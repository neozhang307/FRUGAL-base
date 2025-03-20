#pragma once
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
#include <iostream> // For debugging

#include "../profiling/memoryManager.hpp"
#include "../utilities/types.hpp"

namespace memopt {

// Use existing TaskId type from types.hpp

/**
 * @brief Execution modes for task execution
 */
enum class ExecutionMode {
    Basic,    // Use pointers as-is
    Managed   // Apply MemoryManager::getAddress to marked pointers
};

/**
 * @brief Type trait to check if a type is a pointer type (but not void*)
 */
template <typename T>
struct is_non_void_pointer : std::false_type {};

template <typename T>
struct is_non_void_pointer<T*> : std::bool_constant<!std::is_same_v<T, void>> {};

/**
 * @brief Helper to extract function parameter types
 */
// Base template
template <typename T>
struct FunctionTraits;

// Specialization for function pointers
template <typename R, typename... Args>
struct FunctionTraits<R(*)(Args...)> {
    using return_type = R;
    using args_tuple = std::tuple<Args...>;
    static constexpr std::size_t arity = sizeof...(Args);
    
    template <std::size_t N>
    using arg_type = typename std::tuple_element<N, args_tuple>::type;
};

// Specialization for member function pointers
template <typename C, typename R, typename... Args>
struct FunctionTraits<R(C::*)(Args...)> {
    using return_type = R;
    using args_tuple = std::tuple<Args...>;
    static constexpr std::size_t arity = sizeof...(Args);
    
    template <std::size_t N>
    using arg_type = typename std::tuple_element<N, args_tuple>::type;
};

// Specialization for const member function pointers
template <typename C, typename R, typename... Args>
struct FunctionTraits<R(C::*)(Args...) const> {
    using return_type = R;
    using args_tuple = std::tuple<Args...>;
    static constexpr std::size_t arity = sizeof...(Args);
    
    template <std::size_t N>
    using arg_type = typename std::tuple_element<N, args_tuple>::type;
};

// Specialization for lambdas and functors
template <typename Functor>
struct FunctionTraits {
    using operator_type = decltype(&Functor::operator());
    using return_type = typename FunctionTraits<operator_type>::return_type;
    using args_tuple = typename FunctionTraits<operator_type>::args_tuple;
    static constexpr std::size_t arity = FunctionTraits<operator_type>::arity;
    
    template <std::size_t N>
    using arg_type = typename FunctionTraits<operator_type>::template arg_type<N>;
};

// Helper to check if first argument is cudaStream_t
template <typename Tuple>
struct FirstArgIsCudaStream : std::false_type {};

template <typename... Args>
struct FirstArgIsCudaStream<std::tuple<cudaStream_t, Args...>> : std::true_type {};

template <typename T>
inline constexpr bool first_arg_is_cuda_stream_v = FirstArgIsCudaStream<typename FunctionTraits<T>::args_tuple>::value;

/**
 * @brief Wrapper to mark pointers that should be processed by MemoryManager
 */
template <typename T>
class ManagedPtr {
private:
    T* ptr;

public:
    explicit ManagedPtr(T* p) : ptr(p) {}
    
    // Get the pointer in appropriate mode
    T* get(ExecutionMode mode) const {
        if (mode == ExecutionMode::Managed) {
            return MemoryManager::getAddress(ptr);
        }
        return ptr;
    }
    
    // Implicit conversion for convenience in basic mode
    operator T*() const {
        return ptr;
    }
};

// Base class for task storage
class TaskBase {
public:
    virtual ~TaskBase() = default;
    virtual void execute(ExecutionMode mode) = 0;
    virtual void execute(cudaStream_t stream, ExecutionMode mode) = 0;
    
    // Additional methods to check task type (for dynamic_cast safety)
    virtual bool isRuntimeParameterized() const { return false; }
};

// Implementation for simple tasks with no parameters
class SimpleTask : public TaskBase {
private:
    std::function<void(cudaStream_t)> func;
    
public:
    SimpleTask(std::function<void(cudaStream_t)> f) : func(std::move(f)) {}
    
    void execute(ExecutionMode) override {
        func(0);
    }
    
    void execute(cudaStream_t stream, ExecutionMode) override {
        func(stream);
    }
};

// Implementation for tasks with parameters
template <typename... Args>
class ParameterizedTask : public TaskBase {
private:
    // Determine the corresponding argument types for the function by unwrapping ManagedPtr
    template <typename T>
    struct unwrap_managed_ptr {
        using type = T;
    };
    
    template <typename T>
    struct unwrap_managed_ptr<ManagedPtr<T>> {
        using type = T*;
    };
    
    // Generate function parameter types from stored argument types
    template <typename T>
    using function_param_t = typename unwrap_managed_ptr<std::remove_reference_t<T>>::type;
    
    // The actual stored function with unwrapped parameter types (first param is cudaStream_t)
    std::function<void(cudaStream_t, function_param_t<Args>...)> func;
    
    // The stored arguments
    std::tuple<Args...> args;
    
    // Helper to detect if a type is a ManagedPtr
    template <typename T>
    struct is_managed_ptr : std::false_type {};
    
    template <typename T>
    struct is_managed_ptr<ManagedPtr<T>> : std::true_type {};
    
    // Helper to process a single argument
    template <typename T>
    auto processArgument(T& arg, ExecutionMode mode) {
        if constexpr (is_managed_ptr<std::remove_reference_t<T>>::value) {
            // This is a managed pointer - get it with proper mode
            return arg.get(mode);
        } else {
            // Regular argument - pass through without modification
            return arg;
        }
    }
    
public:
    // Constructor for direct function with arguments
    template <typename F>
    ParameterizedTask(F&& f, Args... arguments)
        : func(std::forward<F>(f)), args(std::forward<Args>(arguments)...) {}
    
    void execute(ExecutionMode mode) override {
        // Call the function directly with processed arguments using default stream (0)
        callWithProcessedArgs(0, mode, std::index_sequence_for<Args...>{});
    }
    
    void execute(cudaStream_t stream, ExecutionMode mode) override {
        // Call the function directly with processed arguments and provided stream
        callWithProcessedArgs(stream, mode, std::index_sequence_for<Args...>{});
    }
    
    // Helper to call function directly with processed arguments
    template <size_t... Is>
    void callWithProcessedArgs(cudaStream_t stream, ExecutionMode mode, std::index_sequence<Is...>) {
        func(stream, processArgument(std::get<Is>(args), mode)...);
    }
};

/**
 * @brief TaskManager system for recording and executing functions with different pointer handling
 */
class TaskManager {
private:
    // Store all tasks with a common interface
    std::unordered_map<TaskId, std::unique_ptr<TaskBase>> tasks;
    
    // Special task ID for initializer
    TaskId initializerId = -1;
    
    // Track if initializer has been executed
    bool initializerExecuted = false;
    
    // Flag to enable/disable initializer execution
    bool useInitializer = false;
    
    // Current execution mode for the task manager
    ExecutionMode currentMode = ExecutionMode::Basic;
    
public:
    // Enable or disable initializer functionality
    void setUseInitializer(bool enable) {
        useInitializer = enable;
        if (!enable) {
            initializerExecuted = false;
        }
    }
    
    // Reset the task manager state
    void reset() {
        initializerExecuted = false;
    }
    
    // Register a task as initializer
    template <typename Func>
    void registerInitializer(Func&& func) {
        registerTask(initializerId, std::forward<Func>(func));
    }
    
    // Register a parameterized initializer
    template <typename Func, typename... Args>
    void registerParameterizedInitializer(Func&& func, Args&&... args) {
        registerParameterizedTask(initializerId, std::forward<Func>(func), std::forward<Args>(args)...);
    }
    
    // Register a simple task with no parameters
    template <typename Func>
    void registerTask(const TaskId& taskId, Func&& func) {
        tasks[taskId] = std::make_unique<SimpleTask>(std::forward<Func>(func));
    }
    
    // Register a parameterized task
    template <typename Func, typename... Args>
    void registerParameterizedTask(const TaskId& taskId, Func&& func, Args&&... args) {
        // Use std::remove_reference_t in template parameters for correct reference handling
        auto task = std::make_unique<ParameterizedTask<std::remove_reference_t<Args>...>>(
            std::forward<Func>(func), std::forward<Args>(args)...);
        tasks[taskId] = std::move(task);
    }
    
    // Create a managed pointer
    template <typename T>
    static ManagedPtr<T> managed(T* ptr) {
        return ManagedPtr<T>(ptr);
    }

    // Configure the task manager to use Basic execution mode
    void configureToBasic() {
        currentMode = ExecutionMode::Basic;
    }
    
    // Configure the task manager to use Managed execution mode
    void configureToManaged() {
        currentMode = ExecutionMode::Managed;
    }
    
    // Get the current execution mode
    ExecutionMode getExecutionMode() const {
        return currentMode;
    }
    
    // Execute a task using the current system execution mode
    bool execute(const TaskId& taskId) {
        return execute(taskId, 0, currentMode);
    }
    
    // Execute a task with stream and using the current system execution mode
    bool execute(const TaskId& taskId, cudaStream_t stream) {
        return execute(taskId, stream, currentMode);
    }
    
    // Execute a task with explicitly specified execution mode
    bool execute(const TaskId& taskId, ExecutionMode mode) {
        return execute(taskId, 0, mode);
    }
    
    // Execute a task with stream and explicitly specified execution mode
    bool execute(const TaskId& taskId, cudaStream_t stream, ExecutionMode mode) {
        // Run initializer if enabled, not yet executed, and this is not the initializer itself
        if (useInitializer && !initializerExecuted && taskId != initializerId) {
            auto initIt = tasks.find(initializerId);
            if (initIt != tasks.end()) {
                initIt->second->execute(stream, mode);
                initializerExecuted = true;
            }
        }
        
        // Execute the requested task
        auto it = tasks.find(taskId);
        if (it != tasks.end()) {
            it->second->execute(stream, mode);
            return true;
        }
        
        return false;
    }
    
    // Finalize by resetting initializer state
    void finalize() {
        reset();
    }
    
    // Implementation for tasks that receive all parameters at runtime
    template <typename... RuntimeArgs>
    class RuntimeParameterizedTask : public TaskBase {
    private:
        // Determine the corresponding argument types for the function by unwrapping ManagedPtr
        template <typename T>
        struct unwrap_managed_ptr {
            using type = T;
        };
        
        template <typename T>
        struct unwrap_managed_ptr<ManagedPtr<T>> {
            using type = T*;
        };
        
        // Generate function parameter types from stored argument types
        template <typename T>
        using function_param_t = typename unwrap_managed_ptr<std::remove_reference_t<T>>::type;
        
        // The actual stored lambda with its original signature
        std::function<void(RuntimeArgs...)> originalFunc;
        
        // Helper to check if first argument is cudaStream_t
        template <typename Tuple>
        struct HasCudaStreamParam : std::false_type {};
        
        template <typename... Args>
        struct HasCudaStreamParam<std::tuple<cudaStream_t, Args...>> : std::true_type {};
        
        // Check if the first parameter of the function is cudaStream_t
        static constexpr bool hasCudaStreamFirstParam = 
            HasCudaStreamParam<std::tuple<RuntimeArgs...>>::value;
        
    public:
        // Constructor for direct function with runtime arguments
        template <typename F>
        RuntimeParameterizedTask(F&& f)
            : originalFunc(std::forward<F>(f)) {}
        
        // This is a placeholder implementation since parameters are provided at runtime
        void execute(ExecutionMode) override {
            std::cerr << "Warning: RuntimeParameterizedTask::execute called without parameters. "
                      << "Use executeWithParams instead." << std::endl;
        }
        
        void execute(cudaStream_t, ExecutionMode) override {
            std::cerr << "Warning: RuntimeParameterizedTask::execute called without parameters. "
                      << "Use executeWithParams instead." << std::endl;
        }
        
        // Helper to detect if a type is a ManagedPtr
        template <typename T>
        struct is_managed_ptr : std::false_type {};
        
        template <typename T>
        struct is_managed_ptr<ManagedPtr<T>> : std::true_type {};
        
        // Helper to process a single argument
        template <typename T>
        auto processArgument(T& arg, ExecutionMode mode) {
            if constexpr (is_managed_ptr<std::remove_reference_t<T>>::value) {
                // This is a managed pointer - get it with proper mode
                return arg.get(mode);
            } else {
                // Regular argument - pass through without modification
                return arg;
            }
        }
    
        // The actual execution method with runtime parameters and execution mode
        void executeWithParams(ExecutionMode mode, RuntimeArgs... args) {
            executeWithParams(0, mode, std::forward<RuntimeArgs>(args)...);
        }
        
        // The actual execution method with stream, runtime parameters and execution mode
        void executeWithParams(cudaStream_t stream, ExecutionMode mode, RuntimeArgs... args) {
            // Process and call the function with the given arguments
            callWithProcessedArgs(stream, mode, std::forward<RuntimeArgs>(args)...);
        }
        
        // Helper to process arguments and call function
        void callWithProcessedArgs(cudaStream_t stream, ExecutionMode mode, RuntimeArgs... args) {
            // Process each argument individually
            originalFunc(processArgument(args, mode)...);
        }
        
        bool isRuntimeParameterized() const override { return true; }
    };
    
    // Register a task that will receive parameters at execution time
    template <typename Func>
    void registerRuntimeTask(const TaskId& taskId, Func&& func) {
        using FuncType = std::decay_t<Func>;
        using FuncTraits = FunctionTraits<FuncType>;
        
        // Create a RuntimeParameterizedTask with parameter types extracted from the function
        registerRuntimeTaskHelper(taskId, std::forward<Func>(func), 
                                typename FuncTraits::args_tuple{});
    }
    
    // Helper to extract parameter types from the function - main version
    template <typename Func, typename... Args>
    void registerRuntimeTaskHelper(const TaskId& taskId, Func&& func, std::tuple<Args...>) {
        // Create the appropriate RuntimeParameterizedTask type based on the function signature
        auto task = std::make_unique<RuntimeParameterizedTask<Args...>>(
            std::forward<Func>(func));
            
        tasks[taskId] = std::move(task);
    }
    

    
    // Execute a task with parameters provided at runtime using the current system mode
    template <typename... Args>
    bool executeWithParams(const TaskId& taskId, Args&&... args) {
        return executeWithParamsInternal(taskId, 0, currentMode, std::forward<Args>(args)...);
    }

    // Execute a task with parameters provided at runtime, stream, and using the current system mode
    template <typename... Args>
    bool executeWithParams(const TaskId& taskId, cudaStream_t stream, Args&&... args) {
        return executeWithParamsInternal(taskId, stream, currentMode, std::forward<Args>(args)...);
    }
    
    // Execute a task with parameters provided at runtime and explicit execution mode
    template <typename... Args>
    bool executeWithParams(const TaskId& taskId, ExecutionMode mode, Args&&... args) {
        return executeWithParamsInternal(taskId, 0, mode, std::forward<Args>(args)...);
    }
    
    // Execute a task with parameters provided at runtime, stream, and explicit execution mode
    template <typename... Args>
    bool executeWithParams(const TaskId& taskId, cudaStream_t stream, ExecutionMode mode, Args&&... args) {
        return executeWithParamsInternal(taskId, stream, mode, std::forward<Args>(args)...);
    }
    
    // Internal implementation to prevent recursion
    template <typename... Args>
    bool executeWithParamsInternal(const TaskId& taskId, cudaStream_t stream, ExecutionMode mode, Args&&... args) {
        // Handle initializer if needed
        if (useInitializer && !initializerExecuted && taskId != initializerId) {
            auto initIt = tasks.find(initializerId);
            if (initIt != tasks.end()) {
                initIt->second->execute(stream, mode);
                initializerExecuted = true;
            }
        }
        
        // Find and execute the task with the provided parameters
        auto it = tasks.find(taskId);
        if (it != tasks.end() && it->second->isRuntimeParameterized()) {
            // Try to execute the task with the provided parameters
            try {
                // Use a generic method to execute runtime tasks with different parameter types
                // This avoids the need for exact type matching with dynamic_cast
                if (executeRuntimeTask(it->second.get(), stream, mode, std::forward<Args>(args)...)) {
                    return true;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error executing runtime task: " << e.what() << std::endl;
            }
            
            std::cerr << "Error: Task parameter types don't match the registered function signature." << std::endl;
        }
        
        return false;
    }
    
 
    
    // Fallback version when no specific overload matches
    template <typename... Args>
    bool executeRuntimeTask(TaskBase* task, ExecutionMode mode, Args&&... args) {
        return executeRuntimeTask(task, 0, mode, std::forward<Args>(args)...);
    }
    
    // Fallback version when no specific overload matches with stream
    template <typename... Args>
    bool executeRuntimeTask(TaskBase* task, cudaStream_t stream, ExecutionMode mode, Args&&... args) {
        // Cast to the appropriate runtime parameterized task type with the exact parameter types
        // Note: In RuntimeParameterizedTask, the cudaStream_t is the first parameter of the lambda
        // but we store it separately and don't include it in the template arguments
        auto* runtimeTask = dynamic_cast<RuntimeParameterizedTask<std::remove_reference_t<Args>...>*>(task);
        if (runtimeTask) {
            runtimeTask->executeWithParams(stream, mode, std::forward<Args>(args)...);
            return true;
        }
        return false;
    }
};

} // namespace memopt