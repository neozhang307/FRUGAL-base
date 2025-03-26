#pragma once

#include <cuda_runtime.h>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <atomic>
#include <iostream>
#include <algorithm> // For std::max
#include <tuple>
#include <type_traits>

#include "../profiling/annotation.hpp"
#include "../profiling/memoryManager.hpp"
#include "../utilities/configurationManager.hpp"
#include "../utilities/types.hpp"

namespace memopt {

/**
 * @brief Enhanced TaskManager with integrated annotation and memory management
 */
class TaskManager_v2 {
public:
    /**
     * @brief Operation mode for executing tasks
     */
    enum class ExecutionMode {
        Basic,      // Original pointers, just with annotation
        Production  // Apply getAddress to all parameters
    };

    /**
     * Helper struct for parameter packs in task functions
     */
    template <typename... Args>
    struct TaskFunctionArgs {
        std::tuple<Args...> args;
        
        TaskFunctionArgs() = default;
        
        template <typename... T>
        explicit TaskFunctionArgs(T&&... params) : args(std::forward<T>(params)...) {}
        
        // Helper to apply getAddress to all pointer arguments
        template <typename Tuple, size_t... Is>
        static auto processArgs(const Tuple& t, std::index_sequence<Is...>) {
            return std::make_tuple(processArg(std::get<Is>(t))...);
        }
        
        // Process a single argument - non-pointer version (no-op)
        template <typename T>
        static auto processArg(const T& arg) {
            return arg;
        }
        
        // Process a single argument - pointer version (apply getAddress)
        template <typename T>
        static T* processArg(T* arg) {
            return memopt::MemoryManager::getAddress(arg);
        }
        
        // Create a new TaskFunctionArgs with all pointers processed with getAddress
        TaskFunctionArgs<Args...> processWithGetAddress() const {
            TaskFunctionArgs<Args...> result;
            result.args = processArgs(args, std::index_sequence_for<Args...>{});
            return result;
        }
    };
    
private:
    // Task ID generation and management
    inline static std::atomic<TaskId> nextTaskId = 1;
    std::unordered_set<TaskId> usedTaskIds;
    
    // Forward declaration of TaskFunction 
    template <typename Func, typename... Args>
    class TaskFunction;
    
    // Store task functions with their metadata
    struct TaskInfo {
        // Base class for type erasure
        struct TaskFunctionBase {
            virtual ~TaskFunctionBase() = default;
            virtual void execute(cudaStream_t stream) = 0;
            virtual void executeWithProcessedArgs(cudaStream_t stream) = 0;
        };
        
        // Type-erased task function
        std::unique_ptr<TaskFunctionBase> taskFunction;
        
        // Memory management info
        std::vector<void*> inputs;
        std::vector<void*> outputs;
        
        // For debugging/logging
        std::string taskName;
    };
    
    // Type-safe task function implementation
    template <typename Func, typename... Args>
    class TaskFunction : public TaskInfo::TaskFunctionBase {
    private:
        Func func;
        TaskFunctionArgs<Args...> defaultArgs;
        TaskManager_v2* manager;
        TaskId taskId;
        
    public:
        TaskFunction(Func&& f, TaskFunctionArgs<Args...> args, TaskManager_v2* mgr, TaskId id)
            : func(std::forward<Func>(f)), defaultArgs(std::move(args)), manager(mgr), taskId(id) {}
            
        void execute(cudaStream_t stream) override {
            // Apply function with unpacked arguments
            std::apply([this, stream](Args... args) {
                func(stream, args...);
            }, defaultArgs.args);
        }
        
        void executeWithProcessedArgs(cudaStream_t stream) override {
            // Apply getAddress to all pointer arguments
            auto processedArgs = defaultArgs.processWithGetAddress();
            
            // Apply function with processed arguments
            std::apply([this, stream](Args... args) {
                func(stream, args...);
            }, processedArgs.args);
        }
        
        // Expose default args for modification
        TaskFunctionArgs<Args...>& getDefaultArgs() {
            return defaultArgs;
        }
    };
    
    // Store all task info by task ID
    std::unordered_map<TaskId, TaskInfo> tasks;
    
    // Reference to memory manager
    MemoryManager& memoryManager;
    
    // Flag to help with debugging
    bool debugMode = false;
    
    // Current execution mode
    ExecutionMode currentMode = ExecutionMode::Basic;
    
    /**
     * @brief Generate a new unique task ID
     * @return A new unique task ID
     */
    TaskId generateTaskId() {
        TaskId id = nextTaskId.fetch_add(1);
        
        if (debugMode && ConfigurationManager::getConfig().execution.enableVerboseOutput) {
            std::cout << "Generating new task ID: " << id << std::endl;
        }
        
        if (usedTaskIds.find(id) != usedTaskIds.end()) {
            if (debugMode) {
                std::cout << "Warning: Task ID " << id << " already exists, generating new one" << std::endl;
            }
            return generateTaskId(); // Recursively try again
        }
        
        if (debugMode && ConfigurationManager::getConfig().execution.enableVerboseOutput) {
            std::cout << "Successfully allocated task ID: " << id << std::endl;
        }
        
        return id;
    }
    
    /**
     * @brief Log pointers for debugging (memory registration now handled externally)
     * @param ptrs Vector of pointers to log
     */
    void logPointers(const std::vector<void*>& ptrs) {
        // Only log detailed pointer information if debugMode is on AND verbose output is enabled
        if (debugMode && ConfigurationManager::getConfig().execution.enableVerboseOutput) {
            std::cout << "Task uses " << ptrs.size() << " pointers (memory registration handled externally)" << std::endl;
            for (void* ptr : ptrs) {
                if (ptr) {
                    std::cout << "  - Pointer: " << ptr << std::endl;
                }
            }
        }
    }
    
    // Helper to extract all pointers from a parameter pack
    template <typename T>
    void collectPointers(std::vector<void*>& ptrs, T* ptr) {
        if (ptr) {
            ptrs.push_back(static_cast<void*>(ptr));
        }
    }
    
    template <typename T>
    void collectPointers(std::vector<void*>& ptrs, const T& arg) {
        // Non-pointer types are ignored
    }
    
    template <typename Tuple, size_t... Is>
    void collectPointersFromTuple(std::vector<void*>& ptrs, const Tuple& t, std::index_sequence<Is...>) {
        (collectPointers(ptrs, std::get<Is>(t)), ...);
    }
    
    template <typename... Args>
    std::vector<void*> extractPointers(const TaskFunctionArgs<Args...>& args) {
        std::vector<void*> ptrs;
        collectPointersFromTuple(ptrs, args.args, std::index_sequence_for<Args...>{});
        return ptrs;
    }
    
public:
    /**
     * @brief Constructor
     * @param enableDebug Whether to enable debug output
     */
    TaskManager_v2(bool enableDebug = false) 
        : memoryManager(MemoryManager::getInstance()), debugMode(enableDebug) {
        // Reset the static task ID to ensure a clean start
        nextTaskId = 1;
        if (debugMode && ConfigurationManager::getConfig().execution.enableVerboseOutput) {
            std::cout << "TaskManager_v2 initialized" << std::endl;
        }
    }
    
    /**
     * @brief Destructor
     */
    ~TaskManager_v2() = default;
    
    /**
     * @brief Register a task with auto-generated task ID
     * 
     * @tparam Func Function type
     * @tparam Args Parameter types
     * @param taskFunction Function to execute (takes cudaStream_t and args)
     * @param inputs Vector of input pointers
     * @param outputs Vector of output pointers
     * @param defaultArgs Default arguments for the function
     * @param taskName Optional name for the task (for debugging)
     * @return Generated task ID
     */
    template <typename Func, typename... Args>
    TaskId registerTask(
        Func&& taskFunction,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        TaskFunctionArgs<Args...> defaultArgs = TaskFunctionArgs<Args...>(),
        const std::string& taskName = ""
    ) {
        TaskId taskId = generateTaskId();
        
        if (debugMode && ConfigurationManager::getConfig().execution.enableVerboseOutput) {
            std::cout << "Registering task with auto-generated ID: " << taskId << std::endl;
        }
        
        // Store the ID as used
        usedTaskIds.insert(taskId);
        
        // Create task info
        TaskInfo taskInfo;
        taskInfo.inputs = inputs;
        taskInfo.outputs = outputs;
        taskInfo.taskName = taskName;
        
        // Log the pointers (no registration, assuming it's handled externally)
        logPointers(inputs);
        logPointers(outputs);
        
        // Also log any pointers in the default arguments
        std::vector<void*> argPointers = extractPointers(defaultArgs);
        logPointers(argPointers);
        
        if (debugMode && ConfigurationManager::getConfig().execution.enableVerboseOutput) {
            std::cout << "Note: Memory registration is assumed to be handled externally" << std::endl;
        }
        
        // Create a type-safe task function
        taskInfo.taskFunction = std::make_unique<TaskFunction<Func, Args...>>(
            std::forward<Func>(taskFunction),
            std::move(defaultArgs),
            this,
            taskId
        );
        
        // Store the task
        tasks[taskId] = std::move(taskInfo);
        
        return taskId;
    }
    
    /**
     * @brief Register a task with a specific task ID
     * 
     * @tparam Func Function type
     * @tparam Args Parameter types
     * @param taskId Task ID to use
     * @param taskFunction Function to execute (takes cudaStream_t and args)
     * @param inputs Vector of input pointers
     * @param outputs Vector of output pointers
     * @param defaultArgs Default arguments for the function
     * @param taskName Optional name for the task (for debugging)
     * @return Provided task ID
     * @throws std::invalid_argument if task ID is already in use
     */
    template <typename Func, typename... Args>
    TaskId registerTaskWithId(
        TaskId taskId,
        Func&& taskFunction,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        TaskFunctionArgs<Args...> defaultArgs = TaskFunctionArgs<Args...>(),
        const std::string& taskName = ""
    ) {
        if (usedTaskIds.find(taskId) != usedTaskIds.end()) {
            if (debugMode) {
                std::cout << "Error: Task ID " << taskId << " already in use" << std::endl;
            }
            throw std::invalid_argument("Task ID already in use");
        }
        
        if (debugMode && ConfigurationManager::getConfig().execution.enableVerboseOutput) {
            std::cout << "Registering task with provided ID: " << taskId << std::endl;
        }
        
        // Store the ID as used
        usedTaskIds.insert(taskId);
        
        // Create task info
        TaskInfo taskInfo;
        taskInfo.inputs = inputs;
        taskInfo.outputs = outputs;
        taskInfo.taskName = taskName;
        
        // Log the pointers (no registration, assuming it's handled externally)
        logPointers(inputs);
        logPointers(outputs);
        
        // Also log any pointers in the default arguments
        std::vector<void*> argPointers = extractPointers(defaultArgs);
        logPointers(argPointers);
        
        if (debugMode && ConfigurationManager::getConfig().execution.enableVerboseOutput) {
            std::cout << "Note: Memory registration is assumed to be handled externally" << std::endl;
        }
        
        // Create a type-safe task function
        taskInfo.taskFunction = std::make_unique<TaskFunction<Func, Args...>>(
            std::forward<Func>(taskFunction),
            std::move(defaultArgs),
            this,
            taskId
        );
        
        // Store the task
        tasks[taskId] = std::move(taskInfo);
        
        return taskId;
    }
    
    /**
     * @brief Set default parameters for a task
     * 
     * @tparam Args Parameter types
     * @param taskId Task ID
     * @param args Default arguments to use
     * @return True if successful, false if task ID doesn't exist or type mismatch
     */
    template <typename... Args>
    bool setDefaultParameters(TaskId taskId, TaskFunctionArgs<Args...> args) {
        auto taskIt = tasks.find(taskId);
        if (taskIt == tasks.end()) {
            if (debugMode) {
                std::cout << "Error: Task ID " << taskId << " not found" << std::endl;
            }
            return false;
        }
        
        // Try to cast to the correct type
        auto* typedTask = dynamic_cast<TaskFunction<std::function<void(cudaStream_t, Args...)>, Args...>*>(
            taskIt->second.taskFunction.get()
        );
        
        if (!typedTask) {
            if (debugMode) {
                std::cout << "Error: Type mismatch for task ID " << taskId << std::endl;
            }
            return false;
        }
        
        // Update the default arguments
        typedTask->getDefaultArgs() = std::move(args);
        
        if (debugMode && ConfigurationManager::getConfig().execution.enableVerboseOutput) {
            std::cout << "Set default parameters for task ID: " << taskId << std::endl;
        }
        
        return true;
    }
    
    /**
     * @brief Execute a task using the current execution mode
     * 
     * @param taskId Task ID
     * @param stream CUDA stream
     * @return True if executed successfully, false otherwise
     */
    bool execute(TaskId taskId, cudaStream_t stream = 0) {
        auto taskIt = tasks.find(taskId);
        if (taskIt == tasks.end()) {
            if (debugMode) {
                std::cout << "Error: Task ID " << taskId << " not found" << std::endl;
            }
            return false;
        }
        
        if (debugMode && ConfigurationManager::getConfig().execution.enableVerboseOutput) {
            std::cout << "Executing task ID: " << taskId << 
                      (currentMode == ExecutionMode::Production ? " in Production mode" : " in Basic mode") << 
                      std::endl;
        }
        
        // Get task information
        auto& taskInfo = taskIt->second;
        
        // Annotate the task with inputs/outputs
        if (currentMode == ExecutionMode::Basic) {
            // Basic mode: use original inputs/outputs
            annotateNextTask(taskId, taskInfo.inputs, taskInfo.outputs, stream);
            
            // Execute with original arguments
            taskInfo.taskFunction->execute(stream);
        } else {
            // Production mode: apply getAddress to inputs/outputs for annotation
            std::vector<void*> processedInputs;
            std::vector<void*> processedOutputs;
            
            // Process inputs
            for (void* ptr : taskInfo.inputs) {
                void* processedPtr = getAddress(ptr);
                processedInputs.push_back(processedPtr);
            }
            
            // Process outputs
            for (void* ptr : taskInfo.outputs) {
                void* processedPtr = getAddress(ptr);
                processedOutputs.push_back(processedPtr);
            }
            
            // Annotate with processed pointers
            annotateNextTask(taskId, processedInputs, processedOutputs, stream);
            
            // Execute with processed arguments
            taskInfo.taskFunction->executeWithProcessedArgs(stream);
        }
        
        return true;
    }
    
    /**
     * @brief Execute a task with custom parameters
     * 
     * @tparam Args Parameter types
     * @param taskId Task ID
     * @param stream CUDA stream
     * @param args Custom arguments for this execution
     * @return True if executed successfully, false otherwise
     */
    template <typename... Args>
    bool executeWithParams(TaskId taskId, cudaStream_t stream, TaskFunctionArgs<Args...> args) {
        auto taskIt = tasks.find(taskId);
        if (taskIt == tasks.end()) {
            if (debugMode) {
                std::cout << "Error: Task ID " << taskId << " not found" << std::endl;
            }
            return false;
        }
        
        // Try to cast to the correct type
        auto* typedTask = dynamic_cast<TaskFunction<std::function<void(cudaStream_t, Args...)>, Args...>*>(
            taskIt->second.taskFunction.get()
        );
        
        if (!typedTask) {
            if (debugMode) {
                std::cout << "Error: Type mismatch for task ID " << taskId << std::endl;
            }
            return false;
        }
        
        if (debugMode && ConfigurationManager::getConfig().execution.enableVerboseOutput) {
            std::cout << "Executing task ID: " << taskId << " with custom parameters" << std::endl;
        }
        
        // Get task information
        auto& taskInfo = taskIt->second;
        
        // Save original args
        auto originalArgs = typedTask->getDefaultArgs();
        
        // Temporarily set the new args
        typedTask->getDefaultArgs() = args;
        
        // Execute the task
        if (currentMode == ExecutionMode::Basic) {
            // Basic mode: use original inputs/outputs and parameters
            annotateNextTask(taskId, taskInfo.inputs, taskInfo.outputs, stream);
            typedTask->execute(stream);
        } else {
            // Production mode: apply getAddress to inputs/outputs and parameters
            std::vector<void*> processedInputs;
            std::vector<void*> processedOutputs;
            
            // Process inputs/outputs for annotation
            for (void* ptr : taskInfo.inputs) {
                processedInputs.push_back(getAddress(ptr));
            }
            
            for (void* ptr : taskInfo.outputs) {
                processedOutputs.push_back(getAddress(ptr));
            }
            
            // Annotate with processed pointers
            annotateNextTask(taskId, processedInputs, processedOutputs, stream);
            
            // Execute with processed arguments
            typedTask->executeWithProcessedArgs(stream);
        }
        
        // Restore original args
        typedTask->getDefaultArgs() = originalArgs;
        
        return true;
    }
    
    /**
     * @brief Check if a task ID is valid
     * 
     * @param taskId Task ID to check
     * @return True if task ID exists, false otherwise
     */
    bool hasTask(TaskId taskId) const {
        return tasks.find(taskId) != tasks.end();
    }
    
    /**
     * @brief Get the number of registered tasks
     * 
     * @return Number of tasks
     */
    size_t taskCount() const {
        return tasks.size();
    }
    
    /**
     * @brief Clear all tasks
     */
    void clearTasks() {
        tasks.clear();
        usedTaskIds.clear();
        if (debugMode && ConfigurationManager::getConfig().execution.enableVerboseOutput) {
            std::cout << "Cleared all tasks" << std::endl;
        }
    }
    
    /**
     * @brief Get all registered task IDs
     * 
     * @return Vector of task IDs
     */
    std::vector<TaskId> getTaskIds() const {
        std::vector<TaskId> result;
        result.reserve(tasks.size());
        for (const auto& pair : tasks) {
            result.push_back(pair.first);
        }
        return result;
    }
    
    /**
     * @brief Get the MemoryManager instance
     * 
     * @return Reference to the MemoryManager
     */
    MemoryManager& getMemoryManager() const {
        return memoryManager;
    }
    
    /**
     * @brief Set the execution mode for the task manager
     * 
     * @param mode The execution mode to set (Basic or Production)
     */
    void setExecutionMode(ExecutionMode mode) {
        currentMode = mode;
        if (debugMode && ConfigurationManager::getConfig().execution.enableVerboseOutput) {
            std::cout << "Execution mode set to: " << 
                (mode == ExecutionMode::Production ? "Production" : "Basic") << std::endl;
        }
    }
    
    /**
     * @brief Get the current execution mode
     * 
     * @return The current execution mode
     */
    ExecutionMode getExecutionMode() const {
        return currentMode;
    }
    
    /**
     * @brief Process a pointer with MemoryManager::getAddress
     * 
     * @tparam T Pointer type
     * @param ptr Pointer to process
     * @return Processed pointer
     */
    template <typename T>
    static T* getAddress(T* ptr) {
        // Use the MemoryManager's getAddress method to get the optimized pointer
        return memopt::MemoryManager::getAddress(ptr);
    }
    
    // Helper function to create TaskFunctionArgs from arguments
    template <typename... Args>
    static TaskFunctionArgs<Args...> makeArgs(Args... args) {
        return TaskFunctionArgs<Args...>(std::forward<Args>(args)...);
    }
};

} // namespace memopt