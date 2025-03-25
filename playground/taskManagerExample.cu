#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include "../profiling/memoryManager.hpp"
#include "../optimization/taskManager.hpp"

using namespace memopt;

// Simple CUDA kernel for vector addition
__global__ void addKernel(float* arr, float value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        arr[idx] += value;
    }
}

int main() {
    // Initialize CUDA
    cudaFree(0);
    
    // Allocate some memory that we'll use in our tasks
    float *d_arrayA, *d_arrayB, *d_arrayC;
    const size_t size = 1024 * sizeof(float);
    const int numElements = 1024;
    
    cudaMalloc(&d_arrayA, size);
    cudaMalloc(&d_arrayB, size);
    cudaMalloc(&d_arrayC, size);
    
    // Register the arrays with the memory manager
    MemoryManager::getInstance().registerManagedMemoryAddress(d_arrayA, size);
    MemoryManager::getInstance().registerManagedMemoryAddress(d_arrayB, size);
    MemoryManager::getInstance().registerManagedMemoryAddress(d_arrayC, size);
    
    // Initialize data
    cudaMemset(d_arrayA, 0, size);
    cudaMemset(d_arrayB, 1, size);
    cudaMemset(d_arrayC, 2, size);

    // Create CUDA streams for asynchronous execution
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    // Create a task manager
    TaskManager taskManager;
    
    // Define task IDs
    const TaskId TASK_PRINT = 1;
    const TaskId TASK_INCREMENT = 2;
    const TaskId TASK_COPY = 3;
    const TaskId TASK_STREAM_OP = 4;
    const TaskId TASK_MULTIPLE_PARAMS = 5;
    const TaskId TASK_STREAM_PARAMETERIZED = 6;
    const TaskId TASK_RUNTIME_PARAMETERIZED = 7;
    
    // Example 1: Simple task with no pointer handling
    taskManager.registerTask(TASK_PRINT, [](cudaStream_t) {
        std::cout << "Executing simple task with no parameters" << std::endl;
    });
    
    // Example 2: Task with a regular pointer parameter
    taskManager.registerParameterizedTask(
        TASK_INCREMENT,
        [](cudaStream_t stream, float* arr) {
            dim3 block(256);
            dim3 grid((1024 + block.x - 1) / block.x);
            addKernel<<<grid, block, 0, stream>>>(arr, 1.0f, 1024);
            cudaDeviceSynchronize();
        },
        d_arrayA  // Regular pointer (will be used as-is in Basic mode)
    );
    
    // Example 3: Task with a managed pointer parameter
    taskManager.registerParameterizedTask(
        TASK_COPY,
        [](cudaStream_t stream, float* src, float* dst) {
            cudaMemcpyAsync(dst, src, 1024 * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            cudaDeviceSynchronize();
        },
        TaskManager::managed<float>(d_arrayB),  // Will be processed with MemoryManager in Managed mode
        d_arrayC                              // Regular pointer
    );
    
    // Example 4: Task with a non-pointer parameter (extra stream parameter - will be ignored)
    taskManager.registerParameterizedTask(
        TASK_STREAM_OP,
        [](cudaStream_t stream, float* arr, cudaStream_t extraStream) {
            dim3 block(256);
            dim3 grid((1024 + block.x - 1) / block.x);
            // Use the first stream parameter (the one provided by TaskManager)
            addKernel<<<grid, block, 0, stream>>>(arr, 2.0f, 1024);
            // No synchronization - using stream
        },
        TaskManager::managed<float>(d_arrayA),  // Managed pointer
        stream1                                 // This will be passed as an extra parameter but won't be used for execution
    );
    
    // Example 5: Task with multiple parameters of different types
    taskManager.registerParameterizedTask(
        TASK_MULTIPLE_PARAMS,
        [](cudaStream_t stream, float* src, float* dst, cudaStream_t extraStream, float multiplier, int elements) {
            // Copy first
            cudaMemcpyAsync(dst, src, elements * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            
            // Then add value
            dim3 block(256);
            dim3 grid((elements + block.x - 1) / block.x);
            addKernel<<<grid, block, 0, stream>>>(dst, multiplier, elements);
            
            // Print the operation details
            std::cout << "Executing task with " << elements << " elements and multiplier " 
                    << multiplier << " using stream " << stream << std::endl;
        },
        TaskManager::managed<float>(d_arrayB),  // Managed source pointer
        d_arrayC,                              // Regular destination pointer
        stream2,                               // Extra CUDA stream parameter (won't be used for execution)
        3.5f,                                  // Float scalar parameter
        numElements                            // Integer parameter
    );
    
 
    
    // Example 7: RuntimeParameterizedTask - all arguments provided at execution time
    taskManager.registerRuntimeTask(
        TASK_RUNTIME_PARAMETERIZED,
        [](float* arr, float value, cudaStream_t stream, int elements) {
            dim3 block(256);
            dim3 grid((elements + block.x - 1) / block.x);
            std::cout << "Executing RuntimeParameterizedTask with value " << value 
                      << " on " << elements << " elements" << std::endl;
            addKernel<<<grid, block, 0, stream>>>(arr, value, elements);
        }
    );
    
    // Execute tasks in Basic mode (original pointers)
    std::cout << "Executing tasks in Basic mode" << std::endl;
    taskManager.execute(TASK_PRINT, ExecutionMode::Basic);
    taskManager.execute(TASK_INCREMENT, ExecutionMode::Basic);
    taskManager.execute(TASK_COPY, ExecutionMode::Basic);
    taskManager.execute(TASK_STREAM_OP, ExecutionMode::Basic);
    taskManager.execute(TASK_MULTIPLE_PARAMS, ExecutionMode::Basic);
    
    // Execute RuntimeParameterizedTask with different parameters
    // When we call executeWithParams, our lambda receives the additional parameters
    taskManager.executeWithParams(TASK_RUNTIME_PARAMETERIZED, ExecutionMode::Basic, 
                                  d_arrayB, 7.0f, stream1, 512);
    taskManager.executeWithParams(TASK_RUNTIME_PARAMETERIZED, ExecutionMode::Basic, 
                                  d_arrayC, 9.0f, stream2, 1024);
    
    // Synchronize all operations
    cudaDeviceSynchronize();
    
    // In a real application, some memory optimization would happen here
    // that would relocate pointers via MemoryManager::managedMemoryAddressToAssignedMap
    
    // Simulate memory relocation that would happen during optimization
    float *h_arrayB;
    cudaMallocHost(&h_arrayB, size);
    cudaMemcpy(h_arrayB, d_arrayB, size, cudaMemcpyDeviceToHost);
    
    // Update memory manager to use the host pointer instead of the device pointer
    MemoryManager::getInstance().updateCurrentMapping(d_arrayB, h_arrayB);
    
    // Execute tasks in Managed mode (pointers processed through MemoryManager::getAddress)
    std::cout << "\nExecuting tasks in Managed mode" << std::endl;
    taskManager.execute(TASK_PRINT, ExecutionMode::Managed);
    taskManager.execute(TASK_INCREMENT, ExecutionMode::Managed);
    taskManager.execute(TASK_COPY, ExecutionMode::Managed);
    taskManager.execute(TASK_STREAM_OP, ExecutionMode::Managed);
    taskManager.execute(TASK_MULTIPLE_PARAMS, ExecutionMode::Managed);
    

    // Print what we're doing in Managed mode with RuntimeParameterizedTask
    std::cout << "Executing RuntimeParameterizedTask in Managed mode..." << std::endl;
    
    // Execute RuntimeParameterizedTask with different parameters in Managed mode
    // Use direct pointers instead of ManagedPtr - the task will handle the MemoryManager lookup
    // ExecuteWithParams order: taskId, (optional mode), args...
    // Args list matches what the function expects
    taskManager.executeWithParams(TASK_RUNTIME_PARAMETERIZED, ExecutionMode::Managed, 
                                  d_arrayB, 10.0f, stream1, 256);
    taskManager.executeWithParams(TASK_RUNTIME_PARAMETERIZED, ExecutionMode::Managed, 
                                  d_arrayC, 12.0f, stream2, 768);
    
    // Synchronize all operations
    cudaDeviceSynchronize();
    
    // Clean up
    MemoryManager::getInstance().clearCurrentMappings();
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFreeHost(h_arrayB);
    cudaFree(d_arrayA);
    cudaFree(d_arrayB);
    cudaFree(d_arrayC);
    
    std::cout << "TaskManager demo completed successfully" << std::endl;
    return 0;
}