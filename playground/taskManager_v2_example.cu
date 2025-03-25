#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include "../profiling/memoryManager.hpp"
#include "../optimization/taskManager_v2.hpp"
#include "../utilities/cudaUtilities.hpp"
#include "../utilities/configurationManager.hpp"

using namespace memopt;

// Simple CUDA kernel for testing
__global__ void addKernel(float* arr, float value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        arr[idx] += value;
    }
}

// Advanced kernel with multiple parameters
__global__ void transformKernel(float* output, float* input1, float* input2, 
                              float scale, int offset, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = (input1[idx] + input2[idx]) * scale + offset;
    }
}

int main() {
    // Create and export configuration
    {
        Configuration config;
        config.optimization.minManagedArraySize = 1024; // 1KB minimum for managed arrays
        
        // Export to config.json
        nlohmann::json j = config;
        std::ofstream f("config.json");
        f << j.dump(2) << std::endl;
    }
    
    // Now load it
    ConfigurationManager::loadConfiguration("config.json");
    
    // Initialize CUDA
    checkCudaErrors(cudaFree(0));
    
    // Create a task manager with debug mode enabled
    TaskManager_v2 taskManager(true);
    
    // Create test arrays
    float *d_array1, *d_array2, *d_output;
    size_t size = 1024 * sizeof(float);
    
    checkCudaErrors(cudaMalloc(&d_array1, size));
    checkCudaErrors(cudaMalloc(&d_array2, size));
    checkCudaErrors(cudaMalloc(&d_output, size));
    
    checkCudaErrors(cudaMemset(d_array1, 0, size));
    checkCudaErrors(cudaMemset(d_array2, 0, size));
    checkCudaErrors(cudaMemset(d_output, 0, size));
    
    // Manually register memory with the memory manager (now required externally)
    // Using the global template functions which are friends of MemoryManager
    registerManagedMemoryAddress(d_array1, size);
    registerManagedMemoryAddress(d_array2, size);
    registerManagedMemoryAddress(d_output, size);
    
    // Mark inputs and outputs for application I/O
    registerApplicationInput(d_array1);
    registerApplicationInput(d_array2);
    registerApplicationOutput(d_output);
    
    std::cout << "Memory registered externally before task registration" << std::endl;
    
    // Create CUDA stream
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));
    
    std::cout << "Registering simple task..." << std::endl;
    
    // Basic task without custom parameters
    TaskId taskId1 = taskManager.registerTask<std::function<void(cudaStream_t)>>(
        [](cudaStream_t stream) {
            std::cout << "Executing basic task without parameters" << std::endl;
        },
        {}, // inputs
        {}, // outputs
        TaskManager_v2::makeArgs(), // No args
        "BasicTask"
    );
    
    std::cout << "Registering single-parameter task..." << std::endl;
    
    // Task with a single parameter
    TaskId taskId2 = taskManager.registerTask<std::function<void(cudaStream_t, float*)>, float*>(
        [](cudaStream_t stream, float* arr) {
            std::cout << "Executing task with array parameter: " << arr << std::endl;
            dim3 block(256);
            dim3 grid(4);
            addKernel<<<grid, block, 0, stream>>>(arr, 1.0f, 1024);
        },
        {d_array1}, // inputs
        {d_array1}, // outputs
        TaskManager_v2::makeArgs(d_array1), // default args
        "SingleParamTask"
    );
    
    std::cout << "Registering multi-parameter task..." << std::endl;
    
    // Task with multiple parameters (mixture of pointers and values)
    using MultiParamFunc = std::function<void(cudaStream_t, float*, float*, float*, float, int)>;
    TaskId taskId3 = taskManager.registerTask<MultiParamFunc, float*, float*, float*, float, int>(
        [](cudaStream_t stream, float* output, float* input1, float* input2, float scale, int offset) {
            std::cout << "Executing task with multiple parameters:" << std::endl;
            std::cout << "  - output: " << output << std::endl;
            std::cout << "  - input1: " << input1 << std::endl; 
            std::cout << "  - input2: " << input2 << std::endl;
            std::cout << "  - scale: " << scale << std::endl;
            std::cout << "  - offset: " << offset << std::endl;
            
            dim3 block(256);
            dim3 grid(4);
            transformKernel<<<grid, block, 0, stream>>>(output, input1, input2, scale, offset, 1024);
        },
        {d_array1, d_array2}, // inputs
        {d_output}, // outputs
        TaskManager_v2::makeArgs(d_output, d_array1, d_array2, 2.0f, 10), // default args
        "MultiParamTask"
    );
    
    // Execute tasks in Basic mode
    std::cout << "\nExecuting tasks in Basic mode..." << std::endl;
    taskManager.execute(taskId1, stream);
    taskManager.execute(taskId2, stream);
    taskManager.execute(taskId3, stream);
    
    // Change parameters for task3
    std::cout << "\nChanging parameters for multi-parameter task..." << std::endl;
    auto newArgs = TaskManager_v2::makeArgs(d_output, d_array1, d_array2, 3.0f, 20);
    taskManager.executeWithParams<float*, float*, float*, float, int>(taskId3, stream, newArgs);
    
    // Now switch to Production mode and execute again
    std::cout << "\nSwitching to Production mode..." << std::endl;
    taskManager.setExecutionMode(TaskManager_v2::ExecutionMode::Production);
    
    std::cout << "Executing tasks in Production mode..." << std::endl;
    taskManager.execute(taskId1, stream);
    taskManager.execute(taskId2, stream);
    taskManager.execute(taskId3, stream);
    
    // Change parameters again and execute in Production mode
    std::cout << "\nChanging parameters for multi-parameter task in Production mode..." << std::endl;
    auto productionArgs = TaskManager_v2::makeArgs(d_output, d_array1, d_array2, 4.0f, 30);
    taskManager.executeWithParams<float*, float*, float*, float, int>(taskId3, stream, productionArgs);
    
    // Clean up
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaStreamDestroy(stream));
    checkCudaErrors(cudaFree(d_array1));
    checkCudaErrors(cudaFree(d_array2));
    checkCudaErrors(cudaFree(d_output));
    
    std::cout << "Example completed successfully" << std::endl;
    return 0;
}