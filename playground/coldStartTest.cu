#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include "../profiling/memoryManager.hpp"
#include "../optimization/executor.hpp"
#include "../optimization/optimizationOutput.hpp"
#include "../utilities/configurationManager.hpp"

using namespace memopt;

// Simple kernel for testing
__global__ void simpleKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

int main() {
    printf("=== Cold Start Execution Test ===\n");
    
    // Step 1: Initialize configuration
    // Note: ConfigurationManager uses static config, we'll rely on defaults
    // The executor will use these from ConfigurationManager::getConfig()
    // mainDeviceId = 0 (default)
    // storageDeviceId = -1 (host memory, default)
    // useNvlink = false (default)
    
    // Step 2: Create test arrays
    printf("\nStep 1: Create test arrays\n");
    
    size_t arraySize = 1024 * 1024 * sizeof(float);  // 1M elements
    int numArrays = 3;
    std::vector<float*> deviceArrays;
    std::vector<float*> cpuArrays;
    
    // Allocate GPU arrays
    for (int i = 0; i < numArrays; i++) {
        float* d_array;
        cudaMalloc(&d_array, arraySize);
        deviceArrays.push_back(d_array);
        printf("  GPU array %d allocated: %p (%zu bytes)\n", i, d_array, arraySize);
    }
    
    // Allocate CPU arrays with test data
    for (int i = 0; i < numArrays; i++) {
        float* h_array = (float*)malloc(arraySize);
        for (size_t j = 0; j < arraySize/sizeof(float); j++) {
            h_array[j] = (float)(i * 1000 + j);
        }
        cpuArrays.push_back(h_array);
        printf("  CPU array %d allocated: %p (%zu bytes)\n", i, h_array, arraySize);
    }
    
    // Step 3: Register arrays with memory manager
    printf("\nStep 2: Register arrays with MemoryManager\n");
    
    MemoryManager& memManager = MemoryManager::getInstance();
    
    for (int i = 0; i < numArrays; i++) {
        memManager.registerManagedMemoryAddress(deviceArrays[i], arraySize);
        printf("  Registered GPU array %d\n", i);
    }
    
    // Step 4: Setup cold start - move all data to CPU
    printf("\nStep 3: Setup cold start - move all data to CPU\n");
    
    memManager.configureStorage(0, -1, false);
    
    // Use re-registration to set CPU data as storage
    for (int i = 0; i < numArrays; i++) {
        bool success = memManager.reregisterManagedArrayWithLargerCPUData(
            deviceArrays[i], cpuArrays[i], arraySize);
        if (success) {
            printf("  Array %d re-registered with CPU data\n", i);
        } else {
            printf("  ERROR: Failed to re-register array %d\n", i);
            return -1;
        }
    }
    
    // Step 5: Create a simple optimization plan
    printf("\nStep 4: Create optimization plan\n");
    
    OptimizationOutput optimizedGraph;
    
    // Create nodes: prefetch array 0, task 0, offload array 0, prefetch array 1, task 1, etc.
    int nodeId = 0;
    for (int i = 0; i < numArrays; i++) {
        // Prefetch node
        optimizedGraph.nodes.push_back(nodeId);
        optimizedGraph.nodeIdToNodeTypeMap[nodeId] = OptimizationOutput::NodeType::dataMovement;
        OptimizationOutput::DataMovement dm;
        dm.arrayId = i;
        dm.direction = OptimizationOutput::DataMovement::Direction::hostToDevice;
        optimizedGraph.nodeIdToDataMovementMap[nodeId] = dm;
        int prefetchNode = nodeId++;
        
        // Task node
        optimizedGraph.nodes.push_back(nodeId);
        optimizedGraph.nodeIdToNodeTypeMap[nodeId] = OptimizationOutput::NodeType::task;
        optimizedGraph.nodeIdToTaskIdMap[nodeId] = i;
        int taskNode = nodeId++;
        
        // Offload node  
        optimizedGraph.nodes.push_back(nodeId);
        optimizedGraph.nodeIdToNodeTypeMap[nodeId] = OptimizationOutput::NodeType::dataMovement;
        OptimizationOutput::DataMovement dm2;
        dm2.arrayId = i;
        dm2.direction = OptimizationOutput::DataMovement::Direction::deviceToHost;
        optimizedGraph.nodeIdToDataMovementMap[nodeId] = dm2;
        int offloadNode = nodeId++;
        
        // Add edges
        optimizedGraph.edges[prefetchNode].push_back(taskNode);
        optimizedGraph.edges[taskNode].push_back(offloadNode);
        
        // Chain arrays together
        if (i < numArrays - 1) {
            optimizedGraph.edges[offloadNode].push_back(nodeId); // Next prefetch
        }
    }
    
    // Initially no arrays on device (cold start)
    optimizedGraph.arraysInitiallyAllocatedOnDevice.clear();
    
    printf("  Created graph with %d nodes\n", nodeId);
    
    // Step 6: Define task execution callback
    printf("\nStep 5: Define task execution callback\n");
    
    ExecuteRandomTask executeTask = [&deviceArrays](int taskId, std::map<void*, void*> addressMapping, cudaStream_t stream) {
        printf("  Executing task %d\n", taskId);
        
        // Get current address of the array
        void* originalAddr = deviceArrays[taskId];
        void* currentAddr = addressMapping[originalAddr];
        
        if (currentAddr == nullptr) {
            printf("    ERROR: No mapping for array %d\n", taskId);
            return;
        }
        
        // Launch kernel
        int size = 1024 * 1024;
        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;
        
        simpleKernel<<<gridSize, blockSize, 0, stream>>>((float*)currentAddr, size);
    };
    
    // Step 7: Execute with cold start
    printf("\nStep 6: Execute with cold start\n");
    
    try {
        float runningTime = 0.0f;
        Executor* executor = Executor::getInstance();
        
        executor->executeOptimizedGraphColdStart(
            optimizedGraph, 
            executeTask, 
            runningTime, 
            memManager
        );
        
        printf("\n✅ SUCCESS: Cold start execution completed!\n");
        printf("   Execution time: %.3f ms\n", runningTime * 1000.0f);
        
    } catch (const std::exception& e) {
        printf("\n❌ FAILED: Exception during cold start execution\n");
        printf("   Error: %s\n", e.what());
        return -1;
    }
    
    // Step 8: Verify data is back in CPU
    printf("\nStep 7: Verify final state\n");
    
    bool allInCpu = true;
    for (int i = 0; i < numArrays; i++) {
        ArrayId arrayId = memManager.getArrayId(deviceArrays[i]);
        const auto& arrayInfo = memManager.getMemoryArrayInfo(arrayId);
        
        printf("  Array %d: deviceAddress=%p, storageAddress=%p\n", 
               i, arrayInfo.deviceAddress, arrayInfo.storageAddress);
        
        if (arrayInfo.deviceAddress != nullptr) {
            printf("    WARNING: Array %d still on device!\n", i);
            allInCpu = false;
        }
    }
    
    if (allInCpu) {
        printf("\n✅ All arrays successfully returned to CPU/storage\n");
    } else {
        printf("\n❌ Some arrays remain on device\n");
    }
    
    // Cleanup
    printf("\nStep 8: Cleanup\n");
    for (auto h_array : cpuArrays) {
        free(h_array);
    }
    for (auto d_array : deviceArrays) {
        cudaFree(d_array);
    }
    
    printf("=== Test Complete ===\n");
    return allInCpu ? 0 : -1;
}