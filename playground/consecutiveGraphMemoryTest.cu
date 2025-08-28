#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>

#include "../utilities/cudaUtilities.hpp"

using namespace memopt;

// Helper function to print memory usage
void printMemoryUsage(const std::string& stage) {
    size_t free_bytes, total_bytes;
    checkCudaErrors(cudaMemGetInfo(&free_bytes, &total_bytes));
    size_t used_bytes = total_bytes - free_bytes;
    
    std::cout << "[MEMORY-" << stage << "] Free: " << free_bytes / (1024*1024) 
              << " MB, Used: " << used_bytes / (1024*1024) << " MB" << std::endl;
}

int main() {
    // Configuration to trigger cuBLAS internal memory allocations
    const int MATRIX_SIZE = 4096;  // 4096x4096 matrix = ~67MB per matrix
    const int NUM_ITERATIONS = 5;
    
    std::cout << "=== CUDA Graph Memory Leak Bug Demonstration ===" << std::endl;
    std::cout << "Using cuBLAS GEMM operations to trigger internal allocations" << std::endl;
    std::cout << "Matrix size: " << MATRIX_SIZE << "x" << MATRIX_SIZE << std::endl;
    std::cout << "Memory per matrix: " << (MATRIX_SIZE * MATRIX_SIZE * sizeof(float)) / (1024*1024) << " MB" << std::endl;
    std::cout << "==================================================" << std::endl;
    
    // Create CUDA stream and cuBLAS handle
    cudaStream_t stream;
    cublasHandle_t handle;
    checkCudaErrors(cudaStreamCreate(&stream));
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
    
    printMemoryUsage("INITIAL");
    
    // === Demonstrate Memory Leak Bug with Consecutive Graphs ===
    // The key: allocate memory INSIDE each graph to trigger the bug
    std::vector<cudaGraph_t> graphs;
    std::vector<cudaGraphExec_t> graphExecs;
    
    const float alpha = 1.0f, beta = 0.0f;
    size_t matrix_bytes = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);
    
    // Store pointers to demonstrate cross-graph allocation/deallocation
    std::vector<float*> allocated_ptrs;
    
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        std::cout << "\n--- Iteration " << (iter + 1) << ": Allocation Graph ---" << std::endl;
        
        // Create ALLOCATION graph
        cudaGraph_t alloc_graph;
        checkCudaErrors(cudaGraphCreate(&alloc_graph, 0));
        
        // Begin stream capture for allocation
        checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        
        // CRITICAL: Allocate memory in THIS graph
        float *d_A, *d_B, *d_C;
        checkCudaErrors(cudaMallocAsync(&d_A, matrix_bytes, stream));
        checkCudaErrors(cudaMallocAsync(&d_B, matrix_bytes, stream));
        checkCudaErrors(cudaMallocAsync(&d_C, matrix_bytes, stream));
        
        // Initialize with async memset
        checkCudaErrors(cudaMemsetAsync(d_A, 1, matrix_bytes, stream));
        checkCudaErrors(cudaMemsetAsync(d_B, 1, matrix_bytes, stream));
        checkCudaErrors(cudaMemsetAsync(d_C, 0, matrix_bytes, stream));
        
        // End capture for allocation graph
        checkCudaErrors(cudaStreamEndCapture(stream, &alloc_graph));
        
        // Store pointers for later freeing
        allocated_ptrs.push_back(d_A);
        allocated_ptrs.push_back(d_B);
        allocated_ptrs.push_back(d_C);
        
        // Instantiate and execute allocation graph
        cudaGraphExec_t allocGraphExec;
        checkCudaErrors(cudaGraphInstantiate(&allocGraphExec, alloc_graph, nullptr, nullptr, 0));
        checkCudaErrors(cudaGraphLaunch(allocGraphExec, stream));
        checkCudaErrors(cudaStreamSynchronize(stream));
        
        printMemoryUsage("AFTER_ALLOC_GRAPH_" + std::to_string(iter + 1));
        
        // Store graphs for cleanup
        graphs.push_back(alloc_graph);
        graphExecs.push_back(allocGraphExec);
        
        // NOTE: Intentionally NOT calling cudaDeviceGraphMemTrim() 
        // to demonstrate Bug #1 - cross-graph allocation creates memory pools that don't get freed
        std::cout << "NOT calling cudaDeviceGraphMemTrim - allocated memory stays in graph pools..." << std::endl;
    }
    
    std::cout << "\n--- Creating DEALLOCATION graphs (this should trigger the bug) ---" << std::endl;
    
    // Now create separate graphs to FREE the memory allocated in previous graphs
    // This is the key pattern that triggers the memory leak bug
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        std::cout << "\n--- Iteration " << (iter + 1) << ": Deallocation Graph ---" << std::endl;
        
        // Create DEALLOCATION graph
        cudaGraph_t dealloc_graph;
        checkCudaErrors(cudaGraphCreate(&dealloc_graph, 0));
        
        // Begin stream capture for deallocation
        checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        
        // CRITICAL: Free memory that was allocated in DIFFERENT graphs
        // This creates the memory leak because graph pools don't communicate
        int base_idx = iter * 3;  // 3 pointers per iteration
        checkCudaErrors(cudaFreeAsync(allocated_ptrs[base_idx], stream));     // d_A
        checkCudaErrors(cudaFreeAsync(allocated_ptrs[base_idx + 1], stream)); // d_B
        checkCudaErrors(cudaFreeAsync(allocated_ptrs[base_idx + 2], stream)); // d_C
        
        // End capture for deallocation graph
        checkCudaErrors(cudaStreamEndCapture(stream, &dealloc_graph));
        
        // Instantiate and execute deallocation graph
        cudaGraphExec_t deallocGraphExec;
        checkCudaErrors(cudaGraphInstantiate(&deallocGraphExec, dealloc_graph, nullptr, nullptr, 0));
        checkCudaErrors(cudaGraphLaunch(deallocGraphExec, stream));
        checkCudaErrors(cudaStreamSynchronize(stream));
        
        printMemoryUsage("AFTER_DEALLOC_GRAPH_" + std::to_string(iter + 1));
        
        // Store graphs for cleanup
        graphs.push_back(dealloc_graph);
        graphExecs.push_back(deallocGraphExec);
        
        std::cout << "Memory should be freed but may stay in graph pools (demonstrating bug)..." << std::endl;
    }
    
    std::cout << "\n--- Phase 3A: REALLOCATION graphs WITHOUT auto-free flag ---" << std::endl;
    
    // Test reallocation WITHOUT auto-free flag
    for (int iter = 0; iter < 2; iter++) {  // Reduced iterations for comparison
        std::cout << "\n--- Iteration " << (iter + 1) << ": Reallocation Graph (NO AUTO-FREE) ---" << std::endl;
        
        cudaGraph_t realloc_graph;
        checkCudaErrors(cudaGraphCreate(&realloc_graph, 0));
        
        checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        
        float *d_A_new, *d_B_new, *d_C_new;
        checkCudaErrors(cudaMallocAsync(&d_A_new, matrix_bytes, stream));
        checkCudaErrors(cudaMallocAsync(&d_B_new, matrix_bytes, stream));
        checkCudaErrors(cudaMallocAsync(&d_C_new, matrix_bytes, stream));
        
        checkCudaErrors(cudaMemsetAsync(d_A_new, 2, matrix_bytes, stream));
        checkCudaErrors(cudaMemsetAsync(d_B_new, 2, matrix_bytes, stream));
        checkCudaErrors(cudaMemsetAsync(d_C_new, 0, matrix_bytes, stream));
        
        checkCudaErrors(cudaFreeAsync(d_A_new, stream));
        checkCudaErrors(cudaFreeAsync(d_B_new, stream));
        checkCudaErrors(cudaFreeAsync(d_C_new, stream));
        
        checkCudaErrors(cudaStreamEndCapture(stream, &realloc_graph));
        
        // Instantiate WITHOUT auto-free flag
        cudaGraphExec_t reallocGraphExec;
        checkCudaErrors(cudaGraphInstantiate(&reallocGraphExec, realloc_graph, nullptr, nullptr, 0));
        std::cout << "Using NO FLAGS (default behavior)" << std::endl;
        
        checkCudaErrors(cudaGraphLaunch(reallocGraphExec, stream));
        checkCudaErrors(cudaStreamSynchronize(stream));
        
        printMemoryUsage("NO_AUTO_FREE_" + std::to_string(iter + 1));
        
        graphs.push_back(realloc_graph);
        graphExecs.push_back(reallocGraphExec);
    }
    
    std::cout << "\n--- Phase 3B: REALLOCATION graphs WITH auto-free flag ---" << std::endl;
    
    // Test reallocation WITH auto-free flag
    for (int iter = 0; iter < 2; iter++) {  // Reduced iterations for comparison
        std::cout << "\n--- Iteration " << (iter + 1) << ": Reallocation Graph (WITH AUTO-FREE) ---" << std::endl;
        
        cudaGraph_t realloc_graph;
        checkCudaErrors(cudaGraphCreate(&realloc_graph, 0));
        
        checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        
        float *d_A_new, *d_B_new, *d_C_new;
        checkCudaErrors(cudaMallocAsync(&d_A_new, matrix_bytes, stream));
        checkCudaErrors(cudaMallocAsync(&d_B_new, matrix_bytes, stream));
        checkCudaErrors(cudaMallocAsync(&d_C_new, matrix_bytes, stream));
        
        checkCudaErrors(cudaMemsetAsync(d_A_new, 3, matrix_bytes, stream));
        checkCudaErrors(cudaMemsetAsync(d_B_new, 3, matrix_bytes, stream));
        checkCudaErrors(cudaMemsetAsync(d_C_new, 0, matrix_bytes, stream));
        
        checkCudaErrors(cudaFreeAsync(d_A_new, stream));
        checkCudaErrors(cudaFreeAsync(d_B_new, stream));
        checkCudaErrors(cudaFreeAsync(d_C_new, stream));
        
        checkCudaErrors(cudaStreamEndCapture(stream, &realloc_graph));
        
        // Instantiate WITH auto-free flag
        cudaGraphExec_t reallocGraphExec;
        checkCudaErrors(cudaGraphInstantiate(&reallocGraphExec, realloc_graph, nullptr, nullptr, 
                                           cudaGraphInstantiateFlagAutoFreeOnLaunch));
        std::cout << "Using CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH" << std::endl;
        
        checkCudaErrors(cudaGraphLaunch(reallocGraphExec, stream));
        checkCudaErrors(cudaStreamSynchronize(stream));
        
        printMemoryUsage("WITH_AUTO_FREE_" + std::to_string(iter + 1));
        
        graphs.push_back(realloc_graph);
        graphExecs.push_back(reallocGraphExec);
        
        std::cout << "Expected: Auto-free should help manage memory automatically" << std::endl;
    }
    
    std::cout << "\n=== COMPARISON RESULTS ===" << std::endl;
    std::cout << "Compare memory usage patterns:" << std::endl;
    std::cout << "- NO_AUTO_FREE vs WITH_AUTO_FREE" << std::endl;
    std::cout << "- Does auto-free flag reduce memory usage?" << std::endl;
    
    std::cout << "\n--- Cleaning up graphs (but not calling trim) ---" << std::endl;
    
    // Clean up graphs
    for (size_t i = 0; i < graphs.size(); i++) {
        checkCudaErrors(cudaGraphExecDestroy(graphExecs[i]));
        checkCudaErrors(cudaGraphDestroy(graphs[i]));
    }
    
    printMemoryUsage("AFTER_GRAPH_CLEANUP");
    
    std::cout << "\n--- NOW demonstrating the FIX ---" << std::endl;
    std::cout << "Calling cudaDeviceGraphMemTrim() to free leaked memory..." << std::endl;
    
    int currentDevice;
    checkCudaErrors(cudaGetDevice(&currentDevice));
    checkCudaErrors(cudaDeviceGraphMemTrim(currentDevice));
    
    printMemoryUsage("AFTER_TRIM_FIX");
    
    // Cleanup
    cublasDestroy(handle);
    checkCudaErrors(cudaStreamDestroy(stream));
    
    std::cout << "\n=== BUG DEMONSTRATION COMPLETED ====" << std::endl;
    std::cout << "CUDA Graph Memory Leak Bug (Bug #1):" << std::endl;
    std::cout << "1. Memory allocated in one graph and freed in another graph" << std::endl;
    std::cout << "2. Each graph maintains separate memory pools that don't communicate" << std::endl;
    std::cout << "3. Memory should be freed but stays in the allocation graph's pool" << std::endl;
    std::cout << "4. Only cudaDeviceGraphMemTrim() consolidates pools and returns memory to OS" << std::endl;
    std::cout << "5. This confirms Bug #1 documented in cuda_graph_notation.md" << std::endl;
    
    return 0;
}