# CUDA Graph Memory Management Issues and Solutions

This document records critical CUDA graph-related bugs encountered during the development of the FRUGAL memory optimization system and their solutions.

## Bug #1: Memory Leak in CUDA Graph Execution

### Problem Description
When CUDA graphs execute operations that involve asynchronous memory allocation (such as cuBLAS operations), the memory allocated internally by the graph is not automatically freed when the graph execution completes. This leads to significant memory leaks that can cause out-of-memory errors in subsequent operations.

### Symptoms
- CUDA error code 2 (`cudaErrorMemoryAllocation`) during memory allocation
- Memory usage increases significantly after graph execution (observed: ~21GB leak)
- GPU memory remains allocated even after `cudaGraphExecDestroy()` and `cudaGraphDestroy()`
- Memory debugging shows no change in free/used memory before and after cleanup operations

### Root Cause
CUDA graphs maintain **separate memory pools per graph**. When memory is allocated in one graph and freed in a different graph, the freed memory remains in the original allocation graph's pool and cannot be accessed by other graphs. This creates cross-graph memory isolation that leads to memory accumulation across multiple graphs.

### Solution
Use `cudaDeviceGraphMemTrim()` to explicitly return unused graph memory to the operating system:

```cpp
// After graph execution and cleanup
int currentDevice;
checkCudaErrors(cudaGetDevice(&currentDevice));
checkCudaErrors(cudaDeviceGraphMemTrim(currentDevice));
```

### Implementation Location
- File: `profiling/memoryManager_v2.cu`
- Function: `offloadRemainedManagedMemoryToStorage()`
- Added after graph execution and cleanup operations

### Alternative APIs
- `cuDeviceGraphMemTrim()` (CUDA Driver API) - equivalent functionality
- Note: Must be called per device, and only frees memory that is not currently in use

### Ineffective Solutions
**CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH does NOT solve cross-graph memory leaks:**
- This flag only frees unfreed allocations within the same graph before relaunching
- It cannot access memory pools from other graphs
- Cross-graph allocation/deallocation patterns remain problematic
- Verified through playground testing: memory usage identical with/without the flag

### Test Case
See `playground/consecutiveGraphMemoryTest.cu` for a comprehensive demonstration of:
- Cross-graph allocation/deallocation memory leak
- Memory pool isolation behavior
- AUTO_FREE_ON_LAUNCH flag limitations
- cudaDeviceGraphMemTrim() effectiveness

---

## Bug #2: Core Dump on Graph Destruction Without Execution

### Problem Description
If a CUDA graph is instantiated (`cudaGraphInstantiate()`) but never executed (`cudaGraphLaunch()`), calling `cudaGraphExecDestroy()` or `cudaGraphDestroy()` can result in a segmentation fault (core dump).

### Symptoms
- Segmentation fault during cleanup operations
- Core dump occurs specifically when destroying unexecuted graphs
- Problem manifests in scenarios where graph construction succeeds but execution is skipped due to errors or early returns

### Root Cause
CUDA graph execution objects (`cudaGraphExec_t`) may have incomplete initialization when never executed. The CUDA runtime expects certain internal state to be established through actual graph execution before cleanup can proceed safely.

### Solution
Ensure graphs are either:
1. **Always executed before destruction**, or
2. **Properly validated before instantiation**, or  
3. **Use try-catch blocks around graph operations** to handle incomplete states

#### Example Safe Pattern:
```cpp
cudaGraph_t graph;
cudaGraphExec_t graphExec;
bool graphExecuted = false;

try {
    checkCudaErrors(cudaGraphCreate(&graph, 0));
    // ... build graph operations ...
    checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    
    // Always execute if instantiated
    checkCudaErrors(cudaGraphLaunch(graphExec, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));
    graphExecuted = true;
    
} catch (...) {
    // Handle errors appropriately
}

// Safe cleanup
if (graphExecuted) {
    checkCudaErrors(cudaGraphExecDestroy(graphExec));
}
checkCudaErrors(cudaGraphDestroy(graph));
```

### Implementation Location
- Affects: `optimization/executor_v2.cu`
- Function: `executeOptimizedGraph()` and related graph execution functions
- Fixed by ensuring `executeGraph()` is always called when graph is successfully instantiated

---

## Additional Notes

### Memory Debugging Tips
1. Use `cudaMemGetInfo()` before and after graph operations to monitor memory usage
2. Enable CUDA memory debugging with `CUDA_LAUNCH_BLOCKING=1`
3. Use `nvidia-smi` to monitor GPU memory usage externally
4. Consider using CUDA memory profiling tools like Nsight Systems

### Performance Considerations
- `cudaDeviceGraphMemTrim()` should be called judiciously as it may impact performance if called too frequently
- Graph memory pools are designed to optimize repeated graph executions by reusing allocations
- Only call trim operations when memory usage is a concern or at the end of major computation phases

### Related Configuration
- Configuration setting `mergeConcurrentCudaGraphNodes` affects graph construction and may influence memory allocation patterns
- LU decomposition workloads (lu_test branch) use `mergeConcurrentCudaGraphNodes: false` for better memory management

---

## References
- CUDA Graph Memory Management Documentation
- CUDA Runtime API Reference: `cudaDeviceGraphMemTrim`
- FRUGAL Paper: Two-Phase Memory Optimization for GPU Applications