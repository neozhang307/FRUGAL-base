# Execution Routine Analysis

## Main Execution Flow
The system follows: Profile → Optimize/Reschedule → Execute Optimized Program

We focus on the **Execute Optimized Program** phase performance.

## Memory Management System

The system uses a **dual approach** for memory management:

### 1. User Code Interface (Address-Based)
```cpp
// User registers arrays by original pointer addresses
float* A = cudaMalloc(...);
registerManagedMemoryAddress(A, size);

// User gets current address mapping in executeRandomTask callback
ExecuteRandomTask callback = [](int taskId, std::map<void*, void*> addressMapping, cudaStream_t stream) {
    auto currentA = addressMapping[A];  // Get remapped address
    myKernel<<<>>>(currentA, ...);      // Use current location
};
```

### 2. Internal Optimization (ArrayId-Based)
```cpp
// OptimizationOutput stores ArrayIds (integers: 0, 1, 2, ...)
struct DataMovement {
    ArrayId arrayId;           // Internal array identifier
    Direction direction;       // hostToDevice or deviceToHost
};

// Executor uses ArrayIds for prefetch/offload operations
memManager.prefetchToDeviceAsync(dataMovement.arrayId, stream);
memManager.offloadFromDeviceAsync(dataMovement.arrayId, stream);
```

### 3. Memory Array Info Structure
```cpp
struct MemoryArrayInfo {
    void* managedMemoryAddress;  // Original user pointer (key for address mapping)
    void* deviceAddress;         // Current device location (nullptr if offloaded)
    void* storageAddress;        // Storage location (CPU/secondary GPU)
    size_t size;                 // Array size in bytes
};

// ArrayId is the index into memoryArrayInfos vector
```

### 4. Address Mapping Flow
1. **Registration**: `managedMemoryAddress` stores original user pointer
2. **Optimization**: Optimization plan references arrays by ArrayId
3. **Execution**: 
   - Internal operations use ArrayId → `memoryArrayInfos[arrayId]`
   - User callbacks get address mapping: `originalPtr → currentDeviceAddress`

## Domain Enlargement Strategy
For domain enlargement, we need to work with **addresses** (not ArrayIds) because:
- User code expects original pointer addresses to remain valid
- executeRandomTask callback receives address mappings
- Cannot change user code to use ArrayIds

## Current Execution Methods

### Single Execution: `executeOptimizedGraph()`
- **Function**: `executeGraph()` (lines 87-131)
- **Flow**: Instantiate → Upload → Launch → Measure
- **Performance**: Optimized for single-shot execution

### Repeated Execution: `executeOptimizedGraphRepeatedly()`
- **Function**: `executeGraphRepeatedly()` (lines 351-407)
- **Issue**: `cudaDeviceSynchronize()` in iteration loop (line 387)
- **Impact**: Kills GPU pipeline efficiency for iterative workloads
- **TODO**: Remove synchronization from iteration loop to maintain GPU pipeline

### Evaluation Execution: `executeOptimizedGraphForEvaluation()`
- **Function**: Lines 513-631
- **Issues**: 
  - Inlined graph construction instead of reusing patterns
  - Launches both initialization and main graphs sequentially (lines 596-597)

## Code Duplication Issues
- `buildOptimizedGraph()` vs `buildRepeatedExecutionGraph()`: ~400 lines duplicated
- Multiple execution variants with similar patterns
- **TODO**: Consolidate duplicate graph building logic

## Notes
- CUDA graphs use single stream but achieve concurrency internally
- Focus on main execution routine performance, not iterative loop optimization for now