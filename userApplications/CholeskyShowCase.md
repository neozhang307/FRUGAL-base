# Tiled Cholesky Decomposition Implementations Showcase

## Overview

This document provides a comprehensive comparison of the different tiled Cholesky decomposition implementations in the FRUGAL framework. Each implementation demonstrates different aspects of CUDA graph construction, memory optimization, and execution strategies.

## Available Implementations

### 1. tiledCholesky.cu - **Baseline Implementation**
The standard implementation with explicit dependency management and full control over graph construction.

**Key Features:**
- Custom `TiledCholeskyGraphCreator` class for explicit dependency tracking
- Manual tile dependency management with `beginCaptureOperation`/`endCaptureOperation`
- Full control over graph construction process
- Supports both optimized and non-optimized execution modes
- Integrated with TaskManager_v2 for task execution

**Architecture:**
```cpp
// Custom graph creator with explicit tile tracking
auto tiledCholeskyGraphCreator = std::make_unique<TiledCholeskyGraphCreator>(s, graph);

// Explicit dependency specification
tiledCholeskyGraphCreator->beginCaptureOperation(
    {k, k},         // Tile to write
    {{k, k}}        // Tiles to read
);
```

**Use Cases:**
- When you need fine-grained control over dependencies
- Debugging complex dependency patterns
- Educational purposes to understand graph construction

**Execution:**
```bash
./build/userApplications/tiledCholesky [N] [T] [optimize: 0/1] [verify: 0/1]
# Example: ./build/userApplications/tiledCholesky 4096 8 1 1
```

---

### 2. tiledCholeskyMemoryOptimized.cu - **Production-Ready Implementation**
A clean, robust implementation with automatic dependency tracking and proper memory management.

**Key Features:**
- `PointerDependencyCudaGraphConstructor` for automatic dependency inference
- Robust memory manager integration with storage pointer resolution
- Clean memory lifecycle management with `freeManagedMemory()`
- Simplified single-phase execution
- Proper address resolution for offloaded memory

**Architecture:**
```cpp
// Automatic pointer-based dependency tracking
auto graphConstructor = std::make_unique<PointerDependencyCudaGraphConstructor>(s, graph);

// Dependencies inferred from input/output pointers
std::vector<void*> inputs = {static_cast<void*>(getMatrixBlock(k, k))};
std::vector<void*> outputs = {static_cast<void*>(getMatrixBlock(k, k))};
graphConstructor->beginCaptureOperation(inputs, outputs);
```

**Memory Management:**
```cpp
// Proper storage pointer resolution
void* srcAddress = memManager.getAddress(d_tiles[i + j * T]);
if (srcAddress == d_tiles[i + j * T]) {
    srcAddress = memManager.getStoragePtr(d_tiles[i + j * T]);
}

// Clean memory deallocation
memManager.freeManagedMemory(d_tile);
```

**Use Cases:**
- Production deployments
- General-purpose Cholesky decomposition
- When reliability and maintainability are priorities

**Execution:**
```bash
./build/userApplications/tiledCholeskyMemoryOptimized [N] [T]
# Example: ./build/userApplications/tiledCholeskyMemoryOptimized 8192 16
```

---

### 3. tiledCholeskyNaiveGraph.cu - **Simplified API Implementation**
The newest addition featuring a simplified graph construction API that automatically infers dependencies from task registration order.

**Key Features:**
- `generateNaiveGraph()` API - simplest graph construction method
- Automatic dependency inference from task registration order
- No explicit dependency specification required
- Integration with profileAndOptimize() for memory optimization
- Peak memory profiling during execution

**Architecture:**
```cpp
// Just register tasks in execution order
TaskManager_v2 tmanager_v2(true);

// Register tasks sequentially - dependencies auto-inferred
for (int k = 0; k < T; k++) {
    // Register POTRF task
    TaskId potrfTaskId = tmanager_v2.registerTask(...);

    // Register TRSM tasks
    for (int i = k + 1; i < T; i++) {
        TaskId trsmTaskId = tmanager_v2.registerTask(...);
    }
    // ... SYRK and GEMM tasks follow
}

// Generate graph automatically
cudaGraph_t graph = tmanager_v2.generateNaiveGraph(s);

// Optimize the naive graph
auto optimizedGraph = profileAndOptimize(graph);
```

**Use Cases:**
- Rapid prototyping and testing
- Simple dependency patterns
- When development speed is more important than fine control

**Execution:**
```bash
./build/userApplications/tiledCholeskyNaiveGraph
# Uses configuration from config.json
```

---

### 4. tiledCholeskyDomainEnlarge.cu - **Domain Enlargement Demonstration**
A specialized implementation demonstrating optimization plan reuse across different problem sizes.

**Key Features:**
- Two-phase execution: optimize small, execute large
- Domain enlargement with `reregisterManagedArrayWithLargerCPUData()`
- Optimization plan preservation across problem sizes
- Comprehensive performance profiling with FLOPS calculation
- Comparison with cuSOLVER direct method
- Timer class for detailed phase-by-phase timing

**Architecture:**
```cpp
// Phase 1: Optimize with small domain
size_t N_small = 2048;  // Fits in GPU memory
auto optimizedGraph = profileAndOptimize(smallGraph);

// Phase 2: Enlarge domain
size_t N_large = 8192;  // May exceed GPU memory
memManager.reregisterManagedArrayWithLargerCPUData(
    d_tiles[idx], h_tiles_large[idx], tileSize_large
);

// Phase 3: Execute with large domain using small optimization plan
executeOptimizedGraphColdStart(optimizedGraph, ...);
```

**Performance Metrics:**
- Theoretical FLOPS calculation: N³/3 + O(N²)
- GFLOPS performance reporting
- Phase-by-phase timing breakdown
- Memory usage profiling with PeakMemoryUsageProfiler

**Use Cases:**
- Problems too large to optimize directly
- Research on optimization plan transferability
- Benchmarking memory optimization strategies
- Conference paper evaluations (CGO26)

**Execution:**
```bash
./build/userApplications/tiledCholeskyDomainEnlarge
# Configuration via config.json or command line
```

---

## Comparison Matrix

| Feature | tiledCholesky | MemoryOptimized | NaiveGraph | DomainEnlarge |
|---------|--------------|-----------------|------------|---------------|
| **Dependency Management** | Manual/Explicit | Automatic (Pointer) | Automatic (Order) | Manual |
| **Graph Construction API** | Custom Creator | PointerDependency | generateNaiveGraph() | Custom |
| **Code Complexity** | Medium | Low | Lowest | Highest |
| **Memory Management** | Standard | Advanced | Standard | Re-registration |
| **Production Ready** | ✓ | ✓✓✓ | ✓ | Research |
| **Fine Control** | ✓✓✓ | ✓✓ | ✓ | ✓✓ |
| **Development Speed** | Medium | Fast | Fastest | Slow |
| **Performance Metrics** | Basic | Basic | Basic | Comprehensive |

## Implementation Details

### Algorithm Structure
All implementations follow the standard tiled Cholesky decomposition:
1. **POTRF**: Cholesky factorization of diagonal tile
2. **TRSM**: Triangular solve for tiles below diagonal
3. **SYRK**: Symmetric rank-k update for diagonal tiles
4. **GEMM**: General matrix multiplication for off-diagonal tiles

### Memory Optimization Strategy
- Register all tiles with MemoryManager
- Profile memory usage patterns
- Apply optimization (offloading/prefetching)
- Execute with optimized memory movement

### Verification
All implementations support verification using cuSOLVER:
- Generate symmetric positive definite matrix
- Run tiled implementation
- Compare with cuSOLVER reference
- Check numerical accuracy

## Selection Guidelines

### Choose **tiledCholesky** when:
- You need explicit control over dependencies
- Debugging complex dependency patterns
- Learning about CUDA graph construction

### Choose **tiledCholeskyMemoryOptimized** when:
- Deploying to production
- Need robust memory management
- Want clean, maintainable code
- **Recommended for most use cases**

### Choose **tiledCholeskyNaiveGraph** when:
- Rapid prototyping is priority
- Dependencies follow natural execution order
- Simplicity is more important than control

### Choose **tiledCholeskyDomainEnlarge** when:
- Working with very large matrices
- Researching optimization transferability
- Need to reuse optimization plans
- Benchmarking for conference papers

## Building and Running

### Build All Implementations
```bash
source ~/miniconda3/bin/activate && conda activate frugal
make config && make build
```

### Configuration
Edit `config.json` to set parameters:
```json
{
  "tiledCholesky": {
    "n": 8192,    // Matrix dimension
    "t": 16,      // Number of tiles
    "mode": 0     // Execution mode
  }
}
```

### Running Examples
```bash
# Baseline with optimization and verification
./build/userApplications/tiledCholesky 4096 8 1 1

# Memory optimized version
./build/userApplications/tiledCholeskyMemoryOptimized 8192 16

# Naive graph version (uses config.json)
./build/userApplications/tiledCholeskyNaiveGraph

# Domain enlargement demonstration
./build/userApplications/tiledCholeskyDomainEnlarge
```

## Common Issues and Solutions

### Out of Memory
- Use MemoryOptimized or DomainEnlarge implementations
- Reduce matrix size or tile count
- Enable memory offloading in configuration

### Verification Failures
- Ensure matrix is positive definite
- Check numerical stability for large matrices
- Verify tile size divides matrix dimension evenly

### Performance Issues
- Check optimization settings in config.json
- Verify CUDA graph is being properly constructed
- Use profiling tools to identify bottlenecks

---

*Last Updated: November 2024*
*Part of the FRUGAL Memory Optimization Framework*