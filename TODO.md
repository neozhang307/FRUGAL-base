# FRUGAL Optimization - TODO List

## ✅ Completed Tasks
- [x] Fixed CUDA graph memory leak (21GB) using cudaDeviceGraphMemTrim
- [x] Fixed iterator bug in memoryManager_v2.cu (reference vs copy issue)
- [x] Fixed missing executeGraph call in executor_v2.cu
- [x] Merged LU decomposition branch with task graph visualization
- [x] Created playground test for CUDA graph memory behavior
- [x] Documented CUDA graph memory management findings

## 🔧 Pending Optimization Tasks

### Phase 1: First Step Solver Performance
- [ ] **Fix Phase 1 optimization time consuming issue**
  - Current issue: First step solver takes excessive time for large task graphs
  - Location: `optimization/strategies/firstStepSolver.cpp`
  - Potential causes:
    - O(n²) or worse complexity in task ordering algorithm
    - Inefficient memory overlap calculations
    - Excessive debug output generation

### Phase 2: Second Step Solver Performance  
- [ ] **Fix Phase 2 optimization time consuming issue**
  - Current issue: Second step solver (Gurobi MIP) takes too long to converge
  - Location: `optimization/strategies/secondStepSolver.cpp`
  
**Priority 1: Core Performance Issues**
- [ ] **Reduce o[i][j][k] variable explosion**
  - Limit offload destinations to next K tasks instead of all future tasks
  - Impact: Reduces variables from O(n²m) to O(nKm)
  - Location: `secondStepSolver.cpp:299-344` (defineDecisionVariables)

- [ ] **Tighten big-M constraints** 
  - Change from `1000×originalRuntime` to `~10×originalRuntime`
  - Impact: Tighter LP bounds → faster convergence  
  - Location: `secondStepSolver.cpp:743` (addLongestPathConstraints)

**Priority 2: Solver Configuration**
- [ ] **Add configurable Gurobi parameters**
  - MIP gap tolerance, time limits, thread count
  - Impact: Stop when "good enough" solution found
  - Location: Add to `configurationManager.hpp` + `secondStepSolver.cpp:1198-1206`

- [ ] **Set variable branching priorities**
  - Optimize large arrays first, small arrays last
  - Impact: Focus solver effort on high-impact decisions
  - Location: `secondStepSolver.cpp:261-345` (after variable creation)

**Priority 3: Advanced Techniques**  
- [ ] **Implement warm start with feasible solution**
  - Provide initial solution from greedy heuristic
  - Impact: Start solver from good point instead of scratch
  - Location: Before `solver->Solve()` call

- [ ] **Add temporal decomposition for large problems**
  - Break into overlapping time windows for very large instances
  - Impact: Handle problems that don't fit in memory
  - Location: New wrapper around existing solver

### Memory Management System
- [ ] **Fix storage cleanup cudaFreeHost error**
  - **Issue**: `cudaFreeHost(storageAddress)` fails with CUDA error code=1
  - **Location**: `profiling/memoryManager_v2.cu:152` in `freeStorage()` method
  - **Root cause**: Storage addresses allocated inconsistently with how they're being freed
  - **Impact**: Causes cleanup errors after domain enlargement operations
  - **Investigation needed**:
    - Check how storage addresses are allocated in `allocateInStorage()`
    - Verify allocation method matches deallocation method
    - Consider tracking allocation type (cudaMallocHost vs cudaMalloc vs malloc)
  - **Priority**: Medium (doesn't affect core functionality but causes error messages)

### Code Refactoring and Organization
- [ ] **Reorganize tiledCholesky applications to reduce code duplication**
  - **Issue**: tiledCholesky.cu, tiledCholeskyDomainEnlarge.cu, tiledCholeskyMemoryOptimized.cu, and tiledCholeskyNaiveGraph.cu contain significant code duplication
  - **Redundant code includes**:
    - Matrix generation functions (generateRandomSymmetricPositiveDefiniteMatrix, makeMatrixSymmetric, addIdenticalMatrix)
    - Data initialization functions (initializeDeviceData)
    - Verification functions (verifyCholeskyDecompositionPartially, cuSOLVER comparison logic)
    - CUDA setup and cleanup code
    - Task registration patterns for POTRF/TRSM/SYRK/GEMM operations
  - **Proposed solution**:
    - Extract common functions into shared header/source files (`userApplications/common/`)
    - Create base class or utility functions for common tiled Cholesky operations
    - Keep application-specific logic (optimization vs naive graph vs domain enlargement) separate
  - **Benefits**: Reduced maintenance burden, consistent behavior across applications, easier testing
  - **Priority**: Medium (code quality improvement, no functional impact)

### Benchmarks and Validation
- [ ] **Add ResNet benchmark**
  - Implement ResNet model as new benchmark
  - Location: Create new file in `userApplications/`
  - Requirements:
    - Support different ResNet variants (ResNet-50, ResNet-101)
    - Implement tiled convolution operations
    - Add memory profiling hooks

- [ ] **Run decomposition validation experiment**
  - Validate two-phase decomposition effectiveness
  - Compare memory usage: baseline vs optimized
  - Measure performance overhead
  - Generate comparison plots and statistics

## 📝 Notes
- All critical memory bugs have been resolved
- System is stable for development
- Current focus: Performance optimization of the two-phase decomposition

## 🔬 Experimental Work - Naive Graph Generation API (2025-01-03)

**Status**: Experimental implementation attempted but not production-ready

### What was attempted:
- Implemented `generateNaiveGraph()` API in TaskManager_v2 to automatically generate CUDA graphs from registered tasks
- Goal: Eliminate manual graph construction by auto-detecting dependencies from input/output memory patterns
- Used PointerDependencyCudaGraphConstructor for automatic RAW/WAW/WAR dependency tracking

### What worked:
- ✅ Clean API design: `cudaGraph_t graph = taskManager.generateNaiveGraph();`
- ✅ Automatic dependency detection based on memory access patterns
- ✅ Successful graph generation and execution for simple operations
- ✅ Perfect accuracy for TRSM, SYRK, GEMM operations (off-diagonal elements)
- ✅ Proper integration with existing TaskManager_v2 infrastructure

### Fundamental limitations discovered:
- ❌ **Complex library operations fail**: cuSOLVER POTRF operations don't execute correctly in captured graphs
- ❌ **Architecture mismatch**: Batch task registration breaks the careful coordination between TaskManager, MemoryManager, and PointerDependencyCudaGraphConstructor
- ❌ **Address translation issues**: MemoryManager address translation during capture vs execution creates inconsistencies
- ❌ **Sequential vs batch execution**: Working implementations require one-by-one task execution during capture, not batch processing

### Technical findings:
- Graph structure is captured correctly (verified via .dot files)
- Graph executes without errors (info=0) but produces incorrect computational results for diagonal operations
- Off-diagonal elements computed perfectly, diagonal elements retain original matrix values instead of Cholesky factors
- Root cause: Batch approach doesn't maintain the orchestrated interaction required for complex cuBLAS/cuSOLVER operations

### Recommendation:
The concept is valuable but requires fundamental architectural changes beyond current scope. For complex library operations, continue using the existing manual PointerDependencyCudaGraphConstructor approach until a more sophisticated automatic system can be designed.

**Branch**: `naive-graph-generation` (changes reverted)
**Files involved**: `optimization/taskManager_v2.hpp`, test applications in `userApplications/`