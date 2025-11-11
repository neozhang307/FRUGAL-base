# FRUGAL Optimization - TODO List

## ✅ Completed Tasks
- [x] Fixed CUDA graph memory leak (21GB) using cudaDeviceGraphMemTrim
- [x] Fixed iterator bug in memoryManager_v2.cu (reference vs copy issue)
- [x] Fixed missing executeGraph call in executor_v2.cu
- [x] Merged LU decomposition branch with task graph visualization
- [x] Created playground test for CUDA graph memory behavior
- [x] Documented CUDA graph memory management findings

## ✔️ Resolved/Proven Unnecessary
- [x] **Phase 1 Optimization** - Solved with Beam Search (configurable width=100 provides fast, good solutions)
- [x] **Phase 2 Variable Reduction** - Reduced variance by using dependencies as constraints
- [x] ~~**Tighten big-M constraints**~~ - **PROVEN USELESS**: Tested but showed no improvement
- [x] ~~**Additional Gurobi parameter tuning**~~ - **PROVEN USELESS**: Current settings (60s timeout, 10% MIP gap) are sufficient

## 🔧 Pending Tasks

### Code Quality & Maintenance
- [ ] **Reorganize tiledCholesky applications**
  - Consolidate duplicate code from tiledCholesky.cu, tiledCholeskyDomainEnlarge.cu, tiledCholeskyMemoryOptimized.cu, tiledCholeskyNaiveGraph.cu
  - Extract common functions (matrix generation, verification, CUDA setup) into shared utilities
  - Priority: Low (code quality improvement, no functional impact)

### High Priority Optimizations

#### Second-Step Solver Improvements
- [x] **Calculate Minimal Memory Usage Bound** ✅ COMPLETED (2024-11-11)
  - Created standalone `MinimalMemoryCalculator` class in `optimization/minimalMemoryCalculator.{hpp,cpp}`
  - Calculates theoretical minimum memory = max(sum of arrays needed per task)
  - Supports single-stage and multi-stage inputs
  - Integrated into `secondStepSolver.cpp` with detailed result output
  - Provides memory reduction potential, critical task identification, and per-task analysis

- [ ] **Implement Heuristic-Based Warm Start for Gurobi**
  - Generate initial feasible solution using heuristic rules
  - **Keep/Offload Logic**:
    - Keep array if used by next task AND task after (lookahead=2)
    - Otherwise schedule offload immediately after use
    - Consider array size as weight factor
  - **Prefetch Scheduling**:
    - Try to prefetch at beginning of previous task
    - Fallback: If array still offloading or no previous task, prefetch at current task
    - Constraint: Ensure memory_at_prev + array_size <= minimal_memory
  - **Implementation Considerations**:
    - Track last_use_task for each array to avoid conflicts
    - Check memory constraints before scheduling prefetch
    - Ensure no race conditions between offload and prefetch
  - Impact: Significantly faster Gurobi convergence by starting from good solution
  - Location: `secondStepSolver.cpp` before `solver->Solve()`

### Potential Future Optimizations
- [ ] **Set variable branching priorities** (Not yet attempted)
  - Optimize large arrays first, small arrays last
  - Impact: Could potentially improve solver convergence
  - Location: `secondStepSolver.cpp:261-345` (after variable creation)

- [ ] **Add temporal decomposition for large problems**
  - Break into overlapping time windows for very large instances
  - Impact: Handle problems that don't fit in memory
  - Location: New wrapper around existing solver

### Known Issues

#### Critical Issues
- [ ] **Stage Logic Bug - Out-of-Core Initialization** 🔴
  - **Issue**: Different behavior between original staged implementation and current staged implementation
  - **Symptoms**: When considering initially out-of-core data, there's a logic difference causing incorrect behavior
  - **Affected Branches**:
    - `CGO26/master` - Quest integration (working correctly)
    - `CGO26/stage-showcase` - Staged Cholesky (has bug when integrated with Quest)
  - **Root Cause (Hypothesis)**: Different logic in how stages are connected between:
    - Original executor/optimizer implementation (used in master)
    - Current optimizer implementation (used in stage-showcase)
  - **Specific Problem**: How initially out-of-core arrays are handled at stage boundaries
  - **Investigation Needed**:
    - Compare stage connection logic in original vs current executor
    - Check data movement scheduling between stages
    - Verify memory state transitions at stage boundaries
  - **Impact**: HIGH - Causes incorrect results for staged applications with out-of-core data
  - **Priority**: High

#### Minor Issues
- [ ] **Fix storage cleanup cudaFreeHost error** (Minor)
  - **Issue**: `cudaFreeHost(storageAddress)` fails with CUDA error code=1
  - **Location**: `profiling/memoryManager_v2.cu:152` in `freeStorage()` method
  - **Impact**: Minor - causes cleanup errors after domain enlargement operations but doesn't affect functionality
  - **Priority**: Low

### Evaluation Experiments - Model Validation

#### Model Accuracy and Saturation Analysis
- [ ] **Experiment 1: Memory-Runtime Tradeoff & Saturation Study**
  - **Part A: Model Accuracy Verification**
    - Configure optimization to use memory savings as constraint (e.g., save 20%, 40%, 60% memory)
    - Optimize for runtime (set weightOfTotalRunningTime higher, weightOfPeakMemoryUsage=0)
    - Test Case: Tiled Cholesky with fixed large domain size
    - Verify if model accurately predicts performance impact

  - **Part B: Saturation Analysis**
    - Fix memory constraint to specific value (e.g., 50% memory savings)
    - Vary domain size: small (not saturating) to large (saturating)
    - Small domains may not generate enough parallelism to hide data movement
    - Large domains should saturate device and better hide overhead

  - **Measurements**:
    - Model-predicted overhead vs actual runtime overhead
    - Prediction error: |predicted_overhead - actual_overhead| / actual_overhead
    - Absolute runtime comparison: predicted vs actual execution time
    - **Heatmap Output**: Domain Size × Memory Constraint → Prediction Error (%)
    - Identify saturation point where prediction accuracy improves

  - **Expected Results**:
    - Small domains: **Higher prediction error** (model assumes saturation that doesn't exist)
    - Large domains: **Lower prediction error** (model assumptions match reality)
    - Model assumes device is saturated, but small workloads cannot achieve this
    - Prediction accuracy improves as domain size increases and device saturates

  - **Comparison**: Include Unified Memory (UM) as baseline

- [ ] **Experiment 2: Compute-Communication Overlap Study (Independence)**
  - **Setup**: Create stream-like workload with increasing compute intensity
    - Base: `A = B * C` (memory-bound)
    - Increase: `A = B * C * B * C` (more compute)
    - Continue: `A = B * C * B * C * B * C...` (compute-bound)
  - **Parameters**:
    - Use large enough matrices to ensure memory pressure
    - Gradually increase computation while keeping memory footprint constant
  - **Measurements**:
    - Compare predicted overhead vs actual overhead
    - Compare predicted runtime vs actual runtime
    - Track when prefetch/offload overhead becomes hidden by computation
    - Measure actual vs predicted overlap efficiency
    - Identify point where adding compute no longer helps
  - **Expected Results**:
    - Low compute: Higher prediction error (data movement dominates)
    - High compute: Lower prediction error (compute hides data movement as model expects)
    - Model accuracy improves as compute intensity increases
  - **Validation**: Verify independence assumption between compute and data movement
  - **Comparison**: UM performance as reference

#### Implementation Details
- [ ] **Create Evaluation Framework**
  - Location: `experiments/model_validation/`
  - Components:
    - Configuration generator for parameter sweep
    - Automated test runner for domain size × memory constraint matrix
    - Heatmap generation scripts
    - Performance comparison with UM baseline

- [ ] **Metrics to Collect**:
  - Model predicted: runtime, memory usage, overlap percentage
  - Actual measured: runtime, peak memory, PCIe bandwidth utilization
  - Derived: prediction error, saturation point, overlap efficiency

### Design Decision Validation Experiments

#### Core Design Validation
- [ ] **Experiment 3: Data Reuse Metric Validation**
  - **Objective**: Verify if data reuse is suitable metric to bridge two-stage optimization
  - **Methodology**:
    1. Generate ALL valid topological orderings (for small graphs)
    2. Sort by data reuse metric
    3. Create optimization plans for each ordering
    4. Systematically test all plans
  - **Analysis**:
    - Plot: Data Reuse Score vs Final Performance
    - Question: Is higher reuse always better? What about order-2 reuse?
    - Identify if there's a threshold where reuse stops mattering
  - **Coding Effort**: HIGH
    - Need to modify firstStepSolver to enumerate all solutions
    - Create systematic testing framework
    - ~3-4 days implementation

- [ ] **Experiment 4: Beam Search Effectiveness Analysis**
  - **Objective**: Validate beam search quality vs computational cost
  - **Methodology**:
    - Test beam sizes K = [1, 5, 10, 20, 50, 100, 200, 500]
    - For each K, measure:
      - Solution quality (data reuse score)
      - Optimization time
      - Final runtime performance
  - **Analysis**:
    - Plot: Beam Size vs Solution Quality
    - Plot: Beam Size vs Optimization Time
    - Identify sweet spot for quality/time tradeoff
  - **Coding Effort**: MEDIUM
    - Parameterize beam width in config
    - Add timing instrumentation
    - ~2 days implementation

- [ ] **Experiment 5: Window Size Impact Study**
  - **Objective**: Understand preprocessing time vs performance tradeoff
  - **Current Issue**: Window size not clearly defined in codebase
  - **Proposed Definition**:
    - Lookahead/lookback distance for prefetch/offload
    - Or task grouping size for optimization
  - **Methodology**:
    - Vary window size parameters
    - Measure preprocessing time and final performance
  - **Analysis**:
    - Plot: Window Size vs Preprocessing Time
    - Plot: Window Size vs Runtime Performance
    - Find optimal window configuration
  - **Coding Effort**: MEDIUM-HIGH
    - Need to clarify window concept in code
    - May require optimizer modifications
    - ~2-3 days implementation

- [ ] **Experiment 6: Top-K Schedule Analysis**
  - **Objective**: Explore alternative scheduling solutions
  - **Methodology**:
    - Configure Gurobi to find top-K solutions
    - Test each solution's actual performance
    - Analyze diversity of solutions
  - **Implementation Needs**:
    - Gurobi solution pool feature
    - Multiple solution extraction
    - Performance testing framework
  - **Coding Effort**: HIGH
    - Gurobi API changes for solution pool
    - Solution management infrastructure
    - ~3-4 days implementation

#### Infrastructure Requirements
- [ ] **Evaluation Framework Development**
  - **Components Needed**:
    1. **Solution Enumerator**: Generate all/many task orderings
    2. **Batch Tester**: Run multiple configurations systematically
    3. **Metric Collector**: Gather all relevant metrics
    4. **Analysis Tools**: Generate plots and statistics

  - **Current Codebase Limitations**:
    - No support for enumerating multiple solutions
    - No batch testing infrastructure
    - Limited metric export capabilities
    - Gurobi configured for single solution only

  - **Total Estimated Effort**: 10-12 days for complete framework

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