# FRUGAL Optimization - TODO List

## ✅ Completed Tasks
- [x] Fixed CUDA graph memory leak (21GB) using cudaDeviceGraphMemTrim
- [x] Fixed iterator bug in memoryManager_v2.cu (reference vs copy issue)
- [x] Fixed missing executeGraph call in executor_v2.cu
- [x] Merged LU decomposition branch with task graph visualization
- [x] Created playground test for CUDA graph memory behavior
- [x] Documented CUDA graph memory management findings

## ✔️ Resolved/Proven Unnecessary
- [x] **Phase 1 (Task Ordering)** - Solved with Beam Search (configurable width=100 provides fast, good solutions)
- [x] **Phase 2 (Migration Scheduling)** - MIP solver improved with variable reduction using dependencies as constraints; Greedy scheduler and warmup MIP alternatives implemented
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
- [x] **Calculate Minimal Memory Usage Bound** ✅ COMPLETED (2025-11-11)
  - Created standalone `MinimalMemoryCalculator` class in `optimization/minimalMemoryCalculator.{hpp,cpp}`
  - Calculates theoretical minimum memory = max(sum of arrays needed per task)
  - Supports single-stage and multi-stage inputs
  - Integrated into `secondStepSolver.cpp` with detailed result output
  - Provides memory reduction potential, critical task identification, and per-task analysis

- [x] **Implement Greedy Scheduler** ✅ COMPLETED (2025-11-11)
  - Created `GreedyScheduler` class in `optimization/strategies/greedyScheduler.{hpp,cpp}`
  - Two modes implemented:
    - **MIN_MEMORY**: Prefetch per task, offload everything (achieves 81% memory reduction)
    - **MAX_PERFORMANCE**: Prefetch all at start, keep everything (maximum performance)
  - Configuration via `config.json`: `secondStepSolverType` and `greedySchedulerMode`
  - **Performance**: ~7,500x faster than MIP solver (<0.01s vs 60s)
  - **Results on tiledCholesky (n=102400)**: 40 prefetches, 40 offloads, 81.25% memory reduction
  - **Verification**: PASSED with zero error
  - Integrated into `secondStepSolver.cpp` with mode selection
  - Location: `optimization/strategies/greedyScheduler.{hpp,cpp}`

- [x] **Implement Warm Start for MIP Solver Using Greedy Solution** ✅ COMPLETED (2025-11-11)
  - Implemented `generateWarmStart()` method in `GreedyScheduler` class
  - Automatically selects warm start mode based on memory constraints:
    - If `maxPeakMemoryUsageInMiB < totalMemory`: Uses MIN_MEMORY greedy mode
    - If `maxPeakMemoryUsageInMiB >= totalMemory`: Uses MAX_PERFORMANCE greedy mode
  - Converts greedy schedule to Gurobi variable format (I, p, o, x, y variables)
  - Applies variable hints to MIP solver using OR-Tools SetInteger() API
  - **Results**: At most ~10-20% reduction in Step 2 preprocessing (MIP solve) time; no change to final schedule quality
  - Greedy provides feasible (not necessarily optimal) solution as starting point
  - Configuration: Set `secondStepSolverType = "GREEDY_WARMSTART"` in config.json
  - Location: `greedyScheduler.cpp::generateWarmStart()` and `secondStepSolver.cpp`
  - Documentation: See WARMUP_IMP.md

#### Top-K Solution Generation (Ablation Study Tool)
- [x] **TopK Solution Pool - First Attempt** ❌ FAILED (2025-11-12)
  - Attempted: Direct Gurobi solution pool integration in Step 2 MIP
  - Configuration: `PoolSolutions=10`, `PoolSearchMode=2`, `PoolGap=0.50`
  - **Problem**: Unreliable - returns 1-2 solutions instead of 10
  - **Root Cause**: Gurobi presolve reduces solution space, non-deterministic results
  - **Lessons Learned**:
    - Multi-weight strategy works better than solution pool
    - Need decoupled pipeline (profile once, iterate rapidly)
    - Both Step 1 and Step 2 Top-K are valuable
  - **Status**: Reverted, clean redesign planned
  - Documentation: See ABLATION.md for detailed failure analysis

- [ ] **TopK Clean Redesign** ⏳ HIGH PRIORITY (For reviewers)
  - **Why**: Reviewers require ablation studies to validate design decisions
  - **Approach**: Three-stage decoupled pipeline + multi-weight strategy
  - **Components Needed**:
    1. `ProfilingSerializer` - Save/load profiling results
    2. `Step1Serializer` - Save/load beam search Top-K candidates
    3. `MultiRefinementSolver` - Refine each candidate with multiple weight configs
    4. `PipelineController` - Mode-based execution (PROFILE/STEP1/STEP2)
  - **Architecture**:
    - Stage 0: Profile once, save JSON, reuse forever
    - Stage 1: Beam search Top-K (fast exploration, ~2s) → produces task orderings
    - Stage 2: Multi-weight refinement per candidate (precise, ~5-10s each)
      - Step 1 task ordering used as INPUT CONSTRAINT (not warm start)
      - Warm start from GreedyScheduler::generateWarmStart() (already implemented)
  - **Benefits**:
    - Fair comparisons (same profiling data)
    - Rapid iteration (don't re-run entire pipeline)
    - Reproducible results (no solution pool unreliability)
  - **Deliverables**: Ablation studies for reviewers
    - Data reuse metric validation
    - Beam width analysis
    - Solution diversity metrics
  - Documentation: See TOPK.md for complete design
  - Priority: **HIGH** - blocking paper acceptance

#### Epsilon-Constraint Refinement (Production Feature)
- [ ] **Epsilon Refinement Implementation** ⏳ Medium Priority (After TopK)
  - **Why**: Provide users explicit control over runtime-migration tradeoffs
  - **Problem**: Weight-based approach is hard to tune, discontinuous jumps
  - **Solution**: Two-step epsilon-constraint optimization
    1. Minimize runtime → get optimal_runtime
    2. Add constraint: runtime ≤ (1+ε) × optimal_runtime, minimize migrations
  - **Components Needed**:
    - `EpsilonRefiner` class with `refine()` and `refineFromSolution()`
    - `SecondStepSolver::SolveOptions` for runtime constraints
    - Configuration: `epsilonRefine.enabled`, `epsilonRefine.epsilon`
  - **Expected Result**: "Fastest schedule, then fewest migrations within 5% runtime"
  - Documentation: See EPSILON_REFINE.md for complete design
  - Priority: Medium - implement after TopK ablation tool

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

> **NOTE**: Experiments 3-6 require TopK tool implementation first (see "TopK Clean Redesign" above)

#### Core Design Validation (For Reviewer Responses)
- [ ] **Experiment 3: Data Reuse Metric Validation**
  - **Objective**: Verify if data reuse is suitable metric to bridge two-stage optimization
  - **Prerequisites**: ⚠️ Requires TopK tool with Step1 Top-K generation
  - **Methodology**:
    1. Use TopK tool to generate multiple task orderings (Step 1 Top-K)
    2. Sort by data reuse metric
    3. Refine each ordering with Step 2 MIP
    4. Measure actual execution performance
  - **Analysis**:
    - Plot: Data Reuse Score vs Final Performance
    - Question: Is higher reuse always better? What about order-2 reuse?
    - Identify correlation strength and threshold effects
  - **Implementation Needs**:
    - TopK tool with profiling serialization (reuse same profiling data)
    - Step 1 Top-K candidate generation (beam search output)
    - Batch execution framework

- [ ] **Experiment 4: Beam Search Effectiveness Analysis**
  - **Objective**: Validate beam search quality vs computational cost
  - **Prerequisites**: ⚠️ Requires TopK tool with configurable beam width
  - **Methodology**:
    - Use TopK tool to test beam sizes K = [1, 5, 10, 20, 50, 100, 200, 500]
    - For each K, measure:
      - Solution quality (data reuse score, final runtime)
      - Step 1 optimization time
      - Step 2 refinement time
  - **Analysis**:
    - Plot: Beam Size vs Solution Quality
    - Plot: Beam Size vs Optimization Time
    - Identify sweet spot for quality/time tradeoff
  - **Implementation Needs**:
    - Configurable beam width in TopK tool
    - Timing instrumentation at each stage
    - Decoupled pipeline for fair comparison

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

- [ ] **Experiment 6: Solution Space Diversity Analysis**
  - **Objective**: Understand diversity of alternative scheduling solutions
  - **Prerequisites**: ⚠️ Requires TopK tool (Step 2 multi-weight refinement)
  - **Methodology**:
    - Use TopK tool to generate diverse solutions via multi-weight strategy
    - For each Step 1 candidate, refine with different weight configs:
      - Pure speed: `{runtime: 1.0, migration: 0.0}`
      - Balanced: `{runtime: 0.5, migration: 0.5}`
      - Minimal migration: `{runtime: 0.0, migration: 1.0}`
    - Test each solution's actual performance
    - Measure diversity: Hamming distance, Pareto frontiers
  - **Analysis**:
    - Plot: Runtime vs Migrations (Pareto frontier)
    - Hamming distance between solutions
    - Identify solution clusters and trade-off regions
  - **Implementation Needs**:
    - TopK tool with multi-weight refinement (see TOPK.md)
    - NOT Gurobi solution pool (unreliable - see ABLATION.md)

#### Infrastructure Requirements
- [x] **Warm Start Infrastructure** ✅ COMPLETED (2025-11-11)
  - Foundation for TopK tool (see WARMUP_IMP.md)

- [ ] **TopK Tool Development** ⏳ HIGH PRIORITY
  - **Components Needed** (see "TopK Clean Redesign" above):
    1. **ProfilingSerializer**: Save/load profiling results
    2. **Step1Serializer**: Save/load beam search Top-K
    3. **MultiRefinementSolver**: Refine with multiple weights (uses existing GreedyScheduler for warm start)
    4. **PipelineController**: Mode-based execution

  - **Note**: Warm start already implemented via `GreedyScheduler::generateWarmStart()` (see WARMUP_IMP.md)
    - Step 1 output (task ordering) = CONSTRAINT for Step 2, not warm start source
    - Warm start = Greedy migration schedule for the given task ordering

  - **Enables All Ablation Studies**:
    - Experiment 3: Data reuse validation
    - Experiment 4: Beam search effectiveness
    - Experiment 6: Solution diversity

  - **Estimated Effort**: 1-2 weeks for complete implementation
  - Documentation: See TOPK.md for detailed design

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
