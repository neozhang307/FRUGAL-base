# FRUGAL Optimization - TODO List

## ✅ Completed Tasks (November 2025)

### Infrastructure & Memory Management
- [x] Fixed CUDA graph memory leak (21GB) using cudaDeviceGraphMemTrim
- [x] Fixed iterator bug in memoryManager_v2.cu (reference vs copy issue)
- [x] Fixed missing executeGraph call in executor_v2.cu
- [x] Merged LU decomposition branch with task graph visualization
- [x] Created playground test for CUDA graph memory behavior
- [x] Documented CUDA graph memory management findings
- [x] **Restructured memory management code** (2025-11-17)
  - Moved MemoryManager from profiling/ to dedicated memory/ folder
  - Better architectural organization

### Offline Optimization Workflow
- [x] **Implemented complete offline optimization workflow** (2025-11-17)
  - Created `standaloneOptimizer` tool for CPU-only optimization
  - Implemented `tiledCholeskyAblation` with three modes:
    - Profile-only mode (saves to JSON)
    - Run-plan mode (loads and executes)
    - Normal mode (profile + optimize + execute)
  - Added `ProfilingContext` for managing dummy kernel handles

### Serialization Infrastructure
- [x] **Implemented comprehensive serialization** (2025-11-17)
  - `OptimizationInput` serialization (profiling data)
  - `OptimizationOutput` serialization (execution plans)
  - `FirstStepSolver::Output` serialization (task scheduling results)
  - All serialization functions in `optimizationSerializer.{hpp,cpp}`

### Ablation Study Support
- [x] **Enabled first/second step separation** (2025-11-17)
  - `--save-first-step=<path>` to save task scheduling
  - `--load-first-step=<path>` to skip task scheduling
  - Inline optimization in standaloneOptimizer (avoids linking issues)
  - Fixed argh command-line parsing (requires `--option=value` syntax)

### Visualization Tools (2025-11-21)
- [x] **DAG and Task Dependency Visualization**
  - Created `scripts/visualize_plan_dag.py` for execution plan visualization
  - Shows task nodes, prefetch/offload operations, and control nodes
  - A4 landscape layout with 3-row format
  - Array sizes displayed in GB on memory operations

## ✔️ Resolved/Proven Unnecessary
- [x] **Phase 1 (Task Ordering)** - Solved with Beam Search (configurable width=100 provides fast, good solutions)
- [x] **Phase 2 (Migration Scheduling)** - MIP solver improved with variable reduction using dependencies as constraints; Greedy scheduler and warmup MIP alternatives implemented
- [x] ~~**Tighten big-M constraints**~~ - **PROVEN USELESS**: Tested but showed no improvement
- [x] ~~**Additional Gurobi parameter tuning**~~ - **PROVEN USELESS**: Current settings (60s timeout, 10% MIP gap) are sufficient

## 🔧 Pending Tasks

### High Priority - Next Implementation
- [x] **Top-K Solutions for First Step** ✅ COMPLETED (2025-11-17)
  - Implemented `solveTopK()` method in FirstStepSolver
  - Extracts multiple solutions from beam search final states
  - Added `--top-k`, `--save-topk`, `--load-topk` flags to standaloneOptimizer
  - Serialization implemented in optimizationSerializer
  - Can extract and save multiple task orderings with data reuse scores

- [x] **Gurobi Solution Pool Support** ✅ COMPLETED (2025-11-17)
  - Implemented `solveWithPool()` method in SecondStepSolver
  - Uses `NextSolution()` API to iterate through Gurobi's solution pool
  - Extracts and saves all solutions as separate files (plan_sol0.json, plan_sol1.json, etc.)
  - Added `--use-pool` flag to standaloneOptimizer
  - Requires MIP solver (not GREEDY) for solution pool functionality
  - Typically finds 1-2 solutions with relaxed pool parameters
  - See ABLATION_README.md for usage guide

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

- [x] **TopK Clean Redesign** ✅ COMPLETED (2025-11-21)
  - **Implemented**: Three-stage decoupled pipeline
  - **Components Delivered**:
    1. `ProfilingSerializer` - Save/load profiling results ✅
    2. `Step1Serializer` - Save/load beam search Top-K candidates ✅
    3. Batch execution scripts for Top-K evaluation ✅
  - **Architecture Implemented**:
    - Stage 0: Profile once, save JSON, reuse forever ✅
    - Stage 1: Beam search Top-K (generates 100 task orderings) ✅
    - Stage 2: MIP refinement for each candidate ✅
  - **Results**: See `results/ablation/exp1/` for Top-100 study results

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

#### ✅ Completed Ablation Studies (November 2025)

- [x] **Ablation Exp 1: Top-K Task Ordering Performance Analysis** ✅ COMPLETED
  - **Results Location**: `results/ablation/exp1/`
  - **Configuration**: N=102400, T=4, Beam Width=100, Top-100 solutions
  - **Key Findings**:
    - Task ordering has minimal impact on final performance (<1% variation)
    - MIP solver effectively compensates for different task orderings
  - **Deliverables**: Top-100 analysis with GPU execution validation

- [x] **Ablation Exp 2: Beam Width Ablation Study** ✅ COMPLETED
  - **Results Location**: `results/ablation/exp2/`
  - **Key Findings**: Beam width 10 achieves 99% of optimal, solve time scales linearly

- [x] **Ablation Exp 3: Task Window Ablation Study** ✅ COMPLETED
  - **Results Location**: `results/ablation/exp3/`
  - **Tests**: Distance-based limits and time-based factors

- [x] **Saturation Analysis Study** ✅ COMPLETED
  - **Code Location**: `experiments/performance_validation/`
  - Validated model accuracy at different saturation levels

- [ ] ~~**Compute-Communication Overlap Study (Independence)**~~ ❌ NOT INCLUDED
  - **Status**: Blocked due to a bug in the implementation
  - **Original Goal**: Validate independence assumption between compute and data movement

#### Infrastructure (Completed)
- [x] **Warm Start Infrastructure** ✅ COMPLETED (2025-11-11)
- [x] **TopK Tool Development** ✅ COMPLETED (2025-11-21)

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

## 🧹 Codebase Cleanup for Open Source (TODO)

### High Priority - Remove from Git Tracking
- [ ] **Remove `results/` directory** - Experiment outputs should not be tracked
- [ ] **Remove internal documentation files**:
  - `TODO.md` - Internal task tracking
  - `TODO_TIMELINE.md` - Internal timeline
  - `VERIFIED_WORKING.md` - Internal verification notes
  - `TOPK_EXECUTION_SUMMARY.md` - Experiment results summary
  - `TOPK_GAPOVERLAP_RESULTS.md` - Experiment results

### Medium Priority - Review and Decide
- [ ] **Review documentation files**:
  - `ABLATION_README.md` - May contain useful info, consider consolidating
  - `EPSILON_REFINE.md` - Algorithm design notes
  - `GAP_OVERLAP_CRITICAL_FINDING.md` - Important finding, may keep
  - `METRICS.md` - Internal metrics docs
  - `PROGRAM.md` - Program structure notes
  - `VALIDATION_USAGE.md` - Usage docs
  - `cuda_graph_notation.md` - Technical docs
  - `execution_routine.md` - Internal notes
  - `secondStepSolver_MIP_Reference.md` - Algorithm reference

### Keep for Open Source
- [x] `CLAUDE.md` - AI guidance for vibe coding developers
- [x] `README.md` - Main project documentation
- [x] All source code in `memory/`, `optimization/`, `profiling/`, `utilities/`, `public/`
- [x] User applications in `userApplications/`
- [x] Build files (`CMakeLists.txt`, `Makefile`, `vcpkg.json`)
- [x] Scripts in `scripts/` (visualization tools)
- [x] Experiment scripts in `experiments/` (for reproducibility)

### Update .gitignore
- [ ] Add patterns for:
  ```
  results/
  *.pdf
  *.png
  *.csv
  *.log
  profile*.json
  *_plan.json
  *_output.json
  ```

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
