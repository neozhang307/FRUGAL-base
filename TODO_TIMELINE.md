# FRUGAL Development Timeline - Code Implementation Focus (5 Days)

## Overview
**Goal**: Complete all code modifications within 5 days. Evaluation/data collection can run in parallel on multiple machines afterward.

### Three Development Tracks:
1. **Track 1**: Minimal Memory & Warm Start (Performance optimization)
2. **Track 2**: Top-K Solution Support (Gurobi solution pool)
3. **Track 3**: Design & Model Verification Infrastructure

---

## PHASE 1: CODE IMPLEMENTATION (5 Days)

### Day 1: Core Optimizations - Minimal Memory & Warm Start Part 1
**Focus: Start main performance improvements**

#### Morning (4 hours)
- [x] **Minimal memory calculation** ✅ COMPLETED (2025-11-11)
  - Created standalone `MinimalMemoryCalculator` class instead of adding to secondStepSolver
  - Calculates sum of memory for required arrays per task
  - Takes maximum across all tasks for theoretical lower bound
  - Exports detailed metrics (reduction potential, critical task, per-task memory)
  - **Deliverable**: Theoretical lower bound available and integrated

- [ ] **Metric export system** (Partially complete)
  - Modify firstStepSolver: Export data reuse score, timing
  - Modify secondStepSolver: Export memory usage, runtime
  - JSON output format
  - **Note**: MinimalMemoryCalculator already provides detailed metrics
  - **Deliverable**: Metrics accessible for analysis

#### Afternoon (4 hours)
- [x] **Greedy Scheduler - Core Implementation** ✅ COMPLETED (2025-11-11)
  - Created `GreedyScheduler` class with two modes
  - **MIN_MEMORY mode**: Prefetch per task, offload everything after each task
  - **MAX_PERFORMANCE mode**: Prefetch all arrays at start, no offloads
  - Simple and reliable algorithm (no complex lookahead)
  - **Deliverable**: Production-ready greedy scheduler

---

### Day 2: Complete Warm Start & Start Top-K
**Focus: Finish warm start, begin Gurobi modifications**

#### Morning (4 hours)
- [x] **Greedy Scheduler - Integration & Testing** ✅ COMPLETED (2025-11-11)
  - Added configuration support (`secondStepSolverType`, `greedySchedulerMode`)
  - Integrated into `secondStepSolver.cpp` with mode selection
  - Updated `CMakeLists.txt` for build system
  - **Testing Results**:
    - MIN_MEMORY: 81.25% memory reduction, <0.01s optimization time
    - MAX_PERFORMANCE: All arrays on device, zero data movement
    - Verification: PASSED on tiledCholesky (n=102400)
  - **Deliverable**: Tested and verified greedy scheduler

- [x] **Warm Start for MIP Solver** ✅ COMPLETED (2025-11-11)
  - Implemented `generateWarmStart()` in GreedyScheduler
  - Automatically selects mode based on memory constraints (MIN_MEMORY vs MAX_PERFORMANCE)
  - Converts greedy solution to Gurobi variable format (I, p, o, x, y)
  - Applies hints via OR-Tools SetInteger() API
  - **Note**: Effectiveness needs benchmarking - may help or hinder depending on problem
  - **Deliverable**: Warm start implementation complete, needs evaluation

#### Afternoon (4 hours)
- [ ] **Gurobi Top-K solution pool setup**
  - Add solution pool parameters to config.json:
    - `poolSolutions`: number of solutions to find
    - `poolSearchMode`: 0/1/2 (different strategies)
    - `poolGap`: acceptable suboptimality gap
  - Modify secondStepSolver initialization
  - **Deliverable**: Gurobi configured for multiple solutions

---

### Day 3: Complete Top-K & Integration
**Focus: Extract multiple solutions, test warm start**

#### Morning (4 hours)
- [ ] **Top-K solution extraction**
  - After solve(), extract all solutions from pool:
    ```cpp
    for (int i = 0; i < model.get(GRB_IntAttr_SolCount); i++) {
        model.set(GRB_IntParam_SolutionNumber, i);
        // Extract and store solution i
    }
    ```
  - Store each solution's decisions
  - Export to JSON with objective values
  - **Deliverable**: Can retrieve K solutions

#### Afternoon (4 hours)
- [ ] **Integration testing**
  - Test warm start with Gurobi
  - Verify warm start reduces solve time
  - Test Top-K extraction
  - Ensure all solutions are valid
  - **Deliverable**: Warm start + Top-K working together

---

### Day 4: Design Verification Features
**Focus: Enable experiments for validation**

#### Morning (4 hours)
- [ ] **Multiple ordering generation**
  - Modify firstStepSolver.cpp:
    - Add `generateRandomOrderings(int count)` method
    - Use random valid topological sorts
    - Calculate data reuse for each
    - Export all orderings with scores
  - **Deliverable**: Can generate N different orderings

#### Afternoon (4 hours)
- [ ] **Beam search parameterization**
  - Make beam width configurable in config.json
  - Add detailed timing for each width
  - Track solution quality at each level
  - Export convergence metrics
  - **Deliverable**: Full beam search instrumentation

---

### Day 5: Model Verification & Batch Infrastructure
**Focus: Complete experiment infrastructure**

#### Morning (4 hours)
- [ ] **Compute intensity workload**
  - Create experiments/compute_workload.cu:
    - Variable compute: A=B*C, A=B*C*D*E, etc.
    - Parameterized via config
    - Memory footprint control
  - **Deliverable**: Configurable compute workloads

- [ ] **Memory constraint configuration**
  - Add to config.json:
    - `memoryConstraintPercent`: target memory savings
    - `optimizationTarget`: "memory" or "runtime"
  - Modify optimizer constraints accordingly
  - **Deliverable**: Can optimize for different objectives

#### Afternoon (4 hours)
- [ ] **Batch runner script**
  - Python script: experiments/batch_runner.py
  - Features:
    - Read parameter sweep from config file
    - Generate config.json for each combination
    - Run experiments sequentially or in parallel
    - Collect results to CSV/JSON
  - Test with small parameter sweep
  - **Deliverable**: Automated experiment execution

---

## PHASE 2: EVALUATION & DATA COLLECTION (After Code Complete)
**Can run on multiple machines in parallel**

### Experiment Set 1: Optimization Improvements
```bash
# Test warm start effectiveness
./test_warm_start.sh
```
- Compare solve time with/without warm start
- Measure solution quality difference
- Runtime: 2-3 hours

### Experiment Set 2: Top-K Analysis
```bash
# Extract top 10 solutions
python batch_runner.py --experiment top_k --solutions 10
```
- Test actual performance of each solution
- Analyze solution diversity
- Runtime: 3-4 hours

### Experiment Set 3: Model Verification
```bash
# Saturation study
python batch_runner.py --experiment saturation \
  --domains 4096,8192,16384,32768 \
  --memory_constraints 30,50,70
```
- Generate prediction error heatmap
- Runtime: 4-6 hours

### Experiment Set 4: Design Verification
```bash
# Data reuse validation
python batch_runner.py --experiment data_reuse --orderings 100
```
- Test correlation with performance
- Runtime: 4-5 hours

```bash
# Beam search analysis
python batch_runner.py --experiment beam_search \
  --widths 1,10,50,100,200,500
```
- Quality vs time tradeoff
- Runtime: 3-4 hours

---

## Code Modifications Priority List

### Must Complete (Critical Path):
1. **Day 1**: ✅ Minimal memory calculation (COMPLETED 2025-11-11)
2. **Day 1-2**: ✅ Greedy scheduler implementation (COMPLETED 2025-11-11)
3. **Day 2**: ✅ Warm start for MIP solver using greedy solution (COMPLETED 2025-11-11)
4. **Day 2-3**: Gurobi Top-K solution pool
5. **Day 4**: Multiple ordering generation
6. **Day 5**: Batch runner for automation

### Nice to Have (Can simplify if needed):
- Detailed beam search instrumentation
- Compute workload generator
- Advanced prefetch scheduling in warm start

---

## Implementation Notes

### Gurobi Solution Pool Setup:
```cpp
// In secondStepSolver.cpp
void SecondStepSolver::configureSolutionPool() {
    model.set(GRB_IntParam_PoolSolutions,
              config.optimization.poolSolutions);
    model.set(GRB_IntParam_PoolSearchMode, 2); // Find K best
    model.set(GRB_DoubleParam_PoolGap,
              config.optimization.poolGap);
}

void SecondStepSolver::extractSolutions() {
    int nSolutions = model.get(GRB_IntAttr_SolCount);
    for (int k = 0; k < nSolutions; k++) {
        model.set(GRB_IntParam_SolutionNumber, k);
        // Extract solution k
        solutions[k] = extractCurrentSolution();
    }
}
```

### Warm Start Integration:
```cpp
// In secondStepSolver.cpp
void SecondStepSolver::setWarmStart() {
    auto heuristic = generateHeuristicSolution();
    for (auto& [var, value] : heuristic) {
        var.set(GRB_DoubleAttr_Start, value);
    }
}
```

---

## Success Criteria
- [ ] Warm start reduces Gurobi solve time by >30%
- [ ] Can extract and test top-10 solutions
- [ ] All metrics exported to JSON
- [ ] Batch runner can execute parameter sweeps
- [ ] Can generate 100+ valid orderings for testing

## Risk Mitigation
- If warm start is complex → Simplify prefetch logic
- If Top-K has issues → At minimum get top-3 working
- If batch runner delayed → Use simple shell scripts
- Focus on getting core features working over perfect implementation