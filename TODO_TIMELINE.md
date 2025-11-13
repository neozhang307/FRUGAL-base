# FRUGAL Development Timeline - REVISED

**Last Updated**: November 2025
**Status**: Days 1-2 complete, Days 2-5 REVISED after ablation lessons

## Overview
**Original Goal**: Complete all code modifications within 5 days
**Reality**: Day 2 afternoon attempt (Gurobi solution pool) failed - pivot to clean redesign

### Revised Development Tracks:
1. **Track 1**: ✅ Minimal Memory & Warm Start (COMPLETE)
2. **Track 2**: 🔄 Top-K Solution Support (REVISED - multi-weight strategy instead of solution pool)
3. **Track 3**: ⏳ Design & Model Verification Infrastructure (Depends on Track 2)

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
- [x] **Gurobi Top-K solution pool setup** ❌ FAILED (2025-11-12)
  - Attempted: Direct solution pool integration in Step 2 MIP
  - Configuration: `PoolSolutions=10`, `PoolSearchMode=2`, `PoolGap=0.50`
  - **Problem**: Unreliable - returns 1-2 solutions instead of 10
  - **Root Cause**: Gurobi presolve reduces solution space, non-deterministic
  - **Decision**: REVERTED - pivot to clean redesign
  - **Lessons Learned**: See ABLATION.md for detailed analysis

---

### Day 3: Complete Top-K & Integration (ORIGINAL PLAN - OBSOLETE)
**Status**: ❌ Day 2 afternoon failure requires plan revision

~~**Focus: Extract multiple solutions, test warm start**~~

**What Happened**:
- Solution pool approach proved unreliable
- Discovered need for decoupled pipeline architecture
- Learned multi-weight strategy is more effective

**New Plan**: See "REVISED TIMELINE" section below

#### Morning (4 hours) - OBSOLETE
- ~~[ ] **Top-K solution extraction**~~ ❌ Not viable with solution pool
  - Problem: Can only extract 1-2 solutions, not 10
  - Cannot build reliable ablation studies on this

#### Afternoon (4 hours) - OBSOLETE
- ~~[ ] **Integration testing**~~ ❌ Cannot test unreliable feature

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

## REVISED TIMELINE (Post-Ablation)

**Current Status**: Foundation complete (warm start), pivot to clean TopK redesign

### Week 1-2: TopK Tool Implementation (HIGH PRIORITY)
**Goal**: Enable ablation studies for reviewer responses

#### Phase 1: Profiling Serialization (2-3 days)
- [ ] Create `ProfilingSerializer` class
  - Save profiling results (task graph, timings, array metadata) to JSON
  - Load profiling results from JSON
  - Ensure reproducibility (same profiling → same optimization results)
- [ ] Add execution mode: `--mode=PROFILE`
- [ ] Test: Profile once, reuse across multiple optimization runs
- **Deliverable**: Can profile once, iterate rapidly on optimization

#### Phase 2: Step1 Top-K Generation (2-3 days)
- [ ] Modify `BeamSearch` to return Top-K candidates (not just best)
- [ ] Create `Step1Serializer` class
  - Save Top-K beam search candidates to JSON
  - Include: task orderings, data reuse scores, estimated metrics
- [ ] Add execution mode: `--mode=STEP1 --top-k=10`
- [ ] Test: Generate 10 diverse candidates from beam search
- **Deliverable**: Step 1 produces Top-K candidates for exploration

#### Phase 3: Step2 Multi-Weight Refinement (2-3 days)
- [ ] Create `MultiRefinementSolver` class
  - For each Step 1 candidate, refine with multiple weight configs
  - Use existing warm start infrastructure (WARMUP_IMP.md)
- [ ] Create `WarmStartConverter` to convert Step1 → MIP hints
- [ ] Define weight strategies:
  - Pure speed: `{runtime: 1.0, migration: 0.0}`
  - Balanced: `{runtime: 0.5, migration: 0.5}`
  - Minimal migration: `{runtime: 0.0, migration: 1.0}`
  - Memory-aware: `{runtime: 0.8, migration: 0.1, memory: 0.1}`
- [ ] Add execution mode: `--mode=STEP2 --refine-top=3`
- [ ] Test: Refine top-3 candidates with 5 weight configs each
- **Deliverable**: 15 refined solutions (3 × 5) for analysis

#### Phase 4: Integration & Testing (1-2 days)
- [ ] Create `PipelineController` for mode-based execution
- [ ] Test full pipeline: Profile → Step1 Top-K → Step2 Refinement
- [ ] Verify warm start works with Step1 → Step2
- [ ] Ensure fair comparison (all use same profiling data)
- **Deliverable**: Complete TopK tool ready for ablation studies

**Documentation**: See TOPK.md for detailed architecture and API specs

---

### Week 3: Run Ablation Studies (Using TopK Tool)

#### Experiment 3: Data Reuse Validation
- [ ] Generate 50+ orderings with Step1 Top-K
- [ ] Measure correlation: data reuse score vs actual performance
- [ ] Create plots for reviewer responses

#### Experiment 4: Beam Width Analysis
- [ ] Test beam widths: [1, 10, 50, 100, 200, 500]
- [ ] Plot quality vs speed tradeoff
- [ ] Justify current beam width choice (100)

#### Experiment 6: Solution Diversity
- [ ] Generate solutions with multi-weight refinement
- [ ] Measure Hamming distance, plot Pareto frontiers
- [ ] Show diversity of solution space

**Deliverable**: Ablation study results for reviewer responses

---

### Week 4+: Epsilon Refinement (Production Feature)
**Priority**: Medium (after TopK for reviewers)

- [ ] Implement `EpsilonRefiner` class
- [ ] Add runtime constraint support to MIP solver
- [ ] Integrate into main workflow
- **Goal**: Give users explicit control over runtime-migration tradeoff

**Documentation**: See EPSILON_REFINE.md for complete design

---

## Success Criteria (REVISED)

### Foundation (COMPLETE ✅)
- [x] Warm start reduces MIP solve time by >30% (achieved 5x speedup)
- [x] Minimal memory calculator implemented
- [x] Greedy scheduler with 81% memory reduction

### TopK Tool (HIGH PRIORITY ⏳)
- [ ] Can profile once, reuse across all experiments (fair comparison)
- [ ] Can generate Step1 Top-K (10+ diverse candidates)
- [ ] Can refine each candidate with multiple weight strategies
- [ ] Warm start works for Step1 → Step2
- [ ] All results exported to JSON with metadata

### Ablation Studies (DEPENDS ON TOPK ⏳)
- [ ] Data reuse correlation measured and plotted
- [ ] Beam width tradeoff analysis complete
- [ ] Solution diversity metrics calculated
- [ ] Reviewer questions answered with data

### Production Features (AFTER TOPK ⏳)
- [ ] Epsilon refinement implemented and tested

## Risk Mitigation (UPDATED)

### What We Learned
- ✅ Solution pool unreliable → Use multi-weight strategy instead
- ✅ Profiling variability → Save and reuse profiling data
- ✅ Tight coupling → Decouple stages for rapid iteration

### Current Risks
- **Risk**: TopK implementation takes longer than 2 weeks
  - **Mitigation**: Phase 1-2 (profiling + Step1) are simpler, can deliver partial tool
- **Risk**: Reviewer questions change
  - **Mitigation**: Modular design allows adding new experiments easily
- **Risk**: Multi-weight strategy doesn't provide enough diversity
  - **Mitigation**: Can try more weight combinations (tested approach shows 3 distinct types)