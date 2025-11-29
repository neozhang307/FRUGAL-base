# Epsilon-Constraint Refinement - Production Feature

**Date**: November 2025
**Feature Type**: Production Feature (user-facing tradeoff control)
**Status**: Design Phase (implement after TopK)
**Priority**: Medium (after TopK ablation tool for reviewers)
**Goal**: Provide users explicit control over runtime-migration tradeoffs

---

## Development Context

**Feature Position in Timeline**:
1. ✅ Warm start infrastructure (WARMUP_IMP.md) - foundation complete
2. ❌ TopK first attempt (ABLATION.md) - lessons learned
3. ⏳ **CURRENT PRIORITY**: TopK redesign (TOPK.md) - HIGH PRIORITY for reviewers
4. ⏳ **THIS FEATURE**: Epsilon refinement - production enhancement

**Why After TopK**:
- TopK is required by reviewers for paper acceptance (higher priority)
- Epsilon refinement enhances user experience (important but not blocking paper)
- Both features are independent and can be developed in sequence

---

## Background: System Overview

**Context**: CUDA memory optimization framework for out-of-core execution:
- Schedules **prefetches** (CPU→GPU) and **offloads** (GPU→CPU) for data movement
- **Migrations**: Total number of prefetch + offload operations
- Optimizes for: memory usage, runtime, and number of migrations

**Current Two-Phase Optimization**:
1. **Step 1 (Task Ordering)**: Beam Search finds optimal task execution ordering
2. **Step 2 (Migration Scheduling)**: MIP Solver decides when to prefetch/offload arrays given the task order from Step 1, using weighted objective: `w₁×runtime + w₂×migrations`

**Problem with Weight-Based Approach**:
- Hard to know what weights give "fastest with fewest migrations"
- Small weight changes cause discontinuous solution jumps
- Cannot guarantee "within X% of optimal runtime"

**User Need**: "Give me the fastest schedule, then among schedules within 5% of that, pick the one with fewest migrations"

**This Document**: Epsilon-constraint refinement - explicit control over runtime-migration tradeoff instead of opaque weight tuning.

**Development Priority**: Medium (implement after TopK ablation tool for reviewers)

---

## Table of Contents

1. [Overview & Motivation](#overview--motivation)
2. [Epsilon-Constraint Method](#epsilon-constraint-method)
3. [Architecture Design](#architecture-design)
4. [API Specification](#api-specification)
5. [Configuration](#configuration)
6. [Implementation Phases](#implementation-phases)
7. [Usage Workflows](#usage-workflows)
8. [Testing Strategy](#testing-strategy)

---

## Overview & Motivation

### Problem Statement

After MIP optimization (Step 2), we have solutions optimized for a weighted combination of runtime and migrations. However, users often want:

1. **Minimal Runtime First**: Find the fastest possible schedule
2. **Then Minimize Migrations**: Among schedules with "near-optimal" runtime, pick the one with fewest migrations

This is a **multi-objective optimization** problem where runtime takes priority over migrations.

### Why Not Just Use Weights?

**Weight-based approach** (current):
```cpp
objective = w_runtime * runtime + w_migration * migrations
```
- **Problem**: Hard to know what weights will give "fastest with fewest migrations"
- **Issue**: Small weight changes can cause discontinuous jumps in solution quality
- **Limitation**: Cannot guarantee "within X% of optimal runtime"

**Epsilon-constraint approach** (proposed):
```
Step 1: minimize runtime
Step 2: add constraint runtime ≤ (1 + ε) * optimal_runtime
        then minimize migrations
```
- **Benefit**: Explicit control over runtime sacrifice
- **Guarantee**: "This is the schedule with fewest migrations among all schedules within 5% of optimal runtime"
- **Interpretability**: Clear tradeoff parameter (ε)

### Design Goals

1. **Modular**: Can be enabled/disabled independently of main optimization
2. **Reusable**: Can refine any existing Step 2 solution (not just fresh runs)
3. **Configurable**: User controls epsilon threshold (e.g., 0%, 5%, 10%)
4. **Non-invasive**: Doesn't change existing SecondStepSolver behavior when disabled

---

## Epsilon-Constraint Method

### Mathematical Formulation

**Step 1: Find Optimal Runtime**
```
minimize: z_T (total runtime)
subject to: [all original MIP constraints]
```
Output: `runtime_optimal`

**Step 2: Minimize Migrations with Runtime Constraint**
```
minimize: Σ migrations
subject to:
  [all original MIP constraints]
  z_T ≤ (1 + ε) * runtime_optimal    ← NEW CONSTRAINT
```
Output: Solution with minimal migrations within runtime budget

### Parameter: Epsilon (ε)

- **ε = 0.00**: Strict - only accept exactly optimal runtime (may not reduce migrations)
- **ε = 0.05**: Relaxed 5% - allow 5% runtime degradation to reduce migrations
- **ε = 0.10**: Relaxed 10% - allow 10% runtime degradation
- **ε = -1.0**: Disabled - no refinement

### Expected Behavior

Given a Step 2 solution with:
- Runtime: 100.0 seconds
- Migrations: 21

After epsilon refinement with ε=0.05:
- Runtime: ≤ 105.0 seconds (within 5% of optimal)
- Migrations: 11 (reduced by using runtime slack)

---

## Architecture Design

### Three-Component Design

```
┌──────────────────────────────────────────────────────────────┐
│  COMPONENT 1: EpsilonRefiner (Core Logic)                   │
│  ────────────────────────────────────────────────────────────│
│  Class: EpsilonRefiner                                       │
│  Purpose: Execute two-step epsilon-constraint optimization   │
│  Input: SecondStepSolver::Input + epsilon value              │
│  Output: RefinedSolution with runtime constraint             │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│  COMPONENT 2: Solution Refiner (Wrapper)                    │
│  ────────────────────────────────────────────────────────────│
│  Function: refineExistingSolution()                          │
│  Purpose: Refine already-computed Step 2 solutions           │
│  Input: SecondStepSolver::Output + epsilon                   │
│  Output: RefinedSolution                                      │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│  COMPONENT 3: Pipeline Integration                          │
│  ────────────────────────────────────────────────────────────│
│  Location: tiledCholesky.cu main()                           │
│  Purpose: Optional post-processing after Step 2             │
│  Trigger: config.optimization.epsilonRefine.enabled          │
└──────────────────────────────────────────────────────────────┘
```

### Component Interactions

**Scenario 1: Fresh Run with Auto-Refine**
```
tiledCholesky.cu
  → SecondStepSolver::solve()        // Normal Step 2
  → Check: config.epsilonRefine.enabled?
  → If yes: EpsilonRefiner::refine()  // Automatic refinement
  → Export both original and refined
```

**Scenario 2: Offline Refinement of Saved Solution**
```
User loads solution_0.json
  → refineExistingSolution(loaded_output, epsilon=0.05)
  → EpsilonRefiner::refineFromSolution()
  → Export refined_solution.json
```

---

## API Specification

### EpsilonRefiner Class

```cpp
namespace memopt {

/**
 * @class EpsilonRefiner
 * @brief Performs epsilon-constraint refinement on MIP solutions
 *
 * Two-step optimization:
 * 1. Find optimal runtime (minimize z_T)
 * 2. Add runtime constraint, minimize migrations
 */
class EpsilonRefiner {
 public:
  /**
   * @struct Config
   * @brief Configuration for epsilon refinement
   */
  struct Config {
    double epsilon = 0.05;           // Runtime slack tolerance (5% default)
    bool enabled = false;            // Enable refinement
    int timeoutSeconds = 60;         // Max time for each step
    double mipGap = 0.10;            // Acceptable optimality gap

    Config() = default;
  };

  /**
   * @struct RefinedSolution
   * @brief Output from epsilon refinement
   */
  struct RefinedSolution {
    SecondStepSolver::Output solution;     // Final refined solution

    // Step 1 results
    double optimalRuntime;                 // Best possible runtime
    SecondStepSolver::Output runtimeOptimalSolution;

    // Step 2 results
    double runtimeBound;                   // (1 + ε) * optimalRuntime
    int migrationsBefore;                  // Migrations in Step 1 solution
    int migrationsAfter;                   // Migrations after refinement
    double migrationReduction;             // Percentage reduction

    // Metadata
    bool successful;
    std::string errorMessage;
    double step1SolveTime;
    double step2SolveTime;
  };

  /**
   * @brief Constructor with configuration
   */
  explicit EpsilonRefiner(const Config& config);

  /**
   * @brief Perform fresh epsilon refinement
   * @param input Original problem input (task graph, arrays, etc.)
   * @return RefinedSolution with both steps' results
   *
   * This executes both steps from scratch:
   * - Step 1: Minimize runtime
   * - Step 2: Minimize migrations with runtime constraint
   */
  RefinedSolution refine(const SecondStepSolver::Input& input);

  /**
   * @brief Refine an existing Step 2 solution
   * @param existingSolution Previously computed solution
   * @param input Original problem input
   * @return RefinedSolution using existing runtime as bound
   *
   * This skips Step 1 and uses existing solution's runtime:
   * - Extract runtime from existingSolution
   * - Apply epsilon to get runtime bound
   * - Solve Step 2 only (minimize migrations)
   */
  RefinedSolution refineFromSolution(
      const SecondStepSolver::Output& existingSolution,
      const SecondStepSolver::Input& input);

 private:
  Config config_;

  /**
   * @brief Step 1: Optimize for minimal runtime
   * @param input Problem data
   * @return Solution optimized purely for runtime
   */
  SecondStepSolver::Output optimizeRuntime(
      const SecondStepSolver::Input& input);

  /**
   * @brief Step 2: Minimize migrations with runtime constraint
   * @param input Problem data
   * @param runtimeUpperBound Maximum allowed runtime
   * @return Solution with minimal migrations within runtime budget
   */
  SecondStepSolver::Output minimizeMigrations(
      const SecondStepSolver::Input& input,
      double runtimeUpperBound);

  /**
   * @brief Calculate total migrations in a solution
   */
  int countMigrations(const SecondStepSolver::Output& solution) const;

  /**
   * @brief Calculate total runtime from solution
   */
  double calculateRuntime(
      const SecondStepSolver::Output& solution,
      const SecondStepSolver::Input& input) const;
};

}  // namespace memopt
```

### Standalone Refine Function

```cpp
namespace memopt {

/**
 * @brief Refine an existing solution with epsilon-constraint method
 * @param existingSolution Solution to refine (from JSON or previous run)
 * @param input Original problem input
 * @param epsilon Runtime slack tolerance (0.0 = strict, 0.1 = 10% slack)
 * @return Refined solution with reduced migrations
 *
 * Usage:
 *   auto loaded = loadSolutionFromJSON("solution_0.json");
 *   auto refined = refineExistingSolution(loaded, input, 0.05);
 *   saveSolutionToJSON(refined.solution, "refined_solution.json");
 */
EpsilonRefiner::RefinedSolution refineExistingSolution(
    const SecondStepSolver::Output& existingSolution,
    const SecondStepSolver::Input& input,
    double epsilon = 0.05);

}  // namespace memopt
```

### SecondStepSolver Extensions

**Add Runtime Constraint Support**:

```cpp
// secondStepSolver.hpp
class SecondStepSolver {
 public:
  struct SolveOptions {
    bool optimizeRuntime = true;        // If false, minimize migrations
    double runtimeUpperBound = -1.0;    // -1 = no constraint, >0 = add constraint

    SolveOptions() = default;
  };

  /**
   * @brief Solve with custom options (for epsilon refinement)
   */
  Output solve(const Input& input, const SolveOptions& options);

  // Existing method (unchanged)
  Output solve(const Input& input);
};
```

**Implementation Changes**:

```cpp
// secondStepSolver.cpp

Output SecondStepSolver::solve(const Input& input, const SolveOptions& options) {
  // ... [variable creation as before]

  // Objective: Choose based on options
  MPObjective* objective = solver->MutableObjective();

  if (options.optimizeRuntime) {
    // Original behavior: weighted combination
    objective->SetCoefficient(z[numTasks - 1], config.weightOfRuntime);

    for (int i = 0; i < numTasks; i++) {
      for (int j = 0; j < numArrays; j++) {
        objective->SetCoefficient(p[i][j], config.weightOfMigrations);
        for (int k = 0; k < numTasks; k++) {
          objective->SetCoefficient(o[i][j][k], config.weightOfMigrations);
        }
      }
    }
  } else {
    // Epsilon refinement: ONLY minimize migrations
    for (int i = 0; i < numTasks; i++) {
      for (int j = 0; j < numArrays; j++) {
        objective->SetCoefficient(p[i][j], 1.0);
        for (int k = 0; k < numTasks; k++) {
          objective->SetCoefficient(o[i][j][k], 1.0);
        }
      }
    }
  }

  objective->SetMinimization();

  // Runtime constraint: Add if specified
  if (options.runtimeUpperBound > 0.0) {
    auto runtimeConstraint = solver->MakeRowConstraint(
      -solver->infinity(),
      options.runtimeUpperBound
    );
    runtimeConstraint->SetCoefficient(z[numTasks - 1], 1.0);

    LOG_TRACE_WITH_INFO("Added runtime constraint: z_T ≤ %.6f",
                        options.runtimeUpperBound);
  }

  // ... [rest of solve logic]
}

// Backward compatibility wrapper
Output SecondStepSolver::solve(const Input& input) {
  SolveOptions defaultOptions;
  return solve(input, defaultOptions);
}
```

---

## Configuration

### JSON Configuration Schema

```json
{
  "optimization": {
    "epsilonRefine": {
      "enabled": false,
      "epsilon": 0.05,
      "timeoutSeconds": 60,
      "mipGap": 0.10,
      "exportPath": "./plans/{app}/epsilon_refined/"
    }
  }
}
```

### ConfigurationManager Extensions

```cpp
// utilities/configurationManager.hpp

struct EpsilonRefineConfig {
  bool enabled = false;
  double epsilon = 0.05;
  int timeoutSeconds = 60;
  double mipGap = 0.10;
  std::string exportPath = "";

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(
    EpsilonRefineConfig,
    enabled,
    epsilon,
    timeoutSeconds,
    mipGap,
    exportPath
  )
};

struct Optimization {
  // ... existing fields ...
  EpsilonRefineConfig epsilonRefine;
};
```

---

## Implementation Phases

### Phase 1: Core Refinement Logic (Week 1)

**Tasks**:
1. Add `SolveOptions` to `SecondStepSolver`
2. Implement runtime constraint in `SecondStepSolver::solve()`
3. Add objective switching (runtime vs migrations)
4. Test individual steps separately

**Deliverables**:
- `secondStepSolver.hpp` with `SolveOptions`
- `secondStepSolver.cpp` with constraint logic
- Unit tests for runtime constraint

**Validation**:
```cpp
// Test 1: Runtime constraint is respected
SolveOptions opts;
opts.runtimeUpperBound = 100.0;
auto result = solver.solve(input, opts);
ASSERT(result.totalRuntime <= 100.0);

// Test 2: Minimize migrations without runtime weight
opts.optimizeRuntime = false;
opts.runtimeUpperBound = 105.0;
auto result2 = solver.solve(input, opts);
ASSERT(countMigrations(result2) <= countMigrations(result));
```

### Phase 2: EpsilonRefiner Class (Week 2)

**Tasks**:
1. Create `epsilonRefiner.hpp` and `epsilonRefiner.cpp`
2. Implement `refine()` method (two-step execution)
3. Implement `refineFromSolution()` method
4. Add migration counting and runtime calculation helpers

**Deliverables**:
- `optimization/strategies/epsilonRefiner.hpp`
- `optimization/strategies/epsilonRefiner.cpp`
- Integration tests with sample task graphs

**Validation**:
```cpp
// Test: Full epsilon refinement
EpsilonRefiner::Config cfg;
cfg.epsilon = 0.05;
EpsilonRefiner refiner(cfg);

auto result = refiner.refine(input);
ASSERT(result.successful);
ASSERT(result.solution.totalRuntime <= result.runtimeBound);
ASSERT(result.migrationsAfter <= result.migrationsBefore);
```

### Phase 3: Standalone Refine Function (Week 3)

**Tasks**:
1. Implement `refineExistingSolution()` wrapper
2. Add JSON serialization for `RefinedSolution`
3. Create export directory structure
4. Add logging and diagnostics

**Deliverables**:
- `refineExistingSolution()` in `epsilonRefiner.cpp`
- `exportRefinedSolution()` in `epsilonRefiner.cpp`
- Documentation in EPSILON_REFINE.md

**File Format**:
```json
{
  "refinementMetadata": {
    "epsilon": 0.05,
    "optimalRuntime": 100.0,
    "runtimeBound": 105.0,
    "migrationsBefore": 21,
    "migrationsAfter": 11,
    "migrationReduction": 47.6,
    "step1SolveTime": 5.2,
    "step2SolveTime": 3.8
  },
  "solution": {
    // Standard SecondStepSolver::Output format
  }
}
```

### Phase 4: Pipeline Integration (Week 4)

**Tasks**:
1. Add `epsilonRefine` to configuration schema
2. Integrate into `tiledCholesky.cu` main loop
3. Add command-line flag: `--epsilon-refine=0.05`
4. Export both original and refined solutions

**Deliverables**:
- Updated `config.json` with `epsilonRefine` section
- Modified `tiledCholesky.cu` with optional refinement
- Updated CLAUDE.md with usage examples

**Integration Code**:
```cpp
// tiledCholesky.cu

int main(int argc, char** argv) {
  // ... [normal execution]

  // Step 2: MIP Optimization
  auto step2Output = secondStepSolver.solve(step2Input);
  exportSolution(step2Output, "solution_0.json");

  // Optional: Epsilon Refinement
  auto& epsilonCfg = ConfigurationManager::getConfig().optimization.epsilonRefine;
  if (epsilonCfg.enabled) {
    LOG_TRACE_WITH_INFO("Starting epsilon refinement (ε=%.2f)", epsilonCfg.epsilon);

    EpsilonRefiner refiner(epsilonCfg);
    auto refined = refiner.refineFromSolution(step2Output, step2Input);

    if (refined.successful) {
      exportRefinedSolution(refined, "refined_solution.json");

      LOG_TRACE_WITH_INFO("Refinement complete:");
      LOG_TRACE_WITH_INFO("  Runtime: %.2f → %.2f (bound: %.2f)",
                          refined.optimalRuntime,
                          refined.solution.totalRuntime,
                          refined.runtimeBound);
      LOG_TRACE_WITH_INFO("  Migrations: %d → %d (%.1f%% reduction)",
                          refined.migrationsBefore,
                          refined.migrationsAfter,
                          refined.migrationReduction);
    }
  }

  return 0;
}
```

---

## Usage Workflows

### Workflow 1: Auto-Refine During Execution

**Config**:
```json
{
  "optimization": {
    "epsilonRefine": {
      "enabled": true,
      "epsilon": 0.05
    }
  }
}
```

**Command**:
```bash
./build/userApplications/tiledCholesky
```

**Output**:
```
./plans/tiledCholesky/20241113_143022/
├── solution_0.json           # Original Step 2 solution
└── refined_solution.json     # Epsilon-refined solution
```

### Workflow 2: Offline Refinement of Saved Solution

**Use Case**: You have an old solution and want to try different epsilon values

**Code**:
```cpp
// refineTool.cpp (new utility)
#include "optimization/strategies/epsilonRefiner.hpp"

int main(int argc, char** argv) {
  // Load existing solution
  auto existingSolution = loadSolutionFromJSON(argv[1]);
  auto input = loadInputFromJSON(argv[2]);

  // Refine with different epsilons
  for (double eps : {0.0, 0.05, 0.10, 0.20}) {
    auto refined = refineExistingSolution(existingSolution, input, eps);

    std::string filename = fmt::format("refined_eps_{:.2f}.json", eps);
    exportRefinedSolution(refined, filename);

    printf("ε=%.2f: Runtime=%.2f, Migrations=%d\n",
           eps, refined.solution.totalRuntime, refined.migrationsAfter);
  }
}
```

**Command**:
```bash
./build/tools/refineTool \
  ./plans/tiledCholesky/20241113_143022/solution_0.json \
  ./plans/tiledCholesky/20241113_143022/profile.json
```

**Output**:
```
ε=0.00: Runtime=100.0, Migrations=21
ε=0.05: Runtime=103.2, Migrations=11
ε=0.10: Runtime=107.5, Migrations=8
ε=0.20: Runtime=115.1, Migrations=5
```

### Workflow 3: Batch Epsilon Sweep

**Script**: `scripts/epsilon_sweep.sh`
```bash
#!/bin/bash

EPSILONS=(0.00 0.01 0.02 0.05 0.10 0.15 0.20)

for eps in "${EPSILONS[@]}"; do
  echo "Testing epsilon=$eps"

  # Update config
  jq ".optimization.epsilonRefine.epsilon = $eps" config.json > tmp.json
  mv tmp.json config.json

  # Run
  ./build/userApplications/tiledCholesky

  # Rename output
  PLAN_DIR=$(ls -td ./plans/tiledCholesky/* | head -1)
  mv "$PLAN_DIR" "${PLAN_DIR}_eps_${eps}"
done

# Analyze results
python3 scripts/analyze_epsilon_tradeoff.py
```

### Workflow 4: Command-Line Override

**Implementation**:
```cpp
// Parse command-line argument in tiledCholesky.cu
for (int i = 1; i < argc; i++) {
  if (strncmp(argv[i], "--epsilon-refine=", 17) == 0) {
    double eps = std::stod(argv[i] + 17);
    auto& cfg = ConfigurationManager::getConfig().optimization.epsilonRefine;
    cfg.enabled = true;
    cfg.epsilon = eps;
  }
}
```

**Command**:
```bash
./build/userApplications/tiledCholesky --epsilon-refine=0.10
```

---

## Testing Strategy

### Unit Tests

**Test 1: Runtime Constraint Enforcement**
```cpp
TEST(SecondStepSolver, RuntimeConstraint) {
  SecondStepSolver solver;
  SecondStepSolver::Input input = createTestInput();

  SecondStepSolver::SolveOptions opts;
  opts.runtimeUpperBound = 50.0;

  auto result = solver.solve(input, opts);

  EXPECT_LE(result.totalRuntime, 50.0);
}
```

**Test 2: Migration Minimization**
```cpp
TEST(SecondStepSolver, MinimizeMigrations) {
  SecondStepSolver solver;
  SecondStepSolver::Input input = createTestInput();

  // Step 1: Normal solve
  auto normalResult = solver.solve(input);

  // Step 2: Migration-only optimization with runtime slack
  SecondStepSolver::SolveOptions opts;
  opts.optimizeRuntime = false;
  opts.runtimeUpperBound = normalResult.totalRuntime * 1.10;

  auto refinedResult = solver.solve(input, opts);

  int normalMigrations = countMigrations(normalResult);
  int refinedMigrations = countMigrations(refinedResult);

  EXPECT_LE(refinedMigrations, normalMigrations);
  EXPECT_LE(refinedResult.totalRuntime, opts.runtimeUpperBound);
}
```

**Test 3: Epsilon Refiner End-to-End**
```cpp
TEST(EpsilonRefiner, FullRefinement) {
  EpsilonRefiner::Config cfg;
  cfg.epsilon = 0.05;

  EpsilonRefiner refiner(cfg);
  SecondStepSolver::Input input = createTestInput();

  auto result = refiner.refine(input);

  EXPECT_TRUE(result.successful);
  EXPECT_GT(result.optimalRuntime, 0.0);
  EXPECT_LE(result.solution.totalRuntime, result.runtimeBound);
  EXPECT_LE(result.migrationsAfter, result.migrationsBefore);
}
```

### Integration Tests

**Test Suite**: tiledCholesky with varying problem sizes

```bash
# Test small problem (quick validation)
./test_epsilon_refine.sh --size=small --epsilon=0.05

# Test medium problem (realistic)
./test_epsilon_refine.sh --size=medium --epsilon=0.10

# Test large problem (stress test)
./test_epsilon_refine.sh --size=large --epsilon=0.05
```

**Validation Criteria**:
1. Runtime constraint always respected: `runtime ≤ (1+ε) * optimal`
2. Migration reduction achieved: `migrations_refined ≤ migrations_original`
3. Solutions remain feasible: All task dependencies satisfied
4. Solve time reasonable: Both steps complete within timeout

### Regression Tests

**Test**: Existing behavior unchanged when disabled

```cpp
TEST(EpsilonRefiner, DisabledDoesNothing) {
  // Run with epsilon refine disabled
  ConfigurationManager::getConfig().optimization.epsilonRefine.enabled = false;

  auto result1 = runTiledCholesky();

  // Should produce identical results to before epsilon refine feature
  auto expectedResult = loadBaselineResult();

  EXPECT_EQ(result1.prefetches, expectedResult.prefetches);
  EXPECT_EQ(result1.offloadings, expectedResult.offloadings);
}
```

---

## Expected Results

### tiledCholesky Example (n=102400, t=4)

**Baseline (Step 2 only)**:
- Runtime: 100.0 seconds
- Migrations: 21
- Objective: 0.36 (with weightOfMigrations=0.1)

**After Epsilon Refinement (ε=0.05)**:
- Step 1 (optimize runtime): 100.0 seconds, 21 migrations
- Step 2 (minimize migrations): 103.2 seconds, 11 migrations
- **Result**: 47% fewer migrations for 3.2% runtime cost

**Pareto Frontier** (expected):
```
ε=0.00: Runtime=100.0, Migrations=21 (strict, no improvement)
ε=0.02: Runtime=101.5, Migrations=15 (small slack)
ε=0.05: Runtime=103.2, Migrations=11 (sweet spot)
ε=0.10: Runtime=107.5, Migrations=8  (more slack)
ε=0.20: Runtime=115.1, Migrations=5  (large slack)
```

### Performance Metrics

**Expected Solve Times**:
- Step 1 (runtime optimization): 5-10 seconds
- Step 2 (migration optimization): 3-8 seconds
- Total overhead: 8-18 seconds

**With Warm Start** (future optimization):
- If using Step 2 solution as warm start for refinement: 2-5 seconds total

---

## Future Enhancements

### Enhancement 1: Automatic Epsilon Selection

Instead of user-specified ε, find the "knee" of the Pareto curve:

```cpp
std::vector<RefinedSolution> findParetoFrontier(
    const SecondStepSolver::Input& input,
    const std::vector<double>& epsilons);

double findOptimalEpsilon(const std::vector<RefinedSolution>& frontier);
```

### Enhancement 2: Multi-Level Refinement

Chain multiple refinements:
1. Minimize runtime
2. Constrain runtime, minimize migrations
3. Constrain runtime + migrations, minimize memory

### Enhancement 3: Interactive Selection

GUI or CLI tool for users to explore tradeoffs:
```bash
./interactiveRefine solution_0.json
> Try epsilon=0.05
  Result: Runtime=103.2, Migrations=11
> Try epsilon=0.10
  Result: Runtime=107.5, Migrations=8
> Accept epsilon=0.05
  Exported to refined_solution.json
```

---

## References

- **Epsilon-Constraint Method**: Haimes, Y. Y., et al. (1971). "On a Bicriterion Formulation of the Problems of Integrated System Identification and System Optimization"
- **Multi-Objective MIP**: Ehrgott, M. (2005). "Multicriteria Optimization". Springer.
- **Gurobi Runtime Bounds**: https://docs.gurobi.com/projects/optimizer/en/current/reference/constraints.html
- **OR-Tools MPSolver**: https://developers.google.com/optimization/reference/linear_solver/linear_solver/MPSolver

---

## Status & Next Steps

**Current**: Design complete, implementation not started

**Next**: Begin Phase 1 - Core refinement logic
1. Add `SolveOptions` to `SecondStepSolver`
2. Implement runtime constraint
3. Test constraint enforcement

**Timeline**: 4 weeks for complete implementation (aligns with TOPK.md phases)
