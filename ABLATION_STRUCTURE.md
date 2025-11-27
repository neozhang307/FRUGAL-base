# FRUGAL Ablation Study Structure

This document describes the ablation studies conducted for the FRUGAL project and the associated code files.

## Overview

| Study | Status | Description |
|-------|--------|-------------|
| Model Validation - Saturation | ✅ Completed | Validate saturation assumption in performance model |
| Model Validation - Independence | ❌ Not Finished | Validate compute-communication independence (bug in CGO26/revision-evaluation-bug) |
| Top-K Task Ordering (Exp1) | ✅ Completed | Evaluate Top-100 first-step solutions with MIP predicted vs actual runtime |
| Beam Width Analysis (Exp2) | ✅ Completed | Analyze beam width impact on solution quality |
| Window Size Analysis (Exp3) | ✅ Completed | Compare different window sizes and lookahead distances |
| Visualization Support | ✅ Completed | Visualize execution plans and DAGs |

---

## 1. Model Validation Studies

### 1.1 Saturation Assumption Validation ✅

**Goal**: Validate that the performance model's saturation assumption holds for large workloads.

**Location**: `experiments/performance_validation/`

| File | Purpose |
|------|---------|
| `experiments/performance_validation/run_validation.py` | Run saturation validation experiments |
| `experiments/performance_validation/analyze_results.py` | Analyze validation results |
| `experiments/performance_validation/generate_paper_plots.py` | Generate publication-quality plots |

### 1.2 Independence Assumption Validation ❌

**Goal**: Validate that compute and communication can overlap independently.

**Status**: Not finished due to a bug. See branch `CGO26/revision-evaluation-bug`.

**Location**: `experiments/model_validate_independence/` (untracked, buggy)

---

## 2. Top-K Task Ordering Study (Exp1) ✅

**Goal**: Generate Top-100 task orderings from beam search, run MIP optimization for each, compare predicted runtime vs actual GPU execution runtime.

**Results Location**: `results/ablation/exp1/`

### Scripts (Run Order)

| Step | File | Purpose |
|------|------|---------|
| 1 | `experiments/ablation/generate_topk100.sh` | Generate Top-100 task orderings from beam search |
| 2 | `experiments/ablation/generate_topk100_plans.sh` | Generate MIP-optimized execution plans for each ordering |
| 3 | `experiments/ablation/run_all_topk100.sh` | Execute all 100 plans on GPU and collect actual runtime |

### Analysis Scripts

| File | Purpose |
|------|---------|
| `experiments/ablation/analyze_topk100.py` | Analyze Top-100 results (score distribution, runtime analysis) |
| `experiments/ablation/parse_topk100_execution.py` | Parse GPU execution logs, compare predicted vs actual runtime |
| `experiments/ablation/plot_topk100.py` | Generate plots (score distribution, runtime histogram, etc.) |

### Key Metrics
- Task scheduling score (data reuse in GB)
- MIP predicted runtime
- Actual GPU execution runtime
- Prediction accuracy

---

## 3. Beam Width Analysis (Exp2) ✅

**Goal**: Analyze how beam width affects solution quality and solve time.

**Results Location**: `results/ablation/exp2/`

### Scripts

| File | Purpose |
|------|---------|
| `experiments/ablation/beam_width_ablation.sh` | Run beam width experiments (width 1-100) |
| `experiments/ablation/run_beam_solutions.sh` | Execute beam solutions on GPU |

### Analysis Scripts

| File | Purpose |
|------|---------|
| `experiments/ablation/analyze_beam_width.py` | Analyze beam width vs score/time tradeoff |
| `experiments/ablation/parse_beam_execution.py` | Parse beam execution results |

### Key Metrics
- Beam width vs task scheduling score
- Beam width vs first-step solve time
- Score convergence point

---

## 4. Window Size Analysis (Exp3) ✅

**Goal**: Understand how prefetch/offload window parameters affect MIP solver performance and solution quality.

**Results Location**: `results/ablation/exp3/`

### Scripts

| File | Purpose |
|------|---------|
| `experiments/ablation/run_test1_abstract_window.sh` | Test distance-based window limits (1, 5, 10, 20, 30) |
| `experiments/ablation/run_test2_time_factor.sh` | Test time-based factors (1.0 - 50.0) |

### Parameters Tested
- `prefetchLookbackDistanceLimit` - How many tasks to look back for prefetch
- `offloadLookaheadDistanceLimit` - How many tasks to look ahead for offload
- `prefetchLookbackTimeBudgetFactor` - Time budget factor for prefetch
- `offloadLookaheadComputeTimeFactor` - Compute time factor for offload

### Key Metrics
- MIP solve time
- Solution quality (predicted runtime)
- Memory constraint satisfaction

---

## 5. Visualization Support ✅

**Goal**: Visualize execution plans, DAGs, and data migration schedules.

### Scripts

| File | Purpose |
|------|---------|
| `scripts/visualize_plan_dag.py` | Visualize execution plan DAG with tasks, prefetch/offload, control nodes |
| `scripts/visualize_dag.py` | Visualize task dependency graph and data flow graph with execution order |

### Features

**visualize_plan_dag.py**:
- A4 landscape layout with 3-row format
- Task nodes (blue), prefetch (green), offload (red), control (gray)
- Array sizes displayed in GB

**visualize_dag.py**:
- Task dependency DAG with execution order overlay
- Data flow graph showing array-task relationships
- Color gradient from red (early) to blue (late) based on execution order
- Execution order computed from topological sort of plan DAG

### Generating Input Files (profile.json and plan.json)

The visualization scripts require two JSON files:
1. **profile.json** - Profiling data (task groups, arrays, dependencies)
2. **plan.json** - Optimized execution plan (DAG with nodes and edges)

#### Method 1: Using tiledCholeskyAblation (GPU required)

```bash
# Step 1: Generate profile only (saves profiling data, no optimization)
./build/userApplications/tiledCholeskyAblation \
  --profile-only \
  --save-profile=profile.json \
  -N 102400 -T 4

# Step 2: Generate optimized plan using standaloneOptimizer (CPU only)
./build/tools/standaloneOptimizer \
  profile.json \
  plan.json \
  --config=config.json
```

#### Method 2: Using standaloneOptimizer only (CPU only, profile must exist)

```bash
# If you already have a profile.json from previous profiling:
./build/tools/standaloneOptimizer \
  profile.json \
  plan.json \
  --config=config.json \
  --memory-bound=15000
```

#### Method 3: Full end-to-end run (generates both internally)

```bash
# Normal mode: profile + optimize + execute (no files saved by default)
./build/userApplications/tiledCholeskyAblation -N 102400 -T 4

# To save intermediate files, use --run-plan mode after profiling:
./build/userApplications/tiledCholeskyAblation \
  --run-plan \
  --load-plan=plan.json \
  -N 102400 -T 4
```

### Command Line Options (tiledCholeskyAblation)

| Option | Description |
|--------|-------------|
| `--profile-only` | Only run profiling, save to JSON, then exit |
| `--save-profile=<path>` | Path to save profiling data (default: `optimization_input.json`) |
| `--run-plan` | Skip profiling/optimization, load and execute existing plan |
| `--load-plan=<path>` | Path to load execution plan (default: `ablation_optimized_plan.json`) |

### Usage Examples

```bash
# Visualize execution plan DAG
python scripts/visualize_plan_dag.py \
  --profile results/ablation/exp1/profile_N102400_T4.json \
  --plan results/ablation/exp3/test1_abstract_window/plans/plan_dist10.json \
  --output results/visualization/

# Visualize task dependency graph with execution order
python scripts/visualize_dag.py \
  --profile results/ablation/exp1/profile_N102400_T4.json \
  --plan results/ablation/exp3/test1_abstract_window/plans/plan_dist10.json \
  --output results/visualization/ \
  --graphs all
```

### Output Files

| Script | Output Files |
|--------|--------------|
| `visualize_plan_dag.py` | `plan_dag_graph.pdf`, `plan_dag_graph.png` |
| `visualize_dag.py` | `task_dependency_graph.pdf/png`, `data_flow_graph.pdf/png` |

---

## File Summary

### Tracked Files for Ablation Studies

```
experiments/ablation/
├── generate_topk100.sh          # Exp1: Generate Top-100 orderings
├── generate_topk100_plans.sh    # Exp1: Generate plans
├── run_all_topk100.sh           # Exp1: Run on GPU
├── analyze_topk100.py           # Exp1: Analysis
├── parse_topk100_execution.py   # Exp1: Parse results
├── plot_topk100.py              # Exp1: Plotting
├── beam_width_ablation.sh       # Exp2: Beam width runner
├── run_beam_solutions.sh        # Exp2: Run solutions
├── analyze_beam_width.py        # Exp2: Analysis
├── parse_beam_execution.py      # Exp2: Parse results
├── run_test1_abstract_window.sh # Exp3: Window distance test
└── run_test2_time_factor.sh     # Exp3: Time factor test

experiments/performance_validation/
├── run_validation.py            # Saturation validation
├── analyze_results.py           # Analysis
└── generate_paper_plots.py      # Paper plots

scripts/
├── visualize_plan_dag.py        # Execution plan DAG visualization
└── visualize_dag.py             # Task dependency and data flow visualization
```

---

## Running the Complete Ablation Study

### Prerequisites
1. Build the project: `make build`
2. Ensure Gurobi license is available
3. Activate conda environment: `conda activate frugal`

### Execution Order

```bash
# 1. Model Validation (Saturation)
cd experiments/performance_validation
python run_validation.py
python analyze_results.py
python generate_paper_plots.py

# 2. Top-K Study (Exp1)
cd experiments/ablation
./generate_topk100.sh
./generate_topk100_plans.sh
./run_all_topk100.sh
python analyze_topk100.py
python parse_topk100_execution.py
python plot_topk100.py

# 3. Beam Width Study (Exp2)
./beam_width_ablation.sh
./run_beam_solutions.sh
python analyze_beam_width.py
python parse_beam_execution.py

# 4. Window Size Study (Exp3)
./run_test1_abstract_window.sh
./run_test2_time_factor.sh

# 5. Visualization
python scripts/visualize_plan_dag.py --profile <profile.json> --plan <plan.json> --output <output_dir>
```
