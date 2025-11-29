# FRUGAL Ablation Study Guide

This document describes the ablation study infrastructure and how to use it.

## Overview

The FRUGAL optimization framework uses a two-step approach:
1. **Step 1**: Task scheduling optimization (finds optimal task execution order)
2. **Step 2**: Memory management optimization (determines when to prefetch/offload arrays)

### Study Status

| Study | Status | Description |
|-------|--------|-------------|
| Model Validation - Saturation | ✅ Completed | Validate saturation assumption in performance model |
| Top-K Task Ordering (Exp1) | ✅ Completed | Evaluate Top-100 first-step solutions |
| Beam Width Analysis (Exp2) | ✅ Completed | Analyze beam width impact on solution quality |
| Window Size Analysis (Exp3) | ✅ Completed | Compare different window sizes |
| Visualization Support | ✅ Completed | Visualize execution plans and DAGs |

---

## Tools Available

### 1. standaloneOptimizer
- CPU-only tool for offline optimization
- No GPU/CUDA required
- Can run first step, second step, or both
- Supports Top-K solution extraction

### 2. tiledCholeskyAblation
- GPU application with three execution modes
- Can profile, optimize, or execute pre-optimized plans
- Supports saving/loading at different stages

---

## Using standaloneOptimizer

### Basic Usage

```bash
# Activate environment
source ~/miniconda3/bin/activate && conda activate frugal

# Basic optimization (both steps)
./build/tools/standaloneOptimizer <input_profile.json> <output_plan.json>

# With custom config
./build/tools/standaloneOptimizer <input_profile.json> <output_plan.json> --config=myconfig.json
```

### Independent Step Execution

```bash
# Run ONLY first step
./build/tools/standaloneOptimizer profile.json --first-step-only --top-k=10 --save-topk=topk.json

# Run ONLY second step (requires loaded first step)
./build/tools/standaloneOptimizer profile.json output.json --second-step-only \
  --load-topk=topk.json --solution-index=0
```

### Top-K Solutions

```bash
# Extract and save top-10 solutions from beam search
./build/tools/standaloneOptimizer profile.json --first-step-only --top-k=10 --save-topk=topk.json

# Load Top-K and use specific solution
./build/tools/standaloneOptimizer profile.json output.json --load-topk=topk.json --solution-index=5
```

### Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--config` | Configuration file path | `--config=config.json` |
| `--first-step-only` | Run only first step | `--first-step-only` |
| `--second-step-only` | Run only second step | `--second-step-only` |
| `--top-k` | Extract K solutions from beam search | `--top-k=10` |
| `--solution-index` | Which solution to use (0-based) | `--solution-index=2` |
| `--save-topk` | Save all Top-K solutions | `--save-topk=topk.json` |
| `--load-topk` | Load Top-K solutions | `--load-topk=topk.json` |
| `--use-pool` | Enable Gurobi solution pool | `--use-pool` |
| `--solver` | Override solver type (MIP/GREEDY) | `--solver=MIP` |
| `--weight-memory` | Weight for memory optimization | `--weight-memory=0.8` |
| `--weight-runtime` | Weight for runtime optimization | `--weight-runtime=0.2` |
| `--max-memory` | Peak memory constraint (MiB) | `--max-memory=1000.0` |

---

## Using tiledCholeskyAblation

### Three Execution Modes

```bash
# 1. Profile-Only Mode
./build/userApplications/tiledCholeskyAblation \
    --N=2048 --T=8 --profile-only --save-profile=profile.json

# 2. Run-Plan Mode (execute pre-optimized plan)
./build/userApplications/tiledCholeskyAblation \
    --N=2048 --T=8 --run-plan --load-plan=optimized.json

# 3. Normal Mode (full pipeline)
./build/userApplications/tiledCholeskyAblation --N=2048 --T=8
```

---

## Experiment Structure

### Exp1: Top-K Task Ordering Study

**Location**: `experiments/ablation/topk/`

**Scripts (Run Order)**:
1. `generate_profile.sh` - Generate profile file (requires GPU)
2. `generate_topk100.sh` - Generate Top-100 task orderings
3. `generate_topk100_plans.sh` - Generate MIP-optimized plans
4. `run_all_topk100.sh` - Execute all plans on GPU

**Analysis**:
- `plot_topk100_predict_runtime.py` - Plots based on MIP prediction
- `plot_topk100_real_runtime.py` - Plots based on real GPU runtime

### Exp2: Beam Width Analysis

**Location**: `experiments/ablation/beam/`

**Scripts**:
1. `generate_beam_plans.sh` - Generate plans for different beam widths
2. `run_beam_solutions.sh` - Execute on GPU

**Analysis**:
- `plot_beam_predict_runtime.py`
- `plot_beam_real_runtime.py`

### Exp3: Window Size Analysis

**Location**: `experiments/ablation/window/`

**Scripts**:
1. `run_test1_abstract_window.sh` - Distance-based window limits
2. `run_test2_time_factor.sh` - Time-based factors
3. `run_test1_solutions.sh` / `run_test2_solutions.sh` - Execute on GPU

**Analysis**:
- `plot_window_real_runtime.py --test test1`
- `plot_window_real_runtime.py --test test2`

### Performance Validation (Saturation)

**Location**: `experiments/performance_validation/`

**Scripts**:
- `run_validation.py` - Run saturation validation
- `analyze_results.py` - Analyze results
- `generate_paper_plots.py` - Generate publication plots

---

## Visualization

**Tool**: `scripts/visualize.py`

```bash
# Generate both visualizations
python scripts/visualize.py \
  --profile profile.json \
  --plan plan.json \
  --output output_dir/

# Task dependency graph only
python scripts/visualize.py --profile profile.json --only-dependency

# Execution plan DAG only
python scripts/visualize.py --profile profile.json --plan plan.json --only-plan
```

**Outputs**:
- `task_dependency_graph.pdf` - Task DAG with execution order
- `plan_dag_graph.pdf` - Full execution plan with memory operations

---

## Running Complete Ablation Study

```bash
# Prerequisites
source ~/miniconda3/bin/activate && conda activate frugal
make build

# 1. Model Validation
python experiments/performance_validation/run_validation.py
python experiments/performance_validation/generate_paper_plots.py

# 2. Top-K Study (Exp1)
./experiments/ablation/topk/generate_profile.sh 102400 4
./experiments/ablation/topk/generate_topk100.sh
./experiments/ablation/topk/generate_topk100_plans.sh
./experiments/ablation/topk/run_all_topk100.sh
python experiments/ablation/topk/plot_topk100_real_runtime.py

# 3. Beam Width Study (Exp2)
./experiments/ablation/beam/generate_beam_plans.sh
./experiments/ablation/beam/run_beam_solutions.sh
python experiments/ablation/beam/plot_beam_real_runtime.py

# 4. Window Size Study (Exp3)
./experiments/ablation/window/run_test1_abstract_window.sh
./experiments/ablation/window/run_test2_time_factor.sh
./experiments/ablation/window/run_test1_solutions.sh
./experiments/ablation/window/run_test2_solutions.sh
python experiments/ablation/window/plot_window_real_runtime.py --test test1
python experiments/ablation/window/plot_window_real_runtime.py --test test2
```

---

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `firstStepSolverType` | Algorithm for Step 1 | `BEAM_SEARCH` |
| `beamWidth` | Beam search width | 100 |
| `secondStepSolverType` | Algorithm for Step 2 | `MIP` |
| `greedySchedulerMode` | Greedy mode | `MIN_MEMORY` |
| `weightOfPeakMemoryUsage` | Memory weight | 0.0 |
| `weightOfTotalRunningTime` | Runtime weight | 1.0 |
| `gurobiTimeLimitSeconds` | MIP solver timeout | 60 |
| `maxPeakMemoryUsageInMiB` | Memory constraint | 0 (unlimited) |

---

## Troubleshooting

1. **"BEAN_SEARCH" error**: Fix typo in config - should be `BEAM_SEARCH`
2. **Memory bound infeasible**: Increase `maxPeakMemoryUsageInMiB` or set to 0
3. **Gurobi not found**: Ensure Gurobi is installed and licensed
4. **CUDA library errors**: Set `LD_LIBRARY_PATH` correctly

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
```
