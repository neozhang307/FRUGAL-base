# Ablation Study Guide

This guide explains how to use the ablation study tools in the FRUGAL framework for analyzing the two-step optimization process.

## Table of Contents
1. [Overview](#overview)
2. [Tools Available](#tools-available)
3. [Using standaloneOptimizer](#using-standaloneoptimizer)
4. [Using tiledCholeskyAblation](#using-tiledcholeskyablation)
5. [Common Workflows](#common-workflows)
6. [Analyzing Results](#analyzing-results)

## Overview

The FRUGAL optimization framework uses a two-step approach:
1. **Step 1**: Task scheduling optimization (finds optimal task execution order)
2. **Step 2**: Memory management optimization (determines when to prefetch/offload arrays)

For ablation studies, we can:
- Run steps independently
- Save/load intermediate results
- Extract multiple solutions (Top-K)
- Test different configurations

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

### Independent Step Execution (**NEW**)

```bash
# Run ONLY first step (no output file needed)
./build/tools/standaloneOptimizer profile.json --first-step-only --top-k=10 --save-topk=topk.json

# Run ONLY second step (requires loaded first step)
./build/tools/standaloneOptimizer profile.json output.json --second-step-only \
  --load-topk=topk.json --solution-index=0
```

### Top-K Solutions (**FULLY IMPLEMENTED**)

#### First Step Top-K (Task Ordering)
```bash
# Extract and save top-10 solutions from beam search
./build/tools/standaloneOptimizer profile.json --first-step-only --top-k=10 --save-topk=topk.json

# Load Top-K and use specific solution
./build/tools/standaloneOptimizer profile.json output.json --load-topk=topk.json --solution-index=5

# Generate Top-K and immediately use one
./build/tools/standaloneOptimizer profile.json output.json --top-k=10 --solution-index=3
```

#### Second Step Top-K (Memory Management) (**NEW**)
```bash
# Generate multiple memory/runtime tradeoffs using multi-weight strategy
./build/tools/standaloneOptimizer profile.json output.json --second-step-topk=5 --use-multi-weight

# Try Gurobi solution pool (may return only 1-2 solutions)
./build/tools/standaloneOptimizer profile.json output.json --second-step-topk=10

# Combine with first step Top-K for comprehensive ablation
./build/tools/standaloneOptimizer profile.json output.json \
  --load-topk=first_step_topk.json --solution-index=0 \
  --second-step-topk=5 --use-multi-weight
```

### Dynamic Parameter Override (**NEW**)

```bash
# Override optimization weights
./build/tools/standaloneOptimizer profile.json output.json \
  --weight-memory=0.8 --weight-runtime=0.2 --weight-migrations=0.0

# Set memory constraint
./build/tools/standaloneOptimizer profile.json output.json --max-memory=1000.0

# Set look-ahead/look-back window size
./build/tools/standaloneOptimizer profile.json output.json --look-window-size=20
```

### Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| **Execution Control** |
| `--config` | Configuration file path | `--config=config.json` |
| `--first-step-only` | Run only first step | `--first-step-only` |
| `--second-step-only` | Run only second step | `--second-step-only` |
| **First Step Options** |
| `--save-first-step` | Save first step output | `--save-first-step=step1.json` |
| `--load-first-step` | Load first step from file | `--load-first-step=step1.json` |
| `--top-k` | Extract K solutions from beam search | `--top-k=10` |
| `--solution-index` | Which solution to use (0-based) | `--solution-index=2` |
| `--save-topk` | Save all Top-K solutions | `--save-topk=topk.json` |
| `--load-topk` | Load Top-K solutions | `--load-topk=topk.json` |
| **Second Step Options** |
| `--use-pool` | Enable Gurobi solution pool to find multiple solutions | `--use-pool` |
| `--solver` | Override second step solver type (MIP/GREEDY) | `--solver=MIP` |
| **Optimization Parameters** |
| `--weight-memory` | Weight for memory optimization (0-1) | `--weight-memory=0.8` |
| `--weight-runtime` | Weight for runtime optimization (0-1) | `--weight-runtime=0.2` |
| `--weight-migrations` | Weight for migration optimization (0-1) | `--weight-migrations=0.0` |
| `--max-memory` | Peak memory constraint (MiB) | `--max-memory=1000.0` |
| `--look-window-size` | Look-ahead/back window size | `--look-window-size=20` |

## Using tiledCholeskyAblation

### Three Execution Modes

#### 1. Profile-Only Mode
Collect profiling data without optimization:
```bash
./build/userApplications/tiledCholeskyAblation \
    --N=2048 --T=8 \
    --profile-only \
    --save-profile=profile.json
```

#### 2. Run-Plan Mode
Execute a pre-optimized plan without re-profiling:
```bash
./build/userApplications/tiledCholeskyAblation \
    --N=2048 --T=8 \
    --run-plan \
    --load-plan=optimized.json
```

#### 3. Normal Mode
Full pipeline (profile + optimize + execute):
```bash
./build/userApplications/tiledCholeskyAblation \
    --N=2048 --T=8
```

### Saving Intermediate Results

```bash
# Save profiling data
./build/userApplications/tiledCholeskyAblation \
    --N=2048 --T=8 \
    --save-profile=profile.json

# Save optimized plan
./build/userApplications/tiledCholeskyAblation \
    --N=2048 --T=8 \
    --save-plan=plan.json

# Save first step output
./build/userApplications/tiledCholeskyAblation \
    --N=2048 --T=8 \
    --save-first-step=step1.json
```

### Loading Intermediate Results

```bash
# Load and continue from first step
./build/userApplications/tiledCholeskyAblation \
    --N=2048 --T=8 \
    --load-first-step=step1.json

# Load pre-optimized plan
./build/userApplications/tiledCholeskyAblation \
    --N=2048 --T=8 \
    --run-plan \
    --load-plan=plan.json
```

## Gurobi Solution Pool Support (**NEW**)

### Overview
The framework now supports Gurobi's solution pool to find multiple near-optimal solutions in a single optimization run. This is useful for exploring different memory/runtime tradeoffs without manually adjusting weights.

### How to Use Solution Pool

```bash
# 1. Profile the application
./build/userApplications/tiledCholeskyAblation --N=2048 --T=8 --profile-only --save-profile=profile.json

# 2. Generate multiple solutions using solution pool
# IMPORTANT: Must use MIP solver (not GREEDY)
./build/tools/standaloneOptimizer profile.json plan.json --use-pool --solver=MIP

# This will create:
#   - plan.json (best solution)
#   - plan_sol0.json (same as best)
#   - plan_sol1.json (second solution, if found)
#   - plan_sol2.json (third solution, if found)
#   - etc.

# 3. Run each solution
for plan in plan_sol*.json; do
    if [ -f "$plan" ]; then
        echo "Testing $plan:"
        ./build/userApplications/tiledCholeskyAblation --N=2048 --T=8 --run-plan --load-plan=$plan
    fi
done
```

### Important Notes
- **Solver Type**: Solution pool only works with MIP solver, not GREEDY
- **Typical Results**: Gurobi usually finds 1-2 solutions (rarely more)
- **Pool Parameters**: Automatically configured with relaxed gaps to find diverse solutions
- **File Naming**: Solutions saved as `plan_sol0.json`, `plan_sol1.json`, etc.

## Common Workflows

### Workflow 1: Analyze Step 1 Impact (**UPDATED**)

```bash
# 1. Profile once
./build/userApplications/tiledCholeskyAblation \
    --N=2048 --T=8 \
    --profile-only \
    --save-profile=profile.json

# 2. Generate Top-K task orderings ONCE
./build/tools/standaloneOptimizer profile.json --first-step-only \
    --top-k=10 --save-topk=topk_orderings.json

# 3. Generate plans from different orderings
for i in {0..9}; do
  ./build/tools/standaloneOptimizer profile.json output_${i}.json \
    --load-topk=topk_orderings.json --solution-index=${i}
done

# 4. Execute and compare
for i in {0..9}; do
  echo "Testing solution ${i}:"
  ./build/userApplications/tiledCholeskyAblation --N=2048 --T=8 \
    --run-plan --load-plan=output_${i}.json
done
```

### Workflow 2: Analyze Step 2 Impact (**ENHANCED WITH TOP-K**)

```bash
# 1. Generate Top-K first step solutions
./build/tools/standaloneOptimizer profile.json --first-step-only \
    --top-k=3 --save-topk=first_step_topk.json

# 2. For each first step solution, generate multiple second step solutions
for idx in {0..2}; do
  # Use multi-weight strategy to get 5 different tradeoffs per task ordering
  ./build/tools/standaloneOptimizer profile.json output_ordering${idx}.json \
    --load-topk=first_step_topk.json --solution-index=${idx} \
    --second-step-topk=5 --use-multi-weight

  echo "Generated 5 memory/runtime tradeoffs for task ordering ${idx}"
done

# 3. Alternative: Manual weight configurations (original method)
for idx in {0..2}; do
  # Memory optimized
  ./build/tools/standaloneOptimizer profile.json output_${idx}_mem.json \
    --load-topk=first_step_topk.json --solution-index=${idx} \
    --weight-memory=1.0 --weight-runtime=0.0 --weight-migrations=0.0

  # Runtime optimized
  ./build/tools/standaloneOptimizer profile.json output_${idx}_runtime.json \
    --load-topk=first_step_topk.json --solution-index=${idx} \
    --weight-memory=0.0 --weight-runtime=1.0 --weight-migrations=0.0

  # Balanced
  ./build/tools/standaloneOptimizer profile.json output_${idx}_balanced.json \
    --load-topk=first_step_topk.json --solution-index=${idx} \
    --weight-memory=0.33 --weight-runtime=0.33 --weight-migrations=0.34
done

# 4. Execute and compare all plans
for plan in output_*.json; do
  echo "Testing ${plan}:"
  ./build/userApplications/tiledCholeskyAblation --N=2048 --T=8 --run-plan --load-plan=${plan}
done
```

### Workflow 3: Skip Optimization Steps

```bash
# Skip first step (use default task order)
cat > config_skip_first.json << EOF
{
  "optimization": {
    "byPassingFirstStep": true
  }
}
EOF
./build/tools/standaloneOptimizer profile.json output_skip1.json --config=config_skip_first.json

# Skip second step (use greedy scheduler)
cat > config_greedy.json << EOF
{
  "optimization": {
    "secondStepSolverType": "GREEDY",
    "greedySchedulerMode": "MIN_MEMORY"
  }
}
EOF
./build/tools/standaloneOptimizer profile.json output_greedy.json --config=config_greedy.json
```

### Workflow 4: Beam Width Analysis

```bash
# Test different beam widths
for width in 1 10 50 100 200; do
  cat > config_beam_${width}.json << EOF
{
  "optimization": {
    "firstStepSolverType": "BEAM_SEARCH",
    "beamWidth": ${width}
  }
}
EOF

  ./build/tools/standaloneOptimizer profile.json output_bw${width}.json \
      --config=config_beam_${width}.json

  echo "Beam width ${width} completed"
done
```

### Workflow 5: Top-K Diversity Analysis

```bash
# Extract top-10 solutions and analyze diversity
./build/tools/standaloneOptimizer profile.json dummy.json --top-k=10

# Process each solution independently
for i in {0..9}; do
  ./build/tools/standaloneOptimizer profile.json output_sol${i}.json \
      --top-k=10 --solution-index=${i}

  echo "Solution ${i} processed"
done

# Execute and compare performance
for i in {0..9}; do
  echo "Testing solution ${i}:"
  ./build/userApplications/tiledCholeskyAblation \
      --N=2048 --T=8 --run-plan --load-plan=output_sol${i}.json
done
```

## Analyzing Results

### Key Metrics to Track

1. **Memory Usage**
   - Peak memory usage (MB)
   - Memory reduction percentage
   - Number of arrays on device

2. **Performance**
   - Total runtime (seconds)
   - Data movement overhead
   - Number of migrations

3. **Solution Quality**
   - Data reuse score (from Step 1)
   - Optimization time
   - Solution optimality

### Output Files

- **Profile JSON**: Contains task graph, timing, and array metadata
- **First Step JSON**: Task execution order and data reuse score
- **Optimized Plan JSON**: Complete execution plan with memory operations
- **Top-K JSON**: Multiple solutions with scores (when implemented)

### Example Analysis Script

```bash
#!/bin/bash
# analyze_ablation.sh

echo "Solution,DataReuseScore,PeakMemory,Runtime,Migrations"

for i in {0..9}; do
  # Extract metrics from output files
  score=$(grep "dataReuseScore" output_sol${i}.json | cut -d: -f2 | tr -d ' ,')
  memory=$(grep "anticipatedPeakMemoryUsage" output_sol${i}.json | cut -d: -f2 | tr -d ' ,')

  # Run and capture runtime
  runtime=$(./build/userApplications/tiledCholeskyAblation \
      --N=2048 --T=8 --run-plan --load-plan=output_sol${i}.json 2>&1 | \
      grep "Total time" | awk '{print $3}')

  echo "${i},${score},${memory},${runtime},-"
done
```

## Configuration Parameters

### Important Optimization Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `firstStepSolverType` | Algorithm for Step 1 | `BEAM_SEARCH` |
| `beamWidth` | Beam search width | 100 |
| `secondStepSolverType` | Algorithm for Step 2 | `MIP` |
| `greedySchedulerMode` | Greedy mode | `MIN_MEMORY` |
| `weightOfPeakMemoryUsage` | Memory weight | 0.0 |
| `weightOfTotalRunningTime` | Runtime weight | 1.0 |
| `weightOfNumberOfMigrations` | Migration weight | 0.0 |
| `gurobiTimeLimitSeconds` | MIP solver timeout | 60 |
| `maxPeakMemoryUsageInMiB` | Memory constraint | 0 (unlimited) |

## Troubleshooting

### Common Issues

1. **"BEAN_SEARCH" error**: Fix typo in config - should be `BEAM_SEARCH`
2. **Memory bound infeasible**: Increase `maxPeakMemoryUsageInMiB` or set to 0
3. **Gurobi not found**: Ensure Gurobi is installed and licensed
4. **CUDA library errors**: Set `LD_LIBRARY_PATH` correctly

### Environment Setup

```bash
# Activate conda environment
source ~/miniconda3/bin/activate && conda activate frugal

# Set CUDA library path if needed
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
```

## Next Steps

Future improvements for ablation studies:
1. Implement `saveTopKSolutions` and `loadTopKSolutions` serialization
2. Add multi-weight refinement for Step 2
3. Create automated analysis scripts
4. Add solution diversity metrics (Hamming distance)
5. Implement parallel execution for multiple configurations