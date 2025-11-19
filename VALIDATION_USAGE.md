# Performance Validation Usage Guide

## Overview

The performance validation script (`experiments/performance_validation/run_validation.py`) validates the FRUGAL optimizer across different problem sizes, comparing baseline vs. optimized performance.

## Recent Fixes Applied

1. **Command-line Arguments**: Fixed tiledCholeskyAblation calls to use `--N=` and `--T=` format
2. **Dual Memory Tracking**: Added managed memory AND peak memory tracking
3. **Configurable Runs**: Added `--num-runs` parameter for flexible testing

## Basic Usage

### Quick Correctness Check (1 run)

Use this for fast verification that everything works:

```bash
python experiments/performance_validation/run_validation.py \
    --domain-sizes 1024,2048 \
    --output-dir results/quick_test \
    --base-dir . \
    --num-runs 1
```

**Time estimate**: ~1-2 minutes per domain size

### Full Validation (10 runs, default)

Use this for publication-quality results with statistical averaging:

```bash
python experiments/performance_validation/run_validation.py \
    --domain-sizes 1024,2048,4096,8192,16384,32768 \
    --output-dir results/full_validation \
    --base-dir . \
    --num-runs 10
```

**Time estimate**: ~5-10 minutes per domain size

### Background Execution

For long-running validation across many domain sizes:

```bash
nohup python experiments/performance_validation/run_validation.py \
    --domain-sizes 1024,2048,4096,8192,16384,32768,51200,65536,81920,102400 \
    --output-dir results/saturation_study \
    --base-dir . \
    --num-runs 10 > validation.log 2>&1 &

# Monitor progress
tail -f validation.log
```

## Parameters

### Required Parameters

- **--base-dir**: Base directory of FRUGAL project (default: `.`)
  - Must contain `build/userApplications/` with executables
  - Must contain `configs/` directory

### Optional Parameters

- **--domain-sizes**: Comma-separated list of N values (default: all sizes)
  - Example: `1024,2048,4096`
  - For testing: use small sizes like `1024,2048`
  - For full study: use `1024,2048,4096,8192,16384,32768,51200,65536,81920,102400`

- **--output-dir**: Output directory for results (default: `results/saturation_study`)
  - Creates directory if it doesn't exist
  - Saves CSV and JSON results here

- **--num-runs**: Number of measurement runs for averaging (default: 10)
  - **1 run**: Quick correctness check (~1-2 min/size)
  - **3 runs**: Moderate confidence (~3-4 min/size)
  - **10 runs**: High confidence, low variance (default, ~5-10 min/size)
  - **20+ runs**: Publication-quality statistics

## Workflow Recommendations

### Step 1: Correctness Check (1 run, 2 sizes)

```bash
python experiments/performance_validation/run_validation.py \
    --domain-sizes 1024,16384 \
    --output-dir results/correctness_check \
    --num-runs 1
```

**Check for**:
- N=1024 shows different runtime than N=16384
- Memory reductions are reasonable (60-75% managed)
- No errors in output

### Step 2: Medium Test (3 runs, 4 sizes)

```bash
python experiments/performance_validation/run_validation.py \
    --domain-sizes 1024,4096,16384,65536 \
    --output-dir results/medium_test \
    --num-runs 3
```

**Check for**:
- Runtime scales with problem size
- Memory metrics are consistent
- Standard deviation is reasonable

### Step 3: Full Validation (10 runs, all sizes)

```bash
nohup python experiments/performance_validation/run_validation.py \
    --domain-sizes 1024,2048,4096,8192,16384,32768,51200,65536,81920,102400 \
    --output-dir results/final_validation \
    --num-runs 10 > final_validation.log 2>&1 &
```

## Output Files

After running, the output directory will contain:

```
results/your_output_dir/
├── performance_validation_results.csv    # Summary table
├── performance_validation_results.json   # Detailed JSON
├── config_baseline_*.json                # Baseline configs
├── config_phase1_*.json                  # Phase 1 (memory min) configs
├── config_phase2_*.json                  # Phase 2 (perf opt) configs
├── baseline_*_run*.log                   # Baseline execution logs
├── phase1_*.log                          # Phase 1 logs
└── phase2_*_run*.log                     # Phase 2 logs
```

## Understanding Results

### CSV Columns

- `n`: Domain size
- `t`: Tile size (always 4)
- `baseline_runtime`: Baseline execution time (seconds)
- `baseline_managed_memory`: Baseline managed memory (MB) - matrix tiles only
- `baseline_peak_memory`: Baseline total GPU memory (MB) - includes libraries
- `optimized_managed`: Optimized managed memory (MB)
- `actual_peak`: Optimized total GPU memory (MB)
- `managed_reduction`: Managed memory reduction (%)
- `peak_reduction`: Peak memory reduction (%)
- `predicted_runtime`: Optimizer's predicted runtime (seconds)
- `actual_runtime`: Measured optimized runtime (seconds)
- `predicted_slowdown`: Predicted slowdown factor (x)
- `actual_slowdown`: Actual slowdown factor (x)
- `prediction_error`: Prediction error (%)
- `baseline_tflops`: Baseline throughput (TFLOPS)
- `optimized_tflops`: Optimized throughput (TFLOPS)

### Expected Metrics

**Good results**:
- Managed memory reduction: 60-75%
- Peak memory reduction: ~0% (expected - dominated by library overhead)
- Actual slowdown: 1.2-2.0x
- Prediction error: <20%

**Red flags**:
- Managed memory reduction < 50%
- Actual slowdown > 3.0x
- Prediction error > 50%
- All runtimes identical (indicates bug - see CMDLINE_ARGS_FIX.md)

## Troubleshooting

### All runtimes are identical
**Problem**: Command-line arguments not being passed correctly
**Fix**: Verify CMDLINE_ARGS_FIX.md was applied (use `--N=` not `-N`)

### Disk quota exceeded
**Problem**: Home directory full
**Fix**: Clean up old results: `rm -rf results/old_*`

### Different N values produce same output
**Problem**: Executables not reading N parameter
**Fix**: Check that tiledCholeskyAblation uses `--N=` format

### Memory metrics don't match
**Problem**: Extracting different memory types
**Fix**: Both should use "Peak GPU memory usage during execution"

## Examples

**Quick test before long run**:
```bash
python experiments/performance_validation/run_validation.py \
    --domain-sizes 1024 \
    --num-runs 1 \
    --output-dir results/test_1run
```

**Full saturation study**:
```bash
nohup python experiments/performance_validation/run_validation.py \
    --domain-sizes 1024,2048,4096,8192,16384,32768,51200,65536,81920,102400 \
    --num-runs 10 \
    --output-dir results/saturation_final > saturation.log 2>&1 &
```

**Focused study on large sizes**:
```bash
python experiments/performance_validation/run_validation.py \
    --domain-sizes 32768,51200,65536,81920,102400 \
    --num-runs 5 \
    --output-dir results/large_scale_test
```
