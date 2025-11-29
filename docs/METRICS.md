# Performance Validation Metrics

## Overview

This document describes all metrics tracked in the performance validation study, explaining what each metric means and why it matters.

---

## Memory Metrics

### 1. Baseline Managed Memory
- **What**: cudaMallocManaged memory used by matrix tiles in baseline (no optimization)
- **Example**: 8.00 MB for N=1024
- **Formula**: `N × N × sizeof(double) / (1024²)` MB
- **Source**: "Total managed memory" from tiledCholeskyNaiveGraph output

### 2. Baseline Peak Memory
- **What**: Total GPU memory usage during baseline execution (includes workspace, libraries, runtime)
- **Example**: 962.19 MB for N=1024
- **Source**: "Peak GPU memory usage during execution" from tiledCholeskyNaiveGraph
- **Components**:
  - Matrix tiles (managed): ~8 MB
  - cuSOLVER workspace: ~hundreds of MB
  - cuBLAS context: ~hundreds of MB
  - CUDA runtime overhead: ~hundreds of MB

### 3. Optimized Managed Memory
- **What**: cudaMallocManaged memory after optimization (anticipated/planned by optimizer)
- **Example**: 2.00 MB for N=1024
- **Source**: "Optimized peak memory usage (MiB)" from optimizer output
- **Note**: This is the optimizer's PLAN, not measured reality

### 4. Optimized Actual Peak Memory
- **What**: Total GPU memory usage during optimized execution (measured)
- **Example**: ~962-965 MB for N=1024
- **Source**: "Peak GPU memory usage during execution" from tiledCholeskyAblation
- **Note**: This is MEASURED reality during execution

### 5. Managed Memory Reduction (%)
- **What**: Percentage reduction in managed memory (matrix tiles only)
- **Formula**: `(1 - Optimized_Managed / Baseline_Managed) × 100`
- **Example**: `(1 - 2.00 / 8.00) × 100 = 75.0%`
- **Significance**: Shows how well the optimizer reduces tile memory
- **Expected Range**: 60-75% for typical workloads

### 6. Peak Memory Reduction (%)
- **What**: Percentage reduction in total GPU memory
- **Formula**: `(1 - Optimized_Peak / Baseline_Peak) × 100`
- **Example**: `(1 - 965.00 / 962.19) × 100 = -0.3%` (negative means increase)
- **Significance**: Shows ACTUAL memory savings on GPU
- **Expected Range**: ~0% or slightly negative (due to migration overhead)
- **Why Minimal**: Dominated by library workspace which cannot be optimized

---

## Runtime Metrics

### 7. Baseline Runtime
- **What**: Average execution time for baseline (no optimization)
- **Measurement**: 10 runs averaged after warmup and dry run
- **Unit**: Seconds
- **Example**: 0.001s for N=1024
- **Source**: "Execution time" from tiledCholeskyNaiveGraph (average of 10 runs)

### 8. Predicted Runtime (Optimized)
- **What**: Optimizer's prediction for execution time with optimization
- **Source**: "Predicted runtime" or "Longest path in optimized graph" from optimizer
- **Unit**: Seconds
- **Example**: 0.0013s for N=1024
- **Note**: This is what the performance model PREDICTS

### 9. Actual Runtime (Optimized)
- **What**: Measured execution time with optimization
- **Measurement**: 10 runs averaged after warmup
- **Unit**: Seconds
- **Example**: 0.0017s for N=1024
- **Source**: "Execution time" from tiledCholeskyAblation (average of 10 runs)

### 10. Predicted Slowdown (x)
- **What**: How much slower the optimizer predicts execution will be
- **Formula**: `Predicted_Runtime / Baseline_Runtime`
- **Example**: `0.0013 / 0.001 = 1.32x`
- **Interpretation**:
  - `> 1.0`: Predicted to be slower
  - `= 1.0`: Predicted same speed
  - `< 1.0`: Predicted to be faster
- **Expected**: >1.0 due to memory-constrained execution

### 11. Actual Slowdown (x)
- **What**: How much slower the optimized version actually runs
- **Formula**: `Actual_Runtime / Baseline_Runtime`
- **Example**: `0.0017 / 0.001 = 1.70x`
- **Interpretation**:
  - `> 1.0`: Optimized is slower (expected - memory reduction has performance cost)
  - `= 1.0`: Same speed
  - `< 1.0`: Optimized is faster (rare)
- **Expected**: 1.2-2.0x for typical workloads

### 12. Prediction Error (%)
- **What**: How accurate the performance model's prediction was
- **Formula**: `|Predicted_Runtime - Actual_Runtime| / Actual_Runtime × 100`
- **Example**: `|0.0013 - 0.0017| / 0.0017 × 100 = 22.3%`
- **Significance**: Measures performance model accuracy
- **Target**: <20% error is good, <10% is excellent

---

## Computational Throughput Metrics

### 13. Baseline TFLOPS
- **What**: Computational throughput for baseline execution
- **Formula**: `(N³ / 3) / (Baseline_Runtime × 10¹²)` TFLOPS
- **Example**: `(1024³ / 3) / (0.001 × 10¹²) = 0.47 TFLOPS`
- **Significance**: Higher is better (more operations per second)

### 14. Optimized TFLOPS
- **What**: Computational throughput for optimized execution
- **Formula**: `(N³ / 3) / (Actual_Runtime × 10¹²)` TFLOPS
- **Example**: `(1024³ / 3) / (0.0017 × 10¹²) = 0.27 TFLOPS`
- **Significance**: Shows performance impact of memory optimization
- **Expected**: Lower than baseline due to slowdown

---

## Optimization Behavior Metrics

### 15. Number of Migrations
- **What**: How many data migrations the optimizer scheduled
- **Source**: "Number of migrations" from optimizer output
- **Example**: 0 for small N, increases with N
- **Significance**: More migrations = more memory reduction but potentially more overhead

---

## Summary Table Format

```
N     Managed Red (%)  Peak Red (%)  Pred Slow (x)  Actual Slow (x)  Pred Err (%)  Base TFLOPS  Opt TFLOPS
========================================================================================================
1024  75.0             -0.3          1.32           1.70             22.3          0.47         0.27
```

---

## Key Interpretations

### What Success Looks Like

1. **Managed Memory Reduction**: 60-75% (optimizer working well)
2. **Peak Memory Reduction**: ~0% (expected - dominated by library overhead)
3. **Actual Slowdown**: 1.2-2.0x (acceptable trade-off for memory reduction)
4. **Prediction Error**: <20% (performance model is reasonably accurate)
5. **TFLOPS**: Decreases proportionally to slowdown (expected)

### Red Flags

1. **Managed Memory Reduction < 50%**: Optimizer may not be working properly
2. **Actual Slowdown > 3.0x**: Too much performance loss
3. **Prediction Error > 50%**: Performance model needs improvement
4. **Peak Memory Increase > 10%**: Migration overhead too high

---

## Measurement Methodology

### Baseline (tiledCholeskyNaiveGraph)
- Warmup: 0.3s cuBLAS GEMM
- Validation script dry run: 1 run (discarded)
- Per measurement run:
  - Internal dry run: 1 run (discarded)
  - Internal measured run: 1 run (reported)
- Total measurements: 10 runs averaged

### Optimized (tiledCholeskyAblation)
- Warmup: 0.3s cuBLAS GEMM (built-in)
- Profiling run: 1 run (for optimization)
- Phase 1 (memory minimization): 1 run
- Phase 2 (performance optimization): 10 runs averaged

### Fair Comparison Achieved
- ✅ Same warmup (0.3s)
- ✅ Same dry run overhead (internal + external)
- ✅ Same measurement protocol (10 runs averaged)
- ✅ Same memory metrics (Peak GPU memory)

---

## CSV Output Columns

The results CSV contains all these metrics:
- `n`: Domain size
- `t`: Tile size
- `baseline_runtime`: Baseline runtime (s)
- `baseline_managed_memory`: Baseline managed memory (MB)
- `baseline_peak_memory`: Baseline peak memory (MB)
- `baseline_tflops`: Baseline TFLOPS
- `optimized_managed`: Optimized managed memory (MB)
- `actual_peak`: Optimized actual peak memory (MB)
- `managed_reduction`: Managed memory reduction (%)
- `peak_reduction`: Peak memory reduction (%)
- `migrations`: Number of migrations
- `predicted_runtime`: Predicted runtime (s)
- `actual_runtime`: Actual runtime (s)
- `predicted_slowdown`: Predicted slowdown (x)
- `actual_slowdown`: Actual slowdown (x)
- `prediction_error`: Prediction error (%)
- `optimized_tflops`: Optimized TFLOPS

---

## Notes

- All memory values in MB (megabytes)
- All time values in seconds
- Slowdown is multiplicative (1.5x = 50% slower)
- Reduction is percentage (75% = reduced to 25% of original)
- TFLOPS = Tera Floating Point Operations Per Second
