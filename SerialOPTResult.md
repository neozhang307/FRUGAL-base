# Optimization Result Serialization Documentation

**Date**: November 2024
**Purpose**: Document the serialization functions for OptimizationOutput (optimization plans)

---

## Overview

The FRUGAL framework has internal functions for saving and loading optimization plans (OptimizationOutput), but these are not exposed in public headers. For the ablation study workflow, we need public access to these serialization functions.

## Existing Internal Functions

### Location: `optimization/optimizer.cu`

```cpp
// Save optimization plan to JSON file
void writeOptimizationOutputToFile(const OptimizationOutput &output, const std::string &path);

// Load optimization plan from JSON file
OptimizationOutput loadOptimizationOutput(const std::string &path);
```

These functions:
- Are used internally by `profileAndOptimize()` when `loadExistingPlan` is set in config
- Handle the complete serialization/deserialization of OptimizationOutput
- Are NOT exposed in any public header (`optimization.hpp` or `optimizer.hpp`)

## Problem

For the ablation study workflow, we need to:
1. **Profile-only mode**: Save profiling data (OptimizationInput) ✅ Done
2. **Standalone optimizer**: Load profiling data, optimize, save plan (OptimizationOutput)
3. **Run-plan mode**: Load and execute existing plan (OptimizationOutput) ❌ Need access

The issue: Step 3 requires loading OptimizationOutput, but `loadOptimizationOutput()` is not accessible.

## Solution Approach

### Option 1: Expose Existing Functions (Not chosen)
- Add declarations to `optimizer.hpp`
- Make existing functions public
- Pro: No code duplication
- Con: Changes core API, may expose internals

### Option 2: Create New Public Serializer (Chosen)
- Add to `optimizationInputSerializer.cpp/hpp`
- Rename to `optimizationSerializer.cpp/hpp` for both Input and Output
- Create public wrappers or duplicate functionality
- Pro: Clean public API, consistent with OptimizationInput approach
- Con: Some code duplication

## Implementation Plan

1. **Rename serializer files**:
   - `optimizationInputSerializer.*` → `optimizationSerializer.*`
   - Handle both OptimizationInput and OptimizationOutput

2. **Add new functions**:
   ```cpp
   namespace memopt {
   // Existing
   void saveOptimizationInput(const OptimizationInput& input, const std::string& path);
   OptimizationInput loadOptimizationInput(const std::string& path);

   // New - wrapping or duplicating internal functions
   void saveOptimizationOutput(const OptimizationOutput& output, const std::string& path);
   OptimizationOutput loadOptimizationOutput(const std::string& path);
   }
   ```

3. **Implementation options**:
   - **Option A**: Make internal functions accessible and wrap them
   - **Option B**: Duplicate the JSON serialization logic
   - **Option C**: Move the functions from optimizer.cu to the serializer

## JSON Format for OptimizationOutput

Based on existing `luOptimizationPlan.json`:
```json
{
  "arraysInitiallyAllocatedOnDevice": [0, 1, 5],
  "nodes": [
    {
      "nodeId": 0,
      "nodeType": 0,      // 0=empty, 1=task, 2=dataMovement
      "taskId": 0,
      "arrayId": 65535,   // 65535 = invalid/none
      "direction": 0,     // 0=hostToDevice, 1=deviceToHost
      "edges": [1, 2, 3]  // Connected node IDs
    }
  ],
  "originalMemoryUsage": 51200.0,
  "anticipatedPeakMemoryUsage": 25600.0
}
```

## Usage in Ablation Workflow

```bash
# Step 1: Profile (requires GPU)
./tiledCholeskyAblation --N=2048 --T=8 --profile-only --save-profile=profile.json

# Step 2: Optimize offline (CPU only, separate tool)
./standalone-optimizer profile.json optimized_plan.json --migration-weight=0.5

# Step 3: Execute plan (requires GPU)
./tiledCholeskyAblation --N=2048 --T=8 --run-plan --load-plan=optimized_plan.json
```

## Current Status

- ✅ OptimizationInput serialization implemented
- ❌ OptimizationOutput serialization needs to be made public
- ❌ --run-plan mode blocked by lack of public loadOptimizationOutput

## Next Steps

1. Implement public OptimizationOutput serialization functions
2. Update tiledCholeskyAblation to use the public functions
3. Test the complete workflow
4. Create standalone optimizer tool

---

*Note: This documents the design decision to create separate public serializer functions rather than exposing internal optimizer functions, maintaining clean API boundaries.*