# Offline Optimization Workflow Status

## Current State (November 2024)

### ✅ Working Components

1. **Profile-only mode** (`--profile-only`)
   - Saves profiling data to JSON with array IDs and dependencies
   - Includes task groups, timing, and array information

2. **Standalone optimizer**
   - Loads profiling data from JSON
   - Runs optimization algorithms
   - Generates optimized execution plans

3. **Serialization functions**
   - `saveOptimizationInput/loadOptimizationInput` for profiling data
   - `saveOptimizationOutput/loadOptimizationOutput` for execution plans

### ⚠️ Architectural Limitation

The fully separated workflow (Profile → Standalone Optimize → Execute Plan) has a fundamental issue:

**The Problem:**
- The optimizer uses `MemoryManager` to map memory addresses to array IDs
- During profiling, real memory addresses are registered (0x7f1234... → array ID 0)
- The standalone optimizer creates fake addresses to satisfy the API (0x1000000 → array ID 0)
- When loading a plan for execution, new real addresses are allocated (0x7f5678... → array ID 0)
- The executor can't properly map between the plan's array references and actual memory

**Root Cause:**
The optimization pipeline was designed to work within a single process where memory addresses remain constant. The core optimization code (`twoStepOptimizationStrategy.cu`) converts addresses to array IDs using:
```cpp
for (auto arrayAddress : node.dataDependency.inputs) {
    secondStepInput.taskGroupInputArrays[i].insert(
        MemoryManager::getInstance().getArrayId(arrayAddress));
}
```

This assumes the MemoryManager has the same address → ID mapping throughout the entire pipeline.

### 🔧 Workarounds

1. **Integrated Mode** (Recommended)
   ```bash
   ./tiledCholeskyAblation --N=1024 --T=8
   ```
   Profile, optimize, and execute in the same process.

2. **Semi-Offline Mode** (Possible future enhancement)
   ```bash
   # On GPU node: Profile and save
   ./tiledCholeskyAblation --profile-only --save-profile=profile.json

   # On same or different node: Load profile, optimize, and execute
   ./tiledCholeskyAblation --load-profile=profile.json --optimize-and-run
   ```
   This would load the profile, run optimization, and execute in the same process.

### 📋 Required Changes for Full Offline Support

To properly support the fully offline workflow, the following changes would be needed:

1. **Modify OptimizationInput structure** to include array ID mappings directly in task groups, not just void* pointers

2. **Update twoStepOptimizationStrategy** to work with array IDs directly when MemoryManager doesn't have real addresses

3. **Enhance the executor** to properly map array IDs in the plan to the current memory layout

4. **Ensure deterministic array ID assignment** so the same arrays get the same IDs across different runs

These changes would require significant refactoring of the core optimization pipeline.

## Summary

The current implementation provides:
- ✅ Ability to save profiling data
- ✅ Standalone optimizer that can process profiles
- ✅ Plan serialization/deserialization

But has limitations:
- ❌ Cannot execute plans generated in a different process
- ❌ Memory address → Array ID mapping is not portable

For production use, the integrated mode is recommended until the architecture can be updated to properly support cross-process optimization workflows.