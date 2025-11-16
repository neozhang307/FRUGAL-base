# Removing MemoryManager Dependency from Optimizer

**Document Type**: Implementation Plan
**Date**: November 2024
**Status**: Planning
**Branch**: CGO26/revision-v4

---

## Executive Summary

Remove all `MemoryManager::getInstance()` calls from the Optimizer by pre-computing array sizes during profiling and storing them in `OptimizationInput`. This enables offline optimization without needing complex virtual memory manager infrastructure.

---

## Problem Analysis

### Current Issue
The Optimizer internally calls `MemoryManager::getInstance()` to get array sizes:
- **Line 704** in `constructOptimizationInput()` - Gets sizes for statistics
- **Line 754** in `writeTaskGraphToDot()` - Gets sizes for visualization

This breaks offline optimization because MemoryManager singleton is empty without GPU context.

### Root Cause
The Optimizer shouldn't need to query MemoryManager during optimization - it should have all necessary data upfront.

---

## Simple Solution

### Core Idea
**Move array size queries from optimization-time to profiling-time**

### Implementation

#### Step 1: Extend OptimizationInput
```cpp
// In optimizationInput.hpp
struct OptimizationInput {
  // ... existing fields ...

  // NEW: Pre-computed array sizes from profiling
  std::map<void*, size_t> arraySizes;
};
```

#### Step 2: Populate During Profiling
```cpp
// In Optimizer::profileGraph() - when we have GPU access
OptimizationInput input;
// ... existing profiling code ...

// NEW: Capture all array sizes while we have MemoryManager access
std::set<void*> allArrays;
for (const auto& taskGroup : input.nodes) {
  for (void* ptr : taskGroup.dataDependency.inputs) {
    allArrays.insert(ptr);
  }
  for (void* ptr : taskGroup.dataDependency.outputs) {
    allArrays.insert(ptr);
  }
}

// Store sizes for offline use
auto& memManager = MemoryManager::getInstance();
for (void* ptr : allArrays) {
  input.arraySizes[ptr] = memManager.getSize(ptr);
}
```

#### Step 3: Use Stored Sizes in Optimizer
```cpp
// BEFORE (line 704 in constructOptimizationInput):
auto &memManager = MemoryManager::getInstance();
size_t s = memManager.getSize(p);

// AFTER:
size_t s = optimizationInput.arraySizes[p];
```

```cpp
// BEFORE (line 754 in writeTaskGraphToDot):
auto &memManager = MemoryManager::getInstance();
uniqueArrays[ptr] = memManager.getSize(ptr);

// AFTER:
uniqueArrays[ptr] = optimizationInput.arraySizes[ptr];
```

---

## Implementation Plan

### Phase 1: Add Array Sizes to OptimizationInput
1. Add `std::map<void*, size_t> arraySizes` to OptimizationInput struct
2. Update serialization to save/load array sizes

### Phase 2: Populate Array Sizes During Profiling
1. In `profileGraph()`, collect all unique array pointers
2. Query MemoryManager for each array size
3. Store in `optimizationInput.arraySizes`

### Phase 3: Remove MemoryManager from Optimization
1. Update `constructOptimizationInput()` to use stored sizes
2. Update `writeTaskGraphToDot()` to use stored sizes
3. Remove all `#include "../memory/memoryManager.hpp"` that are no longer needed

### Phase 4: Test
1. Test GPU mode - ensure sizes are captured correctly
2. Test offline mode - ensure optimization works without GPU
3. Verify serialization preserves array sizes

---

## Benefits

1. **Simple** - No virtual classes or complex refactoring
2. **Clean** - Clear separation between profiling (needs GPU) and optimization (doesn't)
3. **Efficient** - Array sizes computed once during profiling
4. **Compatible** - No changes to external APIs
5. **Testable** - Can easily mock OptimizationInput for tests

---

## Files to Modify

1. `optimization/optimizationInput.hpp` - Add arraySizes field
2. `optimization/optimizationSerializer.cpp` - Serialize/deserialize arraySizes
3. `optimization/optimizer.cu` - Populate arraySizes and use them
4. `tools/standaloneOptimizer.cpp` - No changes needed!

---

## Verification

### Success Criteria
1. ✅ Optimizer.cu has zero calls to `MemoryManager::getInstance()`
2. ✅ standaloneOptimizer works without GPU
3. ✅ Array sizes correctly serialized and deserialized
4. ✅ Existing GPU mode continues to work

### Test Commands
```bash
# GPU mode (should work as before)
./build/userApplications/tiledCholeskyAblation --profile-only

# Offline mode (should now work!)
./build/tools/standaloneOptimizer -i optimization_input.json -o optimized_plan.json
```

---

## Cleanup

After this implementation succeeds, we can remove:
- `memory/IMemoryManager.hpp` (not needed)
- `memory/GpuMemoryManagerAdapter.hpp` (not needed)
- `memory/OfflineMemoryManager.hpp` (not needed)
- `memory/MemoryManagerFactory.hpp` (not needed)
- The stashed changes in git

This is a much simpler solution than the virtual memory manager approach!