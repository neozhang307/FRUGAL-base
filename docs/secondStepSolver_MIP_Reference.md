# SecondStepSolver MIP Variable Reference

## Problem Dimensions
- `numberOfTaskGroups` (n): Number of computational tasks to schedule
- `numberOfArraysManaged` (m): Number of memory arrays to manage
- `numberOfVertices`: Total vertices in execution graph = 2n + 2nm

## Primary Decision Variables

### Initial Placement
- **`initiallyAllocatedOnDevice[j]`**: Binary, whether array j starts on device (0 or 1)
  - Size: O(m) variables
  - Purpose: Determine initial memory placement

### Prefetch Operations
- **`p[i][j]`**: Binary, whether to prefetch array j before task i starts
  - Size: O(nm) variables
  - **Dynamic Lookback Optimization**: Uses preprocessing to eliminate 87-88% of variables
  - Constraints: 
    - `p[i][j] = 0` if task i-1 used array j (already on device)
    - `p[i][j] = 1` if `shouldAllocateWithoutPrefetch[i,j]` is true
    - `p[i][j] = 0` if task i is outside lookback window for array j

### Offload Operations  
- **`o[i][j][k]`**: Binary, whether to offload array j after task i for use at task k
  - Size: O(n²m) variables (main bottleneck!)
  - Constraints:
    - `o[i][j][k] = 0` if k ≤ i (no backward offloads)
    - `o[i][j][k] = 0` if task i doesn't use array j
    - `o[i][j][k] = 0` if k > nextTaskUsingArray[j]
  - Relationship: `sum(o[i][j][k] for all k) ≤ 1` (at most one offload destination)

## State Tracking Variables

### Memory Allocation State
- **`x[i][j]`**: Binary, whether array j is allocated on device during task i
  - Size: O(nm) variables
  - Formula: `x[i][j] = initiallyAllocatedOnDevice[j] + sum(p[u][j] for u≤i) - sum(o[u][j][v] for u<i, v≤i)`
  - Meaning: Array is on device if initially there, or prefetched, minus completed offloads

### Memory Availability State
- **`y[i][j]`**: Binary, whether array j is available for task i to use
  - Size: O(nm) variables  
  - Formula: Similar to x[i][j] but considers ALL offloads (not just completed ones)
  - Constraint: `y[i][j] = 1` for all arrays that task i needs

## Graph Structure Variables

### Vertices (Indexed Operations)
Each vertex has an index in range [0, numberOfVertices):

1. **Task Group Start Vertices**: `getTaskGroupStartVertexIndex(i) = i * 2`
   - Indices: 0, 2, 4, 6...
   - Purpose: Synchronization point before task execution

2. **Task Group Execution Vertices**: `getTaskGroupVertexIndex(i) = i * 2 + 1`
   - Indices: 1, 3, 5, 7...
   - Purpose: Actual computation

3. **Prefetch Vertices**: `getPrefetchVertexIndex(i,j) = 2n + i*m + j`
   - Indices: Start at 2n
   - Purpose: Host-to-device transfers

4. **Offload Vertices**: `getOffloadVertexIndex(i,j) = 2n + nm + i*m + j`
   - Indices: Start at 2n + nm
   - Purpose: Device-to-host transfers

### Edge Variables (Dependencies)
- **`e[u][v]`**: Binary, whether vertex u must complete before vertex v starts
  - Size: O(numberOfVertices²) = O(n²m²) variables (MAJOR BOTTLENECK!)
  - Types of edges:
    - Fixed edges: `e[u][v] = 1` (always exist)
    - Conditional edges: `e[u][v] = p[i][j]` or `e[u][v] = o[i][j][k]`
  - Only used in timing constraints

### Timing Variables
- **`z[u]`**: Continuous, start time of vertex u (in seconds)
  - Size: O(numberOfVertices) = O(nm) variables
  - Constraint: `z[u] ≥ z[v] + w[v] - M(1-e[v][u])` for all v,u pairs
  - Purpose: Determine execution schedule

- **`w[u]`**: Continuous, duration/weight of vertex u (in seconds)
  - Size: O(numberOfVertices) = O(nm) variables
  - Values:
    - Task executions: Fixed to `taskGroupRunningTimes[i]`
    - Prefetches: `p[i][j] * arrayPrefetchingTimes[j]`
    - Offloads: `sum(o[i][j][k]) * arrayOffloadingTimes[j]`
    - Start vertices: 0 (instantaneous)

## Objective Function Variables

- **`peakMemoryUsage`**: Continuous, maximum memory used at any point (MiB)
  - Constraint: `peakMemoryUsage ≥ sum(x[i][j] * arraySizes[j])` for all i
  - Appears in objective with weight `weightOfPeakMemoryUsage`

- **`numberOfDataMovements`**: Continuous, total count of prefetches and offloads
  - Formula: `sum(p[i][j]) + sum(o[i][j][k])`
  - Appears in objective with weight `weightOfNumberOfMigrations`

## Key Relationships

### Prefetch-Task Dependencies
```
TaskStart[i] --p[i][j]--> Prefetch[i][j] --> TaskExecution[k] (if k uses j)
```

### Offload-Task Dependencies  
```
TaskExecution[i] --sum(o[i][j][k])--> Offload[i][j] --o[i][j][k]--> TaskStart[k]
```

### Task Sequencing
```
TaskExecution[i-1] --> TaskStart[i] --> TaskExecution[i]
```

### Memory State Evolution
- Array j on device: Tracks via x[i][j]
- Array j available: Tracks via y[i][j]  
- Prefetch adds array to device
- Offload removes array from device (when destination is reached)

## Serialization Constraints (Critical for Understanding e[u][v])

### Why Memory Operations Must Be Sequential
- **Hardware limitation**: Single memory channel - cannot transfer multiple arrays simultaneously
- **Consequence**: All prefetches and offloads form sequential chains

### Prefetch Serialization
```
// Within task i: Prefetch[i][0] → Prefetch[i][1] → ... → Prefetch[i][m-1]
e[getPrefetchVertexIndex(i, j-1)][getPrefetchVertexIndex(i, j)] = 1

// Across tasks: Prefetch[i][m-1] → Prefetch[i+1][0]  
e[getPrefetchVertexIndex(i, m-1)][getPrefetchVertexIndex(i+1, 0)] = 1
```
**Result**: ALL prefetches form ONE continuous chain across entire execution!

### Offload Serialization
```
// Within task i: Offload[i][0] → Offload[i][1] → ... → Offload[i][m-1]
e[getOffloadVertexIndex(i, j-1)][getOffloadVertexIndex(i, j)] = 1

// Across tasks: Offload[i][m-1] → Offload[i+1][0]
e[getOffloadVertexIndex(i, m-1)][getOffloadVertexIndex(i+1, 0)] = 1
```

### Impact on Timing
If p[i][A]=1, p[i][B]=1, p[i][C]=1 with transfer times 5s, 3s, 7s:
```
TaskStart[i] → Prefetch[A](5s) → Prefetch[B](3s) → Prefetch[C](7s) → TaskExecution[i]
Total delay: 15 seconds (not parallel!)
```

## Why e[u][v] Cannot Be Easily Eliminated

### The Asymmetry Problem
- **p[i][j]**: Only 2 indices - "prefetch j at task i" (implicit destinations to all users)
- **o[i][j][k]**: Has 3 indices - "offload j at task i for task k" (explicit destination)

### What e[u][v] Elegantly Handles
1. **Fixed task sequencing**: TaskExecution[i] → TaskStart[i+1] (always)
2. **Conditional prefetch timing**: TaskStart[i] → Prefetch[i][j] (if p[i][j]=1)
3. **Conditional offload timing**: Offload[i][j] → TaskStart[k] (if o[i][j][k]=1)
4. **Complex serialization**: All memory operations in sequential chains
5. **Uniform constraint formulation**: Single constraint type for all dependencies

### Alternatives Considered and Rejected
1. **p[i][j][k] symmetry**: Would need O(nm²) ordering constraints for serialization
2. **Indicator constraints**: Cannot elegantly handle serialization chains
3. **Direct timing constraints**: Would require complex conditional logic for ordering

## Bottlenecks Identified

1. **`o[i][j][k]`**: O(n²m) variables - partially addressed via dependency constraints
2. **`e[u][v]`**: O(n²m²) variables - necessary evil for serialization handling
3. **Timing constraints**: O(n²m²) constraints from nested loops - follows from e[u][v]

## Optimization Status
- **Completed**: Reduced o[i][j][k] search space by ~60-80% using dependency constraints
- **Completed**: Reduced p[i][j] redundant prefetches for consecutive array usage
- **Completed**: Tightened big-M from 1000x to 100x original runtime
- **Completed**: Reduced prefetch edges to only connect to first user (transitivity handles rest)
- **Completed**: **Dynamic Lookback Preprocessing** - eliminates 87-88% of p[i][j] variables
- **Cannot eliminate**: e[u][v] matrix due to serialization complexity
- **Remaining options**: Temporal decomposition, Gurobi parameter tuning, warm start

## Dynamic Lookback Preprocessing (Implemented)

### Algorithm Overview
The dynamic lookback preprocessing dramatically reduces the search space for prefetch operations from O(nm) to approximately O(30m) by constraining which tasks can prefetch each array.

### Key Concepts
1. **Responsibility Zones**: Each task that uses an array is responsible for ensuring it's prefetched within a limited window before its execution
2. **Dual Constraints**: Combines both distance-based (30 tasks) and time-based (configurable factor × transfer time) limits
3. **Non-overlapping Windows**: Prevents multiple tasks from competing to prefetch the same array

### Implementation Details
```cpp
// For each array j and task g that uses it:
// 1. Calculate distance-based window: [g-30, g]
// 2. Calculate time-based window based on accumulated task runtimes
// 3. Use the more restrictive (smaller) window
// 4. Prevent overlap with previous user's window
// 5. Set p[i][j] = 0 for all i outside the window
```

### Performance Impact
- **Variable Reduction**: Eliminates 87-88% of prefetch variables
- **Solve Time**: Reduces from minutes to ~0.4 seconds
- **Solution Quality**: Maintains optimality within the constrained space
- **Scalability**: Enables solving much larger problems

### Configuration Parameters
- `prefetchLookbackDistanceLimit`: Maximum lookback distance (default: 30 tasks)
- `prefetchLookbackTimeBudgetFactor`: Time budget multiplier (default: 2.0, tested up to 10.0)

## Future Research Directions

### 1. Sparse Storage for e[u][v] Matrix
**Status**: Implemented but reverted - no performance improvement observed
**Reason**: Modern MIP solvers (Gurobi) already optimize away zero variables in presolve
**Learning**: The O(n²m²) edge matrix is not the actual bottleneck - the n^m decision space is

Could potentially revisit with:
- Map-based sparse storage: `std::map<std::pair<int,int>, MPVariable*>`
- Only create constraints for actual edges (~25K instead of 150M)
- Reduces memory from 1.2GB to 0.6MB

### 2. Index-based Serialization with Indicator Constraints
The vertex indices (getPrefetchVertexIndex, getOffloadVertexIndex) already encode the serialization order. Could potentially replace e[u][v] by:
1. Using vertex index ordering directly (consecutive indices = sequential operations)
2. Creating auxiliary variables for pairs of active operations
3. Using indicator constraints: IF (both operations active) THEN (enforce timing)

This approach could eliminate O(n²m²) edge variables but would require significant redesign and performance impact is unclear.

### 3. Deallocation Support (TODO)
Currently, the MIP formulation assumes arrays stay in memory once prefetched until explicitly offloaded. 
We could add deallocation capability where arrays can be freed immediately after their last use:

```cpp
// Additional binary variables
d[i][j] = 1 if array j is deallocated after task i (without offloading)

// Modified allocation state constraint
x[i][j] = initiallyAllocatedOnDevice[j] 
          + sum(p[u][j] for u <= i)     // prefetches
          - sum(o[u][j][v] for u < i, v <= i)  // offloads
          - sum(d[u][j] for u < i)      // deallocations

// Constraint: Can only deallocate if array is on device
d[i][j] <= x[i][j]

// Constraint: Cannot deallocate if array is needed by future tasks
for each task k > i that uses array j:
    d[i][j] = 0  // Force no deallocation

// Benefit: Reduces memory without offload overhead
// Cost: More binary variables, increased complexity
```

This would allow freeing memory without the cost of offloading, particularly useful for arrays that won't be needed again.

## Configuration Parameters (from ConfigurationManager)
- `solver`: "GUROBI_MIXED_INTEGER_PROGRAMMING" or "SCIP"
- `maxPeakMemoryUsageInMiB`: Hard limit on memory usage (if > 0)
- `acceptableRunningTimeFactor`: Max slowdown allowed (if > 0)
- `weightOfPeakMemoryUsage`: Objective weight for memory (default 1.0)
- `weightOfTotalRunningTime`: Objective weight for time (default 0.0001)  
- `weightOfNumberOfMigrations`: Objective weight for transfers (default 0.00001)
- `prefetchingBandwidthInGB`: Host-to-device bandwidth (also used for device-to-host in current implementation)
- `offloadingBandwidth`: Device-to-host bandwidth (NOTE: Current code uses same value as prefetchingBandwidth, but could be configured separately)
- `prefetchLookbackDistanceLimit`: Max tasks to look back for prefetch responsibility (default 30)
- `prefetchLookbackTimeBudgetFactor`: Factor × total prefetch time for time budget (default 2.0)
- `offloadLookaheadDistanceLimit`: Max tasks to look ahead for offload destinations (default 30)
- `offloadLookaheadComputeTimeFactor`: Factor × compute time for offload time budget (default 50.0)
- `gurobiTimeLimitSeconds`: Time limit for Gurobi solver (default 60)
- `gurobiMipGap`: MIP gap tolerance (default 0.10)
- `gurobiThreads`: Number of threads (0 = auto)
- `gurobiEnableHeuristics`: Enable aggressive heuristics