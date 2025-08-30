# FRUGAL Optimization - TODO List

## ✅ Completed Tasks
- [x] Fixed CUDA graph memory leak (21GB) using cudaDeviceGraphMemTrim
- [x] Fixed iterator bug in memoryManager_v2.cu (reference vs copy issue)
- [x] Fixed missing executeGraph call in executor_v2.cu
- [x] Merged LU decomposition branch with task graph visualization
- [x] Created playground test for CUDA graph memory behavior
- [x] Documented CUDA graph memory management findings

## 🔧 Pending Optimization Tasks

### Phase 1: First Step Solver Performance
- [ ] **Fix Phase 1 optimization time consuming issue**
  - Current issue: First step solver takes excessive time for large task graphs
  - Location: `optimization/strategies/firstStepSolver.cpp`
  - Potential causes:
    - O(n²) or worse complexity in task ordering algorithm
    - Inefficient memory overlap calculations
    - Excessive debug output generation

### Phase 2: Second Step Solver Performance  
- [ ] **Fix Phase 2 optimization time consuming issue**
  - Current issue: Second step solver (Gurobi MIP) takes too long to converge
  - Location: `optimization/strategies/secondStepSolver.cpp`
  
**Priority 1: Core Performance Issues**
- [ ] **Reduce o[i][j][k] variable explosion**
  - Limit offload destinations to next K tasks instead of all future tasks
  - Impact: Reduces variables from O(n²m) to O(nKm)
  - Location: `secondStepSolver.cpp:299-344` (defineDecisionVariables)

- [ ] **Tighten big-M constraints** 
  - Change from `1000×originalRuntime` to `~10×originalRuntime`
  - Impact: Tighter LP bounds → faster convergence  
  - Location: `secondStepSolver.cpp:743` (addLongestPathConstraints)

**Priority 2: Solver Configuration**
- [ ] **Add configurable Gurobi parameters**
  - MIP gap tolerance, time limits, thread count
  - Impact: Stop when "good enough" solution found
  - Location: Add to `configurationManager.hpp` + `secondStepSolver.cpp:1198-1206`

- [ ] **Set variable branching priorities**
  - Optimize large arrays first, small arrays last
  - Impact: Focus solver effort on high-impact decisions
  - Location: `secondStepSolver.cpp:261-345` (after variable creation)

**Priority 3: Advanced Techniques**  
- [ ] **Implement warm start with feasible solution**
  - Provide initial solution from greedy heuristic
  - Impact: Start solver from good point instead of scratch
  - Location: Before `solver->Solve()` call

- [ ] **Add temporal decomposition for large problems**
  - Break into overlapping time windows for very large instances
  - Impact: Handle problems that don't fit in memory
  - Location: New wrapper around existing solver

### Benchmarks and Validation
- [ ] **Add ResNet benchmark**
  - Implement ResNet model as new benchmark
  - Location: Create new file in `userApplications/`
  - Requirements:
    - Support different ResNet variants (ResNet-50, ResNet-101)
    - Implement tiled convolution operations
    - Add memory profiling hooks

- [ ] **Run decomposition validation experiment**
  - Validate two-phase decomposition effectiveness
  - Compare memory usage: baseline vs optimized
  - Measure performance overhead
  - Generate comparison plots and statistics

## 📝 Notes
- All critical memory bugs have been resolved
- System is stable for development
- Current focus: Performance optimization of the two-phase decomposition