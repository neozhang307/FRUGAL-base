#include "secondStepSolver.hpp"

#include <fmt/core.h>
#include <ortools/linear_solver/linear_solver.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <numeric>

#include "../../utilities/configurationManager.hpp"
#include "../../utilities/logger.hpp"
#include "../../utilities/utilities.hpp"

using namespace operations_research;

namespace memopt {
/**
 * Note: We separate the OR-Tools implementation into this helper class
 * because NVCC (CUDA compiler) cannot compile OR-Tools code directly.
 * This structure is used to solve the Mixed Integer Programming problem
 * that determines the optimal memory management strategy.
 */
struct IntegerProgrammingSolver {
  // Internal state of the solver
  SecondStepSolver::Input input;  // Input parameters for the optimization

  // Memory usage metrics (in MiB)
  double originalPeakMemoryUsage;         // Peak memory without any optimization
  double lowestPeakMemoryUsagePossible;   // Theoretical lower bound on peak memory
  float originalTotalRunningTime;         // Original execution time (seconds)
  double originalPeakMemoryUsageToTotalRunningTimeRatio;  // Ratio used for objective weighting
  
  // Problem dimensions
  int numberOfTaskGroups;  // Number of task groups to schedule
  int numberOfArraysManaged;      // Number of memory arrays managed
  int numberOfVertices;    // Total vertices in the execution graph
  
  // Memory management heuristics
  // TODO, not yet functioning
  /*
  if data is only allocate to be written : shouldAllocateWithoutPrefetch
  if data is written but no need further use : shouldDeallocateWithoutOffloading
  */
  std::map<std::pair<int, int>, bool> shouldAllocateWithoutPrefetch;    // For shouldAllocateWithoutPrefetch[i][j], whether array j should be allocated at task i, while no need prefetch
  std::map<std::pair<int, int>, bool> shouldDeallocateWithoutOffloading;  // For shouldDeallocateWithoutOffloading[i][j], Whether array j should be deallocated at task i, while no need offloading

  // OR-Tools MIP solver components
  std::unique_ptr<MPSolver> solver;  // Google OR-Tools Mixed Integer Programming solver
  double infinity;                   // Representation of infinity for the solver

  // Decision variables for the integer programming problem
  
  // Binary variables for initial memory placement
  std::vector<MPVariable *> initiallyAllocatedOnDevice;  // Whether array j is initially on device (0/1)
  
  // Binary variables for memory movement operations
  std::vector<std::vector<MPVariable *>> p;  // p[i][j] = 1 if array j is prefetched at task i
  std::vector<std::vector<std::vector<MPVariable *>>> o;  // o[i][j][k] = 1 if array j is offloaded at task i and needed again at task k
  
  // Binary variables for memory state tracking
  std::vector<std::vector<MPVariable *>> x;  // x[i][j] = 1 if array j is allocated on device at task i
  std::vector<std::vector<MPVariable *>> y;  // y[i][j] = 1 if array j is available for task i (on device)
  
  // Variables for timing and scheduling
  std::vector<MPVariable *> w;  // Weights (execution times) for each vertex
  std::vector<double> arrayPrefetchingTimes;  // Time needed to prefetch each array
  std::vector<double> arrayOffloadingTimes;   // Time needed to offload each array
  
  // Graph structure variables
  std::vector<std::vector<MPVariable *>> e;  // e[i][j] = 1 if there's an edge from vertex i to j
  std::vector<MPVariable *> z;  // z[i] = Time when vertex i starts execution
  
  // Objective function components
  MPVariable *peakMemoryUsage;      // Maximum memory used at any point (to minimize)
  MPVariable *numberOfDataMovements;  // Total number of memory transfers (to minimize)

  /**
   * Helper functions for vertex indexing in the execution graph
   * The solver creates a directed graph with different types of vertices:
   * - Task group start vertices: mark the beginning of a task group
   * - Task group vertices: represent the actual execution of task groups
   * - Prefetch vertices: represent memory transfers from host to device
   * - Offload vertices: represent memory transfers from device to host
   * 
   * These functions map logical entities to indices in the linear vertex array
   */
  
  /**
   * @brief Get the vertex index for a task group's execution
   * @param i The task group ID
   * @return Vertex index in the execution graph
   */
  int getTaskGroupVertexIndex(int i) {
    return i * 2 + 1;  // Each task group has two vertices (start + execution)
  }

  /**
   * @brief Get the vertex index for a task group's start point
   * @param i The task group ID
   * @return Vertex index in the execution graph
   */
  int getTaskGroupStartVertexIndex(int i) {
    return i * 2;  // Even indices are task group start vertices
  }

  /**
   * @brief Get the vertex index for a prefetch operation
   * @param i The task group ID where prefetch occurs
   * @param j The array ID being prefetched
   * @return Vertex index in the execution graph
   */
  int getPrefetchVertexIndex(int i, int j) {
    // After all task group vertices, prefetch vertices are arranged as i*numArrays + j
    return numberOfTaskGroups * 2 + i * numberOfArraysManaged + j;
  }

  /**
   * @brief Get the vertex index for an offload operation
   * @param i The task group ID where offload occurs
   * @param j The array ID being offloaded
   * @return Vertex index in the execution graph
   */
  int getOffloadVertexIndex(int i, int j) {
    // After all prefetch vertices, offload vertices are arranged as i*numArrays + j
    return numberOfTaskGroups * 2 + numberOfTaskGroups * numberOfArraysManaged + i * numberOfArraysManaged + j;
  }

  /**
   * @brief Process the input data to prepare for optimization
   * 
   * This function:
   * 1. Calculates problem dimensions
   * 2. Computes memory usage metrics (original peak memory, theoretical minimum)
   * 3. Sets up arrays for tracking memory allocations and deallocations
   */
  void preprocessSecondStepInput() {
    // Determine problem dimensions from input data
    numberOfTaskGroups = input.taskGroupRunningTimes.size();
    numberOfArraysManaged = input.arraySizes.size();
    
    // Total vertices = task starts + task executions + prefetches + offloads
    numberOfVertices = numberOfTaskGroups * 2 + numberOfTaskGroups * numberOfArraysManaged * 2;

    // Record original execution time for comparison
    originalTotalRunningTime = input.originalTotalRunningTime;

    // ------STAGE BEGIN: Calculate the theoretical lower bound on peak memory usage ------
    // This is the maximum memory needed by any individual task group
    lowestPeakMemoryUsagePossible = 0;
    std::set<int> allDependencies, currentDependencies;
    
    for (int i = 0; i < numberOfTaskGroups; i++) {
      // Find all arrays accessed by this task group (both inputs and outputs)
      currentDependencies.clear();
      std::set_union(
        input.taskGroupInputArrays[i].begin(),
        input.taskGroupInputArrays[i].end(),
        input.taskGroupOutputArrays[i].begin(),
        input.taskGroupOutputArrays[i].end(),
        std::inserter(currentDependencies, currentDependencies.begin())
      );
      
      // Add to the set of all arrays accessed by any task
      // union set remove duplicate arrays
      std::set_union(
        currentDependencies.begin(),
        currentDependencies.end(),
        allDependencies.begin(),
        allDependencies.end(),
        std::inserter(allDependencies, allDependencies.begin())
      );
      
      // Calculate memory needed by this task group and update the lower bound
      // Convert bytes to MiB (divide by 1024*1024)
      double taskMemoryUsage = 1.0 / 1024.0 / 1024.0 * 
        std::accumulate(currentDependencies.begin(), currentDependencies.end(), 
                      static_cast<size_t>(0), [&](size_t total, int arrayId) {
          return total + input.arraySizes[arrayId];
        });
      
      // Update the lower bound if this task needs more memory
      lowestPeakMemoryUsagePossible = std::max(taskMemoryUsage, lowestPeakMemoryUsagePossible);
    }
    // ------STAGE END: Calculate the theoretical lower bound on peak memory usage ------


    // Calculate the original peak memory usage (all arrays together)
    // This is the memory usage without any optimization
    originalPeakMemoryUsage = 1.0 / 1024.0 / 1024.0 * 
      std::accumulate(allDependencies.begin(), allDependencies.end(), 
                    static_cast<size_t>(0), [&](size_t total, int arrayId) {
        return total + input.arraySizes[arrayId];
      });

    // This ratio is used to balance memory vs time in the objective function
    originalPeakMemoryUsageToTotalRunningTimeRatio = 
      static_cast<double>(originalPeakMemoryUsage) / static_cast<double>(originalTotalRunningTime);

    shouldAllocateWithoutPrefetch.clear();
    shouldDeallocateWithoutOffloading.clear();

    // TODO: Uncomment the section below to consider allocation and deallocation
    // in the optimization stage after they are supported in the execution stage.

    // std::vector<int> arrayFirstWritingKernel(numberOfArraysManaged, std::numeric_limits<int>::max());

    // for (auto arr : input.applicationInputArrays) {
    //   arrayFirstWritingKernel[arr] = -1;
    // }

    // for (int i = 0; i < numberOfTaskGroups; i++) {
    //   for (auto arr : input.taskGroupOutputArrays[i]) {
    //     if (i < arrayFirstWritingKernel[arr]) {
    //       arrayFirstWritingKernel[arr] = i;
    //       shouldAllocateWithoutPrefetch[std::make_pair(i, arr)] = true;
    //     }
    //   }
    // }

    // std::vector<int> arrayLastReadingKernel(numberOfArraysManaged, -1);

    // for (auto arr : input.applicationOutputArrays) {
    //   arrayLastReadingKernel[arr] = numberOfTaskGroups;
    // }

    // for (int i = numberOfTaskGroups - 1; i >= 0; i--) {
    //   for (auto arr : input.taskGroupInputArrays[i]) {
    //     if (i > arrayLastReadingKernel[arr]) {
    //       arrayLastReadingKernel[arr] = i;
    //       shouldDeallocateWithoutOffloading[std::make_pair(i, arr)] = true;
    //     }
    //   }
    // }
  }

  void initialize() {
    solver = std::unique_ptr<MPSolver>(MPSolver::CreateSolver(
      ConfigurationManager::getConfig().optimization.solver
    ));

    if (!solver) {
      fmt::print("Solver not available\n");
    }

    infinity = solver->infinity(); // Representation of infinity for the solver
  }

  /**
   * @brief Defines the core decision variables for the memory optimization problem
   * 
   * This function creates the primary binary decision variables that represent:
   * 1. Which arrays should be initially allocated on the device
   * 2. When to prefetch arrays from host to device
   * 3. When to offload arrays from device to host
   * 
   * These variables form the foundation of the optimization model and represent
   * the actual decisions that will be made about memory management.
   */
  void defineDecisionVariables() {
    //---------- SECTION 1: Initial Memory Placement Variables ----------
    // Define binary variables I_j that determine if array j starts on the device
    initiallyAllocatedOnDevice.clear();
    for (int i = 0; i < numberOfArraysManaged; i++) {
      // Create a binary variable with name "I_{i}" for each array
      initiallyAllocatedOnDevice.push_back(solver->MakeBoolVar(fmt::format("I_{{{}}}", i)));
      
      // Default is to start with all arrays on host (upper bound of 0 means I_j = 0)
      initiallyAllocatedOnDevice[i]->SetUB(0);
    }

    // Place all arrays on device, unlease set input.forceAllArraysToResideOnHostInitiallyAndFinally=true
    if (!input.forceAllArraysToResideOnHostInitiallyAndFinally) {
      // For application input arrays, allow them to start on device
      // by raising the upper bound to 1, which allows I_arr to be either 0 or 1
      for (auto arr : input.applicationInputArrays) {
        initiallyAllocatedOnDevice[arr]->SetUB(1);
      }
    }

    //---------- SECTION 2: Prefetching Variables ----------
    // Define binary variables p_ij that determine if array j is prefetched at task group i
    
    // PREPROCESSING: Compute valid prefetch ranges using dynamic lookback strategy
    // For each array j, determine which tasks i are allowed to prefetch it
    const int PREFETCH_LOOKBACK_LIMIT = ConfigurationManager::getConfig().optimization.prefetchLookbackDistanceLimit;
    const double TIME_BUDGET_FACTOR = ConfigurationManager::getConfig().optimization.prefetchLookbackTimeBudgetFactor;
    std::vector<std::vector<bool>> allowPrefetch(numberOfTaskGroups, std::vector<bool>(numberOfArraysManaged, false));
    
    // Calculate prefetch time for each array (size / bandwidth)
    std::vector<float> arrayPrefetchTimes(numberOfArraysManaged);
    for (int j = 0; j < numberOfArraysManaged; j++) {
        arrayPrefetchTimes[j] = static_cast<float>(input.arraySizes[j]) / input.prefetchingBandwidth;
    }
    
    for (int j = 0; j < numberOfArraysManaged; j++) {
      // Find all tasks that use array j
      std::vector<int> tasksUsingArray;
      for (int g = 0; g < numberOfTaskGroups; g++) {
        bool taskUsesArray = (input.taskGroupInputArrays[g].count(j) > 0) ||
                             (input.taskGroupOutputArrays[g].count(j) > 0);
        if (taskUsesArray) {
          tasksUsingArray.push_back(g);
        }
      }
      
      // For each task g that uses array j, determine its prefetch responsibility range
      for (int taskIdx = 0; taskIdx < tasksUsingArray.size(); taskIdx++) {
        int g = tasksUsingArray[taskIdx];
        
        // Calculate total prefetch time for all arrays needed by task g
        // Use set union to avoid double-counting arrays that are both input AND output
        std::set<int> allArraysUsedByTaskG;
        std::set_union(
          input.taskGroupInputArrays[g].begin(), input.taskGroupInputArrays[g].end(),
          input.taskGroupOutputArrays[g].begin(), input.taskGroupOutputArrays[g].end(),
          std::inserter(allArraysUsedByTaskG, allArraysUsedByTaskG.begin())
        );
        
        float totalPrefetchTimeForTaskG = 0;
        for (auto arrayId : allArraysUsedByTaskG) {
          totalPrefetchTimeForTaskG += arrayPrefetchTimes[arrayId];
        }
        
        // Time budget: configurable factor × total prefetch time for this task
        float timeBudget = TIME_BUDGET_FACTOR * totalPrefetchTimeForTaskG;
        
        // Distance-based lookback: g to g-30 (or 0 if g < 30) - include task g itself
        int distanceLookbackStart = std::max(0, g - PREFETCH_LOOKBACK_LIMIT);
        
        // Time-based lookback: accumulate task runtimes until budget exceeded
        int timeLookbackStart = g;
        float accumulatedTime = 0;
        for (int i = g - 1; i >= 0; i--) {
          accumulatedTime += input.taskGroupRunningTimes[i];
          if (accumulatedTime > timeBudget) {
            timeLookbackStart = i + 1;  // Stop at the task after the one that exceeded budget
            break;
          }
          timeLookbackStart = i;  // Continue if still within budget
        }
        
        // Use whichever limit is more restrictive (closer to g)
        int lookbackStart = std::max(distanceLookbackStart, timeLookbackStart);
        int lookbackEnd = g;
        
        // If there's a previous task g' that uses the same array, stop at g'+1
        if (taskIdx > 0) {
          int previousUser = tasksUsingArray[taskIdx - 1];
          lookbackStart = std::max(lookbackStart, previousUser + 1);
        }
        
        // Mark all tasks in [lookbackStart, lookbackEnd] as allowed to prefetch array j for task g
        for (int i = lookbackStart; i <= lookbackEnd; i++) {
          allowPrefetch[i][j] = true;
        }
        
        // Debug output for first few arrays to verify logic
        if (j < 3) {
          LOG_TRACE_WITH_INFO(fmt::format("Array {} Task {} range: [{}, {}] (distance: {}, time: {}, budget: {:.3f}s)", 
                              j, g, lookbackStart, lookbackEnd, distanceLookbackStart, timeLookbackStart, timeBudget).c_str());
        }
      }
    }
    
    // Count total prefetch variables eliminated by lookahead optimization
    int totalPrefetchVars = numberOfTaskGroups * numberOfArraysManaged;
    int eliminatedVars = 0;
    for (int i = 0; i < numberOfTaskGroups; i++) {
      for (int j = 0; j < numberOfArraysManaged; j++) {
        if (!allowPrefetch[i][j]) {
          eliminatedVars++;
        }
      }
    }
    LOG_TRACE_WITH_INFO(fmt::format("Lookback optimization: eliminated {}/{} ({:.1f}%) prefetch variables", 
                        eliminatedVars, totalPrefetchVars, 100.0 * eliminatedVars / totalPrefetchVars).c_str());
    
    p.clear();
    for (int i = 0; i < numberOfTaskGroups; i++) {
      p.push_back({});  // Create a new row for task group i
      for (int j = 0; j < numberOfArraysManaged; j++) {
        // Create binary variable p_{i,j} that will be 1 if we prefetch array j at task i
        p[i].push_back(solver->MakeBoolVar(fmt::format("p_{{{}, {}}}", i, j)));
        
        // If this array must be allocated at this task (from preprocessing),
        // force the variable to be 1 by setting its lower bound -> lower bound is true means forcing true
        if (shouldAllocateWithoutPrefetch[std::make_pair(i, j)]) {
          p[i][j]->SetLB(1);  // This forces p[i][j] = 1 (must prefetch)
        } else if (!allowPrefetch[i][j]) {
          // LOOKBACK OPTIMIZATION: Task i is outside the responsible range for array j
          p[i][j]->SetUB(0);  // Force p[i][j] = 0 (outside lookback window)
        } else if (i > 0) {
          // OPTIMIZATION: Check if previous task used this array
          // If previous task used array j, it's already on device - no need to prefetch
          bool previousTaskUsedArray = (input.taskGroupInputArrays[i-1].count(j) > 0) ||
                                       (input.taskGroupOutputArrays[i-1].count(j) > 0);
          
          if (previousTaskUsedArray) {
            // Array j is already on device from previous task execution
            p[i][j]->SetUB(0);  // Force p[i][j] = 0 (no prefetching needed)
          }
        }
      }
    }

    //---------- SECTION 3: Offloading Variables ----------
    // Define binary variables o_ijk that determine if array j is offloaded at task i
    // and needed again at task k
    
    // PREPROCESSING: Dynamic offload lookahead using compute time budget
    const int OFFLOAD_LOOKAHEAD_LIMIT = ConfigurationManager::getConfig().optimization.offloadLookaheadDistanceLimit;
    const double OFFLOAD_COMPUTE_TIME_FACTOR = ConfigurationManager::getConfig().optimization.offloadLookaheadComputeTimeFactor;
    
    // Calculate offload time for each array (size / bandwidth)
    std::vector<float> arrayOffloadTimes(numberOfArraysManaged);
    for (int j = 0; j < numberOfArraysManaged; j++) {
        arrayOffloadTimes[j] = static_cast<float>(input.arraySizes[j]) / input.prefetchingBandwidth;
    }
    
    // Track statistics for offload variable elimination
    int totalOffloadVars = numberOfTaskGroups * numberOfArraysManaged * numberOfTaskGroups;
    int eliminatedOffloadVars = 0;
    
    o.clear();
    for (int i = 0; i < numberOfTaskGroups; i++) {
      o.push_back({});  // Create a new row for task group i
      for (int j = 0; j < numberOfArraysManaged; j++) {
        o[i].push_back({});  // Create a new subrow for array j
        
        // Check if task i actually uses array j (reads or writes it)
        bool taskUsesArray = (input.taskGroupInputArrays[i].count(j) > 0) ||
                             (input.taskGroupOutputArrays[i].count(j) > 0);
        
        // Find the next task after i that uses array j
        int nextTaskUsingArray = numberOfTaskGroups;  // Default: no future task uses it
        for (int iPrime = i + 1; iPrime < numberOfTaskGroups; iPrime++) {
          if ((input.taskGroupInputArrays[iPrime].count(j) > 0) ||
              (input.taskGroupOutputArrays[iPrime].count(j) > 0)) {
            nextTaskUsingArray = iPrime;
            break;
          }
        }
        
        // Calculate total offload time for all arrays that task i produces/touches
        // Use set union to avoid double-counting arrays that are both input AND output
        std::set<int> allArraysUsedByTaskI;
        std::set_union(
          input.taskGroupInputArrays[i].begin(), input.taskGroupInputArrays[i].end(),
          input.taskGroupOutputArrays[i].begin(), input.taskGroupOutputArrays[i].end(),
          std::inserter(allArraysUsedByTaskI, allArraysUsedByTaskI.begin())
        );
        
        float totalOffloadTimeForTaskI = 0;
        for (auto arrayId : allArraysUsedByTaskI) {
          totalOffloadTimeForTaskI += arrayOffloadTimes[arrayId];
        }

        // CRITICAL CONSTRAINT: For each array j at task i, we can perform at most
        // one memory operation - either prefetch it OR offload it to one destination
        // This creates a constraint: 0 <= p_ij + sum(o_ijk for all k) <= 1
        // MakeRowConstraint(0, 1) creates the bounds for this linear constraint
        auto sumLessThanOneConstraint = solver->MakeRowConstraint(0, 1);
        // SetCoefficient(variable, coefficient) adds the term: coefficient * variable to the constraint
        sumLessThanOneConstraint->SetCoefficient(p[i][j], 1);  // Add p_ij to the constraint with coefficient 1

        // For each possible destination task k where we might need the array again
        for (int k = 0; k < numberOfTaskGroups; k++) {
          // Create binary variable o_{i,j,k} that is 1 if we offload array j at task i
          // and need it again at task k
          o[i][j].push_back(solver->MakeBoolVar(fmt::format("o_{{{},{},{}}}", i, j, k)));
          
          // Add this offload variable to our constraint with coefficient 1
          // This builds the sum: p[i][j] + o[i][j][0] + o[i][j][1] + ... + o[i][j][k] <= 1
          sumLessThanOneConstraint->SetCoefficient(o[i][j][k], 1);
          
          // Track whether this variable will be eliminated
          bool willEliminate = false;
          
          // Apply logical constraints based on deallocation requirements
          if (shouldDeallocateWithoutOffloading[std::make_pair(i, j)]) {
            // If this array must be deallocated at this task,
            // we must offload it to the next task, and nowhere else
            if (k == i + 1) {
              o[i][j][k]->SetLB(1);  // Force o[i][j][k] = 1 (must offload to next task)
            } else {
              o[i][j][k]->SetUB(0);  // Force o[i][j][k] = 0 (cannot offload to other tasks)
              willEliminate = true;
            }
          } else if (!taskUsesArray) {
            // OPTIMIZATION: If task i doesn't use array j, it cannot offload it
            // This significantly reduces the search space by eliminating impossible offloads
            o[i][j][k]->SetUB(0);  // Force o[i][j][k] = 0 (task doesn't use this array)
            willEliminate = true;
          } else if (k > nextTaskUsingArray) {
            // OPTIMIZATION: Cannot offload beyond the next task that uses this array
            // If we offload to k > nextTaskUsingArray, the array won't be available
            // when nextTaskUsingArray needs it
            o[i][j][k]->SetUB(0);  // Force o[i][j][k] = 0 (would miss the next use)
            willEliminate = true;
          } else {
            // DYNAMIC LOOKAHEAD OPTIMIZATION: Check if k is within dynamic window
            // Compute time-based limit: accumulate task runtimes until budget exceeded
            float timeBudget = OFFLOAD_COMPUTE_TIME_FACTOR * totalOffloadTimeForTaskI;
            float accumulatedTime = 0;
            int timeLookaheadEnd = i;
            for (int kPrime = i + 1; kPrime <= k && kPrime < numberOfTaskGroups; kPrime++) {
              accumulatedTime += input.taskGroupRunningTimes[kPrime];
              if (accumulatedTime > timeBudget) {
                timeLookaheadEnd = kPrime - 1;
                break;
              }
              timeLookaheadEnd = kPrime;
            }
            
            // Distance-based limit
            int distanceLookaheadEnd = i + OFFLOAD_LOOKAHEAD_LIMIT;
            
            // Use whichever limit is more restrictive
            int effectiveLookaheadEnd = std::min(distanceLookaheadEnd, timeLookaheadEnd);
            
            if (k > effectiveLookaheadEnd) {
              // LOOKAHEAD OPTIMIZATION: Cannot offload beyond dynamic window
              o[i][j][k]->SetUB(0);  // Force o[i][j][k] = 0 (outside dynamic window)
              willEliminate = true;
            } else if (k <= i) {
              // Cannot offload to a task that's earlier than or same as current task
              // (would create a causality violation)
              o[i][j][k]->SetUB(0);  // Force o[i][j][k] = 0 (impossible offload)
              willEliminate = true;
            }
          }
          
          if (willEliminate) {
            eliminatedOffloadVars++;
          }
        }
      }
    }
    
    // Log statistics about offload variable elimination
    LOG_TRACE_WITH_INFO(fmt::format("Offload lookahead optimization: eliminated {}/{} ({:.1f}%) offload variables", 
                        eliminatedOffloadVars, totalOffloadVars, 
                        100.0 * eliminatedOffloadVars / totalOffloadVars).c_str());
  }

  /**
   * @brief Defines variables that track memory state during execution
   * 
   * This function creates two key sets of binary variables:
   * 1. x[i][j]: Whether array j is allocated on the device during task group i
   * 2. y[i][j]: Whether array j is available for task group i to use
   * 
   * These variables are connected to the decision variables (initiallyAllocatedOnDevice,
   * prefetches, and offloads) through constraints that model memory state transitions.
   * They are essential for tracking memory usage and ensuring data dependencies are met.
   */
  void defineXAndY() {
    //---------- SECTION 1: Memory Allocation State Tracking (x variables) ----------
    // Define binary variables tracking whether each array is allocated on device at each task
    x.clear();
    for (int i = 0; i < numberOfTaskGroups; i++) {
      x.push_back({});  // Create new row for task group i
      for (int j = 0; j < numberOfArraysManaged; j++) {
        // Create binary variable x[i][j] that will be 1 if array j is on device at task i
        x[i].push_back(solver->MakeBoolVar(fmt::format("x_{{{}, {}}}", i, j)));

        // Define the equation that determines the value of x[i][j] based on memory operations:
        //
        // x[i][j] = initiallyAllocatedOnDevice[j] + sum(p[u][j] for 0 ≤ u ≤ i) - sum(o[u][j][v] for 0 ≤ u < i, u < v ≤ i)
        //
        // This equation means:
        // - Array j is on device at task i if it was EITHER:
        //   1. Initially allocated on device, OR
        //   2. Prefetched at any task u from 0 to i
        // - BUT it's NOT on device if it was offloaded at some task u before i and 
        //   the destination task v is before or at task i
        auto constraint = solver->MakeRowConstraint(0, 0);  // Create equation: LHS = 0
        
        // Term for x[i][j] with negative coefficient (moves to right side of equation)
        constraint->SetCoefficient(x[i][j], -1);  // -x[i][j]
        
        // Term for initial allocation
        constraint->SetCoefficient(initiallyAllocatedOnDevice[j], 1);  // +initiallyAllocatedOnDevice[j]
        
        // Terms for all possible prefetches that happened before or at task i
        for (int u = 0; u <= i; u++) {
          constraint->SetCoefficient(p[u][j], 1);  // +p[u][j]
        }
        
        // Terms for all possible offloads that happened before task i
        // with destination before or at task i
        for (int u = 0; u <= i - 1; u++) {
          for (int v = u + 1; v <= i; v++) {
            constraint->SetCoefficient(o[u][j][v], -1);  // -o[u][j][v]
          }
        }
        
        // The equation effectively becomes:
        // x[i][j] = initiallyAllocatedOnDevice[j] + sum(prefetches) - sum(offloads)
      }
    }

    // If required, force all arrays to be on host at the end
    if (input.forceAllArraysToResideOnHostInitiallyAndFinally) {
      for (int j = 0; j < numberOfArraysManaged; j++) {
        // Force x[last_task][j] = 0 for all arrays j
        x[numberOfTaskGroups - 1][j]->SetUB(0);  // Upper bound of 0 forces variable to be 0
      }
    }

    //---------- SECTION 2: Memory Availability Tracking (y variables) ----------
    // Define binary variables tracking whether each array is available for each task
    // An array is "available" if it's on device and not scheduled for offloading
    y.clear();
    for (int i = 0; i < numberOfTaskGroups; i++) {
      y.push_back({});  // Create new row for task group i
      for (int j = 0; j < numberOfArraysManaged; j++) {
        // Create binary variable y[i][j] that will be 1 if array j is available at task i
        y[i].push_back(solver->MakeBoolVar(fmt::format("y_{{{}, {}}}", i, j)));

        // First constraint: Array availability based on allocation state
        //
        // y[i][j] = initiallyAllocatedOnDevice[j] + sum(p[u][j] for 0 ≤ u ≤ i) - sum(o[u][j][v] for any u < i, any v)
        //
        // This means the array is available if it's allocated on device (like x[i][j])
        // BUT also considers offloads that haven't completed yet
        auto constraint = solver->MakeRowConstraint(0, 0);  // Create equation: LHS = 0
        
        // Term for y[i][j] with negative coefficient (moves to right side of equation)
        constraint->SetCoefficient(y[i][j], -1);  // -y[i][j]
        
        // Term for initial allocation
        constraint->SetCoefficient(initiallyAllocatedOnDevice[j], 1);  // +initiallyAllocatedOnDevice[j]
        
        // Terms for all possible prefetches that happened before or at task i
        for (int u = 0; u <= i; u++) {
          constraint->SetCoefficient(p[u][j], 1);  // +p[u][j]
        }
        
        // Terms for ALL possible offloads that affect availability
        // (offloads initiated before task i with ANY destination)
        for (int u = 0; u <= i - 1; u++) {
          for (int v = u + 1; v < numberOfTaskGroups; v++) {
            constraint->SetCoefficient(o[u][j][v], -1);  // -o[u][j][v]
          }
        }

        // Second constraint: Array cannot be available if it's scheduled for offloading at task i
        //
        // This adds the constraint: y[i][j] ≤ 1 - sum(o[i][j][k] for all k > i)
        // Which means y[i][j] must be 0 if ANY o[i][j][k] is 1
        auto offloadingConstraint = solver->MakeRowConstraint(0, 1);  // Create inequality: 0 ≤ LHS ≤ 1
        offloadingConstraint->SetCoefficient(y[i][j], 1);  // +y[i][j]
        
        // Add terms for all possible offload destinations from task i
        for (int k = i + 1; k < numberOfTaskGroups; k++) {
          offloadingConstraint->SetCoefficient(o[i][j][k], -1);  // -o[i][j][k]
        }
        // The inequality effectively becomes:
        // y[i][j] + sum(o[i][j][k]) ≤ 1, which means y[i][j] and sum(o[i][j][k]) cannot both be 1
      }
    }
  }

  /**
   * @brief Defines the execution time weights for all operations in the execution graph
   * 
   * This function sets up the runtime performance model for the optimization problem by:
   * 1. Assigning fixed execution times to task groups based on profiling data
   * 2. Calculating data transfer times based on array sizes and bandwidth
   * 3. Linking the binary decision variables to their corresponding time costs
   * 
   * The weights (w vector) represent how long each operation takes, which is
   * essential for calculating the total execution time and ensuring operations
   * happen in the correct sequence.
   */
  void defineWeights() {
    // Initialize the weights array with the correct size
    w.clear();
    w.resize(numberOfVertices);  // One weight for each vertex in the execution graph

    //---------- SECTION 1: Task Execution Times ----------
    // Assign execution time weights to task group vertices
    for (int i = 0; i < numberOfTaskGroups; i++) {
      // Task group start vertices have zero duration (they're just synchronization points)
      w[getTaskGroupStartVertexIndex(i)] = solver->MakeNumVar(
        0, 0,  // Fixed at exactly 0 seconds
        fmt::format("w_{{{}}}", getTaskGroupStartVertexIndex(i))
      );
      
      // Task group execution vertices have fixed durations from the input data
      // Both lower and upper bounds are set to the same value to fix the duration
      w[getTaskGroupVertexIndex(i)] = solver->MakeNumVar(
        input.taskGroupRunningTimes[i],  // Lower bound = task execution time
        input.taskGroupRunningTimes[i],  // Upper bound = task execution time
        fmt::format("w_{{{}}}", getTaskGroupVertexIndex(i))
      );
    }

    //---------- SECTION 2: Data Transfer Times ----------
    // Calculate data transfer times based on array sizes and bandwidth
    arrayPrefetchingTimes.clear();
    arrayOffloadingTimes.clear();
    for (int i = 0; i < numberOfArraysManaged; i++) {
      // Time = Size / Bandwidth (in seconds)
      arrayPrefetchingTimes.push_back(input.arraySizes[i] / input.prefetchingBandwidth);  // Host-to-device time
      arrayOffloadingTimes.push_back(input.arraySizes[i] / input.offloadingBandwidth);    // Device-to-host time
    }

    //---------- SECTION 3: Prefetch Operation Times ----------
    // Assign timing weights to prefetch operations
    for (int i = 0; i < numberOfTaskGroups; i++) {
      for (int j = 0; j < numberOfArraysManaged; j++) {
        if (shouldAllocateWithoutPrefetch[std::make_pair(i, j)]) {
          // If allocation is mandatory, prefetch time is fixed at 0
          // (we assume allocation overhead is negligible)
          w[getPrefetchVertexIndex(i, j)] = solver->MakeNumVar(
            0, 0,  // Fixed at exactly 0 seconds
            fmt::format("w_{{{}}}", getPrefetchVertexIndex(i, j))
          );
        } else {
          // For optional prefetches, time depends on if we decide to prefetch
          // Variable can be between 0 (no prefetch) and the calculated prefetch time
          w[getPrefetchVertexIndex(i, j)] = solver->MakeNumVar(
            0,                         // Lower bound: 0 seconds (if we don't prefetch)
            arrayPrefetchingTimes[j],  // Upper bound: time to prefetch this array
            fmt::format("w_{{{}}}", getPrefetchVertexIndex(i, j))
          );
          
          // Create constraint: w[prefetchVertex] = p[i][j] * arrayPrefetchingTimes[j]
          // This makes the weight equal to the prefetch time if p[i][j]=1, or 0 if p[i][j]=0
          auto constraint = solver->MakeRowConstraint(0, 0);  // Create equation: LHS = 0
          constraint->SetCoefficient(w[getPrefetchVertexIndex(i, j)], -1);  // -w[vertex]
          constraint->SetCoefficient(p[i][j], arrayPrefetchingTimes[j]);    // +p[i][j]*time
          // Result: w[vertex] = p[i][j]*time
        }
      }
    }

    //---------- SECTION 4: Offload Operation Times ----------
    // Assign timing weights to offload operations
    for (int i = 0; i < numberOfTaskGroups; i++) {
      for (int j = 0; j < numberOfArraysManaged; j++) {
        if (shouldDeallocateWithoutOffloading[std::make_pair(i, j)]) {
          // If deallocation is mandatory, offload time is fixed at 0
          // (we assume deallocation overhead is negligible)
          w[getOffloadVertexIndex(i, j)] = solver->MakeNumVar(
            0, 0,  // Fixed at exactly 0 seconds
            fmt::format("w_{{{}}}", getOffloadVertexIndex(i, j))
          );
        } else {
          // For optional offloads, time depends on if we decide to offload
          // Variable can be between 0 (no offload) and the calculated offload time
          w[getOffloadVertexIndex(i, j)] = solver->MakeNumVar(
            0,                         // Lower bound: 0 seconds (if we don't offload)
            arrayOffloadingTimes[j],   // Upper bound: time to offload this array
            fmt::format("w_{{{}}}", getOffloadVertexIndex(i, j))
          );
          
          // Create constraint: w[offloadVertex] = sum(o[i][j][k] * arrayOffloadingTimes[j] for all k)
          // This makes the weight equal to the offload time if any o[i][j][k]=1, or 0 if all o[i][j][k]=0
          auto constraint = solver->MakeRowConstraint(0, 0);  // Create equation: LHS = 0
          constraint->SetCoefficient(w[getOffloadVertexIndex(i, j)], -1);  // -w[vertex]
          
          // Add each possible offload destination with the appropriate time coefficient
          for (int k = 0; k < numberOfTaskGroups; k++) {
            constraint->SetCoefficient(o[i][j][k], arrayOffloadingTimes[j]);  // +o[i][j][k]*time
          }
          // Result: w[vertex] = sum(o[i][j][k]*time for all k)
        }
      }
    }
  }

  /**
   * @brief Defines the execution graph structure by creating edges between operations
   * 
   * This function builds the dependency graph that represents the execution schedule,
   * connecting task groups, prefetch operations, and offload operations with directed edges.
   * The graph structure enforces execution order constraints and links decision variables
   * to the actual control flow.
   * 
   * The execution graph consists of several types of vertices:
   * - Task group start vertices: Synchronization points before task execution
   * - Task group execution vertices: Actual computation tasks
   * - Prefetch vertices: Memory transfers from host to device
   * - Offload vertices: Memory transfers from device to host
   * 
   * Edges in this graph (e[i][j] = 1) indicate that vertex i must complete before vertex j.
   */
  void addEdges() {
    // Create constant variables for edge presence (0) or absence (1)
    auto zeroConstant = solver->MakeIntVar(0, 0, "zero");  // Fixed at 0 (no edge)
    auto oneConstant = solver->MakeIntVar(1, 1, "one");    // Fixed at 1 (definite edge)

    // Initialize the adjacency matrix for the execution graph
    // e[i][j] = 1 means there's a directed edge from vertex i to vertex j
    e.clear();
    e.resize(numberOfVertices, std::vector<MPVariable *>(numberOfVertices, zeroConstant));

    //---------- SECTION 1: Core Task Execution Flow ----------
    // Connect each task's start to its end, and create the main task sequence
    for (int i = 0; i < numberOfTaskGroups; i++) {
      // Each task execution must be preceded by its start vertex
      // This creates the edge: TaskGrouStart[i] -> TaskGroupEnd[i]
      e[getTaskGroupStartVertexIndex(i)][getTaskGroupVertexIndex(i)] = oneConstant;
      
      // Connect sequential tasks in order (except for the first task)
      // This creates the edge: TaskGrouEnd[i-1] -> TaskGrouStart[i]
      if (i > 0) {
        e[getTaskGroupVertexIndex(i - 1)][getTaskGroupStartVertexIndex(i)] = oneConstant;
      }
    }

    //---------- SECTION 2: Prefetch Operation Dependencies ----------
    // Connect prefetch operations to task group starts and executions
    for (int i = 0; i < numberOfTaskGroups; i++) {
      for (int j = 0; j < numberOfArraysManaged; j++) {
        // The prefetch operation is triggered only if p[i][j] = 1
        // This creates a conditional edge: TaskStart[i] -> <Prefetch[i][j]>
        // The edge exists only if we decide to prefetch array j at task i
        e[getTaskGroupStartVertexIndex(i)][getPrefetchVertexIndex(i, j)] = p[i][j];
        
        // OPTIMIZATION: Connect prefetch only to the FIRST task that needs this array
        // Since tasks execute sequentially, if the first task has the array,
        // all subsequent tasks will also have it available (transitivity)
        for (int k = i; k < numberOfTaskGroups; k++) {
          // Check if task group k reads or writes to array j
          // If only write part of the array still need to prefetch all data
          if (input.taskGroupInputArrays[k].count(j) != 0 || input.taskGroupOutputArrays[k].count(j) != 0) {
          // if (input.taskGroupInputArrays[k].count(j) != 0 ) {
            // This creates the edge: <Prefetch[i][j]> -> TaskEnd[k]
            // Only connect to the FIRST task that uses array j
            e[getPrefetchVertexIndex(i, j)][getTaskGroupVertexIndex(k)] = oneConstant;
            break;  // Stop after connecting to first user - reduces edges significantly
          }
        }
      }
    }

    //---------- SECTION 3: Prefetch Operation Serialization ----------
    // Prefetch operations must happen sequentially (no parallel prefetches)
    // This models the constraint that the memory bus can only handle one transfer at a time
    for (int i = 0; i < numberOfTaskGroups; i++) {
      for (int j = 0; j < numberOfArraysManaged; j++) {
        if (j == 0) {  // First array in the task group
          if (i != 0) {  // Not the first task group
            // Connect last prefetch of previous task to first prefetch of current task
            // This creates the edge: <Prefetch[i-1][last]> -> <Prefetch[i][0]>
            e[getPrefetchVertexIndex(i - 1, numberOfArraysManaged - 1)][getPrefetchVertexIndex(i, j)] = oneConstant;
          }
        } else {
          // Connect sequential prefetches within the same task group
          // This creates the edge: <Prefetch[i][j-1]> -> <Prefetch[i][j]>
          e[getPrefetchVertexIndex(i, j - 1)][getPrefetchVertexIndex(i, j)] = oneConstant;
        }
      }
    }

    //---------- SECTION 4: Offload Operation Dependencies ----------
    // Connect offload operations to task executions and future task starts
    for (int i = 0; i < numberOfTaskGroups; i++) {
      for (int j = 0; j < numberOfArraysManaged; j++) {
        // Create a variable edge from task execution to potential offload operation
        // This edge will exist only if we decide to offload this array after task group i
        
        e[getTaskGroupVertexIndex(i)][getOffloadVertexIndex(i, j)] = solver->MakeBoolVar("");

        // Define a constraint: e[TaskExecution[i]][Offload[i][j]] = sum(o[i][j][k] for all k)
        // This means: "There's an edge to offload array j after task i if and only if
        // we decide to offload this array to some future task group k"
        auto constraint = solver->MakeRowConstraint(0, 0);  // Create equation: LHS = 0
        constraint->SetCoefficient(e[getTaskGroupVertexIndex(i)][getOffloadVertexIndex(i, j)], -1);  // -e[...][...]
        
        // Add all possible offload decisions for this task and array
        for (int k = 0; k < numberOfTaskGroups; k++) {
          constraint->SetCoefficient(o[i][j][k], 1);  // +o[i][j][k]
        }
        // And the edge equation effectively becomes: e[...][...] = sum(o[i][j][k])
        // This creates the edge: <TaskEnd[i]> -> <Offload[i][j]> = 

        // Connect each offload operation to all potential destination task starts
        // For each destination task k, create a conditional edge that exists if
        // we decide to offload array j at task i for future use at task k
        for (int k = i + 1; k < numberOfTaskGroups; k++) {
          // This creates the conditional edge: <Offload[i][j]> -> TaskStart[k]
          // The edge exists only if o[i][j][k] = 1
          e[getOffloadVertexIndex(i, j)][getTaskGroupStartVertexIndex(k)] = o[i][j][k];
        }
      }
    }

    //---------- SECTION 5: Offload Operation Serialization ----------
    // Offload operations must happen sequentially (no parallel offloads)
    // This models the constraint that the memory bus can only handle one transfer at a time
    for (int i = 0; i < numberOfTaskGroups; i++) {
      for (int j = 0; j < numberOfArraysManaged; j++) {
        if (j == 0) {  // First array in the task group
          if (i != 0) {  // Not the first task group
            // Connect last offload of previous task to first offload of current task
            // This creates the edge: <Offload[i-1][last]> -> <Offload[i][0]>
            e[getOffloadVertexIndex(i - 1, numberOfArraysManaged - 1)][getOffloadVertexIndex(i, j)] = oneConstant;
          }
        } else {
          // Connect sequential offloads within the same task group
          // This creates the edge: <Offload[i][j-1]> -> <Offload[i][j]>
          e[getOffloadVertexIndex(i, j - 1)][getOffloadVertexIndex(i, j)] = oneConstant;
        }
      }
    }
  }

  /**
   * @brief Defines constraints to model task timing and calculate the critical path
   * 
   * This function creates timing variables and constraints that model the execution
   * schedule represented by the dependency graph. It uses a mathematical formulation
   * based on the "longest path" in a directed acyclic graph to determine when each
   * operation can start, and ultimately calculates the total execution time.
   * 
   * The key concept is that each vertex in the execution graph has a start time (z),
   * and the time at which a vertex can start depends on when all its predecessors
   * finish their execution. This creates a critical path through the graph that
   * determines the overall execution time.
   */
  void addLongestPathConstraints() {
    // Initialize the array of start time variables for each vertex
    z.clear();

    //---------- SECTION 1: Create Start Time Variables ----------
    // For each vertex, create a variable z[u] representing its start time
    for (int u = 0; u < numberOfVertices; u++) {
      if (u == getTaskGroupStartVertexIndex(0)) {
        // The first task starts at time 0 (this is our reference point)
        z.push_back(solver->MakeNumVar(0, 0, ""));
      } else {
        // All other vertices can start at any non-negative time
        z.push_back(solver->MakeNumVar(0, infinity, ""));
      }
    }

    // Define a large number to represent "infinitely late" for timing constraints
    // This is used in the big-M method to conditionally enforce constraints
    // Using 100x original time as a balance between tightness and flexibility
    // This still improves over 1000x while avoiding being too restrictive
    const float infinityForTime = 100.0 * originalTotalRunningTime;

    //---------- SECTION 2: Add Precedence Constraints ----------
    // For each vertex (except the first), add constraints that it cannot start
    // until all its predecessors have completed
    for (int u = 0; u < numberOfVertices; u++) {
      if (u != getTaskGroupStartVertexIndex(0)) {  // Skip the first task start (fixed at 0)
        // For each potential predecessor v of vertex u
        for (int v = 0; v < numberOfVertices; v++) {
          // Create the constraint: z[u] ≥ z[v] + w[u] - M(1 - e[v][u])
          // Where:
          // - z[u] is the start time of vertex u
          // - z[v] is the start time of predecessor v
          // - w[u] is the execution time of vertex u
          // - e[v][u] is 1 if there's an edge from v to u, 0 otherwise
          // - M is a large constant (infinityForTime)
          //
          // This constraint means:
          // - If e[v][u] = 1 (v is a predecessor of u), then: z[u] ≥ z[v] + w[u]
          //   (u must start after v finishes)
          // - If e[v][u] = 0 (v is not a predecessor), the constraint is trivially satisfied
          auto constraint = solver->MakeRowConstraint(-infinityForTime, infinity);
          constraint->SetCoefficient(z[u], 1);            // +z[u]
          constraint->SetCoefficient(z[v], -1);           // -z[v]
          constraint->SetCoefficient(w[u], -1);           // -w[u]
          constraint->SetCoefficient(e[v][u], -infinityForTime);  // -M*e[v][u]
        }
      }
    }

    //---------- SECTION 3: Add Performance Constraint ----------
    // If a performance threshold is specified, add a constraint limiting total execution time
    if (ConfigurationManager::getConfig().optimization.acceptableRunningTimeFactor > std::numeric_limits<double>::epsilon()) {
      // Calculate the maximum acceptable execution time based on the original time
      // and the configured factor (e.g., 1.1 = 10% longer than original is acceptable)
      auto maxAcceptableTime = originalTotalRunningTime * 
                            ConfigurationManager::getConfig().optimization.acceptableRunningTimeFactor;
      
      // Create constraint: z[TaskGroudEnd(last)] ≤ maxAcceptableTime
      // This limits the start time of the last task to ensure total execution time
      // stays within the acceptable range
      auto zLastKernelConstraint = solver->MakeRowConstraint(
        0,            // Lower bound (trivial as start times are always ≥ 0)
        maxAcceptableTime  // Upper bound based on acceptable slowdown
      );
      zLastKernelConstraint->SetCoefficient(z[getTaskGroupVertexIndex(numberOfTaskGroups - 1)], 1);
    }
  }

  /**
   * @brief Ensures all arrays needed by a task are available on the device when the task executes
   * 
   * This function adds constraints to guarantee that any memory array accessed by a task
   * (either as input or output) must be available on the device when that task executes.
   * This is a fundamental constraint for correctness - a task cannot run if its data
   * dependencies are not satisfied.
   * 
   * The constraint uses the y variables which track whether an array is available
   * (allocated and not being offloaded) for a given task group.
   */
  void addKernelDataDependencyConstraints() {
    // Temporary set to hold the union of input and output arrays for each task
    std::set<int> nodeInputOutputUnion;
    
    // For each task group in the execution sequence
    for (int i = 0; i < numberOfTaskGroups; i++) {
      nodeInputOutputUnion.clear();  // Clear set for reuse
      
      // Calculate the union of input and output arrays for this task group
      // This gives us all memory arrays that this task group accesses
      std::set_union(
        input.taskGroupInputArrays[i].begin(),
        input.taskGroupInputArrays[i].end(),
        input.taskGroupOutputArrays[i].begin(),
        input.taskGroupOutputArrays[i].end(),
        std::inserter(nodeInputOutputUnion, nodeInputOutputUnion.begin())
      );
      
      // For each array accessed by this task group
      for (auto &arr : nodeInputOutputUnion) {
        // Create the constraint: y[i][arr] = 1
        // This forces the array to be available (on device) when the task executes
        auto constraint = solver->MakeRowConstraint(1, 1);  // Equal to exactly 1
        constraint->SetCoefficient(y[i][arr], 1);  // y[i][arr] must equal 1
      }
    }
  }

  /**
   * @brief Defines and constrains the peak memory usage variable in the optimization model
   * 
   * This function creates a variable that tracks the maximum memory usage across all
   * task groups, and ensures it correctly represents the peak memory consumption.
   * This is a key part of the objective function, as minimizing peak memory usage
   * is one of the primary goals of the optimization.
   * 
   * The function also adds an optional constraint to enforce a hard limit on
   * memory usage if specified in the configuration.
   */
  void definePeakMemoryUsage() {
    // Create a variable to represent the peak memory usage in MiB
    // This variable will be included in the objective function for minimization
    peakMemoryUsage = solver->MakeNumVar(0, infinity, "");
    
    // For each task group, add a constraint to ensure peakMemoryUsage is at least
    // as large as the total memory used during that task execution
    for (int i = 0; i < numberOfTaskGroups; i++) {
      // Create constraint: peakMemoryUsage ≥ sum(x[i][j] * arraySizes[j] for all j)
      // This ensures peakMemoryUsage is at least as large as the memory usage at each task
      auto constraint = solver->MakeRowConstraint(0, infinity);  // Lower bound 0, no upper bound
      constraint->SetCoefficient(peakMemoryUsage, 1);  // +peakMemoryUsage
      
      // Add terms for each array's potential memory usage
      for (int j = 0; j < numberOfArraysManaged; j++) {
        // The negative coefficient is because we're moving the array terms to the right side:
        // peakMemoryUsage ≥ sum_j(x[i][j] * arraySizes[j]) becomes:
        // peakMemoryUsage - sum_j(x[i][j] * arraySizes[j]) ≥ 0
        // Convert bytes to MiB by dividing by 1024^2 = 1,048,576
        constraint->SetCoefficient(x[i][j], -static_cast<double>(this->input.arraySizes[j]) / 1024.0 / 1024.0);
      }
      // The constraint effectively becomes:
      // peakMemoryUsage ≥ memory used at task i (in MiB)
    }

    // If a maximum memory limit is specified in the configuration, add a constraint
    // to enforce it as an upper bound
    if (ConfigurationManager::getConfig().optimization.maxPeakMemoryUsageInMiB > std::numeric_limits<double>::epsilon()) {
      // Create constraint: peakMemoryUsage ≤ maxPeakMemoryUsageInMiB
      // This enforces a hard limit on total device memory usage
      auto maxPeakMemoryUsageConstraint = solver->MakeRowConstraint(
        0,  // Lower bound (trivial, as peak memory usage is always ≥ 0)
        ConfigurationManager::getConfig().optimization.maxPeakMemoryUsageInMiB  // Upper bound from config
      );
      maxPeakMemoryUsageConstraint->SetCoefficient(peakMemoryUsage, 1);
    }
  }

  /**
   * @brief Defines and constrains the variable tracking total number of memory transfers
   * 
   * This function creates a variable that counts the total number of memory movement 
   * operations (prefetches and offloads) in the solution. Minimizing this count is 
   * often part of the objective function, as each memory transfer has overhead and 
   * can impact performance.
   * 
   * The number is calculated by summing all prefetch and offload decision variables,
   * which equals the total count of memory transfers between host and device.
   */
  void defineNumberOfDataMovements() {
    // Create a variable to represent the total number of data movements
    // This variable will be included in the objective function for minimization
    numberOfDataMovements = solver->MakeNumVar(0, infinity, "");
    
    // Create constraint: numberOfDataMovements = sum of all prefetches and offloads
    auto constraint = solver->MakeRowConstraint(0, infinity);  // Lower bound 0, no upper bound
    constraint->SetCoefficient(numberOfDataMovements, 1);  // +numberOfDataMovements

    // Count all prefetch operations (host-to-device transfers)
    // Each p[i][j] = 1 represents one prefetch operation
    for (int i = 0; i < numberOfTaskGroups; i++) {
      for (int j = 0; j < numberOfArraysManaged; j++) {
        // The negative coefficient is because we're moving the prefetch terms to the right side:
        // numberOfDataMovements = sum(p[i][j]) + sum(o[i][j][k]) becomes:
        // numberOfDataMovements - sum(p[i][j]) - sum(o[i][j][k]) = 0
        constraint->SetCoefficient(p[i][j], -1);  // -p[i][j]
      }
    }

    // Count all offload operations (device-to-host transfers)
    // Each o[i][j][k] = 1 represents one offload operation
    for (int i = 0; i < numberOfTaskGroups; i++) {
      for (int j = 0; j < numberOfArraysManaged; j++) {
        for (int k = i + 1; k < numberOfTaskGroups; k++) {
          constraint->SetCoefficient(o[i][j][k], -1);  // -o[i][j][k]
        }
      }
    }
    // The constraint effectively becomes:
    // numberOfDataMovements = sum of all prefetches and offloads
  }

  void printSolution(MPSolver::ResultStatus resultStatus) {
    // Skip debug output if disabled in configuration
    auto optimizedPeakMemoryUsage = peakMemoryUsage->solution_value();
    auto totalRunningTime = z[getTaskGroupVertexIndex(numberOfTaskGroups - 1)]->solution_value();

    fmt::print( "Original peak memory usage (MiB): {:.6f}\n", originalPeakMemoryUsage);
    fmt::print( "Lowest peak memory usage possible (MiB): {:.6f}\n", lowestPeakMemoryUsagePossible);
    fmt::print( "Optimal peak memory usage (MiB): {:.6f}\n", optimizedPeakMemoryUsage);
    fmt::print( "Optimal peak memory usage  / Original peak memory usage: {:.6f}%\n", optimizedPeakMemoryUsage / originalPeakMemoryUsage * 100.0);

    fmt::print( "Original total running time (s): {:.6f}\n", originalTotalRunningTime);
    fmt::print( "Total running time (s): {:.6f}\n", totalRunningTime);
    fmt::print("Total running time / original: {:.6f}%\n", totalRunningTime / originalTotalRunningTime * 100.0);

    if (!ConfigurationManager::getConfig().execution.enableDebugOutput) {
      return;
    }
    
    std::string outputFilePath = fmt::format("debug/{}.secondStepSolver.out", input.stageIndex);
    LOG_TRACE_WITH_INFO("Printing solution to %s", outputFilePath.c_str());

    auto fp = fopen(outputFilePath.c_str(), "w");
    if (!fp) {
      LOG_TRACE_WITH_INFO("Could not open debug output file %s", outputFilePath.c_str());
      return;
    }

    fmt::print(fp, "Result status: {}\n", resultStatus == MPSolver::OPTIMAL ? "OPTIMAL" : "FEASIBLE");


    fmt::print(fp, "Original peak memory usage (MiB): {:.6f}\n", originalPeakMemoryUsage);
    fmt::print(fp, "Lowest peak memory usage possible (MiB): {:.6f}\n", lowestPeakMemoryUsagePossible);
    fmt::print(fp, "Optimal peak memory usage (MiB): {:.6f}\n", optimizedPeakMemoryUsage);
    fmt::print(fp, "Optimal peak memory usage  / Original peak memory usage: {:.6f}%\n", optimizedPeakMemoryUsage / originalPeakMemoryUsage * 100.0);

    fmt::print(fp, "Original total running time (s): {:.6f}\n", originalTotalRunningTime);
    fmt::print(fp, "Total running time (s): {:.6f}\n", totalRunningTime);
    fmt::print(fp, "Total running time / original: {:.6f}%\n", totalRunningTime / originalTotalRunningTime * 100.0);



    fmt::print(fp, "Solution:\n");

    for (int i = 0; i < numberOfTaskGroups; i++) {
      fmt::print(fp, "{}:\n", i);

      fmt::print(fp, "  Inputs: ");
      for (auto arrayIndex : input.taskGroupInputArrays[i]) {
        fmt::print(fp, "{}, ", arrayIndex);
      }
      fmt::print(fp, "\n");

      fmt::print(fp, "  Outputs: ");
      for (auto arrayIndex : input.taskGroupOutputArrays[i]) {
        fmt::print(fp, "{}, ", arrayIndex);
      }
      fmt::print(fp, "\n");
    }

    fmt::print(fp, "\n");

    for (int i = 0; i < numberOfArraysManaged; i++) {
      if (initiallyAllocatedOnDevice[i]->solution_value() > 0) {
        fmt::print(fp, "I_{{{}}} = 1\n", i);
      }
    }

    fmt::print(fp, "\n");

    for (int i = 0; i < numberOfTaskGroups; i++) {
      for (int j = 0; j < numberOfArraysManaged; j++) {
        if (p[i][j]->solution_value() > 0) {
          fmt::print(fp, "p_{{{}, {}}} = {}; w = {}; z = {}\n", i, j, true, w[getPrefetchVertexIndex(i, j)]->solution_value(), z[getPrefetchVertexIndex(i, j)]->solution_value());
        }
      }
    }

    fmt::print(fp, "\n");

    for (int i = 0; i < numberOfTaskGroups; i++) {
      for (int j = 0; j < numberOfArraysManaged; j++) {
        for (int k = 0; k < numberOfTaskGroups; k++) {
          if (o[i][j][k]->solution_value() > 0) {
            fmt::print(fp, "o_{{{}, {}, {}}} = {}; w = {}; z = {}\n", i, j, k, true, w[getOffloadVertexIndex(i, j)]->solution_value(), z[getOffloadVertexIndex(i, j)]->solution_value());
          }
        }
      }
    }

    fclose(fp);
  }

  /**
   * @brief Prints diagnostic information about task durations and memory sizes
   * 
   * This function generates a detailed report of the task durations and memory array sizes
   * to help with debugging and performance analysis. It's especially important for
   * understanding the diversity and scale of the problem:
   * 
   * 1. Tasks with extremely short durations may have little impact on scheduling decisions
   * 2. Tasks with very long durations may dominate the critical path
   * 3. Memory arrays with very small sizes might be candidates for always keeping on device
   * 4. Memory arrays with very large sizes may need special handling for efficient transfers
   * 
   * Finding the range (min/max) of these values helps identify potential numerical issues
   * in the solver, as extreme ratios between values can sometimes cause precision problems
   * in the Mixed Integer Programming formulation.
   */
  void printDurationsAndSizes() {
    // Skip debug output if disabled in configuration
    if (!ConfigurationManager::getConfig().execution.enableDebugOutput) {
      return;
    }
    
    // Create output file path with the stage index for identification
    std::string outputFilePath = fmt::format("debug/{}.secondStepSolverInput.out", input.stageIndex);
    LOG_TRACE_WITH_INFO("Printing input to %s", outputFilePath.c_str());

    // Open output file for writing
    auto fp = fopen(outputFilePath.c_str(), "w");
    if (!fp) {
      LOG_TRACE_WITH_INFO("Could not open debug output file %s", outputFilePath.c_str());
      return;
    }
    
    // Find min/max task durations to understand the time scale of the problem
    float minDuration = std::numeric_limits<float>::max();
    float maxDuration = 0;
    for (int i = 0; i < numberOfTaskGroups; i++) {
      auto duration = input.taskGroupRunningTimes[i];
      minDuration = std::min(minDuration, duration);
      maxDuration = std::max(maxDuration, duration);
      // Print each task's duration
      fmt::print(fp, "taskGroupRunningTimes[{}] = {}\n", i, duration);
    }

    // Find min/max memory array sizes to understand the memory scale of the problem
    size_t minSize = std::numeric_limits<size_t>::max();
    size_t maxSize = 0;
    for (int i = 0; i < numberOfArraysManaged; i++) {
      auto size = input.arraySizes[i];
      minSize = std::min(minSize, size);
      maxSize = std::max(maxSize, size);
      // Print each array's size
      fmt::print(fp, "arraySizes[{}] = {}\n", i, size);
    }

    // Print summary statistics showing the range of values
    // This helps identify if there are extreme ratios that might cause
    // numerical instability in the solver
    fmt::print(
      fp,
      "minDuration = {}\nmaxDuration = {}\nminSize = {}\nmaxSize = {}\n",
      minDuration,
      maxDuration,
      minSize,
      maxSize
    );

    // Close the output file
    fclose(fp);
  }

  /**
   * @brief Configure Gurobi environment variables for advanced optimization
   * 
   * Sets Gurobi-specific parameters via environment variables to optimize solver
   * performance for large-scale mixed integer programming problems.
   */
  void configureGurobiEnvironment() {
    auto& config = ConfigurationManager::getConfig().optimization;
    
    // Set MIP focus: 1 = feasibility, 2 = optimality, 3 = bound
    // For large problems, focus on finding feasible solutions quickly
    setenv("GRB_MIPFOCUS", "1", 1);
    
    // Enable aggressive heuristics (0.05 = 5% of solve time on heuristics)
    if (config.gurobiEnableHeuristics) {
      setenv("GRB_HEURISTICS", "0.1", 1);  // 10% of time on heuristics
    } else {
      setenv("GRB_HEURISTICS", "0.0", 1);  // Disable heuristics
    }
    
    // Set thread count (0 = automatic, use all available cores)
    char threadStr[32];
    snprintf(threadStr, sizeof(threadStr), "%d", config.gurobiThreads);
    setenv("GRB_THREADS", threadStr, 1);
    
    // Enable aggressive presolve (2 = aggressive, 1 = conservative, 0 = off)
    setenv("GRB_PRESOLVE", "2", 1);
    
    // Enable all cutting planes aggressively (2 = aggressive, 1 = moderate)
    setenv("GRB_CUTS", "2", 1);
    
    // Enable symmetry detection to reduce problem size
    setenv("GRB_SYMMETRY", "2", 1);
    
    // Use dual simplex for root relaxation (generally faster for MIPs)
    setenv("GRB_METHOD", "1", 1);
    
    // Log level for debugging (0 = no output, 1 = minimal, 2 = detailed)
    setenv("GRB_OUTPUTFLAG", "1", 1);
    
    // Set node limit to prevent excessive memory usage in B&B tree
    setenv("GRB_NODELIMIT", "100000", 1);
    
    LOG_TRACE_WITH_INFO("Configured Gurobi environment: MIPFocus=1, Heuristics=%s, Threads=%d", 
                       config.gurobiEnableHeuristics ? "0.1" : "0.0", config.gurobiThreads);
  }

  /**
   * @brief Main solver method for the memory optimization problem
   * @param input Input parameters for the optimization
   * @param verbose Whether to print detailed debugging information
   * @return The optimized memory management strategy
   * 
   * This function sets up and solves the Mixed Integer Programming (MIP) problem
   * that finds the optimal memory management strategy to minimize peak memory usage.
   */
  SecondStepSolver::Output solve(SecondStepSolver::Input &&input, bool verbose = false) {
    // Take ownership of the input parameters
    this->input = std::move(input);

    // Configure Gurobi environment variables for advanced optimization
    configureGurobiEnvironment();

    // Initialize the OR-Tools solver
    initialize();

    // Process input data and calculate baseline metrics
    /*
    * 1. Calculates problem dimensions
    * 2. Computes memory usage metrics (original peak memory, theoretical minimum)
    * 3. Sets up arrays for tracking memory allocations and deallocations (Not implemented)
    */
    preprocessSecondStepInput();

    // Debug output: print task durations/runtime and array sizes/memory consumptions
    printDurationsAndSizes();

    // Set up the MIP problem components
    
    // Define the core decision variables (prefetch/offload operations)
    /*
    * 1. Which arrays should be initially allocated on the device
    * 2. When to prefetch arrays from host to device
    * 3. When to offload arrays from device to host
    */
    defineDecisionVariables();


    // Define variables that track memory state during execution
    /*
    * 1. x[i][j]: Whether array j is allocated on the device during task group i
      * x[i][j] = initiallyAllocatedOnDevice[j] + sum(p[u][j] for 0 ≤ u ≤ i) 
      *           - sum(o[u][j][v] for 0 ≤ u < i, u < v ≤ i)
      * OR 
      * x[i][j] = initiallyAllocatedOnDevice[j] + sum(prefetches) - sum(offloads)
    * 2. y[i][j]: Whether array j is available for task group i to use
      *  y[i][j] = initiallyAllocatedOnDevice[j] + sum(p[u][j] for 0 ≤ u ≤ i) 
      *            - sum(o[u][j][v] for any u < i, any v)
      * CONSTRAINT:
      *  y[i][j] ≤ 1 - sum(o[i][j][k] for all k > i)
      *           , which means y[i][j] and sum(o[i][j][k]) cannot both be 1
    */
    defineXAndY();

    // Define execution times for tasks and data movements
    defineWeights();

    // Define the execution graph structure (precedence constraints)
    /* 
   * The execution graph consists of several types of vertices:
   * - Task group start vertices: Synchronization points before task execution
   * - Task group execution vertices: Actual computation tasks
   * - Prefetch vertices: Memory transfers from host to device
   * - Offload vertices: Memory transfers from device to host * 
   * Edges in this graph (e[i][j] = 1) indicate that vertex i must complete before vertex j.
   */
    addEdges();

    // Add constraints for computing the execution schedule
    // Create constraint: z[TaskGroudEnd(last)] ≤ maxAcceptableTime
    addLongestPathConstraints();

    // Add constraints requiring task data dependencies to be satisfied
    addKernelDataDependencyConstraints();

    // Define constraints and variables for peak memory tracking
    // peakMemoryUsage - sum_j(x[i][j] * arraySizes[j]) ≥ 0, for every i
    definePeakMemoryUsage();

    // Define constraints and variables for tracking total data movements
    defineNumberOfDataMovements();

    // Configure the objective function with unified normalization
    auto objective = solver->MutableObjective();
    
    // All components are normalized to similar scales for balanced optimization:
    // - Memory: normalized by original peak memory (range ~[0.2, 1.0])
    // - Runtime: normalized by original runtime × acceptable factor (range ~[0.8, 2.0])
    // - Migrations: normalized by theoretical maximum (range ~[0.0, 0.1])
    
    LOG_TRACE_WITH_INFO("Objective function normalization factors:");
    LOG_TRACE_WITH_INFO("  Original peak memory: %.2f MiB", originalPeakMemoryUsage);
    LOG_TRACE_WITH_INFO("  Original runtime: %.2f seconds", originalTotalRunningTime);
    LOG_TRACE_WITH_INFO("  Max possible migrations: %d", numberOfArraysManaged * numberOfTaskGroups);
    
    if (ConfigurationManager::getConfig().optimization.weightOfPeakMemoryUsage > std::numeric_limits<double>::epsilon()) {
      // Minimize peak memory usage normalized by original peak memory
      // Normalization: memory_usage / original_peak_memory_usage
      // This gives us a percentage of original memory usage
      double memoryNormalizationFactor = 1.0 / originalPeakMemoryUsage;
      objective->SetCoefficient(peakMemoryUsage, 
                               memoryNormalizationFactor * 
                               ConfigurationManager::getConfig().optimization.weightOfPeakMemoryUsage);
    }
    
    if (ConfigurationManager::getConfig().optimization.weightOfNumberOfMigrations > std::numeric_limits<double>::epsilon()) {
      // Minimize number of data transfers normalized by theoretical maximum
      // Normalization: migrations / (num_arrays × num_tasks)
      // This gives us the migration density as a fraction
      double maxPossibleMigrations = numberOfArraysManaged * numberOfTaskGroups;
      double migrationNormalizationFactor = 1.0 / maxPossibleMigrations;
      objective->SetCoefficient(numberOfDataMovements, 
                               migrationNormalizationFactor *
                               ConfigurationManager::getConfig().optimization.weightOfNumberOfMigrations);
    }
    
    if (ConfigurationManager::getConfig().optimization.weightOfTotalRunningTime > std::numeric_limits<double>::epsilon()) {
      // Minimize total execution time normalized by acceptable runtime
      // Normalization: runtime / (original_runtime × acceptable_factor)
      // Values around 1.0 mean hitting the acceptable slowdown target
      double acceptableFactor = 1.0 + ConfigurationManager::getConfig().optimization.acceptableRunningTimeFactor;
      double runtimeNormalizationFactor = 1.0 / (originalTotalRunningTime * acceptableFactor);
      objective->SetCoefficient(
        z[getTaskGroupVertexIndex(numberOfTaskGroups - 1)],
        runtimeNormalizationFactor *
        ConfigurationManager::getConfig().optimization.weightOfTotalRunningTime
      );
    }
    
    // Set the problem type to minimization
    objective->SetMinimization();

    // Configure solver time limit from configuration (in seconds)
    int timeLimitMs = ConfigurationManager::getConfig().optimization.gurobiTimeLimitSeconds * 1000;
    solver->set_time_limit(timeLimitMs);
    
    // Configure advanced Gurobi parameters
    MPSolverParameters solverParam;
    
    // Set MIP gap tolerance (how close to optimal solution is acceptable)
    double mipGap = ConfigurationManager::getConfig().optimization.gurobiMipGap;
    solverParam.SetDoubleParam(MPSolverParameters::RELATIVE_MIP_GAP, mipGap);
    
    // Set number of threads (0 = auto, use all available)
    int threads = ConfigurationManager::getConfig().optimization.gurobiThreads;
    if (threads > 0) {
      // Note: Thread parameter might need to be set via Gurobi-specific interface
      // For now, document this limitation
    }
    
    // Enable/disable heuristics for faster feasible solution finding
    if (ConfigurationManager::getConfig().optimization.gurobiEnableHeuristics) {
      // Enable aggressive heuristics for faster feasible solution finding
      // Note: These require Gurobi-specific API access beyond OR-Tools
      // For now, rely on the MIP gap tolerance to accept suboptimal solutions
      
      // Alternative: Set solver focus to finding feasible solutions quickly
      // This is a Gurobi-specific parameter that might not be available in OR-Tools
      // solverParam.SetIntegerParam(MPSolverParameters::MIP_STRATEGY, 1); // Focus on feasibility
    }

    // Measure solver execution time
    SystemWallClock clock;
    clock.start();
    auto resultStatus = solver->Solve(solverParam);
    clock.end();

    LOG_TRACE_WITH_INFO("Time for solving the MIP problem (seconds): %.2f", clock.getTimeInSeconds());

    // Process solution results
    SecondStepSolver::Output output;

    if (resultStatus == MPSolver::OPTIMAL || resultStatus == MPSolver::FEASIBLE) {
      // Print detailed solution for debugging
      printSolution(resultStatus);

      // Solution found - mark as optimal
      output.optimal = true;
      
      // Copy memory usage information
      output.originalMemoryUsage = originalPeakMemoryUsage;
      output.anticipatedPeakMemoryUsage = peakMemoryUsage->solution_value();

      // Extract arrays that should be initially on the device
      for (int i = 0; i < numberOfArraysManaged; i++) {
        if (initiallyAllocatedOnDevice[i]->solution_value() > 0) {
          output.indicesOfArraysInitiallyOnDevice.push_back(i);
        }
      }

      // Extract prefetch operations (host-to-device transfers)
      for (int i = 0; i < numberOfTaskGroups; i++) {
        for (int j = 0; j < numberOfArraysManaged; j++) {
          if (p[i][j]->solution_value() > 0) {
            output.prefetches.push_back(std::make_tuple(i, j));
          }
        }
      }

      // Extract offload operations (device-to-host transfers)
      for (int i = 0; i < numberOfTaskGroups; i++) {
        for (int j = 0; j < numberOfArraysManaged; j++) {
          for (int k = 0; k < numberOfTaskGroups; k++) {
            if (o[i][j][k]->solution_value() > 0) {
              output.offloadings.push_back(std::make_tuple(i, j, k));
            }
          }
        }
      }
    } else {
      // No feasible solution found
      LOG_TRACE_WITH_INFO("No optimal solution found. (ResultStatus=%d)", resultStatus);
      output.optimal = false;
    }

    return output;
  }
};

/**
 * @brief Public solve method for the SecondStepSolver
 * @param input Input parameters with task graph and memory requirements
 * @return Output The optimized memory management strategy
 * 
 * This public method creates the internal IntegerProgrammingSolver and delegates
 * the optimization work to it. This separation is necessary because NVCC (CUDA compiler)
 * cannot directly compile OR-Tools code.
 */
SecondStepSolver::Output SecondStepSolver::solve(SecondStepSolver::Input &&input) {
  LOG_TRACE();  // Log entry point for tracing/profiling

  // Create the internal solver that handles the mixed integer programming
  IntegerProgrammingSolver solver;
  
  // Delegate to the MIP solver and return its results
  return solver.solve(std::move(input));
}

}  // namespace memopt
