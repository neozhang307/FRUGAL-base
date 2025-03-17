#pragma once

#include <set>
#include <tuple>
#include <vector>

#include "../../utilities/types.hpp"

namespace memopt {

/**
 * @class SecondStepSolver
 * @brief Determines the optimal memory management strategy for CUDA tasks
 * 
 * This solver is the second phase of the TwoStepOptimizationStrategy. After the
 * FirstStepSolver optimizes task execution order, this solver determines:
 * 1. Which arrays should initially reside on the device
 * 2. When to prefetch arrays from host to device before they're needed
 * 3. When to offload arrays from device to host when they're no longer needed
 * 
 * It uses a mixed integer programming (MIP) approach via Google's OR-Tools
 * to minimize peak GPU memory usage while maintaining acceptable performance.
 */
class SecondStepSolver {
 public:
  /**
   * @struct Input
   * @brief Input parameters for the SecondStepSolver
   * 
   * Contains all the information needed to determine optimal memory management:
   * execution times, memory requirements, data dependencies, and bandwidth constraints
   */
  struct Input {
    std::vector<float> taskGroupRunningTimes;      // Execution time for each task group (in seconds)

    std::vector<size_t> arraySizes;                // Size in bytes of each memory array
    std::set<ArrayId> applicationInputArrays;      // Arrays that are inputs to the entire application
    std::set<ArrayId> applicationOutputArrays;     // Arrays that are outputs of the entire application
    std::vector<std::set<ArrayId>> taskGroupInputArrays;   // Arrays read by each task group
    std::vector<std::set<ArrayId>> taskGroupOutputArrays;  // Arrays written by each task group

    float prefetchingBandwidth;                    // Host-to-device transfer bandwidth (bytes/second)
    float offloadingBandwidth;                     // Device-to-host transfer bandwidth (bytes/second)

    float originalTotalRunningTime;                // Original execution time without optimization

    bool forceAllArraysToResideOnHostInitiallyAndFinally; // Whether arrays must start/end on host
    int stageIndex;                                // For debug output naming
  };

  /**
   * @struct Output
   * @brief Output data from the SecondStepSolver
   * 
   * Contains the complete memory management strategy to minimize peak memory usage
   */
  struct Output {
    // (Task Group Index, Array Index)
    typedef std::tuple<TaskGroupId, ArrayId> Prefetch;  // When to prefetch an array to device

    // (Starting Task Group Index, Array Index, Ending Task Group Index)
    typedef std::tuple<TaskGroupId, ArrayId, TaskGroupId> Offload;  // When to offload array to host

    bool optimal;                                  // Whether a feasible solution was found
    std::vector<ArrayId> indicesOfArraysInitiallyOnDevice;  // Arrays that start on the device
    std::vector<Prefetch> prefetches;              // List of prefetch operations to schedule
    std::vector<Offload> offloadings;              // List of offload operations to schedule
  };

  /**
   * @brief Solves the memory management optimization problem
   * @param input The optimization parameters (moved for efficiency)
   * @return Output The complete memory management strategy
   * 
   * Uses mixed integer programming to solve for the optimal placement and
   * movement of memory arrays to minimize peak GPU memory usage while
   * respecting performance constraints.
   */
  Output solve(Input &&input);
};

}  // namespace memopt
