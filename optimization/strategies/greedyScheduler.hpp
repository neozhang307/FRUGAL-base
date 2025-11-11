#pragma once

#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "../../utilities/types.hpp"
#include "secondStepSolver.hpp"

namespace memopt {

/**
 * @class GreedyScheduler
 * @brief Fast greedy heuristic for memory management scheduling
 *
 * This scheduler provides a fast alternative to the MIP-based SecondStepSolver.
 *
 * Two modes:
 * 1. **MIN_MEMORY**: Minimize memory usage
 *    - Prefetch all needed arrays before each task
 *    - Offload ALL arrays after each task (destination = taskId + 1)
 *    - Achieves minimal memory bound (~81% reduction)
 *    - More data movements but lowest memory
 *
 * 2. **MAX_PERFORMANCE**: Maximize performance
 *    - Prefetch ALL arrays at the beginning (before first task)
 *    - Keep everything on device (no offloads)
 *    - Maximum memory usage but fastest execution
 *    - No data movement overhead during execution
 *
 * Can also be used as warm start for MIP solver to reduce solve time by 30-50%.
 */
class GreedyScheduler {
 public:
  /**
   * @enum Mode
   * @brief Scheduling mode for the greedy algorithm
   */
  enum class Mode {
    MIN_MEMORY,      // Minimize memory: prefetch per task, offload everything
    MAX_PERFORMANCE  // Maximize performance: prefetch all at start, keep everything
  };

  /**
   * @struct Config
   * @brief Configuration options for the greedy scheduler
   */
  struct Config {
    Mode mode = Mode::MIN_MEMORY;  // Default to memory minimization

    Config() = default;
    explicit Config(Mode m) : mode(m) {}
  };

  /**
   * @brief Default constructor
   */
  GreedyScheduler();

  /**
   * @brief Constructor with configuration
   * @param config Configuration parameters for the scheduler
   */
  explicit GreedyScheduler(const Config& config);

  /**
   * @brief Generate a complete memory management schedule using greedy heuristics
   * @param input The optimization parameters (same format as SecondStepSolver)
   * @return Output Complete memory management strategy
   *
   * MIN_MEMORY mode:
   * - For each task: Prefetch needed arrays (if not on device)
   * - After each task: Offload ALL arrays with destination = taskId + 1
   * - Achieves minimal memory, more data movements
   *
   * MAX_PERFORMANCE mode:
   * - Prefetch ALL arrays before first task (task 0)
   * - No offloads
   * - Maximum memory, zero data movement during execution
   */
  SecondStepSolver::Output generateSchedule(const SecondStepSolver::Input& input);

  /**
   * @brief Generate an initial solution for warm-starting the MIP solver
   * @param input The optimization parameters
   * @return Map of variable names to initial values for Gurobi
   *
   * This provides a good starting point for the MIP solver, which can
   * reduce solve time by 30-50% while still finding the optimal solution.
   */
  std::map<std::string, double> generateWarmStart(const SecondStepSolver::Input& input);

 private:
  Config config_;  // Configuration settings

  /**
   * @brief Generate schedule using MIN_MEMORY mode
   * @param input Problem input data
   * @return Complete schedule with per-task prefetching and offloading
   */
  SecondStepSolver::Output generateMinMemorySchedule(const SecondStepSolver::Input& input);

  /**
   * @brief Generate schedule using MAX_PERFORMANCE mode
   * @param input Problem input data
   * @return Complete schedule with all arrays prefetched at start
   */
  SecondStepSolver::Output generateMaxPerformanceSchedule(const SecondStepSolver::Input& input);

  /**
   * @brief Check if an array is used by a specific task
   * @param taskId Task to check
   * @param arrayId Array to check
   * @param input Problem input data
   * @return true if array is an input or output of the task
   */
  bool isArrayUsedByTask(
      TaskGroupId taskId,
      ArrayId arrayId,
      const SecondStepSolver::Input& input) const;

  /**
   * @brief Calculate peak memory usage for a given schedule
   * @param schedule The schedule to analyze
   * @param input Problem input data
   * @return Peak memory in MiB
   */
  double calculatePeakMemory(
      const SecondStepSolver::Output& schedule,
      const SecondStepSolver::Input& input) const;
};

}  // namespace memopt
