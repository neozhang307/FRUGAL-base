#pragma once

#include <map>
#include <string>
#include <vector>

#include "optimizationInput.hpp"
#include "optimizationOutput.hpp"

namespace memopt {

/**
 * @brief Output task graph with execution times, dependencies, and array information in DOT format
 * 
 * This function generates a Graphviz DOT file that visualizes:
 * 1. Task groups as rectangular nodes showing execution time
 * 2. Memory arrays as elliptical nodes showing size
 * 3. Data flow edges (array -> task for inputs, task -> array for outputs)
 * 4. Task dependency edges between task groups
 * 
 * @param optimizationInput The optimization input containing task groups and dependencies
 * @param outputPath Path to write the DOT file
 */
void writeTaskGraphToDot(const OptimizationInput &optimizationInput, const std::string &outputPath);

class Optimizer {
 public:
  static Optimizer *getInstance();
  Optimizer(Optimizer &other) = delete;
  void operator=(const Optimizer &) = delete;

  // Warning: the graph is executed once during profiling.
  OptimizationOutput profileAndOptimize(cudaGraph_t originalGraph);

  /**
   * @brief Profile a CUDA graph to extract optimization input data
   *
   * This method performs only the profiling phase, extracting task groups,
   * dependencies, and timing information without running optimization.
   * The resulting OptimizationInput can be saved for offline optimization.
   *
   * @param originalGraph The CUDA graph to profile
   * @return OptimizationInput Profiling data suitable for optimization
   */
  OptimizationInput profileGraph(cudaGraph_t originalGraph);

  /**
   * @brief Optimize a profiled graph to generate execution plan
   *
   * This method performs only the optimization phase, taking profiling data
   * and generating an optimized execution plan with memory management.
   *
   * @param optimizationInput The profiling data to optimize
   * @return OptimizationOutput The optimized execution plan
   */
  OptimizationOutput optimizeGraph(const OptimizationInput& optimizationInput);

 protected:
  Optimizer() = default;
  static Optimizer *instance;

 private:
  template <typename Strategy>
  OptimizationOutput optimize(OptimizationInput &optimizationInput) {
    Strategy strategyInstance;
    return strategyInstance.run(optimizationInput);
  }
};

}  // namespace memopt
