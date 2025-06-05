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
