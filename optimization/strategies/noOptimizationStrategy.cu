#include "../../profiling/memoryManager.hpp"
#include "../../utilities/logger.hpp"
#include "../../utilities/types.hpp"
#include "strategies.hpp"
#include "strategyUtilities.hpp"

namespace memopt {

/**
 * @brief Creates an execution plan with all data on device and no memory optimization
 *
 * This implementation serves as a baseline strategy that:
 * 1. Places all data on the GPU at the beginning of execution
 * 2. Does not generate any data movement operations (prefetch/offload)
 * 3. Preserves the original task dependencies without reordering
 *
 * This approach works for applications where all data fits in GPU memory
 * but will fail with out-of-memory errors for larger workloads.
 *
 * @param input Structure containing task groups and dependencies
 * @return OptimizationOutput A simple execution plan with all data on device
 */
OptimizationOutput NoOptimizationStrategy::run(OptimizationInput &input) {
  LOG_TRACE();

  // Print input for debugging purposes
  printOptimizationInput(input);

  OptimizationOutput output;
  output.optimal = true;  // Mark the solution as feasible (though unoptimized)

  //---------- MEMORY ALLOCATION STRATEGY ----------
  
  // Place all managed data on the device initially
  // No data movement operations will be generated - everything stays on GPU
  auto& memManager = MemoryManager::getInstance();
  auto& addresses = memManager.getEditableManagedAddresses();
  for (int i = 0; i < addresses.size(); i++) {
    output.arraysInitiallyAllocatedOnDevice.push_back(i);
  }

  //---------- TASK GRAPH CONSTRUCTION ----------
  
  // Create start and end nodes for each task group
  // These act as entry and exit points for task execution flow
  std::vector<int> taskGroupStartNodes, taskGroupEndNodes;
  for (int i = 0; i < input.nodes.size(); i++) {
    taskGroupStartNodes.push_back(output.addEmptyNode());
    taskGroupEndNodes.push_back(output.addEmptyNode());
  }

  // Preserve dependencies between task groups
  // This ensures correct execution order is maintained
  for (const auto &[u, destinations] : input.edges) {
    for (auto v : destinations) {
      output.addEdge(taskGroupEndNodes[u], taskGroupStartNodes[v]);
    }
  }

  //---------- TASK NODE CONSTRUCTION ----------
  
  // Add individual task nodes and their dependencies within each task group
  for (int i = 0; i < input.nodes.size(); i++) {
    auto &taskGroup = input.nodes[i];

    // Maps to track task nodes and their connections
    std::map<TaskId, int> taskIdToOutputNodeId;
    std::map<TaskId, bool> taskHasIncomingEdgeMap;
    
    // Create a node for each task in this task group
    for (TaskId u : taskGroup.nodes) {
      taskIdToOutputNodeId[u] = output.addTaskNode(u);
    }

    // Add edges between tasks based on dependencies
    for (const auto &[u, destinations] : taskGroup.edges) {
      for (auto v : destinations) {
        output.addEdge(taskIdToOutputNodeId[u], taskIdToOutputNodeId[v]);
        taskHasIncomingEdgeMap[v] = true;  // Mark this task as having dependencies
      }
    }

    // Connect task group entry point to tasks with no incoming edges
    // These are the tasks that can execute first in this group
    for (auto u : taskGroup.nodes) {
      if (!taskHasIncomingEdgeMap[u]) {
        output.addEdge(taskGroupStartNodes[i], taskIdToOutputNodeId[u]);
      }

      // Connect tasks with no outgoing edges to the task group exit point
      // These are the final tasks in this group
      if (taskGroup.edges[u].size() == 0) {
        output.addEdge(taskIdToOutputNodeId[u], taskGroupEndNodes[i]);
      }
    }
  }

  // Print the output plan for debugging purposes
  printOptimizationOutput(output, input.stageIndex);

  return output;
}

}  // namespace memopt
