#include <algorithm>
#include <iterator>
#include <numeric>
#include <queue>
#include <utility>

#include "../../profiling/memoryManager.hpp"
#include "../../utilities/configurationManager.hpp"
#include "../../utilities/logger.hpp"
#include "../../utilities/types.hpp"
#include "firstStepSolver.hpp"
#include "secondStepSolver.hpp"
#include "strategies.hpp"
#include "strategyUtilities.hpp"

namespace memopt {

/**
 * @brief Converts OptimizationInput to FirstStepSolver::Input format
 *
 * This function prepares the input for the first optimization step by:
 * 1. Extracting the dependency graph structure between task groups
 * 2. Calculating the memory overlap between each pair of task groups
 *
 * The memory overlap calculation is critical for the first step, as it determines
 * which task groups should execute consecutively to maximize data reuse and
 * minimize memory transfers. The overlap is calculated as the total size of
 * memory buffers that are accessed by both task groups.
 *
 * @param optimizationInput The original optimization input structure
 * @return FirstStepSolver::Input Prepared input for the first optimization step
 */
FirstStepSolver::Input convertToFirstStepInput(OptimizationInput &optimizationInput) {
  // Get the total number of task groups we need to process
  const auto numTaskGroups = optimizationInput.nodes.size();

  // Create the output structure that FirstStepSolver will use
  FirstStepSolver::Input firstStepInput;

  // Initialize the data structures with appropriate sizes
  firstStepInput.n = numTaskGroups;                                                // Number of task groups
  firstStepInput.edges.resize(numTaskGroups);                                      // Array for dependency edges
  firstStepInput.dataDependencyOverlapInBytes.resize(numTaskGroups,                // Create a numTaskGroups x numTaskGroups matrix
                                               std::vector<size_t>(numTaskGroups, 0)); // Initialized to all zeros

  // Copy the dependency edges between task groups
  // These edges represent execution ordering constraints (A must complete before B)
  // For example, if optimizationInput.edges[2] contains {5, 7}, it means
  // task group 2 must execute before task groups 5 and 7
  for (int i = 0; i < numTaskGroups; i++) {
    firstStepInput.edges[i] = optimizationInput.edges[i];  // Direct copy of dependency information
  }

  // Calculate data dependency overlaps between all pairs of task groups
  // This identifies how much memory is shared between different task groups
  // We'll examine each task group (i) and compare it with all other task groups (j)
  for (int i = 0; i < numTaskGroups; i++) {
    // Get information about the current task group (u)
    const auto &u = optimizationInput.nodes[i];
    
    // Combine all memory accessed by this task group (both inputs and outputs)
    // The goal is to create a single set containing all memory addresses this task touches
    std::set<void *> uTotalDataDependency;  // This will hold all memory addresses accessed by task i
    
    // std::set_union combines two sorted ranges into one set without duplicates
    // Here we're combining:
    // 1. Memory addresses that this task reads from (inputs)
    // 2. Memory addresses that this task writes to (outputs)
    std::set_union(
      u.dataDependency.inputs.begin(),        // Start of input addresses for task i
      u.dataDependency.inputs.end(),          // End of input addresses for task i
      u.dataDependency.outputs.begin(),       // Start of output addresses for task i
      u.dataDependency.outputs.end(),         // End of output addresses for task i
      std::inserter(uTotalDataDependency, uTotalDataDependency.begin())  // Where to store the combined result
      // std::inserter creates an iterator that inserts elements into the uTotalDataDependency set
    );

    // Now compare the current task group (i) with each subsequent task group (j)
    // We only need to check pairs once due to symmetry (i,j is same as j,i),
    // so we start j at i+1 to avoid redundant calculations
    for (int j = i + 1; j < numTaskGroups; j++) {
      // Get information about the comparison task group (v)
      const auto &v = optimizationInput.nodes[j];

      // Similar to above, combine all memory accessed by the second task group
      std::set<void *> vTotalDataDependency;  // This will hold all memory addresses accessed by task j
      std::set_union(
        v.dataDependency.inputs.begin(),      // Start of input addresses for task j
        v.dataDependency.inputs.end(),        // End of input addresses for task j
        v.dataDependency.outputs.begin(),     // Start of output addresses for task j
        v.dataDependency.outputs.end(),       // End of output addresses for task j
        std::inserter(vTotalDataDependency, vTotalDataDependency.begin())  // Where to store the combined result
      );

      // Find memory buffers accessed by both task groups (the intersection)
      // std::set_intersection finds elements that exist in BOTH sets
      std::set<void *> dataDependencyIntersection;  // Will hold addresses that both tasks access
      std::set_intersection(
        uTotalDataDependency.begin(),         // Start of all addresses accessed by task i
        uTotalDataDependency.end(),           // End of all addresses accessed by task i
        vTotalDataDependency.begin(),         // Start of all addresses accessed by task j
        vTotalDataDependency.end(),           // End of all addresses accessed by task j
        std::inserter(dataDependencyIntersection, dataDependencyIntersection.begin())
        // Insert results into the intersection set
      );

      // Now that we have the set of memory addresses that both tasks access,
      // we need to calculate the total size of this shared memory in bytes
      
      // Calculate the total size of overlapping memory in bytes using std::accumulate
      // std::accumulate performs a cumulative operation (here: sum) on a sequence
      size_t totalOverlap = std::accumulate(
        dataDependencyIntersection.begin(),   // Start iterating through shared memory addresses
        dataDependencyIntersection.end(),     // End of shared memory addresses
        static_cast<size_t>(0),               // Initial value for the sum (0 bytes)
        [](size_t runningSum, void *memoryAddress) {
          // For each memory address in the intersection:
          // 1. Look up its size in bytes from the global MemoryManager
          // 2. Add this size to our running sum
          return runningSum + MemoryManager::getInstance().getSize(memoryAddress);
        }
      );
      
      // Store this value for both directions (i->j and j->i) since overlap is symmetric
      // If task i and j share 1000 bytes of memory, this relationship is bidirectional
      firstStepInput.dataDependencyOverlapInBytes[i][j] = totalOverlap;
      firstStepInput.dataDependencyOverlapInBytes[j][i] = totalOverlap;
    }
  }

  // Preserve the stage index for logging/debugging
  firstStepInput.stageIndex = optimizationInput.stageIndex;

  return firstStepInput;
}

/**
 * @brief Converts optimization data to SecondStepSolver::Input format
 *
 * This function prepares input for the second optimization step by:
 * 1. Incorporating the optimized task execution order from the first step
 * 2. Mapping memory pointers to array indices for the solver
 * 3. Providing task execution times and memory access patterns
 * 4. Adding bandwidth and memory configuration parameters
 *
 * The second step focuses on memory management decisions:
 * - Which arrays should be initially on the device
 * - When to prefetch arrays from host to device
 * - When to offload arrays from device to host
 *
 * @param optimizationInput The original optimization input structure
 * @param firstStepOutput The output from the first optimization step (task order)
 * @return SecondStepSolver::Input Prepared input for the second optimization step
 */
SecondStepSolver::Input convertToSecondStepInput(OptimizationInput &optimizationInput, FirstStepSolver::Output &firstStepOutput) {
  const int numberOfTaskGroups = firstStepOutput.taskGroupExecutionOrder.size();
  // The last task group is a sentinel representing the end of execution
  const int sentinelTaskGroupIndex = numberOfTaskGroups - 1;

  SecondStepSolver::Input secondStepInput;
  
  // Configure data transfer bandwidths from system configuration
  secondStepInput.prefetchingBandwidth = ConfigurationManager::getConfig().optimization.prefetchingBandwidthInGB * 1e9;
  secondStepInput.offloadingBandwidth = ConfigurationManager::getConfig().optimization.prefetchingBandwidthInGB * 1e9;
  
  // Copy execution parameters from the original input
  secondStepInput.originalTotalRunningTime = optimizationInput.originalTotalRunningTime;
  secondStepInput.forceAllArraysToResideOnHostInitiallyAndFinally = optimizationInput.forceAllArraysToResideOnHostInitiallyAndFinally;
  secondStepInput.stageIndex = optimizationInput.stageIndex;

  // Initialize arrays for task group characteristics
  secondStepInput.taskGroupRunningTimes.resize(numberOfTaskGroups);
  secondStepInput.taskGroupInputArrays.resize(numberOfTaskGroups);
  secondStepInput.taskGroupOutputArrays.resize(numberOfTaskGroups);

  // Process each task group in the execution order from first step
  for (int i = 0; i < numberOfTaskGroups; i++) {
    // Skip the sentinel task group (it doesn't represent real computation)
    if (i == sentinelTaskGroupIndex) continue;

    // Get the actual task group based on the ordering from first step
    auto &node = optimizationInput.nodes[firstStepOutput.taskGroupExecutionOrder[i]];

    // Record the execution time of this task group
    secondStepInput.taskGroupRunningTimes[i] = node.runningTime;

    // Convert memory addresses to array indices for the solver
    // Record which arrays are inputs to this task group
    for (auto arrayAddress : node.dataDependency.inputs) {
      secondStepInput.taskGroupInputArrays[i].insert(MemoryManager::getInstance().getArrayId(arrayAddress));
    }
    // Record which arrays are outputs from this task group
    for (auto arrayAddress : node.dataDependency.outputs) {
      secondStepInput.taskGroupOutputArrays[i].insert(MemoryManager::getInstance().getArrayId(arrayAddress));
    }
  }

  // The sentinel task group has zero execution time
  secondStepInput.taskGroupRunningTimes[sentinelTaskGroupIndex] = 0;

  // Provide information about all memory arrays in the application
  secondStepInput.arraySizes.resize(MemoryManager::getInstance().getManagedAddresses().size());
  for (const auto &[ptr, index] : MemoryManager::getInstance().getAddressToIndexMap()) {
    secondStepInput.arraySizes[index] = MemoryManager::getInstance().getSize(ptr);
  }

  // Identify arrays that are application inputs (must be available at start)
  for (auto ptr : MemoryManager::getInstance().getApplicationInputs()) {
    secondStepInput.applicationInputArrays.insert(MemoryManager::getInstance().getArrayId(ptr));
  }
  // Identify arrays that are application outputs (must be available at end)
  for (auto ptr : MemoryManager::getInstance().getApplicationOutputs()) {
    secondStepInput.applicationOutputArrays.insert(MemoryManager::getInstance().getArrayId(ptr));
  }

  return secondStepInput;
}

float calculateLongestPath(OptimizationOutput &optimizedGraph, std::map<int, float> nodeWeight) {
  std::map<int, int> inDegree;
  for (auto u : optimizedGraph.nodes) {
    for (auto v : optimizedGraph.edges[u]) {
      inDegree[v]++;
    }
  }

  std::map<int, float> dis;

  std::queue<int> q;
  for (auto u : optimizedGraph.nodes) {
    if (inDegree[u] == 0) {
      dis[u] = nodeWeight[u];
      q.push(u);
    }
  }

  while (!q.empty()) {
    int u = q.front();
    q.pop();

    for (auto v : optimizedGraph.edges[u]) {
      dis[v] = std::max(dis[v], dis[u] + nodeWeight[v]);
      inDegree[v]--;
      if (inDegree[v] == 0) {
        q.push(v);
      }
    }
  }

  float longestPathLength = 0;
  for (auto u : optimizedGraph.nodes) {
    longestPathLength = std::max(longestPathLength, dis[u]);
  }
  return longestPathLength;
}

OptimizationOutput convertToOptimizationOutput(
  OptimizationInput &optimizationInput,
  FirstStepSolver::Output &firstStepOutput,
  SecondStepSolver::Output &secondStepOutput
) {
  const int numberOfTaskGroups = firstStepOutput.taskGroupExecutionOrder.size();
  const int sentinelTaskGroupIndex = numberOfTaskGroups - 1;

  OptimizationOutput optimizedGraph;

  if (!secondStepOutput.optimal) {
    optimizedGraph.optimal = false;
    return optimizedGraph;
  }

  optimizedGraph.optimal = true;

  // Add arrays that should be on device initially
  for (auto index : secondStepOutput.indicesOfArraysInitiallyOnDevice) {
    optimizedGraph.arraysInitiallyAllocatedOnDevice.push_back(index);
  }

  std::map<int, float> nodeWeight;

  // Add task groups
  std::vector<int> taskGroupStartNodes, taskGroupBodyNodes, taskGroupEndNodes;
  for (int i = 0; i < numberOfTaskGroups; i++) {
    taskGroupStartNodes.push_back(optimizedGraph.addEmptyNode());
    taskGroupBodyNodes.push_back(optimizedGraph.addEmptyNode());
    taskGroupEndNodes.push_back(optimizedGraph.addEmptyNode());

    nodeWeight[taskGroupStartNodes[i]] = 0;
    nodeWeight[taskGroupBodyNodes[i]] = i == sentinelTaskGroupIndex ? 0 : optimizationInput.nodes[i].runningTime;
    nodeWeight[taskGroupEndNodes[i]] = 0;
  }

  // Add edges between task groups
  for (int i = 1; i < numberOfTaskGroups; i++) {
    const auto previousTaskGroupId = firstStepOutput.taskGroupExecutionOrder[i - 1];
    const auto currentTaskGroupId = firstStepOutput.taskGroupExecutionOrder[i];
    optimizedGraph.addEdge(
      taskGroupEndNodes[previousTaskGroupId],
      taskGroupStartNodes[currentTaskGroupId]
    );
  }

  // Add nodes and edges inside task groups
  for (int i = 0; i < numberOfTaskGroups; i++) {
    if (i == sentinelTaskGroupIndex) continue;

    auto &taskGroup = optimizationInput.nodes[i];

    std::map<TaskId, int> taskIdToOutputNodeIdMap;
    std::map<TaskId, bool> taskHasIncomingEdgeMap;
    for (auto u : taskGroup.nodes) {
      taskIdToOutputNodeIdMap[u] = optimizedGraph.addTaskNode(u);
      nodeWeight[taskIdToOutputNodeIdMap[u]] = 0;
    }

    for (const auto &[u, destinations] : taskGroup.edges) {
      for (auto v : destinations) {
        optimizedGraph.addEdge(taskIdToOutputNodeIdMap[u], taskIdToOutputNodeIdMap[v]);
        taskHasIncomingEdgeMap[v] = true;
      }
    }

    optimizedGraph.addEdge(taskGroupStartNodes[i], taskGroupBodyNodes[i]);

    for (auto u : taskGroup.nodes) {
      if (!taskHasIncomingEdgeMap[u]) {
        optimizedGraph.addEdge(taskGroupBodyNodes[i], taskIdToOutputNodeIdMap[u]);
      }

      if (taskGroup.edges[u].size() == 0) {
        optimizedGraph.addEdge(taskIdToOutputNodeIdMap[u], taskGroupEndNodes[i]);
      }
    }
  }

  optimizedGraph.addEdge(taskGroupStartNodes[sentinelTaskGroupIndex], taskGroupBodyNodes[sentinelTaskGroupIndex]);
  optimizedGraph.addEdge(taskGroupBodyNodes[sentinelTaskGroupIndex], taskGroupEndNodes[sentinelTaskGroupIndex]);

  // Add prefetches
  for (const auto &[startingNodeIndex, arrayIndex] : secondStepOutput.prefetches) {
    auto& memManager = MemoryManager::getInstance();
    auto& addresses = memManager.getEditableManagedAddresses();
    void *arrayAddress = addresses[arrayIndex];
    size_t arraySize = memManager.getSize(arrayAddress);

    int endingNodeIndex = startingNodeIndex;
    while (endingNodeIndex < firstStepOutput.taskGroupExecutionOrder.size()) {
      auto &taskGroup = optimizationInput.nodes[firstStepOutput.taskGroupExecutionOrder[endingNodeIndex]];
      if (taskGroup.dataDependency.inputs.count(arrayAddress) > 0 || taskGroup.dataDependency.outputs.count(arrayAddress) > 0) {
        break;
      }
      endingNodeIndex++;
    }

    // Ignore unnecessary prefetch, which has no dependent kernels after it.
    if (endingNodeIndex == firstStepOutput.taskGroupExecutionOrder.size()) {
      continue;
    }

    auto dataMovementNode = optimizedGraph.addDataMovementNode(
      OptimizationOutput::DataMovement::Direction::hostToDevice,
      arrayIndex,
      taskGroupStartNodes[firstStepOutput.taskGroupExecutionOrder[startingNodeIndex]],
      taskGroupBodyNodes[firstStepOutput.taskGroupExecutionOrder[endingNodeIndex]]
    );

    nodeWeight[dataMovementNode] = static_cast<float>(memManager.getSize(arrayAddress))
                                   / (ConfigurationManager::getConfig().optimization.prefetchingBandwidthInGB * 1e9);
  }

  // Add offloadings
  for (const auto &[startingNodeIndex, arrayIndex, endingNodeIndex] : secondStepOutput.offloadings) {
    auto& memManager = MemoryManager::getInstance();
    auto& addresses = memManager.getEditableManagedAddresses();
    void *arrayAddress = addresses[arrayIndex];
    size_t arraySize = memManager.getSize(arrayAddress);

    auto dataMovementNode = optimizedGraph.addDataMovementNode(
      OptimizationOutput::DataMovement::Direction::deviceToHost,
      arrayIndex,
      taskGroupEndNodes[firstStepOutput.taskGroupExecutionOrder[startingNodeIndex]],
      taskGroupStartNodes[firstStepOutput.taskGroupExecutionOrder[endingNodeIndex]]
    );

    nodeWeight[dataMovementNode] = static_cast<float>(memManager.getSize(arrayAddress))
                                   / (ConfigurationManager::getConfig().optimization.prefetchingBandwidthInGB * 1e9);
  }

  LOG_TRACE_WITH_INFO("Longest path in optimized graph (s): %.6f", calculateLongestPath(optimizedGraph, nodeWeight));

  return optimizedGraph;
}

/**
 * @brief Executes the two-step memory optimization strategy
 * 
 * This function implements the core two-step optimization process:
 * 1. Optimizing task group execution order to maximize data reuse
 * 2. Scheduling memory movement operations based on the optimized order
 * 
 * The process transforms the raw task graph into an optimized execution plan
 * that intelligently manages GPU memory, potentially allowing applications
 * to execute even when their memory requirements exceed available GPU memory.
 * 
 * @param input The OptimizationInput structure with task groups and memory access patterns
 * @return OptimizationOutput A complete execution plan with added memory operations
 */
OptimizationOutput TwoStepOptimizationStrategy::run(OptimizationInput &input) {
  LOG_TRACE();

  // Print the input for debugging/visualization purposes
  printOptimizationInput(input);

  // STEP 1: Task Scheduling Optimization
  // Convert the optimization input to the format expected by FirstStepSolver
  auto firstStepInput = convertToFirstStepInput(input);
  // Create and run the first step solver to find optimal task group execution order
  FirstStepSolver firstStepSolver(std::move(firstStepInput));
  auto firstStepOutput = firstStepSolver.solve();

  // Add a sentinel task group at the end to represent program completion
  // This simplifies boundary condition handling in the second step
  firstStepOutput.taskGroupExecutionOrder.push_back(firstStepOutput.taskGroupExecutionOrder.size());

  // STEP 2: Memory Management Optimization
  // Convert the first step output and original input to the format for SecondStepSolver
  auto secondStepInput = convertToSecondStepInput(input, firstStepOutput);
  // Create and run the second step solver to determine memory movement operations
  SecondStepSolver secondStepSolver;
  auto secondStepOutput = secondStepSolver.solve(std::move(secondStepInput));

  // Combine the outputs from both steps into a unified execution plan
  // This includes task nodes and memory movement operations
  auto output = convertToOptimizationOutput(input, firstStepOutput, secondStepOutput);

  // Print the final optimized plan for debugging/visualization
  printOptimizationOutput(output, input.stageIndex);

  return output;
}

}  // namespace memopt
