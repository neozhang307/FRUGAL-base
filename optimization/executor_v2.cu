#include <cassert>
#include <memory>
#include <queue>

#include "../profiling/memoryManager.hpp"
#include "../profiling/peakMemoryUsageProfiler.hpp"
#include "../utilities/configurationManager.hpp"
#include "../utilities/cudaGraphUtilities.hpp"
#include "../utilities/cudaUtilities.hpp"
#include "../utilities/utilities.hpp"
#include "../utilities/logger.hpp"
#include "executor.hpp"

namespace memopt {

/*
 * executeOptimizedGraph - Executes a CUDA graph with memory optimization
 *
 * This function executes a computation graph that has been optimized to reduce memory usage
 * by dynamically managing data transfers between the main GPU and storage (host memory or secondary GPU).
 * It enables processing of workloads larger than would fit in GPU memory alone.
 * 
 * This implementation uses MemoryManager's internal storage to track device-to-host mappings.
 *
 * Parameters:
 * - optimizedGraph: The optimized computation graph to execute
 * - executeRandomTask: Callback to execute specific computation tasks
 * - runningTime: Output parameter to store the execution time
 * - memManager: Reference to the MemoryManager instance to use
 */
/**
 * @brief Prepares the topological sort structures for graph traversal
 * 
 * Sets up the in-degree map and identifies root nodes for Kahn's algorithm
 *
 * @param optimizedGraph The optimization plan
 * @param inDegrees Output parameter for node in-degrees
 * @param nodesToExecute Output queue of nodes ready to execute
 * @param rootNodes Output vector of root nodes
 */
void Executor::prepareTopologicalSort(
  const OptimizationOutput &optimizedGraph,
  std::map<int, int> &inDegrees,
  std::queue<int> &nodesToExecute,
  std::vector<int> &rootNodes
) {
  // Calculate in-degrees for each node in the optimized graph
  for (auto &[u, outEdges] : optimizedGraph.edges) {
    for (auto &v : outEdges) {
      inDegrees[v] += 1;
    }
  }

  // Find root nodes (nodes with no dependencies)
  for (auto &u : optimizedGraph.nodes) {
    if (inDegrees[u] == 0) {
      nodesToExecute.push(u);
      rootNodes.push_back(u);
    }
  }
}

/**
 * @brief Handles a data movement node in the graph construction
 * 
 * Processes prefetch or offload operations as part of graph construction
 *
 * @param creator The graph creator object
 * @param optimizedGraph The optimization plan
 * @param nodeId The node ID to process
 * @param dependencies Dependencies for this node
 * @param memManager Reference to memory manager
 * @param stream CUDA stream to use
 * @return Vector of new leaf nodes after this operation
 */
std::vector<cudaGraphNode_t> Executor::handleDataMovementNode(
  OptimizedCudaGraphCreator *creator,
  const OptimizationOutput &optimizedGraph,
  int nodeId,
  const std::vector<cudaGraphNode_t> &dependencies,
  MemoryManager &memManager,
  cudaStream_t stream
) {
  // Begin capturing operations with dependencies
  creator->beginCaptureOperation(dependencies);
  
  // Get data movement details
  auto &dataMovement = optimizedGraph.nodeIdToDataMovementMap.at(nodeId);
  
  // Perform the appropriate data movement operation
  if (dataMovement.direction == OptimizationOutput::DataMovement::Direction::hostToDevice) {
    memManager.prefetchToDevice(dataMovement.arrayId, stream);
  } else {
    memManager.offloadFromDevice(dataMovement.arrayId, stream);
  }
  
  // Check for errors and end capture
  checkCudaErrors(cudaPeekAtLastError());
  auto newLeafNodes = creator->endCaptureOperation();
  checkCudaErrors(cudaPeekAtLastError());
  
  return newLeafNodes;
}

/**
 * @brief Handles a computation task node in the graph construction
 * 
 * Executes the specified task as part of graph construction
 *
 * @param creator The graph creator object
 * @param optimizedGraph The optimization plan
 * @param nodeId The node ID to process
 * @param dependencies Dependencies for this node
 * @param executeRandomTask Callback to execute the task
 * @param memManager Reference to memory manager
 * @param stream CUDA stream to use
 * @return Vector of new leaf nodes after this operation
 */
std::vector<cudaGraphNode_t> Executor::handleTaskNode(
  OptimizedCudaGraphCreator *creator,
  const OptimizationOutput &optimizedGraph,
  int nodeId,
  const std::vector<cudaGraphNode_t> &dependencies,
  ExecuteRandomTask executeRandomTask,
  MemoryManager &memManager,
  cudaStream_t stream
) {
  // Begin capturing operations with dependencies
  creator->beginCaptureOperation(dependencies);
  
  // Execute the task with current memory address mapping
  executeRandomTask(
    optimizedGraph.nodeIdToTaskIdMap.at(nodeId),
    memManager.getEditableCurrentAddressMap(),
    stream
  );
  
  return creator->endCaptureOperation();
}

/**
 * @brief Builds the optimized execution graph from the optimization plan
 * 
 * Constructs a CUDA graph by traversing the optimization plan in topological order
 *
 * @param optimizedGraph The optimization plan
 * @param executeRandomTask Callback to execute tasks
 * @param memManager Reference to memory manager
 * @param stream CUDA stream to use
 * @return The created CUDA graph
 */
cudaGraph_t Executor::buildOptimizedGraph(
  OptimizationOutput &optimizedGraph,
  ExecuteRandomTask executeRandomTask,
  MemoryManager &memManager,
  cudaStream_t stream
) {
  LOG_TRACE_WITH_INFO("Record nodes to a new CUDA Graph");
  
  // Create a new CUDA graph
  cudaGraph_t graph;
  checkCudaErrors(cudaGraphCreate(&graph, 0));
  
  // Initialize graph creator
  auto creator = std::make_unique<OptimizedCudaGraphCreator>(stream, graph);
  
  // Prepare for topological traversal
  std::map<int, int> inDegrees;
  std::queue<int> nodesToExecute;
  std::vector<int> rootNodes;
  prepareTopologicalSort(optimizedGraph, inDegrees, nodesToExecute, rootNodes);
  
  // Maps nodes to their dependencies in the CUDA graph
  std::map<int, std::vector<cudaGraphNode_t>> nodeDependencies;
  
  // Process nodes in topological order (Kahn's Algorithm)
  while (!nodesToExecute.empty()) {
    // Get the next node to process
    auto nodeId = nodesToExecute.front();
    nodesToExecute.pop();
    
    std::vector<cudaGraphNode_t> newLeafNodes;
    auto nodeType = optimizedGraph.nodeIdToNodeTypeMap[nodeId];
    
    // Process different node types
    switch (nodeType) {
      case OptimizationOutput::NodeType::dataMovement:
        newLeafNodes = handleDataMovementNode(
          creator.get(), optimizedGraph, nodeId, nodeDependencies[nodeId], memManager, stream);
        break;
        
      case OptimizationOutput::NodeType::task:
        newLeafNodes = handleTaskNode(
          creator.get(), optimizedGraph, nodeId, nodeDependencies[nodeId], 
          executeRandomTask, memManager, stream);
        break;
        
      case OptimizationOutput::NodeType::empty:
        newLeafNodes.push_back(
          creator->addEmptyNode(nodeDependencies[nodeId]));
        break;
        
      default:
        LOG_TRACE_WITH_INFO("Unsupported node type: %d", nodeType);
        exit(-1);
    }
    
    // Update dependencies and process nodes that have all dependencies satisfied
    for (auto &v : optimizedGraph.edges[nodeId]) {
      inDegrees[v]--;
      
      // Add dependencies for the next nodes
      nodeDependencies[v].insert(
        nodeDependencies[v].end(),
        newLeafNodes.begin(),
        newLeafNodes.end()
      );
      
      // If all dependencies are satisfied, add to the queue
      if (inDegrees[v] == 0) {
        nodesToExecute.push(v);
      }
    }
  }
  
  // Export graph for debugging/visualization
  LOG_TRACE_WITH_INFO("Printing the new CUDA Graph to newGraph.dot");
  checkCudaErrors(cudaGraphDebugDotPrint(graph, "newGraph.dot", 0));
  
  return graph;
}

/**
 * @brief Initializes memory setup before graph execution
 * 
 * Configures memory manager, moves data to storage, and prefetches initial data
 *
 * @param optimizedGraph The optimization plan
 * @param memManager Reference to memory manager
 * @param stream CUDA stream to use
 * @return Executable graph for initial data distribution
 */
cudaGraphExec_t Executor::initializeMemory(
  OptimizationOutput &optimizedGraph,
  MemoryManager &memManager,
  cudaStream_t stream
) {
  // Configure device and memory settings
  int mainDeviceId = ConfigurationManager::getConfig().execution.mainDeviceId;
  
  LOG_TRACE_WITH_INFO("Initialize managed data distribution");
  
  // Configure memory manager storage
  memManager.configureStorage(
    ConfigurationManager::getConfig().execution.mainDeviceId,
    ConfigurationManager::getConfig().execution.storageDeviceId,
    ConfigurationManager::getConfig().execution.useNvlink
  );
  
  // Move all managed data to storage (host or secondary GPU)
  memManager.moveAllManagedMemoryToStorage();
  
  // Switch back to main GPU
  checkCudaErrors(cudaSetDevice(mainDeviceId));
  checkCudaErrors(cudaDeviceSynchronize());
  
  // Clear current memory mappings
  memManager.clearCurrentMappings();
  
  // Create a subgraph for initial data prefetching
  checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  
  // Prefetch arrays needed at the start
  memManager.prefetchAllDataToDevice(
    optimizedGraph.arraysInitiallyAllocatedOnDevice,
    stream
  );
  
  // End capture and create graph
  cudaGraph_t graphForInitialDataDistribution;
  checkCudaErrors(cudaStreamEndCapture(stream, &graphForInitialDataDistribution));
  
  // Instantiate and execute the initial data distribution graph
  cudaGraphExec_t graphExec;
  checkCudaErrors(cudaGraphInstantiate(
    &graphExec, graphForInitialDataDistribution, nullptr, nullptr, 0));
  checkCudaErrors(cudaGraphLaunch(graphExec, stream));
  checkCudaErrors(cudaDeviceSynchronize());
  
  // Cleanup the capture graph but keep the executable
  checkCudaErrors(cudaGraphDestroy(graphForInitialDataDistribution));
  
  return graphExec;
}

/**
 * @brief Executes the optimized graph and measures performance
 * 
 * Runs the graph with performance measurement and memory usage profiling
 *
 * @param graph The optimized CUDA graph to execute
 * @param stream CUDA stream to use
 * @param runningTime Output parameter for execution time
 */
void Executor::executeGraph(
  cudaGraph_t graph,
  cudaStream_t stream,
  float &runningTime
) {
  LOG_TRACE_WITH_INFO("Execute the new CUDA Graph");
  
  // Set up profiling if requested
  PeakMemoryUsageProfiler peakMemoryUsageProfiler;
  CudaEventClock cudaEventClock;
  
  // Instantiate the graph for execution
  cudaGraphExec_t graphExec;
  checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  
  // Upload the graph to the device for faster execution
  checkCudaErrors(cudaGraphUpload(graphExec, stream));
  checkCudaErrors(cudaStreamSynchronize(stream));
  
  // Start memory usage profiling if requested
  if (ConfigurationManager::getConfig().execution.measurePeakMemoryUsage) {
    peakMemoryUsageProfiler.start();
  }
  
  // Execute and time the graph
  cudaEventClock.start();
  checkCudaErrors(cudaGraphLaunch(graphExec, stream));
  cudaEventClock.end();
  checkCudaErrors(cudaDeviceSynchronize());
  
  // Report peak memory usage if requested
  if (ConfigurationManager::getConfig().execution.measurePeakMemoryUsage) {
    const auto peakMemoryUsage = peakMemoryUsageProfiler.end();
    LOG_TRACE_WITH_INFO(
      "Peak memory usage (MiB): %.2f",
      static_cast<float>(peakMemoryUsage) / 1024.0 / 1024.0
    );
  }
  
  // Store the execution time
  runningTime = cudaEventClock.getTimeInSeconds();
  
  // Cleanup graph execution resources
  checkCudaErrors(cudaGraphExecDestroy(graphExec));
}

/**
 * @brief Main implementation of executeOptimizedGraph
 * 
 * Executes a computation graph optimized for memory usage
 *
 * @param optimizedGraph The optimization plan
 * @param executeRandomTask Callback to execute tasks
 * @param runningTime Output parameter for execution time
 * @param memManager Reference to memory manager
 */
void Executor::executeOptimizedGraph(
  OptimizationOutput &optimizedGraph,
  ExecuteRandomTask executeRandomTask,
  float &runningTime,
  MemoryManager &memManager
) {
  LOG_TRACE_WITH_INFO("Initialize");
  
  // Create CUDA stream
  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));
  
  // Initialize memory setup and get initial data graph
  cudaGraphExec_t initialDataGraphExec = initializeMemory(
    optimizedGraph, memManager, stream);
  
  // Build the optimized execution graph
  cudaGraph_t graph = buildOptimizedGraph(
    optimizedGraph, executeRandomTask, memManager, stream);
  
  // Execute the optimized graph
  executeGraph(graph, stream, runningTime);
  
  // Clean up resources
  LOG_TRACE_WITH_INFO("Clean up");
  
  // Copy remaining data back to storage
  memManager.moveRemainedManagedMemoryToStorage();
  
  // Clean up CUDA resources
  checkCudaErrors(cudaGraphExecDestroy(initialDataGraphExec));
  checkCudaErrors(cudaGraphDestroy(graph));
  checkCudaErrors(cudaStreamDestroy(stream));
  
  // Clean up memory manager storage
  memManager.cleanStorage();
}

/*
 * executeOptimizedGraph - Executes a CUDA graph with memory optimization
 * 
 * This version uses the singleton instance of MemoryManager.
 *
 * Parameters:
 * - optimizedGraph: The optimized computation graph to execute
 * - executeRandomTask: Callback to execute specific computation tasks
 * - runningTime: Output parameter to store the execution time
 */
void Executor::executeOptimizedGraph(
  OptimizationOutput &optimizedGraph,
  ExecuteRandomTask executeRandomTask,
  float &runningTime
) {
  // Call the implementation version that takes a MemoryManager instance
  executeOptimizedGraph(optimizedGraph, executeRandomTask, runningTime, MemoryManager::getInstance());
}

/*
 * executeOptimizedGraphRepeatedly - Executes a CUDA graph with memory optimization multiple times
 *
 * This function executes a computation graph that has been optimized to reduce memory usage
 * repeatedly until a termination condition is met. It enables benchmarking and iterative
 * workloads that process data larger than would fit in GPU memory alone.
 * 
 * Parameters:
 * - optimizedGraph: The optimized computation graph to execute
 * - executeRandomTask: Callback to execute specific computation tasks
 * - shouldContinue: Function that determines when to stop the execution loop
 * - numIterations: Output parameter that counts how many times the graph was executed
 * - runningTime: Output parameter to store the execution time
 * - memManager: Reference to the MemoryManager instance to use
 */
void Executor::executeOptimizedGraphRepeatedly(
  OptimizationOutput &optimizedGraph,
  ExecuteRandomTask executeRandomTask,
  ShouldContinue shouldContinue,
  int &numIterations,
  float &runningTime,
  MemoryManager &memManager
) {
  LOG_TRACE_WITH_INFO("Initialize");

  // Create CUDA resources
  cudaGraph_t graph;
  checkCudaErrors(cudaGraphCreate(&graph, 0));

  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));

  auto optimizedCudaGraphCreator = std::make_unique<OptimizedCudaGraphCreator>(stream, graph);

  //----------------------------------------------------------------------
  // STEP 1: Prepare the graph for topological traversal using Kahn's algorithm
  //----------------------------------------------------------------------
  
  // Calculate in-degrees for each node in the optimized graph
  std::map<int, int> inDegrees;
  for (auto &[u, outEdges] : optimizedGraph.edges) {
    for (auto &v : outEdges) {
      inDegrees[v] += 1;
    }
  }

  // Find root nodes (nodes with no dependencies)
  std::queue<int> nodesToExecute;
  std::vector<int> rootNodes;
  for (auto &u : optimizedGraph.nodes) {
    if (inDegrees[u] == 0) {
      nodesToExecute.push(u);
      rootNodes.push_back(u);
    }
  }

  //----------------------------------------------------------------------
  // STEP 2: Configure the device and memory settings
  //----------------------------------------------------------------------
  
  // Set up device configuration for data movement
  int mainDeviceId = ConfigurationManager::getConfig().execution.mainDeviceId;
  
  //----------------------------------------------------------------------
  // STEP 3: Initialize managed data distribution
  //----------------------------------------------------------------------
  
  LOG_TRACE_WITH_INFO("Initialize managed data distribution");

  // Configure MemoryManager storage parameters
  memManager.configureStorage(
    ConfigurationManager::getConfig().execution.mainDeviceId,
    ConfigurationManager::getConfig().execution.storageDeviceId,
    ConfigurationManager::getConfig().execution.useNvlink
  );
  
  // Move all managed data to storage (host or secondary GPU) using MemoryManager's internal storage
  memManager.moveAllManagedMemoryToStorage();
  
  // Switch back to main GPU
  checkCudaErrors(cudaSetDevice(mainDeviceId));
  checkCudaErrors(cudaDeviceSynchronize());

  // Start timing the graph creation process
  SystemWallClock clock;
  clock.start();

  LOG_TRACE_WITH_INFO("Record nodes to a new CUDA Graph");

  //----------------------------------------------------------------------
  // STEP 4: Build the optimized execution graph
  //----------------------------------------------------------------------
  
  // Track memory addresses that have been updated (device copies)
  memManager.clearCurrentMappings();
  
  // Initial data allocation is integrated with main graph capture
  // rather than as a separate step like in executeOptimizedGraph
  std::vector<cudaGraphNode_t> newLeafNodes;
  
  // Use MemoryManager's prefetching method with internal storage
  for (auto arrayId : optimizedGraph.arraysInitiallyAllocatedOnDevice) {
    optimizedCudaGraphCreator->beginCaptureOperation(newLeafNodes);
    memManager.prefetchToDevice(arrayId, stream);
    newLeafNodes = optimizedCudaGraphCreator->endCaptureOperation();
  }

  // Maps nodes to their dependencies in the CUDA graph
  std::map<int, std::vector<cudaGraphNode_t>> nodeToDependentNodesMap;

  // Set up dependencies for root nodes
  for (auto u : rootNodes) {
    nodeToDependentNodesMap[u] = newLeafNodes;
  }

  // Process nodes in topological order (Kahn's Algorithm)
  while (!nodesToExecute.empty()) {
    // Get the next node to process
    auto u = nodesToExecute.front();
    nodesToExecute.pop();

    newLeafNodes.clear();

    // Process different node types
    auto nodeType = optimizedGraph.nodeIdToNodeTypeMap[u];
    if (nodeType == OptimizationOutput::NodeType::dataMovement) {
      // Handle data movement nodes (prefetch or offload)
      optimizedCudaGraphCreator->beginCaptureOperation(nodeToDependentNodesMap[u]);
      auto &dataMovement = optimizedGraph.nodeIdToDataMovementMap[u];
      auto dataMovementAddress = memManager.getPointerByArrayId(dataMovement.arrayId);
      auto dataMovementSize = memManager.getSizeByArrayId(dataMovement.arrayId);
      
      if (dataMovement.direction == OptimizationOutput::DataMovement::Direction::hostToDevice) {
        memManager.prefetchToDevice(dataMovement.arrayId,stream);
      } else {
        memManager.offloadFromDevice(dataMovement.arrayId,stream);
      }
      checkCudaErrors(cudaPeekAtLastError());
      newLeafNodes = optimizedCudaGraphCreator->endCaptureOperation();
      checkCudaErrors(cudaPeekAtLastError());
    } else if (nodeType == OptimizationOutput::NodeType::task) {
      // Handle computation task nodes
      optimizedCudaGraphCreator->beginCaptureOperation(nodeToDependentNodesMap[u]);
      // Execute the task with current memory address mapping
      executeRandomTask(
        optimizedGraph.nodeIdToTaskIdMap[u],
        memManager.getEditableCurrentAddressMap(),
        stream
      );
      newLeafNodes = optimizedCudaGraphCreator->endCaptureOperation();
    } else if (nodeType == OptimizationOutput::NodeType::empty) {
      // Handle empty nodes (for dependencies)
      newLeafNodes.push_back(
        optimizedCudaGraphCreator->addEmptyNode(nodeToDependentNodesMap[u])
      );
    } else {
      LOG_TRACE_WITH_INFO("Unsupported node type: %d", nodeType);
      exit(-1);
    }

    // Update dependencies and process nodes that have all dependencies satisfied
    for (auto &v : optimizedGraph.edges[u]) {
      inDegrees[v]--;

      // Add dependencies for the next nodes
      nodeToDependentNodesMap[v].insert(
        nodeToDependentNodesMap[v].end(),
        newLeafNodes.begin(),
        newLeafNodes.end()
      );

      // If all dependencies are satisfied, add to the queue
      if (inDegrees[v] == 0) {
        nodesToExecute.push(v);
      }
    }
  }

  // Add cleanup operations for any remaining device memory
  newLeafNodes = getNodesWithZeroOutDegree(graph);
  optimizedCudaGraphCreator->beginCaptureOperation(newLeafNodes);
  memManager.moveRemainedManagedMemoryToStorage(stream);
  newLeafNodes = optimizedCudaGraphCreator->endCaptureOperation();
  checkCudaErrors(cudaDeviceSynchronize());

  // Report time taken to build the graph
  clock.end();
  LOG_TRACE_WITH_INFO("Time taken for recording graph: %.6f", clock.getTimeInSeconds());

  // Export graph for debugging/visualization
  LOG_TRACE_WITH_INFO("Printing the new CUDA Graph to newGraph.dot");
  checkCudaErrors(cudaGraphDebugDotPrint(graph, "newGraph.dot", 0));

  //----------------------------------------------------------------------
  // STEP 5: Execute the optimized graph repeatedly
  //----------------------------------------------------------------------
  
  LOG_TRACE_WITH_INFO("Execute the new CUDA Graph");
  
  // Set up profiling if requested
  PeakMemoryUsageProfiler peakMemoryUsageProfiler;
  CudaEventClock cudaEventClock;
  
  // Instantiate the graph for execution
  cudaGraphExec_t graphExec;
  checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  // Upload the graph to the device for faster execution
  checkCudaErrors(cudaGraphUpload(graphExec, stream));
  checkCudaErrors(cudaStreamSynchronize(stream));

  // Start memory usage profiling if requested
  if (ConfigurationManager::getConfig().execution.measurePeakMemoryUsage) {
    peakMemoryUsageProfiler.start();
  }

  // Initialize iteration counter
  numIterations = 0;

  // Execute and time the graph
  cudaEventClock.start();
  // Execute graph repeatedly until termination condition is met
  while (shouldContinue()) {
    checkCudaErrors(cudaGraphLaunch(graphExec, stream));
    numIterations++;
    checkCudaErrors(cudaDeviceSynchronize());
  }
  cudaEventClock.end();
  checkCudaErrors(cudaDeviceSynchronize());

  // Report peak memory usage if requested
  if (ConfigurationManager::getConfig().execution.measurePeakMemoryUsage) {
    const auto peakMemoryUsage = peakMemoryUsageProfiler.end();
    LOG_TRACE_WITH_INFO(
      "Peak memory usage (MiB): %.2f",
      static_cast<float>(peakMemoryUsage) / 1024.0 / 1024.0
    );
  }

  //----------------------------------------------------------------------
  // STEP 6: Clean up resources
  //----------------------------------------------------------------------
  
  LOG_TRACE_WITH_INFO("Clean up");
  
  // Clean up CUDA resources
  checkCudaErrors(cudaGraphExecDestroy(graphExec));
  checkCudaErrors(cudaGraphDestroy(graph));
  checkCudaErrors(cudaStreamDestroy(stream));

  // Clean up storage configuration
  memManager.cleanStorage();

  // Store the execution time
  runningTime = cudaEventClock.getTimeInSeconds();
}

/*
 * executeOptimizedGraphRepeatedly - Executes a CUDA graph with memory optimization repeatedly
 * 
 * This version uses the singleton instance of MemoryManager.
 */
void Executor::executeOptimizedGraphRepeatedly(
  OptimizationOutput &optimizedGraph,
  ExecuteRandomTask executeRandomTask,
  ShouldContinue shouldContinue,
  int &numIterations,
  float &runningTime
) {
  // Call the implementation version that takes a MemoryManager instance
  executeOptimizedGraphRepeatedly(
    optimizedGraph, 
    executeRandomTask, 
    shouldContinue, 
    numIterations, 
    runningTime, 
    MemoryManager::getInstance()
  );
}



}  // namespace memopt