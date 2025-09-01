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
  // Only output detailed logs if verbose output is enabled
  if (ConfigurationManager::getConfig().execution.enableVerboseOutput) {
    LOG_TRACE_WITH_INFO("[VERBOSE] Executing handleDataMovementNode for nodeId=%d", nodeId);
  }
  
  // Begin capturing operations with dependencies
  creator->beginCaptureOperation(dependencies);
  
  // Get data movement details
  auto &dataMovement = optimizedGraph.nodeIdToDataMovementMap.at(nodeId);
  
  // Verbose output for data movement operations
  if (ConfigurationManager::getConfig().execution.enableVerboseOutput) {
    if (dataMovement.direction == OptimizationOutput::DataMovement::Direction::hostToDevice) {
      LOG_TRACE_WITH_INFO("[VERBOSE] Prefetching arrayId=%d to device", dataMovement.arrayId);
    } else {
      LOG_TRACE_WITH_INFO("[VERBOSE] Offloading arrayId=%d from device", dataMovement.arrayId);
    }
  }
  
  // Perform the appropriate data movement operation
  if (dataMovement.direction == OptimizationOutput::DataMovement::Direction::hostToDevice) {
    memManager.prefetchToDeviceAsync(dataMovement.arrayId, stream);
  } else {
    memManager.offloadFromDeviceAsync(dataMovement.arrayId, stream);
  }
  
  // Check for errors and end capture
  checkCudaErrors(cudaPeekAtLastError());
  auto newLeafNodes = creator->endCaptureOperation();
  checkCudaErrors(cudaPeekAtLastError());
  
  if (ConfigurationManager::getConfig().execution.enableVerboseOutput) {
    LOG_TRACE_WITH_INFO("[VERBOSE] handleDataMovementNode completed successfully");
  }
  
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
  
  LOG_TRACE_WITH_INFO("Initialize managed data distribution");
  
  // Configure memory manager storage

  // Clear current memory mappings
  // memManager.clearCurrentMappings();
  
  // Create a subgraph for initial data prefetching
  checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  
  // Prefetch arrays needed at the start
  memManager.prefetchAllDataToDeviceAsync(
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
    memManager.configureStorage(
    ConfigurationManager::getConfig().execution.mainDeviceId,
    ConfigurationManager::getConfig().execution.storageDeviceId,
    ConfigurationManager::getConfig().execution.useNvlink
  );
  
  // Move all managed data to storage (host or secondary GPU)
  // The control would automatically go back to the main device. 
  memManager.offloadAllManagedMemoryToStorage();
  fprintf(stderr, "[DEBUG] offloadAllManagedMemoryToStorage done\n");
  cudaGraphExec_t initialDataGraphExec = initializeMemory(
    optimizedGraph, memManager, stream);
  fprintf(stderr, "[DEBUG] initializeMemory done\n");
  // Build the optimized execution graph
  cudaGraph_t graph = buildOptimizedGraph(
    optimizedGraph, executeRandomTask, memManager, stream);
  
  // Execute the optimized graph
  executeGraph(graph, stream, runningTime);
  
  // Clean up resources
  LOG_TRACE_WITH_INFO("Clean up");
  
  // Copy remaining data back to storage
  memManager.offloadRemainedManagedMemoryToStorage();
  
  // Clean up CUDA resources
  checkCudaErrors(cudaGraphExecDestroy(initialDataGraphExec));
  checkCudaErrors(cudaGraphDestroy(graph));
  checkCudaErrors(cudaStreamDestroy(stream));
  
  // Reset Storage Config
  memManager.ResetStorageConfig();
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

/**
 * @brief Cold start execution - asserts data starts in CPU, guarantees data ends in CPU
 * 
 * This method executes the optimized graph with strict CPU→GPU→CPU guarantees:
 * 1. ASSERTS all data starts in CPU/storage (deviceAddress == nullptr)
 * 2. Runs normal optimized execution (handles GPU transfers internally)
 * 3. GUARANTEES all data ends in CPU/storage (explicit final offload + assertion)
 *
 * @param optimizedGraph The optimization plan
 * @param executeRandomTask Callback to execute tasks
 * @param runningTime Output parameter for execution time
 * @param memManager Reference to memory manager
 */
void Executor::executeOptimizedGraphColdStart(
  OptimizationOutput &optimizedGraph,
  ExecuteRandomTask executeRandomTask,
  float &runningTime,
  MemoryManager &memManager
) {
  LOG_TRACE_WITH_INFO("Initialize Cold Start Execution (CPU→GPU→CPU)");
  
  // ASSERTION: Verify all managed arrays start in CPU/storage
  LOG_TRACE_WITH_INFO("Asserting all data starts in CPU/storage");
  const auto& managedAddresses = memManager.getManagedAddresses();
  for (const auto& address : managedAddresses) {
    ArrayId arrayId = memManager.getArrayId(address);
    if (arrayId >= 0) {
      const auto& arrayInfo = memManager.getMemoryArrayInfo(arrayId);
      if (arrayInfo.deviceAddress != nullptr) {
        LOG_TRACE_WITH_INFO("ASSERTION FAILED: Array at %p (arrayId=%d) has deviceAddress=%p, expected nullptr", 
                           address, arrayId, arrayInfo.deviceAddress);
        fprintf(stderr, "ERROR: Cold start assertion failed - array %p is on GPU, expected in CPU/storage\n", address);
        exit(-1);
      }
      if (arrayInfo.storageAddress == nullptr) {
        LOG_TRACE_WITH_INFO("ASSERTION FAILED: Array at %p (arrayId=%d) has no storageAddress", 
                           address, arrayId);
        fprintf(stderr, "ERROR: Cold start assertion failed - array %p has no storage location\n", address);
        exit(-1);
      }
    }
  }
  LOG_TRACE_WITH_INFO("✅ Assertion passed: All %zu managed arrays start in CPU/storage", managedAddresses.size());
  
  // Create CUDA stream
  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));
  
  // Configure storage settings
  memManager.configureStorage(
    ConfigurationManager::getConfig().execution.mainDeviceId,
    ConfigurationManager::getConfig().execution.storageDeviceId,
    ConfigurationManager::getConfig().execution.useNvlink
  );
  
  // Execute normal optimized graph (handles GPU transfers internally)
  cudaGraphExec_t initialDataGraphExec = initializeMemory(
    optimizedGraph, memManager, stream);
  
  LOG_TRACE_WITH_INFO("Building and executing optimized graph");
  cudaGraph_t graph = buildOptimizedGraph(
    optimizedGraph, executeRandomTask, memManager, stream);
  executeGraph(graph, stream, runningTime);
  
  // GUARANTEE: Force all data back to CPU/storage
  LOG_TRACE_WITH_INFO("Forcing all data back to CPU/storage");
  memManager.offloadRemainedManagedMemoryToStorage();
  
  // ASSERTION: Verify all managed arrays end in CPU/storage
  LOG_TRACE_WITH_INFO("Asserting all data ends in CPU/storage");
  for (const auto& address : managedAddresses) {
    ArrayId arrayId = memManager.getArrayId(address);
    if (arrayId >= 0) {
      const auto& arrayInfo = memManager.getMemoryArrayInfo(arrayId);
      if (arrayInfo.deviceAddress != nullptr) {
        LOG_TRACE_WITH_INFO("ASSERTION FAILED: Array at %p (arrayId=%d) still has deviceAddress=%p after offload", 
                           address, arrayId, arrayInfo.deviceAddress);
        fprintf(stderr, "ERROR: Cold start final assertion failed - array %p still on GPU after offload\n", address);
        exit(-1);
      }
    }
  }
  LOG_TRACE_WITH_INFO("✅ Final assertion passed: All %zu managed arrays end in CPU/storage", managedAddresses.size());
  
  // Clean up resources
  LOG_TRACE_WITH_INFO("Cold start execution cleanup");
  checkCudaErrors(cudaGraphExecDestroy(initialDataGraphExec));
  checkCudaErrors(cudaGraphDestroy(graph));
  checkCudaErrors(cudaStreamDestroy(stream));
  
  // Reset storage configuration
  memManager.ResetStorageConfig();
  
  LOG_TRACE_WITH_INFO("Cold start execution completed successfully");
}

/**
 * @brief Cold start execution using singleton MemoryManager
 * 
 * This version uses the singleton MemoryManager instance for cold start execution.
 *
 * @param optimizedGraph The optimization plan
 * @param executeRandomTask Callback to execute tasks
 * @param runningTime Output parameter for execution time
 */
void Executor::executeOptimizedGraphColdStart(
  OptimizationOutput &optimizedGraph,
  ExecuteRandomTask executeRandomTask,
  float &runningTime
) {
  executeOptimizedGraphColdStart(optimizedGraph, executeRandomTask, runningTime, MemoryManager::getInstance());
}

/**
 * @brief Builds the execution graph for repeated execution
 * 
 * This builds a graph similar to buildOptimizedGraph but includes
 * initial data prefetching within the same graph, rather than
 * as a separate initialization step.
 *
 * @param optimizedGraph The optimization plan
 * @param executeRandomTask Callback to execute tasks
 * @param memManager Reference to memory manager
 * @param stream CUDA stream to use
 * @param clock Timer for measuring graph construction time
 * @return The created CUDA graph
 */
cudaGraph_t Executor::buildRepeatedExecutionGraph(
  OptimizationOutput &optimizedGraph,
  ExecuteRandomTask executeRandomTask,
  MemoryManager &memManager,
  cudaStream_t stream,
  SystemWallClock &clock
) {
  LOG_TRACE_WITH_INFO("Record nodes to a new CUDA Graph");
  
  // Start timing the graph creation process
  clock.start();
  
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
  
  // Track memory addresses that have been updated (device copies)
  memManager.clearCurrentMappings();
  
  // Initial data allocation is integrated within the main graph
  std::vector<cudaGraphNode_t> newLeafNodes;
  
  // Prefetch arrays needed at the start as part of the main graph
  for (auto arrayId : optimizedGraph.arraysInitiallyAllocatedOnDevice) {
    creator->beginCaptureOperation(newLeafNodes);
    memManager.prefetchToDeviceAsync(arrayId, stream);
    newLeafNodes = creator->endCaptureOperation();
  }
  
  // Maps nodes to their dependencies in the CUDA graph
  std::map<int, std::vector<cudaGraphNode_t>> nodeDependencies;
  
  // Set up dependencies for root nodes to depend on initial data prefetching
  for (auto u : rootNodes) {
    nodeDependencies[u] = newLeafNodes;
  }
  
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
  
  // Add cleanup operations for any remaining device memory
  newLeafNodes = getNodesWithZeroOutDegree(graph);
  creator->beginCaptureOperation(newLeafNodes);
  memManager.offloadRemainedManagedMemoryToStorageAsync(stream);
  newLeafNodes = creator->endCaptureOperation();
  checkCudaErrors(cudaDeviceSynchronize());
  
  // Finish timing and report
  clock.end();
  LOG_TRACE_WITH_INFO("Time taken for recording graph: %.6f", clock.getTimeInSeconds());
  
  // Export graph for debugging/visualization
  LOG_TRACE_WITH_INFO("Printing the new CUDA Graph to newGraph.dot");
  checkCudaErrors(cudaGraphDebugDotPrint(graph, "newGraph.dot", 0));
  
  return graph;
}

/**
 * @brief Executes the graph repeatedly until termination condition
 * 
 * This method launches the graph in a loop until shouldContinue returns false.
 * It handles performance measurement and synchronization between iterations.
 *
 * @param graph The CUDA graph to execute
 * @param stream CUDA stream to use
 * @param shouldContinue Function that determines when to stop
 * @param numIterations Output parameter counting iterations performed
 * @param runningTime Output parameter for execution time
 */
void Executor::executeGraphRepeatedly(
  cudaGraph_t graph,
  cudaStream_t stream,
  ShouldContinue shouldContinue,
  int &numIterations,
  float &runningTime
) {
  LOG_TRACE_WITH_INFO("Execute the new CUDA Graph repeatedly");
  
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
  
  // Execute and time the graph in a loop
  cudaEventClock.start();
  
  // Execute graph repeatedly until termination condition is met
  while (shouldContinue()) {
    checkCudaErrors(cudaGraphLaunch(graphExec, stream));
    numIterations++;
    checkCudaErrors(cudaDeviceSynchronize());
  }
  
  cudaEventClock.end();
  checkCudaErrors(cudaDeviceSynchronize());
  
  // Store the execution time
  runningTime = cudaEventClock.getTimeInSeconds();
  
  // Report peak memory usage if requested
  if (ConfigurationManager::getConfig().execution.measurePeakMemoryUsage) {
    const auto peakMemoryUsage = peakMemoryUsageProfiler.end();
    LOG_TRACE_WITH_INFO(
      "Peak memory usage (MiB): %.2f",
      static_cast<float>(peakMemoryUsage) / 1024.0 / 1024.0
    );
  }
  
  // Cleanup graph execution resources
  checkCudaErrors(cudaGraphExecDestroy(graphExec));
}

/**
 * @brief Main implementation of executeOptimizedGraphRepeatedly
 *
 * This function executes a computation graph that has been optimized to reduce memory usage
 * repeatedly until a termination condition is met. It enables benchmarking and iterative
 * workloads that process data larger than would fit in GPU memory alone.
 * 
 * @param optimizedGraph The optimized computation graph to execute
 * @param executeRandomTask Callback to execute specific computation tasks
 * @param shouldContinue Function that determines when to stop the execution loop
 * @param numIterations Output parameter that counts how many times the graph was executed
 * @param runningTime Output parameter to store the execution time
 * @param memManager Reference to the MemoryManager instance to use
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
  
  // Create CUDA stream
  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));
  
  // Configure memory manager and initialize data distribution
  int mainDeviceId = ConfigurationManager::getConfig().execution.mainDeviceId;
  
  LOG_TRACE_WITH_INFO("Initialize managed data distribution");
  
  // Configure storage parameters
  memManager.configureStorage(
    ConfigurationManager::getConfig().execution.mainDeviceId,
    ConfigurationManager::getConfig().execution.storageDeviceId,
    ConfigurationManager::getConfig().execution.useNvlink
  );
  
  // Move all managed data to storage
  memManager.offloadAllManagedMemoryToStorage();
  
  // Switch back to main GPU
  checkCudaErrors(cudaDeviceSynchronize());
  
  // Timer for graph creation
  SystemWallClock clock;
  
  // Build the optimized execution graph (with integrated initial data prefetching)
  cudaGraph_t graph = buildRepeatedExecutionGraph(
    optimizedGraph, executeRandomTask, memManager, stream, clock);
  
  // Execute the graph repeatedly
  executeGraphRepeatedly(graph, stream, shouldContinue, numIterations, runningTime);
  
  // Clean up resources
  LOG_TRACE_WITH_INFO("Clean up");
  
  // Clean up CUDA resources
  checkCudaErrors(cudaGraphDestroy(graph));
  checkCudaErrors(cudaStreamDestroy(stream));
  
  // Clean up memory manager storage
  memManager.ResetStorageConfig();
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



/**
 * @brief Implementation of executeOptimizedGraphForEvaluation
 * 
 * This function executes a computation graph that has been optimized to reduce memory usage
 * for evaluation purposes. It is similar to executeOptimizedGraph but with specific
 * modifications for evaluating performance characteristics.
 *
 * @param optimizedGraph The optimized computation graph to execute
 * @param executeRandomTask Callback to execute specific computation tasks
 * @param runningTime Output parameter to store the execution time
 * @param memManager Reference to the MemoryManager instance to use
 */
void Executor::executeOptimizedGraphForEvaluation(
  OptimizationOutput &optimizedGraph,
  ExecuteRandomTask executeRandomTask,
  float &runningTime,
  MemoryManager &memManager
) {
  LOG_TRACE_WITH_INFO("Initialize for evaluation");
  
  // Create CUDA stream
  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));
  
  // Initialize memory setup and get initial data graph
  memManager.configureStorage(
    ConfigurationManager::getConfig().execution.mainDeviceId,
    ConfigurationManager::getConfig().execution.storageDeviceId,
    ConfigurationManager::getConfig().execution.useNvlink
  );
  
  // Move all managed data to storage (host or secondary GPU)
  memManager.offloadAllManagedMemoryToStorage();
  fprintf(stderr, "[DEBUG] offloadAllManagedMemoryToStorage done\n");
  cudaGraphExec_t initialDataGraphExec;// = initializeMemory(
  cudaGraph_t graphForInitialDataDistribution;
    // optimizedGraph, memManager, stream);
  {
  
    LOG_TRACE_WITH_INFO("Initialize managed data distribution");
  
    checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    
    // Prefetch arrays needed at the start
    memManager.prefetchAllDataToDeviceAsync(
      optimizedGraph.arraysInitiallyAllocatedOnDevice,
      stream
    );
    
    // End capture and create graph
    
    checkCudaErrors(cudaStreamEndCapture(stream, &graphForInitialDataDistribution));
    
    // Instantiate and execute the initial data distribution graph
    // cudaGraphExec_t initialDataGraphExec;
    checkCudaErrors(cudaGraphInstantiate(
      &initialDataGraphExec, graphForInitialDataDistribution, nullptr, nullptr, 0));
   
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Cleanup the capture graph but keep the executable

    
  }
  fprintf(stderr, "[DEBUG] initializeMemory done\n");
  
  // Build the optimized execution graph
  cudaGraph_t graph = buildOptimizedGraph(
    optimizedGraph, executeRandomTask, memManager, stream);
  
  // Execute the optimized graph
  // executeGraph(graph, stream, runningTime);
    // Execute the optimized graph
  {
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
    checkCudaErrors(cudaGraphLaunch(initialDataGraphExec, stream));
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

  // Clean up resources
  LOG_TRACE_WITH_INFO("Clean up");
  
  // Copy remaining data back to storage
  memManager.offloadRemainedManagedMemoryToStorage();
  
  // Clean up CUDA resources
  checkCudaErrors(cudaGraphDestroy(graphForInitialDataDistribution));
  checkCudaErrors(cudaGraphExecDestroy(initialDataGraphExec));
  checkCudaErrors(cudaGraphDestroy(graph));
  checkCudaErrors(cudaStreamDestroy(stream));
  
  // Reset Storage Config
  memManager.ResetStorageConfig();
}

/**
 * @brief Executes an optimized graph for evaluation using singleton MemoryManager
 * 
 * This version uses the singleton instance of MemoryManager.
 *
 * @param optimizedGraph The optimized computation graph to execute
 * @param executeRandomTask Callback to execute specific computation tasks
 * @param runningTime Output parameter to store the execution time
 */
void Executor::executeOptimizedGraphForEvaluation(
  OptimizationOutput &optimizedGraph,
  ExecuteRandomTask executeRandomTask,
  float &runningTime
) {
  // Call the implementation version that takes a MemoryManager instance
  executeOptimizedGraphForEvaluation(optimizedGraph, executeRandomTask, runningTime, MemoryManager::getInstance());
}

/**
 * @brief Executes an optimized computation graph for evaluation with simplified callback
 * 
 * This overload accepts an ExecuteRandomTaskBase callback that doesn't require address mapping.
 * It wraps the simplified callback with an adapter that provides the mapping.
 *
 * @param optimizedGraph The optimized computation graph to execute
 * @param executeRandomTaskBase Simplified callback function to execute specific tasks
 * @param runningTime Output parameter for recording execution time
 * @param memManager Reference to the MemoryManager instance to use
 */
void Executor::executeOptimizedGraphForEvaluation(
  OptimizationOutput &optimizedGraph,
  ExecuteRandomTaskBase executeRandomTaskBase,
  float &runningTime,
  MemoryManager &memManager
) {
  // Adapter to convert ExecuteRandomTaskBase to ExecuteRandomTask
  ExecuteRandomTask executeRandomTask = [executeRandomTaskBase](
    int taskId, std::map<void *, void *> addressMapping, cudaStream_t stream
  ) {
    return executeRandomTaskBase(taskId, stream);
  };
  
  // Call the main implementation
  executeOptimizedGraphForEvaluation(optimizedGraph, executeRandomTask, runningTime, memManager);
}

/**
 * @brief Executes an optimized computation graph for evaluation with singleton MemoryManager and simplified callback
 * 
 * This version uses the singleton MemoryManager instance and a simplified callback.
 *
 * @param optimizedGraph The optimized computation graph to execute
 * @param executeRandomTaskBase Simplified callback function to execute specific tasks
 * @param runningTime Output parameter for recording execution time
 */
void Executor::executeOptimizedGraphForEvaluation(
  OptimizationOutput &optimizedGraph,
  ExecuteRandomTaskBase executeRandomTaskBase,
  float &runningTime
) {
  executeOptimizedGraphForEvaluation(
    optimizedGraph, 
    executeRandomTaskBase, 
    runningTime, 
    MemoryManager::getInstance()
  );
}

}  // namespace memopt