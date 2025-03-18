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

/**
 * @brief Helper class for constructing optimized CUDA graphs
 * 
 * OptimizedCudaGraphCreator handles the incremental construction of CUDA graphs
 * by providing methods to:
 * - Start/end stream capture operations with specific dependencies
 * - Track new nodes added during capture
 * - Find leaf nodes (nodes with no outgoing edges) after capture operations
 * - Add empty nodes for dependency management
 * 
 * This class is essential for building the modified execution graph with
 * memory optimization operations inserted between computation tasks.
 */
class OptimizedCudaGraphCreator {
 public:
  /**
   * @brief Constructor that sets up the graph builder
   * 
   * @param stream CUDA stream used for capturing operations
   * @param graph CUDA graph where operations will be recorded
   */
  OptimizedCudaGraphCreator(cudaStream_t stream, cudaGraph_t graph) : stream(stream), graph(graph) {}

  /**
   * @brief Starts capturing CUDA operations to the graph
   * 
   * Begins recording operations executed on the stream into the graph.
   * The captured operations will depend on the specified nodes.
   * 
   * @param dependencies List of nodes that must execute before the captured operations
   */
  void beginCaptureOperation(const std::vector<cudaGraphNode_t> &dependencies) {
    checkCudaErrors(cudaStreamBeginCaptureToGraph(this->stream, this->graph, dependencies.data(), nullptr, dependencies.size(), cudaStreamCaptureModeGlobal));
  }

  /**
   * @brief Ends the capture operation and returns new leaf nodes
   * 
   * Stops recording operations to the graph and identifies the new
   * leaf nodes that were added during this capture operation.
   * 
   * @return Vector of new leaf nodes added during the capture
   */
  std::vector<cudaGraphNode_t> endCaptureOperation() {
    checkCudaErrors(cudaStreamEndCapture(this->stream, &this->graph));
    return this->getNewLeafNodesAddedByLastCapture();
  };

  /**
   * @brief Adds an empty node to the graph
   * 
   * Empty nodes are used for dependency management without
   * performing any actual computation.
   * 
   * @param dependencies List of nodes that the new empty node will depend on
   * @return The newly created empty node
   */
  cudaGraphNode_t addEmptyNode(const std::vector<cudaGraphNode_t> &dependencies) {
    cudaGraphNode_t newEmptyNode;
    checkCudaErrors(cudaGraphAddEmptyNode(&newEmptyNode, this->graph, dependencies.data(), dependencies.size()));
    visited[newEmptyNode] = true;
    return newEmptyNode;
  }

 private:
  cudaStream_t stream;                      ///< CUDA stream for capturing operations
  cudaGraph_t graph;                        ///< CUDA graph being constructed
  std::vector<cudaGraphNode_t> lastDependencies; ///< Previous dependencies used for tracking
  std::map<cudaGraphNode_t, bool> visited;  ///< Tracks which nodes have been processed

  /**
   * @brief Identifies leaf nodes added in the most recent capture
   * 
   * After a capture operation, this method:
   * 1. Gets all nodes and edges in the graph
   * 2. Identifies which nodes have outgoing edges
   * 3. Finds newly added nodes (not previously visited)
   * 4. Returns those that are leaf nodes (no outgoing edges)
   * 
   * These leaf nodes represent the completion points of the
   * operations added in the last capture, and are used as
   * dependencies for subsequent operations.
   * 
   * @return Vector of new leaf nodes from the last capture
   */
  std::vector<cudaGraphNode_t> getNewLeafNodesAddedByLastCapture() {
    // Get all nodes in the graph
    size_t numNodes;
    checkCudaErrors(cudaGraphGetNodes(this->graph, nullptr, &numNodes));
    auto nodes = std::make_unique<cudaGraphNode_t[]>(numNodes);
    checkCudaErrors(cudaGraphGetNodes(this->graph, nodes.get(), &numNodes));

    // Get all edges in the graph
    size_t numEdges;
    checkCudaErrors(cudaGraphGetEdges(this->graph, nullptr, nullptr, &numEdges));
    auto from = std::make_unique<cudaGraphNode_t[]>(numEdges);
    auto to = std::make_unique<cudaGraphNode_t[]>(numEdges);
    checkCudaErrors(cudaGraphGetEdges(this->graph, from.get(), to.get(), &numEdges));

    // Track which nodes have outgoing edges
    std::map<cudaGraphNode_t, bool> hasOutGoingEdge;
    for (int i = 0; i < numEdges; i++) {
      hasOutGoingEdge[from[i]] = true;
    }

    // Find new leaf nodes (not visited before and have no outgoing edges)
    std::vector<cudaGraphNode_t> newLeafNodes;
    for (int i = 0; i < numNodes; i++) {
      auto &node = nodes[i];
      if (!visited[node]) {
        visited[node] = true;
        if (!hasOutGoingEdge[node]) {
          newLeafNodes.push_back(node);
        }
      }
    }

    return newLeafNodes;
  }
};

Executor *Executor::instance = nullptr;

Executor *Executor::getInstance() {
  if (instance == nullptr) {
    instance = new Executor();
  }
  return instance;
}

/*
 * executeOptimizedGraph - Executes a CUDA graph with memory optimization
 *
 * This function executes a computation graph that has been optimized to reduce memory usage
 * by dynamically managing data transfers between the main GPU and storage (host memory or secondary GPU).
 * It enables processing of workloads larger than would fit in GPU memory alone.
 *
 * Parameters:
 * - optimizedGraph: The optimized computation graph to execute
 * - executeRandomTask: Callback to execute specific computation tasks
 * - runningTime: Output parameter to store the execution time
 * - managedDeviceArrayToHostArrayMap: Mapping between device (key) and host (host) memory addresses
 */
void Executor::executeOptimizedGraph(
  OptimizationOutput &optimizedGraph,
  ExecuteRandomTask executeRandomTask,
  float &runningTime,
  std::map<void *, void *> &managedDeviceArrayToHostArrayMap
  // Register if 
) {
  LOG_TRACE_WITH_INFO("Initialize");

  // Reset the memory mapping
  managedDeviceArrayToHostArrayMap.clear();

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
  int storageDeviceId = cudaCpuDeviceId;  // Default: use host memory as storage
  cudaMemcpyKind prefetchMemcpyKind = cudaMemcpyHostToDevice;
  cudaMemcpyKind offloadMemcpyKind = cudaMemcpyDeviceToHost;

  // If NVLink is available, use a second GPU as storage instead of host memory
  if (ConfigurationManager::getConfig().execution.useNvlink) {
    storageDeviceId = ConfigurationManager::getConfig().execution.storageDeviceId;
    prefetchMemcpyKind = cudaMemcpyDeviceToDevice;
    offloadMemcpyKind = cudaMemcpyDeviceToDevice;
    enablePeerAccessForNvlink(ConfigurationManager::getConfig().execution.mainDeviceId, ConfigurationManager::getConfig().execution.storageDeviceId);
  }

  //----------------------------------------------------------------------
  // STEP 3: Initialize managed data distribution
  //----------------------------------------------------------------------
  
  LOG_TRACE_WITH_INFO("Initialize managed data distribution");

  // Move all managed data to storage (host or secondary GPU)
  for (auto ptr : MemoryManager::managedMemoryAddresses) {
    void *newPtr;
    if (ConfigurationManager::getConfig().execution.useNvlink) {
      // Allocate on secondary GPU
      checkCudaErrors(cudaSetDevice(storageDeviceId));
      checkCudaErrors(cudaMalloc(&newPtr, MemoryManager::managedMemoryAddressToSizeMap[ptr]));
    } else {
      // Allocate on host memory
      checkCudaErrors(cudaMallocHost(&newPtr, MemoryManager::managedMemoryAddressToSizeMap[ptr]));
    }

    // Create mapping and copy data to storage
    managedDeviceArrayToHostArrayMap[ptr] = newPtr;
    checkCudaErrors(cudaMemcpy(
      newPtr,
      ptr,
      MemoryManager::managedMemoryAddressToSizeMap[ptr],
      cudaMemcpyDefault
    ));
    checkCudaErrors(cudaFree(ptr));  // Free original device memory
  }
  
  // Switch back to main GPU
  checkCudaErrors(cudaSetDevice(mainDeviceId));
  checkCudaErrors(cudaDeviceSynchronize());

  //----------------------------------------------------------------------
  // STEP 4: Initialize data that needs to be on the device at the start
  //----------------------------------------------------------------------
  
  // Track memory addresses that have been updated (device copies)
  // std::map<void *, void *> MemoryManager::managedMemoryAddressToAssignedMap; //address mapping original to new device address
  MemoryManager::managedMemoryAddressToAssignedMap.clear();
  // Create a subgraph for initial data prefetching
  checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  for (auto arrayId : optimizedGraph.arraysInitiallyAllocatedOnDevice) {
    auto ptr = MemoryManager::managedMemoryAddresses[arrayId];
    auto size = MemoryManager::managedMemoryAddressToSizeMap[ptr];
    auto newPtr = managedDeviceArrayToHostArrayMap[ptr];

    // Allocate on device and copy data from storage
    void *devicePtr;
    checkCudaErrors(cudaMallocAsync(&devicePtr, size, stream));
    checkCudaErrors(cudaMemcpyAsync(devicePtr, newPtr, size, prefetchMemcpyKind, stream));
    MemoryManager::managedMemoryAddressToAssignedMap[ptr] = devicePtr;  // Update the address mapping
  }
  
  // End capture and instantiate the initial data distribution graph
  cudaGraph_t graphForInitialDataDistribution;
  checkCudaErrors(cudaStreamEndCapture(stream, &graphForInitialDataDistribution));

  // Execute the initial data distribution
  cudaGraphExec_t graphExecForInitialDataDistribution;
  checkCudaErrors(cudaGraphInstantiate(&graphExecForInitialDataDistribution, graphForInitialDataDistribution, nullptr, nullptr, 0));
  checkCudaErrors(cudaGraphLaunch(graphExecForInitialDataDistribution, stream));
  checkCudaErrors(cudaDeviceSynchronize());

  //----------------------------------------------------------------------
  // STEP 5: Build the optimized execution graph by processing nodes in topological order
  //----------------------------------------------------------------------
  
  LOG_TRACE_WITH_INFO("Record nodes to a new CUDA Graph");

  // Maps nodes to their dependencies in the CUDA graph
  std::map<int, std::vector<cudaGraphNode_t>> nodeToDependentNodesMap;

  // Kahn's Algorithm for topological sort and graph construction
  while (!nodesToExecute.empty()) {
    // Get the next node to process
    auto u = nodesToExecute.front();
    nodesToExecute.pop();

    std::vector<cudaGraphNode_t> newLeafNodes;
    auto nodeType = optimizedGraph.nodeIdToNodeTypeMap[u];
    
    // Process different node types
    if (nodeType == OptimizationOutput::NodeType::dataMovement) {
      //----------------------------------------------------------------------
      // STEP 5a: Handle data movement nodes (prefetch or offload)
      //----------------------------------------------------------------------
      
      optimizedCudaGraphCreator->beginCaptureOperation(nodeToDependentNodesMap[u]);
      auto &dataMovement = optimizedGraph.nodeIdToDataMovementMap[u];
      auto dataMovementAddress = MemoryManager::managedMemoryAddresses[dataMovement.arrayId];
      auto dataMovementSize = MemoryManager::managedMemoryAddressToSizeMap[dataMovementAddress];
      
      if (dataMovement.direction == OptimizationOutput::DataMovement::Direction::hostToDevice) {
        // PREFETCH: Move data from storage to device
        void *devicePtr;
        checkCudaErrors(cudaMallocAsync(&devicePtr, dataMovementSize, stream));
        checkCudaErrors(cudaMemcpyAsync(
          devicePtr,
          managedDeviceArrayToHostArrayMap[dataMovementAddress],
          dataMovementSize,
          prefetchMemcpyKind,
          stream
        ));
        MemoryManager::managedMemoryAddressToAssignedMap[dataMovementAddress] = devicePtr;
      } else {
        // OFFLOAD: Move data from device back to storage and free device memory
        void *devicePtr = MemoryManager::managedMemoryAddressToAssignedMap[dataMovementAddress];
        checkCudaErrors(cudaMemcpyAsync(
          managedDeviceArrayToHostArrayMap[dataMovementAddress],
          devicePtr,
          dataMovementSize,
          offloadMemcpyKind,
          stream
        ));
        checkCudaErrors(cudaFreeAsync(devicePtr, stream));
        MemoryManager::managedMemoryAddressToAssignedMap.erase(dataMovementAddress);
      }
      
      checkCudaErrors(cudaPeekAtLastError());
      newLeafNodes = optimizedCudaGraphCreator->endCaptureOperation();
      checkCudaErrors(cudaPeekAtLastError());
      
    } else if (nodeType == OptimizationOutput::NodeType::task) {
      //----------------------------------------------------------------------
      // STEP 5b: Handle computation task nodes
      //----------------------------------------------------------------------
      
      optimizedCudaGraphCreator->beginCaptureOperation(nodeToDependentNodesMap[u]);
      // Execute the task with current memory address mapping
      executeRandomTask(
        optimizedGraph.nodeIdToTaskIdMap[u],
        MemoryManager::managedMemoryAddressToAssignedMap,
        stream
      );
      newLeafNodes = optimizedCudaGraphCreator->endCaptureOperation();
      
    } else if (nodeType == OptimizationOutput::NodeType::empty) {
      //----------------------------------------------------------------------
      // STEP 5c: Handle empty nodes (for dependencies)
      //----------------------------------------------------------------------
      
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

  // Export graph for debugging/visualization
  LOG_TRACE_WITH_INFO("Printing the new CUDA Graph to newGraph.dot");
  checkCudaErrors(cudaGraphDebugDotPrint(graph, "newGraph.dot", 0));

  //----------------------------------------------------------------------
  // STEP 6: Execute the optimized graph
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

  //----------------------------------------------------------------------
  // STEP 7: Clean up resources and copy any remaining data back to storage
  //----------------------------------------------------------------------
  
  LOG_TRACE_WITH_INFO("Clean up");
  
  // Copy any remaining device data back to storage
  for (auto &[oldAddr, newAddr] : MemoryManager::managedMemoryAddressToAssignedMap) {
    checkCudaErrors(cudaMemcpy(
      managedDeviceArrayToHostArrayMap[oldAddr],
      newAddr,
      MemoryManager::managedMemoryAddressToSizeMap[oldAddr],
      offloadMemcpyKind
    ));
    checkCudaErrors(cudaFree(newAddr));
  }
  checkCudaErrors(cudaDeviceSynchronize());

  // Clean up CUDA resources
  checkCudaErrors(cudaGraphExecDestroy(graphExecForInitialDataDistribution));
  checkCudaErrors(cudaGraphExecDestroy(graphExec));
  checkCudaErrors(cudaGraphDestroy(graphForInitialDataDistribution));
  checkCudaErrors(cudaGraphDestroy(graph));
  checkCudaErrors(cudaStreamDestroy(stream));

  // Disable peer access if using NVLink
  if (ConfigurationManager::getConfig().execution.useNvlink) {
    disablePeerAccessForNvlink(mainDeviceId, storageDeviceId);
  }

  // Store the execution time
  runningTime = cudaEventClock.getTimeInSeconds();
}

/*
 * executeOptimizedGraphRepeatedly - Executes a CUDA graph with memory optimization multiple times
 *
 * This function executes a computation graph that has been optimized to reduce memory usage
 * repeatedly until a termination condition is met. It enables benchmarking and iterative
 * workloads that process data larger than would fit in GPU memory alone.
 * 
 * ******* DIFFERENCES FROM executeOptimizedGraph: 
 * ******* 1. Accepts a shouldContinue callback to determine when to stop execution
 * ******* 2. Tracks number of iterations via numIterations parameter
 * ******* 3. Executes the graph repeatedly in a loop until shouldContinue() returns false
 * ******* 4. Integrates initial data allocation with the main graph instead of as a separate step
 * ******* 5. Uses SystemWallClock to measure graph recording time separately
 *
 * Parameters:
 * - optimizedGraph: The optimized computation graph to execute
 * - executeRandomTask: Callback to execute specific computation tasks
 * ******* - shouldContinue: Function that determines when to stop the execution loop
 * ******* - numIterations: Output parameter that counts how many times the graph was executed
 * - runningTime: Output parameter to store the execution time
 * - managedDeviceArrayToHostArrayMap: Mapping between device and storage memory addresses
 */
void Executor::executeOptimizedGraphRepeatedly(
  OptimizationOutput &optimizedGraph,
  ExecuteRandomTask executeRandomTask,
  ShouldContinue shouldContinue,
  int &numIterations,
  float &runningTime,
  std::map<void *, void *> &managedDeviceArrayToHostArrayMap
) {
  LOG_TRACE_WITH_INFO("Initialize");

  // Reset the memory mapping
  managedDeviceArrayToHostArrayMap.clear();

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
  int storageDeviceId = cudaCpuDeviceId;  // Default: use host memory as storage
  cudaMemcpyKind prefetchMemcpyKind = cudaMemcpyHostToDevice;
  cudaMemcpyKind offloadMemcpyKind = cudaMemcpyDeviceToHost;

  // If NVLink is available, use a second GPU as storage instead of host memory
  if (ConfigurationManager::getConfig().execution.useNvlink) {
    storageDeviceId = ConfigurationManager::getConfig().execution.storageDeviceId;
    prefetchMemcpyKind = cudaMemcpyDeviceToDevice;
    offloadMemcpyKind = cudaMemcpyDeviceToDevice;
    enablePeerAccessForNvlink(ConfigurationManager::getConfig().execution.mainDeviceId, ConfigurationManager::getConfig().execution.storageDeviceId);
  }

  //----------------------------------------------------------------------
  // STEP 3: Initialize managed data distribution
  //----------------------------------------------------------------------
  
  LOG_TRACE_WITH_INFO("Initialize managed data distribution");

  // Move all managed data to storage (host or secondary GPU)
  for (auto ptr : MemoryManager::managedMemoryAddresses) {
    void *newPtr;
    if (ConfigurationManager::getConfig().execution.useNvlink) {
      // Allocate on secondary GPU
      checkCudaErrors(cudaSetDevice(storageDeviceId));
      checkCudaErrors(cudaMalloc(&newPtr, MemoryManager::managedMemoryAddressToSizeMap[ptr]));
    } else {
      // Allocate on host memory
      checkCudaErrors(cudaMallocHost(&newPtr, MemoryManager::managedMemoryAddressToSizeMap[ptr]));
    }

    // Create mapping and copy data to storage
    managedDeviceArrayToHostArrayMap[ptr] = newPtr;
    checkCudaErrors(cudaMemcpy(
      newPtr,
      ptr,
      MemoryManager::managedMemoryAddressToSizeMap[ptr],
      cudaMemcpyDefault
    ));
    checkCudaErrors(cudaFree(ptr));
  }
  
  // Switch back to main GPU
  checkCudaErrors(cudaSetDevice(mainDeviceId));
  checkCudaErrors(cudaDeviceSynchronize());

  /*******/ // Start timing the graph creation process
  /*******/ SystemWallClock clock;
  /*******/ clock.start();

  LOG_TRACE_WITH_INFO("Record nodes to a new CUDA Graph");

  //----------------------------------------------------------------------
  // STEP 4: Build the optimized execution graph
  //----------------------------------------------------------------------
  
  // Track memory addresses that have been updated (device copies)
  // std::map<void *, void *> addressManagedToAssignedMap;
  MemoryManager::managedMemoryAddressToAssignedMap.clear();
  /*******/ // DIFFERENCE: Initial data allocation is integrated with main graph capture
  /*******/ // rather than as a separate step like in executeOptimizedGraph
  std::vector<cudaGraphNode_t> newLeafNodes;
  for (auto arrayId : optimizedGraph.arraysInitiallyAllocatedOnDevice) {
    auto ptr = MemoryManager::managedMemoryAddresses[arrayId];
    auto size = MemoryManager::managedMemoryAddressToSizeMap[ptr];
    auto newPtr = managedDeviceArrayToHostArrayMap[ptr];

    // Allocate on device and copy data from storage
    void *devicePtr;
    optimizedCudaGraphCreator->beginCaptureOperation(newLeafNodes);
    checkCudaErrors(cudaMallocAsync(&devicePtr, size, stream));
    checkCudaErrors(cudaMemcpyAsync(devicePtr, newPtr, size, prefetchMemcpyKind, stream));
    newLeafNodes = optimizedCudaGraphCreator->endCaptureOperation();
    MemoryManager::managedMemoryAddressToAssignedMap[ptr] = devicePtr;
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
      auto dataMovementAddress = MemoryManager::managedMemoryAddresses[dataMovement.arrayId];
      auto dataMovementSize = MemoryManager::managedMemoryAddressToSizeMap[dataMovementAddress];
      
      if (dataMovement.direction == OptimizationOutput::DataMovement::Direction::hostToDevice) {
        // PREFETCH: Move data from storage to device
        void *devicePtr;
        checkCudaErrors(cudaMallocAsync(&devicePtr, dataMovementSize, stream));
        checkCudaErrors(cudaMemcpyAsync(
          devicePtr,
          managedDeviceArrayToHostArrayMap[dataMovementAddress],
          dataMovementSize,
          prefetchMemcpyKind,
          stream
        ));
        MemoryManager::managedMemoryAddressToAssignedMap[dataMovementAddress] = devicePtr;
      } else {
        // OFFLOAD: Move data from device back to storage and free device memory
        void *devicePtr = MemoryManager::managedMemoryAddressToAssignedMap[dataMovementAddress];
        checkCudaErrors(cudaMemcpyAsync(
          managedDeviceArrayToHostArrayMap[dataMovementAddress],
          devicePtr,
          dataMovementSize,
          offloadMemcpyKind,
          stream
        ));
        checkCudaErrors(cudaFreeAsync(devicePtr, stream));
        MemoryManager::managedMemoryAddressToAssignedMap.erase(dataMovementAddress);
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
        MemoryManager::managedMemoryAddressToAssignedMap,
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
  for (auto &[oldAddr, newAddr] : MemoryManager::managedMemoryAddressToAssignedMap) {
    // Copy any remaining device data back to storage and free device memory
    optimizedCudaGraphCreator->beginCaptureOperation(newLeafNodes);
    checkCudaErrors(cudaMemcpyAsync(
      managedDeviceArrayToHostArrayMap[oldAddr],
      newAddr,
      MemoryManager::managedMemoryAddressToSizeMap[oldAddr],
      offloadMemcpyKind,
      stream
    ));
    checkCudaErrors(cudaFreeAsync(newAddr, stream));
    newLeafNodes = optimizedCudaGraphCreator->endCaptureOperation();
  }
  checkCudaErrors(cudaDeviceSynchronize());

  /*******/ // Report time taken to build the graph
  /*******/ clock.end();
  /*******/ LOG_TRACE_WITH_INFO("Time taken for recording graph: %.6f", clock.getTimeInSeconds());

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

  /*******/ // Initialize iteration counter
  /*******/ numIterations = 0;

  // Execute and time the graph
  cudaEventClock.start();
  /*******/ // DIFFERENCE: Execute graph repeatedly until termination condition is met
  /*******/ while (shouldContinue()) {
    checkCudaErrors(cudaGraphLaunch(graphExec, stream));
    /*******/ numIterations++;
    checkCudaErrors(cudaDeviceSynchronize());
  /*******/ }
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

  // Disable peer access if using NVLink
  if (ConfigurationManager::getConfig().execution.useNvlink) {
    disablePeerAccessForNvlink(mainDeviceId, storageDeviceId);
  }

  // Store the execution time
  runningTime = cudaEventClock.getTimeInSeconds();
}



/*
 * executeOptimizedGraphBase - Executes a CUDA graph with memory optimization
 *
 * This function executes a computation graph that has been optimized to reduce memory usage
 * by dynamically managing data transfers between the main GPU and storage (host memory or secondary GPU).
 * It enables processing of workloads larger than would fit in GPU memory alone.
 *
 * Parameters:
 * - optimizedGraph: The optimized computation graph to execute
 * - executeRandomTask: Callback to execute specific computation tasks
 * - runningTime: Output parameter to store the execution time
 * - managedDeviceArrayToHostArrayMap: Mapping between device (key) and host (host) memory addresses
 */
void Executor::executeOptimizedGraph(
  OptimizationOutput &optimizedGraph,
  ExecuteRandomTaskBase executeRandomTaskBase,
  float &runningTime,
  std::map<void *, void *> &managedDeviceArrayToHostArrayMap
  // Register if 
) {
  LOG_TRACE_WITH_INFO("Initialize");

  // Reset the memory mapping
  managedDeviceArrayToHostArrayMap.clear();

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
  int storageDeviceId = cudaCpuDeviceId;  // Default: use host memory as storage
  cudaMemcpyKind prefetchMemcpyKind = cudaMemcpyHostToDevice;
  cudaMemcpyKind offloadMemcpyKind = cudaMemcpyDeviceToHost;

  // If NVLink is available, use a second GPU as storage instead of host memory
  if (ConfigurationManager::getConfig().execution.useNvlink) {
    storageDeviceId = ConfigurationManager::getConfig().execution.storageDeviceId;
    prefetchMemcpyKind = cudaMemcpyDeviceToDevice;
    offloadMemcpyKind = cudaMemcpyDeviceToDevice;
    enablePeerAccessForNvlink(ConfigurationManager::getConfig().execution.mainDeviceId, ConfigurationManager::getConfig().execution.storageDeviceId);
  }

  //----------------------------------------------------------------------
  // STEP 3: Initialize managed data distribution
  //----------------------------------------------------------------------
  
  LOG_TRACE_WITH_INFO("Initialize managed data distribution");

  // Move all managed data to storage (host or secondary GPU)
  for (auto ptr : MemoryManager::managedMemoryAddresses) {
    void *newPtr;
    if (ConfigurationManager::getConfig().execution.useNvlink) {
      // Allocate on secondary GPU
      checkCudaErrors(cudaSetDevice(storageDeviceId));
      checkCudaErrors(cudaMalloc(&newPtr, MemoryManager::managedMemoryAddressToSizeMap[ptr]));
    } else {
      // Allocate on host memory
      checkCudaErrors(cudaMallocHost(&newPtr, MemoryManager::managedMemoryAddressToSizeMap[ptr]));
    }

    // Create mapping and copy data to storage
    managedDeviceArrayToHostArrayMap[ptr] = newPtr;
    checkCudaErrors(cudaMemcpy(
      newPtr,
      ptr,
      MemoryManager::managedMemoryAddressToSizeMap[ptr],
      cudaMemcpyDefault
    ));
    checkCudaErrors(cudaFree(ptr));  // Free original device memory
  }
  
  // Switch back to main GPU
  checkCudaErrors(cudaSetDevice(mainDeviceId));
  checkCudaErrors(cudaDeviceSynchronize());

  //----------------------------------------------------------------------
  // STEP 4: Initialize data that needs to be on the device at the start
  //----------------------------------------------------------------------
  
  // Track memory addresses that have been updated (device copies)
  // std::map<void *, void *> MemoryManager::managedMemoryAddressToAssignedMap; //address mapping original to new device address
  MemoryManager::managedMemoryAddressToAssignedMap.clear();
  // Create a subgraph for initial data prefetching
  checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  for (auto arrayId : optimizedGraph.arraysInitiallyAllocatedOnDevice) {
    auto ptr = MemoryManager::managedMemoryAddresses[arrayId];
    auto size = MemoryManager::managedMemoryAddressToSizeMap[ptr];
    auto newPtr = managedDeviceArrayToHostArrayMap[ptr];

    // Allocate on device and copy data from storage
    void *devicePtr;
    checkCudaErrors(cudaMallocAsync(&devicePtr, size, stream));
    checkCudaErrors(cudaMemcpyAsync(devicePtr, newPtr, size, prefetchMemcpyKind, stream));
    MemoryManager::managedMemoryAddressToAssignedMap[ptr] = devicePtr;  // Update the address mapping
  }
  
  // End capture and instantiate the initial data distribution graph
  cudaGraph_t graphForInitialDataDistribution;
  checkCudaErrors(cudaStreamEndCapture(stream, &graphForInitialDataDistribution));

  // Execute the initial data distribution
  cudaGraphExec_t graphExecForInitialDataDistribution;
  checkCudaErrors(cudaGraphInstantiate(&graphExecForInitialDataDistribution, graphForInitialDataDistribution, nullptr, nullptr, 0));
  checkCudaErrors(cudaGraphLaunch(graphExecForInitialDataDistribution, stream));
  checkCudaErrors(cudaDeviceSynchronize());

  //----------------------------------------------------------------------
  // STEP 5: Build the optimized execution graph by processing nodes in topological order
  //----------------------------------------------------------------------
  
  LOG_TRACE_WITH_INFO("Record nodes to a new CUDA Graph");

  // Maps nodes to their dependencies in the CUDA graph
  std::map<int, std::vector<cudaGraphNode_t>> nodeToDependentNodesMap;

  // Kahn's Algorithm for topological sort and graph construction
  while (!nodesToExecute.empty()) {
    // Get the next node to process
    auto u = nodesToExecute.front();
    nodesToExecute.pop();

    std::vector<cudaGraphNode_t> newLeafNodes;
    auto nodeType = optimizedGraph.nodeIdToNodeTypeMap[u];
    
    // Process different node types
    if (nodeType == OptimizationOutput::NodeType::dataMovement) {
      //----------------------------------------------------------------------
      // STEP 5a: Handle data movement nodes (prefetch or offload)
      //----------------------------------------------------------------------
      
      optimizedCudaGraphCreator->beginCaptureOperation(nodeToDependentNodesMap[u]);
      auto &dataMovement = optimizedGraph.nodeIdToDataMovementMap[u];
      auto dataMovementAddress = MemoryManager::managedMemoryAddresses[dataMovement.arrayId];
      auto dataMovementSize = MemoryManager::managedMemoryAddressToSizeMap[dataMovementAddress];
      
      if (dataMovement.direction == OptimizationOutput::DataMovement::Direction::hostToDevice) {
        // PREFETCH: Move data from storage to device
        void *devicePtr;
        checkCudaErrors(cudaMallocAsync(&devicePtr, dataMovementSize, stream));
        checkCudaErrors(cudaMemcpyAsync(
          devicePtr,
          managedDeviceArrayToHostArrayMap[dataMovementAddress],
          dataMovementSize,
          prefetchMemcpyKind,
          stream
        ));
        MemoryManager::managedMemoryAddressToAssignedMap[dataMovementAddress] = devicePtr;
      } else {
        // OFFLOAD: Move data from device back to storage and free device memory
        void *devicePtr = MemoryManager::managedMemoryAddressToAssignedMap[dataMovementAddress];
        checkCudaErrors(cudaMemcpyAsync(
          managedDeviceArrayToHostArrayMap[dataMovementAddress],
          devicePtr,
          dataMovementSize,
          offloadMemcpyKind,
          stream
        ));
        checkCudaErrors(cudaFreeAsync(devicePtr, stream));
        MemoryManager::managedMemoryAddressToAssignedMap.erase(dataMovementAddress);
      }
      
      checkCudaErrors(cudaPeekAtLastError());
      newLeafNodes = optimizedCudaGraphCreator->endCaptureOperation();
      checkCudaErrors(cudaPeekAtLastError());
      
    } else if (nodeType == OptimizationOutput::NodeType::task) {
      //----------------------------------------------------------------------
      // STEP 5b: Handle computation task nodes
      //----------------------------------------------------------------------
      
      optimizedCudaGraphCreator->beginCaptureOperation(nodeToDependentNodesMap[u]);
      // Execute the task with current memory address mapping
      executeRandomTaskBase(
        optimizedGraph.nodeIdToTaskIdMap[u],
        stream
      );
      newLeafNodes = optimizedCudaGraphCreator->endCaptureOperation();
      
    } else if (nodeType == OptimizationOutput::NodeType::empty) {
      //----------------------------------------------------------------------
      // STEP 5c: Handle empty nodes (for dependencies)
      //----------------------------------------------------------------------
      
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

  // Export graph for debugging/visualization
  LOG_TRACE_WITH_INFO("Printing the new CUDA Graph to newGraph.dot");
  checkCudaErrors(cudaGraphDebugDotPrint(graph, "newGraph.dot", 0));

  //----------------------------------------------------------------------
  // STEP 6: Execute the optimized graph
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

  //----------------------------------------------------------------------
  // STEP 7: Clean up resources and copy any remaining data back to storage
  //----------------------------------------------------------------------
  
  LOG_TRACE_WITH_INFO("Clean up");
  
  // Copy any remaining device data back to storage
  for (auto &[oldAddr, newAddr] : MemoryManager::managedMemoryAddressToAssignedMap) {
    checkCudaErrors(cudaMemcpy(
      managedDeviceArrayToHostArrayMap[oldAddr],
      newAddr,
      MemoryManager::managedMemoryAddressToSizeMap[oldAddr],
      offloadMemcpyKind
    ));
    checkCudaErrors(cudaFree(newAddr));
  }
  checkCudaErrors(cudaDeviceSynchronize());

  // Clean up CUDA resources
  checkCudaErrors(cudaGraphExecDestroy(graphExecForInitialDataDistribution));
  checkCudaErrors(cudaGraphExecDestroy(graphExec));
  checkCudaErrors(cudaGraphDestroy(graphForInitialDataDistribution));
  checkCudaErrors(cudaGraphDestroy(graph));
  checkCudaErrors(cudaStreamDestroy(stream));

  // Disable peer access if using NVLink
  if (ConfigurationManager::getConfig().execution.useNvlink) {
    disablePeerAccessForNvlink(mainDeviceId, storageDeviceId);
  }

  // Store the execution time
  runningTime = cudaEventClock.getTimeInSeconds();
}

}  // namespace memopt


