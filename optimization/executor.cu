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

// Method implementations for OptimizedCudaGraphCreator

void OptimizedCudaGraphCreator::beginCaptureOperation(const std::vector<cudaGraphNode_t> &dependencies) {
  checkCudaErrors(cudaStreamBeginCaptureToGraph(this->stream, this->graph, dependencies.data(), nullptr, dependencies.size(), cudaStreamCaptureModeGlobal));
}

std::vector<cudaGraphNode_t> OptimizedCudaGraphCreator::endCaptureOperation() {
  checkCudaErrors(cudaStreamEndCapture(this->stream, &this->graph));
  return this->getNewLeafNodesAddedByLastCapture();
}

cudaGraphNode_t OptimizedCudaGraphCreator::addEmptyNode(const std::vector<cudaGraphNode_t> &dependencies) {
  cudaGraphNode_t newEmptyNode;
  checkCudaErrors(cudaGraphAddEmptyNode(&newEmptyNode, this->graph, dependencies.data(), dependencies.size()));
  visited[newEmptyNode] = true;
  return newEmptyNode;
}

std::vector<cudaGraphNode_t> OptimizedCudaGraphCreator::getNewLeafNodesAddedByLastCapture() {
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
  auto& memManager = MemoryManager::getInstance();
  auto& managedAddresses = memManager.getEditableManagedAddresses();
  for (auto ptr : managedAddresses) {
    void *newPtr;
    if (ConfigurationManager::getConfig().execution.useNvlink) {
      // Allocate on secondary GPU
      checkCudaErrors(cudaSetDevice(storageDeviceId));
      checkCudaErrors(cudaMalloc(&newPtr, memManager.getSize(ptr)));
    } else {
      // Allocate on host memory
      checkCudaErrors(cudaMallocHost(&newPtr, memManager.getSize(ptr)));
    }

    // Create mapping and copy data to storage
    managedDeviceArrayToHostArrayMap[ptr] = newPtr;
    checkCudaErrors(cudaMemcpy(
      newPtr,
      ptr,
      memManager.getSize(ptr),
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
  // address mapping original to new device address
  memManager.clearCurrentMappings();
  // Create a subgraph for initial data prefetching
  checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  for (auto arrayId : optimizedGraph.arraysInitiallyAllocatedOnDevice) {
    auto ptr = memManager.getPointerByArrayId(arrayId);
    auto size = memManager.getSizeByArrayId(arrayId);
    auto newPtr = managedDeviceArrayToHostArrayMap[ptr];

    // Allocate on device and copy data from storage
    void *devicePtr;
    checkCudaErrors(cudaMallocAsync(&devicePtr, size, stream));
    checkCudaErrors(cudaMemcpyAsync(devicePtr, newPtr, size, prefetchMemcpyKind, stream));
    memManager.updateCurrentMapping(ptr, devicePtr);  // Update the address mapping
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
      auto dataMovementAddress = memManager.getPointerByArrayId(dataMovement.arrayId);
      auto dataMovementSize = memManager.getSizeByArrayId(dataMovement.arrayId);
      
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
        memManager.updateCurrentMapping(dataMovementAddress, devicePtr);
      } else {
        // OFFLOAD: Move data from device back to storage and free device memory
        void *devicePtr = memManager.getCurrentAddressMap().at(dataMovementAddress);
        checkCudaErrors(cudaMemcpyAsync(
          managedDeviceArrayToHostArrayMap[dataMovementAddress],
          devicePtr,
          dataMovementSize,
          offloadMemcpyKind,
          stream
        ));
        checkCudaErrors(cudaFreeAsync(devicePtr, stream));
        memManager.removeCurrentMapping(dataMovementAddress);
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
        memManager.getEditableCurrentAddressMap(),
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
  auto &currentAddressMap = memManager.getEditableCurrentAddressMap();
  for (auto &[oldAddr, newAddr] : currentAddressMap) {
    checkCudaErrors(cudaMemcpy(
      managedDeviceArrayToHostArrayMap[oldAddr],
      newAddr,
      memManager.getSize(oldAddr),
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
  MemoryManager::getInstance().configureStorage(
    mainDeviceId, 
    storageDeviceId, 
    ConfigurationManager::getConfig().execution.useNvlink
  );
  MemoryManager::getInstance().offloadAllManagedMemoryToStorage(managedDeviceArrayToHostArrayMap);
  
  // Store in internal storage map for future use
  MemoryManager::getInstance().getEditableDeviceToHostArrayMap() = managedDeviceArrayToHostArrayMap;
  
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
  auto &memManager = MemoryManager::getInstance();
  memManager.clearCurrentMappings();
  // DIFFERENCE: Initial data allocation is integrated with main graph capture
  // rather than as a separate step like in executeOptimizedGraph
  std::vector<cudaGraphNode_t> newLeafNodes;
  for (auto arrayId : optimizedGraph.arraysInitiallyAllocatedOnDevice) {
    auto ptr = memManager.getPointerByArrayId(arrayId);
    auto size = memManager.getSizeByArrayId(arrayId);
    auto newPtr = managedDeviceArrayToHostArrayMap[ptr];

    // Allocate on device and copy data from storage
    void *devicePtr;
    optimizedCudaGraphCreator->beginCaptureOperation(newLeafNodes);
    checkCudaErrors(cudaMallocAsync(&devicePtr, size, stream));
    checkCudaErrors(cudaMemcpyAsync(devicePtr, newPtr, size, prefetchMemcpyKind, stream));
    newLeafNodes = optimizedCudaGraphCreator->endCaptureOperation();
    memManager.updateCurrentMapping(ptr, devicePtr);
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
        memManager.updateCurrentMapping(dataMovementAddress, devicePtr);
      } else {
        // OFFLOAD: Move data from device back to storage and free device memory
        void *devicePtr = memManager.getCurrentAddressMap().at(dataMovementAddress);
        checkCudaErrors(cudaMemcpyAsync(
          managedDeviceArrayToHostArrayMap[dataMovementAddress],
          devicePtr,
          dataMovementSize,
          offloadMemcpyKind,
          stream
        ));
        checkCudaErrors(cudaFreeAsync(devicePtr, stream));
        memManager.removeCurrentMapping(dataMovementAddress);
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
  auto &currentAddressMap = memManager.getEditableCurrentAddressMap(); 
  for (auto &[oldAddr, newAddr] : currentAddressMap) {
    // Copy any remaining device data back to storage and free device memory
    optimizedCudaGraphCreator->beginCaptureOperation(newLeafNodes);
    checkCudaErrors(cudaMemcpyAsync(
      managedDeviceArrayToHostArrayMap[oldAddr],
      newAddr,
      memManager.getSize(oldAddr),
      offloadMemcpyKind,
      stream
    ));
    checkCudaErrors(cudaFreeAsync(newAddr, stream));
    newLeafNodes = optimizedCudaGraphCreator->endCaptureOperation();
  }
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
  // DIFFERENCE: Execute graph repeatedly until termination condition is met
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
  MemoryManager::getInstance().configureStorage(
    mainDeviceId, 
    storageDeviceId, 
    ConfigurationManager::getConfig().execution.useNvlink
  );
  MemoryManager::getInstance().offloadAllManagedMemoryToStorage(managedDeviceArrayToHostArrayMap);
  // Old implementation has been replaced with the MemoryManager API call above
  
  // Switch back to main GPU
  checkCudaErrors(cudaSetDevice(mainDeviceId));
  checkCudaErrors(cudaDeviceSynchronize());

  //----------------------------------------------------------------------
  // STEP 4: Initialize data that needs to be on the device at the start
  //----------------------------------------------------------------------
  
  // Track memory addresses that have been updated (device copies)
  auto &memManager = MemoryManager::getInstance();
  memManager.clearCurrentMappings();
  // Create a subgraph for initial data prefetching
  checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  for (auto arrayId : optimizedGraph.arraysInitiallyAllocatedOnDevice) {
    auto ptr = memManager.getPointerByArrayId(arrayId);
    auto size = memManager.getSizeByArrayId(arrayId);
    auto newPtr = managedDeviceArrayToHostArrayMap[ptr];

    // Allocate on device and copy data from storage
    void *devicePtr;
    checkCudaErrors(cudaMallocAsync(&devicePtr, size, stream));
    checkCudaErrors(cudaMemcpyAsync(devicePtr, newPtr, size, prefetchMemcpyKind, stream));
    memManager.updateCurrentMapping(ptr, devicePtr);  // Update the address mapping
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
      auto dataMovementAddress = memManager.getPointerByArrayId(dataMovement.arrayId);
      auto dataMovementSize = memManager.getSizeByArrayId(dataMovement.arrayId);
      
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
        memManager.updateCurrentMapping(dataMovementAddress, devicePtr);
      } else {
        // OFFLOAD: Move data from device back to storage and free device memory
        void *devicePtr = memManager.getCurrentAddressMap().at(dataMovementAddress);
        checkCudaErrors(cudaMemcpyAsync(
          managedDeviceArrayToHostArrayMap[dataMovementAddress],
          devicePtr,
          dataMovementSize,
          offloadMemcpyKind,
          stream
        ));
        checkCudaErrors(cudaFreeAsync(devicePtr, stream));
        memManager.removeCurrentMapping(dataMovementAddress);
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
  auto &currentAddressMap = memManager.getEditableCurrentAddressMap();
  for (auto &[oldAddr, newAddr] : currentAddressMap) {
    checkCudaErrors(cudaMemcpy(
      managedDeviceArrayToHostArrayMap[oldAddr],
      newAddr,
      memManager.getSize(oldAddr),
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

std::vector<cudaGraphNode_t> Executor::getNodesWithZeroOutDegree(cudaGraph_t graph) {
  // Get all nodes in the graph
  size_t numNodes;
  checkCudaErrors(cudaGraphGetNodes(graph, nullptr, &numNodes));
  auto nodes = std::make_unique<cudaGraphNode_t[]>(numNodes);
  checkCudaErrors(cudaGraphGetNodes(graph, nodes.get(), &numNodes));

  // Get all edges in the graph
  size_t numEdges;
  checkCudaErrors(cudaGraphGetEdges(graph, nullptr, nullptr, &numEdges));
  auto from = std::make_unique<cudaGraphNode_t[]>(numEdges);
  auto to = std::make_unique<cudaGraphNode_t[]>(numEdges);
  checkCudaErrors(cudaGraphGetEdges(graph, from.get(), to.get(), &numEdges));

  // Track which nodes have outgoing edges
  std::map<cudaGraphNode_t, bool> hasOutGoingEdge;
  for (size_t i = 0; i < numEdges; i++) {
    hasOutGoingEdge[from[i]] = true;
  }

  // Find leaf nodes (nodes with no outgoing edges)
  std::vector<cudaGraphNode_t> leafNodes;
  for (size_t i = 0; i < numNodes; i++) {
    if (!hasOutGoingEdge[nodes[i]]) {
      leafNodes.push_back(nodes[i]);
    }
  }

  return leafNodes;
}

cudaGraphExec_t Executor::initializeDataDistribution(
  OptimizationOutput &optimizedGraph,
  int mainDeviceId, 
  int storageDeviceId,
  bool useNvlink,
  std::map<void *, void *> &managedDeviceArrayToHostArrayMap,
  cudaStream_t stream,
  cudaMemcpyKind prefetchMemcpyKind
) {
  LOG_TRACE_WITH_INFO("Initialize data distribution");
  
  // Move all managed data to storage (host or secondary GPU) using MemoryManager
  auto& memManager = MemoryManager::getInstance();
  // Configure storage parameters
  memManager.configureStorage(mainDeviceId, storageDeviceId, useNvlink);
  // Use the simplified API that uses the configured parameters
  memManager.offloadAllManagedMemoryToStorage(managedDeviceArrayToHostArrayMap);
  
  // Reset the current assignment map
  memManager.clearCurrentMappings();
  
  // Create a subgraph for initial data prefetching
  cudaGraph_t graphForInitialDataDistribution;
  checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  
  // Prefetch arrays that need to be on device initially
  auto& currentMap = memManager.getEditableCurrentAddressMap();
  memManager.prefetchAllDataToDeviceAsync(
    optimizedGraph.arraysInitiallyAllocatedOnDevice,
    managedDeviceArrayToHostArrayMap,
    currentMap,
    prefetchMemcpyKind,
    stream
  );
  
  // End capture and instantiate graph
  checkCudaErrors(cudaStreamEndCapture(stream, &graphForInitialDataDistribution));
  
  // Create executable graph
  cudaGraphExec_t graphExecForInitialDataDistribution;
  checkCudaErrors(cudaGraphInstantiate(
    &graphExecForInitialDataDistribution, 
    graphForInitialDataDistribution, 
    nullptr, 
    nullptr, 
    0
  ));
  
  // Execute the initial data distribution
  checkCudaErrors(cudaGraphLaunch(graphExecForInitialDataDistribution, stream));
  checkCudaErrors(cudaDeviceSynchronize());
  
  // We'll clean up the graph itself, but return the executable for the caller to clean up
  checkCudaErrors(cudaGraphDestroy(graphForInitialDataDistribution));
  
  return graphExecForInitialDataDistribution;
}

}  // namespace memopt