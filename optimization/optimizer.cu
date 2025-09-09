#include <cuda.h>

#include <algorithm>
#include <cassert>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <utility>

#include "../profiling/annotation.hpp"
#include "../profiling/cudaGraphExecutionTimelineProfiler.hpp"
#include "../profiling/memoryManager.hpp"
#include "../profiling/memred.hpp"
#include "../utilities/configurationManager.hpp"
#include "../utilities/cudaGraphUtilities.hpp"
#include "../utilities/cudaUtilities.hpp"
#include "../utilities/disjointSet.hpp"
#include "../utilities/logger.hpp"
#include "../utilities/utilities.hpp"
#include "optimizer.hpp"
#include "strategies/strategies.hpp"

namespace memopt {

static CUfunction dummyKernelForAnnotationHandle;
static CUfunction dummyKernelForStageSeparatorHandle;

cudaGraph_t dummyKernelForAnnotationGraph;
cudaGraph_t dummyKernelForStageSeparatorGraph;

void registerDummyKernelHandles() {
  cudaStream_t s;
  cudaStreamCreate(&s);

  cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
  TaskAnnotation a{};
  dummyKernelForAnnotation<<<1, 1, 0, s>>>(a, false);
  cudaStreamEndCapture(s, &(dummyKernelForAnnotationGraph));
  CUDA_KERNEL_NODE_PARAMS rootNodeParams;
  getKernelNodeParams(getRootNode(dummyKernelForAnnotationGraph), rootNodeParams);
  dummyKernelForAnnotationHandle = rootNodeParams.func;

  cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
  dummyKernelForStageSeparator<<<1, 1, 0, s>>>();
  cudaStreamEndCapture(s, &(dummyKernelForStageSeparatorGraph));
  getKernelNodeParams(getRootNode(dummyKernelForStageSeparatorGraph), rootNodeParams);
  dummyKernelForStageSeparatorHandle = rootNodeParams.func;

  checkCudaErrors(cudaStreamDestroy(s));
}

void cleanUpDummyKernelFuncHandleRegistrations() {
  checkCudaErrors(cudaGraphDestroy(dummyKernelForAnnotationGraph));
  checkCudaErrors(cudaGraphDestroy(dummyKernelForStageSeparatorGraph));
}

/**
 * Profiles a CUDA graph to obtain its execution timeline
 * 
 * This function profiles the execution of a CUDA graph to collect detailed timing
 * information about when each node in the graph executes. The profiling process
 * involves several steps:
 * 
 * 1. A warm-up run to ensure the GPU is in a steady state
 * 2. Setup of the profiler to track the graph execution
 * 3. An actual profiled run of the graph
 * 4. Collection and processing of the timeline data
 * 
 * The resulting timeline maps each graph node to its execution interval (start/end timestamps),
 * which is useful for analyzing execution patterns, overlaps, and potential optimizations.
 * 
 * @param graph The CUDA graph to profile
 * @return CudaGraphExecutionTimeline A map from graph nodes to their execution time intervals
 */
CudaGraphExecutionTimeline getCudaGraphExecutionTimeline(cudaGraph_t graph) {
  LOG_TRACE();  // Log function entry for debugging purposes

  // Create a stream for graph execution
  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));

  // Step 1: Warm-up execution 
  // This eliminates first-run effects (JIT compilation, caching, etc.)
  // that might affect timing measurements
  cudaGraphExec_t graphExecWarmup;
  checkCudaErrors(cudaGraphInstantiate(&graphExecWarmup, graph, nullptr, nullptr, 0));
  
  //limit cuncurrency
  size_t originalLimit;
  cudaDeviceGetLimit(&originalLimit, cudaLimitDevRuntimePendingLaunchCount);
  cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 1);  // Force sequential

  checkCudaErrors(cudaGraphLaunch(graphExecWarmup, stream));
  checkCudaErrors(cudaDeviceSynchronize());  // Wait for completion
  checkCudaErrors(cudaGraphExecDestroy(graphExecWarmup));  // Clean up
  checkCudaErrors(cudaDeviceSynchronize());  // Wait for completion
  
  // Step 2: Initialize the profiler
  // Get the singleton instance of the timeline profiler
  auto profiler = CudaGraphExecutionTimelineProfiler::getInstance();
  // Setup the profiler to track this specific graph
  // This registers CUPTI callbacks and activity tracking
  profiler->initialize(graph);

  // Step 3: Execute the graph with profiling active
  // This is the run that will be profiled
  cudaGraphExec_t graphExec;
  checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  checkCudaErrors(cudaGraphLaunch(graphExec, stream));
  checkCudaErrors(cudaDeviceSynchronize());  // Wait for completion

  cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, originalLimit);  // Restore concurrency
  // Step 4: Finalize profiling and collect results
  // This flushes activity buffers and disables profiling
  profiler->finalize();

  // Clean up resources
  checkCudaErrors(cudaGraphExecDestroy(graphExec));
  checkCudaErrors(cudaStreamDestroy(stream));

  // Return the collected timeline
  // This will map each original graph node to its execution interval
  return profiler->getTimeline();
}

/**
 * Groups concurrent CUDA graph nodes together to reduce graph complexity
 * 
 * This function identifies nodes that execute concurrently (or with temporal overlap)
 * and groups them together using a disjoint set data structure. This grouping reduces
 * the complexity of the graph while preserving the essential characteristics needed
 * for memory optimization.
 * 
 * The algorithm:
 * 1. Sorts nodes by their execution start time
 * 2. Maintains a "window" of concurrent execution
 * 3. Merges nodes that start execution before the current window ends
 * 4. Creates a new window when nodes no longer overlap or when size limits are reached
 * 
 * This temporal grouping helps simplify the optimization problem without losing
 * critical information about execution dependencies.
 * 
 * @param timeline Execution timeline data for all nodes (start/end timestamps)
 * @param disjointSet Data structure for grouping related nodes (modified by this function)
 * @param logicalNodeSizeLimit Maximum number of nodes allowed in a merged group
 */
void mergeConcurrentCudaGraphNodes(
  CudaGraphExecutionTimeline &timeline,
  DisjointSet<cudaGraphNode_t> &disjointSet,
  const int logicalNodeSizeLimit
) {
  LOG_TRACE();

  // Step 1: Create a map from lifetimes to nodes, which will automatically
  // sort nodes by their execution start time (due to std::map ordering)
  std::map<CudaGraphNodeLifetime, cudaGraphNode_t> lifetimeToCudaGraphNodeMap;
  for (auto &[node, lifetime] : timeline) {
    lifetimeToCudaGraphNodeMap[lifetime] = node;
  }

  // Step 2: Process nodes in order of execution start time
  uint64_t currentWindowEnd = 0;  // Timestamp when current execution window ends
  cudaGraphNode_t currentWindowRepresentativeNode = nullptr;  // Representative node for current window
  
  for (auto &[lifetime, node] : lifetimeToCudaGraphNodeMap) {
    // Special case: Skip memory allocation/deallocation nodes
    // These are handled separately and shouldn't be merged with computation
    cudaGraphNodeType nodeType;
    checkCudaErrors(cudaGraphNodeGetType(node, &nodeType));
    if (nodeType == cudaGraphNodeTypeMemAlloc || nodeType == cudaGraphNodeTypeMemFree) {
      continue;
    }

    // Verify that the node has valid timestamps
    // (first = start time, second = end time)
    assert(lifetime.first != 0 && lifetime.second != 0);

    // Step 3: Determine if this node should be merged with the current window
    // Conditions for merging:
    // 1. We have an active window (not the first node)
    // 2. This node starts before the current window ends (temporal overlap)
    // 3. The merged set size would remain under our size limit
    if (currentWindowRepresentativeNode != nullptr && 
        lifetime.first <= currentWindowEnd && 
        disjointSet.getSetSize(currentWindowRepresentativeNode)+disjointSet.getSetSize(node) < logicalNodeSizeLimit) {
      
      // Merge this node with the current window's representative node
      disjointSet.unionUnderlyingSets(currentWindowRepresentativeNode, node);
      
      // Extend the current window end time if this node finishes later
      currentWindowEnd = std::max(currentWindowEnd, lifetime.second);
    } else {
      // Step 4: Start a new window with this node as the representative
      currentWindowRepresentativeNode = node;
      currentWindowEnd = lifetime.second;
    }
  }
}

/**
 * Helper function that traverses the graph using DFS to map nodes to their controlling annotation nodes
 * 
 * This recursive function walks through the graph and assigns each node to its "governing" annotation node.
 * Annotation nodes in CUDA graphs contain metadata about memory access patterns for a group of nodes.
 * Nodes between two annotation nodes are considered to be under the control of the preceding annotation.
 * 
 * The function works by:
 * 1. Keeping track of the current annotation node as it traverses the graph
 * 2. Updating the current annotation node when a new annotation node is encountered
 * 3. Mapping each traversed node to its governing annotation node
 * 
 * @param currentNode The graph node currently being processed
 * @param currentAnnotationNode The most recent annotation node encountered in this traversal path
 * @param edges Adjacency list representation of the graph's edges
 * @param nodeToAnnotationMap Output map that will associate each node with its governing annotation
 */
void mapNodeToAnnotationDfs(
  cudaGraphNode_t currentNode,
  cudaGraphNode_t currentAnnotationNode,
  std::map<cudaGraphNode_t, std::vector<cudaGraphNode_t>> &edges,
  std::map<cudaGraphNode_t, cudaGraphNode_t> &nodeToAnnotationMap
) {
  // If we've already visited this node, stop to avoid cycles
  if (nodeToAnnotationMap.find(currentNode) != nodeToAnnotationMap.end()) {
    return;
  }

  bool isAnnotationNode = false;

  // Determine if the current node is an annotation node:
  // 1. Either it's the first node we're visiting (root node)
  if (!currentAnnotationNode) {
    isAnnotationNode = true;
  } 
  // 2. Or it's a dummy kernel specifically used for annotation
  else if (compareKernelNodeFunctionHandle(currentNode, dummyKernelForAnnotationHandle)) {
    isAnnotationNode = true;
  }

  // If we found a new annotation node, update our reference
  if (isAnnotationNode) {
    currentAnnotationNode = currentNode;
  }

  // Map this node to its governing annotation node
  nodeToAnnotationMap[currentNode] = currentAnnotationNode;

  // Continue DFS traversal to all children/successors
  for (auto nextNode : edges[currentNode]) {
    mapNodeToAnnotationDfs(nextNode, currentAnnotationNode, edges, nodeToAnnotationMap);
  }
}

/**
 * Maps each node in a CUDA graph to its controlling annotation node
 * 
 * This function creates a mapping between every node in the graph and the annotation node
 * that governs it. Annotation nodes contain metadata about memory access patterns for all
 * operations that follow them in the graph until the next annotation node is encountered.
 * 
 * The mapping is built by performing a depth-first traversal of the graph, tracking
 * the current annotation node at each step. This approach ensures that each node is 
 * correctly associated with the annotation that governs its execution.
 * 
 * The resulting map is crucial for analyzing memory dependencies and optimizing data
 * movement in the execution plan.
 * 
 * @param originalGraph The CUDA graph to analyze
 * @param edges Adjacency list representation of the graph's edges
 * @param nodeToAnnotationMap Output map that will associate each node with its governing annotation
 */
void mapNodeToAnnotation(
  cudaGraph_t originalGraph,
  std::map<cudaGraphNode_t, std::vector<cudaGraphNode_t>> &edges,
  std::map<cudaGraphNode_t, cudaGraphNode_t> &nodeToAnnotationMap
) {
  LOG_TRACE();  // Log function entry for debugging/tracing

  // Start DFS traversal from the root node of the graph
  auto rootNode = getRootNode(originalGraph);
  mapNodeToAnnotationDfs(rootNode, nullptr, edges, nodeToAnnotationMap);
}

void mergeNodesWithSameAnnotation(
  std::vector<cudaGraphNode_t> &nodes,
  std::map<cudaGraphNode_t, cudaGraphNode_t> &nodeToAnnotationMap,
  DisjointSet<cudaGraphNode_t> &disjointSet
) {
  for (auto u : nodes) {
    disjointSet.unionUnderlyingSets(u, nodeToAnnotationMap[u]);
  }
}

TaskId getTaskId(cudaGraphNode_t annotationNode) {
  CUDA_KERNEL_NODE_PARAMS nodeParams;
  getKernelNodeParams(annotationNode, nodeParams);
  assert(nodeParams.func == dummyKernelForAnnotationHandle);

  auto taskAnnotationPtr = reinterpret_cast<TaskAnnotation *>(nodeParams.kernelParams[0]);
  return taskAnnotationPtr->taskId;
}

OptimizationInput::TaskGroup::DataDependency convertTaskAnnotationToTaskGroupDataDependency(
  const TaskAnnotation &taskAnnotation
) {
  OptimizationInput::TaskGroup::DataDependency dep;
  for (int i = 0; i < TaskAnnotation::MAX_NUM_PTR; i++) {
    void *ptr = taskAnnotation.inputs[i];
    if (ptr == nullptr) break;
    dep.inputs.insert(ptr);
  }
  for (int i = 0; i < TaskAnnotation::MAX_NUM_PTR; i++) {
    void *ptr = taskAnnotation.outputs[i];
    if (ptr == nullptr) break;
    dep.outputs.insert(ptr);
  }
  return dep;
}

void getDataDependencyFromAnnotationNode(cudaGraphNode_t annotationNode, OptimizationInput::TaskGroup::DataDependency &dataDependency, bool &inferDataDependency) {
  CUDA_KERNEL_NODE_PARAMS nodeParams;
  getKernelNodeParams(annotationNode, nodeParams);
  assert(nodeParams.func == dummyKernelForAnnotationHandle);

  auto taskAnnotationPtr = reinterpret_cast<TaskAnnotation *>(nodeParams.kernelParams[0]);
  dataDependency = convertTaskAnnotationToTaskGroupDataDependency(*taskAnnotationPtr);
  inferDataDependency = *reinterpret_cast<bool *>(nodeParams.kernelParams[1]);
}

void mergeDataDependency(OptimizationInput::TaskGroup::DataDependency &dataDependency, const OptimizationInput::TaskGroup::DataDependency &additionalDataDependency) {
  dataDependency.inputs.insert(additionalDataDependency.inputs.begin(), additionalDataDependency.inputs.end());
  dataDependency.outputs.insert(additionalDataDependency.outputs.begin(), additionalDataDependency.outputs.end());
}

/**
 * Extracts data dependencies for each computational task in the CUDA graph
 * 
 * This function analyzes the graph to identify which memory buffers are used as inputs
 * and outputs by each task. It works by:
 * 
 * 1. Creating a reverse mapping from annotation nodes to their associated computational nodes
 * 2. For each annotation node:
 *    - Extracting the task ID and explicit data dependencies from the annotation
 *    - If automatic inference is enabled, analyzing the kernel parameters to detect additional
 *      data dependencies not explicitly specified in annotations
 * 
 * The collected data dependency information is critical for memory optimization, as it
 * allows the system to determine:
 * - When memory buffers need to be present on the device
 * - Which buffers can be temporarily moved off the device to free space
 * - How to schedule data transfers to minimize overhead and maximize overlap
 * 
 * @param nodes Vector of all CUDA graph nodes
 * @param nodeToAnnotationMap Mapping from computational nodes to their annotation nodes
 * @param taskIdToDataDependencyMap Output map that will store dependencies for each task ID
 */
void getTaskDataDependencies(
  std::vector<cudaGraphNode_t> &nodes,
  std::map<cudaGraphNode_t, cudaGraphNode_t> &nodeToAnnotationMap,
  std::map<TaskId, OptimizationInput::TaskGroup::DataDependency> &taskIdToDataDependencyMap
) {
  // Create parser for analyzing kernel data dependencies automatically
  MemRedAnalysisParser analysisParser;

  // Build a reverse mapping from annotation nodes to their computational nodes
  // This allows us to process all nodes governed by a single annotation at once
  std::map<cudaGraphNode_t, std::vector<cudaGraphNode_t>> annotationToNodesMap;
  for (const auto [node, annotationNode] : nodeToAnnotationMap) {
    annotationToNodesMap[annotationNode].push_back(node);
  }

  // Process each annotation node and its associated computational nodes
  for (auto [annotationNode, nodes] : annotationToNodesMap) {
    // Extract the task ID from the annotation node
    const TaskId taskId = getTaskId(annotationNode);
    
    // Initialize data dependency structure and extract explicit dependencies
    OptimizationInput::TaskGroup::DataDependency dataDependency;
    bool inferDataDependency;
    getDataDependencyFromAnnotationNode(annotationNode, dataDependency, inferDataDependency);

    // If automatic inference is enabled for this task, analyze the kernel parameters
    if (inferDataDependency) {
      // This task's data dependency should be inferred from compiler pass
      // and kernel parameters.

      for (auto node : nodes) {
        auto nodeType = getNodeType(node);

        // Currently, only kernel's data dependency can be analyzed automatically.
        if (nodeType == cudaGraphNodeTypeKernel) {
          // Skip dummy kernels (annotation and stage separator nodes)
          if (!compareKernelNodeFunctionHandle(node, dummyKernelForAnnotationHandle)
              && !compareKernelNodeFunctionHandle(node, dummyKernelForStageSeparatorHandle)) {
            // Analyze kernel parameters and merge the detected dependencies
            mergeDataDependency(dataDependency, analysisParser.getKernelDataDependency(node));
          }
        }
      }
    }

    // Store the complete data dependency information for this task
    taskIdToDataDependencyMap[taskId] = dataDependency;
  }
}

/**
 * Helper function that traverses the graph with DFS (Depth-first search) to map nodes to execution stages
 * 
 * This recursive function assigns stage indices to nodes in a CUDA graph by traversing
 * the graph in depth-first order. The stage index increments whenever a special
 * "stage separator" node is encountered during traversal.
 * 
 * Stage separation is important for multi-phase applications where different phases
 * have distinct memory requirements and optimization strategies. For example:
 * - Initial data preparation (Stage 0)
 * - Main computation (Stage 1)
 * - Result collection (Stage 2)
 * 
 * @param currentNode The graph node currently being processed
 * @param currentStageIndex The current stage index in this traversal path
 * @param edges Adjacency list representation of the graph's edges
 * @param nodeToStageIndexMap Output map that will associate each node with its stage index
 */
void mapNodeToStageDfs(
  cudaGraphNode_t currentNode,
  int currentStageIndex,
  std::map<cudaGraphNode_t, std::vector<cudaGraphNode_t>> &edges,
  std::map<cudaGraphNode_t, int> &nodeToStageIndexMap
) {
  // If we've already visited this node, stop to avoid cycles and duplicate work
  if (nodeToStageIndexMap.find(currentNode) != nodeToStageIndexMap.end()) {
    return;
  }

  // Check if this node is a stage separator by comparing its function handle
  // Stage separators are special dummy kernel nodes inserted specifically to mark stage boundaries
  bool isStageSeparatorNode = compareKernelNodeFunctionHandle(currentNode, dummyKernelForStageSeparatorHandle);

  // Assign the current stage index to this node
  // Note: The separator node itself belongs to the current (previous) stage
  nodeToStageIndexMap[currentNode] = currentStageIndex;

  // If this was a separator node, increment the stage index for all subsequent nodes
  if (isStageSeparatorNode) {
    currentStageIndex++;
  }

  // Continue DFS traversal to all successors, passing the potentially updated stage index
  for (auto nextNode : edges[currentNode]) {
    mapNodeToStageDfs(nextNode, currentStageIndex, edges, nodeToStageIndexMap);
  }
}

/**
 * Maps each node in a CUDA graph to its execution stage
 * 
 * This function creates a mapping between every node in the graph and the execution
 * stage it belongs to. Stages are separated by special "stage separator" nodes in the graph.
 * 
 * Stage-based processing enables:
 * 1. Breaking down large applications into manageable chunks
 * 2. Applying different memory optimization strategies to different stages
 * 3. Clearing/resetting memory between stages for more efficient usage
 * 4. Handling workflows with dramatically different memory access patterns
 * 
 * @param originalGraph The CUDA graph to analyze
 * @param edges Adjacency list representation of the graph's edges
 * @param nodeToStageIndexMap Output map that will associate each node with its stage index
 */
void mapNodeToStage(
  cudaGraph_t originalGraph,
  std::map<cudaGraphNode_t, std::vector<cudaGraphNode_t>> &edges,
  std::map<cudaGraphNode_t, int> &nodeToStageIndexMap
) {
  LOG_TRACE();  // Log function entry for debugging/tracing

  // Start DFS traversal from the root node of the graph
  // Initial stage index is 0
  auto rootNode = getRootNode(originalGraph);
  mapNodeToStageDfs(rootNode, 0, edges, nodeToStageIndexMap);
}

/**
 * @brief Construct a structured optimization problem from raw CUDA graph profiling data
 * 
 * This function transforms the profiling data collected from a CUDA graph execution
 * into a structured optimization problem. It organizes nodes into task groups,
 * establishes dependencies between tasks and task groups, calculates execution times,
 * and collects memory access patterns. The resulting structure becomes the input to
 * the memory optimization algorithms.
 * 
 * @param originalGraph Original CUDA graph that was profiled
 * @param nodes List of all nodes in the graph
 * @param edges Adjacency list representation of graph edges
 * @param timeline Execution timeline data for each node
 * @param disjointSet Sets of nodes belonged to the same task
 * @param nodeToAnnotationMap Maps nodes to their annotation nodes (containing memory access metadata)
 * @param taskIdToDataDependencyMap Maps tasks to their input/output memory dependencies
 * @return OptimizationInput Structured representation of the optimization problem
 */
OptimizationInput constructOptimizationInput(
  cudaGraph_t originalGraph,
  std::vector<cudaGraphNode_t> &nodes,
  std::map<cudaGraphNode_t, std::vector<cudaGraphNode_t>> &edges,
  CudaGraphExecutionTimeline &timeline,
  DisjointSet<cudaGraphNode_t> &disjointSet, 
  std::map<cudaGraphNode_t, cudaGraphNode_t> &nodeToAnnotationMap,
  std::map<TaskId, OptimizationInput::TaskGroup::DataDependency> &taskIdToDataDependencyMap
) {
  LOG_TRACE();

  OptimizationInput optimizationInput;

  /**
   * Maps disjoint set roots to task group IDs for consistent grouping
   * 
   * This mapping ensures that all CUDA graph nodes that have been grouped together
   * in the same disjoint set are assigned to the same TaskGroup in the OptimizationInput.
   * The key is the root node of a disjoint set (representing a group of related nodes),
   * and the value is the TaskGroupId in optimizationInput.nodes where these nodes belong.
   */
  std::map<cudaGraphNode_t, TaskGroupId> disjointSetRootToTaskGroupIdMap;

  /**
   * Helper function to get or create a task group ID for a CUDA graph node
   * 
   * This function bridges between the disjoint sets (which group related nodes)
   * and the task groups in the OptimizationInput structure. For each node, it:
   * 
   * 1. Looks up the node's TaskId from its annotation
   * 2. Finds which disjoint set the node belongs to (via its root)
   * 3. Creates a new task group if this is the first node from this disjoint set
   *    or uses an existing task group if we've seen nodes from this set before
   * 4. Adds the node's TaskId to the appropriate task group
   * 
   * This ensures that all task nodes (due to executing concurrently (runtime)
   * or sharing annotations (user defined)) are consistently assigned to the same task group,
   * preserving the logical structure of the computation.
   */
  auto getTaskGroupId = [&](cudaGraphNode_t u) {
    // Get the TaskId from the node's annotation
    auto uTaskId = getTaskId(nodeToAnnotationMap[u]);
    
    // Find the root of the disjoint set that contains this node
    // This identifies which group of related nodes this node belongs to
    auto uRoot = disjointSet.findRoot(u);

    // Variable to hold the task group ID (index in optimizationInput.nodes)
    size_t uTaskGroupId;
    
    // Check if we've already created a task group for this disjoint set
    if (disjointSetRootToTaskGroupIdMap.count(uRoot) == 0) {
      // This is the first node we've seen from this disjoint set
      // Create a new task group for this set of related nodes
      optimizationInput.nodes.emplace_back();  // Add a new empty TaskGroup
      uTaskGroupId = optimizationInput.nodes.size() - 1;  // Get its index
      
      // Remember this mapping for future nodes from the same disjoint set
      disjointSetRootToTaskGroupIdMap[uRoot] = uTaskGroupId;
    } else {
      // We've already seen nodes from this disjoint set
      // Use the same task group we assigned previously
      uTaskGroupId = disjointSetRootToTaskGroupIdMap[uRoot];
    }

    // Add this node's TaskId to the appropriate task group
    // Note: The same TaskId might be added multiple times if multiple nodes
    // in the CUDA graph represent the same logical task
    optimizationInput.nodes[uTaskGroupId].nodes.insert(uTaskId);

    return uTaskGroupId;
  };

  //---------- BUILD DEPENDENCY GRAPH ----------
  
  // Track edges we've already added to avoid duplicates between task groups
  std::set<std::pair<TaskGroupId, TaskGroupId>> existingEdges;
  
  // Process each edge in the original graph
  for (const auto &[u, destinations] : edges) {
    auto uTaskId = getTaskId(nodeToAnnotationMap[u]);
    auto uTaskGroupId = getTaskGroupId(u);

    for (auto v : destinations) {
      auto vTaskId = getTaskId(nodeToAnnotationMap[v]);
      auto vTaskGroupId = getTaskGroupId(v);
      
      if (uTaskId == vTaskId) {
        // Skip self-dependencies
        continue;
      } else if (uTaskGroupId == vTaskGroupId) {
        // Internal edge within the same task group
        optimizationInput.nodes[uTaskGroupId].edges[uTaskId].push_back(vTaskId);
      } else {
        // Edge between different task groups (need to deduplicate)
        if (existingEdges.count(std::make_pair(uTaskGroupId, vTaskGroupId)) == 0) {
          existingEdges.insert(std::make_pair(uTaskGroupId, vTaskGroupId));
          optimizationInput.edges[uTaskGroupId].push_back(vTaskGroupId);
        }
      }
    }
  }

  //---------- GATHER EXECUTION TIMELINES ----------
  //---------- TASK LEVEL ---------

  // Maps for storing execution timeline information
  std::map<TaskId, cudaGraphNode_t> taskIdToAnnotationNodeMap;
  std::map<TaskId, std::vector<std::pair<uint64_t, uint64_t>>> taskIdToNodeLifetimes;
  
  // Track global execution span
  uint64_t globalMinStart = std::numeric_limits<uint64_t>::max(), globalMaxEnd = 0;
  
  // Process each node to collect timing information
  for (cudaGraphNode_t u : nodes) {
    auto uTaskId = getTaskId(nodeToAnnotationMap[u]);

    // Remember annotation node for each task
    if (taskIdToAnnotationNodeMap.count(uTaskId) == 0) {
      taskIdToAnnotationNodeMap[uTaskId] = nodeToAnnotationMap[u];
    }

    // Skip annotation nodes (they don't have actual execution time)
    const bool isAnnotationNode = nodeToAnnotationMap[u] == u;
    if (isAnnotationNode) continue;

    // Skip memory allocation/deallocation nodes
    cudaGraphNodeType nodeType;
    checkCudaErrors(cudaGraphNodeGetType(u, &nodeType));
    if (nodeType == cudaGraphNodeTypeMemAlloc || nodeType == cudaGraphNodeTypeMemFree) {
      continue;
    }

    // Record this node's execution lifetime
    taskIdToNodeLifetimes[uTaskId].push_back(timeline[u]);

    // Update global execution span
    globalMinStart = std::min(globalMinStart, timeline[u].first);
    globalMaxEnd = std::max(globalMaxEnd, timeline[u].second);
  }

  //---------- CALCULATE TASK GROUP CHARACTERISTICS ----------
  
  // For each task group, calculate running time and combine data dependencies
  for (auto &taskGroup : optimizationInput.nodes) {
    // Collect lifetimes of all nodes in this task group
    std::vector<std::pair<uint64_t, uint64_t>> nodeLifetimes;
    for (auto taskId : taskGroup.nodes) {
      // Merge data dependencies from all tasks in this group
      mergeDataDependency(taskGroup.dataDependency, taskIdToDataDependencyMap[taskId]);
      
      // Collect execution timelines
      nodeLifetimes.insert(nodeLifetimes.end(), taskIdToNodeLifetimes[taskId].begin(), taskIdToNodeLifetimes[taskId].end());
    }

    // Sort lifetimes by start time for accurate running time calculation
    std::sort(nodeLifetimes.begin(), nodeLifetimes.end());

    // Calculate total running time accounting for overlapping executions
    uint64_t accumulatedRunningTime = 0;
    uint64_t currentWindowEnd = 0;
    for (auto nodeLifetime : nodeLifetimes) {
      if (currentWindowEnd < nodeLifetime.first) {
        // No overlap with previous execution
        accumulatedRunningTime += nodeLifetime.second - nodeLifetime.first;
        currentWindowEnd = nodeLifetime.second;
      } else {
        // Partial overlap - only count the non-overlapping portion
        if (currentWindowEnd < nodeLifetime.second) {
          accumulatedRunningTime += nodeLifetime.second - currentWindowEnd;
          currentWindowEnd = nodeLifetime.second;
        }
      }
    }

    // Convert from nanoseconds to seconds
    taskGroup.runningTime = static_cast<float>(accumulatedRunningTime) * 1e-9f;
  }

  // Calculate total execution time of original graph (in seconds)
  optimizationInput.originalTotalRunningTime = static_cast<float>(globalMaxEnd - globalMinStart) * 1e-9f;

  // Configure initial memory placement policy
  optimizationInput.forceAllArraysToResideOnHostInitiallyAndFinally = false;

  // Set stage index (0 for single-stage graphs)
  optimizationInput.stageIndex = 0;

  //---------- CALCULATE PERFORMANCE STATISTICS ----------
  
  double averageTaskGroupRunningTime = 0.0;
  double averageTaskGroupDataDependencySizeInGiB = 0.0;
  double averageTaskGroupProcessingSpeed = 0.0;
  
  // Process each task group
  for (const auto &tg : optimizationInput.nodes) {
    averageTaskGroupRunningTime += tg.runningTime;

    // Calculate total memory accessed by this task group
    size_t s = 0;
    auto &memManager = MemoryManager::getInstance();
    for (auto p : tg.dataDependency.inputs) {
      s += memManager.getSize(p);
    }
    for (auto p : tg.dataDependency.outputs) {
      s += memManager.getSize(p);
    }

    // Convert bytes to GiB and accumulate statistics
    averageTaskGroupDataDependencySizeInGiB += (double)s / 1024.0 / 1024.0 / 1024.0;
    
    // Calculate data processing speed (GiB/s)
    averageTaskGroupProcessingSpeed += (double)s / 1024.0 / 1024.0 / 1024.0 / tg.runningTime;
  }
  
  // Calculate averages
  averageTaskGroupRunningTime /= optimizationInput.nodes.size();
  averageTaskGroupDataDependencySizeInGiB /= optimizationInput.nodes.size();
  averageTaskGroupProcessingSpeed /= optimizationInput.nodes.size();

  // Log statistics for analysis
  LOG_TRACE_WITH_INFO("Number of task groups: %d", (int)optimizationInput.nodes.size());
  LOG_TRACE_WITH_INFO("Average task group running time (s): %.12lf", averageTaskGroupRunningTime);
  LOG_TRACE_WITH_INFO("Average task group data dependency size (GiB): %.12lf", averageTaskGroupDataDependencySizeInGiB);
  LOG_TRACE_WITH_INFO("Average task group data processing speed (GiB / s): %.12lf", averageTaskGroupProcessingSpeed);

  return optimizationInput;
}

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
void writeTaskGraphToDot(const OptimizationInput &optimizationInput, const std::string &outputPath) {
  LOG_TRACE_WITH_INFO("Writing task graph to DOT file: %s", outputPath.c_str());
  
  std::ofstream dotFile(outputPath);
  if (!dotFile.is_open()) {
    LOG_TRACE_WITH_INFO("Failed to open DOT output file: %s", outputPath.c_str());
    return;
  }

  auto &memManager = MemoryManager::getInstance();
  
  dotFile << "digraph TaskGraph {\n";
  dotFile << "  rankdir=TB;\n";
  dotFile << "  node [fontname=\"Arial\"];\n";
  dotFile << "  edge [fontname=\"Arial\"];\n\n";
  
  // Style definitions
  dotFile << "  // Task group nodes\n";
  dotFile << "  node [shape=box, style=filled, fillcolor=lightblue];\n";
  
  // Create task group nodes
  for (size_t i = 0; i < optimizationInput.nodes.size(); i++) {
    const auto &taskGroup = optimizationInput.nodes[i];
    dotFile << "  task_" << i << " [label=\"Task Group " << i 
            << "\\nTime: " << std::fixed << std::setprecision(6) << taskGroup.runningTime << "s";
    
    // Add task IDs if available
    if (!taskGroup.nodes.empty()) {
      dotFile << "\\nTasks: ";
      bool first = true;
      for (auto taskId : taskGroup.nodes) {
        if (!first) dotFile << ",";
        dotFile << taskId;
        first = false;
      }
    }
    dotFile << "\"];\n";
  }
  
  // Array nodes
  dotFile << "\n  // Array nodes\n";
  dotFile << "  node [shape=ellipse, style=filled, fillcolor=lightgreen];\n";
  
  // Collect all unique arrays and their sizes
  std::map<void*, size_t> uniqueArrays;
  std::map<void*, std::string> arrayLabels;
  int arrayCounter = 0;
  
  for (size_t i = 0; i < optimizationInput.nodes.size(); i++) {
    const auto &taskGroup = optimizationInput.nodes[i];
    
    // Process input arrays
    for (auto ptr : taskGroup.dataDependency.inputs) {
      if (uniqueArrays.find(ptr) == uniqueArrays.end()) {
        uniqueArrays[ptr] = memManager.getSize(ptr);
        arrayLabels[ptr] = "array_" + std::to_string(arrayCounter++);
      }
    }
    
    // Process output arrays
    for (auto ptr : taskGroup.dataDependency.outputs) {
      if (uniqueArrays.find(ptr) == uniqueArrays.end()) {
        uniqueArrays[ptr] = memManager.getSize(ptr);
        arrayLabels[ptr] = "array_" + std::to_string(arrayCounter++);
      }
    }
  }
  
  // Output array nodes
  for (const auto &[ptr, size] : uniqueArrays) {
    double sizeInMB = static_cast<double>(size) / (1024.0 * 1024.0);
    dotFile << "  " << arrayLabels[ptr] << " [label=\"Array\\n" 
            << std::fixed << std::setprecision(2) << sizeInMB << " MB"
            << "\\nPtr: 0x" << std::hex << reinterpret_cast<uintptr_t>(ptr) << std::dec 
            << "\"];\n";
  }
  
  // Data flow edges
  dotFile << "\n  // Data flow edges\n";
  for (size_t i = 0; i < optimizationInput.nodes.size(); i++) {
    const auto &taskGroup = optimizationInput.nodes[i];
    
    // Input edges (array -> task)
    for (auto ptr : taskGroup.dataDependency.inputs) {
      dotFile << "  " << arrayLabels[ptr] << " -> task_" << i 
              << " [color=blue, label=\"input\"];\n";
    }
    
    // Output edges (task -> array)
    for (auto ptr : taskGroup.dataDependency.outputs) {
      dotFile << "  task_" << i << " -> " << arrayLabels[ptr] 
              << " [color=red, label=\"output\"];\n";
    }
  }
  
  // Task dependency edges
  dotFile << "\n  // Task dependency edges\n";
  for (const auto &[fromTaskGroup, toTaskGroups] : optimizationInput.edges) {
    for (auto toTaskGroup : toTaskGroups) {
      dotFile << "  task_" << fromTaskGroup << " -> task_" << toTaskGroup 
              << " [color=black, style=bold, label=\"dependency\"];\n";
    }
  }
  
  // Legend
  dotFile << "\n  // Legend\n";
  dotFile << "  subgraph cluster_legend {\n";
  dotFile << "    label=\"Legend\";\n";
  dotFile << "    style=filled;\n";
  dotFile << "    fillcolor=lightyellow;\n";
  dotFile << "    legend_task [shape=box, fillcolor=lightblue, label=\"Task Group\\n(execution time)\"];\n";
  dotFile << "    legend_array [shape=ellipse, fillcolor=lightgreen, label=\"Memory Array\\n(size)\"];\n";
  dotFile << "    legend_task -> legend_array [color=red, label=\"output\"];\n";
  dotFile << "    legend_array -> legend_task [color=blue, label=\"input\"];\n";
  dotFile << "  }\n";
  
  dotFile << "}\n";
  dotFile.close();
  
  LOG_TRACE_WITH_INFO("Task graph DOT file written successfully");
}

std::vector<OptimizationInput> constructOptimizationInputsForStages(
  cudaGraph_t originalGraph,
  std::vector<cudaGraphNode_t> &allNodes,
  std::map<cudaGraphNode_t, std::vector<cudaGraphNode_t>> &allEdges,
  CudaGraphExecutionTimeline &timeline,
  DisjointSet<cudaGraphNode_t> &disjointSet,
  std::map<cudaGraphNode_t, cudaGraphNode_t> &nodeToAnnotationMap,
  std::map<TaskId, OptimizationInput::TaskGroup::DataDependency> &taskIdToDataDependencyMap,
  std::map<cudaGraphNode_t, int> &nodeToStageIndexMap
) {
  std::vector<std::vector<cudaGraphNode_t>> nodesPerStage;
  for (auto node : allNodes) {
    int stageIndex = nodeToStageIndexMap[node];
    while (nodesPerStage.size() <= stageIndex) nodesPerStage.push_back({});
    nodesPerStage[stageIndex].push_back(node);
  }

  const int numberOfStages = nodesPerStage.size();

  std::vector<std::map<cudaGraphNode_t, std::vector<cudaGraphNode_t>>> edgesPerStage(numberOfStages);
  for (int i = 0; i < numberOfStages; i++) {
    for (auto u : nodesPerStage[i]) {
      for (auto v : allEdges[u]) {
        if (nodeToStageIndexMap[v] == i) {
          edgesPerStage[i][u].push_back(v);
        }
      }
    }
  }

  std::vector<OptimizationInput> optimizationInputs;
  for (int i = 0; i < numberOfStages; i++) {
    optimizationInputs.push_back(
      constructOptimizationInput(
        originalGraph, nodesPerStage[i], edgesPerStage[i], timeline, disjointSet, nodeToAnnotationMap, taskIdToDataDependencyMap
      )
    );
    optimizationInputs.rbegin()->forceAllArraysToResideOnHostInitiallyAndFinally = true;
    optimizationInputs.rbegin()->stageIndex = i;
  }
  return optimizationInputs;
}

OptimizationOutput Optimizer::mergeOptimizationOutputs(std::vector<OptimizationOutput> &optimizationOutputs) {
  auto getRootNodeIndex = [](OptimizationOutput &output) {
    std::map<int, bool> hasIncomingEdge;
    for (auto u : output.nodes) {
      for (auto v : output.edges[u]) {
        hasIncomingEdge[v] = true;
      }
    }
    for (auto u : output.nodes) {
      if (!hasIncomingEdge[u]) return u;
    }
    throw std::runtime_error("Cannot find root node");
  };

  auto getLastNodeIndex = [](OptimizationOutput &output) {
    auto u = output.nodes[0];
    while (output.edges[u].size() != 0) u = output.edges[u][0];
    return u;
  };

  OptimizationOutput mergedOptimizationOutput = optimizationOutputs[0];
  int globalLastNodeIndex = getLastNodeIndex(mergedOptimizationOutput);
  int globalNodeIndexOffset = mergedOptimizationOutput.nodes.size();
  for (int i = 1; i < optimizationOutputs.size(); i++) {
    auto &output = optimizationOutputs[i];
    int rootNodeIndex = getRootNodeIndex(output);
    int lastNodeIndex = getLastNodeIndex(output);
    for (auto u : output.nodes) {
      int newU = u + globalNodeIndexOffset;
      mergedOptimizationOutput.nodes.push_back(newU);
      for (auto v : output.edges[u]) {
        mergedOptimizationOutput.edges[newU].push_back(v + globalNodeIndexOffset);
      }

      auto nodeType = output.nodeIdToNodeTypeMap[u];
      mergedOptimizationOutput.nodeIdToNodeTypeMap[newU] = nodeType;
      if (nodeType == OptimizationOutput::NodeType::task) {
        mergedOptimizationOutput.nodeIdToTaskIdMap[newU] = output.nodeIdToTaskIdMap[u];
      } else if (nodeType == OptimizationOutput::NodeType::dataMovement) {
        mergedOptimizationOutput.nodeIdToDataMovementMap[newU] = output.nodeIdToDataMovementMap[u];
      }
    }

    // Add offload node between stages to move all data to CPU
    int offloadNodeId = globalNodeIndexOffset + output.nodes.size();
    mergedOptimizationOutput.nodes.push_back(offloadNodeId);
    mergedOptimizationOutput.nodeIdToNodeTypeMap[offloadNodeId] = OptimizationOutput::NodeType::dataMovement;
    
    // Create offload data movement for ALL arrays (offload everything to CPU)
    OptimizationOutput::DataMovement offloadMovement;
    offloadMovement.direction = OptimizationOutput::DataMovement::Direction::deviceToHost;
    offloadMovement.arrayId = -1; // Special value to indicate "offload all"
    mergedOptimizationOutput.nodeIdToDataMovementMap[offloadNodeId] = offloadMovement;
    
    // Connect: lastStageNode -> offloadNode -> nextStageRoot
    mergedOptimizationOutput.addEdge(globalLastNodeIndex, offloadNodeId);
    mergedOptimizationOutput.addEdge(offloadNodeId, rootNodeIndex + globalNodeIndexOffset);
    
    globalLastNodeIndex = lastNodeIndex + globalNodeIndexOffset;
    globalNodeIndexOffset += output.nodes.size() + 1; // +1 for the offload node
  }

  mergedOptimizationOutput.arraysInitiallyAllocatedOnDevice.clear();

  return mergedOptimizationOutput;
}

struct SerializableOptimizationOutputNode {
  int nodeId;
  std::vector<int> edges;
  OptimizationOutput::NodeType nodeType;
  int taskId;
  OptimizationOutput::DataMovement::Direction direction;
  int arrayId;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(
    SerializableOptimizationOutputNode,
    nodeId,
    nodeType,
    edges,
    taskId,
    direction,
    arrayId
  );
};

void writeOptimizationOutputToFile(OptimizationOutput &output, const std::string &path) {
  LOG_TRACE_WITH_INFO("Printing optimization plan to %s", path.c_str());

  std::vector<SerializableOptimizationOutputNode> serializableNodes;
  for (auto i : output.nodes) {
    SerializableOptimizationOutputNode node;
    node.nodeId = i;
    node.edges = output.edges[i];
    node.nodeType = output.nodeIdToNodeTypeMap[i];
    if (node.nodeType == OptimizationOutput::NodeType::task) {
      node.taskId = output.nodeIdToTaskIdMap[i];
    } else if (node.nodeType == OptimizationOutput::NodeType::dataMovement) {
      node.direction = output.nodeIdToDataMovementMap[i].direction;
      node.arrayId = output.nodeIdToDataMovementMap[i].arrayId;
    }

    serializableNodes.push_back(node);
  }

  nlohmann::json j;
  j["nodes"] = serializableNodes;
  j["arraysInitiallyAllocatedOnDevice"] = output.arraysInitiallyAllocatedOnDevice;
  std::string s = j.dump(2);
  std::ofstream f(path);
  f << s << std::endl;
}

OptimizationOutput loadOptimizationOutput(const std::string &path) {
  LOG_TRACE_WITH_INFO("Loading optimization plan from %s", path.c_str());

  std::ifstream f(path);
  auto j = nlohmann::json::parse(f);
  auto serializableNodes = j.at("nodes").get<std::vector<SerializableOptimizationOutputNode>>();

  OptimizationOutput output;
  for (const auto &node : serializableNodes) {
    output.nodes.push_back(node.nodeId);
    output.edges[node.nodeId] = node.edges;
    output.nodeIdToNodeTypeMap[node.nodeId] = node.nodeType;
    if (node.nodeType == OptimizationOutput::NodeType::task) {
      output.nodeIdToTaskIdMap[node.nodeId] = node.taskId;
    } else {
      output.nodeIdToDataMovementMap[node.nodeId] = {node.direction, node.arrayId};
    }
  }

  output.arraysInitiallyAllocatedOnDevice = j.at("arraysInitiallyAllocatedOnDevice").get<std::vector<ArrayId>>();

  return output;
}

Optimizer *Optimizer::instance = nullptr;

Optimizer *Optimizer::getInstance() {
  if (instance == nullptr) {
    instance = new Optimizer();
  }
  return instance;
}

/**
 * @brief Profile a CUDA graph execution and generate an optimized execution plan with memory management
 * 
 * This function is the core of the memory optimization system. It performs the following steps:
 * 1. Profile the execution of the original CUDA graph to understand task dependencies, 
 *    execution timelines, and memory access patterns
 * 2. Analyze the annotated graph to extract task groups and data dependencies
 * 3. Generate an optimized execution plan that includes data movement operations to
 *    efficiently manage GPU memory usage
 * 
 * The returned plan allows execution of computations larger than available GPU memory by
 * intelligently scheduling when data should be on GPU and when it can be temporarily moved
 * to host memory or secondary storage.
 * 
 * @param originalGraph The original CUDA graph to optimize
 * @return OptimizationOutput An optimized execution plan with data movement operations
 */
OptimizationOutput Optimizer::profileAndOptimize(cudaGraph_t originalGraph) {
  // Start timing the optimization process
  SystemWallClock clock;
  clock.start();

  // Check if we should load an existing plan instead of recomputing
  if (ConfigurationManager::getConfig().optimization.loadExistingPlan) {
    return loadOptimizationOutput(ConfigurationManager::getConfig().optimization.planPath);
  }

  // Register handles for annotation and stage separator dummy kernels
  registerDummyKernelHandles();
  ScopeGuard scopeGuard([]() -> void { cleanUpDummyKernelFuncHandleRegistrations(); });

  //---------- PROFILING PHASE ----------
  
  // Execute the graph once to capture its execution timeline
  auto timeline = getCudaGraphExecutionTimeline(originalGraph);

  // Extract the graph structure (nodes and edges)
  std::vector<cudaGraphNode_t> nodes;
  std::map<cudaGraphNode_t, std::vector<cudaGraphNode_t>> edges;
  extractGraphNodesAndEdges(originalGraph, nodes, edges);

  // Map each node to its corresponding annotation node (containing metadata)
  std::map<cudaGraphNode_t, cudaGraphNode_t> nodeToAnnotationMap;
  mapNodeToAnnotation(originalGraph, edges, nodeToAnnotationMap);

  // Initialize disjoint set for grouping "task" related nodes
  DisjointSet<cudaGraphNode_t> disjointSet;


  // Merge nodes that share the same annotation (same logical task)
  mergeNodesWithSameAnnotation(nodes, nodeToAnnotationMap, disjointSet);

  // Print the number of task groups after merging nodes with same annotation
  size_t numSets = disjointSet.getNumSets();
  std::cout << "[DEBUG-OUTPUT-OPTIMIZER] Number of task groups after annotation merge: " 
            << numSets << std::endl;

  // Get the configured maximum task group amount (0 means no limit)
  int maxTaskGroupAmount = ConfigurationManager::getConfig().optimization.maxTaskGroupAmount;
  bool shouldMerge = ConfigurationManager::getConfig().optimization.mergeConcurrentCudaGraphNodes;
  
  // Skip merging if number of sets is already small enough (< 100)
  if (numSets < 100) {
    std::cout << "[DEBUG-OUTPUT-OPTIMIZER] Skipping concurrent node merging since task groups < 100" << std::endl;
  }
  // Check if we need to merge due to maxTaskGroupAmount constraint
  else if (maxTaskGroupAmount > 0 && numSets > static_cast<size_t>(maxTaskGroupAmount)) {
    std::cout << "[DEBUG-OUTPUT-OPTIMIZER] Task groups (" << numSets 
              << ") exceed maxTaskGroupAmount (" << maxTaskGroupAmount 
              << "), performing gradual merging" << std::endl;
    
    // Start with a small limit and gradually increase until we meet the target
    // or until increasing doesn't reduce the group count anymore
    int nodeLimit = 2; // Start with merging small groups
    size_t previousNumSets = numSets;
    
    while (numSets > static_cast<size_t>(maxTaskGroupAmount)) {
      mergeConcurrentCudaGraphNodes(timeline, disjointSet, nodeLimit);
      
      size_t newNumSets = disjointSet.getNumSets();
      std::cout << "[DEBUG-OUTPUT-OPTIMIZER] After merging with limit " << nodeLimit 
                << ", task groups: " << newNumSets << std::endl;
      
      // If increasing the limit didn't reduce the count, stop merging
      if (newNumSets >= previousNumSets&&nodeLimit>100) {
        std::cout << "[DEBUG-OUTPUT-OPTIMIZER] No further reduction possible with limit " 
                  << nodeLimit << std::endl;
        break;
      }
      
      previousNumSets = newNumSets;
      numSets = newNumSets;
      
      // Increase the node limit for the next iteration
      nodeLimit = nodeLimit * 2;
    }
  }
  // Standard merging as configured
  else if (shouldMerge) {
    std::cout << "[DEBUG-OUTPUT-OPTIMIZER] Performing standard concurrent node merging" << std::endl;
    mergeConcurrentCudaGraphNodes(timeline, disjointSet, std::numeric_limits<int>::max());
    
    // Print the number of task groups after merging concurrent nodes
    std::cout << "[DEBUG-OUTPUT-OPTIMIZER] Number of task groups after concurrent merge: " 
              << disjointSet.getNumSets() << std::endl;
  }


  // Extract data dependencies for each task from their annotations
  std::map<TaskId, OptimizationInput::TaskGroup::DataDependency> taskIdToDataDependencyMap;
  // Extract from annotation or extract from LLVM preposss pass 
  getTaskDataDependencies(nodes, nodeToAnnotationMap, taskIdToDataDependencyMap);

  //---------- STAGE ANALYSIS ----------
  
  /**
   * Detect if the graph is divided into multiple optimization stages
   * 
   * Stages are user-defined segments of computation separated by special marker nodes
   * (dummyKernelForStageSeparator). Each stage can be independently optimized, which:
   * 1. Simplifies the optimization problem by breaking it into smaller chunks
   * 2. Allows complete memory resets between stages
   * 3. Enables different optimization strategies for different execution phases
   */
  bool hasOnlyOneStage = true;
  for (auto node : nodes) {
    if (compareKernelNodeFunctionHandle(node, dummyKernelForStageSeparatorHandle)) {
      hasOnlyOneStage = false;
      break;
    }
  }

  // If multiple stages exist, map each node to its corresponding stage index
  std::map<cudaGraphNode_t, int> nodeToStageIndexMap;
  if (!hasOnlyOneStage) {
    mapNodeToStage(originalGraph, edges, nodeToStageIndexMap);
  }

  //---------- OPTIMIZATION PHASE ----------
  
  if (hasOnlyOneStage) {
    /**
     * Single-stage optimization path
     * 
     * When the graph has no stage separators, we treat it as a single optimization 
     * problem with these characteristics:
     * - All memory must be managed across the entire execution timeline
     * - Memory constraints must be satisfied for the entire computation
     * - Optimization complexity increases with graph size and complexity
     */
    
    // Construct a single optimization input from the entire graph's profiling data
    auto optimizationInput = constructOptimizationInput(originalGraph, nodes, edges, timeline, disjointSet, nodeToAnnotationMap, taskIdToDataDependencyMap);

    // Generate DOT visualization of the task graph
    writeTaskGraphToDot(optimizationInput, "task_graph.dot");

    // Report profiling time
    clock.end();
    LOG_TRACE_WITH_INFO("Time for profiling (seconds): %.4f", clock.getTimeInSeconds());

    // Apply the two-step optimization strategy to generate a complete execution plan
    auto optimizationOutput = this->optimize<TwoStepOptimizationStrategy>(optimizationInput);

    // Check if optimization succeeded (a feasible plan was found)
    if (optimizationOutput.optimal) {
      // Save the plan to file for potential reuse in future runs
      writeOptimizationOutputToFile(optimizationOutput, ConfigurationManager::getConfig().optimization.planPath);
      return optimizationOutput;
    } else {
      LOG_TRACE_WITH_INFO("Could not find any feasible solution");
      exit(-1);
    }
  } else {
    /**
     * Multi-stage optimization path
     * 
     * When the graph contains stage separators, we:
     * - Break the optimization problem into separate stages
     * - Optimize each stage independently with simpler constraints
     * - Complete memory reset occurs between stages (all arrays move to host)
     * - Merge the individual stage plans into a cohesive execution plan
     * 
     * This approach makes optimization more tractable for complex applications
     * and allows specialized handling of different execution phases.
     */
    
    // Construct separate optimization inputs for each stage
    auto optimizationInputs = constructOptimizationInputsForStages(
      originalGraph, nodes, edges, timeline, disjointSet, nodeToAnnotationMap, taskIdToDataDependencyMap, nodeToStageIndexMap
    );

    // Report profiling time
    clock.end();
    LOG_TRACE_WITH_INFO("Time for profiling (seconds): %.4f", clock.getTimeInSeconds());

    // Apply optimization strategy to each stage independently
    std::vector<OptimizationOutput> optimizationOutputs;
    for (int i = 0; i < optimizationInputs.size(); i++) {
      // Optimize this stage and add its plan to our collection
      optimizationOutputs.push_back(this->optimize<TwoStepOptimizationStrategy>(optimizationInputs[i]));
      
      // Verify that a feasible solution was found for this stage
      if (optimizationOutputs.rbegin()->optimal == false) {
        LOG_TRACE_WITH_INFO("Could not find any feasible solution for stage %d", i);
        exit(-1);
      }
    }

    // Merge the individual stage plans into a single cohesive execution plan
    // This creates a unified plan that properly transitions between stages
    auto mergedOptimizationOutput = mergeOptimizationOutputs(optimizationOutputs);
    writeOptimizationOutputToFile(mergedOptimizationOutput, ConfigurationManager::getConfig().optimization.planPath);
    return mergedOptimizationOutput;
  }
}

// Helper function to estimate memory requirement for a stage
size_t Optimizer::estimateStageMemoryRequirement(
  TaskManager_v2& taskManager,
  const std::vector<TaskId>& stageTasks) {
  
  // TODO: Implement proper memory estimation once TaskManager_v2 API is available
  // For now, return a conservative estimate based on stage size
  size_t estimatedMemoryPerTask = 100 * 1024 * 1024; // 100 MB per task (conservative)
  return stageTasks.size() * estimatedMemoryPerTask;
}

// Helper function to check if memory capacity is sufficient
bool Optimizer::checkMemoryCapacity(size_t requiredMemory, bool hasUM) {
  size_t freeMem, totalMem;
  checkCudaErrors(cudaMemGetInfo(&freeMem, &totalMem));
  
  const double SAFETY_FACTOR = 0.9;
  
  // Only check against available GPU memory
  return requiredMemory <= freeMem * SAFETY_FACTOR;
}

// Removed: hasUnifiedMemory - no longer supporting Unified Memory
// Removed: prefetchStageData - replaced by MemoryManager stage APIs

OptimizationOutput Optimizer::profileAndOptimizeStaged(
  TaskManager_v2& taskManager,
  const std::vector<std::vector<TaskId>>& stageTaskIds,
  cudaStream_t stream) {
  
  LOG_TRACE();
  
  // Create stream if not provided
  cudaStream_t workStream = stream;
  bool createdStream = false;
  if (workStream == nullptr) {
    checkCudaErrors(cudaStreamCreate(&workStream));
    createdStream = true;
  }
  
  auto& memManager = MemoryManager::getInstance();
  
  // Phase 1: Construct and profile all stages with memory management
  std::vector<cudaGraph_t> stageGraphs;
  std::vector<OptimizationOutput> stageOptimizations;
  
  LOG_TRACE_WITH_INFO("Starting stage-based profiling for %zu stages", stageTaskIds.size());
  
  for (size_t stageIdx = 0; stageIdx < stageTaskIds.size(); stageIdx++) {
    const auto& stageTasks = stageTaskIds[stageIdx];
    LOG_TRACE_WITH_INFO("Processing stage %zu with %zu tasks", stageIdx, stageTasks.size());
    
    // Collect arrays needed for this stage by analyzing task inputs/outputs
    std::set<void*> requiredArrays;
    std::set<void*> modifiedArrays;
    
    // Get input and output arrays for all tasks in this stage
    for (TaskId taskId : stageTasks) {
      auto inputs = taskManager.getTaskInputs(taskId);
      auto outputs = taskManager.getTaskOutputs(taskId);
      
      requiredArrays.insert(inputs.begin(), inputs.end());
      requiredArrays.insert(outputs.begin(), outputs.end());
      modifiedArrays.insert(outputs.begin(), outputs.end());
    }
    
    LOG_TRACE_WITH_INFO("Stage %zu requires %zu arrays, modifies %zu arrays", 
                        stageIdx, requiredArrays.size(), modifiedArrays.size());
    
    // Check if stage memory requirements fit in GPU
    if (!memManager.checkStageMemoryRequirement(requiredArrays)) {
      size_t freeMem, totalMem;
      checkCudaErrors(cudaMemGetInfo(&freeMem, &totalMem));
      
      // Calculate actual requirement
      size_t totalRequired = 0;
      for (void* addr : requiredArrays) {
        ArrayId arrayId = memManager.getArrayId(addr);
        if (arrayId >= 0) {
          totalRequired += memManager.getMemoryArrayInfos()[arrayId].size;
        }
      }
      
      if (createdStream) {
        cudaStreamDestroy(workStream);
      }
      
      throw std::runtime_error(
        "Stage " + std::to_string(stageIdx) + " exceeds GPU memory capacity. " +
        "Required: " + std::to_string(totalRequired/1e9) + " GB for " + 
        std::to_string(requiredArrays.size()) + " arrays, " +
        "Available: " + std::to_string((freeMem * 0.9)/1e9) + " GB. " +
        "Please add more stage separators to reduce per-stage memory requirement."
      );
    }
    
    // Prepare stage: fetch required data from CPU to GPU
    LOG_TRACE_WITH_INFO("Fetching data to GPU for stage %zu", stageIdx);
    memManager.prepareStage(stageIdx, requiredArrays, workStream);
    
    // Construct subgraph for this stage
    cudaGraph_t stageGraph;
    checkCudaErrors(cudaGraphCreate(&stageGraph, 0));
    
    // Begin capture
    checkCudaErrors(cudaStreamBeginCapture(workStream, cudaStreamCaptureModeGlobal));
    
    // Execute all tasks in this stage to capture them
    for (TaskId taskId : stageTasks) {
      taskManager.execute(taskId, workStream);
    }
    
    // End capture
    checkCudaErrors(cudaStreamEndCapture(workStream, &stageGraph));
    
    // Profile and optimize this stage
    LOG_TRACE_WITH_INFO("Profiling and optimizing stage %zu", stageIdx);
    auto stageOptimization = profileAndOptimize(stageGraph);
    stageOptimizations.push_back(stageOptimization);
    
    // Finalize stage: offload ALL data to CPU to free GPU memory completely
    LOG_TRACE_WITH_INFO("Offloading ALL data to CPU after stage %zu", stageIdx);
    memManager.finalizeStage(stageIdx, requiredArrays, {}, workStream);  // Pass all arrays, keep none
    
    LOG_TRACE_WITH_INFO("Stage %zu - Original: %.2f MB, Optimized: %.2f MB, Reduction: %.1f%%",
      stageIdx,
      stageOptimization.originalMemoryUsage,
      stageOptimization.anticipatedPeakMemoryUsage,
      stageOptimization.originalMemoryUsage > 0 ? 
        ((stageOptimization.originalMemoryUsage - stageOptimization.anticipatedPeakMemoryUsage) / 
         stageOptimization.originalMemoryUsage) * 100 : 0);
    
    // Store graph for potential future use
    stageGraphs.push_back(stageGraph);
  }
  
  // Clean up graphs
  for (auto& graph : stageGraphs) {
    checkCudaErrors(cudaGraphDestroy(graph));
  }
  
  // Clean up stream if we created it
  if (createdStream) {
    checkCudaErrors(cudaStreamDestroy(workStream));
  }
  
  // Phase 2: Merge results
  LOG_TRACE_WITH_INFO("Merging %zu stage optimization results", stageOptimizations.size());
  auto mergedOutput = mergeOptimizationOutputs(stageOptimizations);
  
  LOG_TRACE_WITH_INFO("Overall - Original: %.2f MB, Optimized: %.2f MB, Reduction: %.1f%%",
    mergedOutput.originalMemoryUsage,
    mergedOutput.anticipatedPeakMemoryUsage,
    mergedOutput.originalMemoryUsage > 0 ?
      ((mergedOutput.originalMemoryUsage - mergedOutput.anticipatedPeakMemoryUsage) / 
       mergedOutput.originalMemoryUsage) * 100 : 0);
  
  return mergedOutput;
}

OptimizationOutput Optimizer::profileAndOptimizeStaged(
  const std::vector<cudaGraph_t>& stageGraphs,
  bool enablePipelining) {
  
  LOG_TRACE();
  
  // For pre-constructed graphs, we skip the construction phase
  std::vector<OptimizationOutput> stageOptimizations;
  
  LOG_TRACE_WITH_INFO("Starting stage-based profiling for %zu pre-constructed stage graphs", stageGraphs.size());
  
  // TODO: If enablePipelining is true in the future, we could implement
  // concurrent profiling/optimization using async operations
  
  for (size_t stageIdx = 0; stageIdx < stageGraphs.size(); stageIdx++) {
    LOG_TRACE_WITH_INFO("Profiling and optimizing stage %zu", stageIdx);
    
    // Profile and optimize this stage
    auto stageOptimization = profileAndOptimize(stageGraphs[stageIdx]);
    stageOptimizations.push_back(stageOptimization);
    
    LOG_TRACE_WITH_INFO("Stage %zu - Original: %.2f MB, Optimized: %.2f MB, Reduction: %.1f%%",
      stageIdx,
      stageOptimization.originalMemoryUsage,
      stageOptimization.anticipatedPeakMemoryUsage,
      stageOptimization.originalMemoryUsage > 0 ?
        ((stageOptimization.originalMemoryUsage - stageOptimization.anticipatedPeakMemoryUsage) / 
         stageOptimization.originalMemoryUsage) * 100 : 0);
  }
  
  // Merge results
  LOG_TRACE_WITH_INFO("Merging %zu stage optimization results", stageOptimizations.size());
  auto mergedOutput = mergeOptimizationOutputs(stageOptimizations);
  
  LOG_TRACE_WITH_INFO("Overall - Original: %.2f MB, Optimized: %.2f MB, Reduction: %.1f%%",
    mergedOutput.originalMemoryUsage,
    mergedOutput.anticipatedPeakMemoryUsage,
    mergedOutput.originalMemoryUsage > 0 ?
      ((mergedOutput.originalMemoryUsage - mergedOutput.anticipatedPeakMemoryUsage) / 
       mergedOutput.originalMemoryUsage) * 100 : 0);
  
  return mergedOutput;
}

}  // namespace memopt
