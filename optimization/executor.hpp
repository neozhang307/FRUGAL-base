#pragma once

#include <functional>
#include <map>
#include <memory>
#include <queue>
#include <vector>

#include "../profiling/memoryManager.hpp"
#include "../utilities/utilities.hpp"
#include "optimizationOutput.hpp"

namespace memopt {

/**
 * @brief Callback function type for executing computational tasks
 * 
 * This function is provided by the application and called by the Executor
 * to perform specific computation tasks within an optimized graph.
 * 
 * @param taskId The identifier of the task to execute
 * @param addressMapping Mapping from original memory addresses to current locations (for data that has been moved)
 * @param stream CUDA stream on which to execute the task
 */
typedef std::function<void(int taskId, std::map<void *, void *> addressMapping, cudaStream_t stream)> ExecuteRandomTask;

/**
 * @brief Simplified task execution callback that doesn't require address mapping
 * 
 * This variant is useful when the application handles memory address translation internally
 * or when using the TaskManager for memory management.
 * 
 * @param taskId The identifier of the task to execute
 * @param stream CUDA stream on which to execute the task
 */
typedef std::function<void(int taskId, cudaStream_t stream)> ExecuteRandomTaskBase;

/**
 * @brief Callback function type for determining when to stop iterative execution
 * 
 * This function is used with executeOptimizedGraphRepeatedly to control
 * when the execution loop should terminate.
 * 
 * @return true if execution should continue, false if it should stop
 */
typedef std::function<bool()> ShouldContinue;

// Forward declaration
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
  void beginCaptureOperation(const std::vector<cudaGraphNode_t> &dependencies);

  /**
   * @brief Ends the capture operation and returns new leaf nodes
   * 
   * Stops recording operations to the graph and identifies the new
   * leaf nodes that were added during this capture operation.
   * 
   * @return Vector of new leaf nodes added during the capture
   */
  std::vector<cudaGraphNode_t> endCaptureOperation();

  /**
   * @brief Adds an empty node to the graph
   * 
   * Empty nodes are used for dependency management without
   * performing any actual computation.
   * 
   * @param dependencies List of nodes that the new empty node will depend on
   * @return The newly created empty node
   */
  cudaGraphNode_t addEmptyNode(const std::vector<cudaGraphNode_t> &dependencies);

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
  std::vector<cudaGraphNode_t> getNewLeafNodesAddedByLastCapture();
};

/**
 * @brief Executes optimized CUDA graphs with memory management
 *
 * The Executor is responsible for running computation graphs that have been
 * optimized to reduce GPU memory usage. It handles:
 * 
 * 1. Data movement between GPU and storage (host memory or secondary GPU)
 * 2. Task scheduling according to the optimized execution plan
 * 3. Dynamic memory allocation and deallocation during execution
 * 4. Address translation between original and current memory locations
 * 
 * This class enables execution of computations that would otherwise exceed
 * available GPU memory by intelligently managing data movement operations.
 */
class Executor {
 public:
  /**
   * @brief Gets the singleton instance of the Executor
   * 
   * @return Pointer to the Executor instance
   */
  static Executor *getInstance();
  
  /**
   * @brief Copy constructor (deleted to enforce singleton pattern)
   */
  Executor(Executor &other) = delete;
  
  /**
   * @brief Assignment operator (deleted to enforce singleton pattern)
   */
  void operator=(const Executor &) = delete;

  /**
   * @brief Executes an optimized computation graph once with provided MemoryManager
   * 
   * This method runs a graph that has been optimized to reduce memory usage
   * by dynamically managing data transfers between the main GPU and storage.
   * Takes a specific MemoryManager instance to use for memory management.
   * 
   * @param optimizedGraph The optimized computation graph to execute
   * @param executeRandomTask Callback function to execute specific tasks
   * @param runningTime Output parameter for recording execution time
   * @param memManager Reference to the MemoryManager instance to use
   */
  void executeOptimizedGraph(
    OptimizationOutput &optimizedGraph,
    ExecuteRandomTask executeRandomTask,
    float &runningTime,
    MemoryManager &memManager
  );
  
  // Helper methods for executeOptimizedGraph
  void prepareTopologicalSort(
    const OptimizationOutput &optimizedGraph,
    std::map<int, int> &inDegrees,
    std::queue<int> &nodesToExecute,
    std::vector<int> &rootNodes
  );
  
  std::vector<cudaGraphNode_t> handleDataMovementNode(
    OptimizedCudaGraphCreator *creator,
    const OptimizationOutput &optimizedGraph,
    int nodeId,
    const std::vector<cudaGraphNode_t> &dependencies,
    MemoryManager &memManager,
    cudaStream_t stream
  );
  
  std::vector<cudaGraphNode_t> handleTaskNode(
    OptimizedCudaGraphCreator *creator,
    const OptimizationOutput &optimizedGraph,
    int nodeId,
    const std::vector<cudaGraphNode_t> &dependencies,
    ExecuteRandomTask executeRandomTask,
    MemoryManager &memManager,
    cudaStream_t stream
  );
  
  cudaGraph_t buildOptimizedGraph(
    OptimizationOutput &optimizedGraph,
    ExecuteRandomTask executeRandomTask,
    MemoryManager &memManager,
    cudaStream_t stream
  );
  
  cudaGraphExec_t initializeMemory(
    OptimizationOutput &optimizedGraph,
    MemoryManager &memManager,
    cudaStream_t stream
  );
  
  void executeGraph(
    cudaGraph_t graph,
    cudaStream_t stream,
    float &runningTime
  );
  
  cudaGraph_t buildRepeatedExecutionGraph(
    OptimizationOutput &optimizedGraph,
    ExecuteRandomTask executeRandomTask,
    MemoryManager &memManager,
    cudaStream_t stream,
    SystemWallClock &clock
  );
  
  void executeGraphRepeatedly(
    cudaGraph_t graph,
    cudaStream_t stream,
    ShouldContinue shouldContinue,
    int &numIterations,
    float &runningTime
  );
  
  /**
   * @brief Executes an optimized computation graph once
   * 
   * This method runs a graph that has been optimized to reduce memory usage
   * by dynamically managing data transfers between the main GPU and storage.
   * This version uses the singleton MemoryManager instance.
   * 
   * @param optimizedGraph The optimized computation graph to execute
   * @param executeRandomTask Callback function to execute specific tasks
   * @param runningTime Output parameter for recording execution time
   */
  void executeOptimizedGraph(
    OptimizationOutput &optimizedGraph,
    ExecuteRandomTask executeRandomTask,
    float &runningTime
  );
  
  /**
   * @brief Executes an optimized computation graph once
   * 
   * This method runs a graph that has been optimized to reduce memory usage
   * by dynamically managing data transfers between the main GPU and storage.
   * This overload accepts an external mapping for backward compatibility.
   * 
   * @param optimizedGraph The optimized computation graph to execute
   * @param executeRandomTask Callback function to execute specific tasks
   * @param runningTime Output parameter for recording execution time
   * @param managedDeviceArrayToHostArrayMap Mapping between device and storage memory addresses
   */
  void executeOptimizedGraph(
    OptimizationOutput &optimizedGraph,
    ExecuteRandomTask executeRandomTask,
    float &runningTime,
    std::map<void *, void *> &managedDeviceArrayToHostArrayMap
  );

  /**
   * @brief Executes an optimized computation graph once with base task execution
   * 
   * This method runs a graph that has been optimized to reduce memory usage
   * by dynamically managing data transfers between the main GPU and storage.
   * Uses the simplified task execution callback that does not require address mapping.
   * 
   * @param optimizedGraph The optimized computation graph to execute
   * @param executeRandomTaskBase Simplified callback function to execute specific tasks
   * @param runningTime Output parameter for recording execution time
   * @param managedDeviceArrayToHostArrayMap Mapping between device and storage memory addresses
   */
  void executeOptimizedGraph(
    OptimizationOutput &optimizedGraph,
    ExecuteRandomTaskBase executeRandomTaskBase,
    float &runningTime,
    std::map<void *, void *> &managedDeviceArrayToHostArrayMap
  );

  /**
   * @brief Executes an optimized computation graph repeatedly with provided MemoryManager
   * 
   * Similar to executeOptimizedGraph, but runs the graph multiple times
   * until the shouldContinue callback returns false. This is useful for:
   * 1. Iterative algorithms that converge over multiple passes
   * 2. Benchmarking with multiple runs
   * 3. Processing streaming data in batches
   * 
   * @param optimizedGraph The optimized computation graph to execute
   * @param executeRandomTask Callback function to execute specific tasks
   * @param shouldContinue Function that determines when to stop execution
   * @param numIterations Output parameter counting iterations performed
   * @param runningTime Output parameter for recording execution time
   * @param memManager Reference to the MemoryManager instance to use
   */
  void executeOptimizedGraphRepeatedly(
    OptimizationOutput &optimizedGraph,
    ExecuteRandomTask executeRandomTask,
    ShouldContinue shouldContinue,
    int &numIterations,
    float &runningTime,
    MemoryManager &memManager
  );
  
  /**
   * @brief Executes an optimized computation graph repeatedly using singleton MemoryManager
   * 
   * Similar to executeOptimizedGraph, but runs the graph multiple times
   * until the shouldContinue callback returns false. This version uses the
   * singleton MemoryManager instance for storage management.
   * 
   * @param optimizedGraph The optimized computation graph to execute
   * @param executeRandomTask Callback function to execute specific tasks
   * @param shouldContinue Function that determines when to stop execution
   * @param numIterations Output parameter counting iterations performed
   * @param runningTime Output parameter for recording execution time
   */
  void executeOptimizedGraphRepeatedly(
    OptimizationOutput &optimizedGraph,
    ExecuteRandomTask executeRandomTask,
    ShouldContinue shouldContinue,
    int &numIterations,
    float &runningTime
  );
  
  /**
   * @brief Executes an optimized computation graph repeatedly with external storage map
   * 
   * Similar to executeOptimizedGraph, but runs the graph multiple times
   * until the shouldContinue callback returns false. This is useful for:
   * 1. Iterative algorithms that converge over multiple passes
   * 2. Benchmarking with multiple runs
   * 3. Processing streaming data in batches
   * 
   * @param optimizedGraph The optimized computation graph to execute
   * @param executeRandomTask Callback function to execute specific tasks
   * @param shouldContinue Function that determines when to stop execution
   * @param numIterations Output parameter counting iterations performed
   * @param runningTime Output parameter for recording execution time
   * @param managedDeviceArrayToHostArrayMap Mapping between device and storage memory addresses
   */
  void executeOptimizedGraphRepeatedly(
    OptimizationOutput &optimizedGraph,
    ExecuteRandomTask executeRandomTask,
    ShouldContinue shouldContinue,
    int &numIterations,
    float &runningTime,
    std::map<void *, void *> &managedDeviceArrayToHostArrayMap
  );
  
  /**
   * @brief Executes an optimized computation graph for evaluation with provided MemoryManager
   * 
   * This method is a variant of executeOptimizedGraph specifically designed for evaluation purposes.
   * It follows the same basic flow but may include additional instrumentation or logging.
   * 
   * @param optimizedGraph The optimized computation graph to execute
   * @param executeRandomTask Callback function to execute specific tasks
   * @param runningTime Output parameter for recording execution time
   * @param memManager Reference to the MemoryManager instance to use
   */
  void executeOptimizedGraphForEvaluation(
    OptimizationOutput &optimizedGraph,
    ExecuteRandomTask executeRandomTask,
    float &runningTime,
    MemoryManager &memManager
  );
  
  /**
   * @brief Executes an optimized computation graph for evaluation using singleton MemoryManager
   * 
   * This is a convenience wrapper that uses the singleton MemoryManager instance.
   * 
   * @param optimizedGraph The optimized computation graph to execute
   * @param executeRandomTask Callback function to execute specific tasks
   * @param runningTime Output parameter for recording execution time
   */
  void executeOptimizedGraphForEvaluation(
    OptimizationOutput &optimizedGraph,
    ExecuteRandomTask executeRandomTask,
    float &runningTime
  );
  
  /**
   * @brief Executes an optimized computation graph for evaluation with simplified callback
   * 
   * This overload accepts an ExecuteRandomTaskBase callback that doesn't require address mapping.
   * 
   * @param optimizedGraph The optimized computation graph to execute
   * @param executeRandomTaskBase Simplified callback function to execute specific tasks
   * @param runningTime Output parameter for recording execution time
   * @param memManager Reference to the MemoryManager instance to use
   */
  void executeOptimizedGraphForEvaluation(
    OptimizationOutput &optimizedGraph,
    ExecuteRandomTaskBase executeRandomTaskBase,
    float &runningTime,
    MemoryManager &memManager
  );
  
  /**
   * @brief Executes an optimized computation graph for evaluation with singleton MemoryManager and simplified callback
   * 
   * This version uses the singleton MemoryManager instance and a simplified callback.
   * 
   * @param optimizedGraph The optimized computation graph to execute
   * @param executeRandomTaskBase Simplified callback function to execute specific tasks
   * @param runningTime Output parameter for recording execution time
   */
  void executeOptimizedGraphForEvaluation(
    OptimizationOutput &optimizedGraph,
    ExecuteRandomTaskBase executeRandomTaskBase,
    float &runningTime
  );

  /**
   * @brief Executes an optimized computation graph with cold start pattern (CPU→GPU→CPU)
   * 
   * This method implements a cold start execution pattern where:
   * 1. All managed arrays start in CPU/storage (cold state)
   * 2. Data is transferred to GPU as needed during optimized execution
   * 3. Final results are transferred back to CPU/storage (return to cold state)
   * 
   * This pattern is useful for batch processing where GPU resources are shared
   * and data should not remain on GPU between executions.
   * 
   * @param optimizedGraph The optimized computation graph to execute
   * @param executeRandomTask Callback function to execute specific tasks
   * @param runningTime Output parameter for recording execution time
   * @param memManager Reference to the MemoryManager instance to use
   */
  void executeOptimizedGraphColdStart(
    OptimizationOutput &optimizedGraph,
    ExecuteRandomTask executeRandomTask,
    float &runningTime,
    MemoryManager &memManager
  );

  /**
   * @brief Executes an optimized computation graph with cold start pattern using singleton MemoryManager
   * 
   * This version uses the singleton MemoryManager instance for cold start execution.
   * 
   * @param optimizedGraph The optimized computation graph to execute
   * @param executeRandomTask Callback function to execute specific tasks
   * @param runningTime Output parameter for recording execution time
   */
  void executeOptimizedGraphColdStart(
    OptimizationOutput &optimizedGraph,
    ExecuteRandomTask executeRandomTask,
    float &runningTime
  );

 protected:
  /**
   * @brief Helper method for initializing data distribution
   * 
   * Moves all managed memory to storage and initializes needed data on device.
   * 
   * @param optimizedGraph The optimization plan to execute
   * @param mainDeviceId ID of the main computation device
   * @param storageDeviceId ID of the storage device
   * @param useNvlink Whether to use NVLink for data transfers
   * @param managedDeviceArrayToHostArrayMap Output map for device-to-storage mappings
   * @param stream CUDA stream to use for operations
   * @return cudaGraphExec_t Executable graph for initial data setup
   */
  cudaGraphExec_t initializeDataDistribution(
    OptimizationOutput &optimizedGraph,
    int mainDeviceId, 
    int storageDeviceId,
    bool useNvlink,
    std::map<void *, void *> &managedDeviceArrayToHostArrayMap,
    cudaStream_t stream,
    cudaMemcpyKind prefetchMemcpyKind
  );
  
  /**
   * @brief Helper method for initializing data distribution using internal storage
   * 
   * Moves all managed memory to storage and initializes needed data on device.
   * Uses MemoryManager's internal storage map.
   * 
   * @param optimizedGraph The optimization plan to execute
   * @param mainDeviceId ID of the main computation device
   * @param storageDeviceId ID of the storage device
   * @param useNvlink Whether to use NVLink for data transfers
   * @param stream CUDA stream to use for operations
   * @return cudaGraphExec_t Executable graph for initial data setup
   */
  cudaGraphExec_t initializeDataDistribution(
    OptimizationOutput &optimizedGraph,
    int mainDeviceId, 
    int storageDeviceId,
    bool useNvlink,
    cudaStream_t stream,
    cudaMemcpyKind prefetchMemcpyKind
  );

  /**
   * @brief Default constructor (protected to enforce singleton pattern)
   */
  Executor() = default;
  
  /**
   * @brief Gets nodes with no outgoing edges in a CUDA graph
   * 
   * This helper function identifies all nodes in the graph that have
   * no outgoing edges (leaf nodes), which can be used as dependencies
   * for subsequent operations.
   * 
   * @param graph The CUDA graph to analyze
   * @return Vector of nodes with zero out-degree
   */
  static std::vector<cudaGraphNode_t> getNodesWithZeroOutDegree(cudaGraph_t graph);
  
  /**
   * @brief Singleton instance pointer
   */
  static Executor *instance;
};

}  // namespace memopt