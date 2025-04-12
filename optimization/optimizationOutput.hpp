#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "../include/json.hpp"
#include "../utilities/types.hpp"

namespace memopt {

/**
 * @brief OptimizationOutput represents an optimized CUDA computation graph
 * 
 * This structure defines a directed acyclic graph (DAG) that represents the optimized 
 * execution plan for a CUDA computation, including both computational tasks and the 
 * data movement operations necessary to manage GPU memory efficiently.
 * 
 * The optimization enables execution of computations that exceed available GPU memory
 * by intelligently scheduling when data should be moved between the GPU and storage
 * (either host memory or a secondary GPU connected via NVLink).
 */
struct OptimizationOutput {
  /**
   * @brief Types of nodes in the computation graph
   */
  enum class NodeType {
    empty = 0,        ///< Empty node used for dependency management
    task,             ///< Computational task node that executes on the GPU
    dataMovement      ///< Data movement operation between GPU and storage
  };

  /**
   * @brief Defines a data movement operation
   * 
   * Represents the transfer of data between the GPU and storage (host or secondary GPU)
   */
  struct DataMovement {
    /**
     * @brief Direction of data movement
     */
    enum class Direction {
      hostToDevice = 0,  ///< Prefetch data from storage to GPU (before use)
      deviceToHost       ///< Offload data from GPU to storage (after use)
    };
    Direction direction;  ///< Direction of the data movement
    ArrayId arrayId;      ///< Identifier of the data array being moved
  };

  /**
   * @brief Indicates if the graph represents an optimal solution
   * 
   * When true, the memory optimization algorithm found an optimal solution.
   * When false, the solution may be a heuristic approximation.
   */
  bool optimal;
  
  /**
   * @brief Original memory usage before optimization (in MiB)
   * 
   * The peak memory usage of the original computation graph before any optimization
   */
  double originalMemoryUsage;
  
  /**
   * @brief Anticipated peak memory usage after optimization (in MiB)
   * 
   * The expected peak memory usage after applying the optimization strategy
   */
  double anticipatedPeakMemoryUsage;

  /**
   * @brief List of all node IDs in the computation graph
   */
  std::vector<int> nodes;
  
  /**
   * @brief Adjacency list representation of the graph edges
   * 
   * Maps a node ID to a list of its successor nodes, defining execution order
   */
  std::map<int, std::vector<int>> edges;
  
  /**
   * @brief Maps node IDs to their types (task, dataMovement, empty)
   */
  std::map<int, NodeType> nodeIdToNodeTypeMap;
  
  /**
   * @brief Maps node IDs to task identifiers (for task nodes)
   */
  std::map<int, TaskId> nodeIdToTaskIdMap;
  
  /**
   * @brief Maps node IDs to data movement details (for dataMovement nodes)
   */
  std::map<int, DataMovement> nodeIdToDataMovementMap;

  /**
   * @brief Data arrays that should be on the GPU at the start of execution
   * 
   * These arrays are prefetched to the GPU before the main computation begins
   */
  std::vector<ArrayId> arraysInitiallyAllocatedOnDevice;

  /**
   * @brief Adds an empty node to the graph
   * 
   * Empty nodes are used primarily for dependency management
   * 
   * @return The ID of the newly created node
   */
  int addEmptyNode() {
    auto u = this->nodes.size();
    this->nodes.push_back(u);
    this->nodeIdToNodeTypeMap[u] = NodeType::empty;
    return u;
  }

  /**
   * @brief Adds a computational task node to the graph
   * 
   * @param taskId The task identifier for the computation
   * @return The ID of the newly created node
   */
  int addTaskNode(TaskId taskId) {
    auto u = this->addEmptyNode();
    this->nodeIdToNodeTypeMap[u] = NodeType::task;
    this->nodeIdToTaskIdMap[u] = taskId;
    return u;
  }

  /**
   * @brief Adds a data movement node to the graph
   * 
   * @param dataMovement The data movement operation details
   * @return The ID of the newly created node
   */
  int addDataMovementNode(DataMovement dataMovement) {
    auto u = this->addEmptyNode();
    this->nodeIdToNodeTypeMap[u] = NodeType::dataMovement;
    this->nodeIdToDataMovementMap[u] = dataMovement;
    return u;
  }

  /**
   * @brief Adds a data movement node with connected edges
   * 
   * This is a convenience method that creates a data movement node
   * and adds edges from the start node to the movement node and
   * from the movement node to the end node.
   * 
   * @param direction Direction of data transfer (hostToDevice or deviceToHost)
   * @param arrayId Identifier of the data array being moved
   * @param start Node ID of the operation preceding the data movement
   * @param end Node ID of the operation following the data movement
   * @return The ID of the newly created node
   */
  int addDataMovementNode(DataMovement::Direction direction, ArrayId arrayId, int start, int end) {
    DataMovement dataMovement;
    dataMovement.direction = direction;
    dataMovement.arrayId = arrayId;

    auto u = this->addDataMovementNode(dataMovement);
    this->addEdge(start, u);
    this->addEdge(u, end);

    return u;
  }

  /**
   * @brief Adds a directed edge between two nodes
   * 
   * Edges define the execution order and dependencies in the graph
   * 
   * @param from The source node ID
   * @param to The destination node ID
   */
  void addEdge(int from, int to) {
    this->edges[from].push_back(to);
  }
};

}  // namespace memopt
