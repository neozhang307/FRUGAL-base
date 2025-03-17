#include <cuda.h>

#include <cassert>
#include <map>
#include <memory>
#include <vector>

#include "cudaGraphUtilities.hpp"
#include "cudaUtilities.hpp"

namespace memopt {

/**
 * Extracts nodes and edges from a CUDA graph into convenient C++ containers
 * 
 * This function transforms the raw CUDA graph structure into more accessible
 * C++ data structures for easier analysis and manipulation. It populates:
 * 1. A vector containing all nodes in the graph
 * 2. An adjacency list representation of the graph's edges
 * 
 * The adjacency list maps each source node to a vector of its destination nodes,
 * which makes graph traversal and analysis algorithms much simpler to implement.
 * 
 * @param graph  The CUDA graph to analyze
 * @param nodes  Output vector that will be filled with all graph nodes
 * @param edges  Output map representing an adjacency list of graph edges
 */
void extractGraphNodesAndEdges(
  cudaGraph_t graph,
  std::vector<cudaGraphNode_t> &nodes,
  std::map<cudaGraphNode_t, std::vector<cudaGraphNode_t>> &edges
) {
  // Step 1: Query the size of the graph (number of nodes and edges)
  // This is a common CUDA pattern: first call to get counts, second call to get data
  size_t numNodes, numEdges;
  checkCudaErrors(cudaGraphGetNodes(graph, nullptr, &numNodes));
  checkCudaErrors(cudaGraphGetEdges(graph, nullptr, nullptr, &numEdges));
  
  // Step 2: Allocate memory for temporary arrays to hold the CUDA graph data
  // Using smart pointers for automatic cleanup when function exits
  auto rawNodes = std::make_unique<cudaGraphNode_t[]>(numNodes);  // Array for all nodes
  auto from = std::make_unique<cudaGraphNode_t[]>(numEdges);      // Array for edge source nodes
  auto to = std::make_unique<cudaGraphNode_t[]>(numEdges);        // Array for edge destination nodes
  
  // Step 3: Actually retrieve the graph data from CUDA
  checkCudaErrors(cudaGraphGetNodes(graph, rawNodes.get(), &numNodes));
  // CUDA stores edges as parallel arrays: from[i]->to[i] represents one edge
  checkCudaErrors(cudaGraphGetEdges(graph, from.get(), to.get(), &numEdges));

  // Step 4: Transfer nodes to the output vector
  nodes.clear();  // Ensure the output vector is empty
  for (int i = 0; i < numNodes; i++) {
    nodes.push_back(rawNodes[i]);
  }

  // Step 5: Build an adjacency list representation of the graph edges
  // This maps each node to a vector of all its direct successors
  edges.clear();  // Ensure the output map is empty
  for (int i = 0; i < numEdges; i++) {
    // For each edge, add the destination node to the source node's adjacency list
    // If this is the first edge for this source node, a new vector is automatically created
    edges[from[i]].push_back(to[i]);
  }
  // Note: Nodes with no outgoing edges will have empty vectors in the map
  // or might not appear as keys in the map at all (depending on std::map implementation)
}

/**
 * Retrieves the root node of a CUDA graph
 * 
 * This function identifies and returns the root node of a CUDA graph. In CUDA graph terminology,
 * a root node is one that has no incoming edges (no predecessors). The function assumes that
 * the graph has exactly one root node, which is the typical case for most CUDA applications
 * where graphs represent a directed flow of computation.
 * 
 * The implementation uses the standard CUDA pattern of:
 * 1. First calling cudaGraphGetRootNodes with nullptr to get the count
 * 2. Allocating memory based on that count
 * 3. Calling again to retrieve the actual node(s)
 * 
 * An assertion ensures that the graph structure is as expected (single entry point).
 * 
 * @param graph The CUDA graph to analyze
 * @return The single root node of the graph
 * @throws Assertion error if the graph has multiple root nodes
 */
cudaGraphNode_t getRootNode(cudaGraph_t graph) {
  // Step 1: Query how many root nodes exist in the graph
  size_t numRootNodes;
  checkCudaErrors(cudaGraphGetRootNodes(graph, nullptr, &numRootNodes));
  
  // Verify our assumption that this graph has exactly one root node
  // This is important for traversal algorithms that need a single entry point
  assert(numRootNodes == 1);

  // Step 2: Allocate memory and retrieve the root node
  // Using smart pointer for automatic cleanup
  auto rootNodes = std::make_unique<cudaGraphNode_t[]>(numRootNodes);
  checkCudaErrors(cudaGraphGetRootNodes(graph, rootNodes.get(), &numRootNodes));
  
  // Return the single root node (safe because of our assertion)
  return rootNodes[0];
}

std::vector<cudaGraphNode_t> getNodesWithZeroOutDegree(cudaGraph_t graph) {
  std::vector<cudaGraphNode_t> nodesWithZeroOutDegree;

  std::vector<cudaGraphNode_t> nodes;
  std::map<cudaGraphNode_t, std::vector<cudaGraphNode_t>> edges;
  extractGraphNodesAndEdges(graph, nodes, edges);

  for (auto u : nodes) {
    if (edges[u].size() == 0) {
      nodesWithZeroOutDegree.push_back(u);
    }
  }

  return nodesWithZeroOutDegree;
}

cudaGraphNodeType getNodeType(cudaGraphNode_t kernelNode) {
  cudaGraphNodeType nodeType;
  checkCudaErrors(cudaGraphNodeGetType(kernelNode, &nodeType));
  return nodeType;
}

void getKernelNodeParams(cudaGraphNode_t kernelNode, CUDA_KERNEL_NODE_PARAMS &nodeParams) {
  cudaGraphNodeType nodeType = getNodeType(kernelNode);
  assert(nodeType == cudaGraphNodeTypeKernel);

  // Why switch to driver API:
  // https://forums.developer.nvidia.com/t/cuda-runtime-api-error-for-cuda-graph-and-opencv/215408/13
  checkCudaErrors(cuGraphKernelNodeGetParams(kernelNode, &nodeParams));
}

bool compareKernelNodeFunctionHandle(cudaGraphNode_t kernelNode, CUfunction functionHandle) {
  cudaGraphNodeType nodeType = getNodeType(kernelNode);
  if (nodeType == cudaGraphNodeTypeKernel) {
    CUDA_KERNEL_NODE_PARAMS nodeParams;
    getKernelNodeParams(kernelNode, nodeParams);

    if (nodeParams.func == functionHandle) {
      return true;
    }
  }
  return false;
}

}  // namespace memopt
