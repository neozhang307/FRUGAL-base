#pragma once

#include <cuda.h>

#include <map>
#include <vector>

namespace memopt {

void extractGraphNodesAndEdges(
  cudaGraph_t graph,
  std::vector<cudaGraphNode_t> &nodes,
  std::map<cudaGraphNode_t, std::vector<cudaGraphNode_t>> &edges
);

cudaGraphNode_t getRootNode(cudaGraph_t graph);

std::vector<cudaGraphNode_t> getNodesWithZeroOutDegree(cudaGraph_t graph);

cudaGraphNodeType getNodeType(cudaGraphNode_t kernelNode);

void getKernelNodeParams(cudaGraphNode_t kernelNode, CUDA_KERNEL_NODE_PARAMS &nodeParams);

bool compareKernelNodeFunctionHandle(cudaGraphNode_t kernelNode, CUfunction functionHandle);

/**
 * @brief Check if a CUDA graph node is an annotation node
 *
 * Annotation nodes are dummy kernels used to mark task boundaries and carry
 * metadata about the task (TaskId, inputs, outputs). This function compares
 * the kernel function handle to identify annotation nodes.
 *
 * Note: registerAnnotationKernelHandle() must be called before using this function.
 *
 * @param node The CUDA graph node to check
 * @return true if the node is an annotation kernel, false otherwise
 */
bool isAnnotationNode(cudaGraphNode_t node);

/**
 * @brief Register the annotation kernel handle for isAnnotationNode() checks
 *
 * @param handle The CUfunction handle of the annotation kernel
 */
void registerAnnotationKernelHandle(CUfunction handle);

}  // namespace memopt
