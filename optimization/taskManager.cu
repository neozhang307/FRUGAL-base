#include <cuda.h>

#include <cassert>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <queue>
#include <utility>
#include <vector>

#include "../utilities/cudaGraphUtilities.hpp"
#include "../utilities/cudaUtilities.hpp"
#include "../utilities/logger.hpp"
#include "optimizer.hpp"
#include "taskManager.hpp"

TaskManager *TaskManager::instance = nullptr;

TaskManager *TaskManager::getInstance() {
  if (instance == nullptr) {
    instance = new TaskManager();
  }
  return instance;
}

void TaskManager::registerDummyKernelHandle(cudaGraph_t graph) {
  auto rootNode = getRootNode(graph);

  CUgraphNodeType nodeType;
  checkCudaErrors(cuGraphNodeGetType(rootNode, &nodeType));
  assert(nodeType == CU_GRAPH_NODE_TYPE_KERNEL);

  CUDA_KERNEL_NODE_PARAMS rootNodeParams;
  checkCudaErrors(cuGraphKernelNodeGetParams(rootNode, &rootNodeParams));
  this->dummyKernelHandle = rootNodeParams.func;
}

CUfunction TaskManager::getDummyKernelHandle() {
  assert(this->dummyKernelHandle != nullptr);
  return this->dummyKernelHandle;
}

void TaskManager::initializeSequentialExecutionEnvironment() {
  checkCudaErrors(cudaStreamCreate(&(this->sequentialStream)));
}

void TaskManager::finalizeSequentialExecutionEnvironment() {
  checkCudaErrors(cudaStreamDestroy(this->sequentialStream));
}

bool TaskManager::executeNodeSequentially(CUgraphNode node) {
  CUgraphNodeType nodeType;
  checkCudaErrors(cuGraphNodeGetType(node, &nodeType));
  if (nodeType == CU_GRAPH_NODE_TYPE_KERNEL) {
    CUDA_KERNEL_NODE_PARAMS params;
    checkCudaErrors(cuGraphKernelNodeGetParams(node, &params));

    if (params.func == this->dummyKernelHandle) {
      return false;
    }

    CUlaunchConfig config;
    config.gridDimX = params.gridDimX;
    config.gridDimY = params.gridDimY;
    config.gridDimZ = params.gridDimZ;
    config.blockDimX = params.blockDimX;
    config.blockDimY = params.blockDimY;
    config.blockDimZ = params.blockDimZ;
    config.sharedMemBytes = params.sharedMemBytes;
    config.hStream = this->sequentialStream;

    // Currently kernel attributes are ignored
    config.attrs = NULL;
    config.numAttrs = 0;

    if (params.func != nullptr) {
      checkCudaErrors(cuLaunchKernelEx(
        &config,
        params.func,
        params.kernelParams,
        params.extra
      ));
    } else if (params.kern != nullptr) {
      checkCudaErrors(cuLaunchKernelEx(
        &config,
        reinterpret_cast<CUfunction>(params.kern),
        params.kernelParams,
        params.extra
      ));
    } else {
      LOG_TRACE_WITH_INFO("Currently only support params.func != NULL or params.kernel != NULL");
      exit(-1);
    }
  } else {
    LOG_TRACE_WITH_INFO("Unsupported node type: %d", nodeType);
    exit(-1);
  }

  return true;
}

Optimizer::CuGraphNodeToKernelDurationMap TaskManager::getCuGraphNodeToKernelDurationMap(cudaGraph_t graph) {
  std::vector<CUgraphNode> nodes;
  std::map<CUgraphNode, std::vector<CUgraphNode>> edges;
  extractGraphNodesAndEdges(graph, nodes, edges);

  std::map<CUgraphNode, int> inDegrees;
  for (auto &[u, outEdges] : edges) {
    for (auto &v : outEdges) {
      inDegrees[v] += 1;
    }
  }

  std::queue<CUgraphNode> nodesToExecute;
  for (auto &u : nodes) {
    if (inDegrees[u] == 0) {
      nodesToExecute.push(u);
    }
  }

  this->initializeSequentialExecutionEnvironment();

  // Kahn Algorithm
  Optimizer::CuGraphNodeToKernelDurationMap kernelRunningTimes;
  CudaEventClock clock;
  while (!nodesToExecute.empty()) {
    auto u = nodesToExecute.front();
    nodesToExecute.pop();

    clock.start(this->sequentialStream);
    auto isExecuted = this->executeNodeSequentially(u);
    clock.end(this->sequentialStream);
    checkCudaErrors(cudaStreamSynchronize(this->sequentialStream));
    if (isExecuted) {
      kernelRunningTimes[u] = clock.getTimeInSeconds();
    }

    for (auto &v : edges[u]) {
      inDegrees[v]--;
      if (inDegrees[v] == 0) {
        nodesToExecute.push(v);
      }
    }
  }

  this->finalizeSequentialExecutionEnvironment();

  return kernelRunningTimes;
}

void TaskManager::executeOptimizedGraph(const CustomGraph &optimizedGraph) {
}
