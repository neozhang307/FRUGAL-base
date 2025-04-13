#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <cusolverDn.h>
#include <fmt/core.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <vector>

#include "../include/argh.h"
#include "memopt.hpp"

using namespace memopt;

const std::string INPUT_MATRIX_FILE_PATH = "tiledCholeskyInputMatrix.in";

size_t N; //
size_t B; // batch size in 1d
size_t T; // tile amount in 1d

__global__ void makeMatrixSymmetric(double *d_matrix, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t x = idx / n;
  size_t y = idx % n;

  if (x >= y || x >= n || y >= n) {
    return;
  }

  double average = 0.5 * (d_matrix[x * n + y] + d_matrix[y * n + x]);
  d_matrix[x * n + y] = average;
  d_matrix[y * n + x] = average;
}

__global__ void addIdenticalMatrix(double *d_matrix, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  d_matrix[idx * n + idx] += n;
}

// Credit to: https://math.stackexchange.com/questions/357980/how-to-generate-random-symmetric-positive-definite-matrices-using-matlab
void generateRandomSymmetricPositiveDefiniteMatrix(double *h_A, const size_t n) {
  double *d_A;
  checkCudaErrors(cudaMalloc(&d_A, n * n * sizeof(double)));

  // Generate random matrix d_A
  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
  curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());
  curandGenerateUniformDouble(prng, d_A, n * n);

  // d_A = (d_A + d_A^T) / 2
  size_t numThreads = 1024;
  size_t numBlocks = (N * N + numThreads - 1) / numThreads;
  makeMatrixSymmetric<<<numBlocks, numThreads>>>(d_A, N);

  // d_A = d_A + n * I
  numThreads = 1024;
  numBlocks = (N + numThreads - 1) / numThreads;
  addIdenticalMatrix<<<numBlocks, numThreads>>>(d_A, N);

  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaMemcpy(h_A, d_A, n * n * sizeof(double), cudaMemcpyDefault));

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaFree(d_A));
}

// Only verify the last row of L * L^T = A
bool verifyCholeskyDecompositionPartially(double *A, std::vector<double *> &d_tiles, const size_t n, const size_t b) {
  const size_t t = n / b;

  std::vector<std::unique_ptr<double[]>> h_tiles;
  for (int i = 0; i < t * t; i++) {
    h_tiles.push_back(std::move(std::make_unique<double[]>(b * b)));
    checkCudaErrors(cudaMemcpy(h_tiles[i].get(), d_tiles[i], B * B * sizeof(double), cudaMemcpyDefault));
    checkCudaErrors(cudaDeviceSynchronize());
  }

  auto getAEntry = [&](size_t row, size_t col) {
    return A[row + col * n];
  };

  auto getLEntry = [&](size_t row, size_t col) {
    if (row < col) {
      return static_cast<double>(0);
    }
    const size_t i = row / b;
    const size_t k = row - (i * b);
    const size_t j = col / b;
    const size_t l = col - (j * b);

    return h_tiles[i + j * t][k + l * b];
  };

  // Only check the last row;
  const size_t rowIndex = n - 1;

  const size_t rowLength = min((size_t)1024, n);

  auto firstRow = std::make_unique<double[]>(rowLength);
  memset(firstRow.get(), 0, rowLength * sizeof(double));
  for (int j = 0; j < rowLength; j++) {
    for (int k = 0; k < n; k++) {
      firstRow[j] += getLEntry(rowIndex, k) * getLEntry(j, k);
    }
  }

  double error = 0;
  for (int j = 0; j < rowLength; j++) {
    error += fabs(getAEntry(rowIndex, j) - firstRow[j]);
  }

  fmt::print("error = {:.6f}\n", error);

  return error <= 1e-6;
}

typedef std::pair<int, int> MatrixTile;

class TiledCholeskyGraphCreator {
 public:
  TiledCholeskyGraphCreator(cudaStream_t stream, cudaGraph_t graph) : stream(stream), graph(graph) {
    this->lastModifiedTile = {-1, -1};
  }

  void beginCaptureOperation(MatrixTile tileToWrite, std::initializer_list<MatrixTile> tilesToRead) {
    auto tiles = std::vector<MatrixTile>(tilesToRead);
    tiles.push_back(tileToWrite);
    auto dependencies = this->getDependencies(tiles);

    this->lastModifiedTile = tileToWrite;
    this->lastDependencies = dependencies;

    checkCudaErrors(cudaStreamBeginCaptureToGraph(this->stream, this->graph, dependencies.data(), nullptr, dependencies.size(), cudaStreamCaptureModeGlobal));
  }

  void endCaptureOperation() {
    assert(this->lastModifiedTile.first != -1 && this->lastModifiedTile.second != -1);
    checkCudaErrors(cudaStreamEndCapture(this->stream, &this->graph));
    this->tileLastModifiedByMap[this->lastModifiedTile] = this->getTailOfLastCapturedNodeChain();
    this->lastModifiedTile = {-1, -1};
  };

 private:
  std::map<MatrixTile, cudaGraphNode_t> tileLastModifiedByMap;
  std::map<cudaGraphNode_t, bool> visited;
  cudaStream_t stream;
  cudaGraph_t graph;
  MatrixTile lastModifiedTile;
  std::vector<cudaGraphNode_t> lastDependencies;

  std::vector<cudaGraphNode_t> getDependencies(std::vector<MatrixTile> tiles) {
    std::vector<cudaGraphNode_t> dependencies;
    for (auto tile : tiles) {
      auto it = this->tileLastModifiedByMap.find(tile);
      if (it != this->tileLastModifiedByMap.end()) {
        dependencies.push_back(it->second);
      }
    }

    auto dedupedEnd = std::unique(dependencies.begin(), dependencies.end());
    dependencies.resize(std::distance(dependencies.begin(), dedupedEnd));
    return dependencies;
  }

  cudaGraphNode_t getTailOfLastCapturedNodeChain() {
    if (lastDependencies.size() == 0) {
      size_t numEdges;
      checkCudaErrors(cudaGraphGetEdges(this->graph, nullptr, nullptr, &numEdges));
      auto from = std::make_unique<cudaGraphNode_t[]>(numEdges);
      auto to = std::make_unique<cudaGraphNode_t[]>(numEdges);
      checkCudaErrors(cudaGraphGetEdges(this->graph, from.get(), to.get(), &numEdges));

      std::map<cudaGraphNode_t, bool> hasOutGoingEdge;
      std::set<cudaGraphNode_t> noOutGoingEdgeNodes;
      for (int i = 0; i < numEdges; i++) {
        hasOutGoingEdge[from[i]] = true;
        noOutGoingEdgeNodes.erase(from[i]);
        if (!hasOutGoingEdge[to[i]])
          noOutGoingEdgeNodes.insert(to[i]);
      }

      assert(noOutGoingEdgeNodes.size() == 1);

      return *noOutGoingEdgeNodes.begin();
    } else {
      auto nodeBeforeChain = lastDependencies[0];
      size_t numDependentNodes;
      checkCudaErrors(cudaGraphNodeGetDependentNodes(nodeBeforeChain, nullptr, &numDependentNodes));

      assert(numDependentNodes > 0);

      auto dependentNodes = std::make_unique<cudaGraphNode_t[]>(numDependentNodes);
      checkCudaErrors(cudaGraphNodeGetDependentNodes(nodeBeforeChain, dependentNodes.get(), &numDependentNodes));

      cudaGraphNode_t chainBeginningNode;
      for (int i = 0; i < numDependentNodes; i++) {
        if (!visited[dependentNodes[i]]) {
          chainBeginningNode = dependentNodes[i];
          break;
        }
      }

      auto u = chainBeginningNode;
      while (true) {
        visited[u] = true;
        checkCudaErrors(cudaGraphNodeGetDependentNodes(u, nullptr, &numDependentNodes));
        if (numDependentNodes == 0) break;

        assert(numDependentNodes == 1);

        cudaGraphNode_t v;
        checkCudaErrors(cudaGraphNodeGetDependentNodes(u, &v, &numDependentNodes));
        u = v;
      }

      return u;
    }
  }
};

class TiledCholeskyTaskManager {
 public:
  struct Task {
    enum class OperationType {
      portf,  // a = PORTF(a)
      trsm,   // a = a * b^T
      syrk,   // a = a - b * b^T
      gemm    // a = a - b * c^T
    };

    OperationType operation;
    MatrixTile a, b, c;
  };

  TiledCholeskyTaskManager(
    cusolverDnHandle_t cusolverDnHandle,
    cusolverDnParams_t cusolverDnParams,
    cublasHandle_t cublasHandle,
    size_t workspaceInBytesOnDevice,
    size_t workspaceInBytesOnHost,
    void *h_workspace,
    void *d_workspace,
    int *d_info,
    double *one,
    double *minusOne
  ) {
    this->cusolverDnHandle = cusolverDnHandle;
    this->cusolverDnParams = cusolverDnParams;
    this->cublasHandle = cublasHandle;
    this->workspaceInBytesOnDevice = workspaceInBytesOnDevice;
    this->workspaceInBytesOnHost = workspaceInBytesOnHost;
    this->h_workspace = h_workspace;
    this->d_workspace = d_workspace;
    this->d_info = d_info;
    this->one = one;
    this->minusOne = minusOne;
  }

  int addTask(
    Task::OperationType operation,
    MatrixTile a,
    MatrixTile b = {0, 0},
    MatrixTile c = {0, 0}
  ) {
    Task t;
    t.operation = operation;
    t.a = a;
    t.b = b;
    t.c = c;

    this->tasks.push_back(t);

    return this->tasks.size() - 1;
  }

  void executeRandomTask(std::function<double *(int, int)> getMatrixBlock, int taskId, std::map<void *, void *> addressUpdate, cudaStream_t stream) {
    const auto &task = this->tasks[taskId];

    checkCudaErrors(cusolverDnSetStream(cusolverDnHandle, stream));
    checkCudaErrors(cublasSetStream(cublasHandle, stream));

    if (task.operation == Task::OperationType::portf) {
      checkCudaErrors(cusolverDnXpotrf(
        cusolverDnHandle,
        cusolverDnParams,
        CUBLAS_FILL_MODE_LOWER,
        B,
        CUDA_R_64F,
        tryGettingUpdatedAddress(addressUpdate, getMatrixBlock(task.a.first, task.a.second)),
        B,
        CUDA_R_64F,
        d_workspace,
        workspaceInBytesOnDevice,
        h_workspace,
        workspaceInBytesOnHost,
        d_info
      ));
    } else if (task.operation == Task::OperationType::trsm) {
      checkCudaErrors(cublasDtrsm(
        cublasHandle,
        CUBLAS_SIDE_RIGHT,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_T,
        CUBLAS_DIAG_NON_UNIT,
        B, B,
        one,
        tryGettingUpdatedAddress(addressUpdate, getMatrixBlock(task.b.first, task.b.second)), B,
        tryGettingUpdatedAddress(addressUpdate, getMatrixBlock(task.a.first, task.a.second)), B
      ));
    } else if (task.operation == Task::OperationType::syrk) {
      checkCudaErrors(cublasDsyrk(
        cublasHandle,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,
        B, B,
        minusOne, tryGettingUpdatedAddress(addressUpdate, getMatrixBlock(task.b.first, task.b.second)), B,
        one, tryGettingUpdatedAddress(addressUpdate, getMatrixBlock(task.a.first, task.a.second)), B
      ));
    } else if (task.operation == Task::OperationType::gemm) {
      checkCudaErrors(cublasGemmEx(
        cublasHandle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        B, B, B,
        minusOne,
        tryGettingUpdatedAddress(addressUpdate, getMatrixBlock(task.b.first, task.b.second)), CUDA_R_64F, B,
        tryGettingUpdatedAddress(addressUpdate, getMatrixBlock(task.c.first, task.c.second)), CUDA_R_64F, B,
        one,
        tryGettingUpdatedAddress(addressUpdate, getMatrixBlock(task.a.first, task.a.second)), CUDA_R_64F, B,
        CUBLAS_COMPUTE_64F,
        CUBLAS_GEMM_DEFAULT
      ));
    }
  }

 private:
  std::vector<Task> tasks;

  cusolverDnHandle_t cusolverDnHandle;
  cusolverDnParams_t cusolverDnParams;
  cublasHandle_t cublasHandle;
  size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
  void *h_workspace, *d_workspace;
  int *d_info;
  double *one, *minusOne;

  template <typename T>
  T *tryGettingUpdatedAddress(std::map<void *, void *> &addressUpdate, T *oldAddress) {
    auto it = addressUpdate.find(static_cast<void *>(oldAddress));
    if (it != addressUpdate.end()) {
      return static_cast<T *>(it->second);
    }
    return oldAddress;
  }
};

void initializeHostData(double *h_originalMatrix) {
  generateRandomSymmetricPositiveDefiniteMatrix(h_originalMatrix, N);
}

void initializeDeviceData(double *h_originalMatrix, std::vector<double *> &d_tiles) {
  for (int i = 0; i < T; i++) {
    for (int j = 0; j < T; j++) {
      for (int k = 0; k < B; k++) {
        checkCudaErrors(cudaMemcpy(
          d_tiles[i + j * T] + B * k,
          h_originalMatrix + N * (j * B + k) + B * i,
          B * sizeof(double),
          cudaMemcpyDefault
        ));
      }
    }
  }
  checkCudaErrors(cudaDeviceSynchronize());
}


void initializeHostData(double *h_originalMatrix, std::vector<double *> &d_tiles) {
  for (int i = 0; i < T; i++) {
    for (int j = 0; j < T; j++) {
      for (int k = 0; k < B; k++) {
        memcpy(d_tiles[i + j * T] + B * k,h_originalMatrix + N * (j * B + k) + B * i,B * sizeof(double));
        // checkCudaErrors(cudaMemcpy(
        //   d_tiles[i + j * T] + B * k,
        //   h_originalMatrix + N * (j * B + k) + B * i,
        //   B * sizeof(double),
        //   cudaMemcpyDefault
        // ));
      }
    }
  }
  checkCudaErrors(cudaDeviceSynchronize());
}

/*
 * TILED CHOLESKY DECOMPOSITION ALGORITHM
 * 
 * OVERVIEW:
 * This function implements a tiled Cholesky decomposition for large matrices on GPUs.
 * It decomposes a symmetric positive-definite matrix A into a product L×L^T
 * where L is a lower triangular matrix.
 * 
 * The implementation uses a tiled approach where an N×N matrix is divided into
 * T×T tiles, each of size B×B (where N = B×T). This improves memory locality
 * and enables processing of matrices larger than GPU memory.
 * 
 * ALGORITHM STEPS:
 * For each tile-column k = 0 to T-1:
 *   1. POTRF: Factor diagonal tile A[k,k] = L[k,k]×L[k,k]^T
 *   2. TRSM: For each tile below diagonal (i > k), solve L[i,k]×L[k,k]^T = A[i,k]
 *   3. SYRK: For each future diagonal tile (i > k), update A[i,i] = A[i,i] - L[i,k]×L[i,k]^T
 *   4. GEMM: For remaining tiles (j > i > k), update A[j,i] = A[j,i] - L[j,k]×L[i,k]^T
 * 
 * KEY COMPONENTS:
 * - CUDA Graph Construction: Records operations with dependencies
 * - Task Management: Tracks operations for replay during optimization
 * - Memory Management: Handles data placement and movement
 * - Multiple Execution Paths: Standard, optimized, and beyond-device-capacity modes
 * 
 * EXECUTION MODES:
 * 1. Standard Execution: Direct CUDA graph execution
 * 2. Optimized Execution: With memory optimization (prefetching/offloading)
 * 3. Beyond-Device-Capacity: For matrices exceeding GPU memory
 * 
 * PERFORMANCE FEATURES:
 * - High-performance GPU libraries (cuSOLVER, cuBLAS)
 * - Parallelism through independent tile operations
 * - Memory optimization through CUDA graphs
 * - Optional performance profiling
 */
void tiledCholesky(bool optimize, bool verify) {
  // Configuration check for beyond-device-capacity mode
  // When matrix is too large for GPU memory, we need both optimization and a pre-computed plan
  if (ConfigurationManager::getConfig().tiledCholesky.mode
      == Configuration::TiledCholesky::Mode::readInputMatrixFromFileAndRunBeyondDeviceCapacity) {
    if (!ConfigurationManager::getConfig().generic.optimize
        || !ConfigurationManager::getConfig().optimization.loadExistingPlan) {
      LOG_TRACE_WITH_INFO("Must enable optimization and load existing plan when problem size is beyond device memory capacity");
      exit(-1);
    }
  }

  // Initialize timing and CUDA device
  SystemWallClock clock;
  clock.start();

  initializeCudaDevice();

  const size_t tileSize = B * B * sizeof(double);

  // SECTION 1: HOST DATA INITIALIZATION
  // Allocate and initialize the input matrix - either generated or loaded from file
  clock.logWithCurrentTime("Initialzing host data");
  auto h_originalMatrix = std::make_unique<double[]>(N * N);  // Column-major

  if (ConfigurationManager::getConfig().tiledCholesky.mode
      == Configuration::TiledCholesky::Mode::readInputMatrixFromFileAndRunBeyondDeviceCapacity) {
    clock.logWithCurrentTime("Loading input matrix from file");
    std::ifstream fin(INPUT_MATRIX_FILE_PATH);
    std::string s;
    int i = 0;
    while (std::getline(fin, s)) {
      h_originalMatrix[i++] = std::stod(s);
    }
    clock.logWithCurrentTime("Input matrix loaded");
  } else {
    initializeHostData(h_originalMatrix.get());
  }

  clock.logWithCurrentTime("Host data initialized");

  if (ConfigurationManager::getConfig().tiledCholesky.mode
      == Configuration::TiledCholesky::Mode::dumpInputMatrixToFile) {
    clock.logWithCurrentTime("Dumping input matrix");
    std::ofstream fout(INPUT_MATRIX_FILE_PATH);
    fout << std::setprecision(10);
    for (size_t i = 0; i < N * N; i++) {
      fout << h_originalMatrix[i] << '\n';
    }
    clock.logWithCurrentTime("Input matrix dumped");
    return;
  }

  // SECTION 2: DEVICE MEMORY ALLOCATION
  // Allocate GPU memory for each tile - either standard device memory or unified memory
  clock.logWithCurrentTime("Initialzing device data");
  std::vector<double *> d_tiles(T * T);
  for (int i = 0; i < T * T; i++) {
    if (ConfigurationManager::getConfig().generic.useUM
        || ConfigurationManager::getConfig().tiledCholesky.mode == Configuration::TiledCholesky::Mode::readInputMatrixFromFileAndRunBeyondDeviceCapacity) {
      // Unified Memory allows easier management of data that exceeds GPU memory
      checkCudaErrors(cudaMallocManaged(&d_tiles[i], tileSize));
    } else {
      // Standard device memory for better performance when data fits in GPU
      checkCudaErrors(cudaMalloc(&d_tiles[i], tileSize));
    }
  }

  // Data will be initialized later before execution
  // initializeDeviceData(h_originalMatrix.get(), d_tiles);

  clock.logWithCurrentTime("Device data initialized");

  // Helper function to access tiles by logical indices
  auto getMatrixBlock = [&](int i, int j) {
    return d_tiles[i + j * T];
  };

  // SECTION 3: MEMORY REGISTRATION
  // Register memory addresses with the memory optimization framework
  // This allows the framework to track and manage memory during optimization
  
  // Register all tile addresses in the memory optimization system
  for (int i = 0; i < T; i++)
    for (int j = 0; j < T; j++)
      registerManagedMemoryAddress(getMatrixBlock(i, j), tileSize);

  // Register which tiles are inputs and outputs for the application
  // Only the lower triangular portion is relevant for Cholesky
  for (int i = 0; i < T; i++) {
    for (int j = 0; j <= i; j++) {
      registerApplicationInput(getMatrixBlock(i, j));
      registerApplicationOutput(getMatrixBlock(i, j));
    }
  }
  clock.logWithCurrentTime("Addresses registered");

  // SECTION 4: CUDA LIBRARY INITIALIZATION
  // Initialize CUDA libraries for linear algebra operations
  cusolverDnHandle_t cusolverDnHandle;  // For Cholesky factorization (POTRF)
  cusolverDnParams_t cusolverDnParams;
  cublasHandle_t cublasHandle;          // For matrix operations (TRSM, SYRK, GEMM)
  checkCudaErrors(cusolverDnCreate(&cusolverDnHandle));
  checkCudaErrors(cusolverDnCreateParams(&cusolverDnParams));
  checkCudaErrors(cublasCreate(&cublasHandle));

  // Constants used in matrix operations
  double *one, *minusOne;
  checkCudaErrors(cudaMallocManaged(&one, sizeof(double)));
  checkCudaErrors(cudaMallocManaged(&minusOne, sizeof(double)));
  *one = 1.0;
  *minusOne = -1.0;

  // Allocate workspace memory for Cholesky factorization (POTRF)
  size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
  checkCudaErrors(cusolverDnXpotrf_bufferSize(
    cusolverDnHandle,
    cusolverDnParams,
    CUBLAS_FILL_MODE_LOWER,
    B,
    CUDA_R_64F,
    d_tiles[0],
    B,
    CUDA_R_64F,
    &workspaceInBytesOnDevice,
    &workspaceInBytesOnHost
  ));
  void *h_workspace, *d_workspace;
  int *d_info;
  checkCudaErrors(cudaMallocHost(&h_workspace, workspaceInBytesOnHost));
  checkCudaErrors(cudaMalloc(&d_workspace, workspaceInBytesOnDevice));
  checkCudaErrors(cudaMallocManaged(&d_info, sizeof(int)));

  // SECTION 5: CUDA GRAPH CREATION
  // Create a CUDA graph to record all operations with dependencies
  cudaGraph_t graph;
  checkCudaErrors(cudaGraphCreate(&graph, 0));

  // Create a stream for asynchronous execution
  cudaStream_t s;
  checkCudaErrors(cudaStreamCreate(&s));

  // Set streams for CUDA libraries
  checkCudaErrors(cusolverDnSetStream(cusolverDnHandle, s));
  checkCudaErrors(cublasSetStream(cublasHandle, s));

  clock.logWithCurrentTime("Preparation done, start to record graph");

  // Initialize the graph creator to manage operation dependencies
  auto tiledCholeskyGraphCreator = std::make_unique<TiledCholeskyGraphCreator>(s, graph);

  // Initialize the task manager for tracking operations
  auto tiledCholeskyTaskManager = std::make_unique<TiledCholeskyTaskManager>(
    cusolverDnHandle,
    cusolverDnParams,
    cublasHandle,
    workspaceInBytesOnDevice,
    workspaceInBytesOnHost,
    h_workspace,
    d_workspace,
    d_info,
    one,
    minusOne
  );

  // SECTION 6: TILED CHOLESKY ALGORITHM IMPLEMENTATION
  // Records the entire algorithm as a CUDA graph with dependencies
  
  int nextTaskId;
  for (int k = 0; k < T; k++) {
    // =====================================================================
    // STEP 1: POTRF - Cholesky factorization of diagonal tile
    // A[k][k] = POTRF(A[k][k])
    // L[k][k] = POTRF(A[k][k])
    // =====================================================================
    
    // Begin graph capture with dependencies
    tiledCholeskyGraphCreator->beginCaptureOperation(
      {k, k},         // Tile to write
      {{k, k}}        // Tiles to read
    );
    
    // Record task for optimization
    nextTaskId = tiledCholeskyTaskManager->addTask(TiledCholeskyTaskManager::Task::OperationType::portf, {k, k});
    annotateNextTask(nextTaskId, {getMatrixBlock(k, k)}, {getMatrixBlock(k, k)}, s);
    
    // Actual computation using cuSOLVER
    checkCudaErrors(cusolverDnXpotrf(
      cusolverDnHandle,
      cusolverDnParams,
      CUBLAS_FILL_MODE_LOWER,  // Lower triangular (for Cholesky)
      B,                       // Matrix size
      CUDA_R_64F,              // Data type (double)
      getMatrixBlock(k, k),    // Input/output matrix
      B,                       // Leading dimension
      CUDA_R_64F,              // Output data type
      d_workspace,             // Device workspace
      workspaceInBytesOnDevice,
      h_workspace,             // Host workspace
      workspaceInBytesOnHost,
      d_info                   // Error information
    ));
    
    // End graph capture
    tiledCholeskyGraphCreator->endCaptureOperation();

    // =====================================================================
    // STEP 2: TRSM - Triangular solve for tiles below diagonal
    // For each tile below diagonal (i > k)
    // =====================================================================
    for (int i = k + 1; i < T; i++) {
      // A[i][k] = TRSM(A[k][k], A[i][k])
      // L[i][k] * L[k][k]^T = A[i][k]
      // Solving for L[i][k]
      
      tiledCholeskyGraphCreator->beginCaptureOperation(
        {i, k},               // Tile to write
        {{k, k}, {i, k}}      // Tiles to read
      );
      
      nextTaskId = tiledCholeskyTaskManager->addTask(TiledCholeskyTaskManager::Task::OperationType::trsm, {i, k}, {k, k});
      annotateNextTask(nextTaskId, {getMatrixBlock(i, k), getMatrixBlock(k, k)}, {getMatrixBlock(i, k)}, s);
      
      // Triangular solve using cuBLAS
      checkCudaErrors(cublasDtrsm(
        cublasHandle,
        CUBLAS_SIDE_RIGHT,     // Multiply from right
        CUBLAS_FILL_MODE_LOWER, // Lower triangular
        CUBLAS_OP_T,           // Transpose
        CUBLAS_DIAG_NON_UNIT,  // Non-unit diagonal
        B, B,                  // Matrix dimensions
        one,                   // Alpha = 1.0
        getMatrixBlock(k, k), B, // Triangular matrix
        getMatrixBlock(i, k), B  // Input/output matrix
      ));
      
      tiledCholeskyGraphCreator->endCaptureOperation();
    }

    // =====================================================================
    // STEP 3 & 4: SYRK and GEMM - Update remaining submatrix
    // =====================================================================
    for (int i = k + 1; i < T; i++) {
      // STEP 3: SYRK - Update diagonal tiles
      // A[i][i] = SYRK(A[i][k], A[i][i])
      // A[i][i] = A[i][i] - L[i][k] * L[i][k]^T
      
      tiledCholeskyGraphCreator->beginCaptureOperation(
        {i, i},              // Tile to write
        {{i, i}, {i, k}}     // Tiles to read
      );
      
      nextTaskId = tiledCholeskyTaskManager->addTask(TiledCholeskyTaskManager::Task::OperationType::syrk, {i, i}, {i, k});
      annotateNextTask(nextTaskId, {getMatrixBlock(i, i), getMatrixBlock(i, k)}, {getMatrixBlock(i, i)}, s);
      
      // Symmetric rank-k update using cuBLAS
      checkCudaErrors(cublasDsyrk(
        cublasHandle,
        CUBLAS_FILL_MODE_LOWER, // Lower triangular
        CUBLAS_OP_N,           // No transpose
        B, B,                  // Matrix dimensions
        minusOne,              // Alpha = -1.0
        getMatrixBlock(i, k), B, // Input matrix
        one,                   // Beta = 1.0
        getMatrixBlock(i, i), B  // Input/output matrix
      ));
      
      tiledCholeskyGraphCreator->endCaptureOperation();

      // STEP 4: GEMM - Update off-diagonal tiles
      // For each tile below diagonal in column i
      for (int j = i + 1; j < T; j++) {
        // A[j][i] = GEMM(A[j][k], A[i][k])
        // A[j][i] = A[j][i] - L[j][k] * L[i][k]^T
        
        tiledCholeskyGraphCreator->beginCaptureOperation(
          {j, i},                     // Tile to write
          {{j, i}, {j, k}, {i, k}}    // Tiles to read
        );
        
        nextTaskId = tiledCholeskyTaskManager->addTask(TiledCholeskyTaskManager::Task::OperationType::gemm, {j, i}, {j, k}, {i, k});
        annotateNextTask(nextTaskId, {getMatrixBlock(j, i), getMatrixBlock(j, k), getMatrixBlock(i, k)}, {getMatrixBlock(j, i)}, s);
        
        // General matrix multiplication using cuBLAS
        checkCudaErrors(cublasGemmEx(
          cublasHandle,
          CUBLAS_OP_N,           // No transpose for first matrix
          CUBLAS_OP_T,           // Transpose second matrix
          B, B, B,               // Matrix dimensions
          minusOne,              // Alpha = -1.0
          getMatrixBlock(j, k), CUDA_R_64F, B, // First input matrix
          getMatrixBlock(i, k), CUDA_R_64F, B, // Second input matrix
          one,                   // Beta = 1.0
          getMatrixBlock(j, i), CUDA_R_64F, B, // Input/output matrix
          CUBLAS_COMPUTE_64F,    // Computation precision
          CUBLAS_GEMM_DEFAULT    // Algorithm selection
        ));
        
        tiledCholeskyGraphCreator->endCaptureOperation();
      }
    }
  }

  // Graph creation completed - now we can export it for debugging
  clock.logWithCurrentTime("Graph recorded");

  LOG_TRACE_WITH_INFO("Printing original graph to graph.dot");
  checkCudaErrors(cudaGraphDebugDotPrint(graph, "./graph.dot", 0));

  clock.logWithCurrentTime("Graph printed");

  // SECTION 7: EXECUTION PATHS
  // The function has three different execution paths depending on configuration:
  // 1. Beyond-device-capacity mode
  // 2. Optimized execution mode
  // 3. Standard execution mode

  // =====================================================================
  // EXECUTION PATH 1: Beyond-device-capacity mode
  // For matrices that exceed GPU memory capacity
  // =====================================================================
  if (ConfigurationManager::getConfig().tiledCholesky.mode
      == Configuration::TiledCholesky::Mode::readInputMatrixFromFileAndRunBeyondDeviceCapacity) {
    // Optimize the CUDA graph for memory usage
    auto optimizedGraph = profileAndOptimize(graph);

    // Initialize device data from the host matrix
    initializeDeviceData(h_originalMatrix.get(), d_tiles);

    // Execute the optimized graph with memory management
    float runningTime;
    std::map<void *, void *> managedDeviceArrayToHostArrayMap;
    executeOptimizedGraph(
      optimizedGraph,
      // Callback for executing specific tasks
      [&](int taskId, std::map<void *, void *> addressUpdate, cudaStream_t stream) {
        tiledCholeskyTaskManager->executeRandomTask(getMatrixBlock, taskId, addressUpdate, stream);
      },
      runningTime,
      managedDeviceArrayToHostArrayMap
    );

    checkCudaErrors(cudaDeviceSynchronize());
    fmt::print("Total time used (s): {}\n", runningTime);

    return;
  }

  // =====================================================================
  // EXECUTION PATH 2: Optimized execution mode
  // Uses memory optimization framework to reduce memory usage
  // =====================================================================
  if (optimize) {
    // Create an optimized version of the graph
    auto optimizedGraph = profileAndOptimize(graph);

    // Run the optimized graph for each repetition
    for (int i = 0; i < ConfigurationManager::getConfig().generic.repeat; i++) {
      // Initialize device data from host matrix
      initializeDeviceData(h_originalMatrix.get(), d_tiles);

      // Execute optimized graph with memory management
      float runningTime;
      std::map<void *, void *> managedDeviceArrayToHostArrayMap;
      executeOptimizedGraph(
        optimizedGraph,
        [&](int taskId, std::map<void *, void *> addressUpdate, cudaStream_t stream) {
          tiledCholeskyTaskManager->executeRandomTask(getMatrixBlock, taskId, addressUpdate, stream);
        },
        runningTime,
        managedDeviceArrayToHostArrayMap
      );

      checkCudaErrors(cudaDeviceSynchronize());

      // Reset memory status after optimization run
      // For simplicity, all data copy back to device memory
      std::map<void *, void *> oldManagedDeviceArrayToNewManagedDeviceArrayMap;
      for (int j = 0; j < T * T; j++) {
        auto oldPtr = d_tiles[j];
        auto newPtr = managedDeviceArrayToHostArrayMap[oldPtr];
        // Copy data back to device memory
        checkCudaErrors(cudaMalloc(&d_tiles[j], tileSize));
        checkCudaErrors(cudaMemcpy(d_tiles[j], newPtr, tileSize, cudaMemcpyDefault));
        // Free temporary memory
        if (ConfigurationManager::getConfig().execution.useNvlink) {
          checkCudaErrors(cudaFree(newPtr));
        } else {
          checkCudaErrors(cudaFreeHost(newPtr));
        }
        oldManagedDeviceArrayToNewManagedDeviceArrayMap[oldPtr] = d_tiles[j];
      }

      checkCudaErrors(cudaDeviceSynchronize());

      // Update memory manager with new addresses
      updateManagedMemoryAddress(oldManagedDeviceArrayToNewManagedDeviceArrayMap);

      fmt::print("Total time used (s): {}\n", runningTime);
      // double gflops_pdpotrf = 1.0 / 6.0 * ((double) N * (double) N * (double) N) / (1000000000.0);
      double gflops_pdpotrf = 1.0 / 6.0 * (((double)T*(double)T*(double)T+(double)T) *(double)B*(double)B*(double)B) / (1000000000.0);
      std::cout << "[PDPOTRF] ELAPSED: " << runningTime
            << " s, GFLOPS: " << gflops_pdpotrf / (runningTime) << std::endl;
    }
  }
  // =====================================================================
  // EXECUTION PATH 3: Standard execution mode
  // Direct CUDA graph execution without optimization
  // =====================================================================
  else {
    
    // For memory usage tracking
    PeakMemoryUsageProfiler peakMemoryUsageProfiler;
    CudaEventClock cudaEventClock;
    
    // Instantiate the CUDA graph for execution
    cudaGraphExec_t graphExec;
    checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    clock.logWithCurrentTime("Graph instantiated, start execution");

    // Run for specified number of repetitions
    for (int i = 0; i < ConfigurationManager::getConfig().generic.repeat; i++) {
      // Initialize device data from host matrix
      initializeHostData(h_originalMatrix.get(), d_tiles);
      double* ptrwast;
      cudaMalloc(&ptrwast,(long)1024*1024*1024*20);

      // Unified Memory prefetching (if enabled)
      if (ConfigurationManager::getConfig().generic.useUM) {
        // Limit available memory for unified memory
        size_t available = 1024ULL * 1024ULL * ConfigurationManager::getConfig().generic.availableMemoryForUMInMiB;
        // reduceAvailableMemoryForUM(available);

        // Prefetch data to device (up to available memory limit)
        size_t sum = 0;
        for (int i = 0; i < T * T; i++) {
          if (sum + (tileSize) > available) {
            break;
          }
          checkCudaErrors(cudaMemPrefetchAsync(
            d_tiles[i],
            tileSize,
            ConfigurationManager::getConfig().execution.mainDeviceId,
            s
          ));
        }
        checkCudaErrors(cudaStreamSynchronize(s));
      }

      // Optional memory usage profiling
      if (ConfigurationManager::getConfig().execution.measurePeakMemoryUsage) {
        peakMemoryUsageProfiler.start();
      }

      // Execute and time the graph
      cudaEventClock.start();
      checkCudaErrors(cudaGraphLaunch(graphExec, s));
      cudaEventClock.end();

      checkCudaErrors(cudaDeviceSynchronize());

      // Report peak memory usage if measured
      if (ConfigurationManager::getConfig().execution.measurePeakMemoryUsage) {
        const auto peakMemoryUsage = peakMemoryUsageProfiler.end();
        fmt::print(
          "Peak memory usage (MiB): {:.2f}\n",
          static_cast<float>(peakMemoryUsage) / 1024.0 / 1024.0
        );
      }

      // Report execution time
      auto runningTime=cudaEventClock.getTimeInSeconds();
      fmt::print("Total time used (s): {}\n", cudaEventClock.getTimeInSeconds());
      double gflops_pdpotrf = 1.0 / 6.0 * (((double)T*(double)T*(double)T+(double)T) *(double)B*(double)B*(double)B) / (1000000000.0);
      std::cout << "[PDPOTRF] ELAPSED: " << runningTime
            << " s, GFLOPS: " << gflops_pdpotrf / (runningTime) << std::endl;

      // Reset memory limit for unified memory
      if (ConfigurationManager::getConfig().generic.useUM) {
        // resetAvailableMemoryForUM();
      }
    }
  }

  clock.logWithCurrentTime("Synchronization done");

  // SECTION 8: RESULT VERIFICATION
  // Optionally verify the correctness of the Cholesky factorization
  if (verify) {
    clock.logWithCurrentTime("Start verification");
    fmt::print("Result passes verification: {}\n", verifyCholeskyDecompositionPartially(h_originalMatrix.get(), d_tiles, N, B));
    clock.logWithCurrentTime("Verification done");
  }

  clock.logWithCurrentTime("All finished");

  // SECTION 9: CLEANUP
  // Free all allocated resources
  checkCudaErrors(cudaFreeHost(h_workspace));
  checkCudaErrors(cudaFree(d_workspace));
  for (auto d_tile : d_tiles) {
    checkCudaErrors(cudaFree(d_tile));
  }
}

int main(int argc, char **argv) {
  auto cmdl = argh::parser(argc, argv);
  std::string configFilePath;
  cmdl("configFile", "config.json") >> configFilePath;

  ConfigurationManager::exportDefaultConfiguration();
  ConfigurationManager::loadConfiguration(configFilePath);

  N = ConfigurationManager::getConfig().tiledCholesky.n;
  T = ConfigurationManager::getConfig().tiledCholesky.t;
  B = N / T;

  tiledCholesky(
    ConfigurationManager::getConfig().generic.optimize,
    ConfigurationManager::getConfig().generic.verify
  );

  return 0;
}
