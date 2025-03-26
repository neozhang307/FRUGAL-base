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
// #include "../optimization/taskManager_v2.hpp"

using namespace memopt;

const std::string INPUT_MATRIX_FILE_PATH = "tiledCholeskyInputMatrix.in";

// Implementation Note: This file has been updated to use TaskManager_v2 for all operations.
// The algorithm records a CUDA graph by executing tasks through TaskManager_v2, which can then
// be executed directly or with memory optimization.

size_t N; // total matrix size (N×N)
size_t B; // batch size in 1d (B×B per tile)
size_t T; // tile amount in 1d (T×T tiles)

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

  // Use pinned memory for h_tiles for faster data transfer
  std::vector<double*> h_tiles(t * t, nullptr);
  for (int i = 0; i < t * t; i++) {
    checkCudaErrors(cudaMallocHost(&h_tiles[i], b * b * sizeof(double)));
    checkCudaErrors(cudaMemcpy(h_tiles[i], MemoryManager::getInstance().getAddress(d_tiles[i]), B * B * sizeof(double), cudaMemcpyDefault));
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

  // Use pinned memory for firstRow as well
  double* firstRow = nullptr;
  checkCudaErrors(cudaMallocHost(&firstRow, rowLength * sizeof(double)));
  memset(firstRow, 0, rowLength * sizeof(double));
  
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
  
  // Free pinned memory for h_tiles and firstRow
  for (int i = 0; i < t * t; i++) {
    checkCudaErrors(cudaFreeHost(h_tiles[i]));
  }
  checkCudaErrors(cudaFreeHost(firstRow));

  return error <= 1e-6;
}

// Define a matrix tile as a pair of coordinates (i,j)
typedef std::pair<int, int> MatrixTile;

void initializeHostData(double *h_originalMatrix) {
  generateRandomSymmetricPositiveDefiniteMatrix(h_originalMatrix, N);
}

void initializeDeviceData(double *h_originalMatrix, std::vector<double *> &d_tiles) {
  for (int i = 0; i < T; i++) {
    for (int j = 0; j < T; j++) {
      for (int k = 0; k < B; k++) {
        checkCudaErrors(cudaMemcpy(
          MemoryManager::getAddress(d_tiles[i + j * T]) + B * k,
          h_originalMatrix + N * (j * B + k) + B * i,
          B * sizeof(double),
          cudaMemcpyDefault
        ));
      }
    }
  }
  checkCudaErrors(cudaDeviceSynchronize());
}

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
  // Allocate and initialize the input matrix using pinned memory for faster host-device transfers
  clock.logWithCurrentTime("Initialzing host data");
  
  // Use pinned memory (cudaMallocHost) instead of pageable memory for better transfer performance
  double* h_originalMatrix = nullptr;
  checkCudaErrors(cudaMallocHost(&h_originalMatrix, N * N * sizeof(double)));  // Column-major

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
    initializeHostData(h_originalMatrix);
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
    // Free allocated memory before return
    checkCudaErrors(cudaFreeHost(h_originalMatrix));
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
  {
    for (int j = 0; j < T; j++)
    {
      MemoryManager::getInstance().registerManagedMemoryAddress(getMatrixBlock(i, j), tileSize);
      MemoryManager::getInstance().registerApplicationInput(getMatrixBlock(i, j));
      // MemoryManager::getInstance().registerApplicaftionOutput(getMatrixBlock(i, j));
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

  // Create TaskManager_v2 with debug mode enabled
  TaskManager_v2 tmanager_v2(true);

  // SECTION 6: TILED CHOLESKY ALGORITHM IMPLEMENTATION
  // Begin recording algorithm operations to CUDA graph
  // checkCudaErrors(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  auto tiledCholeskyGraphCreator = std::make_unique<TiledCholeskyGraphCreator>(s, graph);

  for (int k = 0; k < T; k++) {
    // =====================================================================
    // STEP 1: POTRF - Cholesky factorization of diagonal tile
    // A[k][k] = POTRF(A[k][k])
    // L[k][k] = POTRF(A[k][k])
    // =====================================================================
    
    // Register POTRF task using TaskManager_v2
    tiledCholeskyGraphCreator->beginCaptureOperation(
      {k, k},         // Tile to write
      {{k, k}}        // Tiles to read
    );

    TaskId taskId_v2 = tmanager_v2.registerTask<std::function<void(cudaStream_t, double*)>, double*>(
      [cusolverDnHandle, cusolverDnParams, d_workspace, workspaceInBytesOnDevice, 
       h_workspace, workspaceInBytesOnHost, d_info](cudaStream_t stream, double* matrixblock_k_k) {
        checkCudaErrors(cusolverDnSetStream(cusolverDnHandle, stream));

        checkCudaErrors(cusolverDnXpotrf(
            cusolverDnHandle,
            cusolverDnParams,
            CUBLAS_FILL_MODE_LOWER,  // Lower triangular (for Cholesky)
            B,               // Matrix size
            CUDA_R_64F,              // Data type (double)
            matrixblock_k_k,         // Input/output matrix
            B,               // Leading dimension
            CUDA_R_64F,              // Output data type
            d_workspace,             // Device workspace
            workspaceInBytesOnDevice,
            h_workspace,             // Host workspace
            workspaceInBytesOnHost,
            d_info                   // Error information
          ));
      },
      {getMatrixBlock(k, k)},  // inputs
      {getMatrixBlock(k, k)},  // outputs
      TaskManager_v2::makeArgs(getMatrixBlock(k, k)),  // default args
      "POTRF_task"  // task name
    );
    
    // Execute the task using TaskManager_v2
    tmanager_v2.execute(taskId_v2, s);
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
      
      // Register TRSM task using TaskManager_v2
      TaskId trsmTaskId = tmanager_v2.registerTask<std::function<void(cudaStream_t, double*, double*)>, double*, double*>(
        [cublasHandle, one](cudaStream_t stream, double* matrixblock_k_k, double* matrixblock_i_k) {
          checkCudaErrors(cublasSetStream(cublasHandle, stream));
          checkCudaErrors(cublasDtrsm(
            cublasHandle,
            CUBLAS_SIDE_RIGHT,      // Multiply from right
            CUBLAS_FILL_MODE_LOWER, // Lower triangular
            CUBLAS_OP_T,            // Transpose
            CUBLAS_DIAG_NON_UNIT,   // Non-unit diagonal
            B, B,   // Matrix dimensions
            one,                    // Alpha = 1.0
            matrixblock_k_k, B, // Triangular matrix
            matrixblock_i_k, B  // Input/output matrix
          ));
        },
        {getMatrixBlock(k, k), getMatrixBlock(i, k)},  // inputs
        {getMatrixBlock(i, k)},  // outputs
        TaskManager_v2::makeArgs(getMatrixBlock(k, k), getMatrixBlock(i, k)),  // default args
        "TRSM_task_" + std::to_string(i) + "_" + std::to_string(k)  // task name
      );
      
      // Execute the task using TaskManager_v2
      tmanager_v2.execute(trsmTaskId, s);
      tiledCholeskyGraphCreator->endCaptureOperation();
    }

    // =====================================================================
    // STEP 3 & 4: SYRK and GEMM - Update remaining submatrix
    // =====================================================================
    for (int i = k + 1; i < T; i++) {
      tiledCholeskyGraphCreator->beginCaptureOperation(
        {i, i},              // Tile to write
        {{i, i}, {i, k}}     // Tiles to read
      );
      // STEP 3: SYRK - Update diagonal tiles
      // A[i][i] = SYRK(A[i][k], A[i][i])
      // A[i][i] = A[i][i] - L[i][k] * L[i][k]^T
      
      // Register SYRK task using TaskManager_v2
      TaskId syrkTaskId = tmanager_v2.registerTask<std::function<void(cudaStream_t, double*, double*)>, double*, double*>(
        [cublasHandle, minusOne, one](cudaStream_t stream, double* matrixblock_i_k, double* matrixblock_i_i) {
          checkCudaErrors(cublasSetStream(cublasHandle, stream));
          checkCudaErrors(cublasDsyrk(
            cublasHandle,
            CUBLAS_FILL_MODE_LOWER, // Lower triangular
            CUBLAS_OP_N,           // No transpose
            B, B,                  // Matrix dimensions
            minusOne,              // Alpha = -1.0
            matrixblock_i_k, B, // Input matrix
            one,                   // Beta = 1.0
            matrixblock_i_i, B  // Input/output matrix
          ));
        },
        {getMatrixBlock(i, i), getMatrixBlock(i, k)},  // inputs
        {getMatrixBlock(i, i)},  // outputs
        TaskManager_v2::makeArgs(getMatrixBlock(i, k), getMatrixBlock(i, i)),  // default args
        "SYRK_task_" + std::to_string(i) + "_" + std::to_string(k)  // task name
      );
      

      // Execute the task using TaskManager_v2
      tmanager_v2.execute(syrkTaskId, s);
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
        
        // Register GEMM task using TaskManager_v2
        TaskId gemmTaskId = tmanager_v2.registerTask<std::function<void(cudaStream_t, double*, double*, double*)>, double*, double*, double*>(
          [cublasHandle, minusOne, one](cudaStream_t stream, double* matrixblock_j_k, double* matrixblock_i_k, double* matrixblock_j_i) {
            // General matrix multiplication using cuBLAS
            checkCudaErrors(cublasSetStream(cublasHandle, stream));
            checkCudaErrors(cublasGemmEx(
              cublasHandle,
              CUBLAS_OP_N,           // No transpose for first matrix
              CUBLAS_OP_T,           // Transpose second matrix
              B, B, B,               // Matrix dimensions
              minusOne,              // Alpha = -1.0
              matrixblock_j_k, CUDA_R_64F, B, // First input matrix
              matrixblock_i_k, CUDA_R_64F, B, // Second input matrix
              one,                   // Beta = 1.0
              matrixblock_j_i, CUDA_R_64F, B, // Input/output matrix
              CUBLAS_COMPUTE_64F,    // Computation precision
              CUBLAS_GEMM_DEFAULT    // Algorithm selection
            ));
          },
          {getMatrixBlock(j, i), getMatrixBlock(j, k), getMatrixBlock(i, k)},  // inputs
          {getMatrixBlock(j, i)},  // outputs
          TaskManager_v2::makeArgs(getMatrixBlock(j, k), getMatrixBlock(i, k), getMatrixBlock(j, i)),  // default args
          "GEMM_task_" + std::to_string(j) + "_" + std::to_string(i) + "_" + std::to_string(k)  // task name
        );
        
   
        // Execute the task using TaskManager_v2
        tmanager_v2.execute(gemmTaskId, s);
        tiledCholeskyGraphCreator->endCaptureOperation();
      }
    }
  }
  // Finish recording operations to CUDA graph
  // checkCudaErrors(cudaStreamEndCapture(s, &graph));
  // checkCudaErrors(cudaStreamDestroy(s));

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
  tmanager_v2.setExecutionMode(TaskManager_v2::ExecutionMode::Production);
  if (ConfigurationManager::getConfig().tiledCholesky.mode
      == Configuration::TiledCholesky::Mode::readInputMatrixFromFileAndRunBeyondDeviceCapacity) {
    // Optimize the CUDA graph for memory usage
    auto optimizedGraph = profileAndOptimize(graph);

    // Initialize device data from the host matrix
    initializeDeviceData(h_originalMatrix, d_tiles);

    // Execute the optimized graph with memory management
    float runningTime;
    auto& memManager = MemoryManager::getInstance();
     
    executeOptimizedGraph(
      optimizedGraph,
      // Create a lambda that matches the ExecuteRandomTaskBase signature
      [&tmanager_v2](int taskId, cudaStream_t stream) {
        // Execute the task using TaskManager_v2
        tmanager_v2.execute(taskId, stream);
        return true; // Indicates successful execution
      },
      runningTime,
      memManager
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
      initializeDeviceData(h_originalMatrix, d_tiles);

      // Execute optimized graph with memory management
      float runningTime;
      auto& memManager = MemoryManager::getInstance();
      
      executeOptimizedGraph(
        optimizedGraph,
        // Create a lambda that matches the ExecuteRandomTaskBase signature
        [&tmanager_v2](int taskId, cudaStream_t stream) {
          // Execute the task using TaskManager_v2
          tmanager_v2.execute(taskId, stream);
          return true; // Indicates successful execution
        },
        runningTime,
        memManager
      );
      checkCudaErrors(cudaDeviceSynchronize());
      fmt::print("Total time used (s): {}\n", runningTime);
      memManager.prefetchAllDataToDevice();
      checkCudaErrors(cudaDeviceSynchronize());
      fmt::print("Finalized iteration\n");
      
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
      initializeDeviceData(h_originalMatrix, d_tiles);

      // Create a new stream for execution
      cudaStream_t execStream;
      checkCudaErrors(cudaStreamCreate(&execStream));
      
      // Unified Memory prefetching (if enabled)
      if (ConfigurationManager::getConfig().generic.useUM) {
        // Limit available memory for unified memory
        size_t available = 1024ULL * 1024ULL * ConfigurationManager::getConfig().generic.availableMemoryForUMInMiB;
        reduceAvailableMemoryForUM(available);

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
            execStream
          ));
        }
        checkCudaErrors(cudaStreamSynchronize(execStream));
      }

      // Optional memory usage profiling
      if (ConfigurationManager::getConfig().execution.measurePeakMemoryUsage) {
        peakMemoryUsageProfiler.start();
      }
      
      // Execute and time the graph
      cudaEventClock.start();
      checkCudaErrors(cudaGraphLaunch(graphExec, execStream));
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
      fmt::print("Total time used (s): {}\n", cudaEventClock.getTimeInSeconds());

      // Reset memory limit for unified memory
      if (ConfigurationManager::getConfig().generic.useUM) {
        resetAvailableMemoryForUM();
      }
      
      // Clean up the execution stream
      checkCudaErrors(cudaStreamDestroy(execStream));
    }
  }

  clock.logWithCurrentTime("Synchronization done");

  // SECTION 8: RESULT VERIFICATION
  // Optionally verify the correctness of the Cholesky factorization
  if (verify) {
    clock.logWithCurrentTime("Start verification");
    fmt::print("Result passes verification: {}\n", verifyCholeskyDecompositionPartially(h_originalMatrix, d_tiles, N, B));
    clock.logWithCurrentTime("Verification done");
  }

  clock.logWithCurrentTime("All finished");

  // SECTION 9: CLEANUP
  // Free all allocated resources
  checkCudaErrors(cudaFreeHost(h_workspace));
  checkCudaErrors(cudaFree(d_workspace));
  checkCudaErrors(cudaFreeHost(h_originalMatrix));  // Free the pinned host memory
  
  // Use freeManagedMemory to free both device and storage memory for each tile
  auto& memManager = MemoryManager::getInstance();
  for (auto d_tile : d_tiles) {
    memManager.freeManagedMemory(d_tile);
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
