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

// Implementation Note: This file has been updated to use the generic 
// PointerDependencyCudaGraphConstructor instead of the application-specific
// TiledCholeskyGraphCreator. This demonstrates how to use memory pointer-based
// dependency tracking across different application domains.

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

/*
 * TILED CHOLESKY DECOMPOSITION ALGORITHM
 * 
 * This function implements a tiled Cholesky decomposition for large matrices on GPUs.
 * It decomposes a symmetric positive-definite matrix A into a product L×L^T
 * where L is a lower triangular matrix.
 * 
 * This implementation uses the generic PointerDependencyCudaGraphConstructor for dependency tracking.
 */
void tiledCholesky(bool optimize, bool verify) {
  // Configuration check for beyond-device-capacity mode
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
  clock.logWithCurrentTime("Initialzing host data");
  
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
    checkCudaErrors(cudaFreeHost(h_originalMatrix));
    return;
  }

  // SECTION 2: DEVICE MEMORY ALLOCATION
  clock.logWithCurrentTime("Initialzing device data");
  std::vector<double *> d_tiles(T * T);
  for (int i = 0; i < T * T; i++) {
    if (ConfigurationManager::getConfig().generic.useUM
        || ConfigurationManager::getConfig().tiledCholesky.mode == Configuration::TiledCholesky::Mode::readInputMatrixFromFileAndRunBeyondDeviceCapacity) {
      checkCudaErrors(cudaMallocManaged(&d_tiles[i], tileSize));
    } else {
      checkCudaErrors(cudaMalloc(&d_tiles[i], tileSize));
    }
  }

  clock.logWithCurrentTime("Device data initialized");

  // Helper function to access tiles by logical indices
  auto getMatrixBlock = [&](int i, int j) {
    return d_tiles[i + j * T];
  };

  // SECTION 3: MEMORY REGISTRATION
  for (int i = 0; i < T; i++) {
    for (int j = 0; j < T; j++) {
      MemoryManager::getInstance().registerManagedMemoryAddress(getMatrixBlock(i, j), tileSize);
      // MemoryManager::getInstance().registerApplicationInput(getMatrixBlock(i, j));
    }  
  }  
  
  double totalManagedMemoryMB = MemoryManager::getInstance().GetMemoryManagedSizeInMB();
  fmt::print("[MEMORY-INFO] Total managed memory size: {:.2f} MB\n", totalManagedMemoryMB);
  
  clock.logWithCurrentTime("Addresses registered");

  // SECTION 4: CUDA LIBRARY INITIALIZATION
  cusolverDnHandle_t cusolverDnHandle;
  cusolverDnParams_t cusolverDnParams;
  cublasHandle_t cublasHandle;
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
  cudaGraph_t graph;
  checkCudaErrors(cudaGraphCreate(&graph, 0));

  cudaStream_t s;
  checkCudaErrors(cudaStreamCreate(&s));

  checkCudaErrors(cusolverDnSetStream(cusolverDnHandle, s));
  checkCudaErrors(cublasSetStream(cublasHandle, s));

  clock.logWithCurrentTime("Preparation done, start to record graph");

  // Create TaskManager_v2 with debug mode enabled
  TaskManager_v2 tmanager_v2(true);

  // SECTION 6: TILED CHOLESKY ALGORITHM IMPLEMENTATION USING POINTER-BASED DEPENDENCY TRACKING
  auto graphConstructor = std::make_unique<PointerDependencyCudaGraphConstructor>(s, graph);

  for (int k = 0; k < T; k++) {
    // =====================================================================
    // STEP 1: POTRF - Cholesky factorization of diagonal tile
    // =====================================================================
    
    // Define input and output pointers for dependency tracking
    std::vector<void*> inputs = {static_cast<void*>(getMatrixBlock(k, k))};
    std::vector<void*> outputs = {static_cast<void*>(getMatrixBlock(k, k))};
    
    // Begin capturing operation with pointer-based dependencies
    graphConstructor->beginCaptureOperation(inputs, outputs);

    // Register and execute the POTRF task
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
    
    tmanager_v2.execute(taskId_v2, s);
    graphConstructor->endCaptureOperation();

    // =====================================================================
    // STEP 2: TRSM - Triangular solve for tiles below diagonal
    // =====================================================================
    for (int i = k + 1; i < T; i++) {
      // Define input and output pointers for dependency tracking
      inputs = {static_cast<void*>(getMatrixBlock(k, k)), static_cast<void*>(getMatrixBlock(i, k))};
      outputs = {static_cast<void*>(getMatrixBlock(i, k))};
      
      // Begin capturing operation with pointer-based dependencies
      graphConstructor->beginCaptureOperation(inputs, outputs);
      
      // Register and execute the TRSM task
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
        inputs,  // inputs
        outputs,  // outputs
        TaskManager_v2::makeArgs(getMatrixBlock(k, k), getMatrixBlock(i, k)),  // default args
        "TRSM_task_" + std::to_string(i) + "_" + std::to_string(k)  // task name
      );
      
      tmanager_v2.execute(trsmTaskId, s);
      graphConstructor->endCaptureOperation();
    }

    // =====================================================================
    // STEP 3: SYRK - Update diagonal tiles
    // =====================================================================
    for (int i = k + 1; i < T; i++) {
      // Define input and output pointers for dependency tracking
      inputs = {static_cast<void*>(getMatrixBlock(i, i)), static_cast<void*>(getMatrixBlock(i, k))};
      outputs = {static_cast<void*>(getMatrixBlock(i, i))};
      
      // Begin capturing operation with pointer-based dependencies
      graphConstructor->beginCaptureOperation(inputs, outputs);
      
      // Register and execute the SYRK task
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
        inputs,  // inputs
        outputs,  // outputs
        TaskManager_v2::makeArgs(getMatrixBlock(i, k), getMatrixBlock(i, i)),  // default args
        "SYRK_task_" + std::to_string(i) + "_" + std::to_string(k)  // task name
      );
      
      tmanager_v2.execute(syrkTaskId, s);
      graphConstructor->endCaptureOperation();

      // =====================================================================
      // STEP 4: GEMM - Update off-diagonal tiles
      // =====================================================================
      for (int j = i + 1; j < T; j++) {
        // Define input and output pointers for dependency tracking
        inputs = {
          static_cast<void*>(getMatrixBlock(j, i)), 
          static_cast<void*>(getMatrixBlock(j, k)), 
          static_cast<void*>(getMatrixBlock(i, k))
        };
        outputs = {static_cast<void*>(getMatrixBlock(j, i))};
        
        // Begin capturing operation with pointer-based dependencies
        graphConstructor->beginCaptureOperation(inputs, outputs);
        
        // Register and execute the GEMM task
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
          inputs,  // inputs
          outputs,  // outputs
          TaskManager_v2::makeArgs(getMatrixBlock(j, k), getMatrixBlock(i, k), getMatrixBlock(j, i)),  // default args
          "GEMM_task_" + std::to_string(j) + "_" + std::to_string(i) + "_" + std::to_string(k)  // task name
        );
        
        tmanager_v2.execute(gemmTaskId, s);
        graphConstructor->endCaptureOperation();
      }
    }
  }

  // Graph creation completed - now we can export it for debugging
  clock.logWithCurrentTime("Graph recorded");
  LOG_TRACE_WITH_INFO("Printing original graph to graph.dot");
  checkCudaErrors(cudaGraphDebugDotPrint(graph, "./graph.dot", 0));

  clock.logWithCurrentTime("Graph printed");

  // SECTION 7: EXECUTION PATHS
  // The function has three different execution paths depending on configuration

  // =====================================================================
  // EXECUTION PATH 1: Beyond-device-capacity mode
  // =====================================================================
  tmanager_v2.setExecutionMode(TaskManager_v2::ExecutionMode::Production);
  if (ConfigurationManager::getConfig().tiledCholesky.mode
      == Configuration::TiledCholesky::Mode::readInputMatrixFromFileAndRunBeyondDeviceCapacity) {
    auto optimizedGraph = profileAndOptimize(graph);

    initializeDeviceData(h_originalMatrix, d_tiles);

    float runningTime;
    auto& memManager = MemoryManager::getInstance();
     
    executeOptimizedGraph(
      optimizedGraph,
      [&tmanager_v2](int taskId, cudaStream_t stream) {
        tmanager_v2.execute(taskId, stream);
        return true;
      },
      runningTime,
      memManager
    );

    checkCudaErrors(cudaDeviceSynchronize());
    fmt::print("Total time used (s): {}\n", runningTime);

    // Cleanup
    checkCudaErrors(cudaFreeHost(h_workspace));
    checkCudaErrors(cudaFree(d_workspace));
    checkCudaErrors(cudaFreeHost(h_originalMatrix));
    
    for (auto d_tile : d_tiles) {
      memManager.freeManagedMemory(d_tile);
    }
    
    return;
  }

  // =====================================================================
  // EXECUTION PATH 2: Optimized execution mode
  // =====================================================================
  if (optimize) {
    auto optimizedGraph = profileAndOptimize(graph);

    for (int i = 0; i < ConfigurationManager::getConfig().generic.repeat; i++) {
      initializeDeviceData(h_originalMatrix, d_tiles);

      float runningTime;
      auto& memManager = MemoryManager::getInstance();
      
      executeOptimizedGraph(
        optimizedGraph,
        [&tmanager_v2](int taskId, cudaStream_t stream) {
          tmanager_v2.execute(taskId, stream);
          return true;
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
  // =====================================================================
  else {
    PeakMemoryUsageProfiler peakMemoryUsageProfiler;
    CudaEventClock cudaEventClock;
    
    cudaGraphExec_t graphExec;
    checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    clock.logWithCurrentTime("Graph instantiated, start execution");

    for (int i = 0; i < ConfigurationManager::getConfig().generic.repeat; i++) {
      initializeDeviceData(h_originalMatrix, d_tiles);

      cudaStream_t execStream;
      checkCudaErrors(cudaStreamCreate(&execStream));
      
      if (ConfigurationManager::getConfig().generic.useUM) {
        size_t available = 1024ULL * 1024ULL * ConfigurationManager::getConfig().generic.availableMemoryForUMInMiB;
        reduceAvailableMemoryForUM(available);

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

      if (ConfigurationManager::getConfig().execution.measurePeakMemoryUsage) {
        peakMemoryUsageProfiler.start();
      }
      
      cudaEventClock.start();
      checkCudaErrors(cudaGraphLaunch(graphExec, execStream));
      cudaEventClock.end();

      checkCudaErrors(cudaDeviceSynchronize());

      if (ConfigurationManager::getConfig().execution.measurePeakMemoryUsage) {
        const auto peakMemoryUsage = peakMemoryUsageProfiler.end();
        fmt::print(
          "Peak memory usage (MiB): {:.2f}\n",
          static_cast<float>(peakMemoryUsage) / 1024.0 / 1024.0
        );
      }

      fmt::print("Total time used (s): {}\n", cudaEventClock.getTimeInSeconds());

      if (ConfigurationManager::getConfig().generic.useUM) {
        resetAvailableMemoryForUM();
      }
      
      checkCudaErrors(cudaStreamDestroy(execStream));
    }
  }

  clock.logWithCurrentTime("Synchronization done");
  fmt::print("[MEMORY-INFO] Total managed memory size: {:.2f} MB\n", totalManagedMemoryMB);

  // SECTION 8: RESULT VERIFICATION
  if (verify) {
    clock.logWithCurrentTime("Start verification");
    fmt::print("Result passes verification: {}\n", verifyCholeskyDecompositionPartially(h_originalMatrix, d_tiles, N, B));
    clock.logWithCurrentTime("Verification done");
  }

  clock.logWithCurrentTime("All finished");

  // SECTION 9: CLEANUP
  checkCudaErrors(cudaFreeHost(h_workspace));
  checkCudaErrors(cudaFree(d_workspace));
  checkCudaErrors(cudaFreeHost(h_originalMatrix));
  
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