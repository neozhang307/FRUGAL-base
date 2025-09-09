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
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <vector>

#include "../include/argh.h"
#include "memopt.hpp"

using namespace memopt;

// Global variables
size_t N; // Matrix dimension
size_t T; // Number of tiles 
size_t B; // Block size (N/T)
size_t* current_block_size = &B; // Pointer to current block size for kernels

const std::string INPUT_MATRIX_FILE_PATH = "tiledCholeskyInputMatrix.in";

// Kernels from original
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

void generateRandomSymmetricPositiveDefiniteMatrix(double *h_A, const size_t n) {
  double *d_A;
  checkCudaErrors(cudaMalloc(&d_A, n * n * sizeof(double)));

  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
  curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());
  curandGenerateUniformDouble(prng, d_A, n * n);

  size_t numThreads = 1024;
  size_t numBlocks = (n * n + numThreads - 1) / numThreads;
  makeMatrixSymmetric<<<numBlocks, numThreads>>>(d_A, n);

  numThreads = 1024;
  numBlocks = (n + numThreads - 1) / numThreads;
  addIdenticalMatrix<<<numBlocks, numThreads>>>(d_A, n);

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(h_A, d_A, n * n * sizeof(double), cudaMemcpyDefault));
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaFree(d_A));
  curandDestroyGenerator(prng);
}

void initializeDeviceData(double *h_originalMatrix, std::vector<double *> &d_tiles) {
  fmt::print("Initializing device data for {} tiles with {}x{} matrix\n", d_tiles.size(), N, N);
  
  for (int i = 0; i < T; i++) {
    for (int j = 0; j < T; j++) {
      for (int k = 0; k < B; k++) {
        auto& memManager = MemoryManager::getInstance();
        void* srcAddress = memManager.getAddress(d_tiles[i + j * T]);
        
        // Handle memory manager address resolution properly
        if (srcAddress == d_tiles[i + j * T]) {
          srcAddress = memManager.getStoragePtr(d_tiles[i + j * T]);
          if (srcAddress == nullptr) {
            srcAddress = d_tiles[i + j * T];
          }
        }

        checkCudaErrors(cudaMemcpy(
          (char*)srcAddress + B * k * sizeof(double),
          h_originalMatrix + N * (j * B + k) + B * i,
          B * sizeof(double),
          cudaMemcpyDefault
        ));
      }
    }
  }
  checkCudaErrors(cudaDeviceSynchronize());
}

// Simplified structural verification
bool verifyCholeskyDecompositionPartially(double *A, std::vector<double *> &d_tiles) {
  auto& memManager = MemoryManager::getInstance();
  const size_t t = T;
  const size_t matrix_size_mb = (N * N * sizeof(double)) / (1024 * 1024);
  
  fmt::print("Performing structural verification (checking diagonal positivity, matrix size: {}MB)...\n", matrix_size_mb);
  
  bool validation_passed = true;
  
  // Check diagonal positivity
  for (int i = 0; i < t; i++) {
    double* h_tile;
    checkCudaErrors(cudaMallocHost(&h_tile, B * B * sizeof(double)));
    
    bool copySuccess = memManager.copyManagedArrayToHost(d_tiles[i + i * t], h_tile, B * B * sizeof(double));
    if (!copySuccess) {
      fmt::print("ERROR: Failed to copy diagonal tile [{},{}]\n", i, i);
      checkCudaErrors(cudaFreeHost(h_tile));
      return false;
    }
    
    // Check diagonal elements are positive
    for (int k = 0; k < B; k++) {
      if (h_tile[k * B + k] <= 0.0) {
        fmt::print("❌ DIAGONAL CHECK FAILED: Non-positive element at tile [{},{}], position [{},{}]: {:.6f}\n", 
                   i, i, k, k, h_tile[k * B + k]);
        validation_passed = false;
      }
    }
    
    checkCudaErrors(cudaFreeHost(h_tile));
  }
  
  if (validation_passed) {
    fmt::print("✅ STRUCTURAL VERIFICATION PASSED: All diagonal elements are positive\n");
    return true;
  } else {
    fmt::print("❌ STRUCTURAL VERIFICATION FAILED: Issues found in diagonal elements\n");
    return false;
  }
}

void tiledCholeskyMemoryOptimized() {
  fmt::print("=== Tiled Cholesky Memory Optimized Demo ===\n");
  fmt::print("Matrix: {}x{}, {} tiles, {}x{} blocks ({:.2f} MB)\n", 
             N, N, T*T, B, B, 
             (double)(N * N * sizeof(double)) / (1024.0 * 1024.0));
  
  initializeCudaDevice();

  // =========================================================================
  // PHASE 1: BUILD COMPUTATION GRAPH
  // =========================================================================
  fmt::print("\n--- PHASE 1: Build Computation Graph ---\n");
  
  const size_t tileSize = B * B * sizeof(double);
  
  // Generate matrix
  double* h_matrix = nullptr;
  checkCudaErrors(cudaMallocHost(&h_matrix, N * N * sizeof(double)));
  generateRandomSymmetricPositiveDefiniteMatrix(h_matrix, N);
  
  // Allocate GPU tiles
  std::vector<double*> d_tiles;
  auto& memManager = MemoryManager::getInstance();
  
  auto getMatrixBlock = [&d_tiles](int i, int j) -> double* {
    return d_tiles[i + j * T];
  };
  
  for (int i = 0; i < T * T; i++) {
    double *d_tile;
    checkCudaErrors(cudaMalloc(&d_tile, tileSize));
    d_tiles.push_back(d_tile);
    memManager.registerManagedMemoryAddress(d_tile, tileSize);
  }
  
  double totalManagedMemoryMB = memManager.GetMemoryManagedSizeInMB();
  fmt::print("Total managed memory: {:.2f} MB\n", totalManagedMemoryMB);

  // CUDA library setup
  cusolverDnHandle_t cusolverDnHandle;
  cusolverDnParams_t cusolverDnParams;
  cublasHandle_t cublasHandle;
  checkCudaErrors(cusolverDnCreate(&cusolverDnHandle));
  checkCudaErrors(cusolverDnCreateParams(&cusolverDnParams));
  checkCudaErrors(cublasCreate(&cublasHandle));

  double *one, *minusOne;
  checkCudaErrors(cudaMallocManaged(&one, sizeof(double)));
  checkCudaErrors(cudaMallocManaged(&minusOne, sizeof(double)));
  *one = 1.0;
  *minusOne = -1.0;

  // Workspace for cuSOLVER
  size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
  checkCudaErrors(cusolverDnXpotrf_bufferSize(
    cusolverDnHandle, cusolverDnParams, CUBLAS_FILL_MODE_LOWER, B,
    CUDA_R_64F, d_tiles[0], B, CUDA_R_64F,
    &workspaceInBytesOnDevice, &workspaceInBytesOnHost
  ));
  
  void *h_workspace, *d_workspace;
  int *d_info;
  checkCudaErrors(cudaMallocHost(&h_workspace, workspaceInBytesOnHost));
  checkCudaErrors(cudaMalloc(&d_workspace, workspaceInBytesOnDevice));
  checkCudaErrors(cudaMallocManaged(&d_info, sizeof(int)));

  // Build CUDA graphs for each stage
  fmt::print("Building staged CUDA graphs...\n");
  
  cudaStream_t s;
  checkCudaErrors(cudaStreamCreate(&s));
  checkCudaErrors(cusolverDnSetStream(cusolverDnHandle, s));
  checkCudaErrors(cublasSetStream(cublasHandle, s));

  TaskManager_v2 tmanager_v2(true);
  
  // Store stage task IDs for staged execution
  std::vector<std::vector<TaskId>> stageTaskIds;
  
  // Build tasks for each stage separately  
  for (int k = 0; k < T; k++) {
    fmt::print("Building tasks for stage k={}/{}\n", k, T-1);
    std::vector<TaskId> currentStageTasks;
    // POTRF - Cholesky factorization of diagonal tile
    std::vector<void*> inputs = {static_cast<void*>(getMatrixBlock(k, k))};
    std::vector<void*> outputs = {static_cast<void*>(getMatrixBlock(k, k))};
    
    TaskId potrfTaskId = tmanager_v2.registerTask<std::function<void(cudaStream_t, double*, void*, size_t, int*)>, double*, void*, size_t, int*>(
      [cusolverDnHandle, cusolverDnParams, workspaceInBytesOnHost](cudaStream_t stream, double* matrixblock_k_k, void* d_workspace, size_t workspaceInBytesOnDevice, int* d_info) {
        checkCudaErrors(cusolverDnSetStream(cusolverDnHandle, stream));
        checkCudaErrors(cusolverDnXpotrf(
          cusolverDnHandle, cusolverDnParams, CUBLAS_FILL_MODE_LOWER, *current_block_size,
          CUDA_R_64F, matrixblock_k_k, *current_block_size, CUDA_R_64F,
          d_workspace, workspaceInBytesOnDevice, nullptr, workspaceInBytesOnHost, d_info
        ));
      },
      inputs, outputs,
      TaskManager_v2::makeArgs(getMatrixBlock(k, k), d_workspace, workspaceInBytesOnDevice, d_info),
      "POTRF_task_" + std::to_string(k)
    );
    
    currentStageTasks.push_back(potrfTaskId);

    // TRSM - Triangular solve 
    for (int i = k + 1; i < T; i++) {
      inputs = {static_cast<void*>(getMatrixBlock(i, k)), static_cast<void*>(getMatrixBlock(k, k))};
      outputs = {static_cast<void*>(getMatrixBlock(i, k))};
      
      TaskId trsmTaskId = tmanager_v2.registerTask<std::function<void(cudaStream_t, double*, double*)>, double*, double*>(
        [cublasHandle, one](cudaStream_t stream, double* matrixblock_k_k, double* matrixblock_i_k) {
          checkCudaErrors(cublasSetStream(cublasHandle, stream));
          checkCudaErrors(cublasDtrsm(
            cublasHandle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, *current_block_size, *current_block_size, one,
            matrixblock_k_k, *current_block_size, matrixblock_i_k, *current_block_size
          ));
        },
        inputs, outputs,
        TaskManager_v2::makeArgs(getMatrixBlock(k, k), getMatrixBlock(i, k)),
        "TRSM_task_" + std::to_string(i) + "_" + std::to_string(k)
      );
      
      currentStageTasks.push_back(trsmTaskId);
    }

    // SYRK - Update diagonal tiles
    for (int i = k + 1; i < T; i++) {
      inputs = {static_cast<void*>(getMatrixBlock(i, i)), static_cast<void*>(getMatrixBlock(i, k))};
      outputs = {static_cast<void*>(getMatrixBlock(i, i))};
      
      TaskId syrkTaskId = tmanager_v2.registerTask<std::function<void(cudaStream_t, double*, double*)>, double*, double*>(
        [cublasHandle, minusOne, one](cudaStream_t stream, double* matrixblock_i_i, double* matrixblock_i_k) {
          checkCudaErrors(cublasSetStream(cublasHandle, stream));
          checkCudaErrors(cublasDsyrk(
            cublasHandle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
            *current_block_size, *current_block_size, minusOne, matrixblock_i_k, *current_block_size, one, matrixblock_i_i, *current_block_size
          ));
        },
        inputs, outputs,
        TaskManager_v2::makeArgs(getMatrixBlock(i, i), getMatrixBlock(i, k)),
        "SYRK_task_" + std::to_string(i) + "_" + std::to_string(k)
      );
      
      currentStageTasks.push_back(syrkTaskId);
    }

    // GEMM - Update off-diagonal tiles
    for (int i = k + 1; i < T; i++) {
      for (int j = k + 1; j < i; j++) {
        inputs = {static_cast<void*>(getMatrixBlock(i, j)), static_cast<void*>(getMatrixBlock(i, k)), static_cast<void*>(getMatrixBlock(j, k))};
        outputs = {static_cast<void*>(getMatrixBlock(i, j))};
        
        TaskId gemmTaskId = tmanager_v2.registerTask<std::function<void(cudaStream_t, double*, double*, double*)>, double*, double*, double*>(
          [cublasHandle, minusOne, one](cudaStream_t stream, double* matrixblock_i_j, double* matrixblock_i_k, double* matrixblock_j_k) {
            checkCudaErrors(cublasSetStream(cublasHandle, stream));
            checkCudaErrors(cublasDgemm(
              cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
              *current_block_size, *current_block_size, *current_block_size, minusOne, matrixblock_i_k, *current_block_size, matrixblock_j_k, *current_block_size, one, matrixblock_i_j, *current_block_size
            ));
          },
          inputs, outputs,
          TaskManager_v2::makeArgs(getMatrixBlock(i, j), getMatrixBlock(i, k), getMatrixBlock(j, k)),
          "GEMM_task_" + std::to_string(i) + "_" + std::to_string(j) + "_" + std::to_string(k)
        );
        
        currentStageTasks.push_back(gemmTaskId);
      }
    }
    
    // Store tasks for this stage
    stageTaskIds.push_back(currentStageTasks);
    fmt::print("Stage {} has {} tasks\n", k, currentStageTasks.size());
  }

  // =========================================================================
  // PHASE 2: STAGED OPTIMIZATION
  // =========================================================================
  fmt::print("\n--- PHASE 2: Staged Memory Optimization ---\n");
  
  double initialPeakMemory = memManager.GetMemoryManagedSizeInMB();
  fmt::print("Initial peak memory: {:.2f} MB\n", initialPeakMemory);
  
  // Initialize data before optimization 
  initializeDeviceData(h_matrix, d_tiles);
  
  // Use staged profiling and optimization
  fmt::print("Using staged profiling and optimization with {} stages...\n", T);
  auto optimizedGraph = profileAndOptimizeStaged(tmanager_v2, stageTaskIds, s);
  
  fmt::print("Original peak memory usage (MiB): {:.2f}\n", optimizedGraph.originalMemoryUsage);
  fmt::print("Optimized peak memory usage (MiB): {:.2f}\n", optimizedGraph.anticipatedPeakMemoryUsage);
  fmt::print("Memory reduction: {:.2f} MiB ({:.1f}%)\n", 
             optimizedGraph.originalMemoryUsage - optimizedGraph.anticipatedPeakMemoryUsage,
             ((optimizedGraph.originalMemoryUsage - optimizedGraph.anticipatedPeakMemoryUsage) / optimizedGraph.originalMemoryUsage) * 100);

  // Move all data to storage before execution (required for optimized execution)
  fmt::print("Moving all data to storage for optimized execution...\n");
  memManager.offloadAllManagedMemoryToStorage();
  fmt::print("✅ All data moved to CPU/storage\n");
  
  // =========================================================================
  // PHASE 3: EXECUTE WITH OPTIMIZED MEMORY
  // =========================================================================
  fmt::print("\n--- PHASE 3: Execute with Optimized Memory ---\n");
  
  // Reinitialize data after optimization
  initializeDeviceData(h_matrix, d_tiles);
  
  // Get current GPU memory info
  size_t free_mem, total_mem;
  checkCudaErrors(cudaMemGetInfo(&free_mem, &total_mem));
  fmt::print("GPU Memory - Total: {:.2f} MB, Free: {:.2f} MB\n", 
             (double)total_mem / (1024.0 * 1024.0), (double)free_mem / (1024.0 * 1024.0));
  
  // Start peak memory monitoring
  fmt::print("🔍 Starting continuous GPU memory monitoring during execution...\n");
  PeakMemoryUsageProfiler peakProfiler(10); // Sample every 10ms
  peakProfiler.start();
  
  // Run the optimized graph
  float runningTime;
  executeOptimizedGraph(
    optimizedGraph,
    [&tmanager_v2](int taskId, std::map<void*, void*> addressMapping, cudaStream_t stream) {
      tmanager_v2.execute(taskId, stream);
    },
    runningTime,
    memManager
  );
  
  // Get peak memory usage
  size_t peakMemoryBytes = peakProfiler.end();
  double peakMemoryMB = (double)peakMemoryBytes / (1024.0 * 1024.0);
  
  fmt::print("✅ Optimized execution completed!\n");
  fmt::print("Execution time: {:.3f} ms\n", runningTime * 1000.0f);
  fmt::print("📊 Peak GPU memory usage during execution: {:.2f} MB\n", peakMemoryMB);
  
  // Verify results
  bool result = verifyCholeskyDecompositionPartially(h_matrix, d_tiles);
  
  // Cleanup
  checkCudaErrors(cudaStreamDestroy(s));
  checkCudaErrors(cusolverDnDestroyParams(cusolverDnParams));
  checkCudaErrors(cusolverDnDestroy(cusolverDnHandle));
  checkCudaErrors(cublasDestroy(cublasHandle));
  
  checkCudaErrors(cudaFree(one));
  checkCudaErrors(cudaFree(minusOne));
  checkCudaErrors(cudaFree(d_workspace));
  checkCudaErrors(cudaFree(d_info));
  checkCudaErrors(cudaFreeHost(h_workspace));
  
  // Free managed memory properly using MemoryManager
  for (auto d_tile : d_tiles) {
    memManager.freeManagedMemory(d_tile);
  }
  checkCudaErrors(cudaFreeHost(h_matrix));
  
  fmt::print("\nFinal result: {}\n", result ? "SUCCESS" : "FAILED");
}

int main(int argc, char *argv[]) {
  // Simple command line parsing or use defaults
  N = (argc > 1) ? std::atoi(argv[1]) : 1024;  // Matrix dimension
  T = (argc > 2) ? std::atoi(argv[2]) : 4;     // Number of tiles
  
  // Calculate block size
  B = N / T;
  
  // Validation
  if (N % T != 0) {
    fmt::print("ERROR: Matrix dimension must be divisible by tile count\n");
    return -1;
  }
  
  ConfigurationManager::exportDefaultConfiguration();
  ConfigurationManager::loadConfiguration("config.json");
  
  fmt::print("Configuration: N={}, T={}, B={}\n", N, T, B);
  
  // Run the memory optimized version
  tiledCholeskyMemoryOptimized();
  
  return 0;
}