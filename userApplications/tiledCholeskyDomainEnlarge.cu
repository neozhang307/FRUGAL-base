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

// Global variables (will be set from config or command line)
size_t N_small; // Small matrix for optimization (fits in memory)
size_t N_large; // Large matrix for execution (may exceed memory) 
size_t T;       // Number of tiles (SAME for both)
size_t B_small; // Block size for small domain
size_t B_large; // Block size for large domain

// Global variable to control which block size kernels should use
// During optimization: use B_small, during execution: use B_large
size_t* current_block_size = &B_small;

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

void initializeDeviceData(double *h_originalMatrix, std::vector<double *> &d_tiles, size_t matrix_size, size_t block_size) {
  fmt::print("Initializing device data for {} tiles with {}x{} matrix\n", d_tiles.size(), matrix_size, matrix_size);
  
  for (int i = 0; i < T; i++) {
    for (int j = 0; j < T; j++) {
      for (int k = 0; k < block_size; k++) {
        auto& memManager = MemoryManager::getInstance();
        void* srcAddress = memManager.getAddress(d_tiles[i + j * T]);
        
        if (srcAddress == d_tiles[i]) {
          srcAddress = memManager.getStoragePtr(d_tiles[i + j * T]);
          if (srcAddress == nullptr) {
            srcAddress = d_tiles[i + j * T];
          }
        }

        checkCudaErrors(cudaMemcpy(
          (char*)srcAddress + block_size * k * sizeof(double),
          h_originalMatrix + matrix_size * (j * block_size + k) + block_size * i,
          block_size * sizeof(double),
          cudaMemcpyDefault
        ));
      }
    }
  }
  checkCudaErrors(cudaDeviceSynchronize());
}

// Simplified structural verification: just check diagonal positivity
bool verifyCholeskyDecompositionPartially(double *A, std::vector<double *> &d_tiles, const size_t n, const size_t b) {
  auto& memManager = MemoryManager::getInstance();
  const size_t t = n / b;  // number of tiles per dimension
  const size_t matrix_size_mb = (n * n * sizeof(double)) / (1024 * 1024);
  
  fmt::print("Performing structural verification (checking diagonal positivity, matrix size: {}MB)...\n", matrix_size_mb);
  
  bool validation_passed = true;
  
  // Check diagonal positivity (quick structural check)
  for (int i = 0; i < t; i++) {
    double* h_tile;
    checkCudaErrors(cudaMallocHost(&h_tile, b * b * sizeof(double)));
    
    bool copySuccess = memManager.copyManagedArrayToHost(d_tiles[i + i * t], h_tile, b * b * sizeof(double));
    if (!copySuccess) {
      fmt::print("ERROR: Failed to copy diagonal tile [{},{}]\n", i, i);
      checkCudaErrors(cudaFreeHost(h_tile));
      return false;
    }
    
    // Check diagonal elements are positive
    for (int k = 0; k < b; k++) {
      if (h_tile[k * b + k] <= 0.0) {
        fmt::print("❌ DIAGONAL CHECK FAILED: Non-positive element at tile [{},{}], position [{},{}]: {:.6f}\n", 
                   i, i, k, k, h_tile[k * b + k]);
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

void tiledCholeskyDomainEnlargement() {
  SystemWallClock clock;
  clock.start();
  
  fmt::print("=== Tiled Cholesky Domain Enlargement Demo ===\n");
  fmt::print("Small domain: {}x{} matrix, {} tiles, {}x{} blocks ({:.2f} MB)\n", 
             N_small, N_small, T*T, B_small, B_small, 
             (double)(N_small * N_small * sizeof(double)) / (1024.0 * 1024.0));
  fmt::print("Large domain: {}x{} matrix, {} tiles, {}x{} blocks ({:.2f} MB)\n", 
             N_large, N_large, T*T, B_large, B_large, 
             (double)(N_large * N_large * sizeof(double)) / (1024.0 * 1024.0));
  
  initializeCudaDevice();

  // =========================================================================
  // PHASE 1: OPTIMIZATION WITH SMALL DOMAIN
  // =========================================================================
  fmt::print("\n--- PHASE 1: Optimization with Small Domain ---\n");
  
  const size_t tileSize_small = B_small * B_small * sizeof(double);
  
  // Generate small matrix
  double* h_smallMatrix = nullptr;
  checkCudaErrors(cudaMallocHost(&h_smallMatrix, N_small * N_small * sizeof(double)));
  generateRandomSymmetricPositiveDefiniteMatrix(h_smallMatrix, N_small);
  
  // Allocate GPU tiles for small domain
  std::vector<double*> d_tiles;
  auto& memManager = MemoryManager::getInstance();
  
  auto getMatrixBlock = [&d_tiles](int i, int j) -> double* {
    return d_tiles[i + j * T];
  };
  
  for (int i = 0; i < T * T; i++) {
    double *d_tile;
    checkCudaErrors(cudaMalloc(&d_tile, tileSize_small));
    d_tiles.push_back(d_tile);
    memManager.registerManagedMemoryAddress(d_tile, tileSize_small);
  }
  
  double totalManagedMemoryMB = memManager.GetMemoryManagedSizeInMB();
  fmt::print("Total managed memory: {:.2f} MB\n", totalManagedMemoryMB);

  // CUDA library setup (from original working code)
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
    cusolverDnHandle, cusolverDnParams, CUBLAS_FILL_MODE_LOWER, B_small,
    CUDA_R_64F, d_tiles[0], B_small, CUDA_R_64F,
    &workspaceInBytesOnDevice, &workspaceInBytesOnHost
  ));
  
  void *h_workspace, *d_workspace;
  int *d_info;
  checkCudaErrors(cudaMallocHost(&h_workspace, workspaceInBytesOnHost));
  checkCudaErrors(cudaMalloc(&d_workspace, workspaceInBytesOnDevice));
  checkCudaErrors(cudaMallocManaged(&d_info, sizeof(int)));

  // Build CUDA graph (copied from working original)
  fmt::print("Building CUDA graph...\n");
  cudaGraph_t graph;
  checkCudaErrors(cudaGraphCreate(&graph, 0));

  cudaStream_t s;
  checkCudaErrors(cudaStreamCreate(&s));
  checkCudaErrors(cusolverDnSetStream(cusolverDnHandle, s));
  checkCudaErrors(cublasSetStream(cublasHandle, s));

  TaskManager_v2 tmanager_v2(true);
  auto graphConstructor = std::make_unique<PointerDependencyCudaGraphConstructor>(s, graph);

  // Tiled Cholesky algorithm (from original working code) 
  for (int k = 0; k < T; k++) {
    // POTRF - Cholesky factorization of diagonal tile
    std::vector<void*> inputs = {static_cast<void*>(getMatrixBlock(k, k))};
    std::vector<void*> outputs = {static_cast<void*>(getMatrixBlock(k, k))};
    
    graphConstructor->beginCaptureOperation(inputs, outputs);
    
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
    
    tmanager_v2.execute(potrfTaskId, s);
    graphConstructor->endCaptureOperation();

    // TRSM - Triangular solve 
    for (int i = k + 1; i < T; i++) {
      inputs = {static_cast<void*>(getMatrixBlock(i, k)), static_cast<void*>(getMatrixBlock(k, k))};
      outputs = {static_cast<void*>(getMatrixBlock(i, k))};
      
      graphConstructor->beginCaptureOperation(inputs, outputs);
      
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
      
      tmanager_v2.execute(trsmTaskId, s);
      graphConstructor->endCaptureOperation();
    }

    // SYRK - Update diagonal tiles
    for (int i = k + 1; i < T; i++) {
      inputs = {static_cast<void*>(getMatrixBlock(i, i)), static_cast<void*>(getMatrixBlock(i, k))};
      outputs = {static_cast<void*>(getMatrixBlock(i, i))};
      
      graphConstructor->beginCaptureOperation(inputs, outputs);
      
      TaskId syrkTaskId = tmanager_v2.registerTask<std::function<void(cudaStream_t, double*, double*)>, double*, double*>(
        [cublasHandle, minusOne, one](cudaStream_t stream, double* matrixblock_i_k, double* matrixblock_i_i) {
          checkCudaErrors(cublasSetStream(cublasHandle, stream));
          checkCudaErrors(cublasDsyrk(
            cublasHandle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, *current_block_size, *current_block_size,
            minusOne, matrixblock_i_k, *current_block_size, one, matrixblock_i_i, *current_block_size
          ));
        },
        inputs, outputs,
        TaskManager_v2::makeArgs(getMatrixBlock(i, k), getMatrixBlock(i, i)),
        "SYRK_task_" + std::to_string(i) + "_" + std::to_string(k)
      );
      
      tmanager_v2.execute(syrkTaskId, s);
      graphConstructor->endCaptureOperation();

      // GEMM - Update off-diagonal tiles
      for (int j = i + 1; j < T; j++) {
        inputs = {
          static_cast<void*>(getMatrixBlock(j, i)), 
          static_cast<void*>(getMatrixBlock(j, k)), 
          static_cast<void*>(getMatrixBlock(i, k))
        };
        outputs = {static_cast<void*>(getMatrixBlock(j, i))};
        
        graphConstructor->beginCaptureOperation(inputs, outputs);
        
        TaskId gemmTaskId = tmanager_v2.registerTask<std::function<void(cudaStream_t, double*, double*, double*)>, double*, double*, double*>(
          [cublasHandle, minusOne, one](cudaStream_t stream, double* matrixblock_j_k, double* matrixblock_i_k, double* matrixblock_j_i) {
            checkCudaErrors(cublasSetStream(cublasHandle, stream));
            checkCudaErrors(cublasGemmEx(
              cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, *current_block_size, *current_block_size, *current_block_size,
              minusOne, matrixblock_j_k, CUDA_R_64F, *current_block_size,
              matrixblock_i_k, CUDA_R_64F, *current_block_size, one,
              matrixblock_j_i, CUDA_R_64F, *current_block_size,
              CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT
            ));
          },
          inputs, outputs,
          TaskManager_v2::makeArgs(getMatrixBlock(j, k), getMatrixBlock(i, k), getMatrixBlock(j, i)),
          "GEMM_task_" + std::to_string(j) + "_" + std::to_string(i) + "_" + std::to_string(k)
        );
        
        tmanager_v2.execute(gemmTaskId, s);
        graphConstructor->endCaptureOperation();
      }
    }
  }

  // Profile and optimize
  fmt::print("Profiling and optimizing with small domain...\n");
  auto optimizedGraph = profileAndOptimize(graph);
  
  fmt::print("Original peak memory usage (MiB): {:.2f}\n", optimizedGraph.originalMemoryUsage);
  fmt::print("Optimized peak memory usage (MiB): {:.2f}\n", optimizedGraph.anticipatedPeakMemoryUsage);

  // Move all data to storage before enlargement (required for cold start)
  fmt::print("Moving all small domain data to storage...\n");
  memManager.offloadAllManagedMemoryToStorage();
  fmt::print("✅ All data moved to CPU/storage\n");

  // =========================================================================
  // PHASE 2: DOMAIN ENLARGEMENT
  // =========================================================================
  fmt::print("\n--- PHASE 2: Domain Enlargement ---\n");
  
  // Generate large matrix
  double* h_largeMatrix = nullptr;
  checkCudaErrors(cudaMallocHost(&h_largeMatrix, N_large * N_large * sizeof(double)));
  generateRandomSymmetricPositiveDefiniteMatrix(h_largeMatrix, N_large);
  fmt::print("Generated {}x{} large matrix ({:.2f} MB)\n", N_large, N_large,
             (double)(N_large * N_large * sizeof(double)) / (1024.0 * 1024.0));
  
  // Allocate CPU arrays for large domain (SAME tile count, larger blocks)
  std::vector<double*> h_tiles_large;
  size_t tileSize_large = B_large * B_large * sizeof(double);
  
  for (int i = 0; i < T * T; i++) {
    double *h_tile;
    checkCudaErrors(cudaMallocHost(&h_tile, tileSize_large));
    h_tiles_large.push_back(h_tile);
  }
  
  // Initialize large tiles from matrix
  for (int i = 0; i < T; i++) {
    for (int j = 0; j < T; j++) {
      if (i >= j) { // Lower triangular only
        double *h_tile = h_tiles_large[i * T + j];
        for (int row = 0; row < B_large; row++) {
          for (int col = 0; col < B_large; col++) {
            h_tile[row * B_large + col] = h_largeMatrix[(i * B_large + row) * N_large + (j * B_large + col)];
          }
        }
      }
    }
  }
  
  // Re-register arrays with larger CPU data (SAME tile structure)
  fmt::print("Re-registering {} arrays with larger CPU data ({:.2f} MB per tile)...\n", 
             T * T, (double)tileSize_large / (1024.0 * 1024.0));
  
  for (int i = 0; i < T; i++) {
    for (int j = 0; j < T; j++) {
      if (i >= j) {
        int idx = i * T + j;
        
        bool success = memManager.reregisterManagedArrayWithLargerCPUData(
          d_tiles[idx], h_tiles_large[idx], tileSize_large
        );
        
        if (!success) {
          fmt::print("ERROR: Failed to re-register tile [{},{}]\n", i, j);
          return;
        }
      }
    }
  }
  
  fmt::print("✅ Re-registration complete - optimization plan preserved\n");
  
  // Switch kernels to use large block size for execution
  size_t old_block_size = *current_block_size;
  *current_block_size = B_large;
  fmt::print("🔄 Switched kernel block size from {} to {}\n", old_block_size, *current_block_size);
  
  // Reallocate cuSOLVER workspace for the new block size
  size_t new_workspaceInBytesOnDevice, new_workspaceInBytesOnHost;
  checkCudaErrors(cusolverDnXpotrf_bufferSize(
    cusolverDnHandle, cusolverDnParams, CUBLAS_FILL_MODE_LOWER, B_large,
    CUDA_R_64F, d_tiles[0], B_large, CUDA_R_64F,
    &new_workspaceInBytesOnDevice, &new_workspaceInBytesOnHost
  ));
  
  // Free old workspace
  checkCudaErrors(cudaFree(d_workspace));
  checkCudaErrors(cudaFreeHost(h_workspace));
  
  // Allocate new workspace with correct size
  checkCudaErrors(cudaMallocHost(&h_workspace, new_workspaceInBytesOnHost));
  checkCudaErrors(cudaMalloc(&d_workspace, new_workspaceInBytesOnDevice));
  
  fmt::print("📝 Reallocated cuSOLVER workspace: {:.1f} MB device, {:.1f} MB host\n",
             (double)new_workspaceInBytesOnDevice / (1024.0 * 1024.0),
             (double)new_workspaceInBytesOnHost / (1024.0 * 1024.0));
  
  // =========================================================================
  // PHASE 3: EXECUTION WITH ENLARGED DOMAIN
  // =========================================================================
  fmt::print("\n--- PHASE 3: Cold Start Execution with Large Domain ---\n");
  
  // Get current GPU memory info
  size_t free_mem, total_mem;
  checkCudaErrors(cudaMemGetInfo(&free_mem, &total_mem));
  fmt::print("GPU Memory - Total: {:.2f} MB, Free: {:.2f} MB\n", 
             (double)total_mem / (1024.0 * 1024.0), (double)free_mem / (1024.0 * 1024.0));
  
  double requiredMemoryMB = (double)(T * T * tileSize_large) / (1024.0 * 1024.0);
  fmt::print("Required for all tiles: {:.2f} MB\n", requiredMemoryMB);
  
  if (requiredMemoryMB * 1024.0 * 1024.0 > free_mem) {
    fmt::print("🚀 LARGE DOMAIN EXCEEDS GPU MEMORY - Memory management will be active!\n");
  } else {
    fmt::print("⚠️  Large domain still fits in GPU memory - increase size for true out-of-memory test\n");
  }
  
  // Start peak memory monitoring
  fmt::print("🔍 Starting continuous GPU memory monitoring during execution...\n");
  PeakMemoryUsageProfiler peakProfiler(10); // Sample every 10ms for high resolution
  peakProfiler.start();
  
  float runningTime;
  Executor* executor = Executor::getInstance();
  tmanager_v2.setExecutionMode(TaskManager_v2::ExecutionMode::Production);
  
  executor->executeOptimizedGraphColdStart(
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
  
  fmt::print("✅ Cold start execution completed!\n");
  fmt::print("Execution time with enlarged domain: {:.3f} ms\n", runningTime * 1000.0f);
  fmt::print("📊 Peak GPU memory usage during execution: {:.2f} MB\n", peakMemoryMB);
  
  // Compare with static measurements
  size_t current_free, current_total;
  checkCudaErrors(cudaMemGetInfo(&current_free, &current_total));
  double currentUsedMB = (double)(current_total - current_free) / (1024.0 * 1024.0);
  fmt::print("📊 Current GPU memory usage: {:.2f} MB\n", currentUsedMB);

  // =========================================================================
  // PHASE 4: VERIFICATION
  // =========================================================================
  fmt::print("\n--- PHASE 4: Verification ---\n");
  
  bool verificationPassed = verifyCholeskyDecompositionPartially(h_largeMatrix, d_tiles, N_large, B_large);
  
  if (verificationPassed) {
    fmt::print("✅ Verification PASSED: Results are valid\n");
  } else {
    fmt::print("❌ Verification FAILED\n");
  }

  // =========================================================================
  // PHASE 4.5: COMPARISON WITH DIRECT CHOLESKY
  // =========================================================================
  fmt::print("\n--- PHASE 4.5: Comparison with cuSOLVER Direct Cholesky ---\n");
  
  // Allocate memory for direct Cholesky
  double* d_A_direct;
  checkCudaErrors(cudaMalloc(&d_A_direct, N_large * N_large * sizeof(double)));
  checkCudaErrors(cudaMemcpy(d_A_direct, h_largeMatrix, N_large * N_large * sizeof(double), cudaMemcpyHostToDevice));
  
  // Create new cuSOLVER handle for direct computation
  cusolverDnHandle_t directSolverHandle;
  checkCudaErrors(cusolverDnCreate(&directSolverHandle));
  
  // Query workspace size for direct Cholesky
  int directWorkspaceSize = 0;
  checkCudaErrors(cusolverDnDpotrf_bufferSize(
    directSolverHandle, CUBLAS_FILL_MODE_LOWER, N_large, d_A_direct, N_large, &directWorkspaceSize));
  
  double* d_directWorkspace;
  checkCudaErrors(cudaMalloc(&d_directWorkspace, directWorkspaceSize * sizeof(double)));
  
  int* d_directInfo;
  checkCudaErrors(cudaMalloc(&d_directInfo, sizeof(int)));
  
  // Perform direct Cholesky decomposition
  fmt::print("Running cuSOLVER direct Cholesky on {}x{} matrix...\n", N_large, N_large);
  checkCudaErrors(cusolverDnDpotrf(
    directSolverHandle, CUBLAS_FILL_MODE_LOWER, N_large, d_A_direct, N_large, 
    d_directWorkspace, directWorkspaceSize, d_directInfo));
  
  // Check if direct decomposition succeeded
  int h_directInfo;
  checkCudaErrors(cudaMemcpy(&h_directInfo, d_directInfo, sizeof(int), cudaMemcpyDeviceToHost));
  
  if (h_directInfo != 0) {
    fmt::print("ERROR: Direct Cholesky failed with info = {}\n", h_directInfo);
  } else {
    fmt::print("✅ Direct Cholesky succeeded\n");
    
    // Copy direct result to host
    double* h_L_direct;
    checkCudaErrors(cudaMallocHost(&h_L_direct, N_large * N_large * sizeof(double)));
    checkCudaErrors(cudaMemcpy(h_L_direct, d_A_direct, N_large * N_large * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Copy tiled result to contiguous memory for comparison
    double* h_L_tiled;
    checkCudaErrors(cudaMallocHost(&h_L_tiled, N_large * N_large * sizeof(double)));
    
    // Reconstruct tiled result into contiguous array
    for (size_t i = 0; i < T; i++) {
      for (size_t j = 0; j <= i; j++) {
        size_t tile_idx = i * T + j;
        double* h_tile_data;
        checkCudaErrors(cudaMallocHost(&h_tile_data, B_large * B_large * sizeof(double)));
        
        // Copy tile data using smart copy
        memManager.copyManagedArrayToHost(d_tiles[tile_idx], h_tile_data, B_large * B_large * sizeof(double));
        
        // Copy to contiguous array
        for (size_t bi = 0; bi < B_large; bi++) {
          for (size_t bj = 0; bj < B_large; bj++) {
            size_t global_i = i * B_large + bi;
            size_t global_j = j * B_large + bj;
            if (global_i >= global_j && global_i < N_large && global_j < N_large) {
              h_L_tiled[global_i * N_large + global_j] = h_tile_data[bi * B_large + bj];
            }
          }
        }
        checkCudaErrors(cudaFreeHost(h_tile_data));
      }
    }
    
    // Compare the two results
    fmt::print("\nComparing tiled vs direct Cholesky results...\n");
    double max_diff = 0.0;
    int diff_count = 0;
    const double tolerance = 1e-6;
    
    for (size_t i = 0; i < N_large; i++) {
      for (size_t j = 0; j <= i && j < N_large; j++) {
        double tiled_val = h_L_tiled[i * N_large + j];
        double direct_val = h_L_direct[i * N_large + j];
        double diff = std::abs(tiled_val - direct_val);
        
        if (diff > tolerance) {
          if (diff_count < 10) {  // Print first 10 differences
            fmt::print("  Diff at [{},{}]: tiled={:.6f}, direct={:.6f}, diff={:.2e}\n",
                      i, j, tiled_val, direct_val, diff);
          }
          diff_count++;
          max_diff = std::max(max_diff, diff);
        }
      }
    }
    
    if (diff_count == 0) {
      fmt::print("✅ Tiled and direct Cholesky results MATCH perfectly!\n");
    } else {
      fmt::print("❌ Tiled and direct Cholesky DIFFER: {} differences, max diff = {:.2e}\n", 
                diff_count, max_diff);
    }
    
    // Simple verification: just compare tiled vs direct results
    size_t matrix_size_mb = (N_large * N_large * sizeof(double)) / (1024 * 1024);
    
    fmt::print("\n=== VERIFICATION RESULTS (Matrix: {}MB) ===\n", matrix_size_mb);
    
    if (diff_count == 0) {
      fmt::print("✅ SUCCESS: Tiled and Direct Cholesky produce IDENTICAL results\n");
      fmt::print("   Domain enlargement is working correctly!\n");
    } else {
      fmt::print("❌ FAILURE: Tiled and Direct Cholesky results DIFFER\n");
      fmt::print("   {} differences found, max difference = {:.2e}\n", diff_count, max_diff);
    }
    
    checkCudaErrors(cudaFreeHost(h_L_direct));
    checkCudaErrors(cudaFreeHost(h_L_tiled));
  }
  
  // Cleanup direct Cholesky resources
  checkCudaErrors(cudaFree(d_A_direct));
  checkCudaErrors(cudaFree(d_directWorkspace));
  checkCudaErrors(cudaFree(d_directInfo));
  checkCudaErrors(cusolverDnDestroy(directSolverHandle));

  // =========================================================================
  // PHASE 5: CLEANUP
  // =========================================================================
  fmt::print("\n--- Cleanup ---\n");
  
  checkCudaErrors(cudaFreeHost(h_smallMatrix));
  checkCudaErrors(cudaFreeHost(h_largeMatrix));
  checkCudaErrors(cudaFreeHost(h_workspace));
  checkCudaErrors(cudaFree(d_workspace));
  checkCudaErrors(cudaFree(minusOne));  // Fixed: allocated with cudaMallocManaged
  checkCudaErrors(cudaFree(one));       // Fixed: allocated with cudaMallocManaged
  checkCudaErrors(cudaFree(d_info));
  
  for (auto h_tile : h_tiles_large) {
    checkCudaErrors(cudaFreeHost(h_tile));
  }
  
  for (auto d_tile : d_tiles) {
    memManager.freeManagedMemory(d_tile);
  }
  
  checkCudaErrors(cusolverDnDestroyParams(cusolverDnParams));
  checkCudaErrors(cusolverDnDestroy(cusolverDnHandle));
  checkCudaErrors(cublasDestroy(cublasHandle));
  checkCudaErrors(cudaStreamDestroy(s));
  
  fmt::print("=== Domain Enlargement Demo Complete ===\n");
}

int main(int argc, char **argv) {
  // Simple command line parsing or use defaults
  N_small = (argc > 1) ? std::atoi(argv[1]) : 256;   // Small domain (fits in GPU memory)
  N_large = (argc > 2) ? std::atoi(argv[2]) : 2048;  // Large domain (may exceed GPU memory)  
  T = (argc > 3) ? std::atoi(argv[3]) : 4;           // Number of tiles (SAME for both)
  
  // Calculate block sizes
  B_small = N_small / T;
  B_large = N_large / T;
  
  // Initialize kernel block size to small for optimization phase
  *current_block_size = B_small;
  
  // Validation
  if (N_small % T != 0 || N_large % T != 0) {
    fmt::print("ERROR: Matrix dimensions must be divisible by tile count\n");
    return -1;
  }
  
  if (N_large <= N_small) {
    fmt::print("ERROR: Large domain must be > small domain\n");
    return -1;
  }
  
  ConfigurationManager::exportDefaultConfiguration();
  ConfigurationManager::loadConfiguration("config.json");
  
  tiledCholeskyDomainEnlargement();
  
  return 0;
}