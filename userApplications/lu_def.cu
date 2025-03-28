#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <cusolverDn.h>
#include <fmt/core.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <tuple>
#include <vector>
#include <omp.h>

#include "../include/argh.h"
#include "memopt.hpp"

using namespace memopt;

const std::string INPUT_MATRIX_FILE_PATH = "luInputMatrix.in";

// Implementation Note: This file implements the LU decomposition algorithm
// using the generic PointerDependencyCudaGraphConstructor for memory tracking
// instead of the application-specific TiledLUGraphCreator.

size_t N; // total matrix size (N×N)
size_t B; // batch size in 1d (B×B per tile)
size_t T; // tile amount in 1d (T×T tiles)

// Global flag for GPU verification
bool forceGpuVerify = false;


// Using utilities from the memopt namespace


// Generate a random symmetric positive definite matrix
// Credit to: https://math.stackexchange.com/questions/357980/how-to-generate-random-symmetric-positive-definite-matrices-using-matlab
// 
// A symmetric positive definite matrix is required for LU decomposition to be stable.
// This function creates such a matrix in three steps:
// 1. Generate a random matrix
// 2. Make it symmetric by averaging with its transpose
// 3. Add n to the diagonal elements to ensure positive definiteness
void generateRandomSymmetricPositiveDefiniteMatrix(double *h_A, const size_t n)
{
    memopt::SystemWallClock clock;
    clock.start();

    // Clear the matrix first
    std::memset(h_A, 0, n * n * sizeof(double));
    
    // Use fixed seed for reproducibility
    const unsigned int seed = 420;
    
    clock.logWithCurrentTime("Matrix initialized, starting random generation");
    
    // Only compute upper triangular portion (including diagonal)
    // and mirror it to lower triangular portion
    #pragma omp parallel
    {
        // Each thread needs its own random generator to avoid collisions
        unsigned int thread_seed = seed + omp_get_thread_num();
        std::mt19937 local_rng(thread_seed);
        std::uniform_real_distribution<double> local_dist(0.0, 1.0);
        
        #pragma omp for
        for (int i = 0; i < n; i++) {
            // Set diagonal elements directly (random value + n for positive definiteness)
            h_A[i * n + i] = local_dist(local_rng) + n;
            
            // Upper triangular elements
            for (int j = i + 1; j < n; j++) {
                // Generate one random value and use it for both symmetric positions
                double value = local_dist(local_rng);
                h_A[i * n + j] = value;
                h_A[j * n + i] = value;
            }
        }
    }
    
    clock.logWithCurrentTime("Matrix generation completed");
}

// Print a square matrix with formatting for readability
// Each element is displayed with 3 decimal places and a width of 6 characters
void printSquareMatrix(double *h_A, const size_t n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (j != 0)
                std::cout << " ";  // Add space between columns
            std::cout << std::setw(6) << std::setprecision(3) << h_A[i * n + j];
        }
        std::cout << std::endl;  // New line for each row
    }
}

// Process the LU decomposition result from cuSOLVER into separate L and U matrices
// cuSOLVER returns the LU decomposition as a single matrix in column-major format
// This function:
// 1. Transposes from column-major to row-major format
// 2. Extracts the lower triangular L matrix (with unit diagonal)
// 3. Extracts the upper triangular U matrix
void cleanCusolverLUDecompositionResult(double *L, double *U, const int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = i; j < n; j++)
        {
            // Convert from column-major to row-major format
            std::swap(L[i + j * n], L[i * n + j]);
            
            // Copy the upper triangular part to U matrix
            U[i * n + j] = L[i * n + j];
            
            // Zero out the upper triangular part in L matrix
            L[i * n + j] = 0;
        }
        // Set diagonal entries of L to 1 (unit triangular)
        L[i * n + i] = 1;
    }
}

// Verify the correctness of LU decomposition by checking if L*U equals the original matrix A
// Returns true if the decomposition is valid (error is small enough)
bool verifyLUDecomposition(double *A, double *L, double *U, const int n)
{
    // Reconstruct matrix A as L*U to verify correctness
    auto newA = std::make_unique<double[]>(n * n);
    memset(newA.get(), 0, n * n * sizeof(double));
    
    // Matrix multiplication L*U
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
            {
                newA[i * n + j] += L[i * n + k] * U[k * n + j];
            }
        }
    }

    // Calculate total absolute error between original A and reconstructed L*U
    double error = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            error += fabs(A[i * n + j] - newA[i * n + j]);
        }
    }

    // Only print matrices when verbose mode is enabled
    if (ConfigurationManager::getConfig().execution.enableVerboseOutput) {
        printf("A:\n");
        printSquareMatrix(A, n);

        printf("\nnewA:\n");
        printSquareMatrix(newA.get(), n);

        printf("\nL:\n");
        printSquareMatrix(L, n);
        printf("\n");

        printf("\nU:\n");
        printSquareMatrix(U, n);
        printf("\n");
    }

    printf("error = %.6f}\n", error);

    // Consider the decomposition correct if error is less than threshold
    return error <= 1e-6;
}

// CUDA kernel to compute the absolute difference between two matrices
// A is in row-major format, while B is in column-major format
__global__ void computeAbsoluteDifferenceKernel(double* A, double* B, double* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        // Compute row and column from linear index
        int row = idx / n;
        int col = idx % n;
        
        // Get the elements from A (row-major) and B (column-major)
        double a_val = A[row * n + col];      // row-major access
        double b_val = B[col * n + row];      // column-major access is transposed
        
        // Store the absolute difference
        result[idx] = fabs(a_val - b_val);
    }
}

// GPU-accelerated verification of LU decomposition using cuBLAS
// Significantly faster than the CPU version for large matrices
bool verifyLUDecompositionGPU(double *A, double *L, double *U, const int n)
{
    // Use a system clock for timing - more reliable than CUDA events in this context
    auto startTime = std::chrono::high_resolution_clock::now();
    
    if (ConfigurationManager::getConfig().execution.enableDebugOutput) {
        LOG_TRACE_WITH_INFO("Starting LU verification (GPU)");
    }

    // Create a stream for this operation
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));

    // Create CUBLAS handle
    cublasHandle_t cublasHandle;
    checkCudaErrors(cublasCreate(&cublasHandle));
    checkCudaErrors(cublasSetStream(cublasHandle, stream));
    
    // Allocate device memory for matrices
    double *d_A, *d_L, *d_U, *d_LU, *d_diff;
    size_t matrixBytes = n * n * sizeof(double);
    
    checkCudaErrors(cudaMalloc(&d_A, matrixBytes));
    checkCudaErrors(cudaMalloc(&d_L, matrixBytes));
    checkCudaErrors(cudaMalloc(&d_U, matrixBytes));
    checkCudaErrors(cudaMalloc(&d_LU, matrixBytes));
    checkCudaErrors(cudaMalloc(&d_diff, matrixBytes));
    
    // Copy matrices to device using the stream
    checkCudaErrors(cudaMemcpyAsync(d_A, A, matrixBytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_L, L, matrixBytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_U, U, matrixBytes, cudaMemcpyHostToDevice, stream));
    
    // Constants for matrix multiplication: C = alpha*A*B + beta*C
    double alpha = 1.0;
    double beta = 0.0;
    
    // For CUBLAS, we need to handle the row-major to column-major conversion
    // We do a trick with the op parameters: cublas_op(row_major) = cublas_op_transpose(col_major)
    // B(m,k) A(k,n) computes C(m,n) in row major, which is equivalent to:
    // A^T(n,k) B^T(k,m) computes C^T(n,m) in column major
    
    // We want to compute L*U in row-major, which is equivalent to:
    // computing U^T * L^T = (L*U)^T in column-major
    // Here L is m×k, U is k×n in row-major
    checkCudaErrors(cublasDgemm(
        cublasHandle, 
        CUBLAS_OP_N, CUBLAS_OP_N,   // No transposition in column-major results in transposition for row-major
        n, n, n,                    // Matrix dimensions (result is n×n)
        &alpha,                     // Alpha multiplier
        d_U, n,                     // First matrix (U in column-major format)
        d_L, n,                     // Second matrix (L in column-major format)
        &beta,                      // Beta multiplier
        d_LU, n                     // Result matrix (LU in column-major format)
    ));
    
    // Compute absolute difference: |A - L*U|
    int blockSize = 256;
    int gridSize = (n * n + blockSize - 1) / blockSize;
    computeAbsoluteDifferenceKernel<<<gridSize, blockSize, 0, stream>>>(d_A, d_LU, d_diff, n);
    
    // Sum the absolute differences to get total error
    double error = 0.0;
    checkCudaErrors(cublasDasum(
        cublasHandle,
        n * n,                     // Number of elements
        d_diff, 1,                 // Input array and stride
        &error                     // Result
    ));

    // Wait for all operations to complete
    checkCudaErrors(cudaStreamSynchronize(stream));

    // Only print matrices when verbose mode is enabled
    if (ConfigurationManager::getConfig().execution.enableVerboseOutput) {
        // For debug output, we need to copy results back to host
        auto h_LU = std::make_unique<double[]>(n * n);
        checkCudaErrors(cudaMemcpy(h_LU.get(), d_LU, matrixBytes, cudaMemcpyDeviceToHost));
        
        printf("A:\n");
        printSquareMatrix(A, n);

        printf("\nL*U (GPU):\n");
        printSquareMatrix(h_LU.get(), n);

        printf("\nL:\n");
        printSquareMatrix(L, n);
        printf("\n");

        printf("\nU:\n");
        printSquareMatrix(U, n);
        printf("\n");
    }

    // Clean up GPU resources
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_L));
    checkCudaErrors(cudaFree(d_U));
    checkCudaErrors(cudaFree(d_LU));
    checkCudaErrors(cudaFree(d_diff));
    checkCudaErrors(cublasDestroy(cublasHandle));
    checkCudaErrors(cudaStreamDestroy(stream));
    
    // Calculate elapsed time using system clock
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;
    
    printf("CUDA verification time (s): %.6f\n", elapsed.count());
    printf("GPU error = %.6f\n", error);

    // Consider the decomposition correct if error is less than threshold
    return error <= 1e-6;
}


// Perform LU decomposition using a single CUDA kernel call (non-tiled approach)
// This demonstrates the baseline implementation using cuSOLVER's getrf function
void trivialLU(bool verify)
{
    // Initialize timing and CUDA device
    memopt::SystemWallClock clock;
    clock.start();

    memopt::initializeCudaDevice(true); // Display device info
    
    clock.logWithCurrentTime("Initializing host data");
    
    // Initialize cuSOLVER library handles
    cusolverDnHandle_t cusolverDnHandle;
    checkCudaErrors(cusolverDnCreate(&cusolverDnHandle));

    cusolverDnParams_t cusolverDnParams;
    checkCudaErrors(cusolverDnCreateParams(&cusolverDnParams));

    // Create and initialize the input matrix - using pinned memory for better performance
    double *h_A = nullptr;
    checkCudaErrors(cudaMallocHost(&h_A, N * N * sizeof(double)));
    generateRandomSymmetricPositiveDefiniteMatrix(h_A, N);
    
    clock.logWithCurrentTime("Host data initialized");
    clock.logWithCurrentTime("Initializing device data");

    // Allocate device memory and copy the matrix
    double *d_A;
    checkCudaErrors(cudaMalloc(&d_A, N * N * sizeof(double)));
    checkCudaErrors(cudaMemcpy(d_A, h_A, N * N * sizeof(double), cudaMemcpyHostToDevice));
    
    clock.logWithCurrentTime("Device data initialized");

    // Query the required workspace size for LU decomposition
    size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
    checkCudaErrors(cusolverDnXgetrf_bufferSize(
        cusolverDnHandle,
        cusolverDnParams,
        N,
        N,
        CUDA_R_64F,
        d_A,
        N,
        CUDA_R_64F,
        &workspaceInBytesOnDevice,
        &workspaceInBytesOnHost));
        
    // Print workspace sizes
    fmt::print("Device workspace size: {:.2f} MB\n", workspaceInBytesOnDevice / (1024.0 * 1024.0));
    fmt::print("Host workspace size: {:.2f} MB\n", workspaceInBytesOnHost / (1024.0 * 1024.0));

    // Allocate workspace memory on host and device
    void *h_workspace = nullptr;
    checkCudaErrors(cudaMallocHost(&h_workspace, workspaceInBytesOnHost));
    void *d_workspace;
    checkCudaErrors(cudaMalloc(&d_workspace, workspaceInBytesOnDevice));

    // Allocate memory for status information
    int *d_info;
    checkCudaErrors(cudaMalloc(&d_info, sizeof(int)));
    memopt::CudaEventClock cudaEventClock;

    // Perform LU decomposition and measure time
    cudaEventClock.start();
    checkCudaErrors(cusolverDnXgetrf(
        cusolverDnHandle,
        cusolverDnParams,
        N,                  // Number of rows
        N,                  // Number of columns
        CUDA_R_64F,         // Data type of A
        d_A,                // Input/output matrix
        N,                  // Leading dimension
        NULL,               // No pivoting is used
        CUDA_R_64F,         // Data type for computation
        d_workspace,        // Device workspace
        workspaceInBytesOnDevice,
        h_workspace,        // Host workspace
        workspaceInBytesOnHost,
        d_info));           // Status information
    cudaEventClock.end();

    // Check if decomposition was successful
    int h_info = 0;
    checkCudaErrors(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0)
    {
        std::cout << "Unsuccessful potrf execution\n\n"
                  << "d_info = " << h_info << "\n\n";
    }

    // Verify the decomposition if requested
    if (verify) {
        double *h_L = nullptr;
        double *h_U = nullptr;
        checkCudaErrors(cudaMallocHost(&h_L, N * N * sizeof(double)));
        checkCudaErrors(cudaMallocHost(&h_U, N * N * sizeof(double)));
        
        // Copy result back to host
        checkCudaErrors(cudaMemcpy(h_L, d_A, N * N * sizeof(double), cudaMemcpyDeviceToHost));
        
        // Extract L and U matrices from the result
        cleanCusolverLUDecompositionResult(h_L, h_U, N);
        
        if (forceGpuVerify || N > 1000) {
            // Use the faster GPU-based verification
            fmt::print("Using GPU-accelerated verification\n");
            fmt::print("Result passes verification: {}\n", verifyLUDecompositionGPU(h_A, h_L, h_U, N));
        } else {
            // For small matrices, CPU verification is fine
            fmt::print("Using CPU verification\n");
            fmt::print("Result passes verification: {}\n", verifyLUDecomposition(h_A, h_L, h_U, N));
        }

        // Clean up verification memory
        checkCudaErrors(cudaFreeHost(h_L));
        checkCudaErrors(cudaFreeHost(h_U));
    }
    
    // Print execution time
    fmt::print("Total time used (s): {:.4f}\n", cudaEventClock.getTimeInSeconds());

    // Clean up resources
    checkCudaErrors(cudaFreeHost(h_A));
    checkCudaErrors(cudaFreeHost(h_workspace));
    checkCudaErrors(cusolverDnDestroy(cusolverDnHandle));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_workspace));
    checkCudaErrors(cudaFree(d_info));
}

// Define a type to represent a tile within the matrix
// First int = row index of tile, Second int = column index of tile
typedef std::pair<int, int> MatrixTile;

// Class for creating a CUDA graph that represents the dependencies between operations on matrix tiles
// This is used to build a graph for the tiled LU decomposition algorithm with proper dependencies
class TiledLUGraphCreator
{
public:
    // Constructor initializes the graph capture mechanism
    TiledLUGraphCreator(cudaStream_t stream, cudaGraph_t graph) : stream(stream), graph(graph)
    {
        this->lastModifiedTile = std::make_pair(-1, -1);  // Invalid initial value
    }
    
    // Begin capturing a CUDA operation that writes to one tile and reads from several others
    // This establishes dependencies to ensure correct execution order based on data flow
    void beginCaptureOperation(MatrixTile tileToWrite, std::initializer_list<MatrixTile> tilesToRead)
    {
        // Track all tiles involved in this operation
        auto tiles = std::vector<MatrixTile>(tilesToRead);
        tiles.push_back(tileToWrite);  // Also include the output tile
        
        // Get graph nodes that previously modified these tiles (dependencies)
        auto dependencies = this->getDependencies(tiles);

        // Store state for endCaptureOperation
        this->lastModifiedTile = tileToWrite;
        this->lastDependencies = dependencies;

        // Begin capturing operations into the CUDA graph with proper dependencies
        checkCudaErrors(cudaStreamBeginCaptureToGraph(this->stream, this->graph, dependencies.data(), nullptr, dependencies.size(), cudaStreamCaptureModeGlobal));
    }

    // End the capture of a CUDA operation and update dependency tracking
    void endCaptureOperation()
    {
        // Verify that we're in a valid capture state
        assert(this->lastModifiedTile.first != -1 && this->lastModifiedTile.second != -1);
        
        // Finish the graph capture
        checkCudaErrors(cudaStreamEndCapture(this->stream, &this->graph));
        
        // Record that this tile was last modified by the captured operation
        this->tileLastModifiedByMap[this->lastModifiedTile] = this->getTailOfLastCapturedNodeChain();
        
        // Reset the state
        this->lastModifiedTile = std::make_pair(-1, -1);
    };

private:
    // Maps each tile to the graph node that last modified it
    std::map<MatrixTile, cudaGraphNode_t> tileLastModifiedByMap;
    
    // Tracks which graph nodes have been visited during graph traversal
    std::map<cudaGraphNode_t, bool> visited;
    
    // CUDA stream used for operations
    cudaStream_t stream;
    
    // The CUDA graph being constructed
    cudaGraph_t graph;
    
    // The tile being modified in the current capture
    MatrixTile lastModifiedTile;
    
    // Dependencies for the current capture
    std::vector<cudaGraphNode_t> lastDependencies;

    // Get the graph nodes that represent dependencies for the specified tiles
    std::vector<cudaGraphNode_t> getDependencies(std::vector<MatrixTile> tiles)
    {
        std::vector<cudaGraphNode_t> dependencies;
        
        // For each tile, find the last operation that modified it
        for (auto tile : tiles)
        {
            auto it = this->tileLastModifiedByMap.find(tile);
            if (it != this->tileLastModifiedByMap.end())
            {
                dependencies.push_back(it->second);
            }
        }

        // Remove duplicate dependencies
        auto dedupedEnd = std::unique(dependencies.begin(), dependencies.end());
        dependencies.resize(std::distance(dependencies.begin(), dedupedEnd));
        return dependencies;
    }

    // Find the last node in the most recently captured chain of operations
    // This is used to track dependencies accurately when multiple operations modify the same tile
    cudaGraphNode_t getTailOfLastCapturedNodeChain()
    {
        if (lastDependencies.size() == 0)
        {
            // For the first operation with no dependencies, find the sink node of the graph
            size_t numEdges;
            checkCudaErrors(cudaGraphGetEdges(this->graph, nullptr, nullptr, &numEdges));
            auto from = std::make_unique<cudaGraphNode_t[]>(numEdges);
            auto to = std::make_unique<cudaGraphNode_t[]>(numEdges);
            checkCudaErrors(cudaGraphGetEdges(this->graph, from.get(), to.get(), &numEdges));

            // Find nodes with no outgoing edges (sink nodes)
            std::map<cudaGraphNode_t, bool> hasOutGoingEdge;
            std::set<cudaGraphNode_t> noOutGoingEdgeNodes;
            for (int i = 0; i < numEdges; i++)
            {
                hasOutGoingEdge[from[i]] = true;
                noOutGoingEdgeNodes.erase(from[i]);
                if (!hasOutGoingEdge[to[i]])
                    noOutGoingEdgeNodes.insert(to[i]);
            }

            // There should be exactly one sink node
            assert(noOutGoingEdgeNodes.size() == 1);

            return *noOutGoingEdgeNodes.begin();
        }
        else
        {
            // For operations with dependencies, find the path from the dependency
            // to the new operation, and return the last node in that path
            auto nodeBeforeChain = lastDependencies[0];
            size_t numDependentNodes;
            checkCudaErrors(cudaGraphNodeGetDependentNodes(nodeBeforeChain, nullptr, &numDependentNodes));

            assert(numDependentNodes > 0);

            auto dependentNodes = std::make_unique<cudaGraphNode_t[]>(numDependentNodes);
            checkCudaErrors(cudaGraphNodeGetDependentNodes(nodeBeforeChain, dependentNodes.get(), &numDependentNodes));

            // Find an unvisited dependent node (should be part of our new chain)
            cudaGraphNode_t chainBeginningNode;
            for (int i = 0; i < numDependentNodes; i++)
            {
                if (!visited[dependentNodes[i]])
                {
                    chainBeginningNode = dependentNodes[i];
                    break;
                }
            }

            // Follow the chain of nodes until reaching the end (sink node)
            auto u = chainBeginningNode;
            while (true)
            {
                visited[u] = true;
                checkCudaErrors(cudaGraphNodeGetDependentNodes(u, nullptr, &numDependentNodes));
                if (numDependentNodes == 0)
                    break;  // Reached the end of the chain

                // Each node in our chain should have exactly one dependent
                assert(numDependentNodes == 1);

                cudaGraphNode_t v;
                checkCudaErrors(cudaGraphNodeGetDependentNodes(u, &v, &numDependentNodes));
                u = v;  // Move to the next node in the chain
            }

            return u;  // Return the last node in the chain
        }
    }
};

// Forward declaration of tiledLU_Optimized
void tiledLU_Optimized(bool verify);

// Perform LU decomposition using a tiled approach with CUDA Graphs
// This demonstrates an implementation that operates on matrix tiles
// and builds a CUDA graph to represent the computations and their dependencies
void tiledLU(bool verify)
{
    // Initialize timing and CUDA device
    memopt::SystemWallClock clock;
    clock.start();
    
    memopt::initializeCudaDevice(true); // Display device info
    
    clock.logWithCurrentTime("Initializing host data");
    
    // Initialize data - create the original matrix with pinned memory
    double* originalMatrix = nullptr;
    checkCudaErrors(cudaMallocHost(&originalMatrix, N * N * sizeof(double))); // Column-major format
    generateRandomSymmetricPositiveDefiniteMatrix(originalMatrix, N);
    
    clock.logWithCurrentTime("Host data initialized");
    clock.logWithCurrentTime("Initializing device data");
    
    // Copy matrix to device using unified memory for simplicity
    double *d_matrix;
    checkCudaErrors(cudaMallocManaged(&d_matrix, N * N * sizeof(double)));
    checkCudaErrors(cudaMemcpy(d_matrix, originalMatrix, N * N * sizeof(double), cudaMemcpyHostToDevice));
    
    clock.logWithCurrentTime("Device data initialized");

    // Helper lambda to get a pointer to a specific block/tile within the matrix
    // Row i, column j of the tile grid (each tile is B×B elements)
    auto getMatrixBlock = [&](int i, int j)
    {
        return d_matrix + i * B + j * B * N; // Locate the start of the block
    };

    // Initialize CUDA libraries
    cusolverDnHandle_t cusolverDnHandle;
    cusolverDnParams_t cusolverDnParams;
    cublasHandle_t cublasHandle;
    checkCudaErrors(cusolverDnCreate(&cusolverDnHandle));
    checkCudaErrors(cusolverDnCreateParams(&cusolverDnParams));
    checkCudaErrors(cublasCreate(&cublasHandle));

    // Constants for matrix operations
    // Alternatives that were considered but not used:
    // double *one, *minusOne;
    // checkCudaErrors(cudaMallocManaged(&one, sizeof(double)));
    // checkCudaErrors(cudaMallocManaged(&minusOne, sizeof(double)));
    double one = 1.0;        // For scaling operations
    double minusOne = -1.0;  // For subtraction in GEMM

    // Calculate required workspace for LU decomposition
    size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
    checkCudaErrors(cusolverDnXgetrf_bufferSize(
        cusolverDnHandle,
        cusolverDnParams,
        B,                  // Tile size (rows)
        B,                  // Tile size (columns)
        CUDA_R_64F,         // Data type
        d_matrix,           // Matrix pointer
        N,                  // Leading dimension
        CUDA_R_64F,         // Compute type
        &workspaceInBytesOnDevice,
        &workspaceInBytesOnHost));
    
    // Print workspace sizes for tiled implementation
    fmt::print("Tiled implementation - Device workspace size: {:.2f} MB\n", workspaceInBytesOnDevice / (1024.0 * 1024.0));
    fmt::print("Tiled implementation - Host workspace size: {:.2f} MB\n", workspaceInBytesOnHost / (1024.0 * 1024.0));

    // Allocate workspace memory
    void *h_workspace, *d_workspace;
    int *d_info;  // For error status
    checkCudaErrors(cudaMallocManaged(&h_workspace, workspaceInBytesOnHost));
    checkCudaErrors(cudaMallocManaged(&d_workspace, workspaceInBytesOnDevice));
    checkCudaErrors(cudaMallocManaged(&d_info, sizeof(int)));

    // Create CUDA graph and stream
    cudaGraph_t graph;
    checkCudaErrors(cudaGraphCreate(&graph, 0));

    cudaStream_t s;
    checkCudaErrors(cudaStreamCreate(&s));

    // Associate libraries with the stream
    checkCudaErrors(cusolverDnSetStream(cusolverDnHandle, s));
    checkCudaErrors(cublasSetStream(cublasHandle, s));
    // Not used: checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace, workspaceInBytesOnDevice));

    // Create helper for building the tiled LU graph
    auto tiledLUGraphCreator = std::make_unique<TiledLUGraphCreator>(s, graph);

    // The tiled LU decomposition algorithm is expressed using CUDA graph operations
    // It follows the standard block LU decomposition with right-looking updates
    for (int k = 0; k < T; k++)
    {
        // Step 1: LU decomposition of diagonal block A[k][k] (becomes L[k][k] and U[k][k])
        // A[k][k] = GETRF(A[k][k])
        // L[k][k]*U[k][k] = A[k][k]
        tiledLUGraphCreator->beginCaptureOperation(
            std::make_pair(k, k),           // This operation modifies block (k,k)
            {std::make_pair(k, k)});        // And reads from block (k,k)
        checkCudaErrors(cusolverDnXgetrf(
            cusolverDnHandle,
            cusolverDnParams,
            B, B,                           // Tile dimensions
            CUDA_R_64F,                     // Data type
            getMatrixBlock(k, k),           // Input/output matrix block
            N,                              // Leading dimension
            NULL,                           // No pivoting used
            CUDA_R_64F,                     // Compute type
            d_workspace,
            workspaceInBytesOnDevice,
            h_workspace,
            workspaceInBytesOnHost,
            d_info));
        tiledLUGraphCreator->endCaptureOperation();

        // Step 2: Update column blocks below diagonal
        for (int i = k + 1; i < T; i++)
        {
            // Solve for L[i][k] using the upper triangular part of A[k][k]
            // L[i][k] = A[i][k] / U[k][k]
            tiledLUGraphCreator->beginCaptureOperation(
                std::make_pair(k, i),                        // Modifies block (k,i)
                {std::make_pair(k, k), std::make_pair(k, i)}); // Reads from blocks (k,k) and (k,i)
            checkCudaErrors(cublasDtrsm(
                cublasHandle,
                CUBLAS_SIDE_LEFT,           // Apply operation from left
                CUBLAS_FILL_MODE_LOWER,      // A[k][k] is lower triangular
                CUBLAS_OP_N,                // No transpose
                CUBLAS_DIAG_UNIT,           // Unit diagonal (LU specific)
                B, B,                       // Tile dimensions
                &one,                       // Scalar alpha
                getMatrixBlock(k, k), N,    // Triangle matrix
                getMatrixBlock(k, i), N));  // Right-hand side and result
            tiledLUGraphCreator->endCaptureOperation();
        }

        // Step 3: Update row blocks to the right of diagonal
        for (int i = k + 1; i < T; i++)
        {
            // Solve for U[k][i] using the lower triangular part of A[k][k]
            // U[k][i] = L[k][k]⁻¹ * A[k][i]
            tiledLUGraphCreator->beginCaptureOperation(
                std::make_pair(i, k),                        // Modifies block (i,k)
                {std::make_pair(k, k), std::make_pair(i, k)}); // Reads from blocks (k,k) and (i,k)
            checkCudaErrors(cublasDtrsm(
                cublasHandle,
                CUBLAS_SIDE_RIGHT,          // Apply operation from right
                CUBLAS_FILL_MODE_UPPER,      // A[k][k] is upper triangular
                CUBLAS_OP_N,                // No transpose 
                CUBLAS_DIAG_NON_UNIT,       // Non-unit diagonal (LU specific)
                B, B,                       // Tile dimensions
                &one,                       // Scalar alpha
                getMatrixBlock(k, k), N,    // Triangle matrix
                getMatrixBlock(i, k), N));  // Right-hand side and result
            tiledLUGraphCreator->endCaptureOperation();

            // Step 4: Update the remaining blocks in the trailing submatrix
            for (int j = k + 1; j < T; j++)
            {
                // Update A[i][j] with the outer product of L[i][k] and U[k][j]
                // A[i][j] = A[i][j] - L[i][k] * U[k][j]
                tiledLUGraphCreator->beginCaptureOperation(
                    std::make_pair(i, j),                                               // Modifies block (i,j)
                    {std::make_pair(i, k), std::make_pair(k, j), std::make_pair(i, j)}); // Reads from blocks (i,k), (k,j), and (i,j)
                checkCudaErrors(cublasGemmEx(
                    cublasHandle,
                    CUBLAS_OP_N,              // No transpose for first matrix
                    CUBLAS_OP_N,              // No transpose for second matrix
                    B, B, B,                  // Dimensions m,n,k
                    &minusOne,                // Alpha (negative for subtraction)
                    getMatrixBlock(i, k), CUDA_R_64F, N,  // First matrix (L[i][k])
                    getMatrixBlock(k, j), CUDA_R_64F, N,  // Second matrix (U[k][j])
                    &one,                     // Beta (add to existing values)
                    getMatrixBlock(i, j), CUDA_R_64F, N,  // Result matrix (A[i][j])
                    CUBLAS_COMPUTE_64F,       // Compute precision
                    CUBLAS_GEMM_DEFAULT));    // Algorithm selection
                tiledLUGraphCreator->endCaptureOperation();
            }
        }
    }

    // Export the graph as a DOT file for visualization
    clock.logWithCurrentTime("Graph recorded");
    if (ConfigurationManager::getConfig().execution.enableDebugOutput) {
      checkCudaErrors(cudaGraphDebugDotPrint(graph, "./graph.dot", 0));
    }
    clock.logWithCurrentTime("Graph printed");

    // Instantiate the graph for execution
    cudaGraphExec_t graphExec;
    checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    
    // Execute the graph and measure time
    clock.logWithCurrentTime("Start execution");
    memopt::CudaEventClock cudaEventClock;
    cudaEventClock.start(s);
    checkCudaErrors(cudaGraphLaunch(graphExec, s));
    checkCudaErrors(cudaStreamSynchronize(s));
    cudaEventClock.end(s);
    checkCudaErrors(cudaDeviceSynchronize());
    clock.logWithCurrentTime("Execution completed");

    // Verify the result if requested
    if (verify) {
        clock.logWithCurrentTime("Starting verification");
        double *h_U = nullptr;
        checkCudaErrors(cudaMallocHost(&h_U, N * N * sizeof(double)));
        memset(h_U, 0, N * N * sizeof(double));
        cleanCusolverLUDecompositionResult(d_matrix, h_U, N);
        
        extern bool forceGpuVerify;
        
        if (forceGpuVerify || N > 1000) {
            // Use the faster GPU-based verification
            fmt::print("Using GPU-accelerated verification\n");
            fmt::print("Result passes verification: {}\n", verifyLUDecompositionGPU(originalMatrix, d_matrix, h_U, N));
        } else {
            // For small matrices, CPU verification is fine
            fmt::print("Using CPU verification\n");
            fmt::print("Result passes verification: {}\n", verifyLUDecomposition(originalMatrix, d_matrix, h_U, N));
        }
        
        checkCudaErrors(cudaFreeHost(h_U));
        clock.logWithCurrentTime("Verification completed");
    }
    
    // Report performance
    fmt::print("Total time used (s): {:.4f}\n", cudaEventClock.getTimeInSeconds());

    // Clean up resources
    checkCudaErrors(cudaFreeHost(originalMatrix));
    checkCudaErrors(cudaFreeHost(h_workspace));
    checkCudaErrors(cudaFree(d_matrix));
    checkCudaErrors(cudaFree(d_workspace));
}

// Perform LU decomposition using a tiled approach with memory optimization
// This implementation uses the PointerDependencyCudaGraphConstructor and TaskManager_v2
void tiledLU_Optimized(bool verify) {
  // Configuration check
  if (!ConfigurationManager::getConfig().generic.optimize) {
    LOG_TRACE_WITH_INFO("This function requires optimization to be enabled");
    return;
  }

  // Initialize timing and CUDA device
  memopt::SystemWallClock clock;
  clock.start();

  memopt::initializeCudaDevice(true); // Display device info

  const size_t tileSize = B * B * sizeof(double);

  // SECTION 1: HOST DATA INITIALIZATION
  clock.logWithCurrentTime("Initializing host data");
  
  double* h_originalMatrix = nullptr;
  checkCudaErrors(cudaMallocHost(&h_originalMatrix, N * N * sizeof(double)));  // Column-major, using pinned memory
  
  // Generate input matrix
  generateRandomSymmetricPositiveDefiniteMatrix(h_originalMatrix, N);
  
  clock.logWithCurrentTime("Host data initialized");

  // SECTION 2: DEVICE MEMORY ALLOCATION
  clock.logWithCurrentTime("Initializing device data");
  std::vector<double *> d_tiles(T * T);
  for (int i = 0; i < T * T; i++) {
    if (ConfigurationManager::getConfig().generic.useUM) {
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
      MemoryManager::getInstance().registerApplicationInput(getMatrixBlock(i, j));
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

  // Calculate required workspace for LU decomposition
  size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
  checkCudaErrors(cusolverDnXgetrf_bufferSize(
    cusolverDnHandle,
    cusolverDnParams,
    B,
    B,
    CUDA_R_64F,
    d_tiles[0],
    B,
    CUDA_R_64F,
    &workspaceInBytesOnDevice,
    &workspaceInBytesOnHost
  ));
  
  // Print workspace sizes for optimized implementation
  fmt::print("Optimized implementation - Device workspace size: {:.2f} MB\n", workspaceInBytesOnDevice / (1024.0 * 1024.0));
  fmt::print("Optimized implementation - Host workspace size: {:.2f} MB\n", workspaceInBytesOnHost / (1024.0 * 1024.0));

  void *h_workspace = nullptr, *d_workspace;
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
  TaskManager_v2 tmanager_v2(ConfigurationManager::getConfig().execution.enableDebugOutput);

  // SECTION 6: TILED LU ALGORITHM IMPLEMENTATION USING POINTER-BASED DEPENDENCY TRACKING
  auto graphConstructor = std::make_unique<PointerDependencyCudaGraphConstructor>(s, graph);

  for (int k = 0; k < T; k++) {
    // =====================================================================
    // STEP 1: GETRF - LU factorization of diagonal tile
    // =====================================================================
    
    // Define input and output pointers for dependency tracking
    std::vector<void*> inputs = {static_cast<void*>(getMatrixBlock(k, k))};
    std::vector<void*> outputs = {static_cast<void*>(getMatrixBlock(k, k))};
    
    // Begin capturing operation with pointer-based dependencies
    graphConstructor->beginCaptureOperation(inputs, outputs);

    // Register and execute the GETRF task
    TaskId taskId_v2 = tmanager_v2.registerTask<std::function<void(cudaStream_t, double*)>, double*>(
      [cusolverDnHandle, cusolverDnParams, d_workspace, workspaceInBytesOnDevice, 
       h_workspace, workspaceInBytesOnHost, d_info](cudaStream_t stream, double* matrixblock_k_k) {
        checkCudaErrors(cusolverDnSetStream(cusolverDnHandle, stream));

        checkCudaErrors(cusolverDnXgetrf(
            cusolverDnHandle,
            cusolverDnParams,
            B,               // Matrix size
            B,               // Matrix size
            CUDA_R_64F,              // Data type (double)
            matrixblock_k_k,         // Input/output matrix
            B,               // Leading dimension
            NULL,                    // No pivoting
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
      "GETRF_task"  // task name
    );
    
    tmanager_v2.execute(taskId_v2, s);
    graphConstructor->endCaptureOperation();

    // =====================================================================
    // STEP 2: Update column blocks below diagonal (L)
    // =====================================================================
    for (int i = k + 1; i < T; i++) {
      // Define input and output pointers for dependency tracking
      inputs = {static_cast<void*>(getMatrixBlock(k, k)), static_cast<void*>(getMatrixBlock(i, k))};
      outputs = {static_cast<void*>(getMatrixBlock(i, k))};
      
      // Begin capturing operation with pointer-based dependencies
      graphConstructor->beginCaptureOperation(inputs, outputs);
      
      // Register and execute the TRSM task for L
      TaskId trsmLTaskId = tmanager_v2.registerTask<std::function<void(cudaStream_t, double*, double*)>, double*, double*>(
        [cublasHandle, one](cudaStream_t stream, double* matrixblock_k_k, double* matrixblock_i_k) {
          checkCudaErrors(cublasSetStream(cublasHandle, stream));
          checkCudaErrors(cublasDtrsm(
            cublasHandle,
            CUBLAS_SIDE_RIGHT,      // Multiply from right
            CUBLAS_FILL_MODE_UPPER, // Upper triangular
            CUBLAS_OP_N,            // No transpose
            CUBLAS_DIAG_NON_UNIT,   // Non-unit diagonal
            B, B,                   // Matrix dimensions
            one,                    // Alpha = 1.0
            matrixblock_k_k, B,     // Triangular matrix
            matrixblock_i_k, B      // Input/output matrix
          ));
        },
        inputs,  // inputs
        outputs,  // outputs
        TaskManager_v2::makeArgs(getMatrixBlock(k, k), getMatrixBlock(i, k)),  // default args
        "TRSM_L_task_" + std::to_string(i) + "_" + std::to_string(k)  // task name
      );
      
      tmanager_v2.execute(trsmLTaskId, s);
      graphConstructor->endCaptureOperation();
    }

    // =====================================================================
    // STEP 3: Update row blocks to the right of diagonal (U)
    // =====================================================================
    for (int j = k + 1; j < T; j++) {
      // Define input and output pointers for dependency tracking
      inputs = {static_cast<void*>(getMatrixBlock(k, k)), static_cast<void*>(getMatrixBlock(k, j))};
      outputs = {static_cast<void*>(getMatrixBlock(k, j))};
      
      // Begin capturing operation with pointer-based dependencies
      graphConstructor->beginCaptureOperation(inputs, outputs);
      
      // Register and execute the TRSM task for U
      TaskId trsmUTaskId = tmanager_v2.registerTask<std::function<void(cudaStream_t, double*, double*)>, double*, double*>(
        [cublasHandle, one](cudaStream_t stream, double* matrixblock_k_k, double* matrixblock_k_j) {
          checkCudaErrors(cublasSetStream(cublasHandle, stream));
          checkCudaErrors(cublasDtrsm(
            cublasHandle,
            CUBLAS_SIDE_LEFT,       // Multiply from left
            CUBLAS_FILL_MODE_LOWER, // Lower triangular
            CUBLAS_OP_N,            // No transpose
            CUBLAS_DIAG_UNIT,       // Unit diagonal
            B, B,                   // Matrix dimensions
            one,                    // Alpha = 1.0
            matrixblock_k_k, B,     // Triangular matrix
            matrixblock_k_j, B      // Input/output matrix
          ));
        },
        inputs,  // inputs
        outputs,  // outputs
        TaskManager_v2::makeArgs(getMatrixBlock(k, k), getMatrixBlock(k, j)),  // default args
        "TRSM_U_task_" + std::to_string(k) + "_" + std::to_string(j)  // task name
      );
      
      tmanager_v2.execute(trsmUTaskId, s);
      graphConstructor->endCaptureOperation();
    }

    // =====================================================================
    // STEP 4: Update the remaining blocks in the trailing submatrix
    // =====================================================================
    for (int i = k + 1; i < T; i++) {
      for (int j = k + 1; j < T; j++) {
        // Define input and output pointers for dependency tracking
        inputs = {
          static_cast<void*>(getMatrixBlock(i, k)), 
          static_cast<void*>(getMatrixBlock(k, j)), 
          static_cast<void*>(getMatrixBlock(i, j))
        };
        outputs = {static_cast<void*>(getMatrixBlock(i, j))};
        
        // Begin capturing operation with pointer-based dependencies
        graphConstructor->beginCaptureOperation(inputs, outputs);
        
        // Register and execute the GEMM task
        TaskId gemmTaskId = tmanager_v2.registerTask<std::function<void(cudaStream_t, double*, double*, double*)>, double*, double*, double*>(
          [cublasHandle, minusOne, one](cudaStream_t stream, double* matrixblock_i_k, double* matrixblock_k_j, double* matrixblock_i_j) {
            // General matrix multiplication using cuBLAS
            checkCudaErrors(cublasSetStream(cublasHandle, stream));
            checkCudaErrors(cublasGemmEx(
              cublasHandle,
              CUBLAS_OP_N,           // No transpose for first matrix
              CUBLAS_OP_N,           // No transpose second matrix
              B, B, B,               // Matrix dimensions
              minusOne,              // Alpha = -1.0
              matrixblock_i_k, CUDA_R_64F, B, // First input matrix
              matrixblock_k_j, CUDA_R_64F, B, // Second input matrix
              one,                   // Beta = 1.0
              matrixblock_i_j, CUDA_R_64F, B, // Input/output matrix
              CUBLAS_COMPUTE_64F,    // Computation precision
              CUBLAS_GEMM_DEFAULT    // Algorithm selection
            ));
          },
          inputs,  // inputs
          outputs,  // outputs
          TaskManager_v2::makeArgs(getMatrixBlock(i, k), getMatrixBlock(k, j), getMatrixBlock(i, j)),  // default args
          "GEMM_task_" + std::to_string(i) + "_" + std::to_string(j) + "_" + std::to_string(k)  // task name
        );
        
        tmanager_v2.execute(gemmTaskId, s);
        graphConstructor->endCaptureOperation();
      }
    }
  }

  // Graph creation completed - now we can export it for debugging
  clock.logWithCurrentTime("Graph recorded");
  if (ConfigurationManager::getConfig().execution.enableDebugOutput) {
    LOG_TRACE_WITH_INFO("Printing original graph to graph.dot");
    checkCudaErrors(cudaGraphDebugDotPrint(graph, "./graph.dot", 0));
  }

  clock.logWithCurrentTime("Graph printed");

  // SECTION 7: OPTIMIZED EXECUTION
  tmanager_v2.setExecutionMode(TaskManager_v2::ExecutionMode::Production);
  
  auto optimizedGraph = profileAndOptimize(graph);

  // Copy data to matrix tiles
  for (int i = 0; i < T; i++) {
    for (int j = 0; j < T; j++) {
      for (int k = 0; k < B; k++) {
        checkCudaErrors(cudaMemcpy(
          MemoryManager::getInstance().getAddress(getMatrixBlock(i, j)) + B * k,
          h_originalMatrix + N * (j * B + k) + B * i,
          B * sizeof(double),
          cudaMemcpyDefault
        ));
      }
    }
  }
  checkCudaErrors(cudaDeviceSynchronize());
  
  // Run the optimized graph
  for (int i = 0; i < ConfigurationManager::getConfig().generic.repeat; i++) {
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
    fmt::print("Finalized iteration {}\n", i+1);
  }
    
  // Verification of the result if requested
  if (verify) {
    clock.logWithCurrentTime("Starting verification");
    
    // Reconstruct the full matrix from tiles for verification
    double *h_resultMatrix = nullptr;
    double *h_U = nullptr;
    checkCudaErrors(cudaMallocHost(&h_resultMatrix, N * N * sizeof(double)));
    checkCudaErrors(cudaMallocHost(&h_U, N * N * sizeof(double)));
    
    // Copy data from device tiles to host result matrix
    for (int i = 0; i < T; i++) {
      for (int j = 0; j < T; j++) {
        for (int k = 0; k < B; k++) {
          checkCudaErrors(cudaMemcpy(
            h_resultMatrix + N * (j * B + k) + B * i,
            MemoryManager::getInstance().getAddress(getMatrixBlock(i, j)) + B * k,
            B * sizeof(double),
            cudaMemcpyDeviceToHost
          ));
        }
      }
    }
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Extract L and U matrices from the result
    memset(h_U, 0, N * N * sizeof(double));

    cleanCusolverLUDecompositionResult(h_resultMatrix, h_U, N);
    double totalManagedMemoryMB = MemoryManager::getInstance().GetMemoryManagedSizeInMB();
    fmt::print("[MEMORY-INFO] Total managed memory size: {:.2f} MB\n", totalManagedMemoryMB);  

    extern bool forceGpuVerify;
    
    if (forceGpuVerify || N > 1000) {
      // Use the faster GPU-based verification
      fmt::print("Using GPU-accelerated verification\n");
      fmt::print("Result passes verification: {}\n", 
                verifyLUDecompositionGPU(h_originalMatrix, h_resultMatrix, h_U, N));
    } else {
      // For small matrices, CPU verification is fine
      fmt::print("Using CPU verification\n");
      fmt::print("Result passes verification: {}\n", 
                verifyLUDecomposition(h_originalMatrix, h_resultMatrix, h_U, N));
    }
    
    // Clean up verification memory
    checkCudaErrors(cudaFreeHost(h_resultMatrix));
    checkCudaErrors(cudaFreeHost(h_U));
    clock.logWithCurrentTime("Verification completed");
  }

  // Cleanup
  checkCudaErrors(cudaFreeHost(h_workspace));
  checkCudaErrors(cudaFree(d_workspace));
  checkCudaErrors(cudaFree(d_info));
  checkCudaErrors(cudaFree(one));
  checkCudaErrors(cudaFree(minusOne));
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
  
  bool useGpuVerify = cmdl["use-gpu-verify"];

  ConfigurationManager::exportDefaultConfiguration();
  ConfigurationManager::loadConfiguration(configFilePath);

  N = ConfigurationManager::getConfig().lu.n;
  T = ConfigurationManager::getConfig().lu.t;
  B = N / T;

  fmt::print("LU Decomposition with N={}, T={}, B={}\n", N, T, B);
  
  // Use GPU for verification if requested or matrix is large
  bool verifyEnabled = ConfigurationManager::getConfig().generic.verify;
  // Set the global flag for GPU verification
  forceGpuVerify = useGpuVerify || N > 1000;

  // Determine which implementation to use
  if (cmdl["trivial"] || ConfigurationManager::getConfig().lu.mode == Configuration::LU::Mode::trivial) {
    fmt::print("Using single-kernel (trivial) implementation\n");
    trivialLU(verifyEnabled);
  } else {
    if (ConfigurationManager::getConfig().generic.optimize) {
      fmt::print("Using tiled implementation with memory optimization\n");
      tiledLU_Optimized(verifyEnabled);
    } else {
      fmt::print("Using tiled implementation without optimization\n");
      tiledLU(verifyEnabled);
    }
  }
  
  return 0;
}