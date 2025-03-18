#include <cublas_v2.h>      // CUDA BLAS library for linear algebra operations
#include <cuda_runtime.h>    // Core CUDA runtime API
#include <cuda_runtime_api.h> // Extended CUDA runtime API
#include <cusolverDn.h>      // CUDA solver for dense matrices
#include <fmt/core.h>     // Commented out formatting library

#include <algorithm>         // STL algorithms
#include <cassert>           // Assert macro for error checking
#include <cmath>             // Math functions
#include <cstdlib>           // Standard library functions
#include <initializer_list>  // Initializer list support
#include <iomanip>           // IO formatting
#include <iostream>          // Standard IO streams
#include <limits>            // Numeric limits
#include <map>               // Map container
#include <memory>            // Smart pointers
#include <set>               // Set container
#include <tuple>             // Tuple support
#include <vector>            // Vector container

#include "../include/argh.h" // Command line argument parsing
// #include "../utilities/cudaUtilities.hpp" // Commented out
#include "memopt.hpp"

using namespace memopt;

// Matrix dimensions and tiling parameters
// N: Total matrix size (N x N)
// B: Block/tile size
// T: Number of tiles per dimension (T = N/B)
size_t N = 15 * 1;           // Default matrix size (15x15)
size_t B = N / 5;            // Default block size derived from N
size_t T = N / B;            // Default number of tiles


// Error checking function for CUDA API calls
// Checks if the result code indicates an error, and if so, prints the error and exits
// template <typename T>
// void __check(T result, char const *const func, const char *const file, int const line)
// {
//     if (result)
//     {
//         fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n", file, line, static_cast<unsigned int>(result), func);
//         exit(EXIT_FAILURE);
//     }
// }

// Macro for checking CUDA errors
// Passes the result of a CUDA function call to __check along with source information
#define checkCudaErrors(val) __check((val), #val, __FILE__, __LINE__)

// Simple CUDA kernel to warm up the GPU
// This avoids timing fluctuations for the first CUDA operations
// __global__ void warmUp()
// {
//     unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     float ia, ib;
//     ia = ib = 0.0f;
//     ib += ia + static_cast<float>(tid);
// }

// Executes a warm-up kernel on the GPU
// This initializes the CUDA runtime and makes subsequent kernel launches more consistent in timing
// void warmUpCudaDevice()
// {
//     warmUp<<<32, 32>>>();
//     cudaDeviceSynchronize();
// }

// Set up the CUDA device and optionally display its information
// This function selects GPU 0, prints its properties if requested, and warms it up
void initializeCudaDevice(bool displayDeviceInfo)
{
    // Select device 0 (first GPU)
    checkCudaErrors(cudaSetDevice(0));

    if (displayDeviceInfo)
    {
        cudaDeviceProp deviceProp;
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));
        printf("GPU Device %d: %s\n", 0, deviceProp.name);
        printf("Compute Capability: %d.%d\n\n", deviceProp.major, deviceProp.minor);
    }

    warmUpCudaDevice();
}

// Class for measuring execution time of CUDA operations using CUDA events
// This provides more accurate GPU timing than CPU-based timers
class LUCudaEventClock
{
public:
    // Constructor - creates timing events
    LUCudaEventClock() {
        checkCudaErrors(cudaEventCreate(&startEvent));
        checkCudaErrors(cudaEventCreate(&endEvent));
    }
    
    // Destructor - cleans up events
    ~LUCudaEventClock() {
        checkCudaErrors(cudaEventDestroy(startEvent));
        checkCudaErrors(cudaEventDestroy(endEvent));
    }
    
    // Mark the start time on a specific stream
    void start(cudaStream_t stream = 0) {
        checkCudaErrors(cudaEventRecord(startEvent, stream));
    }
    
    // Mark the end time on a specific stream
    void end(cudaStream_t stream = 0) {
        checkCudaErrors(cudaEventRecord(endEvent, stream));
    }
    
    // Calculate elapsed time in seconds
    float getTimeInSeconds() {
        float time;
        checkCudaErrors(cudaEventElapsedTime(&time, startEvent, endEvent));
        return time * 1e-3f;  // Convert milliseconds to seconds
    }

private:
    cudaEvent_t startEvent, endEvent;           // CUDA events for timing
};


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
    // Use fixed seed for reproducibility
    // srand(time(NULL));
    srand(420);

    // Temporary matrix for initial random values
    double *h_A_temp = (double *)malloc(n * n * sizeof(double));

    // Fill with random values between 0 and 1
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            h_A_temp[i * n + j] = (float)rand() / (float)RAND_MAX;

    // Make the matrix symmetric by averaging with its transpose
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            h_A[i * n + j] = 0.5 * (h_A_temp[i * n + j] + h_A_temp[j * n + i]);

    // Add n to diagonal elements to ensure positive definiteness
    // This shifts all eigenvalues by n, making them positive
    for (int i = 0; i < n; i++)
        h_A[i * n + i] = h_A[i * n + i] + n;
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

    // Print matrices for inspection
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

    printf("error = %.6f}\n", error);

    // Consider the decomposition correct if error is less than threshold
    return error <= 1e-6;
}

// Perform LU decomposition using a single CUDA kernel call (non-tiled approach)
// This demonstrates the baseline implementation using cuSOLVER's getrf function
void trivialLU(bool verify)
{
    // Initialize cuSOLVER library handles
    cusolverDnHandle_t cusolverDnHandle;
    checkCudaErrors(cusolverDnCreate(&cusolverDnHandle));

    cusolverDnParams_t cusolverDnParams;
    checkCudaErrors(cusolverDnCreateParams(&cusolverDnParams));

    // Create and initialize the input matrix
    double *h_A = (double *)malloc(N * N * sizeof(double));
    generateRandomSymmetricPositiveDefiniteMatrix(h_A, N);

    // Allocate device memory and copy the matrix
    double *d_A;
    checkCudaErrors(cudaMalloc(&d_A, N * N * sizeof(double)));
    checkCudaErrors(cudaMemcpy(d_A, h_A, N * N * sizeof(double), cudaMemcpyHostToDevice));

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

    // Allocate workspace memory on host and device
    void *h_workspace = malloc(workspaceInBytesOnHost);
    void *d_workspace;
    checkCudaErrors(cudaMalloc(&d_workspace, workspaceInBytesOnDevice));

    // Allocate memory for status information
    int *d_info;
    checkCudaErrors(cudaMalloc(&d_info, sizeof(int)));
    LUCudaEventClock clock;

    // Perform LU decomposition and measure time
    clock.start();
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
    clock.end();

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
        double *h_L = (double *)malloc(N * N * sizeof(double));
        double *h_U = (double *)malloc(N * N * sizeof(double));
        
        // Copy result back to host
        checkCudaErrors(cudaMemcpy(h_L, d_A, N * N * sizeof(double), cudaMemcpyDeviceToHost));
        
        // Extract L and U matrices from the result
        cleanCusolverLUDecompositionResult(h_L, h_U, N);
        
        // Verify L*U = A
        printf("Result passes verification: %d\n", verifyLUDecomposition(h_A, h_L, h_U, N));

        // Clean up verification memory
        free(h_L);
        free(h_U);
    }
    
    // Print execution time
    printf("Total time used (s): %4.4f\n", clock.getTimeInSeconds());

    // Clean up resources
    free(h_A);
    free(h_workspace);
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

// Perform LU decomposition using a tiled approach with CUDA Graphs
// This demonstrates an optimized implementation that operates on matrix tiles
// and builds a CUDA graph to represent the computations and their dependencies
void tiledLU(bool verify)
{
    // Initialize data - create the original matrix
    auto originalMatrix = std::make_unique<double[]>(N * N); // Column-major format
    generateRandomSymmetricPositiveDefiniteMatrix(originalMatrix.get(), N);

    // Copy matrix to device using unified memory for simplicity
    double *d_matrix;
    checkCudaErrors(cudaMallocManaged(&d_matrix, N * N * sizeof(double)));
    checkCudaErrors(cudaMemcpy(d_matrix, originalMatrix.get(), N * N * sizeof(double), cudaMemcpyHostToDevice));

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

    // Timer for performance measurement
    LUCudaEventClock clock;
    
    // Export the graph as a DOT file for visualization
    checkCudaErrors(cudaGraphDebugDotPrint(graph, "./graph.dot", 0));

    // Instantiate the graph for execution
    cudaGraphExec_t graphExec;
    checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    
    // Execute the graph and measure time
    clock.start(s);
    checkCudaErrors(cudaGraphLaunch(graphExec, s));
    checkCudaErrors(cudaStreamSynchronize(s));
    clock.end(s);
    checkCudaErrors(cudaDeviceSynchronize());

    // Verify the result if requested
    if (verify) {
        double *h_U = (double *)malloc(N * N * sizeof(double));
        memset(h_U, 0, N * N * sizeof(double));
        cleanCusolverLUDecompositionResult(d_matrix, h_U, N);
        printf("Result passes verification: %d\n", verifyLUDecomposition(originalMatrix.get(), d_matrix, h_U, N));
        free(h_U);
    }
    
    // Report performance
    printf("Total time used (s): %4.4f\n", clock.getTimeInSeconds());

    // Clean up resources
    free(h_workspace);
    cudaFree(d_matrix);
    cudaFree(d_workspace);
}

// Main entry point for LU decomposition
// Selects between tiled and non-tiled implementations based on parameter
// Parameters:
//   tiled - if true, use tiled implementation; if false, use single-kernel implementation
//   verify - if true, verify the correctness of the result
void LU(bool tiled, bool verify)
{
    if (tiled)
    {
        tiledLU(verify);
    }
    else
    {
        trivialLU(verify);
    }
}

// Main program entry point
// Parses command line arguments and runs the selected LU decomposition method
int main(int argc, char **argv)
{
    // Set up command line argument parsing
    argh::parser cmdl({"n", "N", "t", "T"});
    cmdl.parse(argc, argv);

    // Parse matrix size N
    if (!(cmdl({"N", "n"}, N) >> N)) {
        std::cerr << "Must provide a valid N value! Got '" << cmdl({"N", "n"}).str() << "'" << std::endl;
        return 0;
    }
    
    // Parse number of tiles T
    if (!(cmdl({"t", "T"}, T) >> T)) {
        std::cerr << "Must provide a valid T value! Got '" << cmdl({"T", "t"}).str() << "'" << std::endl;
        return 0;
    }
    
    // Ensure N is divisible by T for proper tiling
    if (N % T > 0) {
        std::cerr << "N must be divisible by T! Got 'N=" << N << " & T=" << T << "'" << std::endl;
        return 0;
    }
    
    // Calculate tile size B
    B = N / T;
    
    // Print configuration information
    if (cmdl["tiled"])
        std::cout << "TILED ";
    else
        std::cout << "Single-kernel ";
    std::cout << "with 'N=" << N << " & T=" << T << " & B=" << B << "'" << std::endl;

    // Execute LU decomposition with the selected options
    LU(cmdl["tiled"], cmdl["verify"]);

    return 0;
}