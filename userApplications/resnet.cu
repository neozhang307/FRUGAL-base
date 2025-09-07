#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <cudnn.h>
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

// Global configuration
size_t BATCH_SIZE = 1;
size_t NUM_CLASSES = 10;
size_t INPUT_HEIGHT = 32;
size_t INPUT_WIDTH = 32;
size_t INPUT_CHANNELS = 3;

// ResNet layer structure (based on verified resnet_simple.cu)
struct ResNetLayer {
    size_t in_channels;
    size_t out_channels;
    size_t input_h;
    size_t input_w;
    size_t kernel_size;
    size_t stride;
    size_t padding;
    size_t output_h;
    size_t output_w;
};

// ResNet architecture definition (proven correct in resnet_simple.cu)
std::vector<ResNetLayer> resnet_layers = {
    // Layer 0: 3->16, 32x32 -> 32x32
    {3, 16, 32, 32, 3, 1, 1, 32, 32},
    // Layer 1: 16->32, 32x32 -> 16x16 (with stride 2)
    {16, 32, 32, 32, 3, 2, 1, 16, 16},
    // Layer 2: 32->64, 16x16 -> 8x8 (with stride 2) 
    {32, 64, 16, 16, 3, 2, 1, 8, 8},
};

// CUDA kernels (from verified resnet_simple.cu)
__global__ void normalizeInput(float* input, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        input[idx] = input[idx] / 255.0f;
    }
}

__global__ void relu_activation(float* data, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

// Helper functions (from verified resnet_simple.cu)
cudnnTensorDescriptor_t createTensorDescriptor(int n, int c, int h, int w) {
    cudnnTensorDescriptor_t desc;
    checkCudaErrors(cudnnCreateTensorDescriptor(&desc));
    checkCudaErrors(cudnnSetTensorNdDescriptor(desc, CUDNN_DATA_FLOAT, 4,
        new int[4]{n, c, h, w}, new int[4]{c*h*w, h*w, w, 1}));
    return desc;
}

cudnnFilterDescriptor_t createFilterDescriptor(int k, int c, int h, int w) {
    cudnnFilterDescriptor_t desc;
    checkCudaErrors(cudnnCreateFilterDescriptor(&desc));
    checkCudaErrors(cudnnSetFilterNdDescriptor(desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4,
        new int[4]{k, c, h, w}));
    return desc;
}

cudnnConvolutionDescriptor_t createConvDescriptor(int pad_h, int pad_w, int stride_h, int stride_w) {
    cudnnConvolutionDescriptor_t desc;
    checkCudaErrors(cudnnCreateConvolutionDescriptor(&desc));
    checkCudaErrors(cudnnSetConvolution2dDescriptor(desc, pad_h, pad_w, stride_h, stride_w, 1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    return desc;
}

cudnnPoolingDescriptor_t createPoolingDescriptor(int window_h, int window_w, int pad_h, int pad_w, int stride_h, int stride_w) {
    cudnnPoolingDescriptor_t desc;
    checkCudaErrors(cudnnCreatePoolingDescriptor(&desc));
    checkCudaErrors(cudnnSetPooling2dDescriptor(desc, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN,
        window_h, window_w, pad_h, pad_w, stride_h, stride_w));
    return desc;
}

void initializeRandomData(float* data, size_t size, float range = 255.0f) {
    for (size_t i = 0; i < size; i++) {
        data[i] = (float(rand()) / RAND_MAX) * range;
    }
}

void initializeWeights(float* weights, size_t size, float std_dev = 0.1f) {
    for (size_t i = 0; i < size; i++) {
        // Simple normal distribution approximation
        float u1 = float(rand()) / RAND_MAX;
        float u2 = float(rand()) / RAND_MAX;
        float z = sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI * u2);
        weights[i] = z * std_dev;
    }
}

void resnetNaiveGraph() {
    fmt::print("=== ResNet Naive Graph Demo ===\n");
    fmt::print("Input: {}x{}x{}x{} (batch x channels x height x width)\n", 
               BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH);
    fmt::print("Architecture: {} layers\n", resnet_layers.size());
    
    initializeCudaDevice();

    // =========================================================================
    // PHASE 1: SETUP AND ALLOCATE MEMORY
    // =========================================================================
    fmt::print("\n--- PHASE 1: Setup and Allocate Memory ---\n");
    
    auto& memManager = MemoryManager::getInstance();
    
    // Initialize cuDNN and cuBLAS
    cudnnHandle_t cudnnHandle;
    cublasHandle_t cublasHandle;
    checkCudaErrors(cudnnCreate(&cudnnHandle));
    checkCudaErrors(cublasCreate(&cublasHandle));
    
    // Input tensor
    size_t input_size = BATCH_SIZE * INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH;
    float* d_input;
    checkCudaErrors(cudaMalloc(&d_input, input_size * sizeof(float)));
    memManager.registerManagedMemoryAddress(d_input, input_size * sizeof(float));
    
    // Initialize input data
    std::vector<float> h_input(input_size);
    srand(42); // Fixed seed for consistency
    initializeRandomData(h_input.data(), input_size, 255.0f);
    checkCudaErrors(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice));
    
    fmt::print("Allocated input tensor: {:.2f} MB\n", 
               (input_size * sizeof(float)) / (1024.0 * 1024.0));
    
    // Allocate memory for each layer
    std::vector<float*> layer_outputs;
    std::vector<float*> layer_weights;
    std::vector<float*> layer_biases;
    
    size_t total_memory = input_size * sizeof(float);
    
    for (size_t i = 0; i < resnet_layers.size(); i++) {
        const auto& layer = resnet_layers[i];
        
        // Output tensor
        size_t output_size = BATCH_SIZE * layer.out_channels * layer.output_h * layer.output_w;
        float* d_output;
        checkCudaErrors(cudaMalloc(&d_output, output_size * sizeof(float)));
        memManager.registerManagedMemoryAddress(d_output, output_size * sizeof(float));
        layer_outputs.push_back(d_output);
        total_memory += output_size * sizeof(float);
        
        // Weight tensor
        size_t weight_size = layer.out_channels * layer.in_channels * layer.kernel_size * layer.kernel_size;
        float* d_weights;
        checkCudaErrors(cudaMalloc(&d_weights, weight_size * sizeof(float)));
        memManager.registerManagedMemoryAddress(d_weights, weight_size * sizeof(float));
        layer_weights.push_back(d_weights);
        total_memory += weight_size * sizeof(float);
        
        // Initialize weights
        std::vector<float> h_weights(weight_size);
        float std_dev = sqrt(2.0f / (layer.in_channels * layer.kernel_size * layer.kernel_size)); // He initialization
        initializeWeights(h_weights.data(), weight_size, std_dev);
        checkCudaErrors(cudaMemcpy(d_weights, h_weights.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice));
        
        // Bias tensor (initialize to zero)
        size_t bias_size = layer.out_channels;
        float* d_bias;
        checkCudaErrors(cudaMalloc(&d_bias, bias_size * sizeof(float)));
        memManager.registerManagedMemoryAddress(d_bias, bias_size * sizeof(float));
        checkCudaErrors(cudaMemset(d_bias, 0, bias_size * sizeof(float)));
        layer_biases.push_back(d_bias);
        total_memory += bias_size * sizeof(float);
        
        fmt::print("Layer {}: {}->{}x{}x{}, weights: {:.2f}KB, output: {:.2f}KB\n", 
                   i, layer.in_channels, layer.out_channels, layer.output_h, layer.output_w,
                   (weight_size * sizeof(float)) / 1024.0,
                   (output_size * sizeof(float)) / 1024.0);
    }
    
    // Final classification layer (GAP + FC)
    const auto& last_layer = resnet_layers.back();
    size_t fc_input_size = BATCH_SIZE * last_layer.out_channels;  
    size_t fc_output_size = BATCH_SIZE * NUM_CLASSES;
    
    float* d_gap_output;
    float* d_fc_weights;
    float* d_fc_bias;
    float* d_fc_output;
    
    checkCudaErrors(cudaMalloc(&d_gap_output, fc_input_size * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc_weights, last_layer.out_channels * NUM_CLASSES * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc_bias, NUM_CLASSES * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc_output, fc_output_size * sizeof(float)));
    
    memManager.registerManagedMemoryAddress(d_gap_output, fc_input_size * sizeof(float));
    memManager.registerManagedMemoryAddress(d_fc_weights, last_layer.out_channels * NUM_CLASSES * sizeof(float));
    memManager.registerManagedMemoryAddress(d_fc_bias, NUM_CLASSES * sizeof(float));
    memManager.registerManagedMemoryAddress(d_fc_output, fc_output_size * sizeof(float));
    
    // Initialize FC weights and bias
    std::vector<float> h_fc_weights(last_layer.out_channels * NUM_CLASSES);
    initializeWeights(h_fc_weights.data(), h_fc_weights.size(), 0.01f);
    checkCudaErrors(cudaMemcpy(d_fc_weights, h_fc_weights.data(), h_fc_weights.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_fc_bias, 0, NUM_CLASSES * sizeof(float)));
    
    total_memory += (h_fc_weights.size() + NUM_CLASSES + fc_output_size + fc_input_size) * sizeof(float);
    
    double totalManagedMemoryMB = memManager.GetMemoryManagedSizeInMB();
    fmt::print("Total managed memory: {:.2f} MB\n", totalManagedMemoryMB);

    // =========================================================================
    // PHASE 2: BUILD NAIVE CUDA GRAPH USING TASKMANAGER_V2
    // =========================================================================
    fmt::print("\n--- PHASE 2: Build Naive CUDA Graph ---\n");
    
    cudaStream_t s;
    checkCudaErrors(cudaStreamCreate(&s));
    checkCudaErrors(cudnnSetStream(cudnnHandle, s));
    checkCudaErrors(cublasSetStream(cublasHandle, s));

    // Initialize TaskManager_v2 for naive graph construction (like in tiledCholeskyNaiveGraph.cu)
    TaskManager_v2 tmanager_v2(true);

    // Register all tasks in the correct dependency order
    fmt::print("Registering tasks for ResNet inference...\n");
    
    // Task 1: Simple memory operation to ensure graph has nodes
    TaskId normalizeTaskId = tmanager_v2.registerTask<std::function<void(cudaStream_t, float*, size_t)>, float*, size_t>(
        [](cudaStream_t stream, float* input, size_t size) {
            // Simple memset operation to ensure graph capture
            checkCudaErrors(cudaMemsetAsync(input, 0, sizeof(float), stream));
            size_t numThreads = 256;
            size_t numBlocks = (size + numThreads - 1) / numThreads;
            normalizeInput<<<numBlocks, numThreads, 0, stream>>>(input, size);
        },
        {static_cast<void*>(d_input)},
        {static_cast<void*>(d_input)},
        TaskManager_v2::makeArgs(d_input, input_size),
        "normalize_input"
    );
    fmt::print("✓ Registered input normalization task\n");
    
    // Tasks 2-4: Convolution layers with proper dependency chain
    float* current_input = d_input;
    for (size_t i = 0; i < resnet_layers.size(); i++) {
        const auto& layer = resnet_layers[i];
        float* current_output = layer_outputs[i];
        float* current_weights = layer_weights[i];
        float* current_bias = layer_biases[i];
        
        // Convolution task
        TaskId convTaskId = tmanager_v2.registerTask<std::function<void(cudaStream_t, cudnnHandle_t, float*, float*, float*, float*, size_t)>, 
                                                      cudnnHandle_t, float*, float*, float*, float*, size_t>(
            [](cudaStream_t stream, cudnnHandle_t handle, float* input, float* weights, float* bias, float* output, size_t layer_idx) {
                const auto& layer = resnet_layers[layer_idx];
                
                // Create cuDNN descriptors
                auto input_desc = createTensorDescriptor(BATCH_SIZE, layer.in_channels, layer.input_h, layer.input_w);
                auto filter_desc = createFilterDescriptor(layer.out_channels, layer.in_channels, layer.kernel_size, layer.kernel_size);
                auto output_desc = createTensorDescriptor(BATCH_SIZE, layer.out_channels, layer.output_h, layer.output_w);
                auto conv_desc = createConvDescriptor(layer.padding, layer.padding, layer.stride, layer.stride);
                
                // Find best convolution algorithm
                cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
                
                // Get workspace size
                size_t workspace_size = 0;
                checkCudaErrors(cudnnGetConvolutionForwardWorkspaceSize(handle, input_desc, filter_desc, conv_desc, output_desc, algo, &workspace_size));
                
                void* workspace = nullptr;
                if (workspace_size > 0) {
                    checkCudaErrors(cudaMalloc(&workspace, workspace_size));
                }
                
                // Perform convolution
                const float alpha = 1.0f, beta = 0.0f;
                checkCudaErrors(cudnnConvolutionForward(handle, &alpha, input_desc, input, 
                    filter_desc, weights, conv_desc, algo, workspace, workspace_size, &beta, output_desc, output));
                
                // Add bias
                auto bias_desc = createTensorDescriptor(1, layer.out_channels, 1, 1);
                checkCudaErrors(cudnnAddTensor(handle, &alpha, bias_desc, bias, &alpha, output_desc, output));
                
                // Apply ReLU activation
                size_t output_size = BATCH_SIZE * layer.out_channels * layer.output_h * layer.output_w;
                size_t numThreads = 256;
                size_t numBlocks = (output_size + numThreads - 1) / numThreads;
                relu_activation<<<numBlocks, numThreads, 0, stream>>>(output, output_size);
                
                // Cleanup
                if (workspace) checkCudaErrors(cudaFree(workspace));
                checkCudaErrors(cudnnDestroyTensorDescriptor(input_desc));
                checkCudaErrors(cudnnDestroyFilterDescriptor(filter_desc));
                checkCudaErrors(cudnnDestroyTensorDescriptor(output_desc));
                checkCudaErrors(cudnnDestroyConvolutionDescriptor(conv_desc));
                checkCudaErrors(cudnnDestroyTensorDescriptor(bias_desc));
            },
            {static_cast<void*>(current_input), static_cast<void*>(current_weights), static_cast<void*>(current_bias)},
            {static_cast<void*>(current_output)},
            TaskManager_v2::makeArgs(cudnnHandle, current_input, current_weights, current_bias, current_output, i),
            "conv_layer_" + std::to_string(i)
        );
        
        fmt::print("✓ Registered convolution layer {} task\n", i);
        current_input = current_output; // Update for next layer to create dependency chain
    }
    
    // Task 5: Global Average Pooling
    TaskId gapTaskId = tmanager_v2.registerTask<std::function<void(cudaStream_t, cudnnHandle_t, float*, float*)>, 
                                                cudnnHandle_t, float*, float*>(
        [](cudaStream_t stream, cudnnHandle_t handle, float* input, float* output) {
            const auto& last_layer = resnet_layers.back();
            auto input_desc = createTensorDescriptor(BATCH_SIZE, last_layer.out_channels, last_layer.output_h, last_layer.output_w);
            auto output_desc = createTensorDescriptor(BATCH_SIZE, last_layer.out_channels, 1, 1);
            auto pool_desc = createPoolingDescriptor(last_layer.output_h, last_layer.output_w, 0, 0, 1, 1);
            
            const float alpha = 1.0f, beta = 0.0f;
            checkCudaErrors(cudnnPoolingForward(handle, pool_desc, &alpha, input_desc, input, &beta, output_desc, output));
            
            // Cleanup
            checkCudaErrors(cudnnDestroyTensorDescriptor(input_desc));
            checkCudaErrors(cudnnDestroyTensorDescriptor(output_desc));
            checkCudaErrors(cudnnDestroyPoolingDescriptor(pool_desc));
        },
        {static_cast<void*>(current_input)},
        {static_cast<void*>(d_gap_output)},
        TaskManager_v2::makeArgs(cudnnHandle, current_input, d_gap_output),
        "global_avg_pool"
    );
    fmt::print("✓ Registered Global Average Pooling task\n");
    
    // Task 6: Final classification layer
    TaskId fcTaskId = tmanager_v2.registerTask<std::function<void(cudaStream_t, cublasHandle_t, float*, float*, float*, float*)>, 
                                               cublasHandle_t, float*, float*, float*, float*>(
        [](cudaStream_t stream, cublasHandle_t handle, float* input, float* weights, float* bias, float* output) {
            const auto& last_layer = resnet_layers.back();
            const float alpha = 1.0f, beta = 0.0f;
            
            // output = weights^T * input + bias
            checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, NUM_CLASSES, BATCH_SIZE, last_layer.out_channels,
                &alpha, weights, last_layer.out_channels, input, last_layer.out_channels, &beta, output, NUM_CLASSES));
            
            // Add bias (for each batch element)
            for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
                checkCudaErrors(cublasSaxpy(handle, NUM_CLASSES, &alpha, bias, 1, output + batch * NUM_CLASSES, 1));
            }
        },
        {static_cast<void*>(d_gap_output), static_cast<void*>(d_fc_weights), static_cast<void*>(d_fc_bias)},
        {static_cast<void*>(d_fc_output)},
        TaskManager_v2::makeArgs(cublasHandle, d_gap_output, d_fc_weights, d_fc_bias, d_fc_output),
        "final_classification"
    );
    fmt::print("✓ Registered final classification task\n");
    
    fmt::print("✅ Task registration completed! Total tasks: {}\n", tmanager_v2.taskCount());
    
    // Initialize data before graph generation to avoid corrupting graph state
    fmt::print("Initializing data for graph generation...\n");
    checkCudaErrors(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Generate naive CUDA graph from registered tasks (like in tiledCholeskyNaiveGraph.cu)
    fmt::print("Generating naive CUDA graph from task sequence...\n");
    cudaGraph_t graph = tmanager_v2.generateNaiveGraph(s);
    
    // Get graph stats
    size_t numNodes;
    checkCudaErrors(cudaGraphGetNodes(graph, nullptr, &numNodes));
    fmt::print("📊 Generated graph contains {} nodes\n", numNodes);

    // =========================================================================
    // PHASE 3: OPTIMIZE MEMORY ALLOCATION WITH NAIVE GRAPH
    // =========================================================================
    fmt::print("\n--- PHASE 3: Memory Optimization with Naive Graph ---\n");
    
    double initialPeakMemory = memManager.GetMemoryManagedSizeInMB();
    fmt::print("Initial peak memory: {:.2f} MB\n", initialPeakMemory);
    
    // Profile and optimize the naive graph
    // (Data already initialized before graph generation)
    fmt::print("Profiling and optimizing naive CUDA graph...\n");
    auto optimizedGraph = profileAndOptimize(graph);
    
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
    // PHASE 4: EXECUTE WITH OPTIMIZED NAIVE GRAPH
    // =========================================================================
    fmt::print("\n--- PHASE 4: Execute with Optimized Naive Graph ---\n");
    
    // Reinitialize data after optimization
    checkCudaErrors(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Get current GPU memory info
    size_t free_mem, total_mem;
    checkCudaErrors(cudaMemGetInfo(&free_mem, &total_mem));
    fmt::print("GPU Memory - Total: {:.2f} MB, Free: {:.2f} MB\n", 
               (double)total_mem / (1024.0 * 1024.0), (double)free_mem / (1024.0 * 1024.0));
    
    // Start peak memory monitoring
    fmt::print("🔍 Starting continuous GPU memory monitoring during execution...\n");
    PeakMemoryUsageProfiler peakProfiler(10); // Sample every 10ms
    peakProfiler.start();
    
    // Run the optimized naive graph (like in tiledCholeskyNaiveGraph.cu)
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
    
    fmt::print("✅ Optimized ResNet inference execution completed!\n");
    fmt::print("Execution time: {:.3f} ms\n", runningTime * 1000.0f);
    fmt::print("📊 Peak GPU memory usage during execution: {:.2f} MB\n", peakMemoryMB);
    
    // =========================================================================
    // PHASE 5: VERIFY RESULTS
    // =========================================================================
    fmt::print("\n--- PHASE 5: Verify Results ---\n");
    
    // Copy results to host for verification
    std::vector<float> h_output(fc_output_size);
    memManager.copyManagedArrayToHost(d_fc_output, h_output.data(), fc_output_size * sizeof(float));
    
    // Apply softmax and find top predictions for each batch
    for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
        std::vector<float> batch_logits(NUM_CLASSES);
        for (size_t i = 0; i < NUM_CLASSES; i++) {
            batch_logits[i] = h_output[batch * NUM_CLASSES + i];
        }
        
        // Apply softmax
        float max_logit = *std::max_element(batch_logits.begin(), batch_logits.end());
        float sum_exp = 0.0f;
        std::vector<float> softmax_probs(NUM_CLASSES);
        
        for (size_t i = 0; i < NUM_CLASSES; i++) {
            softmax_probs[i] = exp(batch_logits[i] - max_logit);
            sum_exp += softmax_probs[i];
        }
        
        for (size_t i = 0; i < NUM_CLASSES; i++) {
            softmax_probs[i] /= sum_exp;
        }
        
        // Find top prediction
        size_t top_class = std::distance(softmax_probs.begin(), 
            std::max_element(softmax_probs.begin(), softmax_probs.end()));
        
        fmt::print("Batch {} - Top prediction: Class {} with probability {:.2f}%\n", 
                   batch, top_class, softmax_probs[top_class] * 100.0f);
        
        // Basic sanity check
        float prob_sum = 0.0f;
        for (float p : softmax_probs) prob_sum += p;
        if (abs(prob_sum - 1.0f) > 1e-5) {
            fmt::print("❌ ERROR: Probabilities sum to {:.6f}, expected 1.0\n", prob_sum);
        }
    }
    
    fmt::print("✅ Results verification completed!\n");
    
    // =========================================================================
    // PHASE 6: CLEANUP
    // =========================================================================
    fmt::print("\n--- PHASE 6: Cleanup ---\n");
    
    checkCudaErrors(cudaGraphDestroy(graph));
    checkCudaErrors(cudaStreamDestroy(s));
    checkCudaErrors(cudnnDestroy(cudnnHandle));
    checkCudaErrors(cublasDestroy(cublasHandle));
    
    // Free all managed memory properly using MemoryManager (like in tiledCholeskyNaiveGraph.cu)
    memManager.freeManagedMemory(d_input);
    for (auto output : layer_outputs) {
        memManager.freeManagedMemory(output);
    }
    for (auto weight : layer_weights) {
        memManager.freeManagedMemory(weight);
    }
    for (auto bias : layer_biases) {
        memManager.freeManagedMemory(bias);
    }
    memManager.freeManagedMemory(d_gap_output);
    memManager.freeManagedMemory(d_fc_weights);
    memManager.freeManagedMemory(d_fc_bias);
    memManager.freeManagedMemory(d_fc_output);
    
    fmt::print("\nFinal result: SUCCESS\n");
}

int main(int argc, char *argv[]) {
    // Parse command line arguments (like in tiledCholeskyNaiveGraph.cu)
    BATCH_SIZE = (argc > 1) ? std::atoi(argv[1]) : 1;
    INPUT_HEIGHT = INPUT_WIDTH = (argc > 2) ? std::atoi(argv[2]) : 32;
    NUM_CLASSES = (argc > 3) ? std::atoi(argv[3]) : 10;
    
    // Validation
    if (BATCH_SIZE < 1 || BATCH_SIZE > 64) {
        fmt::print("ERROR: Batch size must be between 1 and 64\n");
        return -1;
    }
    
    if (INPUT_HEIGHT < 16 || INPUT_HEIGHT > 128) {
        fmt::print("ERROR: Input size must be between 16 and 128\n");
        return -1;
    }
    
    if (NUM_CLASSES < 2 || NUM_CLASSES > 1000) {
        fmt::print("ERROR: Number of classes must be between 2 and 1000\n");
        return -1;
    }
    
    // Update layer input dimensions based on actual input size
    if (INPUT_HEIGHT != 32 || INPUT_WIDTH != 32) {
        // Update first layer input dimensions
        resnet_layers[0].input_h = INPUT_HEIGHT;
        resnet_layers[0].input_w = INPUT_WIDTH;
        resnet_layers[0].output_h = INPUT_HEIGHT;
        resnet_layers[0].output_w = INPUT_WIDTH;
        
        // Recalculate subsequent layer dimensions
        for (size_t i = 1; i < resnet_layers.size(); i++) {
            auto& prev_layer = resnet_layers[i-1];
            auto& curr_layer = resnet_layers[i];
            curr_layer.input_h = prev_layer.output_h;
            curr_layer.input_w = prev_layer.output_w;
            curr_layer.output_h = (curr_layer.input_h + 2 * curr_layer.padding - curr_layer.kernel_size) / curr_layer.stride + 1;
            curr_layer.output_w = (curr_layer.input_w + 2 * curr_layer.padding - curr_layer.kernel_size) / curr_layer.stride + 1;
        }
    }
    
    ConfigurationManager::exportDefaultConfiguration();
    ConfigurationManager::loadConfiguration("config.json");
    
    fmt::print("Configuration: Batch={}, Input={}x{}x{}, Classes={}\n", 
               BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH, NUM_CLASSES);
    
    // Run the ResNet with naive graph version
    resnetNaiveGraph();
    
    return 0;
}