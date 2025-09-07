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
size_t BATCH_SIZE = 32;
size_t NUM_CLASSES = 1000;
size_t INPUT_HEIGHT = 224;
size_t INPUT_WIDTH = 224;
size_t INPUT_CHANNELS = 3;

// ResNet-18 layer dimensions
struct LayerDimensions {
    size_t in_channels;
    size_t out_channels;
    size_t height;
    size_t width;
    size_t kernel_size;
    size_t stride;
    size_t padding;
};

// ResNet-18 architecture definition
std::vector<LayerDimensions> resnet18_layers = {
    // Initial conv layer: 3->64, 224x224 -> 112x112
    {3, 64, 224, 224, 7, 2, 3},
    
    // Layer 1 blocks: 64->64, 56x56 (after maxpool)
    {64, 64, 56, 56, 3, 1, 1},
    {64, 64, 56, 56, 3, 1, 1},
    {64, 64, 56, 56, 3, 1, 1},
    {64, 64, 56, 56, 3, 1, 1},
    
    // Layer 2 blocks: 64->128, 28x28
    {64, 128, 56, 56, 3, 2, 1},   // First block with stride 2
    {128, 128, 28, 28, 3, 1, 1},
    {128, 128, 28, 28, 3, 1, 1},
    {128, 128, 28, 28, 3, 1, 1},
    
    // Layer 3 blocks: 128->256, 14x14
    {128, 256, 28, 28, 3, 2, 1},  // First block with stride 2
    {256, 256, 14, 14, 3, 1, 1},
    {256, 256, 14, 14, 3, 1, 1},
    {256, 256, 14, 14, 3, 1, 1},
    
    // Layer 4 blocks: 256->512, 7x7
    {256, 512, 14, 14, 3, 2, 1},  // First block with stride 2
    {512, 512, 7, 7, 3, 1, 1},
    {512, 512, 7, 7, 3, 1, 1},
    {512, 512, 7, 7, 3, 1, 1},
};

// CUDA kernels for data preprocessing and utility functions
__global__ void normalizeInput(float* input, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        size_t channel = (idx / (INPUT_HEIGHT * INPUT_WIDTH)) % INPUT_CHANNELS;
        float mean[3] = {0.485f, 0.456f, 0.406f};
        float std[3] = {0.229f, 0.224f, 0.225f};
        input[idx] = (input[idx] / 255.0f - mean[channel]) / std[channel];
    }
}

__global__ void relu_activation(float* data, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

__global__ void add_residual(float* output, const float* input, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] += input[idx];
    }
}

// Helper function to create cuDNN tensor descriptor
cudnnTensorDescriptor_t createTensorDescriptor(int n, int c, int h, int w) {
    cudnnTensorDescriptor_t desc;
    checkCudaErrors(cudnnCreateTensorDescriptor(&desc));
    checkCudaErrors(cudnnSetTensorNdDescriptor(desc, CUDNN_DATA_FLOAT, 4,
        new int[4]{n, c, h, w}, new int[4]{c*h*w, h*w, w, 1}));
    return desc;
}

// Helper function to create cuDNN filter descriptor
cudnnFilterDescriptor_t createFilterDescriptor(int k, int c, int h, int w) {
    cudnnFilterDescriptor_t desc;
    checkCudaErrors(cudnnCreateFilterDescriptor(&desc));
    checkCudaErrors(cudnnSetFilterNdDescriptor(desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4,
        new int[4]{k, c, h, w}));
    return desc;
}

// Helper function to create cuDNN convolution descriptor
cudnnConvolutionDescriptor_t createConvDescriptor(int pad_h, int pad_w, int stride_h, int stride_w) {
    cudnnConvolutionDescriptor_t desc;
    checkCudaErrors(cudnnCreateConvolutionDescriptor(&desc));
    checkCudaErrors(cudnnSetConvolution2dDescriptor(desc, pad_h, pad_w, stride_h, stride_w, 1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    return desc;
}

// Helper function to create cuDNN pooling descriptor
cudnnPoolingDescriptor_t createPoolingDescriptor(int window_h, int window_w, int stride_h, int stride_w) {
    cudnnPoolingDescriptor_t desc;
    checkCudaErrors(cudnnCreatePoolingDescriptor(&desc));
    checkCudaErrors(cudnnSetPooling2dDescriptor(desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,
        window_h, window_w, 0, 0, stride_h, stride_w));
    return desc;
}

void generateRandomInput(float* h_input, size_t size) {
    curandGenerator_t prng;
    float* d_temp;
    
    checkCudaErrors(cudaMalloc(&d_temp, size * sizeof(float)));
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());
    curandGenerateUniform(prng, d_temp, size);
    
    // Scale to [0, 255] range for image data
    size_t numThreads = 1024;
    size_t numBlocks = (size + numThreads - 1) / numThreads;
    
    // Scale to 0-255 range
    checkCudaErrors(cublasSscal(nullptr, size, &(const float){255.0f}, d_temp, 1));
    
    checkCudaErrors(cudaMemcpy(h_input, d_temp, size * sizeof(float), cudaMemcpyDefault));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(d_temp));
    curandDestroyGenerator(prng);
}

void initializeWeights(float* weights, size_t size, float std_dev) {
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());
    curandGenerateNormal(prng, weights, size, 0.0f, std_dev);
    curandDestroyGenerator(prng);
}

void resnetInference() {
    fmt::print("=== ResNet-18 Inference Demo ===\n");
    fmt::print("Input: {}x{}x{}x{} (batch x channels x height x width)\n", 
               BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH);
    fmt::print("Architecture: ResNet-18 with {} layers\n", resnet18_layers.size());
    
    initializeCudaDevice();
    
    // =========================================================================
    // PHASE 1: SETUP AND ALLOCATE MEMORY
    // =========================================================================
    fmt::print("\n--- PHASE 1: Setup and Allocate Memory ---\n");
    
    // Initialize cuDNN and cuBLAS
    cudnnHandle_t cudnnHandle;
    cublasHandle_t cublasHandle;
    checkCudaErrors(cudnnCreate(&cudnnHandle));
    checkCudaErrors(cublasCreate(&cublasHandle));
    
    auto& memManager = MemoryManager::getInstance();
    std::vector<float*> layer_outputs;
    std::vector<float*> layer_weights;
    std::vector<float*> layer_biases;
    
    // Calculate memory requirements for each layer
    size_t total_memory = 0;
    
    // Input tensor
    size_t input_size = BATCH_SIZE * INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH;
    float* d_input;
    checkCudaErrors(cudaMalloc(&d_input, input_size * sizeof(float)));
    memManager.registerManagedMemoryAddress(d_input, input_size * sizeof(float));
    total_memory += input_size * sizeof(float);
    
    fmt::print("Allocated input tensor: {:.2f} MB\n", 
               (input_size * sizeof(float)) / (1024.0 * 1024.0));
    
    // Allocate memory for each layer
    for (size_t i = 0; i < resnet18_layers.size(); i++) {
        const auto& layer = resnet18_layers[i];
        
        // Calculate output dimensions
        size_t out_h = (layer.height + 2 * layer.padding - layer.kernel_size) / layer.stride + 1;
        size_t out_w = (layer.width + 2 * layer.padding - layer.kernel_size) / layer.stride + 1;
        
        // Output tensor
        size_t output_size = BATCH_SIZE * layer.out_channels * out_h * out_w;
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
        
        // Bias tensor
        size_t bias_size = layer.out_channels;
        float* d_bias;
        checkCudaErrors(cudaMalloc(&d_bias, bias_size * sizeof(float)));
        memManager.registerManagedMemoryAddress(d_bias, bias_size * sizeof(float));
        layer_biases.push_back(d_bias);
        total_memory += bias_size * sizeof(float);
        
        fmt::print("Layer {}: {}->{}x{}x{}, weights: {:.2f}MB, output: {:.2f}MB\n", 
                   i, layer.in_channels, layer.out_channels, out_h, out_w,
                   (weight_size * sizeof(float)) / (1024.0 * 1024.0),
                   (output_size * sizeof(float)) / (1024.0 * 1024.0));
    }
    
    // Final classification layer (Global Average Pool + FC)
    size_t fc_input_size = BATCH_SIZE * 512;  // After GAP: 512 features
    size_t fc_output_size = BATCH_SIZE * NUM_CLASSES;
    float* d_fc_weights;
    float* d_fc_bias;
    float* d_fc_output;
    
    checkCudaErrors(cudaMalloc(&d_fc_weights, 512 * NUM_CLASSES * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc_bias, NUM_CLASSES * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc_output, fc_output_size * sizeof(float)));
    
    memManager.registerManagedMemoryAddress(d_fc_weights, 512 * NUM_CLASSES * sizeof(float));
    memManager.registerManagedMemoryAddress(d_fc_bias, NUM_CLASSES * sizeof(float));
    memManager.registerManagedMemoryAddress(d_fc_output, fc_output_size * sizeof(float));
    
    total_memory += (512 * NUM_CLASSES + NUM_CLASSES + fc_output_size) * sizeof(float);
    
    fmt::print("Total managed memory: {:.2f} MB\n", total_memory / (1024.0 * 1024.0));
    
    // Initialize input data and weights
    fmt::print("Initializing random input and weights...\n");
    float* h_input;
    checkCudaErrors(cudaMallocHost(&h_input, input_size * sizeof(float)));
    generateRandomInput(h_input, input_size);
    checkCudaErrors(cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize weights for all layers
    for (size_t i = 0; i < layer_weights.size(); i++) {
        const auto& layer = resnet18_layers[i];
        size_t weight_size = layer.out_channels * layer.in_channels * layer.kernel_size * layer.kernel_size;
        float std_dev = sqrtf(2.0f / (layer.in_channels * layer.kernel_size * layer.kernel_size)); // He initialization
        initializeWeights(layer_weights[i], weight_size, std_dev);
        
        // Initialize bias to zero
        checkCudaErrors(cudaMemset(layer_biases[i], 0, layer.out_channels * sizeof(float)));
    }
    
    // Initialize FC layer weights
    initializeWeights(d_fc_weights, 512 * NUM_CLASSES, 0.01f);
    checkCudaErrors(cudaMemset(d_fc_bias, 0, NUM_CLASSES * sizeof(float)));
    
    // =========================================================================
    // PHASE 2: BUILD CUDA GRAPH USING TASKMANAGER_V2
    // =========================================================================
    fmt::print("\n--- PHASE 2: Build CUDA Graph ---\n");
    
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));
    checkCudaErrors(cudnnSetStream(cudnnHandle, stream));
    checkCudaErrors(cublasSetStream(cublasHandle, stream));
    
    TaskManager_v2 taskManager(true);
    
    // Normalize input data
    TaskId normalizeTaskId = taskManager.registerTask<std::function<void(cudaStream_t, float*, size_t)>, float*, size_t>(
        [](cudaStream_t stream, float* input, size_t size) {
            size_t numThreads = 1024;
            size_t numBlocks = (size + numThreads - 1) / numThreads;
            normalizeInput<<<numBlocks, numThreads, 0, stream>>>(input, size);
        },
        {static_cast<void*>(d_input)},
        {static_cast<void*>(d_input)},
        TaskManager_v2::makeArgs(d_input, input_size),
        "normalize_input"
    );
    
    fmt::print("✓ Registered input normalization task\n");
    
    // First convolution layer (7x7, stride 2)
    float* current_input = d_input;
    float* current_output = layer_outputs[0];
    
    TaskId conv1TaskId = taskManager.registerTask<std::function<void(cudaStream_t, cudnnHandle_t, float*, float*, float*, size_t, size_t)>, 
                                                   cudnnHandle_t, float*, float*, float*, size_t, size_t>(
        [](cudaStream_t stream, cudnnHandle_t handle, float* input, float* weights, float* output, 
           size_t layer_idx, size_t batch_size) {
            // This is a simplified convolution - in practice, you'd use cudnnConvolutionForward
            // For demonstration, we'll use a placeholder that shows the structure
            const auto& layer = resnet18_layers[layer_idx];
            
            // Create descriptors
            auto input_desc = createTensorDescriptor(batch_size, layer.in_channels, layer.height, layer.width);
            auto filter_desc = createFilterDescriptor(layer.out_channels, layer.in_channels, layer.kernel_size, layer.kernel_size);
            
            size_t out_h = (layer.height + 2 * layer.padding - layer.kernel_size) / layer.stride + 1;
            size_t out_w = (layer.width + 2 * layer.padding - layer.kernel_size) / layer.stride + 1;
            auto output_desc = createTensorDescriptor(batch_size, layer.out_channels, out_h, out_w);
            
            auto conv_desc = createConvDescriptor(layer.padding, layer.padding, layer.stride, layer.stride);
            
            const float alpha = 1.0f, beta = 0.0f;
            
            // Find best algorithm
            cudnnConvolutionFwdAlgo_t algo;
            checkCudaErrors(cudnnGetConvolutionForwardAlgorithm(handle, input_desc, filter_desc, conv_desc, output_desc,
                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));
            
            // Get workspace size
            size_t workspace_size;
            checkCudaErrors(cudnnGetConvolutionForwardWorkspaceSize(handle, input_desc, filter_desc, conv_desc, output_desc, algo, &workspace_size));
            
            void* workspace = nullptr;
            if (workspace_size > 0) {
                checkCudaErrors(cudaMalloc(&workspace, workspace_size));
            }
            
            // Perform convolution
            checkCudaErrors(cudnnConvolutionForward(handle, &alpha, input_desc, input, filter_desc, weights,
                conv_desc, algo, workspace, workspace_size, &beta, output_desc, output));
            
            if (workspace) {
                checkCudaErrors(cudaFree(workspace));
            }
            
            // Cleanup descriptors
            checkCudaErrors(cudnnDestroyTensorDescriptor(input_desc));
            checkCudaErrors(cudnnDestroyFilterDescriptor(filter_desc));
            checkCudaErrors(cudnnDestroyTensorDescriptor(output_desc));
            checkCudaErrors(cudnnDestroyConvolutionDescriptor(conv_desc));
        },
        {static_cast<void*>(current_input), static_cast<void*>(layer_weights[0])},
        {static_cast<void*>(current_output)},
        TaskManager_v2::makeArgs(cudnnHandle, current_input, layer_weights[0], current_output, (size_t)0, BATCH_SIZE),
        "conv1_7x7"
    );
    
    // Add ReLU activation
    size_t out_h = (resnet18_layers[0].height + 2 * resnet18_layers[0].padding - resnet18_layers[0].kernel_size) / resnet18_layers[0].stride + 1;
    size_t out_w = (resnet18_layers[0].width + 2 * resnet18_layers[0].padding - resnet18_layers[0].kernel_size) / resnet18_layers[0].stride + 1;
    size_t activation_size = BATCH_SIZE * resnet18_layers[0].out_channels * out_h * out_w;
    
    TaskId relu1TaskId = taskManager.registerTask<std::function<void(cudaStream_t, float*, size_t)>, float*, size_t>(
        [](cudaStream_t stream, float* data, size_t size) {
            size_t numThreads = 1024;
            size_t numBlocks = (size + numThreads - 1) / numThreads;
            relu_activation<<<numBlocks, numThreads, 0, stream>>>(data, size);
        },
        {static_cast<void*>(current_output)},
        {static_cast<void*>(current_output)},
        TaskManager_v2::makeArgs(current_output, activation_size),
        "relu1"
    );
    
    fmt::print("✓ Registered conv1 + relu1 tasks\n");
    
    // Max pooling layer (3x3, stride 2)
    float* pool_output = nullptr;
    size_t pool_out_h = out_h / 2;
    size_t pool_out_w = out_w / 2;
    size_t pool_output_size = BATCH_SIZE * resnet18_layers[0].out_channels * pool_out_h * pool_out_w;
    checkCudaErrors(cudaMalloc(&pool_output, pool_output_size * sizeof(float)));
    memManager.registerManagedMemoryAddress(pool_output, pool_output_size * sizeof(float));
    
    TaskId maxpoolTaskId = taskManager.registerTask<std::function<void(cudaStream_t, cudnnHandle_t, float*, float*)>, 
                                                     cudnnHandle_t, float*, float*>(
        [out_h, out_w, pool_out_h, pool_out_w](cudaStream_t stream, cudnnHandle_t handle, float* input, float* output) {
            auto input_desc = createTensorDescriptor(BATCH_SIZE, 64, out_h, out_w);
            auto output_desc = createTensorDescriptor(BATCH_SIZE, 64, pool_out_h, pool_out_w);
            auto pool_desc = createPoolingDescriptor(3, 3, 2, 2);
            
            const float alpha = 1.0f, beta = 0.0f;
            checkCudaErrors(cudnnPoolingForward(handle, pool_desc, &alpha, input_desc, input, &beta, output_desc, output));
            
            checkCudaErrors(cudnnDestroyTensorDescriptor(input_desc));
            checkCudaErrors(cudnnDestroyTensorDescriptor(output_desc));
            checkCudaErrors(cudnnDestroyPoolingDescriptor(pool_desc));
        },
        {static_cast<void*>(current_output)},
        {static_cast<void*>(pool_output)},
        TaskManager_v2::makeArgs(cudnnHandle, current_output, pool_output),
        "maxpool"
    );
    
    fmt::print("✓ Registered maxpool task\n");
    
    // Register basic ResNet blocks (simplified version)
    current_input = pool_output;
    for (size_t i = 1; i < std::min(resnet18_layers.size(), (size_t)5); i++) {
        current_output = layer_outputs[i];
        
        TaskId blockTaskId = taskManager.registerTask<std::function<void(cudaStream_t, cudnnHandle_t, float*, float*, float*)>, 
                                                      cudnnHandle_t, float*, float*, float*>(
            [i](cudaStream_t stream, cudnnHandle_t handle, float* input, float* weights, float* output) {
                // Simplified basic block - in practice this would be more complex
                const auto& layer = resnet18_layers[i];
                
                auto input_desc = createTensorDescriptor(BATCH_SIZE, layer.in_channels, layer.height, layer.width);
                auto filter_desc = createFilterDescriptor(layer.out_channels, layer.in_channels, layer.kernel_size, layer.kernel_size);
                
                size_t out_h = (layer.height + 2 * layer.padding - layer.kernel_size) / layer.stride + 1;
                size_t out_w = (layer.width + 2 * layer.padding - layer.kernel_size) / layer.stride + 1;
                auto output_desc = createTensorDescriptor(BATCH_SIZE, layer.out_channels, out_h, out_w);
                
                auto conv_desc = createConvDescriptor(layer.padding, layer.padding, layer.stride, layer.stride);
                
                const float alpha = 1.0f, beta = 0.0f;
                
                cudnnConvolutionFwdAlgo_t algo;
                checkCudaErrors(cudnnGetConvolutionForwardAlgorithm(handle, input_desc, filter_desc, conv_desc, output_desc,
                    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));
                
                size_t workspace_size;
                checkCudaErrors(cudnnGetConvolutionForwardWorkspaceSize(handle, input_desc, filter_desc, conv_desc, output_desc, algo, &workspace_size));
                
                void* workspace = nullptr;
                if (workspace_size > 0) {
                    checkCudaErrors(cudaMalloc(&workspace, workspace_size));
                }
                
                checkCudaErrors(cudnnConvolutionForward(handle, &alpha, input_desc, input, filter_desc, weights,
                    conv_desc, algo, workspace, workspace_size, &beta, output_desc, output));
                
                if (workspace) {
                    checkCudaErrors(cudaFree(workspace));
                }
                
                checkCudaErrors(cudnnDestroyTensorDescriptor(input_desc));
                checkCudaErrors(cudnnDestroyFilterDescriptor(filter_desc));
                checkCudaErrors(cudnnDestroyTensorDescriptor(output_desc));
                checkCudaErrors(cudnnDestroyConvolutionDescriptor(conv_desc));
            },
            {static_cast<void*>(current_input), static_cast<void*>(layer_weights[i])},
            {static_cast<void*>(current_output)},
            TaskManager_v2::makeArgs(cudnnHandle, current_input, layer_weights[i], current_output),
            "resnet_block_" + std::to_string(i)
        );
        
        fmt::print("✓ Registered ResNet block {} task\n", i);
        current_input = current_output;
    }
    
    // Global Average Pooling
    float* gap_output;
    checkCudaErrors(cudaMalloc(&gap_output, fc_input_size * sizeof(float)));
    memManager.registerManagedMemoryAddress(gap_output, fc_input_size * sizeof(float));
    
    TaskId gapTaskId = taskManager.registerTask<std::function<void(cudaStream_t, cudnnHandle_t, float*, float*)>, 
                                                cudnnHandle_t, float*, float*>(
        [](cudaStream_t stream, cudnnHandle_t handle, float* input, float* output) {
            // Global Average Pooling: 7x7 -> 1x1
            auto input_desc = createTensorDescriptor(BATCH_SIZE, 512, 7, 7);
            auto output_desc = createTensorDescriptor(BATCH_SIZE, 512, 1, 1);
            auto pool_desc = createPoolingDescriptor(7, 7, 1, 1);
            
            const float alpha = 1.0f, beta = 0.0f;
            checkCudaErrors(cudnnPoolingForward(handle, pool_desc, &alpha, input_desc, input, &beta, output_desc, output));
            
            checkCudaErrors(cudnnDestroyTensorDescriptor(input_desc));
            checkCudaErrors(cudnnDestroyTensorDescriptor(output_desc));
            checkCudaErrors(cudnnDestroyPoolingDescriptor(pool_desc));
        },
        {static_cast<void*>(current_input)},
        {static_cast<void*>(gap_output)},
        TaskManager_v2::makeArgs(cudnnHandle, current_input, gap_output),
        "global_avg_pool"
    );
    
    // Final classification layer
    TaskId fcTaskId = taskManager.registerTask<std::function<void(cudaStream_t, cublasHandle_t, float*, float*, float*, float*)>, 
                                               cublasHandle_t, float*, float*, float*, float*>(
        [](cudaStream_t stream, cublasHandle_t handle, float* input, float* weights, float* bias, float* output) {
            const float alpha = 1.0f, beta = 0.0f, beta_bias = 1.0f;
            
            // output = weights^T * input + bias
            checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, NUM_CLASSES, BATCH_SIZE, 512,
                &alpha, weights, 512, input, 512, &beta, output, NUM_CLASSES));
            
            // Add bias
            checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, NUM_CLASSES, BATCH_SIZE, 1,
                &alpha, bias, NUM_CLASSES, &(const float){1.0f}, 1, &beta_bias, output, NUM_CLASSES));
        },
        {static_cast<void*>(gap_output), static_cast<void*>(d_fc_weights), static_cast<void*>(d_fc_bias)},
        {static_cast<void*>(d_fc_output)},
        TaskManager_v2::makeArgs(cublasHandle, gap_output, d_fc_weights, d_fc_bias, d_fc_output),
        "final_fc"
    );
    
    fmt::print("✓ Registered GAP + FC tasks\n");
    fmt::print("✅ Task registration completed! Total tasks: {}\n", taskManager.taskCount());
    
    // =========================================================================
    // PHASE 3: GENERATE AND OPTIMIZE CUDA GRAPH
    // =========================================================================
    fmt::print("\n--- PHASE 3: Generate and Optimize CUDA Graph ---\n");
    
    fmt::print("Generating CUDA graph from task sequence...\n");
    cudaGraph_t graph = taskManager.generateNaiveGraph(stream);
    
    size_t numNodes;
    checkCudaErrors(cudaGraphGetNodes(graph, nullptr, &numNodes));
    fmt::print("📊 Generated graph contains {} nodes\n", numNodes);
    
    double initialPeakMemory = memManager.GetMemoryManagedSizeInMB();
    fmt::print("Initial peak memory: {:.2f} MB\n", initialPeakMemory);
    
    fmt::print("Profiling and optimizing CUDA graph...\n");
    auto optimizedGraph = profileAndOptimize(graph);
    
    fmt::print("Original peak memory usage (MiB): {:.2f}\n", optimizedGraph.originalMemoryUsage);
    fmt::print("Optimized peak memory usage (MiB): {:.2f}\n", optimizedGraph.anticipatedPeakMemoryUsage);
    fmt::print("Memory reduction: {:.2f} MiB ({:.1f}%)\n", 
               optimizedGraph.originalMemoryUsage - optimizedGraph.anticipatedPeakMemoryUsage,
               ((optimizedGraph.originalMemoryUsage - optimizedGraph.anticipatedPeakMemoryUsage) / optimizedGraph.originalMemoryUsage) * 100);
    
    // Move data to storage for optimized execution
    fmt::print("Moving all data to storage for optimized execution...\n");
    memManager.offloadAllManagedMemoryToStorage();
    fmt::print("✅ All data moved to CPU/storage\n");
    
    // =========================================================================
    // PHASE 4: EXECUTE OPTIMIZED RESNET
    // =========================================================================
    fmt::print("\n--- PHASE 4: Execute Optimized ResNet ---\n");
    
    // Get current GPU memory info
    size_t free_mem, total_mem;
    checkCudaErrors(cudaMemGetInfo(&free_mem, &total_mem));
    fmt::print("GPU Memory - Total: {:.2f} MB, Free: {:.2f} MB\n", 
               (double)total_mem / (1024.0 * 1024.0), (double)free_mem / (1024.0 * 1024.0));
    
    // Start peak memory monitoring
    fmt::print("🔍 Starting continuous GPU memory monitoring during execution...\n");
    PeakMemoryUsageProfiler peakProfiler(10); // Sample every 10ms
    peakProfiler.start();
    
    // Execute optimized ResNet
    float runningTime;
    executeOptimizedGraph(
        optimizedGraph,
        [&taskManager](int taskId, std::map<void*, void*> addressMapping, cudaStream_t stream) {
            taskManager.execute(taskId, stream);
        },
        runningTime,
        memManager
    );
    
    size_t peakMemoryBytes = peakProfiler.end();
    double peakMemoryMB = (double)peakMemoryBytes / (1024.0 * 1024.0);
    
    fmt::print("✅ ResNet inference execution completed!\n");
    fmt::print("Execution time: {:.3f} ms\n", runningTime * 1000.0f);
    fmt::print("📊 Peak GPU memory usage during execution: {:.2f} MB\n", peakMemoryMB);
    
    // =========================================================================
    // PHASE 5: VERIFY OUTPUT AND CLEANUP
    // =========================================================================
    fmt::print("\n--- PHASE 5: Verify Output and Cleanup ---\n");
    
    // Copy results to host for verification
    float* h_output;
    checkCudaErrors(cudaMallocHost(&h_output, fc_output_size * sizeof(float)));
    memManager.copyManagedArrayToHost(d_fc_output, h_output, fc_output_size * sizeof(float));
    
    // Find top-5 predictions for first batch element
    std::vector<std::pair<float, int>> predictions;
    for (size_t i = 0; i < NUM_CLASSES; i++) {
        predictions.push_back({h_output[i], static_cast<int>(i)});
    }
    std::partial_sort(predictions.begin(), predictions.begin() + 5, predictions.end(), 
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    
    fmt::print("Top-5 predictions for batch 0:\n");
    for (int i = 0; i < 5; i++) {
        fmt::print("  Class {}: {:.4f}\n", predictions[i].second, predictions[i].first);
    }
    
    // Cleanup
    fmt::print("Cleaning up resources...\n");
    
    checkCudaErrors(cudaGraphDestroy(graph));
    checkCudaErrors(cudaStreamDestroy(stream));
    checkCudaErrors(cudnnDestroy(cudnnHandle));
    checkCudaErrors(cublasDestroy(cublasHandle));
    
    // Free all managed memory
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
    memManager.freeManagedMemory(d_fc_weights);
    memManager.freeManagedMemory(d_fc_bias);
    memManager.freeManagedMemory(d_fc_output);
    memManager.freeManagedMemory(pool_output);
    memManager.freeManagedMemory(gap_output);
    
    checkCudaErrors(cudaFreeHost(h_input));
    checkCudaErrors(cudaFreeHost(h_output));
    
    fmt::print("✅ ResNet inference demo completed successfully!\n");
}

int main(int argc, char *argv[]) {
    // Parse command line arguments
    BATCH_SIZE = (argc > 1) ? std::atoi(argv[1]) : 32;
    INPUT_HEIGHT = INPUT_WIDTH = (argc > 2) ? std::atoi(argv[2]) : 224;
    
    // Validation
    if (BATCH_SIZE < 1 || BATCH_SIZE > 256) {
        fmt::print("ERROR: Batch size must be between 1 and 256\n");
        return -1;
    }
    
    if (INPUT_HEIGHT < 32 || INPUT_HEIGHT > 512) {
        fmt::print("ERROR: Input size must be between 32 and 512\n");
        return -1;
    }
    
    ConfigurationManager::exportDefaultConfiguration();
    ConfigurationManager::loadConfiguration("config.json");
    
    fmt::print("Configuration: Batch={}, Input={}x{}x{}, Classes={}\n", 
               BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH, NUM_CLASSES);
    
    resnetInference();
    
    return 0;
}