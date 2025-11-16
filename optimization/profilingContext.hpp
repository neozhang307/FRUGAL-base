#pragma once

namespace memopt {

/**
 * @brief RAII wrapper for dummy kernel handle lifecycle management
 *
 * This class manages the registration and cleanup of dummy kernel handles
 * used during profiling. It ensures proper resource management through RAII.
 *
 * The context must be created before calling profileGraph() as that function
 * expects the dummy kernels to already be registered.
 *
 * Usage:
 * ```cpp
 * {
 *     ProfilingContext ctx;  // Registers handles
 *     auto input = optimizer->profileGraph(graph);
 *     auto output = optimizer->optimizeGraph(input);
 * }  // Automatically cleans up handles
 * ```
 */
class ProfilingContext {
public:
    /**
     * @brief Constructor - registers dummy kernel handles
     *
     * Calls registerDummyKernelHandles() to create:
     * - dummyKernelForAnnotationHandle
     * - dummyKernelForStageSeparatorHandle
     *
     * These handles are used to identify special marker nodes in CUDA graphs
     * during the profiling phase.
     */
    ProfilingContext();

    /**
     * @brief Destructor - cleans up dummy kernel handles
     *
     * Calls cleanUpDummyKernelFuncHandleRegistrations() to destroy
     * the CUDA graph resources created during registration.
     */
    ~ProfilingContext();

    // Prevent copying (resource ownership)
    ProfilingContext(const ProfilingContext&) = delete;
    ProfilingContext& operator=(const ProfilingContext&) = delete;

    // Allow moving if needed in the future
    ProfilingContext(ProfilingContext&&) = default;
    ProfilingContext& operator=(ProfilingContext&&) = default;
};

}  // namespace memopt