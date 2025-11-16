#include "profilingContext.hpp"

// Forward declarations for the dummy kernel registration functions
// These are defined in optimizer.cu
namespace memopt {
    extern void registerDummyKernelHandles();
    extern void cleanUpDummyKernelFuncHandleRegistrations();
}

namespace memopt {

ProfilingContext::ProfilingContext() {
    // Register the dummy kernel handles when context is created
    registerDummyKernelHandles();
}

ProfilingContext::~ProfilingContext() {
    // Clean up the dummy kernel handles when context is destroyed
    cleanUpDummyKernelFuncHandleRegistrations();
}

}  // namespace memopt