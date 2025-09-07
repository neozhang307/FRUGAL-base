cmake_minimum_required(VERSION 3.21)

project(OptimizeCudaMemoryUsage LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_STANDARD 17)
SET(CMAKE_CUDA_ARCHITECTURES 90)
SET(CUDA_SEPARABLE_COMPILATION ON)

# Add OpenMP support
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

enable_language(C CXX CUDA)

find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(fmt REQUIRED)
find_package(ortools REQUIRED)

# Find cuDNN
find_library(CUDNN_LIBRARY cudnn
    HINTS ${CUDA_TOOLKIT_ROOT_DIR} $ENV{CONDA_PREFIX}
    PATH_SUFFIXES lib64 lib)
    
find_path(CUDNN_INCLUDE_DIR cudnn.h
    HINTS ${CUDA_TOOLKIT_ROOT_DIR} $ENV{CONDA_PREFIX}
    PATH_SUFFIXES include)
    
if(CUDNN_LIBRARY AND CUDNN_INCLUDE_DIR)
    message(STATUS "Found cuDNN: ${CUDNN_LIBRARY}")
    message(STATUS "Found cuDNN headers: ${CUDNN_INCLUDE_DIR}")
    add_library(CUDA::cudnn UNKNOWN IMPORTED)
    set_target_properties(CUDA::cudnn PROPERTIES
        IMPORTED_LOCATION ${CUDNN_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES ${CUDNN_INCLUDE_DIR})
else()
    message(WARNING "cuDNN not found - ResNet application may not build properly")
endif()

add_subdirectory(utilities)
add_subdirectory(profiling)
add_subdirectory(optimization)
add_subdirectory(public)
add_subdirectory(userApplications)
add_subdirectory(playground)
add_subdirectory(experiments)
