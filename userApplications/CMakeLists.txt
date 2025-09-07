add_executable(tiledCholesky tiledCholesky.cu)
target_link_libraries(
  tiledCholesky
  memopt
  CUDA::cublas
  CUDA::cusolver
  CUDA::curand
  fmt::fmt
)


add_executable(tiledCholeskyDomainEnlarge tiledCholeskyDomainEnlarge.cu)
target_link_libraries(
  tiledCholeskyDomainEnlarge
  memopt
  CUDA::cublas
  CUDA::cusolver
  CUDA::curand
  fmt::fmt
)

add_executable(tiledCholeskyMemoryOptimized tiledCholeskyMemoryOptimized.cu)
target_link_libraries(
  tiledCholeskyMemoryOptimized
  memopt
  CUDA::cublas
  CUDA::cusolver
  CUDA::curand
  fmt::fmt
)

add_executable(tiledCholeskyNaiveGraph tiledCholeskyNaiveGraph.cu)
target_link_libraries(
  tiledCholeskyNaiveGraph
  memopt
  CUDA::cublas
  CUDA::cusolver
  CUDA::curand
  fmt::fmt
)

add_executable(lu_def lu_def.cu)
target_link_libraries(
  lu_def
  memopt
  CUDA::cublas
  CUDA::cusolver
  CUDA::curand
  fmt::fmt
  OpenMP::OpenMP_CXX
)

add_executable(resnet resnet.cu)
target_link_libraries(
  resnet
  memopt
  CUDA::cublas
  CUDA::cusolver
  CUDA::curand
  fmt::fmt
)

add_executable(resnet_simple resnet_simple.cu)
target_link_libraries(
  resnet_simple
  memopt
  CUDA::cublas
  CUDA::cusolver
  CUDA::curand
  fmt::fmt
)

# Link cuDNN if available
if(TARGET CUDA::cudnn)
  target_link_libraries(resnet CUDA::cudnn)
  target_link_libraries(resnet_simple CUDA::cudnn)
else()
  message(WARNING "cuDNN not found - ResNet may not function correctly without cuDNN library")
endif()

#add_subdirectory(lulesh)
