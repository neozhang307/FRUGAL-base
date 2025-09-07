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
  CUDA::cudnn
  fmt::fmt
)

#add_subdirectory(lulesh)
