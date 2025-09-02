add_executable(tiledCholesky tiledCholesky.cu)
target_link_libraries(
  tiledCholesky
  memopt
  CUDA::cublas
  CUDA::cusolver
  CUDA::curand
  fmt::fmt
)

add_executable(tiledCholesky_with_pointer_deps tiledCholesky_with_pointer_deps.cu)
target_link_libraries(
  tiledCholesky_with_pointer_deps
  memopt
  CUDA::cublas
  CUDA::cusolver
  CUDA::curand
  fmt::fmt
)

add_executable(tiledCholeskyEnlarge tiledCholeskyEnlarge.cu)
target_link_libraries(
  tiledCholeskyEnlarge
  memopt
  CUDA::cublas
  CUDA::cusolver
  CUDA::curand
  fmt::fmt
)

add_executable(tiledCholeskyColdStart tiledCholeskyColdStart.cu)
target_link_libraries(
  tiledCholeskyColdStart
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

#add_subdirectory(lulesh)
