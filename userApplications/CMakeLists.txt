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

add_executable(LUDecomposition lu_def.cu)
target_link_libraries(
  LUDecomposition
  memopt
  CUDA::cublas
  CUDA::cusolver
  CUDA::curand
  fmt::fmt
)

#add_subdirectory(lulesh)
