add_executable(helloWorld helloWorld.cu)

add_executable(captureGraph captureGraph.cu)
target_link_libraries(captureGraph memopt)

add_executable(modifyKernelNodeParams modifyKernelNodeParams.cu)
target_link_libraries(modifyKernelNodeParams memopt CUDA::cuda_driver)

add_executable(monitorMemoryUsage monitorMemoryUsage.cu)
target_link_libraries(monitorMemoryUsage memopt fmt::fmt)

add_executable(assertion assertion.cu)

add_executable(orToolsPlayground orToolsPlayground.cpp)
target_link_libraries(orToolsPlayground ortools::ortools fmt::fmt)

add_executable(orToolsRoutingPlayground orToolsRoutingPlayground.cpp)
target_link_libraries(orToolsRoutingPlayground ortools::ortools fmt::fmt)

add_executable(tiledCholeskyPlayground tiledCholeskyPlayground.cu)
target_link_libraries(
  tiledCholeskyPlayground
  memopt
  CUDA::cublas
  CUDA::cusolver
  CUDA::curand
  fmt::fmt
)

add_executable(taskManagerExample taskManagerExample.cu)
target_link_libraries(
  taskManagerExample
  memopt
  fmt::fmt
)

add_executable(taskManager_v2_example taskManager_v2_example.cu)
target_link_libraries(
  taskManager_v2_example
  memopt
  fmt::fmt
)

add_executable(consecutiveGraphMemoryTest consecutiveGraphMemoryTest.cu)
target_link_libraries(
  consecutiveGraphMemoryTest
  memopt
  CUDA::cublas
)

add_executable(domainEnlargementTest domainEnlargementTest.cu)
target_link_libraries(
  domainEnlargementTest
  memopt
)

add_executable(coldStartTest coldStartTest.cu)
target_link_libraries(
  coldStartTest
  memopt
)
