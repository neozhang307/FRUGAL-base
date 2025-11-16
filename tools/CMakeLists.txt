# Standalone optimizer executable
add_executable(
  standaloneOptimizer
  standaloneOptimizer.cpp
)

target_link_libraries(
  standaloneOptimizer
  PRIVATE
  optimization
  utilities
  fmt::fmt
)

# Set include directories
target_include_directories(
  standaloneOptimizer
  PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/include
)