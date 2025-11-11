add_library(minimalMemoryCalculator STATIC ../minimalMemoryCalculator.cpp)
target_link_libraries(
  minimalMemoryCalculator
  PUBLIC
  fmt::fmt
  utilities
)

add_library(strategyUtilities STATIC strategyUtilities.cu)
target_link_libraries(
  strategyUtilities
  PRIVATE
  fmt::fmt
)

add_library(strategies STATIC noOptimizationStrategy.cu twoStepOptimizationStrategy.cu)
target_link_libraries(
  strategies
  PRIVATE
  strategyUtilities
  firstStepSolver
  secondStepSolver
)

add_library(firstStepSolver STATIC firstStepSolver.cpp)
target_link_libraries(
  firstStepSolver
  PRIVATE
  fmt::fmt
)

add_library(secondStepSolver STATIC secondStepSolver.cpp)
target_link_libraries(
  secondStepSolver
  PUBLIC
  utilities
  fmt::fmt
  ortools::ortools
  minimalMemoryCalculator
)
