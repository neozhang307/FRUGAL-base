#pragma once

#include <string>
#include "optimizationInput.hpp"
#include "optimizationOutput.hpp"
#include "strategies/firstStepSolver.hpp"

namespace memopt {

// ========== OptimizationInput Serialization ==========

/**
 * @brief Save OptimizationInput to JSON file for offline optimization
 *
 * This function serializes the profiling data (task groups, dependencies, arrays)
 * to a JSON file that can be loaded later for offline optimization.
 *
 * @param input The optimization input containing profiling data
 * @param path Path to the output JSON file
 */
void saveOptimizationInput(const OptimizationInput& input, const std::string& path);

/**
 * @brief Load OptimizationInput from JSON file
 *
 * This function deserializes profiling data from a JSON file created by
 * saveOptimizationInput, allowing offline optimization without re-profiling.
 *
 * @param path Path to the input JSON file
 * @return OptimizationInput The loaded profiling data
 */
OptimizationInput loadOptimizationInput(const std::string& path);

// ========== OptimizationOutput Serialization ==========

/**
 * @brief Save OptimizationOutput to JSON file
 *
 * This function serializes an optimized execution plan to a JSON file
 * that can be loaded and executed later without re-optimization.
 *
 * @param output The optimized execution plan
 * @param path Path to the output JSON file
 */
void saveOptimizationOutput(const OptimizationOutput& output, const std::string& path);

/**
 * @brief Load OptimizationOutput from JSON file
 *
 * This function deserializes an optimized execution plan from a JSON file
 * created by saveOptimizationOutput, allowing direct execution of a
 * pre-optimized plan.
 *
 * @param path Path to the input JSON file
 * @return OptimizationOutput The loaded optimized execution plan
 */
OptimizationOutput loadOptimizationOutput(const std::string& path);

// ========== FirstStepSolver::Output Serialization ==========

/**
 * @brief Save FirstStepSolver::Output to JSON file
 *
 * This function serializes the task scheduling results from the first
 * optimization step, allowing it to be reused in subsequent runs.
 *
 * @param output The first step output containing task execution order
 * @param path Path to the output JSON file
 */
void saveFirstStepOutput(const FirstStepSolver::Output& output, const std::string& path);

/**
 * @brief Load FirstStepSolver::Output from JSON file
 *
 * This function deserializes task scheduling results from a JSON file
 * created by saveFirstStepOutput, allowing the second optimization step
 * to run with a pre-computed task order.
 *
 * @param path Path to the input JSON file
 * @return FirstStepSolver::Output The loaded task scheduling results
 */
FirstStepSolver::Output loadFirstStepOutput(const std::string& path);

}  // namespace memopt
