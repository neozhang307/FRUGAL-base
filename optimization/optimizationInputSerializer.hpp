#pragma once

#include <string>
#include "optimizationInput.hpp"

namespace memopt {

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

}  // namespace memopt
