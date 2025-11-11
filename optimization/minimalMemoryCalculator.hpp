#pragma once

#include <set>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>

#include "../utilities/types.hpp"
#include "optimizationInput.hpp"
#include "strategies/secondStepSolver.hpp"

namespace memopt {

/**
 * @class MinimalMemoryCalculator
 * @brief Calculates the theoretical minimal memory required for task execution
 *
 * This calculator determines the theoretical lower bound on memory usage by finding
 * the maximum memory required by any single task or task group. It can work with
 * various input formats and supports both single-stage and multi-stage applications.
 *
 * The minimal memory represents the best possible memory usage that any optimization
 * strategy could achieve, serving as a benchmark for evaluating optimization quality.
 */
class MinimalMemoryCalculator {
public:
    /**
     * @struct TaskMemoryInfo
     * @brief Memory access pattern for a single task or task group
     */
    struct TaskMemoryInfo {
        std::set<ArrayId> inputArrays;   ///< Arrays read by this task
        std::set<ArrayId> outputArrays;  ///< Arrays written by this task

        /**
         * @brief Get all arrays accessed (union of inputs and outputs)
         */
        std::set<ArrayId> getAllArrays() const {
            std::set<ArrayId> allArrays;
            std::set_union(inputArrays.begin(), inputArrays.end(),
                          outputArrays.begin(), outputArrays.end(),
                          std::inserter(allArrays, allArrays.begin()));
            return allArrays;
        }
    };

    /**
     * @struct StageInfo
     * @brief Information about a single execution stage
     */
    struct StageInfo {
        std::vector<TaskMemoryInfo> tasks;  ///< Task memory access patterns
        std::vector<size_t> arraySizes;     ///< Size in bytes for each array
        std::string stageName;              ///< Optional stage identifier
    };

    /**
     * @struct Result
     * @brief Results of minimal memory calculation
     */
    struct Result {
        double minimalMemoryMiB;              ///< Theoretical minimum memory in MiB
        double originalMemoryMiB;             ///< Memory if all arrays loaded at once
        double reductionPotential;            ///< originalMemory / minimalMemory ratio
        std::vector<double> perTaskMemoryMiB; ///< Memory needed for each task
        int criticalTaskIndex;                ///< Index of task requiring most memory
        std::set<ArrayId> allArraysUsed;      ///< All arrays accessed across all tasks

        /**
         * @brief Get memory savings percentage
         */
        double getMemorySavingsPercent() const {
            return 100.0 * (1.0 - minimalMemoryMiB / originalMemoryMiB);
        }

        /**
         * @brief Print result summary to stdout
         */
        void printSummary() const;
    };

    /**
     * @struct MultiStageResult
     * @brief Results for multi-stage applications
     */
    struct MultiStageResult {
        std::vector<Result> stageResults;     ///< Results for each stage
        double overallMinimalMemoryMiB;       ///< Maximum minimal memory across stages
        double overallOriginalMemoryMiB;      ///< Maximum original memory across stages
        int criticalStageIndex;               ///< Stage requiring most memory

        /**
         * @brief Print multi-stage result summary
         */
        void printSummary() const;
    };

    // ============= Main Calculation Methods =============

    /**
     * @brief Calculate minimal memory for a single stage
     * @param stage Stage information containing tasks and array sizes
     * @return Result containing minimal memory analysis
     */
    Result calculateForSingleStage(const StageInfo& stage) const;

    /**
     * @brief Calculate minimal memory for multiple stages
     * @param stages Vector of stage information
     * @return MultiStageResult containing analysis for all stages
     */
    MultiStageResult calculateForMultiStage(const std::vector<StageInfo>& stages) const;

    // ============= Convenience Methods for Different Input Types =============

    /**
     * @brief Calculate from OptimizationInput structure
     * @param input Optimization input from profiling phase
     * @return Result containing minimal memory analysis
     */
    Result calculateFromOptimizationInput(const OptimizationInput& input) const;

    /**
     * @brief Calculate from SecondStepSolver::Input structure
     * @param input Second step solver input
     * @return Result containing minimal memory analysis
     */
    Result calculateFromSecondStepInput(const SecondStepSolver::Input& input) const;

    /**
     * @brief Convert OptimizationInput to StageInfo format
     * @param input Optimization input to convert
     * @return StageInfo representation
     */
    StageInfo convertOptimizationInputToStageInfo(const OptimizationInput& input) const;

    /**
     * @brief Convert SecondStepSolver::Input to StageInfo format
     * @param input Second step input to convert
     * @return StageInfo representation
     */
    StageInfo convertSecondStepInputToStageInfo(const SecondStepSolver::Input& input) const;

private:
    /**
     * @brief Convert bytes to MiB
     * @param bytes Size in bytes
     * @return Size in MiB
     */
    static constexpr double bytesToMiB(size_t bytes) {
        return static_cast<double>(bytes) / (1024.0 * 1024.0);
    }

    /**
     * @brief Calculate memory for a specific set of arrays
     * @param arrayIds Set of array IDs
     * @param arraySizes Vector of array sizes indexed by array ID
     * @return Total memory in bytes
     */
    size_t calculateMemoryForArrays(const std::set<ArrayId>& arrayIds,
                                    const std::vector<size_t>& arraySizes) const;
};

}  // namespace memopt