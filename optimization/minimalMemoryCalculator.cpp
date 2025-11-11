#include "minimalMemoryCalculator.hpp"

#include <fmt/core.h>
#include <algorithm>
#include <numeric>
#include <iostream>

#include "../utilities/logger.hpp"

namespace memopt {

// ============= Result Methods =============

void MinimalMemoryCalculator::Result::printSummary() const {
    fmt::print("===== Minimal Memory Calculation Result =====\n");
    fmt::print("Minimal memory usage (theoretical lower bound): {:.2f} MiB\n", minimalMemoryMiB);
    fmt::print("Original peak memory usage (without optimization): {:.2f} MiB\n", originalMemoryMiB);
    fmt::print("Memory reduction potential: {:.2f}x\n", reductionPotential);
    fmt::print("Memory savings potential: {:.1f}%\n", getMemorySavingsPercent());
    fmt::print("Critical task index: {} (requires {:.2f} MiB)\n",
               criticalTaskIndex, perTaskMemoryMiB[criticalTaskIndex]);
    fmt::print("Total arrays accessed: {}\n", allArraysUsed.size());
    fmt::print("===========================================\n");
}

void MinimalMemoryCalculator::MultiStageResult::printSummary() const {
    fmt::print("===== Multi-Stage Minimal Memory Calculation =====\n");
    fmt::print("Number of stages: {}\n", stageResults.size());
    fmt::print("Overall minimal memory: {:.2f} MiB\n", overallMinimalMemoryMiB);
    fmt::print("Overall original memory: {:.2f} MiB\n", overallOriginalMemoryMiB);
    fmt::print("Critical stage: {} (requires {:.2f} MiB)\n",
               criticalStageIndex, stageResults[criticalStageIndex].minimalMemoryMiB);

    for (size_t i = 0; i < stageResults.size(); ++i) {
        fmt::print("\n  Stage {}: minimal={:.2f} MiB, original={:.2f} MiB, reduction={:.2f}x\n",
                   i, stageResults[i].minimalMemoryMiB,
                   stageResults[i].originalMemoryMiB,
                   stageResults[i].reductionPotential);
    }
    fmt::print("==================================================\n");
}

// ============= Main Calculation Methods =============

MinimalMemoryCalculator::Result
MinimalMemoryCalculator::calculateForSingleStage(const StageInfo& stage) const {
    LOG_TRACE_WITH_INFO("MinimalMemoryCalculator::calculateForSingleStage");

    Result result;
    result.perTaskMemoryMiB.reserve(stage.tasks.size());
    result.minimalMemoryMiB = 0.0;
    result.criticalTaskIndex = -1;

    // Track all arrays used across all tasks
    std::set<ArrayId> allArraysUsed;

    // Calculate memory requirement for each task
    for (size_t taskIdx = 0; taskIdx < stage.tasks.size(); ++taskIdx) {
        const auto& task = stage.tasks[taskIdx];

        // Get all arrays accessed by this task (union of inputs and outputs)
        std::set<ArrayId> taskArrays = task.getAllArrays();

        // Add to overall set of arrays
        allArraysUsed.insert(taskArrays.begin(), taskArrays.end());

        // Calculate memory needed for this task
        size_t taskMemoryBytes = calculateMemoryForArrays(taskArrays, stage.arraySizes);
        double taskMemoryMiB = bytesToMiB(taskMemoryBytes);

        result.perTaskMemoryMiB.push_back(taskMemoryMiB);

        // Update minimal memory if this task needs more
        if (taskMemoryMiB > result.minimalMemoryMiB) {
            result.minimalMemoryMiB = taskMemoryMiB;
            result.criticalTaskIndex = static_cast<int>(taskIdx);
        }

        LOG_TRACE_WITH_INFO(fmt::format("Task {}: {} arrays, {:.2f} MiB",
                                        taskIdx, taskArrays.size(), taskMemoryMiB).c_str());
    }

    // Calculate original memory (all arrays at once)
    size_t originalMemoryBytes = calculateMemoryForArrays(allArraysUsed, stage.arraySizes);
    result.originalMemoryMiB = bytesToMiB(originalMemoryBytes);
    result.allArraysUsed = allArraysUsed;

    // Calculate reduction potential
    if (result.minimalMemoryMiB > 0) {
        result.reductionPotential = result.originalMemoryMiB / result.minimalMemoryMiB;
    } else {
        result.reductionPotential = 1.0;
    }

    LOG_TRACE_WITH_INFO(fmt::format("Minimal memory: {:.2f} MiB, Original: {:.2f} MiB, Reduction: {:.2f}x",
                                    result.minimalMemoryMiB, result.originalMemoryMiB,
                                    result.reductionPotential).c_str());

    return result;
}

MinimalMemoryCalculator::MultiStageResult
MinimalMemoryCalculator::calculateForMultiStage(const std::vector<StageInfo>& stages) const {
    LOG_TRACE_WITH_INFO("MinimalMemoryCalculator::calculateForMultiStage");

    MultiStageResult multiResult;
    multiResult.stageResults.reserve(stages.size());
    multiResult.overallMinimalMemoryMiB = 0.0;
    multiResult.overallOriginalMemoryMiB = 0.0;
    multiResult.criticalStageIndex = -1;

    // Calculate for each stage
    for (size_t stageIdx = 0; stageIdx < stages.size(); ++stageIdx) {
        Result stageResult = calculateForSingleStage(stages[stageIdx]);
        multiResult.stageResults.push_back(stageResult);

        // Update overall minimal memory
        if (stageResult.minimalMemoryMiB > multiResult.overallMinimalMemoryMiB) {
            multiResult.overallMinimalMemoryMiB = stageResult.minimalMemoryMiB;
            multiResult.criticalStageIndex = static_cast<int>(stageIdx);
        }

        // Update overall original memory
        multiResult.overallOriginalMemoryMiB = std::max(multiResult.overallOriginalMemoryMiB,
                                                        stageResult.originalMemoryMiB);
    }

    return multiResult;
}

// ============= Convenience Methods for Different Input Types =============

MinimalMemoryCalculator::Result
MinimalMemoryCalculator::calculateFromOptimizationInput(const OptimizationInput& input) const {
    LOG_TRACE_WITH_INFO("MinimalMemoryCalculator::calculateFromOptimizationInput");

    StageInfo stage = convertOptimizationInputToStageInfo(input);
    stage.stageName = fmt::format("Stage_{}", input.stageIndex);
    return calculateForSingleStage(stage);
}

MinimalMemoryCalculator::Result
MinimalMemoryCalculator::calculateFromSecondStepInput(const SecondStepSolver::Input& input) const {
    LOG_TRACE_WITH_INFO("MinimalMemoryCalculator::calculateFromSecondStepInput");

    StageInfo stage = convertSecondStepInputToStageInfo(input);
    stage.stageName = fmt::format("Stage_{}", input.stageIndex);
    return calculateForSingleStage(stage);
}

MinimalMemoryCalculator::StageInfo
MinimalMemoryCalculator::convertOptimizationInputToStageInfo(const OptimizationInput& input) const {
    StageInfo stage;

    // First, we need to build a mapping from void* to array indices
    std::map<void*, ArrayId> pointerToArrayId;
    std::vector<size_t> arraySizes;
    ArrayId nextArrayId = 0;

    // Collect all unique pointers from all task groups
    for (const auto& taskGroup : input.nodes) {
        for (void* ptr : taskGroup.dataDependency.inputs) {
            if (pointerToArrayId.find(ptr) == pointerToArrayId.end()) {
                pointerToArrayId[ptr] = nextArrayId++;
                // We don't have size information in OptimizationInput
                // This would need to be obtained from memory manager or profiling data
                // For now, use a placeholder size
                arraySizes.push_back(1024 * 1024);  // 1MB placeholder
            }
        }
        for (void* ptr : taskGroup.dataDependency.outputs) {
            if (pointerToArrayId.find(ptr) == pointerToArrayId.end()) {
                pointerToArrayId[ptr] = nextArrayId++;
                arraySizes.push_back(1024 * 1024);  // 1MB placeholder
            }
        }
    }

    stage.arraySizes = arraySizes;

    // Convert each task group to TaskMemoryInfo
    for (const auto& taskGroup : input.nodes) {
        TaskMemoryInfo taskInfo;

        // Convert input pointers to array IDs
        for (void* ptr : taskGroup.dataDependency.inputs) {
            taskInfo.inputArrays.insert(pointerToArrayId[ptr]);
        }

        // Convert output pointers to array IDs
        for (void* ptr : taskGroup.dataDependency.outputs) {
            taskInfo.outputArrays.insert(pointerToArrayId[ptr]);
        }

        stage.tasks.push_back(taskInfo);
    }

    return stage;
}

MinimalMemoryCalculator::StageInfo
MinimalMemoryCalculator::convertSecondStepInputToStageInfo(const SecondStepSolver::Input& input) const {
    StageInfo stage;
    stage.arraySizes = input.arraySizes;

    // Convert each task group to TaskMemoryInfo
    size_t numTasks = input.taskGroupInputArrays.size();
    for (size_t i = 0; i < numTasks; ++i) {
        TaskMemoryInfo taskInfo;
        taskInfo.inputArrays = input.taskGroupInputArrays[i];
        taskInfo.outputArrays = input.taskGroupOutputArrays[i];
        stage.tasks.push_back(taskInfo);
    }

    return stage;
}

// ============= Private Helper Methods =============

size_t MinimalMemoryCalculator::calculateMemoryForArrays(const std::set<ArrayId>& arrayIds,
                                                         const std::vector<size_t>& arraySizes) const {
    size_t totalBytes = 0;
    for (ArrayId id : arrayIds) {
        if (id < arraySizes.size()) {
            totalBytes += arraySizes[id];
        } else {
            LOG_TRACE_WITH_INFO(fmt::format("Warning: Array ID {} exceeds array sizes vector size {}",
                                           id, arraySizes.size()).c_str());
        }
    }
    return totalBytes;
}

}  // namespace memopt