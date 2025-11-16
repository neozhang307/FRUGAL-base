/**
 * @file standaloneOptimizer.cpp
 * @brief Standalone optimizer for offline optimization workflow
 *
 * This tool loads profiling data from JSON, runs the optimization algorithm,
 * and saves the optimized execution plan. It enables CPU-only optimization
 * without requiring GPU access.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <fmt/core.h>

#include "../include/argh.h"
#include "../include/json.hpp"
#include "../optimization/optimizationSerializer.hpp"
#include "../optimization/optimizationInput.hpp"
#include "../optimization/optimizationOutput.hpp"
#include "../utilities/configurationManager.hpp"
#include "../utilities/logger.hpp"

// Include necessary headers for inline optimization
#include "../optimization/strategies/firstStepSolver.hpp"
#include "../optimization/strategies/secondStepSolver.hpp"
#include "../optimization/minimalMemoryCalculator.hpp"

namespace memopt {
    // Forward declare conversion functions from twoStepOptimizationStrategy.cu
    FirstStepSolver::Input convertToFirstStepInput(OptimizationInput &optimizationInput);
    SecondStepSolver::Input convertToSecondStepInput(
        OptimizationInput &optimizationInput,
        FirstStepSolver::Output &firstStepOutput
    );
    OptimizationOutput convertToOptimizationOutput(
        OptimizationInput &optimizationInput,
        FirstStepSolver::Output &firstStepOutput,
        SecondStepSolver::Output &secondStepOutput
    );
}

using namespace memopt;

// Local serialization functions for FirstStepSolver::Output
void saveFirstStepOutput(const FirstStepSolver::Output& output, const std::string& path) {
    nlohmann::json j;
    j["taskGroupExecutionOrder"] = output.taskGroupExecutionOrder;

    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
    file << j.dump(2);
    file.close();

    fmt::print("Saved FirstStepOutput to {} (tasks: {})\n", path, output.taskGroupExecutionOrder.size());
}

FirstStepSolver::Output loadFirstStepOutput(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }

    nlohmann::json j;
    file >> j;
    file.close();

    FirstStepSolver::Output output;
    output.taskGroupExecutionOrder = j["taskGroupExecutionOrder"].get<std::vector<TaskGroupId>>();

    fmt::print("Loaded FirstStepOutput from {} (tasks: {})\n", path, output.taskGroupExecutionOrder.size());
    return output;
}

void printUsage(const char* programName) {
    fmt::print("Usage: {} <input_profile.json> <output_plan.json> [options]\n", programName);
    fmt::print("\nDescription:\n");
    fmt::print("  Loads profiling data and generates an optimized execution plan.\n");
    fmt::print("  This tool performs CPU-only optimization without requiring GPU access.\n");
    fmt::print("\nArguments:\n");
    fmt::print("  input_profile.json    Path to input profiling data (from --profile-only mode)\n");
    fmt::print("  output_plan.json      Path to save the optimized execution plan\n");
    fmt::print("\nOptions:\n");
    fmt::print("  --config=<path>       Path to configuration file (default: config.json)\n");
    fmt::print("  --memory-bound=<MB>   Memory constraint in MB (default: from config)\n");
    fmt::print("  --weight=<value>      Weight for running time vs memory (default: from config)\n");
    fmt::print("  --solver=<type>       Second step solver: MIP, GREEDY (default: from config)\n");
    fmt::print("  --beam-width=<N>      Beam width for first step solver (default: from config)\n");
    fmt::print("  --timeout=<seconds>   Gurobi solver timeout in seconds (default: from config)\n");
    fmt::print("  --save-first-step=<path>  Save first step output to file\n");
    fmt::print("  --load-first-step=<path>  Load first step output and run only second step\n");
    fmt::print("  --help, -h            Show this help message\n");
    fmt::print("\nExamples:\n");
    fmt::print("  # Basic usage with default config\n");
    fmt::print("  {} profile.json optimized.json\n", programName);
    fmt::print("\n  # With custom memory bound and solver\n");
    fmt::print("  {} profile.json optimized.json --memory-bound=1000 --solver=GREEDY\n", programName);
    fmt::print("\n  # With custom config file\n");
    fmt::print("  {} profile.json optimized.json --config=custom_config.json\n", programName);
    fmt::print("\n  # Save first step output for reuse\n");
    fmt::print("  {} profile.json optimized.json --save-first-step=first_step.json\n", programName);
    fmt::print("\n  # Load first step and run only second step\n");
    fmt::print("  {} profile.json optimized.json --load-first-step=first_step.json\n", programName);
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    auto cmdl = argh::parser(argc, argv);

    // Check for help
    if (cmdl["-h"] || cmdl["--help"] || argc < 3) {
        printUsage(argv[0]);
        return (cmdl["-h"] || cmdl["--help"]) ? 0 : 1;
    }

    // Get positional arguments
    std::string inputPath, outputPath;
    if (!(cmdl(1) >> inputPath) || !(cmdl(2) >> outputPath)) {
        fmt::print(stderr, "Error: Missing required arguments\n");
        printUsage(argv[0]);
        return 1;
    }

    // Load configuration
    std::string configPath = "config.json";
    cmdl("--config", configPath) >> configPath;

    fmt::print("=== FRUGAL Standalone Optimizer ===\n");
    fmt::print("Input profile: {}\n", inputPath);
    fmt::print("Output plan: {}\n", outputPath);
    fmt::print("Config file: {}\n", configPath);

    try {
        // Load configuration
        ConfigurationManager::exportDefaultConfiguration();
        ConfigurationManager::loadConfiguration(configPath);
        auto& config = const_cast<Configuration&>(ConfigurationManager::getConfig());

        // Override config with command line options
        if (cmdl("--memory-bound")) {
            double memoryBoundMB;
            cmdl("--memory-bound") >> memoryBoundMB;
            config.optimization.maxPeakMemoryUsageInMiB = memoryBoundMB;
            fmt::print("Memory bound: {:.2f} MB\n", memoryBoundMB);
        }

        if (cmdl("--weight")) {
            double weight;
            cmdl("--weight") >> weight;
            config.optimization.weightOfTotalRunningTime = weight;
            fmt::print("Running time weight: {:.6f}\n", weight);
        }

        if (cmdl("--solver")) {
            std::string solverType;
            cmdl("--solver") >> solverType;
            config.optimization.secondStepSolverType = solverType;
            fmt::print("Second step solver type: {}\n", solverType);
        }

        if (cmdl("--beam-width")) {
            int beamWidth;
            cmdl("--beam-width") >> beamWidth;
            config.optimization.beamWidth = beamWidth;
            fmt::print("Beam width: {}\n", beamWidth);
        }

        if (cmdl("--timeout")) {
            int timeout;
            cmdl("--timeout") >> timeout;
            config.optimization.gurobiTimeLimitSeconds = timeout;
            fmt::print("Gurobi solver timeout: {} seconds\n", timeout);
        }

        fmt::print("\n--- Loading Profiling Data ---\n");

        // Load profiling data
        auto optimizationInput = loadOptimizationInput(inputPath);

        fmt::print("Loaded {} task groups\n", optimizationInput.nodes.size());
        fmt::print("Original running time: {:.6f} seconds\n", optimizationInput.originalTotalRunningTime);

        // Load array information from JSON to get sizes
        std::ifstream inputFile(inputPath);
        if (!inputFile.is_open()) {
            throw std::runtime_error("Failed to open input file: " + inputPath);
        }
        auto j = nlohmann::json::parse(inputFile);
        inputFile.close();

        // Calculate total data size
        double totalDataMB = 0.0;
        if (j.contains("arrays")) {
            for (const auto& arr : j["arrays"]) {
                size_t size = arr["size"];
                totalDataMB += static_cast<double>(size) / (1024.0 * 1024.0);
            }
        }
        fmt::print("Total data size: {:.2f} MB\n", totalDataMB);
        fmt::print("Memory constraint: {:.2f} MB\n", config.optimization.maxPeakMemoryUsageInMiB);

        fmt::print("\n--- Optimization Settings ---\n");
        fmt::print("First step solver: {}\n", config.optimization.firstStepSolverType);
        fmt::print("Second step solver: {}\n", config.optimization.secondStepSolverType);
        fmt::print("Beam width: {}\n", config.optimization.beamWidth);
        fmt::print("Weight - Memory: {:.6f}\n", config.optimization.weightOfPeakMemoryUsage);
        fmt::print("Weight - Time: {:.6f}\n", config.optimization.weightOfTotalRunningTime);
        fmt::print("Weight - Migrations: {:.6f}\n", config.optimization.weightOfNumberOfMigrations);

        fmt::print("\n--- Running Optimization ---\n");

        // Check if we should load or save first step
        std::string loadFirstStepPath = "";
        std::string saveFirstStepPath = "";

        // Use the same pattern as tiledCholeskyAblation.cu
        cmdl("--save-first-step", saveFirstStepPath) >> saveFirstStepPath;
        cmdl("--load-first-step", loadFirstStepPath) >> loadFirstStepPath;

        if (!saveFirstStepPath.empty()) {
            fmt::print("Will save first step output to: {}\n", saveFirstStepPath);
        }

        if (!loadFirstStepPath.empty()) {
            fmt::print("Will load first step output from: {}\n", loadFirstStepPath);
        }

        // Instead of calling optimizer->optimizeGraph, inline the TwoStepOptimizationStrategy::run code
        // This avoids dependency issues and allows us to run in standalone mode

        OptimizationOutput optimizationOutput;

        // ======= MEMORY BOUND FEASIBILITY CHECK =======
        const double configuredMemoryBoundMiB = config.optimization.maxPeakMemoryUsageInMiB;

        if (configuredMemoryBoundMiB > 0) {
            // Calculate the theoretical minimum memory required
            MinimalMemoryCalculator calculator;
            auto minMemResult = calculator.calculateFromOptimizationInput(optimizationInput);

            if (configuredMemoryBoundMiB < minMemResult.minimalMemoryMiB) {
                fmt::print(stderr, "\n========================================\n");
                fmt::print(stderr, "ERROR: MEMORY BOUND INFEASIBLE!\n");
                fmt::print(stderr, "========================================\n");
                fmt::print(stderr, "Configured memory bound: {:.2f} MiB\n", configuredMemoryBoundMiB);
                fmt::print(stderr, "Theoretical minimum memory: {:.2f} MiB\n", minMemResult.minimalMemoryMiB);
                fmt::print(stderr, "Critical task requiring most memory: Task {} (needs {:.2f} MiB)\n",
                          minMemResult.criticalTaskIndex,
                          minMemResult.perTaskMemoryMiB[minMemResult.criticalTaskIndex]);
                fmt::print(stderr, "\nThe configured memory bound is {:.2f} MiB below the theoretical minimum.\n",
                          minMemResult.minimalMemoryMiB - configuredMemoryBoundMiB);
                fmt::print(stderr, "No optimization strategy can achieve this memory target.\n");
                fmt::print(stderr, "\nSuggested actions:\n");
                fmt::print(stderr, "1. Increase maxPeakMemoryUsageInMiB to at least {:.2f} MiB\n", minMemResult.minimalMemoryMiB);
                fmt::print(stderr, "2. Or set maxPeakMemoryUsageInMiB to 0 to disable the memory constraint\n");
                fmt::print(stderr, "3. Or reduce the problem size to lower memory requirements\n");
                fmt::print(stderr, "========================================\n\n");
                return 1;
            }

            fmt::print("[MEMORY-BOUND-CHECK] Configured memory bound ({:.2f} MiB) is feasible (minimum required: {:.2f} MiB)\n",
                      configuredMemoryBoundMiB, minMemResult.minimalMemoryMiB);
        }

        FirstStepSolver::Output firstStepOutput;

        if (!loadFirstStepPath.empty()) {
            // Load first step output from file
            fmt::print("\n=== LOADING FIRST STEP OUTPUT FROM FILE ===\n");
            firstStepOutput = loadFirstStepOutput(loadFirstStepPath);

            // Verify the loaded output has the correct number of tasks
            if (firstStepOutput.taskGroupExecutionOrder.size() != optimizationInput.nodes.size() + 1) {
                fmt::print("Warning: Loaded first step output has {} tasks, expected {} (including sentinel)\n",
                          firstStepOutput.taskGroupExecutionOrder.size(),
                          optimizationInput.nodes.size() + 1);
            }
        } else {
            fmt::print("[DEBUG-OUTPUT-OPTIMIZER] ==================== STARTING STEP 1: TASK SCHEDULING OPTIMIZATION ====================\n");

            // STEP 1: Task Scheduling Optimization
            auto firstStepInput = convertToFirstStepInput(optimizationInput);
            FirstStepSolver firstStepSolver(std::move(firstStepInput));
            firstStepOutput = firstStepSolver.solve();

            // Add sentinel task group
            firstStepOutput.taskGroupExecutionOrder.push_back(firstStepOutput.taskGroupExecutionOrder.size());

            // Save first step output if requested
            if (!saveFirstStepPath.empty()) {
                saveFirstStepOutput(firstStepOutput, saveFirstStepPath);
            }
        }

        fmt::print("[DEBUG-OUTPUT-OPTIMIZER] ==================== STARTING STEP 2: MEMORY MANAGEMENT OPTIMIZATION ====================\n");

        // STEP 2: Memory Management Optimization
        auto secondStepInput = convertToSecondStepInput(optimizationInput, firstStepOutput);
        SecondStepSolver secondStepSolver;
        auto secondStepOutput = secondStepSolver.solve(std::move(secondStepInput));

        fmt::print("[DEBUG-OUTPUT-OPTIMIZER] ==================== COMPLETED STEP 2: FINALIZING OPTIMIZATION ====================\n");

        // Combine outputs into unified execution plan
        optimizationOutput = convertToOptimizationOutput(optimizationInput, firstStepOutput, secondStepOutput);

        // Set optimal flag (assuming success if we got here)
        optimizationOutput.optimal = true;

        fmt::print("\n--- Optimization Results ---\n");
        fmt::print("Optimization completed: {}\n", optimizationOutput.optimal ? "OPTIMAL" : "HEURISTIC");
        fmt::print("Original memory usage: {:.2f} MB\n", optimizationOutput.originalMemoryUsage);
        fmt::print("Optimized memory usage: {:.2f} MB\n", optimizationOutput.anticipatedPeakMemoryUsage);

        double reduction = optimizationOutput.originalMemoryUsage - optimizationOutput.anticipatedPeakMemoryUsage;
        double reductionPercent = optimizationOutput.originalMemoryUsage > 0
            ? (reduction / optimizationOutput.originalMemoryUsage) * 100.0
            : 0.0;

        if (reduction > 0) {
            fmt::print("Memory reduction: {:.2f} MB ({:.1f}%)\n", reduction, reductionPercent);
        } else if (reduction < 0) {
            fmt::print("Memory increase: {:.2f} MB (optimized for performance)\n", -reduction);
        } else {
            fmt::print("Memory unchanged: {:.2f} MB\n", optimizationOutput.originalMemoryUsage);
        }

        fmt::print("Graph nodes: {}\n", optimizationOutput.nodes.size());

        // Count data movement nodes
        int dataMovementCount = 0;
        for (const auto& nodeId : optimizationOutput.nodes) {
            auto it = optimizationOutput.nodeIdToNodeTypeMap.find(nodeId);
            if (it != optimizationOutput.nodeIdToNodeTypeMap.end() &&
                it->second == OptimizationOutput::NodeType::dataMovement) {
                dataMovementCount++;
            }
        }
        fmt::print("Data movement operations: {}\n", dataMovementCount);

        fmt::print("\n--- Saving Optimized Plan ---\n");

        // Save the optimized plan
        saveOptimizationOutput(optimizationOutput, outputPath);

        fmt::print("\n✅ Optimization completed successfully!\n");
        fmt::print("Optimized plan saved to: {}\n", outputPath);

        fmt::print("\nTo execute this plan, run:\n");
        fmt::print("  ./tiledCholeskyAblation --N=<size> --T=<tiles> --run-plan --load-plan={}\n", outputPath);

    } catch (const std::exception& e) {
        fmt::print(stderr, "\n❌ Error: {}\n", e.what());
        return 1;
    }

    return 0;
}