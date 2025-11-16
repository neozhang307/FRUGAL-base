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

// Forward declare the optimizeGraph function
namespace memopt {
    class Optimizer {
    public:
        static Optimizer* getInstance();
        OptimizationOutput optimizeGraph(const OptimizationInput& input);
    };
}

using namespace memopt;

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
    fmt::print("  --help, -h            Show this help message\n");
    fmt::print("\nExamples:\n");
    fmt::print("  # Basic usage with default config\n");
    fmt::print("  {} profile.json optimized.json\n", programName);
    fmt::print("\n  # With custom memory bound and solver\n");
    fmt::print("  {} profile.json optimized.json --memory-bound=1000 --solver=GREEDY\n", programName);
    fmt::print("\n  # With custom config file\n");
    fmt::print("  {} profile.json optimized.json --config=custom_config.json\n", programName);
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

        // Get optimizer instance and run optimization
        auto optimizer = Optimizer::getInstance();
        auto optimizationOutput = optimizer->optimizeGraph(optimizationInput);

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