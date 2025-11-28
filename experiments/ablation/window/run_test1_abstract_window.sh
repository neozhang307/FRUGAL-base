#!/bin/bash
# Test 1: Abstract Window Size Ablation
# Variable: Distance limits (1, 5, 10, 20, 30)
# Fixed: Time factors = 1000.0 (effectively infinite)

set -e

PROJECT_ROOT="/home/users/lingqi.zhang/workspace_x64/FRUGAL/BASE/optimize-cuda-memory-usage-v1"
cd "$PROJECT_ROOT"

# Configuration
PROFILE="results/ablation/exp1/profile_gapoverlap_enabled.json"
FIRST_STEP_SOLUTION="results/ablation/exp1/topk_extensive/first_step/topk100.json"
SOLUTION_ID=84  # 6th fastest solution
CONFIG="experiments/ablation/window/config.json"
EXP_DIR="results/ablation/exp3/test1_abstract_window"

# Create directories
mkdir -p "$EXP_DIR/logs"
mkdir -p "$EXP_DIR/plans"

# Distance limits to test
DISTANCES=(1 5 10 20 30)

# Fixed time factors (set to very large = infinite)
TIME_FACTOR=1000.0

echo "=========================================="
echo "Test 1: Abstract Window Size Ablation"
echo "=========================================="
echo "Base solution: Top-100 Solution #$SOLUTION_ID"
echo "Profile: $PROFILE"
echo "Distance limits: ${DISTANCES[@]}"
echo "Time factors (fixed): $TIME_FACTOR"
echo "Output: $EXP_DIR"
echo ""

# Test each distance limit
for DIST in "${DISTANCES[@]}"; do
    echo ""
    echo "=== Testing Distance Limit: $DIST ==="

    PLAN_FILE="$EXP_DIR/plans/plan_dist${DIST}.json"
    LOG_FILE="$EXP_DIR/logs/dist${DIST}.log"
    CONFIG_TEMP="$EXP_DIR/config_dist${DIST}.json"

    # Create temporary config with current window settings
    jq "
        .optimization.prefetchLookbackDistanceLimit = $DIST |
        .optimization.offloadLookaheadDistanceLimit = $DIST |
        .optimization.prefetchLookbackTimeBudgetFactor = $TIME_FACTOR |
        .optimization.offloadLookaheadComputeTimeFactor = $TIME_FACTOR
    " "$CONFIG" > "$CONFIG_TEMP"

    echo "Running optimization with distance limit = $DIST..."

    # Run standalone optimizer with specific first-step solution
    build/tools/standaloneOptimizer \
        "$PROFILE" \
        "$PLAN_FILE" \
        --config="$CONFIG_TEMP" \
        --load-topk="$FIRST_STEP_SOLUTION" \
        --solution-index=$SOLUTION_ID \
        > "$LOG_FILE" 2>&1

    # Extract key metrics
    if grep -q "OPTIMAL" "$LOG_FILE"; then
        STATUS="OPTIMAL"
    elif grep -q "INFEASIBLE" "$LOG_FILE"; then
        STATUS="INFEASIBLE"
    else
        STATUS="UNKNOWN"
    fi

    MIP_TIME=$(grep "Time for solving the MIP problem" "$LOG_FILE" | awk '{print $8}' || echo "N/A")
    PREDICTED=$(grep "Total running time (s):" "$LOG_FILE" | head -1 | awk '{print $5}' || echo "N/A")
    PEAK_MEM=$(grep "Optimal peak memory usage (MiB):" "$LOG_FILE" | awk '{print $6}' || echo "N/A")

    echo "  Distance: $DIST"
    echo "  MIP solve time: ${MIP_TIME}s"
    echo "  Predicted runtime: ${PREDICTED}s"
    echo "  Peak memory: ${PEAK_MEM} MiB"
    echo "  Status: $STATUS"

    # Clean up temp config
    rm "$CONFIG_TEMP"
done

echo ""
echo "=========================================="
echo "✅ Test 1 complete"
echo "Results saved to: $EXP_DIR"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Parse results: experiments/ablation/parse_window_results.sh test1"
echo "  2. Run on GPU: experiments/ablation/run_window_solutions.sh test1"
echo "  3. Analyze: python experiments/ablation/analyze_task_window.py"
