#!/bin/bash
# Execute all 100 Top-K solutions and measure actual GPU performance

set -e

PROJECT_ROOT="/home/users/lingqi.zhang/workspace_x64/FRUGAL/BASE/optimize-cuda-memory-usage-v1"
cd "$PROJECT_ROOT"

# Experiment settings
N=102400
T=4
EXP_DIR="results/ablation/exp1/topk_extensive"
PROFILE="results/ablation/exp1/profile_gapoverlap_enabled.json"
TOPK_FILE="$EXP_DIR/first_step/topk100.json"
PLAN_DIR="$EXP_DIR/second_step_plans"
CONFIG="config.json"
OUTPUT_LOG="$EXP_DIR/all_topk100_execution.log"

echo "=========================================="
echo "Execute All Top-100 Solutions on GPU"
echo "=========================================="
echo "Profile: $PROFILE"
echo "Top-K file: $TOPK_FILE"
echo "Plans directory: $PLAN_DIR"
echo "Output log: $OUTPUT_LOG"
echo ""

# Check dependencies
if [ ! -f "$TOPK_FILE" ]; then
    echo "❌ ERROR: Top-K file not found: $TOPK_FILE"
    exit 1
fi

if [ ! -d "$PLAN_DIR" ]; then
    echo "❌ ERROR: Plans directory not found: $PLAN_DIR"
    exit 1
fi

# Count plans
NUM_PLANS=$(ls -1 $PLAN_DIR/plan_sol*.json 2>/dev/null | wc -l)
echo "Found $NUM_PLANS optimized plans"
echo ""

# Start execution
echo "=== Executing All Solutions ===" | tee "$OUTPUT_LOG"
START_TIME=$(date +%s)

for i in $(seq 0 99); do
    PLAN_FILE="$PLAN_DIR/plan_sol${i}.json"

    if [ ! -f "$PLAN_FILE" ]; then
        echo "Solution #$i: ❌ SKIP (plan not found)" | tee -a "$OUTPUT_LOG"
        continue
    fi

    echo "" | tee -a "$OUTPUT_LOG"
    echo "=== Solution #$i ===" | tee -a "$OUTPUT_LOG"

    # Get beam score from TopK file
    SCORE=$(jq -r ".solutions[$i].dataReuseScore" "$TOPK_FILE")
    echo "Beam search score: $SCORE" | tee -a "$OUTPUT_LOG"

    # Get predicted runtime from optimization log
    OPT_LOG="$PLAN_DIR/optimization_sol${i}.log"
    if [ -f "$OPT_LOG" ]; then
        PREDICTED=$(grep "Total running time (s):" "$OPT_LOG" | head -1 | awk '{print $5}')
        echo "Predicted runtime: ${PREDICTED}s" | tee -a "$OUTPUT_LOG"
    fi

    # Execute the plan and capture actual runtime
    echo "Executing solution #$i..." | tee -a "$OUTPUT_LOG"

    build/userApplications/tiledCholeskyAblation \
        --N=$N --T=$T \
        --run-plan \
        --load-plan="$PLAN_FILE" \
        2>&1 | tee -a "$OUTPUT_LOG" | grep -E "Execution time:|Peak GPU memory"

    echo "✅ Solution #$i completed" | tee -a "$OUTPUT_LOG"
done

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo "" | tee -a "$OUTPUT_LOG"
echo "=========================================="  | tee -a "$OUTPUT_LOG"
echo "✅ All 100 solutions executed" | tee -a "$OUTPUT_LOG"
echo "Total execution time: ${TOTAL_TIME} seconds" | tee -a "$OUTPUT_LOG"
echo "=========================================="  | tee -a "$OUTPUT_LOG"
echo "" | tee -a "$OUTPUT_LOG"

echo "Output saved to: $OUTPUT_LOG"
echo ""
echo "Next step: Run parse script to analyze results"
echo "  python experiments/ablation/parse_topk100_execution.py"
