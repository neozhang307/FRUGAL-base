#!/bin/bash
# Execute Test 1 (distance limit) plans on GPU to get actual runtime

set -e

PROJECT_ROOT="/home/users/lingqi.zhang/workspace_x64/FRUGAL/BASE/optimize-cuda-memory-usage-v1"
cd "$PROJECT_ROOT"

# Configuration
N=102400
T=4
EXP_DIR="results/ablation/exp3/test1_abstract_window"
PLAN_DIR="$EXP_DIR/plans"
OUTPUT_LOG="$EXP_DIR/execution.log"

# Distance limits tested
DISTANCES=(1 5 10 20 30)

echo "==========================================" | tee "$OUTPUT_LOG"
echo "Execute Test 1 (Distance Limit) Plans on GPU" | tee -a "$OUTPUT_LOG"
echo "==========================================" | tee -a "$OUTPUT_LOG"
echo "Workload: Tiled Cholesky N=$N, T=$T" | tee -a "$OUTPUT_LOG"
echo "Plans directory: $PLAN_DIR" | tee -a "$OUTPUT_LOG"
echo "Output log: $OUTPUT_LOG" | tee -a "$OUTPUT_LOG"
echo "" | tee -a "$OUTPUT_LOG"

# Execute each distance limit plan
for DIST in "${DISTANCES[@]}"; do
    PLAN_FILE="$PLAN_DIR/plan_dist${DIST}.json"

    if [ ! -f "$PLAN_FILE" ]; then
        echo "⚠️  Plan file not found: $PLAN_FILE" | tee -a "$OUTPUT_LOG"
        continue
    fi

    echo "=== Distance Limit: $DIST ===" | tee -a "$OUTPUT_LOG"

    # Get predicted runtime from middle results CSV
    MIDDLE_CSV="$EXP_DIR/window_distance_middle_results.csv"
    if [ -f "$MIDDLE_CSV" ]; then
        PREDICTED=$(awk -F',' -v d="$DIST" '$1 == d {print $3}' "$MIDDLE_CSV")
        echo "Predicted runtime: ${PREDICTED}s" | tee -a "$OUTPUT_LOG"
    fi

    echo "Executing solution..." | tee -a "$OUTPUT_LOG"

    # Run on GPU
    build/userApplications/tiledCholeskyAblation \
        --N=$N --T=$T \
        --run-plan \
        --load-plan="$PLAN_FILE" \
        2>&1 | tee -a "$OUTPUT_LOG"

    echo "✅ Distance limit $DIST completed" | tee -a "$OUTPUT_LOG"
    echo "" | tee -a "$OUTPUT_LOG"
done

echo "==========================================" | tee -a "$OUTPUT_LOG"
echo "✅ All distance limit plans executed" | tee -a "$OUTPUT_LOG"
echo "==========================================" | tee -a "$OUTPUT_LOG"
echo "" | tee -a "$OUTPUT_LOG"
echo "Next step: Generate plots" | tee -a "$OUTPUT_LOG"
echo "  python experiments/ablation/window/plot_window_real_runtime.py --test test1" | tee -a "$OUTPUT_LOG"
