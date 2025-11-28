#!/bin/bash
# Generate Top-100 solutions from beam search and create optimized plans for all

set -e  # Exit on error

PROJECT_ROOT="/home/users/lingqi.zhang/workspace_x64/FRUGAL/BASE/optimize-cuda-memory-usage-v1"
cd "$PROJECT_ROOT"

# Experiment settings
N=102400
T=4
BEAM_WIDTH=100
TOPK=100

# Directories
EXP_DIR="results/ablation/exp1/topk_extensive"
PROFILE="results/ablation/exp1/profile_gapoverlap_enabled.json"
TOPK_FILE="$EXP_DIR/first_step/topk100.json"
CONFIG="config.json"

# Create directories
mkdir -p "$EXP_DIR"/{first_step,second_step_plans,logs}

echo "=========================================="
echo "Top-100 Generation Experiment"
echo "=========================================="
echo "Profile: $PROFILE"
echo "Beam width: $BEAM_WIDTH"
echo "Top-K: $TOPK"
echo "Output: $TOPK_FILE"
echo ""

# Step 1: Generate Top-100 solutions using standaloneOptimizer
echo "=== Step 1: Generate Top-100 Solutions ==="
echo "Using standaloneOptimizer with beam search..."

build/tools/standaloneOptimizer \
    "$PROFILE" \
    "$EXP_DIR/first_step/dummy_output.json" \
    --config="$CONFIG" \
    --top-k=$TOPK \
    --save-topk="$TOPK_FILE" \
    --first-step-only \
    2>&1 | tee "$EXP_DIR/logs/step1_topk100_generation.log"

# Verify Top-K file was created
if [ ! -f "$TOPK_FILE" ]; then
    echo "❌ ERROR: Top-K file was not created: $TOPK_FILE"
    exit 1
fi

echo "✅ Top-100 solutions generated: $TOPK_FILE"

# Show summary
NUM_SOLUTIONS=$(jq '.numSolutions' "$TOPK_FILE")
echo ""
echo "Top-K Summary:"
echo "  Solutions: $NUM_SOLUTIONS"
echo "  File: $TOPK_FILE"
echo "  Size: $(du -h "$TOPK_FILE" | cut -f1)"

# Extract score distribution
echo ""
echo "Score Distribution:"
jq -r '.solutions[] | "\(.rank): \(.dataReuseScore)"' "$TOPK_FILE" | \
    awk '{bytes=$2; gb=bytes/(1024^3); print $1, sprintf("%.2f GB", gb)}' | \
    head -10
echo "..."
jq -r '.solutions[] | "\(.rank): \(.dataReuseScore)"' "$TOPK_FILE" | \
    awk '{bytes=$2; gb=bytes/(1024^3); print $1, sprintf("%.2f GB", gb)}' | \
    tail -5

echo ""
echo "✅ Step 1 Complete: Top-100 solutions generated"
echo "=========================================="
