#!/bin/bash
# Generate profile file for ablation studies
# Must be run from project root directory

set -e  # Exit on error

PROJECT_ROOT="/home/users/lingqi.zhang/workspace_x64/FRUGAL/BASE/optimize-cuda-memory-usage-v1"
cd "$PROJECT_ROOT"

# Default parameters
N=${1:-102400}
T=${2:-4}

# Output directory and file
OUTPUT_DIR="results/ablation/exp1"
PROFILE_FILE="$OUTPUT_DIR/profile_gapoverlap_enabled.json"

echo "=========================================="
echo "Generate Profile for Ablation Study"
echo "=========================================="
echo "Matrix size: N=$N, T=$T"
echo "Output: $PROFILE_FILE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate profile using tiledCholeskyAblation
echo "Generating profile (requires GPU)..."
./build/userApplications/tiledCholeskyAblation \
    --profile-only \
    --save-profile="$PROFILE_FILE" \
    --N=$N --T=$T

# Verify profile was created
if [ -f "$PROFILE_FILE" ]; then
    echo ""
    echo "✅ Profile generated successfully: $PROFILE_FILE"
    echo ""

    # Show profile summary
    echo "Profile Summary:"
    ARRAY_SIZE=$(jq '.arrays[0].size' "$PROFILE_FILE")
    NUM_ARRAYS=$(jq '.arrays | length' "$PROFILE_FILE")
    NUM_TASKS=$(jq '.taskGroups | length' "$PROFILE_FILE")
    ORIG_TIME=$(jq '.metadata.originalTotalRunningTime' "$PROFILE_FILE")

    echo "  Arrays: $NUM_ARRAYS"
    echo "  Array size: $ARRAY_SIZE bytes ($(echo "scale=2; $ARRAY_SIZE / 1024 / 1024 / 1024" | bc) GB)"
    echo "  Task groups: $NUM_TASKS"
    echo "  Original running time: ${ORIG_TIME}s"
else
    echo "❌ ERROR: Profile was not created"
    exit 1
fi
