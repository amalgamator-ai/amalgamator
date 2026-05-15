#!/bin/bash
#
# Full pipeline orchestrator: profile, parse, and analyze.
# Runs all 3 steps sequentially. Each step can also be run individually.
#
# Usage: ./run_all.sh <run_config>
# Example: ./run_all.sh ./auto_profile/test_configs/deepseek_r1_nvfp4/run_deepseek_r1_nvfp4.json

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <run_config>"
    echo "Example: $0 ./auto_profile/test_configs/deepseek_r1_nvfp4/run_deepseek_r1_nvfp4.json"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
run_config="$1"

echo "========================================"
echo "  AI Auto Performance Analysis Pipeline"
echo "========================================"
echo "  Run config: ${run_config}"
echo "========================================"

# Step 1: Run benchmarks and profiling
echo ""
echo "=== STEP 1: Run benchmarks and profiling ==="
"${SCRIPT_DIR}/run_step1_profile.sh" "${run_config}"

# Step 2: Parse results and generate analysis configs
echo ""
echo "=== STEP 2: Parse results and generate analysis configs ==="
"${SCRIPT_DIR}/run_step2_parse.sh" "${run_config}"

# Step 3: Run analysis pipeline
echo ""
echo "=== STEP 3: Run analysis pipeline ==="
"${SCRIPT_DIR}/run_step3_analyze.sh" "${run_config}"

echo ""
echo "========================================"
echo "  Pipeline complete"
echo "========================================"
