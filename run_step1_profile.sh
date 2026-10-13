#!/bin/bash
# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

#
# Step 1: Run benchmarks and profiling across frameworks (vllm, sgl, trt)
#
# Usage: ./run_step1_profile.sh <run_config>
# Example: ./run_step1_profile.sh ./llm/auto_profile/test_configs/deepseek_r1_nvfp4/run_deepseek_r1_nvfp4.json

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <run_config>"
    echo "Example: $0 ./llm/auto_profile/test_configs/deepseek_r1_nvfp4/run_deepseek_r1_nvfp4.json"
    exit 1
fi


run_config="$1"

echo "========================================"
echo "  STEP 1: Run benchmarks and profiling"
echo "========================================"
echo "  Run config: ${run_config}"
echo "========================================"

./llm/auto_profile/run_profile_core.sh "${run_config}"

echo ""
echo "  Step 1 complete."
