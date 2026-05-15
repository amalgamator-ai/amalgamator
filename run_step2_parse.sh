#!/bin/bash
#
# Step 2: Parse profiling results, generate summary tables, and generate
#          per-test-case analysis configs (single-trace + cross-trace).
#
# Usage: ./run_step2_parse.sh <run_config>
# Example: ./run_step2_parse.sh ./auto_profile/test_configs/deepseek_r1_nvfp4/run_deepseek_r1_nvfp4.json

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <run_config>"
    echo "Example: $0 ./auto_profile/test_configs/deepseek_r1_nvfp4/run_deepseek_r1_nvfp4.json"
    exit 1
fi

source "$(dirname "$0")/env.sh"

run_config="$1"
output_dir=$(python -c "from common.utils import output_dir_from_run_config; print(output_dir_from_run_config('${run_config}'))")

echo "========================================"
echo "  STEP 2: Parse results and generate analysis configs"
echo "========================================"
echo "  Run config: ${run_config}"
echo "  Output dir: ${output_dir}"
echo "========================================"

python -m auto_profile.run_profile_summary \
    --run-config "${run_config}" \
    --override-output-dir

echo ""
echo "  Step 2 complete."
