#!/bin/bash
#
# Full pipeline orchestrator: profile, parse, and analyze.
# Runs all 3 steps sequentially. Each step can also be run individually.
#
# Usage: ./run_all.sh <run_config> [options]
# Example: ./run_all.sh ./auto_profile/test_configs/deepseek_r1_nvfp4/run_deepseek_r1_nvfp4.json
#          ./run_all.sh ./auto_profile/test_configs/deepseek_r1_nvfp4/run_deepseek_r1_nvfp4.json --skip-profile --skip-pdf --skip-jiras
#
# Options:
#   --skip-profile  Skip step 1: benchmarks and profiling
#   --skip-parse    Skip step 2: parse results and generate analysis configs
#   --skip-analyze  Skip step 3: analysis pipeline (all sub-steps)
#   --skip-pdf      Skip step 3c: summary PDF report generation
#   --skip-trace    Skip step 3d: Chrome trace visualization generation
#   --skip-jiras    Skip step 3e: JIRA task creation

set -euo pipefail

run_config=""
skip_profile=false
skip_parse=false
skip_analyze=false
step3_flags=()

for arg in "$@"; do
    case "$arg" in
        --skip-profile) skip_profile=true ;;
        --skip-parse)   skip_parse=true ;;
        --skip-analyze) skip_analyze=true ;;
        --skip-pdf)     step3_flags+=("--skip-pdf") ;;
        --skip-trace)   step3_flags+=("--skip-trace") ;;
        --skip-jiras)   step3_flags+=("--skip-jiras") ;;
        -*)
            echo "Unknown option: $arg"
            echo "Usage: $0 <run_config> [--skip-profile] [--skip-parse] [--skip-analyze] [--skip-pdf] [--skip-trace] [--skip-jiras]"
            exit 1
            ;;
        *)
            if [ -z "$run_config" ]; then
                run_config="$arg"
            else
                echo "Error: unexpected argument: $arg"
                exit 1
            fi
            ;;
    esac
done

if [ -z "$run_config" ]; then
    echo "Usage: $0 <run_config> [--skip-profile] [--skip-parse] [--skip-analyze] [--skip-pdf] [--skip-trace] [--skip-jiras]"
    echo "Example: $0 ./auto_profile/test_configs/deepseek_r1_nvfp4/run_deepseek_r1_nvfp4.json"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================"
echo "  AI Auto Performance Analysis Pipeline"
echo "========================================"
echo "  Run config:     ${run_config}"
echo "  Skip profile:   ${skip_profile}"
echo "  Skip parse:     ${skip_parse}"
echo "  Skip analyze:   ${skip_analyze}"
if [ "${#step3_flags[@]}" -gt 0 ]; then
echo "  Step 3 flags:   ${step3_flags[*]}"
fi
echo "========================================"

# Step 1: Run benchmarks and profiling
if [ "$skip_profile" = false ]; then
    echo ""
    echo "=== STEP 1: Run benchmarks and profiling ==="
    "${SCRIPT_DIR}/run_step1_profile.sh" "${run_config}"
else
    echo ""
    echo "=== STEP 1: SKIPPED ==="
fi

# Step 2: Parse results and generate analysis configs
if [ "$skip_parse" = false ]; then
    echo ""
    echo "=== STEP 2: Parse results and generate analysis configs ==="
    "${SCRIPT_DIR}/run_step2_parse.sh" "${run_config}"
else
    echo ""
    echo "=== STEP 2: SKIPPED ==="
fi

# Step 3: Run analysis pipeline
if [ "$skip_analyze" = false ]; then
    echo ""
    echo "=== STEP 3: Run analysis pipeline ==="
    "${SCRIPT_DIR}/run_step3_analyze.sh" "${run_config}" ${step3_flags[@]+"${step3_flags[@]}"}
else
    echo ""
    echo "=== STEP 3: SKIPPED ==="
fi

echo ""
echo "========================================"
echo "  Pipeline complete"
echo "========================================"
