#!/bin/bash
# Run ONLY the feature detection step
#
# Usage:
#   ./kagenti_to_openshell/run_detect_features.sh
#   ./kagenti_to_openshell/run_detect_features.sh --resume
#   ./kagenti_to_openshell/run_detect_features.sh --config path/to/config

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

CONFIG="${SCRIPT_DIR}/configs/default_config.json"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "============================================"
echo "Kagenti -> OpenShell Feature Detection"
echo "============================================"
echo "Config: ${CONFIG}"
echo "============================================"

python -m kagenti_to_openshell.run_detect_features --config "${CONFIG}" "${EXTRA_ARGS[@]}"
