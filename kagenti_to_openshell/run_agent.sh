#!/bin/bash
# Run the Kagenti -> OpenShell feature porting agent
#
# Usage:
#   ./kagenti_to_openshell/run_agent.sh                          # Full pipeline (detect + port feature #1)
#   ./kagenti_to_openshell/run_agent.sh --detect-only            # Only detect features
#   ./kagenti_to_openshell/run_agent.sh --feature-id 2           # Port specific feature
#   ./kagenti_to_openshell/run_agent.sh --resume                 # Resume interrupted run
#   ./kagenti_to_openshell/run_agent.sh --config path/to/config  # Custom config

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

CONFIG="${SCRIPT_DIR}/configs/default_config.json"
EXTRA_ARGS=()

# Parse arguments
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
echo "Kagenti -> OpenShell Feature Porting Agent"
echo "============================================"
echo "Config: ${CONFIG}"
echo "Project root: ${PROJECT_ROOT}"
echo "============================================"

python -m kagenti_to_openshell.run_agent --config "${CONFIG}" "${EXTRA_ARGS[@]}"
