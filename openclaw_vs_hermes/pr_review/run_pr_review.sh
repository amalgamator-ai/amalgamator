#!/bin/bash
# Address PR review comments for a ported feature
#
# Usage:
#   ./openclaw_vs_hermes/pr_review/run_pr_review.sh --config openclaw_vs_hermes/pr_review/configs/pr_review_config.json
#   ./openclaw_vs_hermes/pr_review/run_pr_review.sh --config openclaw_vs_hermes/pr_review/configs/pr_review_config.json --round 2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"

CONFIG=""
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

if [[ -z "${CONFIG}" ]]; then
    echo "Usage: $0 --config <path-to-pr-review-config.json> [--round N]"
    exit 1
fi

echo "============================================"
echo "PR Review Automation"
echo "============================================"
echo "Config: ${CONFIG}"
echo "Project root: ${PROJECT_ROOT}"
echo "============================================"

python -m openclaw_vs_hermes.pr_review.run_pr_review --config "${CONFIG}" "${EXTRA_ARGS[@]}"
