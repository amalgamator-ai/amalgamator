#!/bin/bash
# Push PR review fixes to upstream and post a summary comment.
#
# Usage:
#   ./openclaw_vs_hermes/pr_review/push_pr_update.sh <PR_REVIEW_CONFIG> [ROUND_ID]
#
# Examples:
#   ./openclaw_vs_hermes/pr_review/push_pr_update.sh openclaw_vs_hermes/pr_review/configs/pr_review_config_iteration-budget-agent-loop-safety.json
#   ./openclaw_vs_hermes/pr_review/push_pr_update.sh openclaw_vs_hermes/pr_review/configs/pr_review_config_iteration-budget-agent-loop-safety.json 1
#   ./openclaw_vs_hermes/pr_review/push_pr_update.sh openclaw_vs_hermes/pr_review/configs/pr_review_config_multi-pass-context-compaction.json 2
#
# If ROUND_ID is omitted, auto-detects from the highest PR_fixes_N.txt in the output dir.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <PR_REVIEW_CONFIG> [ROUND_ID]"
    echo ""
    echo "  PR_REVIEW_CONFIG  Path to the PR review JSON config"
    echo "  ROUND_ID          Review round to post (auto-detected if omitted)"
    exit 1
fi

CONFIG_FILE="$1"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Parse all fields from config
OUTPUT_DIR=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['output_dir'])")
SOURCE_DIR=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['openclaw_source_dir'])")
BRANCH=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['branch_name'])")
PR_URL=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['pr_url'])")

# Extract owner/repo/pr_number from PR URL
if [[ ! "$PR_URL" =~ ^https://github\.com/([^/]+)/([^/]+)/pull/([0-9]+)$ ]]; then
    echo "ERROR: Invalid pr_url in config: $PR_URL"
    echo "Expected: https://github.com/<owner>/<repo>/pull/<number>"
    exit 1
fi

OWNER="${BASH_REMATCH[1]}"
REPO="${BASH_REMATCH[2]}"
PR_NUMBER="${BASH_REMATCH[3]}"

# Auto-detect round ID if not provided
if [[ $# -ge 2 ]]; then
    ROUND_ID="$2"
else
    ROUND_ID=0
    for f in "$OUTPUT_DIR"/PR_fixes_*.txt; do
        if [[ -f "$f" ]]; then
            base="$(basename "$f")"
            num="${base#PR_fixes_}"
            num="${num%.txt}"
            if [[ "$num" =~ ^[0-9]+$ ]] && [[ "$num" -gt "$ROUND_ID" ]]; then
                ROUND_ID="$num"
            fi
        fi
    done
    if [[ "$ROUND_ID" -eq 0 ]]; then
        echo "ERROR: No PR_fixes_N.txt found in $OUTPUT_DIR"
        echo "Run the PR review pipeline first."
        exit 1
    fi
fi

FIXES_FILE="${OUTPUT_DIR}/PR_fixes_${ROUND_ID}.txt"

if [[ ! -f "$FIXES_FILE" ]]; then
    echo "ERROR: $FIXES_FILE not found."
    echo "Run the PR review pipeline for round $ROUND_ID first."
    exit 1
fi

echo "============================================"
echo "Push PR Review Fixes"
echo "============================================"
echo "PR:          ${OWNER}/${REPO}#${PR_NUMBER}"
echo "URL:         ${PR_URL}"
echo "Branch:      ${BRANCH}"
echo "Source dir:  ${SOURCE_DIR}"
echo "Round:       ${ROUND_ID}"
echo "Fixes file:  ${FIXES_FILE}"
echo "============================================"
echo ""

# Verify we're on the correct branch
CURRENT_BRANCH=$(git -C "$SOURCE_DIR" branch --show-current)
if [[ "$CURRENT_BRANCH" != "$BRANCH" ]]; then
    echo "ERROR: Source dir is on branch '$CURRENT_BRANCH', expected '$BRANCH'."
    exit 1
fi

# Show what will be pushed
echo "Commits to push:"
git -C "$SOURCE_DIR" log --oneline origin/main..HEAD
echo ""

# Push
echo "Pushing $BRANCH to myfork..."
git -C "$SOURCE_DIR" push myfork "$BRANCH" --force-with-lease
echo ""
echo "Push complete."
echo ""

# Post comment
echo "Posting round $ROUND_ID summary as PR comment..."
gh pr comment "$PR_NUMBER" --repo "${OWNER}/${REPO}" --body-file "$FIXES_FILE"
echo ""
echo "============================================"
echo "Done."
echo "  PR updated: ${PR_URL}"
echo "  Summary posted as comment on the PR."
echo "============================================"
