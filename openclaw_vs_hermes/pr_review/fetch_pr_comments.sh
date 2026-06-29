#!/bin/bash
# Fetch all PR comments from a GitHub PR and write to PR_comments_<ROUND>.txt
#
# Usage:
#   ./openclaw_vs_hermes/pr_review/fetch_pr_comments.sh <PR_URL> <OUTPUT_DIR> [ROUND_ID]
#
# Examples:
#   ./openclaw_vs_hermes/pr_review/fetch_pr_comments.sh https://github.com/openclaw/openclaw/pull/97580 ./openclaw_vs_hermes/output_iteration_budget 1
#   ./openclaw_vs_hermes/pr_review/fetch_pr_comments.sh https://github.com/openclaw/openclaw/pull/97580 ./openclaw_vs_hermes/output_iteration_budget 2
#
# If ROUND_ID is omitted, auto-detects the next round from existing PR_comments_N.txt files.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <PR_URL> <OUTPUT_DIR> [ROUND_ID]"
    echo ""
    echo "  PR_URL     GitHub PR URL (e.g., https://github.com/openclaw/openclaw/pull/97580)"
    echo "  OUTPUT_DIR Directory where PR_comments_<ROUND>.txt will be written"
    echo "  ROUND_ID   Review round number (auto-detected if omitted)"
    exit 1
fi

PR_URL="$1"
OUTPUT_DIR="$2"

# Parse owner/repo/pr_number from URL
if [[ ! "$PR_URL" =~ ^https://github\.com/([^/]+)/([^/]+)/pull/([0-9]+)$ ]]; then
    echo "ERROR: Invalid PR URL format: $PR_URL"
    echo "Expected: https://github.com/<owner>/<repo>/pull/<number>"
    exit 1
fi

OWNER="${BASH_REMATCH[1]}"
REPO="${BASH_REMATCH[2]}"
PR_NUMBER="${BASH_REMATCH[3]}"

# Validate prerequisites
if ! command -v gh &>/dev/null; then
    echo "ERROR: gh CLI is not installed. Install from https://cli.github.com/"
    exit 1
fi

if ! gh auth status &>/dev/null; then
    echo "ERROR: gh CLI is not authenticated. Run: gh auth login"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Auto-detect round ID if not provided
if [[ $# -ge 3 ]]; then
    ROUND_ID="$3"
else
    ROUND_ID=1
    for f in "$OUTPUT_DIR"/PR_comments_*.txt; do
        if [[ -f "$f" ]]; then
            base="$(basename "$f")"
            num="${base#PR_comments_}"
            num="${num%.txt}"
            if [[ "$num" =~ ^[0-9]+$ ]] && [[ "$num" -ge "$ROUND_ID" ]]; then
                ROUND_ID=$((num + 1))
            fi
        fi
    done
fi

OUTPUT_FILE="${OUTPUT_DIR}/PR_comments_${ROUND_ID}.txt"

echo "============================================"
echo "Fetching PR Comments"
echo "============================================"
echo "PR:         ${OWNER}/${REPO}#${PR_NUMBER}"
echo "URL:        ${PR_URL}"
echo "Round:      ${ROUND_ID}"
echo "Output:     ${OUTPUT_FILE}"
echo "============================================"

# Start building the output file
cat > "$OUTPUT_FILE" <<HEADER
================================================================================
PR REVIEW COMMENTS — Round ${ROUND_ID}
================================================================================
PR:         ${OWNER}/${REPO}#${PR_NUMBER}
URL:        ${PR_URL}
Fetched:    $(date -u '+%Y-%m-%d %H:%M:%S UTC')
Round:      ${ROUND_ID}
================================================================================

HEADER

# ---------------------------------------------------------------------------
# Fetch inline code review comments
# ---------------------------------------------------------------------------

INLINE_COUNT=0
{
    echo "================================================================================"
    echo "INLINE CODE REVIEW COMMENTS"
    echo "================================================================================"
    echo ""
} >> "$OUTPUT_FILE"

INLINE_OUTPUT=$(gh api --paginate "repos/${OWNER}/${REPO}/pulls/${PR_NUMBER}/comments" \
    --jq '.[] | "--- INLINE COMMENT ---\nAuthor: \(.user.login)\nCreated: \(.created_at)\nFile: \(.path)\nLine: \(.line // .original_line // "N/A")\nSide: \(.side // "N/A")\nIn reply to: \(.in_reply_to_id // "N/A")\n\nDiff hunk:\n\(.diff_hunk)\n\nComment:\n\(.body)\n"' 2>/dev/null || true)

if [[ -n "$INLINE_OUTPUT" ]]; then
    echo "$INLINE_OUTPUT" >> "$OUTPUT_FILE"
    INLINE_COUNT=$(echo "$INLINE_OUTPUT" | grep -c "^--- INLINE COMMENT ---$" || true)
fi
echo "" >> "$OUTPUT_FILE"

# ---------------------------------------------------------------------------
# Fetch review summaries
# ---------------------------------------------------------------------------

REVIEW_COUNT=0
{
    echo "================================================================================"
    echo "REVIEW SUMMARIES"
    echo "================================================================================"
    echo ""
} >> "$OUTPUT_FILE"

REVIEW_OUTPUT=$(gh api --paginate "repos/${OWNER}/${REPO}/pulls/${PR_NUMBER}/reviews" \
    --jq '.[] | select(.body != "" and .body != null) | "--- REVIEW SUMMARY ---\nAuthor: \(.user.login)\nState: \(.state)\nSubmitted: \(.submitted_at)\n\nBody:\n\(.body)\n"' 2>/dev/null || true)

if [[ -n "$REVIEW_OUTPUT" ]]; then
    echo "$REVIEW_OUTPUT" >> "$OUTPUT_FILE"
    REVIEW_COUNT=$(echo "$REVIEW_OUTPUT" | grep -c "^--- REVIEW SUMMARY ---$" || true)
fi
echo "" >> "$OUTPUT_FILE"

# ---------------------------------------------------------------------------
# Fetch discussion comments
# ---------------------------------------------------------------------------

DISCUSSION_COUNT=0
{
    echo "================================================================================"
    echo "DISCUSSION COMMENTS"
    echo "================================================================================"
    echo ""
} >> "$OUTPUT_FILE"

DISCUSSION_OUTPUT=$(gh api --paginate "repos/${OWNER}/${REPO}/issues/${PR_NUMBER}/comments" \
    --jq '.[] | "--- DISCUSSION COMMENT ---\nAuthor: \(.user.login)\nCreated: \(.created_at)\nUpdated: \(.updated_at)\n\nBody:\n\(.body)\n"' 2>/dev/null || true)

if [[ -n "$DISCUSSION_OUTPUT" ]]; then
    echo "$DISCUSSION_OUTPUT" >> "$OUTPUT_FILE"
    DISCUSSION_COUNT=$(echo "$DISCUSSION_OUTPUT" | grep -c "^--- DISCUSSION COMMENT ---$" || true)
fi
echo "" >> "$OUTPUT_FILE"

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

TOTAL=$((INLINE_COUNT + REVIEW_COUNT + DISCUSSION_COUNT))

{
    echo "================================================================================"
    echo "SUMMARY"
    echo "================================================================================"
    echo "Inline code comments: ${INLINE_COUNT}"
    echo "Review summaries:     ${REVIEW_COUNT}"
    echo "Discussion comments:  ${DISCUSSION_COUNT}"
    echo "Total:                ${TOTAL}"
    echo "================================================================================"
} >> "$OUTPUT_FILE"

echo ""
echo "============================================"
echo "Done. ${TOTAL} comments fetched."
echo "  Inline:     ${INLINE_COUNT}"
echo "  Reviews:    ${REVIEW_COUNT}"
echo "  Discussion: ${DISCUSSION_COUNT}"
echo "Output: ${OUTPUT_FILE}"
echo ""
echo "Next step:"
echo "  python -m openclaw_vs_hermes.pr_review.run_pr_review --config <config> --round ${ROUND_ID}"
echo "============================================"
