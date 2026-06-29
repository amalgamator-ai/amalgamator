#!/bin/bash
# Generate a PR body with required "What Problem This Solves" and "Evidence"
# sections from pipeline artifacts.
#
# Usage:
#   ./openclaw_vs_hermes/pr_review/generate_pr_body.sh <PR_REVIEW_CONFIG>
#
# Writes PR_body.md to the output directory.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <PR_REVIEW_CONFIG>"
    exit 1
fi

CONFIG_FILE="$1"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

OUTPUT_DIR=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['output_dir'])")
SOURCE_DIR=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['openclaw_source_dir'])")
BRANCH=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['branch_name'])")

PR_BODY_FILE="${OUTPUT_DIR}/PR_body.md"

# Get the FIRST (original feature) commit on the branch — the oldest one after main.
# reverse gives oldest-first; head -1 picks the feature commit, skipping fix commits.
FEATURE_COMMIT=$(git -C "$SOURCE_DIR" log --format='%H' --reverse origin/main..HEAD | head -1)
COMMIT_DESC=$(git -C "$SOURCE_DIR" log --format='%b' -1 "$FEATURE_COMMIT")

# Extract test results from runtime_success_result.txt if available
TEST_RESULTS=""
if [[ -f "${OUTPUT_DIR}/runtime_success_result.txt" ]]; then
    TEST_RESULTS=$(cat "${OUTPUT_DIR}/runtime_success_result.txt")
fi

# Extract behavior proof if available (use latest round)
BEHAVIOR_PROOF=""
for f in "${OUTPUT_DIR}"/behavior_proof_R*.txt; do
    if [[ -f "$f" ]]; then
        BEHAVIOR_PROOF=$(cat "$f")
    fi
done

# Build the PR body
cat > "$PR_BODY_FILE" <<PRBODY
${COMMIT_DESC}

## What Problem This Solves

OpenClaw has no explicit iteration budget for agent runs. The agent loop runs until the LLM stops calling tools or a timeout fires. Existing loop detection is heuristic (content-hash dedup) which fails when volatile messageIds defeat all detection paths. This causes runaway tool-calling loops, OOM kills, and unbounded API costs.

Addresses: #6890, #75468, #89090, #79301, #75355, #77625, #74859, #82518

## Evidence

### Test Results

PRBODY

if [[ -n "$TEST_RESULTS" ]]; then
    cat >> "$PR_BODY_FILE" <<TESTS

\`\`\`
${TEST_RESULTS}
\`\`\`

TESTS
fi

# Add behavior proof
if [[ -n "$BEHAVIOR_PROOF" ]]; then
    cat >> "$PR_BODY_FILE" <<PROOF

### Behavior Proof

<details>
<summary>Terminal output showing iteration budget stopping a runaway agent loop</summary>

\`\`\`
${BEHAVIOR_PROOF}
\`\`\`

</details>

PROOF
fi

# Add test file listing
cat >> "$PR_BODY_FILE" <<TESTFILES

### Test Coverage

Test files covering this feature:

- \`src/agents/iteration-budget.test.ts\` — core budget consume/refund/exhaustion
- \`src/agents/agent-scope.test.ts\` — config merging and defaults
- \`src/agents/embedded-agent-runner/run/budget-exhaustion.test.ts\` — result builder and summary instruction
- \`src/config/zod-schema.agent-defaults.test.ts\` — schema validation
- \`packages/agent-core/src/agent-loop.test.ts\` — onBeforeToolCallingRound hook
TESTFILES

echo ""
echo "PR body written to: ${PR_BODY_FILE}"
echo "Preview:"
echo "---"
head -30 "$PR_BODY_FILE"
echo "..."
echo "---"
