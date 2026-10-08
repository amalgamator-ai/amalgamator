#!/bin/bash
# Step 2b -- suggest reviewers: generate the draft comment for the PR.
#
#   Network access : YES -- read-only 'gh api' searches (HTTPS + gh token).
#                    No write to GitHub. Nothing is posted here.
#   Writes         : local only, <output_dir>/pr-generate/pr_reviewers.txt
#                    (the draft comment, human-editable) and pr_reviewers.json
#                    (the evidence and the command step 2c will run).
#
# Usage: pr_flow_step_2b_suggest_reviewers.sh --config <cfg> [--max-reviewers 5]

source "$(dirname "$0")/lib_pr_flow.sh"

MAX_REVIEWERS="5"
while [ $# -gt 0 ]; do
    case "$1" in
        --config)         PF_CONFIG="$2"; shift 2 ;;
        --max-reviewers)  MAX_REVIEWERS="$2"; shift 2 ;;
        *) pf_die "unknown argument: $1" ;;
    esac
done

pf_resolve_state
pf_check gh-auth

PR_NUMBER="$(pf_get pr_number)"
PR_URL="$(pf_get pr_url)"
[ -n "$PR_NUMBER" ] || pf_die "no pr_number in state -- run pr_flow_step_2_push.sh first"

pf_banner "STEP 2B -- SUGGEST REVIEWERS (draft only, nothing posted)"
pf_note "PR          : $PR_URL"
pf_note "max names   : $MAX_REVIEWERS"

echo
echo "Runs read-only detection (git blame/history in the worktree, 'gh api'"
echo "searches for past reviewers and CODEOWNERS), then writes the draft."
echo "The ONLY network traffic is read-only 'gh api' queries."
echo
echo "\$ $PF_PYTHON -m llm.auto_code_gen.pr_flow.run_reviewer_suggest generate \\"
echo "      --state $PF_STATE --max-reviewers $MAX_REVIEWERS"

pf_run "$PF_PYTHON" -m llm.auto_code_gen.pr_flow.run_reviewer_suggest generate \
    --state "$PF_STATE" --max-reviewers "$MAX_REVIEWERS" \
    || pf_die "reviewer suggestion failed"

DRAFT="$(pf_dir descs_dir "$(pf_output_dir)")/pr_reviewers.txt"
pf_banner "DRAFT WRITTEN (not posted)"
cat "$DRAFT"
pf_banner "NEXT"
echo "  1. Review (and edit if needed): $DRAFT"
echo "  2. Post it:"
echo "     bash $(cd "$(dirname "$0")" && pwd)/pr_flow_step_2c_post_reviewers.sh --config $PF_CONFIG"
