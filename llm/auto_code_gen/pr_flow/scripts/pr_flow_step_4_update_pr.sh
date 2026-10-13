#!/bin/bash
# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

# Step 4 -- push the review-fix commits and update the PR text.
#
#   Network access : YES -- 'git push' over SSH (your passphrase) plus gh calls
#                    that edit the PR body / post comments.
#   Precondition   : the round reached a green verdict AND the fixes are already
#                    committed (step 1 does that). A dirty tree or a FAIL verdict
#                    stops this script; nothing is ever pushed unverified.
#   Rebased branch : if the local head is no longer a descendant of what the fork
#                    holds, only --force-with-lease (leased to the published sha)
#                    is accepted.
#
# Usage: pr_flow_step_4_update_pr.sh --config <cfg> [--force-with-lease]
#        [--reply-threads] [--skip-comment] [--dry-run]

source "$(dirname "$0")/lib_pr_flow.sh"

FORCE_LEASE="" REPLY_THREADS="" SKIP_COMMENT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --config)           PF_CONFIG="$2"; shift 2 ;;
        --force-with-lease) FORCE_LEASE=1; shift ;;
        --reply-threads)    REPLY_THREADS=1; shift ;;
        --skip-comment)     SKIP_COMMENT=1; shift ;;
        --dry-run)          DRY_RUN=1; shift ;;
        *) pf_die "unknown argument: $1" ;;
    esac
done

pf_resolve_state

REPO="$(pf_get repo_dir)"
OUTPUT_DIR="$(pf_get output_dir)"
LOCAL_BRANCH="$(pf_get local_branch)"
FORK_REMOTE="$(pf_get fork_remote)"
REMOTE_BRANCH="$(pf_get remote_branch)"
UPSTREAM="$(pf_get upstream_repo)"
PR_NUMBER="$(pf_get pr_number)"
BASE_COMMIT="$(pf_get base_commit)"
OLD_SHA="$(pf_get head_sha)"
# What the fork actually holds: after a rebase this is not an ancestor of HEAD.
PUBLISHED_SHA="$(pf_get remote_head_sha)"
[ -n "$PUBLISHED_SHA" ] || PUBLISHED_SHA="$OLD_SHA"
DESCS_DIR="$(pf_descs_dir)"
COMMENT_DIR="$(pf_comments_dir)"
DESCS_SHA="$(sha256sum "$DESCS_DIR/pr_desc.txt" 2>/dev/null | cut -d' ' -f1)"
STORED_DESCS_SHA="$(pf_get descs_sha256)"
ROUND="$(pf_get round)"
ITERATION="$(pf_get iteration)"
[ -n "$REMOTE_BRANCH" ] || pf_die "no remote_branch in state -- run pr_flow_step_2_push.sh first"
[ -n "$PR_NUMBER" ] || pf_die "no pr_number in state -- run pr_flow_step_2_push.sh first"
[ -n "$OLD_SHA" ] || pf_die "no head_sha in state"

pf_check rebase-absent --state "$PF_STATE"
pf_check verdict --state "$PF_STATE"
echo
echo "-- guardrail: checks dco"
(cd "$PF_ROOT" && "$PF_PYTHON" -m llm.auto_code_gen.pr_flow.checks dco --state "$PF_STATE")
rc=$?
if [ "$rc" -ne 0 ]; then
    echo
    echo "Refusing to continue (guardrail exit $rc)." >&2
    echo "The upstream DCO bot would reject this PR: fix the signoffs BEFORE pushing." >&2
    exit "$rc"
fi
echo
echo "-- guardrail: checks clean-tree --expect-ahead${FORCE_LEASE:+ --allow-rewritten}"
(cd "$PF_ROOT" && "$PF_PYTHON" -m llm.auto_code_gen.pr_flow.checks clean-tree \
    --state "$PF_STATE" --expect-ahead ${FORCE_LEASE:+--allow-rewritten})
rc=$?
if [ "$rc" -ne 0 ]; then
    echo
    echo "Refusing to continue (guardrail exit $rc)." >&2
    echo "Before updating the PR:" >&2
    echo "  * uncommitted fixes -> commit them:  bash pr_flow_step_1_commit.sh --config $PF_CONFIG \\" >&2
    echo "        --message-file $OUTPUT_DIR/runtime/V${ITERATION}/followup_commit_msg.txt" >&2
    echo "  * HEAD not a descendant of the published head -> the branch was rebased:" >&2
    echo "        re-run this with --force-with-lease (it leases $OLD_SHA)" >&2
    exit "$rc"
fi
pf_check descs --state "$PF_STATE"

# git push needs SSH only; gh is needed solely for PR text. A commit-only push
# therefore never asks for an API token.
NEEDS_GH=""
[ "$DESCS_SHA" != "$STORED_DESCS_SHA" ] && NEEDS_GH="the PR description changed"
[ -z "$SKIP_COMMENT" ] && NEEDS_GH="${NEEDS_GH:-a round comment will be posted}"
[ -n "$REPLY_THREADS" ] && NEEDS_GH="${NEEDS_GH:-thread replies will be posted}"
if [ -n "$NEEDS_GH" ]; then
    pf_check gh-auth
else
    echo "  gh not required (commit-only push: --skip-comment, description unchanged)"
fi

VERDICT_FILE="$(cd "$PF_ROOT" && "$PF_PYTHON" -c \
    "from llm.auto_code_gen.pr_flow import gates; print(gates.latest_verdict('$OUTPUT_DIR') or '')")"

pf_banner "STEP 4 -- UPDATE PR #$PR_NUMBER (round $ROUND, iteration V$ITERATION)"
pf_note "PR            : $(pf_get pr_url)"
pf_note "push target   : ${FORK_REMOTE}:${REMOTE_BRANCH} (never ${UPSTREAM})"
pf_note "PR head       : ${PUBLISHED_SHA:0:12} (on fork) -> $(pf_head_sha | cut -c1-12) (local)"
pf_note "description   : $([ "$DESCS_SHA" != "$STORED_DESCS_SHA" ] && echo 'changed -- will update body' || echo 'unchanged')"
pf_note "thread replies: $([ -n "$REPLY_THREADS" ] && echo 'will post' || echo 'skipped (use --reply-threads)')"

pf_banner "COMMITS ADDED THIS ROUND"
pf_git log --oneline "${PUBLISHED_SHA}..HEAD" 2>/dev/null \
    || pf_git log --oneline -5
echo
pf_banner "DIFF vs CURRENT PR HEAD"
pf_git diff --stat "${PUBLISHED_SHA}..HEAD" 2>/dev/null \
    || pf_git diff --stat "$PUBLISHED_SHA" HEAD
echo
pf_banner "VERIFICATION THAT GATES THIS PUSH"
[ -f "$VERDICT_FILE" ] && cat "$VERDICT_FILE" || echo "  (no verdict snapshot; guardrail above already passed)"
RESOLUTION="$OUTPUT_DIR/runtime/V${ITERATION}/comment_resolution.md"
if [ -f "$RESOLUTION" ]; then
    echo
    echo "-- per-comment disposition --"
    grep -E '^## Comment |^STATUS: ' "$RESOLUTION" | paste - - | sed 's/^/  /'
fi
if [ "$DESCS_SHA" != "$STORED_DESCS_SHA" ] && [ -f "$DESCS_DIR/pr_desc.txt" ]; then
    echo
    echo "-- new PR description (first 40 lines) --"
    head -40 "$DESCS_DIR/pr_desc.txt" | sed 's/^/  | /'
fi

COMMENT_FILE="$COMMENT_DIR/round_${ROUND}_comment.md"
"$PF_PYTHON" -c "import os,sys; os.makedirs(sys.argv[1], exist_ok=True)" "$COMMENT_DIR"
if [ -z "$SKIP_COMMENT" ]; then
    "$PF_PYTHON" -m llm.auto_code_gen.pr_flow.gh_publish render-comment \
        --state "$PF_STATE" --round "$ROUND" --iteration "$ITERATION" --out "$COMMENT_FILE" \
        >/dev/null 2>&1 && {
        echo
        echo "-- round comment to be posted --"
        sed 's/^/  | /' "$COMMENT_FILE"
    }
fi

REPLIES_FILE="$OUTPUT_DIR/runtime/V${ITERATION}/thread_replies.json"
pf_banner "WILL RUN (in this order)"
PUSH_PREVIEW=""
[ -n "$FORCE_LEASE" ] && PUSH_PREVIEW="--force-with-lease=${REMOTE_BRANCH}:${PUBLISHED_SHA} "
echo "  git -C $REPO push ${PUSH_PREVIEW}$FORK_REMOTE $LOCAL_BRANCH:$REMOTE_BRANCH"
[ "$DESCS_SHA" != "$STORED_DESCS_SHA" ] && echo "  gh pr edit $PR_NUMBER --repo $UPSTREAM --body-file $DESCS_DIR/pr_desc.txt"
[ -z "$SKIP_COMMENT" ] && echo "  gh pr comment $PR_NUMBER --repo $UPSTREAM --body-file $COMMENT_FILE"
[ -n "$REPLY_THREADS" ] && { [ -f "$REPLIES_FILE" ] && echo "  gh api --method POST .../pulls/$PR_NUMBER/comments/<id>/replies  (for each reply in $REPLIES_FILE)" || echo "  (--reply-threads: no $REPLIES_FILE -- nothing to post)"; }
echo

pf_confirm

PUSH_ARGS=()
[ -n "$FORCE_LEASE" ] && PUSH_ARGS+=("--force-with-lease=${REMOTE_BRANCH}:${PUBLISHED_SHA}")
pf_run pf_git push "${PUSH_ARGS[@]}" "$FORK_REMOTE" "${LOCAL_BRANCH}:${REMOTE_BRANCH}" || exit $?
pf_set head_sha "$(pf_head_sha)"
pf_set remote_head_sha "$(pf_head_sha)"

if [ "$DESCS_SHA" != "$STORED_DESCS_SHA" ]; then
    pf_run gh pr edit "$PR_NUMBER" --repo "$UPSTREAM" --body-file "$DESCS_DIR/pr_desc.txt" || exit $?
    pf_set descs_sha256 "$DESCS_SHA"
fi

if [ -z "$SKIP_COMMENT" ] && [ -f "$COMMENT_FILE" ]; then
    pf_run "$PF_PYTHON" -m llm.auto_code_gen.pr_flow.gh_publish comment \
        --state "$PF_STATE" --body-file "$COMMENT_FILE" --execute \
        || echo "  (round comment failed -- post it manually)" >&2
fi

if [ -n "$REPLY_THREADS" ] && [ -f "$REPLIES_FILE" ]; then
    pf_run "$PF_PYTHON" -m llm.auto_code_gen.pr_flow.gh_publish replies \
        --replies "$REPLIES_FILE" --state "$PF_STATE" --execute \
        || echo "  (some thread replies failed -- see output above)" >&2
fi

echo
echo "PR updated: $(pf_get pr_url)"
echo
echo "NEXT -- if reviewers replied again, start the next round:"
echo "  bash llm/auto_code_gen/pr_flow/scripts/pr_flow_step_3_fetch_reviews.sh --config $PF_CONFIG"
echo "If the PR is merged, close the loop with:"
echo "  bash llm/auto_code_gen/pr_flow/scripts/pr_flow_step_3_fetch_reviews.sh --config $PF_CONFIG   # expect 'state: merged'"
