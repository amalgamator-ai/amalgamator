#!/bin/bash
# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

# Step 1 -- commit the verified worktree as the PR commit (local git only).
#
#   Network access : NONE
#   Writes         : one commit in the repo working tree + pr_state.json
#
# Run it again after every successful review round, passing the message the fix
# step produced:
#   bash pr_flow_step_1_commit.sh --config <cfg> \
#       --message-file <output_dir>/runtime/V<n>/followup_commit_msg.txt
#
# Usage: pr_flow_step_1_commit.sh --config <cfg> [--message-file PATH]
#        [--patch PATH] [--allow PATH]... [--squash-into-base] [--dry-run]

source "$(dirname "$0")/lib_pr_flow.sh"

MESSAGE_FILE="" PATCH_REF="" ALLOW_ARGS=() SQUASH=""
while [ $# -gt 0 ]; do
    case "$1" in
        --config)        PF_CONFIG="$2"; shift 2 ;;
        --message-file)  MESSAGE_FILE="$2"; shift 2 ;;
        --patch)         PATCH_REF="$2"; shift 2 ;;
        --allow)         ALLOW_ARGS+=(--allow "$2"); shift 2 ;;
        --squash-into-base) SQUASH=1; shift ;;
        --dry-run)       DRY_RUN=1; shift ;;
        *) pf_die "unknown argument: $1" ;;
    esac
done

pf_resolve_state
REPO="$(pf_get repo_dir)"
OUTPUT_DIR="$(pf_get output_dir)"
BRANCH="$(pf_git rev-parse --abbrev-ref HEAD)"
case "$BRANCH" in
    main|master|HEAD)
        if pf_rebase_paused "$REPO"; then
            pf_die "a rebase is paused in $REPO -- finish pr_flow_step_r1_rebase.sh" \
                   "--continue or --abort before committing"
        fi
        pf_die "refusing to commit on '$BRANCH'" ;;
esac

DESCS_DIR="$(pf_descs_dir)"
[ -n "$MESSAGE_FILE" ] || MESSAGE_FILE="$DESCS_DIR/commit_desc.txt"
[ -s "$MESSAGE_FILE" ] || pf_die "commit message file missing or empty: $MESSAGE_FILE"

# `diff HEAD` sees staged and unstaged edits alike -- a fixer hook or an
# earlier aborted run of this script may have left changes staged.
CHANGED="$(pf_git diff --name-only HEAD)"
UNTRACKED="$(pf_git ls-files --others --exclude-standard)"
[ -n "$CHANGED$UNTRACKED" ] || pf_die "nothing to commit -- the worktree is clean"

pf_banner "STEP 1 -- COMMIT (local git only, no network)"
pf_note "repo         : $REPO"
pf_note "branch       : $BRANCH"
pf_note "message file : $MESSAGE_FILE"
echo
echo "-- files that will be committed (explicit 'git add', never -A) --"
printf '%s\n' "$CHANGED" "$UNTRACKED" | sed '/^$/d' | sort -u | sed 's/^/    /'
echo
echo "-- git diff --stat --"
pf_git diff --stat | sed 's/^/    /'
echo
echo "-- message (first 15 lines) --"
head -15 "$MESSAGE_FILE" | sed 's/^/    /'

pf_check rebase-absent --state "$PF_STATE"
pf_check tree-scope --state "$PF_STATE" ${PATCH_REF:+--patch "$PATCH_REF"} "${ALLOW_ARGS[@]}"
if [ "$MESSAGE_FILE" = "$DESCS_DIR/commit_desc.txt" ]; then
    pf_check descs --state "$PF_STATE"
fi
# The upstream repo's own hooks (the same suite its CI runs on the PR).
# Fixer hooks (ruff --fix, ruff-format, ...) may rewrite files in place, so
# re-verify scope on the possibly-fixed tree before committing. pf_check
# aborts this script itself when a hook fails.
pf_check pre-commit --state "$PF_STATE"
pf_check tree-scope --state "$PF_STATE" ${PATCH_REF:+--patch "$PATCH_REF"} "${ALLOW_ARGS[@]}"

# Files may have been rewritten by fixer hooks above; collect the final set
# (changes + untracked) the commit will actually contain. `diff HEAD` covers
# staged and unstaged edits alike -- fixer hooks may leave changes staged from
# an earlier aborted run of this script.
CHANGED="$(pf_git diff --name-only HEAD)"
UNTRACKED="$(pf_git ls-files --others --exclude-standard)"
[ -n "$CHANGED$UNTRACKED" ] || pf_die "nothing to commit -- the worktree is clean"

FILES=()
while IFS= read -r line; do
    [ -n "$line" ] && FILES+=("$line")
done < <(printf '%s\n' "$CHANGED" "$UNTRACKED" | sed '/^$/d' | sort -u)

pf_banner "WILL RUN"
echo "  git -C $REPO add ${FILES[*]}"
echo "  git -C $REPO commit -F $MESSAGE_FILE"
echo "  git -C $REPO commit --amend --no-edit -s        # DCO Signed-off-by"
[ -n "$SQUASH" ] && echo "  git -C $REPO reset --soft $(pf_get base_commit) && git -C $REPO commit -F $MESSAGE_FILE  # squash"

pf_confirm

pf_run pf_git add "${FILES[@]}" || exit $?
pf_run pf_git commit -F "$MESSAGE_FILE" || exit $?
pf_run pf_git commit --amend --no-edit -s || exit $?

if [ -n "$SQUASH" ]; then
    BASE="$(pf_get base_commit)"
    pf_require_arg base-commit "$BASE"
    pf_run pf_git reset --soft "$BASE" || exit $?
    pf_run pf_git commit -F "$MESSAGE_FILE" || exit $?
    pf_run pf_git commit --amend --no-edit -s || exit $?
fi

SHA="$(pf_head_sha)"
pf_set head_sha "$SHA"
pf_set local_branch "$BRANCH"
if [ -z "$(pf_get base_commit)" ]; then
    BASE="$(pf_git rev-parse "$(pf_get upstream_remote)/$(pf_get upstream_branch)" 2>/dev/null \
           || pf_git merge-base HEAD "$(pf_get upstream_branch)")"
    [ -n "$BASE" ] && pf_set base_commit "$BASE"
fi

echo
echo "Committed $(echo "$SHA" | cut -c1-12) on $BRANCH"
if [ -n "$(pf_get pr_number)" ]; then
    echo "NEXT: bash llm/auto_code_gen/pr_flow/scripts/pr_flow_step_4_update_pr.sh --config $PF_CONFIG"
else
    echo "NEXT: bash llm/auto_code_gen/pr_flow/scripts/pr_flow_step_2_push.sh --config $PF_CONFIG --remote-branch perf/<short-name>"
fi
