#!/bin/bash
# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

# Step R1 -- move the PR branch onto a newer upstream base. Human-run, local git.
#
#   Network access : only with --fetch (git fetch over SSH, your passphrase)
#   Writes         : the rebase itself + pr_state.json. A safety ref is created
#                    first, so an abort or a later --abort always works.
#   It never resolves conflicts: that is the AI's job in a paused worktree.
#
# Modes:
#   (default)      start the rebase
#   --continue     stage the resolutions and continue after a conflict stop
#   --abort        abandon the rebase and return to the pre-rebase head
#   --status       show where the rebase stands
#
# Usage: pr_flow_step_r1_rebase.sh --config <cfg> [--fetch] [--target upstream/main]
#        [--continue] [--abort] [--status] [--dry-run]

source "$(dirname "$0")/lib_pr_flow.sh"

DO_FETCH="" CONTINUE="" ABORT="" STATUS="" TARGET=""
while [ $# -gt 0 ]; do
    case "$1" in
        --config)    PF_CONFIG="$2"; shift 2 ;;
        --fetch)     DO_FETCH=1; shift ;;
        --target)    TARGET="$2"; shift 2 ;;
        --continue)  CONTINUE=1; shift ;;
        --abort)     ABORT=1; shift ;;
        --status)    STATUS=1; shift ;;
        --dry-run)   DRY_RUN=1; shift ;;
        *) pf_die "unknown argument: $1" ;;
    esac
done

pf_resolve_state
REPO="$(pf_get repo_dir)"
OUTPUT_DIR="$(pf_get output_dir)"
LOCAL_BRANCH="$(pf_get local_branch)"
BASE_COMMIT="$(pf_get base_commit)"
UPSTREAM_REMOTE="$(pf_get upstream_remote)"
UPSTREAM_BRANCH="$(pf_get upstream_branch)"
[ -n "$LOCAL_BRANCH" ] || LOCAL_BRANCH="$(pf_git rev-parse --abbrev-ref HEAD)"
[ -n "$TARGET" ] && pf_set rebase_target "$TARGET"

if [ -n "$STATUS" ]; then
    echo "repo: $REPO"
    "$PF_PYTHON" -m llm.auto_code_gen.pr_flow.rebase status --state "$PF_STATE"
    "$PF_PYTHON" -m llm.auto_code_gen.pr_flow.rebase plan --state "$PF_STATE"
    exit 0
fi

if [ -n "$ABORT" ]; then
    if ! pf_rebase_paused "$REPO"; then
        echo "No rebase is paused in $REPO -- nothing to abandon."
        "$PF_PYTHON" -m llm.auto_code_gen.pr_flow.rebase status --state "$PF_STATE" | sed 's/^/  /'
        echo "  (state is untouched; if a previous rebase was already abandoned, the"
        echo "   safety tag pr-flow/pre-rebase-* still holds the old head)"
        exit 0
    fi
    pf_banner "STEP R1 -- ABORT THE REBASE"
    pf_note "repo: $REPO"
    "$PF_PYTHON" -m llm.auto_code_gen.pr_flow.rebase status --state "$PF_STATE" | sed 's/^/  /'
    pf_banner "WILL RUN"
    echo "  git -C $REPO rebase --abort"
    pf_confirm
    pf_run pf_git rebase --abort || exit $?
    PRE="$(pf_get pre_rebase_head)"
    if [ -n "$PRE" ]; then
        pf_run pf_git reset --hard "$PRE" || exit $?
        pf_set head_sha "$PRE"
    fi
    echo
    echo "Rebase abandoned; back at $(pf_head_sha | cut -c1-12)."
    echo "NEXT: bash llm/auto_code_gen/pr_flow/scripts/pr_flow_step_r1_rebase.sh --config $PF_CONFIG --status"
    exit 0
fi

if [ -n "$CONTINUE" ]; then
    pf_check rebase-resume --state "$PF_STATE"
    # The newest stage the resolver worked on -- resolved via the module so the
    # layout is spelled in exactly one place.
    STAGE_DIR="$("$PF_PYTHON" -c '
import sys
from llm.auto_code_gen.pr_flow import rebase, run_rebase_conflict_fix as fixer
stages = fixer.existing_stages(sys.argv[1])
print(rebase.rebase_dir(sys.argv[1], stages[-1] if stages else 1))
' "$OUTPUT_DIR")"
    RESOLUTION="$STAGE_DIR/rebase_resolution.md"
    if [ -f "$RESOLUTION" ]; then
        pf_check rebase-resolved --state "$PF_STATE" --resolution "$RESOLUTION"
    else
        echo "  (no rebase_resolution.md -- assuming you resolved the conflicts by hand;"
        echo "   no report means no coverage check -- be sure you edited them all)"
        "$PF_PYTHON" - "$REPO" <<'PY' || exit 30
import sys
from llm.auto_code_gen.pr_flow import rebase
repo = sys.argv[1]
markers = rebase.scan_markers(repo)
for f in markers:
    print("CONFLICT MARKERS LEFT: " + f)
unmerged = rebase.unmerged_files(repo)
if unmerged:
    print("  unmerged index entries (staged by 'git add' below): {}".format(
        ", ".join(unmerged)))
sys.exit(30 if markers else 0)
PY
    fi

    CONFLICTED="$(pf_git diff --name-only --diff-filter=U)"
    MODIFIED="$(pf_git diff --name-only)"
    UNTRACKED="$(pf_git ls-files --others --exclude-standard)"
    FILES="$(printf '%s\n%s\n%s\n' "$CONFLICTED" "$MODIFIED" "$UNTRACKED" | sed '/^$/d' | sort -u)"

    pf_banner "STEP R1 -- CONTINUE THE REBASE"
    pf_note "repo      : $REPO"
    pf_note "stage dir : $STAGE_DIR"
    echo
    if [ -n "$FILES" ]; then
        echo "-- files that will be staged --"
        printf '%s\n' "$FILES" | sed 's/^/    /'
    else
        echo "-- nothing left to stage -- the resolutions are staged already, so this"
        echo "   run only replays the remaining commits (a safe re-run after an abort"
        echo "   of the previous attempt)"
    fi
    pf_banner "WILL RUN"
    [ -n "$FILES" ] && printf '%s\n' "$FILES" | sed "s|^|  git -C $REPO add |"
    echo "  git -C $REPO -c core.editor=true rebase --continue   (no editor opens)"
    echo "  (repeats while more commits in this PR still conflict)"
    pf_confirm

    if [ -n "$FILES" ]; then
        while IFS= read -r f; do
            [ -n "$f" ] && pf_run pf_git add -- "$f" || exit $?
        done <<< "$FILES"
    fi

    for _ in 1 2 3 4 5 6 7 8 9 10; do
        pf_rebase_paused "$REPO" || break
        "$PF_PYTHON" -m llm.auto_code_gen.pr_flow.rebase status --state "$PF_STATE" | sed 's/^/  /'
        pf_run pf_git -c core.editor=true rebase --continue
        RC=$?
        pf_rebase_paused "$REPO" || break
        "$PF_PYTHON" -m llm.auto_code_gen.pr_flow.rebase snapshot --state "$PF_STATE" >/dev/null
        if [ -n "$(pf_git diff --name-only --diff-filter=U)" ]; then
            echo
            echo "The rebase stopped again with new conflicts."
            echo "NEXT (AI, no git writes):"
            echo "  $PF_PYTHON -m llm.auto_code_gen.pr_flow.run_rebase_conflict_fix --config $PF_CONFIG"
            echo "THEN: bash $0 --config $PF_CONFIG --continue"
            exit 30
        fi
        # Paused with nothing unmerged: git stopped for its own reasons (an
        # empty commit, a cherry-pick it cannot apply). Never guess -- hand it
        # back with the facts and leave the choice of --skip/--abort to a human.
        echo
        echo "The rebase is still paused but nothing is in conflict"
        echo "(git exited $RC). Inspect before deciding:"
        pf_git status --short | sed 's/^/    /'
        echo "    git -C $REPO rebase --skip     # drop the commit git is stuck on"
        echo "    git -C $REPO rebase --abort    # or abandon and retry later"
        echo "    bash $0 --config $PF_CONFIG --abort"
        exit 30
    done
    _r1_finished
    exit 0
fi

# ---- default mode: start the rebase ------------------------------------------
pf_check remotes --state "$PF_STATE"
pf_check rebase-plan --state "$PF_STATE"
"$PF_PYTHON" -m llm.auto_code_gen.pr_flow.rebase plan --state "$PF_STATE" | sed 's/^/  /'
PLAN_JSON="$("$PF_PYTHON" -c "
import json, sys
from llm.auto_code_gen.pr_flow import rebase, pr_state
s = pr_state.load_state(sys.argv[1])
print(json.dumps(rebase.plan(s)))
" "$PF_STATE")"
TARGET_REF="$(printf '%s' "$PLAN_JSON" | "$PF_PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["target"])')"
TARGET_SHA="$(printf '%s' "$PLAN_JSON" | "$PF_PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["target_sha"])')"
OVERLAP="$(printf '%s' "$PLAN_JSON" | "$PF_PYTHON" -c 'import json,sys; print(" ".join(json.load(sys.stdin)["overlap_files"]))')"

pf_banner "STEP R1 -- START THE REBASE"
pf_note "repo        : $REPO"
pf_note "branch      : $LOCAL_BRANCH"
pf_note "current base: ${BASE_COMMIT:0:12}"
pf_note "new base    : $TARGET_REF @ ${TARGET_SHA:0:12}"
if [ -n "$OVERLAP" ]; then
    pf_note "conflict risk   : files also changed upstream -> $OVERLAP"
else
    pf_note "conflict risk   : none expected (no file overlap)"
fi

pf_banner "WILL RUN (in this order)"
[ -n "$DO_FETCH" ] && echo "  git -C $REPO fetch $UPSTREAM_REMOTE $UPSTREAM_BRANCH   (SSH passphrase)"
echo "  git -C $REPO tag pr-flow/pre-rebase-\$(git -C $REPO rev-parse --short HEAD)   # safety anchor"
echo "  git -C $REPO rebase --onto $TARGET_REF ${BASE_COMMIT:0:12} $LOCAL_BRANCH"
echo
pf_note "If commits conflict, the rebase stays PAUSED and this script hands off to"
pf_note "the AI conflict resolver; nothing is lost and --abort always works."

pf_confirm

if [ -n "$DO_FETCH" ]; then
    pf_run pf_git fetch "$UPSTREAM_REMOTE" "$UPSTREAM_BRANCH" || exit $?
    TARGET_SHA="$(pf_git rev-parse "$TARGET_REF")"
fi

SHORT="$(pf_git rev-parse --short HEAD)"
pf_run pf_git tag "pr-flow/pre-rebase-$SHORT" || echo "  (safety tag already present)"
pf_set pre_rebase_head "$(pf_head_sha)"
# NOTE: remote_head_sha belongs to step 2/4 only -- it is what the fork holds.
# Stamping it here would make a rebase look like an unpushed local change.

pf_run pf_git rebase --onto "$TARGET_REF" "$BASE_COMMIT" "$LOCAL_BRANCH" || true

if "$PF_PYTHON" -c "
import sys
from llm.auto_code_gen.pr_flow import rebase
sys.exit(0 if rebase.in_progress('$REPO') else 1)
"; then
    "$PF_PYTHON" -m llm.auto_code_gen.pr_flow.rebase snapshot --state "$PF_STATE"
    echo
    echo "Rebase PAUSED on conflicts."
    echo "NEXT (AI, no git writes; it edits the paused worktree):"
    echo "  $PF_PYTHON -m llm.auto_code_gen.pr_flow.run_rebase_conflict_fix --config $PF_CONFIG"
    echo "THEN (you): bash $0 --config $PF_CONFIG --continue"
    exit 30
fi

_r1_finished
exit 0
