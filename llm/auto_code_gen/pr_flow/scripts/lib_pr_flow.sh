#!/bin/bash
# Shared helpers for the human-run PR flow scripts.
#
# Contract every step script follows:
#   1. announce      -- print exactly what will happen, including the commands
#   2. guardrails    -- read-only local checks (llm.auto_code_gen.pr_flow.checks)
#   3. confirm       -- type 'yes'; anything else aborts
#   4. execute       -- run the commands visibly, so SSH/password prompts reach you
#   5. record        -- write the outcome to pr_state.json and logs/
#
# Nothing in this library talks to GitHub: only the step scripts do, and only
# after a confirmation.

set -o pipefail

PF_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PF_ROOT="$(cd "${PF_LIB_DIR}/../../../.." && pwd)"
PF_PYTHON="${PF_PYTHON:-python}"
command -v "$PF_PYTHON" >/dev/null 2>&1 || PF_PYTHON=python3

PF_CONFIG=""
PF_STATE=""
DRY_RUN=0

pf_die() {
    echo "ABORT: $*" >&2
    exit 21
}

pf_note() { echo "  $*"; }

pf_banner() {
    echo
    echo "=============================================================================="
    echo "$*"
    echo "=============================================================================="
}

pf_require_arg() {
    [ -n "$2" ] || pf_die "missing --$1"
}

pf_resolve_state() {
    pf_require_arg config "$PF_CONFIG"
    [ -f "$PF_CONFIG" ] || pf_die "config not found: $PF_CONFIG"
    PF_STATE="$(cd "$PF_ROOT" && "$PF_PYTHON" -m llm.auto_code_gen.pr_flow.pr_state path \
        --config "$PF_CONFIG")" || pf_die "cannot resolve pr_state path from $PF_CONFIG"
    [ -f "$PF_STATE" ] || pf_die "no state at $PF_STATE -- run pr_flow_step_0_prepare.sh first"
}

pf_get() {
    (cd "$PF_ROOT" && "$PF_PYTHON" -m llm.auto_code_gen.pr_flow.pr_state get \
        --state "$PF_STATE" --field "$1")
}

pf_set() {
    (cd "$PF_ROOT" && "$PF_PYTHON" -m llm.auto_code_gen.pr_flow.pr_state set \
        --state "$PF_STATE" --set "$1=$2") >/dev/null
    echo "  state: $1 = $2"
}

# Canonical paths come from pr_state.py so no script ever re-spells them.
pf_dir() {
    (cd "$PF_ROOT" && "$PF_PYTHON" -c '
import sys
from llm.auto_code_gen.pr_flow import pr_state
print(getattr(pr_state, sys.argv[1])(*sys.argv[2:]))
' "$@")
}

pf_output_dir()    { pf_get output_dir; }
pf_descs_dir()     { pf_dir descs_dir "$(pf_get output_dir)"; }
pf_comments_dir()  { pf_dir comments_dir "$(pf_get output_dir)"; }
pf_inbox_dir()     { pf_dir inbox_dir "$(pf_get output_dir)" "$1"; }

pf_check() {
    echo
    echo "-- guardrail: checks $*"
    (cd "$PF_ROOT" && "$PF_PYTHON" -m llm.auto_code_gen.pr_flow.checks "$@") || {
        local rc=$?
        echo
        echo "Refusing to continue (guardrail exit $rc)." >&2
        exit $rc
    }
}

pf_confirm() {
    if [ "$DRY_RUN" = "1" ]; then
        echo
        echo "[dry-run] nothing executed."
        exit 0
    fi
    echo
    printf "Type 'yes' to run the commands above, anything else to abort: "
    local answer
    read -r answer
    if [ "$answer" != "yes" ]; then
        echo "Aborted by operator. Nothing was changed."
        exit 130
    fi
}

pf_run() {
    echo
    echo "\$ $*"
    local log
    log="$(pf_logfile)"
    echo "$(date '+%F %T') RUN $*" >> "$log"
    "$@"
    local rc=$?
    echo "$(date '+%F %T') EXIT=$rc $*" >> "$log"
    return $rc
}

# Like pf_run, but the command's stdout goes to a file. The trace line must not
# go through that stdout or it would be captured into the JSON artifact.
pf_run_to() {
    local out="$1"
    shift
    echo
    echo "\$ $* > $out"
    local log
    log="$(pf_logfile)"
    echo "$(date '+%F %T') RUN $* > $out" >> "$log"
    "$@" > "$out"
    local rc=$?
    echo "$(date '+%F %T') EXIT=$rc $*" >> "$log"
    return $rc
}

pf_logfile() {
    local dir
    dir="$(pf_get logs_dir)"
    [ -n "$dir" ] || dir="/tmp"
    mkdir -p "$dir"
    echo "$dir/$(basename "$0" .sh)_$(date +%Y%m%d_%H%M%S).log"
}

pf_head_sha() {
    pf_git rev-parse HEAD
}

pf_gh_repo() { pf_get upstream_repo; }

pf_git() {
    local repo
    repo="$(pf_get repo_dir)"
    git -C "$repo" "$@"
}

pf_show_pending() {
    local inbox
    inbox="$(pf_inbox_dir "$1")"
    echo "  inbox: $inbox"
    if [ -f "$inbox/NEW_COMMENTS.md" ]; then
        echo "  pending comment ids:"
        grep -E '^## Comment ' "$inbox/NEW_COMMENTS.md" | sed 's/^/    /'
    fi
}

# True while git is paused mid-rebase. The repo path is passed as argv, never
# interpolated into the Python source, so odd paths cannot break the quote.
pf_rebase_paused() {
    "$PF_PYTHON" -c '
import sys
from llm.auto_code_gen.pr_flow import rebase
sys.exit(0 if rebase.in_progress(sys.argv[1]) else 1)
' "$1"
}

# Shared by the rebase script once the branch is on the new base.
_r1_finished() {
    local sha base target
    sha="$(pf_head_sha)"
    target="$(pf_get rebase_target)"
    base="$(pf_git rev-parse "$target" 2>/dev/null || true)"
    pf_set head_sha "$sha"
    [ -n "$base" ] && pf_set base_commit "$base"
    echo
    echo "Rebase complete: HEAD ${sha:0:12} now sits on ${target:-base} ${base:0:12}"
    echo "The fork still holds $(pf_get remote_head_sha | cut -c1-12); publishing this"
    echo "needs a leased force push, which step 4 does with --force-with-lease."
    echo
    echo "The shared OFF baselines were measured on the OLD base, so the next stage"
    echo "re-measures them (the expensive part, ~32 min) and re-proves the win -- and"
    echo "it may also repair what the rebase broke semantically, like any normal"
    echo "runtime iteration: fix first, then measure, until the gates are green."
    echo "NEXT (AI, no network):"
    echo "  $PF_PYTHON -m llm.auto_code_gen.pr_flow.run_rebase_runtime_iters --config $PF_CONFIG"
}
