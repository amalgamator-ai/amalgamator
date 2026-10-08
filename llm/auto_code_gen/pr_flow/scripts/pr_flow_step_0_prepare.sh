#!/bin/bash
# Step 0 -- create/refresh pr_state.json for this improvement's PR.
#
#   Network access : NONE
#   Writes         : <output_dir>/pr_p3/pr_state.json (+ comments/, logs/ dirs)
#
# Usage:
#   bash llm/auto_code_gen/pr_flow/scripts/pr_flow_step_0_prepare.sh \
#       --config llm/auto_code_gen/configs/code_gen_config_example.json \
#       --repo-dir /home/<you>/code/vllm_deepseek \
#       --fork-url git@github.com:<you>/vllm.git \
#       [--upstream-repo vllm-project/vllm] [--upstream-branch main] \
#       [--local-branch <branch>] [--base-commit <sha>] [--force] [--dry-run]

source "$(dirname "$0")/lib_pr_flow.sh"

REPO_DIR="" FORK_URL="" FORK_OWNER="" UPSTREAM_REPO="vllm-project/vllm"
UPSTREAM_BRANCH="main" LOCAL_BRANCH="" BASE_COMMIT="" FORCE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --config)            PF_CONFIG="$2"; shift 2 ;;
        --repo-dir)          REPO_DIR="$2"; shift 2 ;;
        --fork-url)          FORK_URL="$2"; shift 2 ;;
        --fork-owner)        FORK_OWNER="$2"; shift 2 ;;
        --upstream-repo)     UPSTREAM_REPO="$2"; shift 2 ;;
        --upstream-branch)   UPSTREAM_BRANCH="$2"; shift 2 ;;
        --local-branch)      LOCAL_BRANCH="$2"; shift 2 ;;
        --base-commit)       BASE_COMMIT="$2"; shift 2 ;;
        --force)             FORCE="--force"; shift ;;
        --dry-run)           DRY_RUN=1; shift ;;
        *) pf_die "unknown argument: $1" ;;
    esac
done

pf_require_arg config "$PF_CONFIG"
pf_require_arg repo-dir "$REPO_DIR"
pf_require_arg fork-url "$FORK_URL"
[ -f "$PF_CONFIG" ] || pf_die "config not found: $PF_CONFIG"
# A worktree's .git is a file (pointer to the main repo's gitdir), so accept
# either form; "git -C rev-parse" below is the authoritative check anyway.
[ -e "$REPO_DIR/.git" ] || pf_die "not a git repository: $REPO_DIR"
git -C "$REPO_DIR" rev-parse --git-dir >/dev/null 2>&1 \
    || pf_die "not a git repository: $REPO_DIR"

pf_banner "STEP 0 -- PREPARE PR STATE"
pf_note "config     : $PF_CONFIG"
pf_note "repo       : $REPO_DIR"
pf_note "fork       : $FORK_URL"
pf_note "upstream   : ${UPSTREAM_REPO}:${UPSTREAM_BRANCH}"
pf_note "network    : none (this step writes one JSON file)"

CASE="$(cd "$PF_ROOT" && "$PF_PYTHON" -c '
import json, sys
from llm.auto_code_gen.code_gen_prompts import RUNTIME_DIR, RUNTIME_SUCCESS_FILE
print(json.load(open(sys.argv[1]))["output_dir"] + "/" + RUNTIME_DIR + "/" + RUNTIME_SUCCESS_FILE)
' "$PF_CONFIG")"
[ -f "$CASE" ] || pf_die "no successful runtime result yet: $CASE"
pf_note "verified   : $CASE"

CUR_BRANCH="$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD)"
case "$CUR_BRANCH" in
    main|master|HEAD) pf_die "refusing to prepare a PR from '$CUR_BRANCH'; check out the work branch" ;;
esac
[ -n "$LOCAL_BRANCH" ] || LOCAL_BRANCH="$CUR_BRANCH"
pf_note "local branch: $LOCAL_BRANCH"
pf_note "HEAD        : $(git -C "$REPO_DIR" rev-parse --short HEAD) $(git -C "$REPO_DIR" log -1 --format=%s | cut -c1-60)"

pf_banner "WILL RUN"
echo "  $PF_PYTHON -m llm.auto_code_gen.pr_flow.pr_state init \\
      --config $PF_CONFIG --repo-dir $REPO_DIR --fork-url $FORK_URL \\
      --upstream-repo $UPSTREAM_REPO --upstream-branch $UPSTREAM_BRANCH \\
      --local-branch $LOCAL_BRANCH${BASE_COMMIT:+ --base-commit $BASE_COMMIT}${FORCE:+ $FORCE}"

pf_confirm

(cd "$PF_ROOT" && "$PF_PYTHON" -m llm.auto_code_gen.pr_flow.pr_state init \
    --config "$PF_CONFIG" --repo-dir "$REPO_DIR" --fork-url "$FORK_URL" \
    ${FORK_OWNER:+--fork-owner "$FORK_OWNER"} \
    --upstream-repo "$UPSTREAM_REPO" --upstream-branch "$UPSTREAM_BRANCH" \
    --local-branch "$LOCAL_BRANCH" ${BASE_COMMIT:+--base-commit "$BASE_COMMIT"} $FORCE) \
    || pf_die "pr_state init failed"

pf_resolve_state
DESCS_DIR="$(pf_descs_dir)"
pf_note "state written: $PF_STATE"

if [ ! -s "$DESCS_DIR/commit_desc.txt" ] || [ ! -s "$DESCS_DIR/pr_desc.txt" ]; then
    echo
    echo "Descriptions are missing. Ask the AI to write them (no network, no git):"
    echo "  cd $PF_ROOT && $PF_PYTHON -m llm.auto_code_gen.run_generate_pr_commands \\
      --config $PF_CONFIG --descs-only"
fi

echo
echo "NEXT: review $DESCS_DIR/{commit_desc.txt,pr_desc.txt}, then"
echo "  bash llm/auto_code_gen/pr_flow/scripts/pr_flow_step_1_commit.sh --config $PF_CONFIG"
