#!/bin/bash
# Step 2 -- push the PR branch to the fork and open the PR upstream.
#
#   Network access : YES -- git push over SSH (your passphrase, typed by you)
#                    and one 'gh pr create' over HTTPS.
#   Safety         : the branch name must be non-protected, prefix-conforming and
#                    unused on the fork; the push target must be a fork, never
#                    upstream, never a local path.
#
# Usage: pr_flow_step_2_push.sh --config <cfg> --remote-branch perf/<short-name>
#        [--unique-suffix] [--skip-create-pr] [--dry-run]

source "$(dirname "$0")/lib_pr_flow.sh"

REMOTE_BRANCH="" UNIQUE_SUFFIX="" SKIP_PR=""
while [ $# -gt 0 ]; do
    case "$1" in
        --config)           PF_CONFIG="$2"; shift 2 ;;
        --remote-branch)    REMOTE_BRANCH="$2"; shift 2 ;;
        --unique-suffix)    UNIQUE_SUFFIX=1; shift ;;
        --skip-create-pr)   SKIP_PR=1; shift ;;
        --dry-run)          DRY_RUN=1; shift ;;
        *) pf_die "unknown argument: $1" ;;
    esac
done

pf_resolve_state
pf_require_arg remote-branch "$REMOTE_BRANCH"

REPO="$(pf_get repo_dir)"
OUTPUT_DIR="$(pf_get output_dir)"
LOCAL_BRANCH="$(pf_get local_branch)"
FORK_REMOTE="$(pf_get fork_remote)"
UPSTREAM="$(pf_get upstream_repo)"
UPSTREAM_BRANCH="$(pf_get upstream_branch)"
FORK_OWNER="$(pf_get fork_owner)"
DESCS_DIR="$(pf_descs_dir)"
[ -n "$LOCAL_BRANCH" ] || LOCAL_BRANCH="$(pf_git rev-parse --abbrev-ref HEAD)"

pf_check remotes --state "$PF_STATE"
pf_check rebase-absent --state "$PF_STATE"
pf_check branch-name --state "$PF_STATE" --name "$REMOTE_BRANCH"
pf_check clean-tree --state "$PF_STATE"
pf_check descs --state "$PF_STATE"
pf_check dco --state "$PF_STATE"

pf_banner "STEP 2 -- PUSH + CREATE PR"
pf_note "push target : ${FORK_REMOTE} (${FORK_OWNER}) -- NEVER ${UPSTREAM}"
pf_note "refspec     : ${LOCAL_BRANCH}:${REMOTE_BRANCH}"
pf_note "PR          : ${UPSTREAM}:${UPSTREAM_BRANCH} <- ${FORK_OWNER}:${REMOTE_BRANCH}"
pf_note "commit      : $(pf_head_sha | cut -c1-12)  $(pf_git log -1 --format=%s)"
pf_note "auth        : SSH passphrase will be requested by git; gh uses its own token"

pf_banner "WILL RUN (in this order)"
echo "  git -C $REPO push -u $FORK_REMOTE $LOCAL_BRANCH:$REMOTE_BRANCH"
[ -z "$SKIP_PR" ] && echo "  gh pr create --repo $UPSTREAM --base $UPSTREAM_BRANCH \\
      --head $FORK_OWNER:$REMOTE_BRANCH --title \"\$(head -1 $DESCS_DIR/commit_desc.txt)\" \\
      --body-file $DESCS_DIR/pr_desc.txt"
echo
echo "First, two read-only uniqueness probes on the fork/PR list."

pf_confirm

pf_note "probing fork for an existing '${REMOTE_BRANCH}' branch..."
EXISTING="$(pf_git ls-remote --heads "$FORK_REMOTE" "$REMOTE_BRANCH" 2>/dev/null)"
if [ -n "$EXISTING" ]; then
    if [ -n "$UNIQUE_SUFFIX" ]; then
        REMOTE_BRANCH="${REMOTE_BRANCH}-$(date +%Y%m%dT%H%M)"
        pf_note "taken -- retrying as ${REMOTE_BRANCH}"
        pf_check branch-name --state "$PF_STATE" --name "$REMOTE_BRANCH"
        EXISTING="$(pf_git ls-remote --heads "$FORK_REMOTE" "$REMOTE_BRANCH" 2>/dev/null)"
        [ -n "$EXISTING" ] && pf_die "even ${REMOTE_BRANCH} exists on the fork"
    else
        echo "Branch '${REMOTE_BRANCH}' already exists on ${FORK_REMOTE}." >&2
        echo "Choose another --remote-branch, or re-run with --unique-suffix." >&2
        exit 25
    fi
fi

if [ -z "$SKIP_PR" ] && (cd "$PF_ROOT" && "$PF_PYTHON" -m llm.auto_code_gen.pr_flow.checks gh-auth) >/dev/null 2>&1; then
    OPEN_PR="$(gh pr list --repo "$UPSTREAM" --head "${FORK_OWNER}:${REMOTE_BRANCH}" \
        --state all --json number --jq '.[].number' 2>/dev/null)"
    if [ -n "$OPEN_PR" ]; then
        echo "PR #$OPEN_PR already uses ${FORK_OWNER}:${REMOTE_BRANCH}." >&2
        echo "Use --unique-suffix, or skip to step 4 to update that PR." >&2
        exit 25
    fi
fi

pf_run pf_git push -u "$FORK_REMOTE" "${LOCAL_BRANCH}:${REMOTE_BRANCH}" || exit $?
pf_set remote_branch "$REMOTE_BRANCH"
pf_set remote_head_sha "$(pf_head_sha)"
pf_set head_sha "$(pf_head_sha)"

if [ -n "$SKIP_PR" ]; then
    echo
    echo "Pushed ${FORK_OWNER}:${REMOTE_BRANCH}; PR creation skipped (--skip-create-pr)."
    exit 0
fi

TITLE="$(head -1 "$DESCS_DIR/commit_desc.txt")"
OUT="$(mktemp)"
echo
echo "\$ gh pr create --repo $UPSTREAM --base $UPSTREAM_BRANCH --head ${FORK_OWNER}:${REMOTE_BRANCH}"
if ! gh pr create --repo "$UPSTREAM" --base "$UPSTREAM_BRANCH" \
        --head "${FORK_OWNER}:${REMOTE_BRANCH}" --title "$TITLE" \
        --body-file "$DESCS_DIR/pr_desc.txt" | tee "$OUT"; then
    echo
    echo "Push succeeded but the PR was not created. Branch: ${FORK_OWNER}:${REMOTE_BRANCH}" >&2
    echo "Create it manually or re-run this script with --skip-create-pr already covered." >&2
    exit 22
fi

PR_URL="$(grep -Eo 'https://github\.com/[^[:space:]]+/pull/[0-9]+' "$OUT" | tail -1)"
rm -f "$OUT"
[ -n "$PR_URL" ] || pf_die "could not read the PR URL from gh output -- record it manually"

pf_set pr_url "$PR_URL"
pf_set pr_number "${PR_URL##*/}"
pf_set descs_sha256 "$(sha256sum "$DESCS_DIR/pr_desc.txt" | cut -d' ' -f1)"

echo
echo "PR open: $PR_URL"
echo "NEXT (when reviewers have replied):"
echo "  bash llm/auto_code_gen/pr_flow/scripts/pr_flow_step_3_fetch_reviews.sh --config $PF_CONFIG"
