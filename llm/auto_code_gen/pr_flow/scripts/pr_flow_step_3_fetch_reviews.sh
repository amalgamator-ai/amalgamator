#!/bin/bash
# Step 3 -- fetch every review artifact for the PR into the local inbox.
#
#   Network access : YES -- read-only 'gh api' calls (HTTPS + gh token) and,
#                    with --pull, one 'git fetch' over SSH (your passphrase).
#   Writes         : local only, under <output_dir>/pr_p3/inbox/round_N/ (and the
#                    round summary next to it). The git worktree is untouched.
#                    Nothing in the git worktree changes unless you pass --pull.
#
# After this script, the AI runtime iteration runs; it reads the inbox and
# never talks to GitHub itself.
#
# Usage: pr_flow_step_3_fetch_reviews.sh --config <cfg> [--round N] [--pull] [--dry-run]

source "$(dirname "$0")/lib_pr_flow.sh"

ROUND="" DO_PULL=""
while [ $# -gt 0 ]; do
    case "$1" in
        --config)   PF_CONFIG="$2"; shift 2 ;;
        --round)    ROUND="$2"; shift 2 ;;
        --pull)     DO_PULL=1; shift ;;
        --dry-run)  DRY_RUN=1; shift ;;
        *) pf_die "unknown argument: $1" ;;
    esac
done

pf_resolve_state
pf_check gh-auth
pf_check remotes --state "$PF_STATE"

UPSTREAM="$(pf_get upstream_repo)"
PR_NUMBER="$(pf_get pr_number)"
PR_URL="$(pf_get pr_url)"
OUTPUT_DIR="$(pf_get output_dir)"
REPO="$(pf_get repo_dir)"
FORK_REMOTE="$(pf_get fork_remote)"
REMOTE_BRANCH="$(pf_get remote_branch)"
[ -n "$PR_NUMBER" ] || pf_die "no pr_number in state -- run pr_flow_step_2_push.sh first"

OWNER="${UPSTREAM%%/*}"
NAME="${UPSTREAM##*/}"
API="repos/${OWNER}/${NAME}"

CURRENT_ROUND="$(pf_get round)"
CURRENT_ROUND="${CURRENT_ROUND:-0}"
[ -n "$ROUND" ] || ROUND=$((CURRENT_ROUND + 1))
INBOX="$(pf_inbox_dir "$ROUND")"
HEAD_SHA="$(pf_head_sha)"

pf_banner "STEP 3 -- FETCH PR #$PR_NUMBER REVIEWS (round $ROUND)"
pf_note "PR          : $PR_URL"
pf_note "inbox       : $INBOX"
pf_note "previous rnd: $CURRENT_ROUND (comments already answered are not re-fed)"
pf_note "local HEAD  : ${HEAD_SHA:0:12}"

pf_banner "WILL RUN (in this order)"
echo "  mkdir -p $INBOX"
echo "  gh pr view $PR_NUMBER --repo $UPSTREAM --json \\
      number,url,state,baseRefName,headRefOid,mergeable,reviewDecision,isDraft --> pr_view.json"
echo "  gh api --paginate --slurp $API/pulls/$PR_NUMBER/comments        --> comments.json (inline review comments)"
echo "  gh api --paginate --slurp $API/pulls/$PR_NUMBER/reviews         --> reviews.json (approve/request-changes)"
echo "  gh api --paginate --slurp $API/issues/$PR_NUMBER/comments       --> issue_comments.json (general comments)"
echo "  gh api --paginate --slurp --jq '[.[].check_runs]' \\
      $API/commits/$HEAD_SHA/check-runs                                --> checks.json (CI)"
echo "  gh api graphql (reviewThreads isResolved, best effort)         --> threads.json"
[ -n "$DO_PULL" ] && echo "  git -C $REPO fetch $FORK_REMOTE $REMOTE_BRANCH  (SSH passphrase; merge --ff-only only if the tree is clean)"
echo "  python -m llm.auto_code_gen.pr_flow.review_comments summarize --inbox ... --state ... --round $ROUND"
echo
pf_note "All of the above is READ-ONLY against GitHub. No PR is modified."

pf_confirm

mkdir -p "$INBOX"
LOG="$(pf_logfile)"
echo "$(date '+%F %T') round=$ROUND head=$HEAD_SHA" >> "$LOG"

pf_run_to "$INBOX/pr_view.json" gh pr view "$PR_NUMBER" --repo "$UPSTREAM" \
    --json number,url,state,baseRefName,headRefOid,mergeable,reviewDecision,isDraft \
    || pf_die "gh pr view failed"
pf_run_to "$INBOX/comments.json" gh api --paginate --slurp "$API/pulls/$PR_NUMBER/comments" \
    || pf_die "gh api pulls/comments failed"
pf_run_to "$INBOX/reviews.json" gh api --paginate --slurp "$API/pulls/$PR_NUMBER/reviews" \
    || pf_die "gh api pulls/reviews failed"
pf_run_to "$INBOX/issue_comments.json" \
    gh api --paginate --slurp "$API/issues/$PR_NUMBER/comments" \
    || pf_die "gh api issues/comments failed"

HEAD_REF="$HEAD_SHA"
PR_HEAD="$(sed -n 's/.*"headRefOid"[^"]*"\([0-9a-f]\{40\}\)".*/\1/p' "$INBOX/pr_view.json" | head -1)"
[ -n "$PR_HEAD" ] && HEAD_REF="$PR_HEAD"
pf_run_to "$INBOX/checks.json" gh api --paginate --slurp --jq '[.[].check_runs]' \
    "$API/commits/$HEAD_REF/check-runs" || echo "  (CI checks unavailable -- continuing)"

pf_run_to "$INBOX/threads.json" gh api graphql -f owner="$OWNER" -f repo="$NAME" \
    -F number="$PR_NUMBER" \
    -f query='query($owner:String!,$repo:String!,$number:Int!){
      repository(owner:$owner, name:$repo){
        pullRequest(number:$number){
          reviewThreads(first:100){ nodes { isResolved
            comments(first=20){ nodes { databaseId } } } }
        }
      }
    }' || echo "  (review threads unavailable -- continuing)"

if [ -n "$DO_PULL" ]; then
    if [ -n "$(pf_git status --porcelain)" ]; then
        echo
        echo "  --pull skipped: working tree is dirty, refusing to touch it." >&2
    else
        pf_run pf_git fetch "$FORK_REMOTE" "$REMOTE_BRANCH" || exit $?
        REMOTE_SHA="$(pf_git rev-parse FETCH_HEAD)"
        LOCAL_SHA="$(pf_head_sha)"
        if [ "$REMOTE_SHA" != "$LOCAL_SHA" ]; then
            pf_run pf_git merge --ff-only FETCH_HEAD || {
                echo "  The fork branch diverged; resolve it manually before continuing." >&2
                exit 24
            }
            pf_set head_sha "$(pf_head_sha)"
        fi
    fi
fi

echo
echo "-- summarizing for the AI iteration"
"$PF_PYTHON" -m llm.auto_code_gen.pr_flow.review_comments summarize \
    --inbox "$INBOX" --state "$PF_STATE" --round "$ROUND"
SUM_RC=$?
pf_set round "$ROUND"

echo
if [ "$SUM_RC" = "28" ]; then
    echo "No new review comments to address in round $ROUND."
    echo "Re-run later, or once reviewers have replied."
    exit 28
fi

pf_show_pending "$ROUND"
echo
echo "NEXT -- run the AI fix iteration (no repo/GitHub access, it is blocked):"
echo "  $PF_PYTHON -m llm.auto_code_gen.pr_flow.run_pr_review_iters --config $PF_CONFIG --round $ROUND"
exit 0
