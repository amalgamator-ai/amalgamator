#!/bin/bash
# Step 2c -- post the reviewer-suggestion comment drafted by step 2b.
#
#   Network access : YES -- read-only 'gh api' to check idempotency, then ONE
#                    'gh pr comment' write (after the 'yes' confirmation).
#   Safety         : refuses if the draft is missing, already posted (state or
#                    marker scan), or the PR is not OPEN.
#
# If you edited pr_reviewers.txt after step 2b, your edited file is what gets
# posted; the flow re-stamps the hash so it knows the posted version.
#
# Usage: pr_flow_step_2c_post_reviewers.sh --config <cfg> [--dry-run]

source "$(dirname "$0")/lib_pr_flow.sh"

while [ $# -gt 0 ]; do
    case "$1" in
        --config)   PF_CONFIG="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=1; shift ;;
        *) pf_die "unknown argument: $1" ;;
    esac
done

pf_resolve_state
pf_check gh-auth

PR_NUMBER="$(pf_get pr_number)"
PR_URL="$(pf_get pr_url)"
UPSTREAM="$(pf_get upstream_repo)"
[ -n "$PR_NUMBER" ] || pf_die "no pr_number in state -- run pr_flow_step_2_push.sh first"

DRAFT="$(pf_dir descs_dir "$(pf_output_dir)")/pr_reviewers.txt"
[ -f "$DRAFT" ] || pf_die "no draft at $DRAFT -- run pr_flow_step_2b_suggest_reviewers.sh first"

POSTED_URL="$(pf_get reviewers_comment_url)"
if [ -n "$POSTED_URL" ]; then
    pf_die "reviewer comment already posted: $POSTED_URL (idempotent; refusing to post twice)"
fi

OWNER="${UPSTREAM%%/*}"
NAME="${UPSTREAM%%/*}"; NAME="${UPSTREAM##*/}"

# Idempotency + PR-state checks are read-only; run them before confirming.
echo
echo "-- guardrail: reviewer-comment not already on the PR, PR is OPEN"
"$PF_PYTHON" - "$PR_NUMBER" "$OWNER" "$NAME" "$DRAFT" <<'PYEOF' || exit $?
import json, subprocess, sys

pr_number, owner, name, draft = sys.argv[1:5]
marker = "<!-- ai-auto-analyze:reviewer-suggestion"
api = "repos/{}/{}/issues/{}/comments".format(owner, name, pr_number)

r = subprocess.run(["gh", "api", "--paginate", "--slurp", api],
                   capture_output=True, text=True)
if r.returncode != 0:
    print("REFUSED: cannot read PR comments: " + r.stderr.strip()[:200])
    sys.exit(25)
pages = json.loads(r.stdout or "[]")
comments = [c for page in pages for c in (page if isinstance(page, list) else [])]
existing = [c for c in comments if marker in (c.get("body") or "")]
if existing:
    print("REFUSED: a reviewer-suggestion comment already exists: {}".format(
        existing[0].get("html_url")))
    sys.exit(25)

v = subprocess.run(["gh", "pr", "view", pr_number, "--repo",
                    "{}/{}".format(owner, name), "--json", "state"],
                   capture_output=True, text=True)
if v.returncode != 0:
    print("REFUSED: cannot read PR state: " + v.stderr.strip()[:200])
    sys.exit(25)
state = json.loads(v.stdout).get("state")
if state != "OPEN":
    print("REFUSED: PR is {} (not OPEN)".format(state))
    sys.exit(25)
print("no existing reviewer comment; PR is OPEN")
PYEOF

pf_banner "STEP 2C -- POST REVIEWER SUGGESTION"
pf_note "PR          : $PR_URL"
pf_note "draft       : $DRAFT"
pf_note "comment cmd : gh pr comment $PR_NUMBER --repo $UPSTREAM --body-file $DRAFT"

pf_banner "COMMENT BODY (posted verbatim)"
cat "$DRAFT"

pf_banner "WILL RUN"
echo "  gh pr comment $PR_NUMBER --repo $UPSTREAM --body-file $DRAFT"

pf_confirm

POST_OUT="$(pf_run gh pr comment "$PR_NUMBER" --repo "$UPSTREAM" --body-file "$DRAFT" 2>&1)" \
    || pf_die "gh pr comment failed"
echo "$POST_OUT"
COMMENT_URL="$(echo "$POST_OUT" | grep -o 'https://[^ ]*pull[^ ]*#comments\|https://[^ ]*/issues/[^ ]*' | head -1)"

# Re-stamp the hash of what was actually posted (the draft may have been
# hand-edited between 2b and 2c -- the posted file is the source of truth).
(cd "$PF_ROOT" && "$PF_PYTHON" - "$PF_STATE" "$DRAFT" "$COMMENT_URL" "$POST_OUT" <<'PYEOF'
import hashlib, re, sys
sys.path.insert(0, ".")
from llm.auto_code_gen.pr_flow import pr_state

state_path, draft, comment_url, post_out = sys.argv[1:5]
body = open(draft).read()
url = comment_url
if not url:
    m = re.search(r"https://\S+", post_out)
    url = m.group(0) if m else ""
m = re.search(r"/issues/(\d+)", url or "")
pr_state.update(state_path,
                reviewers_draft_sha256=hashlib.sha256(body.encode()).hexdigest(),
                reviewers_comment_url=url,
                reviewers_comment_id=(m.group(1) if m else ""))
print("  state: reviewers_comment_url = {}".format(url))
PYEOF
)

pf_banner "POSTED"
echo "  $COMMENT_URL"
echo "NEXT (when reviewers have replied):"
echo "  bash $(cd "$(dirname "$0")" && pwd)/pr_flow_step_3_fetch_reviews.sh --config $PF_CONFIG"
