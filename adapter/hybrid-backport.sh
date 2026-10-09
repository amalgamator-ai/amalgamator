#!/bin/bash
# hybrid-backport.sh — standalone hybrid backport driver.
#
# Runs the full hybrid pipeline against any git repo:
#
#   1. worktree at --base (default: repo HEAD)
#   2. git cherry-pick -x <upstream>
#   3. on conflict: a conflict resolver finishes the cherry-pick
#      (PLUGGABLE — see "Resolver contract" below; tested implementation:
#      synthmerge, https://gitlab.com/aarcange/synthmerge)
#   4. deterministic checks (check-backport.py: live worktree,
#      ai.diff extracted via git, metadata via captured resolver stdout)
#   5. clean  -> keep the resolver's commit, done (fast path)
#      dirty  -> emit external-context.md + case yaml, run amalgamator
#                seeded with the context; result preserved on the
#                amalgamator/<issue-id> branch of the source repo
#
# Resolver contract (pluggable front-end):
#   The gate, escalation and arbiter are resolver-agnostic. Their only
#   contract with the resolver is the git commit it leaves behind: a
#   completed cherry-pick on top of --base (HEAD != base, no unmerged
#   paths). Anything that produces that works:
#     - plain git on a clean pick (no resolver involved at all — this
#       path runs end-to-end today, gate included);
#     - a human resolving in the worktree and committing (the checker
#       and the escalation artifacts are usable directly; only this
#       unattended driver cannot pause mid-run to wait for a human);
#     - any mechanical applier or LLM resolver wired into step 3.
#   The tested automated resolver is synthmerge
#   (https://gitlab.com/aarcange/synthmerge), the default for the
#   unattended path. It is invoked as
#     synthmerge -c <config> --cache <lmdb> --vibe --continue
#   and requires merge.conflictStyle=diff3 (set below, scoped to this
#   phase; harmless for other resolvers). The B1 arbiter re-scans the
#   amalgamator's own result tree with the same checker, so the spine is
#   already exercised over more than one resolver's output.
#
# Designed to run inside an environment where synthmerge (linux binary) and
# a python with the amalgamator deps are available: natively on a Linux prod
# host, or inside one container on other hosts (see run-standalone-hybrid.sh).
#
# Usage:
#   hybrid-backport.sh --repo <path> --upstream <sha> [options]
#
# Options:
#   --base REF              base to build on (default: repo HEAD)
#   --workdir PATH          artifact dir (logs, MD, yaml, ai.diff) — created
#   --gitscratch PATH       where git worktrees live — MUST be on a
#                           case-SENSITIVE filesystem (these repos contain
#                           Makefile/makefile-style collisions). On a Linux
#                           prod host any dir works; when driven from macOS
#                           via docker, use a container-internal path.
#                           Default: same as --workdir.
#   --issue-id ID           default: hybrid-<upstream-sha12>
#   --synthmerge-bin PATH   default: synthmerge on PATH — the three
#   --synthmerge-config P   synthmerge options exist only because the
#   --synthmerge-cache PATH default/tested resolver is synthmerge
#                           (https://gitlab.com/aarcange/synthmerge);
#                           a different resolver plugged into step 3
#                           would carry its own options instead
#   --python PATH           python with amalgamator deps (default: python3)
#   --amalgamator-root PATH dir containing auto_bug_fix/ (PYTHONPATH);
#                           default: the parent dir when this script is
#                           vendored as <amalgamator>/adapter/hybrid-backport.sh
#   --model STR             litellm model (e.g. openai/deepseek-...)
#   --api-base URL          endpoint base URL
#   --api-key-file PATH     endpoint key file
#   --context-length N      model context window (default 128000)
set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "[hybrid-backport] $*"; }

REPO="" UPSTREAM="" BASE="" WORKDIR="" GITSCRATCH="" ISSUE_ID=""
SYNTHMERGE_BIN="${SYNTHMERGE_BIN:-synthmerge}"
SYNTHMERGE_CONFIG="${SYNTHMERGE_CONFIG:-}"
SYNTHMERGE_CACHE="${SYNTHMERGE_CACHE:-}"
PYTHON="${PYTHON:-python3}"
AMALGAMATOR_ROOT="${AMALGAMATOR_ROOT:-}"
MODEL="${MODEL:-}"
API_BASE="${API_BASE:-}"
API_KEY_FILE="${API_KEY_FILE:-}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-128000}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Amalgamator root: default to the parent directory when this script is
# vendored inside an amalgamator checkout (<amalgamator>/adapter/) — the
# parent contains auto_bug_fix/, which is all --amalgamator-root needs.
if [ -z "$AMALGAMATOR_ROOT" ] && [ -d "$SCRIPT_DIR/../auto_bug_fix" ]; then
    AMALGAMATOR_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

while [ $# -gt 0 ]; do
    case "$1" in
        --repo) REPO="$2"; shift 2 ;;
        --upstream) UPSTREAM="$2"; shift 2 ;;
        --base) BASE="$2"; shift 2 ;;
        --workdir) WORKDIR="$2"; shift 2 ;;
        --gitscratch) GITSCRATCH="$2"; shift 2 ;;
        --issue-id) ISSUE_ID="$2"; shift 2 ;;
        --synthmerge-bin) SYNTHMERGE_BIN="$2"; shift 2 ;;
        --synthmerge-config) SYNTHMERGE_CONFIG="$2"; shift 2 ;;
        --synthmerge-cache) SYNTHMERGE_CACHE="$2"; shift 2 ;;
        --python) PYTHON="$2"; shift 2 ;;
        --amalgamator-root) AMALGAMATOR_ROOT="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --api-base) API_BASE="$2"; shift 2 ;;
        --api-key-file) API_KEY_FILE="$2"; shift 2 ;;
        --context-length) CONTEXT_LENGTH="$2"; shift 2 ;;
        *) die "unknown option: $1" ;;
    esac
done

[ -n "$REPO" ] && [ -n "$UPSTREAM" ] || die "--repo and --upstream are required"
[ -d "$REPO/.git" ] || [ -f "$REPO/.git" ] || die "$REPO is not a git repo"
[ -n "$SYNTHMERGE_CONFIG" ] || die "--synthmerge-config is required"
[ -n "$SYNTHMERGE_CACHE" ] || die "--synthmerge-cache is required"
[ -n "$MODEL" ] || die "--model is required (e.g. openai/<model-id>)"
[ -n "$API_BASE" ] || die "--api-base is required"
[ -n "$API_KEY_FILE" ] || die "--api-key-file is required"
[ -n "$AMALGAMATOR_ROOT" ] || die "--amalgamator-root is required"
[ -x "$SYNTHMERGE_BIN" ] || [ "$(command -v "$SYNTHMERGE_BIN" || true)" ] || die "synthmerge binary not found: $SYNTHMERGE_BIN"

[ -n "$BASE" ] || BASE="$(git -C "$REPO" rev-parse HEAD)"
ISSUE_ID="${ISSUE_ID:-hybrid-$(echo "$UPSTREAM" | cut -c1-12)}"
WORKDIR="${WORKDIR:-$(mktemp -d "hybrid-${ISSUE_ID}.XXXXXX")}"
mkdir -p "$WORKDIR"
WORKDIR="$(cd "$WORKDIR" && pwd)"
GITSCRATCH="${GITSCRATCH:-$WORKDIR}"
mkdir -p "$GITSCRATCH"
WT="$GITSCRATCH/synthmerge-wt"

log "repo=$REPO upstream=$UPSTREAM base=$BASE issue=$ISSUE_ID workdir=$WORKDIR"

# ---------------------------------------------------------------------------
# Deterministic git settings for the resolver phase.
# merge.conflictStyle=diff3 is REQUIRED by synthmerge (the tested
# resolver); harmless for other resolvers. Scoped: unset before the
# amalgamator phase, which ran without these in validated runs (diff3
# would change its conflict marker format).
# ---------------------------------------------------------------------------
export GIT_CONFIG_COUNT=7
GIT_CONFIG_KEY_0="merge.conflictStyle";  GIT_CONFIG_VALUE_0="diff3"
GIT_CONFIG_KEY_1="user.name";            GIT_CONFIG_VALUE_1="${GIT_USER_NAME:-hybrid-backport}"
GIT_CONFIG_KEY_2="user.email";           GIT_CONFIG_VALUE_2="${GIT_USER_EMAIL:-hybrid-backport@localhost}"
GIT_CONFIG_KEY_3="diff.renames";         GIT_CONFIG_VALUE_3="copies"
GIT_CONFIG_KEY_4="diff.renameLimit";     GIT_CONFIG_VALUE_4="100000"
GIT_CONFIG_KEY_5="diff.algorithm";       GIT_CONFIG_VALUE_5="histogram"
GIT_CONFIG_KEY_6="advice.detachedHead";  GIT_CONFIG_VALUE_6="false"
export GIT_CONFIG_KEY_0 GIT_CONFIG_VALUE_0 GIT_CONFIG_KEY_1 GIT_CONFIG_VALUE_1 \
       GIT_CONFIG_KEY_2 GIT_CONFIG_VALUE_2 GIT_CONFIG_KEY_3 GIT_CONFIG_VALUE_3 \
       GIT_CONFIG_KEY_4 GIT_CONFIG_VALUE_4 GIT_CONFIG_KEY_5 GIT_CONFIG_VALUE_5 \
       GIT_CONFIG_KEY_6 GIT_CONFIG_VALUE_6

# ---------------------------------------------------------------------------
# 1-3. worktree, cherry-pick, resolver
#      (resolver contract: cherry-pick -x, then the resolver finishes it;
#       default and tested resolver: synthmerge — see the header's
#       "Resolver contract" for what a replacement must satisfy)
# ---------------------------------------------------------------------------
log "creating worktree at $WT"
git -C "$REPO" worktree prune
git -C "$REPO" worktree add --detach "$WT" "$BASE" >/dev/null

log "cherry-pick $UPSTREAM"
set +e
git -C "$WT" cherry-pick -x "$UPSTREAM" >"$WORKDIR/cherry-pick.log" 2>&1
CHERRY_RC=$?
set -e
cat "$WORKDIR/cherry-pick.log"

if [ "$CHERRY_RC" -ne 0 ]; then
    # Distinguish a real conflict (unmerged paths exist) from a hard failure
    # (bad object, missing alternate, bad sha...). A hard failure must die:
    # running the resolver on a broken worktree yields "No conflicts found"
    # (observed with synthmerge) and an empty result that would sail
    # through the gate (case 6155).
    UNMERGED=$(git -C "$WT" diff --name-only --diff-filter=U | wc -l | tr -d ' ')
    if [ "$UNMERGED" -eq 0 ]; then
        git -C "$WT" cherry-pick --abort 2>/dev/null || true
        git -C "$REPO" worktree remove --force "$WT" 2>/dev/null || true
        die "cherry-pick failed without conflicts (rc=$CHERRY_RC) — bad object/ref, missing alternates, or empty patch; see $WORKDIR/cherry-pick.log"
    fi
    # PLUGGABLE RESOLVER SEAM — replace this block to use a different
    # resolver. Contract: finish the cherry-pick in $WT with a commit on
    # top of $BASE (HEAD != base, no unmerged paths), exit 0 on success.
    # Default/tested resolver: synthmerge
    # (https://gitlab.com/aarcange/synthmerge).
    log "conflicts — running resolver (synthmerge)"
    set +e
    (cd "$WT" && "$SYNTHMERGE_BIN" -c "$SYNTHMERGE_CONFIG" \
        --cache "$SYNTHMERGE_CACHE" --vibe --continue \
        >"$WORKDIR/synthmerge-stdout.txt" 2>"$WORKDIR/synthmerge-stderr.txt")
    SM_RC=$?
    set -e
    cat "$WORKDIR/synthmerge-stdout.txt"
    [ "$SM_RC" -eq 0 ] || {
        git -C "$WT" cherry-pick --abort 2>/dev/null || true
        git -C "$REPO" worktree remove --force "$WT" || true
        die "synthmerge exited $SM_RC (see $WORKDIR/synthmerge-stderr.txt)"
    }
    STDOUT_FILE="$WORKDIR/synthmerge-stdout.txt"
    # synthmerge claims success — the result must be a NEW commit on top of
    # base. HEAD == base means the cherry-pick was silently lost (observed
    # with missing alternates: synthmerge reports "No conflicts found").
    RESULT_SHA="$(git -C "$WT" rev-parse HEAD)"
    BASE_SHA="$(git -C "$WT" rev-parse "$BASE")"
    if [ "$RESULT_SHA" = "$BASE_SHA" ]; then
        git -C "$REPO" worktree remove --force "$WT" 2>/dev/null || true
        die "synthmerge finished but HEAD == base (no commit created) — the cherry-pick was lost; see $WORKDIR/synthmerge-stdout.txt"
    fi
else
    log "clean cherry-pick — synthmerge not needed"
    STDOUT_FILE="$WORKDIR/cherry-pick.log"
    RESULT_SHA="$(git -C "$WT" rev-parse HEAD)"
    BASE_SHA="$(git -C "$WT" rev-parse "$BASE")"
    if [ "$RESULT_SHA" = "$BASE_SHA" ]; then
        # empty cherry-pick: fix already present at base. Legitimate no-op —
        # report and exit 0 (nothing to check, nothing to port).
        log "cherry-pick is empty (fix already at base) — nothing to do"
        git -C "$REPO" worktree remove --force "$WT" 2>/dev/null || true
        exit 0
    fi
fi

# ---------------------------------------------------------------------------
# 4. deterministic checks (live worktree)
# ---------------------------------------------------------------------------
# Always preserve the pre-gate (synthmerge/cherry-pick) result commit on
# hybrid/<issue-id> — even when escalating — so post-hoc analysis can score
# synthmerge-alone vs the amalgamator's seeded result.
git -C "$REPO" branch -f "hybrid/$ISSUE_ID" "$(git -C "$WT" rev-parse HEAD)"
log "pre-gate result preserved on branch hybrid/$ISSUE_ID"

log "running deterministic checks"
set +e
"$PYTHON" "$SCRIPT_DIR/check-backport.py" \
    --worktree "$WT" --base "$BASE" \
    --upstream "$UPSTREAM" \
    --synthmerge-stdout "$STDOUT_FILE" \
    --ai-diff-out "$WORKDIR/ai.diff" \
    --emit-external-context "$WORKDIR/external-context.md" \
    | tee "$WORKDIR/checker-pre.log"
CHECK_RC=${PIPESTATUS[0]}
PRE_FINDINGS=$(grep -c "FINDING:" "$WORKDIR/checker-pre.log" || true)
# strong findings = objective breakage classes (undefined symbols, broken
# Makefile objects). Dead-symbol findings are advisory (correlation
# signals) — they drive escalation but must not decide arbitration: a
# correct port that legitimately adds a header fires dozens of
# "resurrected" dead-decl findings (case 6155, byte-exact vs downstream).
PRE_STRONG=$(grep -cE "FINDING: .*(used in added line but not defined|listed but no)" \
    "$WORKDIR/checker-pre.log" || true)
set -e

if [ "$CHECK_RC" -eq 0 ]; then
    log "CLEAN GATE: synthmerge result stands (fast path)"
    log "result commit: $(git -C "$WT" rev-parse HEAD) (worktree kept at $WT)"
    log "preserved on branch hybrid/$ISSUE_ID"
    exit 0
fi

# Non-zero checker exit must mean findings (MD written). If the checker
# died with an error (unresolved conflicts, no HEAD...), the MD does not
# exist — escalating then would hand the amalgamator a missing context
# file. Die instead.
[ -s "$WORKDIR/external-context.md" ] || die "checker exited $CHECK_RC without writing external-context.md (checker error, not findings)"

log "findings — escalating to amalgamator (seeded)"

# Amalgamator phase: clean git env (validated runs used container defaults;
# diff3 here would change the amalgamator's conflict marker format)
unset GIT_CONFIG_COUNT GIT_CONFIG_KEY_0 GIT_CONFIG_VALUE_0 \
      GIT_CONFIG_KEY_1 GIT_CONFIG_VALUE_1 GIT_CONFIG_KEY_2 GIT_CONFIG_VALUE_2 \
      GIT_CONFIG_KEY_3 GIT_CONFIG_VALUE_3 GIT_CONFIG_KEY_4 GIT_CONFIG_VALUE_4 \
      GIT_CONFIG_KEY_5 GIT_CONFIG_VALUE_5 GIT_CONFIG_KEY_6 GIT_CONFIG_VALUE_6

# ---------------------------------------------------------------------------
# 5. build the case yaml and run amalgamator (its own worktree, own
#    cherry-pick; seeded via external_context_file)
# ---------------------------------------------------------------------------
YAML="$WORKDIR/$ISSUE_ID.yaml"
cat >"$YAML" <<EOF
# Hybrid escalation case — generated by hybrid-backport.sh
issue_id: "$ISSUE_ID"

bug_description: "backport of upstream commit $UPSTREAM"

repository:
  source_path: "$REPO"
  workdir: "$GITSCRATCH/amalgamator-worktrees"

branches:
  source: "$BASE"
  target: "$BASE"

fix:
  commit: "$UPSTREAM"

build:
  command: ""
  test_command: ""
  lint_command: ""
  auto_discover_tests: false

model: "$MODEL"
api_base: "$API_BASE"
api_key_file: "$API_KEY_FILE"

advanced:
  context_length: $CONTEXT_LENGTH

external_context_file: "external-context.md"
EOF
log "case yaml: $YAML"

mkdir -p "$WORKDIR/amalgamator-worktrees" "$WORKDIR/amalgamator-runs"
log "running amalgamator"
set +e
(
    cd "$WORKDIR/amalgamator-runs"
    PYTHONPATH="$AMALGAMATOR_ROOT" "$PYTHON" -m auto_bug_fix.run_bug_fix "$YAML"
)
AM_RC=$?
set -e

log "amalgamator exit code: $AM_RC"
if git -C "$REPO" rev-parse --verify -q "amalgamator/$ISSUE_ID" >/dev/null; then
    # Strip stray agent artifacts the amalgamator may have committed into
    # the result tree (observed: triage_assessment.json/.md at repo root,
    # case 10018 — code was exact, tree polluted). Plumbing-only, no
    # checkout: rebuild the root tree without those entries and amend.
    # run_and_fix_{success,failure}.txt: phase 4b markers written into the
    # worktree; harmless when untracked but strip them if committed.
    STRAY=$(git -C "$REPO" ls-tree --name-only "amalgamator/$ISSUE_ID" \
        | grep -E '^(triage_assessment\.(json|md)|run_and_fix_(success|failure)\.txt)$' \
        || true)
    if [ -n "$STRAY" ]; then
        log "stripping stray artifacts from result: $STRAY"
        BR="amalgamator/$ISSUE_ID"
        ROOT_TREE=$(git -C "$REPO" rev-parse "$BR^{tree}")
        NEW_TREE=$(git -C "$REPO" ls-tree "$ROOT_TREE" \
            | grep -v -E '^(triage_assessment[.](json|md)|run_and_fix_(success|failure)[.]txt)$' \
            | git -C "$REPO" mktree)
        PARENT=$(git -C "$REPO" rev-parse "$BR^")
        NEW_COMMIT=$(git -C "$REPO" log -1 --format=%B "$BR" \
            | git -C "$REPO" commit-tree "$NEW_TREE" -p "$PARENT" -F -)
        git -C "$REPO" branch -f "$BR" "$NEW_COMMIT"
        log "stripped result commit: $NEW_COMMIT"
    fi
    log "result preserved on branch amalgamator/$ISSUE_ID"

    # ------------------------------------------------------------------
    # B1: post-escalation verification (best-of-both arbiter).
    # The escalation exists to fix the pre-gate findings; verify that it
    # actually did. Re-run the same deterministic scans on the amalgamator's
    # result tree and compare finding counts:
    #   post < pre  -> escalation paid off, keep the amalgamator result
    #   post >= pre -> no improvement (or regression: wholesale replacement
    #                  shows up as many new dead/undefined symbols) — keep
    #                  the pre-gate tree as the final result, preserve the
    #                  rejected tree on amalgamator-rejected/<id> for study.
    # Deterministic, ~50s, no LLM. The scans' blind spots mean this errs
    # toward the pre-gate result — the measured-safer default.
    # ------------------------------------------------------------------
    RES_SHA=$(git -C "$REPO" rev-parse "amalgamator/$ISSUE_ID")
    PCWT="$GITSCRATCH/postcheck-$ISSUE_ID"
    git -C "$REPO" worktree add --detach --quiet "$PCWT" "$RES_SHA" 2>/dev/null || PCWT=""
    if [ -n "$PCWT" ] && [ "$PRE_FINDINGS" -gt 0 ]; then
        log "post-check: verifying amalgamator result (pre-gate findings: $PRE_FINDINGS)"
        set +e
        "$PYTHON" "$SCRIPT_DIR/check-backport.py" \
            --worktree "$PCWT" --base "$BASE" \
            --upstream "$UPSTREAM" \
            >"$WORKDIR/checker-post.log" 2>&1
        POST_RC=$?
        set -e
        POST_FINDINGS=$(grep -c "FINDING:" "$WORKDIR/checker-post.log" || true)
        POST_STRONG=$(grep -cE "FINDING: .*(used in added line but not defined|listed but no)" \
            "$WORKDIR/checker-post.log" || true)

        # Diff-shape check (semantic-equivalence gate): the port's diff must
        # be a comparable-size adaptation of the upstream fix. A wholesale
        # upstream replacement (force-resolve --theirs damage) is
        # self-consistent — symbol scans see nothing wrong — but shows up
        # as a massive size blowup. Measured on the 400-case batch: all
        # fixed/same results land at <=1.0x upstream size; blowups at
        # 4-112x. The >100-line absolute guard protects small-diff fixes
        # that legitimately scale (unit adaptations, prerequisite drops).
        shape_bad=0
        UP_INS=$(git -C "$REPO" diff --numstat "$UPSTREAM^1" "$UPSTREAM" | awk '{i+=($1=="-")?0:$1; d+=($2=="-")?0:$2} END{print i+d+0}')
        RES_INS=$(git -C "$REPO" diff --numstat "$BASE" "$RES_SHA" | awk '{i+=($1=="-")?0:$1; d+=($2=="-")?0:$2} END{print i+d+0}')
        if [ "$RES_INS" -gt $(( UP_INS * 3 )) ] && [ $(( RES_INS - UP_INS )) -gt 100 ]; then
            shape_bad=1
        fi

        # Arbitration rule: reject iff the escalation made things objectively
        # WORSE — new strong findings (undefined symbols / broken Makefiles)
        # appeared, or the diff shape blew up (wholesale replacement). A
        # result that merely kept the same advisory findings is kept: the
        # arbiter prevents harm, it does not punish wasted escalations.
        if [ "$POST_STRONG" -le "$PRE_STRONG" ] && [ "$shape_bad" -eq 0 ]; then
            log "post-check: kept amalgamator result (strong findings $PRE_STRONG -> $POST_STRONG, advisory $PRE_FINDINGS -> $POST_FINDINGS, shape ok)"
        else
            log "post-check: REJECTED amalgamator result (strong findings $PRE_STRONG -> $POST_STRONG, shape_bad=$shape_bad size=$RES_INS/up=$UP_INS) — keeping pre-gate tree"
            git -C "$REPO" branch "amalgamator-rejected/$ISSUE_ID" "$RES_SHA"
            git -C "$REPO" branch -f "amalgamator/$ISSUE_ID" "hybrid/$ISSUE_ID"
            log "rejected tree preserved on amalgamator-rejected/$ISSUE_ID"
        fi
        git -C "$REPO" worktree remove --force "$PCWT" 2>/dev/null || true
    else
        [ -z "$PCWT" ] && log "post-check: skipped (worktree add failed)"
        [ "$PRE_FINDINGS" -eq 0 ] 2>/dev/null && log "post-check: skipped (no pre-gate findings recorded)"
    fi
    log "result commit: $(git -C "$REPO" rev-parse "amalgamator/$ISSUE_ID")"
fi
exit "$AM_RC"
