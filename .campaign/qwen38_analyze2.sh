#!/usr/bin/env bash
# The analysis checks out the tree's commit, so the PR-1 edits have to be a
# commit rather than a dirty worktree. Commit them in the measurement tree
# (content is unchanged, so running measurements stay valid) and rerun the
# analysis against that commit.
set -u
cd "$HOME/amalgamator" || exit 1
LOG=~/qwen38/logs/qwen38_analyze.log
TREE=/home/amatveev/work/wt76_pr1
RES=llm/auto_profile_server_client/results/qwen38_post_p1
OUT=llm/auto_analyze/results/results_analyze_qwen38_post_p1

echo "ANALYZE2 START $(date -u +%H:%M:%S)" >> "$LOG"

cd "$TREE" || exit 1
if [ -n "$(git status --porcelain)" ]; then
  GIT_AUTHOR_NAME="Alexander Matveev" GIT_AUTHOR_EMAIL="amatveev@redhat.com" \
  GIT_COMMITTER_NAME="Alexander Matveev" GIT_COMMITTER_EMAIL="amatveev@redhat.com" \
  git commit -q -am "measurement tree: AITER attention backend ordering" || exit 1
fi
COMMIT=$(git rev-parse --short=10 HEAD)
echo "TREE COMMIT $COMMIT" >> "$LOG"
cd "$HOME/amalgamator" || exit 1

python3 ~/qwen38/gen_single_trace_config.py \
  "$RES" "$TREE" "$COMMIT" "$OUT/single_trace_config.json" >> "$LOG" 2>&1

bash -lc "set -a; source ~/amalgamator_env.sh; set +a; \
  PYTHONPATH=. .venv/bin/python llm/auto_analyze/run_single_trace.py \
  --config $OUT/single_trace_config.json" >> "$LOG" 2>&1
echo "ANALYZE2 RC=$? $(date -u +%H:%M:%S)" >> "$LOG"
ls "$OUT" >> "$LOG"
echo "ANALYZE2_DONE $(date -u +%H:%M:%S)" >> "$LOG"
