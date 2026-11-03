#!/bin/bash
# Plan-driven codegen for Qwen3.5-35B-A3B (cross-framework plan: vLLM vs SGLang).
# The tree already carries P1 on the run branch, so the baseline is the fast
# attention path and the plan's attention item is not re-derived.
# Extra args pass through (e.g. --resume).
set -u
cd "$HOME/amalgamator" || exit 1
source "$HOME/amalgamator_env.sh"
LOG=~/qwen38/logs/codegen_q355.log
echo "START $(date -u +%H:%M:%S) args: $*" > "$LOG"
bash ~/qwen38/preflight_images.sh >> "$LOG" 2>&1 || {
  echo "IMAGE PREFLIGHT FAILED" >> "$LOG"; exit 1; }
.venv/bin/python -m llm.auto_code_gen.run_code_gen \
  --config llm/auto_code_gen/configs/qwen35_35b_a3b_mi300x_decode8k.json "$@" >> "$LOG" 2>&1
rc=$?
echo "CODE_GEN_Q355_DONE rc=$rc $(date -u +%H:%M:%S)" >> "$LOG"
