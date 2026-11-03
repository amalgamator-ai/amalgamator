#!/bin/bash
# auto_profile for Qwen3.5-35B-A3B-FP8 (vLLM vs SGLang, 8K/1K, trace on).
set -u
cd "$HOME/amalgamator" || exit 1
source "$HOME/amalgamator_env.sh"
echo "START $(date -u +%H:%M:%S)" > ~/qwen38/logs/auto_profile_q355.log
.venv/bin/python -m llm.auto_profile_server_client \
  --config llm/auto_profile_server_client/test_configs/qwen35_35b_a3b_mi300x_decode8k.json \
  >> ~/qwen38/logs/auto_profile_q355.log 2>&1
echo "PROFILE_Q355_DONE $? $(date -u +%H:%M:%S)" >> ~/qwen38/logs/auto_profile_q355.log
