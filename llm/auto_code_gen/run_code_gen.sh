#!/bin/bash

source "$(dirname "$0")/../auto_profile/utils.sh"

# Clear outputs
# clean_dir_contents "llm/auto_code_gen/prs" "vllm" "sglang"

# Run
python -m llm.auto_code_gen.run_code_gen