---
description: "Parse GPU profiling results and generate analysis configs (single-trace + cross-trace)"
---

# Parse Profiling Results

Parse the output of GPU profiling runs, generate summary tables, and create per-test-case analysis configs for downstream trace analysis.

## Arguments

The user provides a path to a run config JSON file. If not provided, help them find one.

## Instructions

1. Parse the argument to get the run config path. If not provided:
   - List available configs: `ls auto_profile/test_configs/*/run_*.json`
   - Ask the user to pick one.

2. Validate the config exists.

3. Compute the output directory:
   ```bash
   python -c "from common.utils import output_dir_from_run_config; print(output_dir_from_run_config('<run_config>'))"
   ```

4. Run the parse step:
   ```bash
   python -m auto_profile.run_profile_summary --run-config "<run_config>" --override-output-dir
   ```

5. Report what was generated:
   - Show the output directory path
   - List the generated analysis configs (single_trace_config_*.json, cross_trace_config.json)

## Example Usage

```
/parse auto_profile/test_configs/deepseek_r1_nvfp4/run_deepseek_r1_nvfp4.json
```
