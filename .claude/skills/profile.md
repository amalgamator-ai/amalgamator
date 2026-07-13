---
description: "Run GPU profiling benchmarks across inference frameworks (vLLM, SGLang, TRT-LLM)"
---

# GPU Profiling

Run GPU benchmarking and profiling for LLM inference frameworks.

## Arguments

The user provides a path to a run config JSON file. If not provided, help them find one.

## Instructions

1. Parse the argument to get the run config path. If the user did not provide one:
   - List available configs: `ls auto_profile/test_configs/*/run_*.json`
   - Ask the user to pick one or provide a path.

2. Validate the run config exists:
   ```bash
   test -f "<run_config>" && echo "OK" || echo "NOT FOUND"
   ```

3. Run the profiling pipeline:
   ```bash
   source env.sh && ./run_step1_profile.sh "<run_config>"
   ```

4. Report results: show which frameworks were profiled and where output files are located.

## Example Configs

Run configs live in `auto_profile/test_configs/` with names like `run_deepseek_r1_nvfp4.json`.

## Example Usage

```
/profile auto_profile/test_configs/deepseek_r1_nvfp4/run_deepseek_r1_nvfp4.json
```
