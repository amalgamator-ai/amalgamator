---
description: "Run standalone runtime iterations — apply patch, run benchmark, investigate and fix issues"
---

# Runtime Iterations

Run the runtime iteration loop as a standalone step (Phase 5 of the code generation pipeline). This applies a code patch, runs benchmarks, investigates output, and iteratively fixes issues.

Use this when you already have a code patch from the code gen pipeline and want to re-run or continue runtime iterations without re-running the full pipeline.

## Arguments

The user provides:
- A path to the code gen config JSON file (required)
- Optional: `--resume` to continue from last iteration

## Instructions

1. Parse the arguments for the config path and resume flag. If not provided:
   - List available configs: `ls auto_code_gen/configs/*.json`
   - Ask the user to provide one.

2. Validate the config exists and contains the required fields.

3. Run runtime iterations:
   ```bash
   python -m auto_code_gen.run_runtime_iters --config "<config_path>" [--resume]
   ```

4. Report results:
   - Show iteration outcomes (success/failure)
   - Show generated patches and investigation summaries
   - Report if a `runtime_success.txt` was created (indicating benchmark success)

## Example Usage

```
/runtime auto_code_gen/configs/code_gen_config_example.json
/runtime auto_code_gen/configs/code_gen_config_example.json --resume
```
