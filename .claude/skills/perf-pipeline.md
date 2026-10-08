---
description: "Run the full performance analysis pipeline: profile → parse → analyze all traces"
---

# Full Performance Analysis Pipeline

Run the complete end-to-end pipeline: GPU profiling, result parsing, single-trace analysis for each framework, cross-trace comparison, and optionally PDF reports, Chrome traces, and JIRA tasks.

## Arguments

The user provides:
- A path to a run config JSON file (required)
- Optional skip flags: `--skip-profile`, `--skip-parse`, `--skip-analyze`, `--skip-pdf`, `--skip-trace`, `--skip-jiras`

## Instructions

1. Parse the arguments to get the run config path and any skip flags. If no config provided:
   - List available configs: `ls llm/auto_profile/test_configs/*/run_*.json`
   - Ask the user to pick one.

2. Validate the config exists.

3. Build the command with any skip flags:
   ```bash
   ./run_all.sh "<run_config>" [skip flags]
   ```

4. Run the pipeline. This is a long-running operation — inform the user it may take a while.

5. Report results:
   - Show the output directory
   - Summarize which steps completed
   - List key output files

## Skip Flags

| Flag | Effect |
|------|--------|
| `--skip-profile` | Skip step 1: benchmarks and profiling |
| `--skip-parse` | Skip step 2: parse results and generate configs |
| `--skip-analyze` | Skip step 3: all analysis |
| `--skip-pdf` | Skip step 3c: PDF report generation |
| `--skip-trace` | Skip step 3d: Chrome trace generation |
| `--skip-jiras` | Skip step 3e: JIRA task creation |

## Example Usage

```
/perf-pipeline llm/auto_profile/test_configs/deepseek_r1_nvfp4/run_deepseek_r1_nvfp4.json
/perf-pipeline llm/auto_profile/test_configs/deepseek_r1_nvfp4/run_deepseek_r1_nvfp4.json --skip-profile --skip-pdf --skip-jiras
```
