---
description: "Run the AI code generation pipeline — LLM framework optimization porting or bug fix porting"
---

# AI Code Generation Pipeline

Run the multi-phase AI code generation pipeline with iterative generate+review convergence. Supports two modes:

- **LLM Framework** (`use_case: llm_framework`): Port GPU kernel optimizations between inference frameworks (vLLM, SGLang, TRT-LLM). Requires prior cross-trace analysis results.
- **Bug Fix** (`use_case: bug_fix`): Port bug fixes between branches of C/systems projects (gcc, openssl, glibc, curl).

## Arguments

The user provides:
- A path to a code gen config JSON file (required)
- Optional: `--resume` to resume from last incomplete step

## Instructions

1. Parse the arguments to get the config path and resume flag. If no config provided:
   - List available configs: `ls auto_code_gen/configs/*.json`
   - Show examples of both modes:
     - LLM framework: `auto_code_gen/configs/code_gen_config_example.json`
     - Bug fix: `auto_code_gen/configs/bug_fix_config_example.json`
   - Ask the user which mode they want and help them create a config if needed.

2. Validate the config:
   - Check the file exists
   - Read it and identify the `use_case` field (defaults to `llm_framework`)
   - Verify required fields are present:
     - **LLM framework**: `cross_trace_config`, `improvement_id`, `source_code_dir`, `output_dir`
     - **Bug fix**: `repo_path`, `source_branch`, `target_branch`, `source_fix_commit`, `output_dir`, `build_command`, `test_command`

3. Run the pipeline:
   ```bash
   source env.sh && python -m auto_code_gen.run_code_gen --config "<config_path>" [--resume]
   ```

4. This is a long-running operation (can take hours for complex ports). Inform the user.

5. Report results:
   - Show the output directory
   - Summarize phase results (iterations, convergence, duration, cost)
   - Note key output files: code patches, review summaries, runtime logs

## Pipeline Phases

| Phase | Description |
|-------|-------------|
| Phase 0 | Baseline test snapshot (bug fix only, optional) |
| Phase 1 | Code traces — analyze source/target code paths |
| Phase 1.5 | Port tests from source (bug fix only) |
| Phase 1.7 | Test validation / RED check (bug fix only) |
| Phase 2 | Code port plan — iterative generate+review |
| Phase 3 | Test plan — iterative generate+review |
| Phase 4 | Code generation — iterative generate+review + compile |
| Phase 5 | Runtime iterations — apply, run, investigate, fix |
| Phase 6 | Baseline diff — do-no-harm check (bug fix only) |

## Config Options

Key config fields (both modes):
- `thinking-mode`: `"deep"` (Opus, max effort) or `"normal"` (Sonnet, medium effort)
- `num_code_port_plan_iterations`: Default 3
- `num_code_gen_iterations`: Default 3
- `code_port_plan_skip_review`: Skip review iterations for faster runs
- `code_gen_skip_review`: Skip code gen reviews

## Example Usage

```
/codegen auto_code_gen/configs/code_gen_config_example.json
/codegen auto_code_gen/configs/bug_fix_config_example.json --resume
```
