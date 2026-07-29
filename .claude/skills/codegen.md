---
description: "Run the AI code generation pipeline — LLM framework optimization porting"
---

# AI Code Generation Pipeline

Run the multi-phase AI code generation pipeline with iterative generate+review convergence. One mode:

- **LLM Framework** (`use_case: llm_framework`): Port GPU kernel optimizations between inference frameworks (vLLM, SGLang, TRT-LLM). Requires prior cross-trace analysis results.

For backporting bug fixes between branches, use `/bugfix` (the `auto_bug_fix` pipeline) instead.

## Arguments

The user provides:
- A path to a code gen config JSON file (required)
- Optional: `--resume` to resume from last incomplete step

## Instructions

1. Parse the arguments to get the config path and resume flag. If no config provided:
   - List available configs: `ls auto_code_gen/configs/*.json`
   - Show the example: `auto_code_gen/configs/code_gen_config_example.json`
   - Help them create a config if needed.

2. Validate the config:
   - Check the file exists
   - Read it and identify the `use_case` field (defaults to `llm_framework`; `bug_fix` is no longer accepted here — point the user at `/bugfix`)
   - Verify required fields are present: `cross_trace_config`, `improvement_id`, `source_code_dir`, `output_dir`

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
| Phase 1 | Code traces — analyze source/target code paths |
| Phase 2 | Code port plan — iterative generate+review |
| Phase 3 | Test plan — iterative generate+review |
| Phase 4 | Code generation — iterative generate+review + compile |
| Phase 5 | Runtime iterations — apply, run, investigate, fix |

## Config Options

Key config fields:
- `thinking-mode`: `"deep"` (Opus, max effort) or `"normal"` (Sonnet, medium effort)
- `num_code_port_plan_iterations`: Default 3
- `num_code_gen_iterations`: Default 3
- `code_port_plan_skip_review`: Skip review iterations for faster runs
- `code_gen_skip_review`: Skip code gen reviews

## Example Usage

```
/codegen auto_code_gen/configs/code_gen_config_example.json
/codegen auto_code_gen/configs/code_gen_config_example.json --resume
```
