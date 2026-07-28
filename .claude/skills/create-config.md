---
description: "Generate config files for analysis or code generation pipelines"
---

# Create Config

Interactively generate JSON or YAML configuration files for the various pipelines.

## Arguments

The user specifies which config type they want. If not specified, ask them.

## Instructions

1. Determine which config type the user needs:
   - **single-trace** — for single-trace GPU analysis
   - **cross-trace** — for cross-trace comparison
   - **codegen** — for AI code generation (LLM framework mode)
   - **bugfix** — for the bug fix backport pipeline (YAML)

2. Based on the type, gather required fields from the user interactively.

3. Generate the config:

### Single-Trace Config

Use the CLI helper:
```bash
python -m auto_analyze.create_single_trace_config \
  --framework-name <name> \
  --model <model> \
  --gpu-type <gpu> \
  --batch-size <bs> \
  --prefill-size <ps> \
  --output-size <os> \
  --trace-file <path> \
  --framework-source-code <path> \
  --output-dir <path> \
  [--commit-id <sha>] \
  [--run-log <path>] \
  [--trace-gpu-focus <gpu_id>] \
  [--max-gpu-ops <n>] \
  [--skip-perf-analysis]
```

### Cross-Trace Config

Use the CLI helper:
```bash
python -m auto_analyze.create_cross_trace_config \
  --result-dirs <dir1> <dir2> \
  --target-trace-id <idx> \
  --output-dir <path> \
  [--make-improvement-plan]
```

### Code Gen Config (LLM Framework)

Create a JSON file based on `auto_code_gen/configs/code_gen_config_example.json`:
```json
{
    "use_case": "llm_framework",
    "cross_trace_config": "<path_to_cross_trace_config>",
    "improvement_id": 1,
    "source_code_dir": "<path_to_target_framework_source>",
    "output_dir": "<output_path>",
    "thinking-mode": "deep"
}
```

### Bug Fix Config (YAML)

Create a YAML file based on `auto_bug_fix/configs/example_backport.yaml`:
```yaml
issue_id: "<id>"
bug_description: "<description>"
repository:
  source_path: "<repo_path>"
  workdir: "<worktree_dir>"
branches:
  source: "<source_branch>"
  target: "<target_branch>"
fix:
  commit: "<sha>"
build:
  command: "<build_cmd>"
  test_command: "<test_cmd>"
model: "claude-sonnet-4-6"
```

4. Write the config file to the path specified by the user.

5. Suggest the next skill to run:
   - Single-trace → `/analyze`
   - Cross-trace → `/analyze`
   - Code gen → `/codegen`
   - Bug fix → `/bugfix`

## Example Usage

```
/create-config single-trace
/create-config bugfix
/create-config codegen
```
