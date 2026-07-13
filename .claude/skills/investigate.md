---
description: "Investigate post-pipeline issues — analyze runtime failures or unexpected behavior"
---

# Investigate Issue

Run a post-pipeline investigation to analyze runtime failures, unexpected benchmark results, or other issues that occurred during code generation.

## Arguments

The user provides:
- A path to the code gen config JSON file (required)
- Additional paths are auto-discovered from the output directory

## Instructions

1. Parse the arguments for the config path. If not provided:
   - List available configs: `ls auto_code_gen/configs/*.json`
   - Ask the user to provide one.

2. Load the config to find the output directory:
   ```bash
   python -c "import json; print(json.load(open('<config_path>'))['output_dir'])"
   ```

3. Find the required files in the output directory:
   - Framework code trace files: `ls <output_dir>/code_trace_*.txt`
   - Code port plan file: `ls <output_dir>/code_port_plan_*.txt | tail -1`
   - Test plan file: `ls <output_dir>/test_plan_*.txt | tail -1`

4. Run the investigation with all required arguments:
   ```bash
   source env.sh && python -m auto_code_gen.run_investigate_issue \
     --config "<config_path>" \
     --framework-code-trace-files <trace_file_1> <trace_file_2> \
     --code-port-plan-file "<plan_file>" \
     --test-plan-file "<test_plan_file>"
   ```

5. Report findings:
   - Summarize the investigation results
   - Highlight root cause if identified
   - Suggest next steps: `/fix` to apply fixes, `/runtime` to re-run benchmarks

## Example Usage

```
/investigate auto_code_gen/configs/code_gen_config_example.json
```
