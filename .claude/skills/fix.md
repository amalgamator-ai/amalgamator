---
description: "Fix a specific post-pipeline issue — apply targeted fixes based on investigation results"
---

# Fix Issue

Apply targeted fixes for issues identified during post-pipeline investigation. Reads investigation results and applies code changes to the target framework.

## Arguments

The user provides:
- A path to the code gen config JSON file (required)
- Additional paths are auto-discovered from the output directory

## Instructions

1. Parse the argument for the config path. If not provided:
   - List available configs: `ls auto_code_gen/configs/*.json`
   - Ask the user to provide one.

2. Load the config to find the output directory:
   ```bash
   python -c "import json; print(json.load(open('<config_path>'))['output_dir'])"
   ```

3. Find the required files in the output directory:
   - Code plan file: `ls <output_dir>/code_port_plan_*.txt | tail -1`
   - PRs directory: `<output_dir>/prs/` (if it exists)
   - Issue file: `ls <output_dir>/investigate_issue_*.txt | tail -1`

4. Run the fix:
   ```bash
   source env.sh && python -m auto_code_gen.run_fix_issue \
     --config "<config_path>" \
     --high_level_code_plan_file "<plan_file>" \
     --prs_dir "<prs_dir>" \
     --issue_to_fix_file "<issue_file>"
   ```

5. Report results:
   - Show what changes were applied
   - Show compilation and test results
   - Suggest running `/runtime` to verify the fix in benchmarks

## Example Usage

```
/fix auto_code_gen/configs/code_gen_config_example.json
```
