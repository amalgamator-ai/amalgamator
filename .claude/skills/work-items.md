---
description: "Execute arbitrary work items within the code generation pipeline framework"
---

# Work Items

Execute arbitrary work items within the pipeline framework. Work items are custom tasks like error investigation, PR review preparation, or rebasing operations.

## Arguments

The user provides:
- A path to the code gen config JSON file (required)
- A path to a work items file (or one will be located in the output directory)

## Instructions

1. Parse the arguments. If no config provided:
   - List available configs: `ls llm/auto_code_gen/configs/*.json`
   - Ask the user to provide one.

2. Load the config to find the output directory:
   ```bash
   python -c "import json; print(json.load(open('<config_path>'))['output_dir'])"
   ```

3. Find the required files in the output directory:
   - Code gen directory: `<output_dir>`
   - Work items file: Ask the user for the path, or check for existing files:
     ```bash
     ls <output_dir>/work_items*.txt 2>/dev/null
     ```

4. Check example work item scripts for patterns:
   ```bash
   ls llm/auto_code_gen/run_work_item_*.sh
   ```
   These show patterns for error investigation, PR review, and rebase operations.

5. Run the work items:
   ```bash
   python -m llm.auto_code_gen.run_work_items \
     --config "<config_path>" \
     --code-gen-dir "<output_dir>" \
     --work-items-file "<work_items_file>"
   ```

6. Report what work items were executed and their outcomes.

## Example Usage

```
/work-items llm/auto_code_gen/configs/code_gen_config_example.json
```
