---
description: "Create JIRA tasks from cross-trace analysis improvement plans"
---

# Create JIRA Tasks

Generate JIRA tasks from the improvement plan produced by cross-trace analysis. Each improvement item becomes a structured JIRA task with description, acceptance criteria, and priority.

## Arguments

The user provides:
- A path to a cross-trace config JSON file (required)

## Instructions

1. Parse the argument to get the config path. If not provided:
   - Find available cross-trace configs with improvement plans:
     ```bash
     find auto_analyze/results -name "cross_trace_config.json" 2>/dev/null | head -20
     ```
   - Ask the user to pick one.

2. Validate the config exists and that the cross-trace analysis has been completed with `make_improvement_plan: true`.

3. Run the JIRA creator:
   ```bash
   source env.sh && python -m auto_analyze.run_jiras --cross-config "<config_path>"
   ```

4. Report results:
   - List the JIRA tasks that were created
   - Show the output files

## Example Usage

```
/jiras auto_analyze/results/test_results/test_1/cross_trace_config.json
```
