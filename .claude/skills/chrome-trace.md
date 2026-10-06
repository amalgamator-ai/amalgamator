---
description: "Generate Chrome trace JSON for Perfetto visualization from cross-trace analysis results"
---

# Chrome Trace Generation

Generate Chrome trace JSON files that can be visualized in [Perfetto](https://ui.perfetto.dev). This combines transformer block traces from multiple frameworks into a single timeline view.

## Arguments

The user provides:
- A path to a cross-trace config JSON file (required)

## Instructions

1. Parse the argument to get the config path. If not provided:
   - Find available cross-trace configs:
     ```bash
     find auto_analyze/results -name "cross_trace_config.json" 2>/dev/null | head -20
     ```
   - Ask the user to pick one.

2. Validate the config exists and is a cross-trace config (contains `"traces"` key).

3. Run the Chrome trace generator:
   ```bash
   python -m auto_analyze.run_chrome_trace --cross-config "<config_path>"
   ```

4. Report results:
   - Show the path to the generated JSON file
   - Remind the user they can view it at https://ui.perfetto.dev

## Example Usage

```
/chrome-trace auto_analyze/results/results_analyze_test/test_results/test_1/cross_trace_config.json
```
