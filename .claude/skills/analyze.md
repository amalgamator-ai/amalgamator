---
description: "Run GPU trace analysis — single-trace analysis or cross-trace comparison"
---

# Trace Analysis

Analyze GPU execution traces from LLM inference frameworks. Supports two modes:
- **Single-trace**: Analyze one framework's trace to extract transformer block structure, GPU ops, and performance insights
- **Cross-trace**: Compare two or more traces across frameworks or commits

## Arguments

The user provides:
- A path to a config JSON file (single-trace or cross-trace format)
- Optional: `--no-clean` to preserve existing output

The config type is auto-detected from the JSON content.

## Instructions

1. Parse the argument to get the config path. If not provided:
   - Check if there are generated configs from a previous parse step:
     ```bash
     find auto_analyze/results -name "single_trace_config_*.json" -o -name "cross_trace_config.json" 2>/dev/null | head -20
     ```
   - Also check example configs: `ls auto_analyze/configs/*_example.json`
   - Ask the user to pick one.

2. Determine the config type by reading the file:
   - If it contains `"traces"` and `"target_trace_id"` keys → **cross-trace**
   - If it contains `"framework_name"` and `"trace_file"` keys → **single-trace**

3. Run the appropriate analysis:

   **Single-trace:**
   ```bash
   source env.sh && python -m auto_analyze.run_single_trace --config "<config_path>"
   ```

   **Cross-trace:**
   ```bash
   source env.sh && python -m auto_analyze.run_cross_trace --config "<config_path>"
   ```

   Add `--no-clean` if the user requested it.

4. Report results:
   - Show the output directory
   - List key output files and their status (OK/MISSING)
   - For single-trace: transformer_block_high_level_ops.txt, gpu_ops.txt, gpu_ops_to_blocks.txt, perf_analysis_single_trace.txt
   - For cross-trace: cross_matching_blocks.txt, cross_compare_blocks.txt, cross_improvement_plan.txt

## Example Usage

```
/analyze auto_analyze/results/results_analyze_deepseek_r1_nvfp4/test_results/test_1/single_trace_config_vllm.json
/analyze auto_analyze/results/results_analyze_deepseek_r1_nvfp4/test_results/test_1/cross_trace_config.json
```
