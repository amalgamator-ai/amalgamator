---
description: "Generate summary reports — PDF from trace analysis or PPTX from code generation"
---

# Summary Report Generation

Generate summary reports from pipeline results. Supports two formats:

- **PDF** — from cross-trace analysis results (performance comparison reports)
- **PPTX** — from code generation results (code gen process summary)

## Arguments

The user provides:
- A config path
- For PDF: cross-trace config JSON
- For PPTX: code gen config JSON (additional paths auto-discovered from output directory)

## Instructions

1. Parse the arguments. If not provided, ask the user what kind of summary they want.

2. Determine the format by reading the config:
   - If the config contains `"traces"` → **PDF** (cross-trace analysis summary)
   - If the config contains `"use_case"` or `"cross_trace_config"` → **PPTX** (code gen summary)

3. Run the appropriate generator:

   **PDF (trace analysis):**
   ```bash
   python -m auto_analyze.run_summary_pdf --cross-config "<config_path>"
   ```

   **PPTX (code generation):**
   First, find the required files in the output directory:
   ```bash
   python -c "import json; print(json.load(open('<config_path>'))['output_dir'])"
   ```
   Then locate framework code trace files in the output directory:
   ```bash
   ls <output_dir>/code_trace_*.txt
   ```
   Then run:
   ```bash
   python -m auto_code_gen.run_summary \
     --config "<config_path>" \
     --framework-code-trace-files <trace_file_1> <trace_file_2>
   ```

4. Report the path to the generated file.

## Example Usage

```
/summary auto_analyze/results/test_results/test_1/cross_trace_config.json
/summary auto_code_gen/configs/code_gen_config_example.json
```
