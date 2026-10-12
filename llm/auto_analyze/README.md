# GPU Performance Analysis

End-to-end automation for profiling, analyzing, and comparing GPU traces of LLM inference frameworks (vLLM, SGLang, TensorRT-LLM). Uses Claude as an AI agent.

Manual analysis of GPU profile traces is labor-intensive, particularly when correlating hundreds of low-level CUDA kernels with high-level Python/C++/CUDA code. This pipeline automates the entire workflow:

1. **Single-Trace Analysis** — Analyze a single framework's GPU trace to extract transformer block structure, correlate every GPU kernel to its high-level operation, and produce an annotated trace viewable in Perfetto
2. **Cross-Trace Analysis** — Compare two single-trace results (different commits or different frameworks) to identify all performance differences with root cause analysis and optional improvement plans
3. **Profiling** *(optional)* — Run inference benchmarks with GPU profiling and auto-generate analysis configs
4. **Code Generation** *(optional)* — Port optimizations from a faster framework to a slower one (see [`auto_code_gen`](../auto_code_gen/README.md))

All commands below are run from the repository root. For setup, see the [root README](../../README.md#getting-started).

## Guides

Detailed guides with copy-pasteable commands are in [`examples/`](examples/):

| Guide | Description |
|-------|-------------|
| [Single-Trace Analysis and Annotation](examples/single_trace_analysis_and_annotation_guide.md) | Analyze a GPU trace and produce an annotated Perfetto visualization |
| [Cross-Commit Comparison](examples/cross_commit_comparison_guide.md) | Compare two commits of the same framework |
| [Cross-Framework Comparison](examples/cross_framework_comparison_guide.md) | Compare different frameworks (e.g., vLLM vs SGLang) |
| [vLLM Trace Generation](examples/vllm_generate_trace_example.md) | How to capture PyTorch profiling traces with vLLM |

## Prerequisites

- Python 3.12+ and the project dependencies (see the [root README](../../README.md#getting-started))
- Claude access (the analysis agents use the Claude Agent SDK)
- Framework source code and GPU trace files to analyze

## Quick Start

### Single-Trace Analysis

Analyze one framework's GPU trace to produce an annotated trace with high-level operation labels:

```bash
# 1. Create the analysis config
python -m llm.auto_analyze.create_single_trace_config \
    --model nvidia/Kimi-K2.5-NVFP4 \
    --gpu-type B200 \
    --batch-size-range 1 \
    --prefill-size-range 4 \
    --output-size-range 1024 \
    --trace-file /path/to/trace.json.gz \
    --run-log-file /path/to/run_log.txt \
    --clean-source-code-path /path/to/vllm \
    --commit-id HEAD \
    --analyze-output-dir /path/to/output \
    --output-config-file /path/to/config

# 2. Run the analysis
python -m llm.auto_analyze.run_single_trace --config /path/to/config.json

# 3. Open the annotated trace in Perfetto
#    -> /path/to/output/single_trace_transformer_block.json
```

### Cross-Trace Analysis

Compare two single-trace results to identify performance differences:

```bash
# 1. Run single-trace analysis for each commit (see above)

# 2. Create the cross-trace config
python -m llm.auto_analyze.create_cross_trace_config \
    --trace-result-dir /path/to/commit_a/analyze \
    --trace-result-dir /path/to/commit_b/analyze \
    --target-trace-id 0 \
    --analyze-output-dir /path/to/cross_output \
    --output-config-file /path/to/cross_config

# 3. Run the cross-trace analysis
python -m llm.auto_analyze.run_cross_trace --config /path/to/cross_config.json
```

## Analysis Pipeline

### Single-Trace Analysis (`run_single_trace.py`)

Analyzes a single framework's GPU trace through 4 automated steps:

1. **High-level operations** — Reads framework source code to identify the sequence of logical operations in each transformer block type
2. **GPU operations extraction** — Parses the trace file to extract all GPU kernel events with timestamps, streams, and launch parameters
3. **Operation correlation** — Correlates every low-level GPU kernel to its high-level transformer block operation through source code analysis; selects the median block
4. **Annotated trace generation** — Produces a Chrome trace JSON for Perfetto with every kernel labeled with its high-level operation, source code references, and call chain

**Output files:**

| File | Description |
|------|-------------|
| `single_trace_transformer_block.json` | Annotated Chrome trace — open in [Perfetto](https://ui.perfetto.dev) |
| `single_trace_transformer_block.txt` | Human-readable annotated trace summary |
| `transformer_block_high_level_ops.txt` | High-level operation sequence from source code |
| `gpu_ops.txt` | Extracted GPU operations from trace |
| `gpu_ops_to_blocks.txt` | Full correlation of GPU ops to transformer blocks |
| `median_block.txt` | Selected median transformer block |

Optional single-trace performance analysis (enabled with `--enable-single-trace-perf-analysis`) adds bottleneck identification and improvement proposals.

### Cross-Trace Analysis (`run_cross_trace.py`)

Compares two or more single-trace results. Automatically detects the analysis mode:
- **Cross-commit** — same framework, different commits (e.g., vLLM v0.16.0 vs main)
- **Cross-framework** — different frameworks, same model (e.g., vLLM vs SGLang)

The pipeline runs 2-3 steps:

1. **Block matching** — Matches operations across median transformer blocks one-by-one
2. **Performance comparison** — Analyzes all differences (positive and negative) with root cause analysis
3. **Improvement plan** *(optional, `make_improvement_plan: true`)* — Generates ranked improvement proposals with step-by-step coding guides

**Output files:**

| File | Description |
|------|-------------|
| `cross_matching_blocks.txt` | Operation-by-operation matching across traces |
| `cross_compare_blocks.txt` | Performance comparison with root causes and code references |
| `cross_improvement_plan.txt` | *(optional)* Improvement plan with coding guides |

### Additional Tools

| Script | Description |
|--------|-------------|
| `run_chrome_trace.py` | Generate Chrome trace JSON for Perfetto (single or cross mode) |
| `run_summary_pdf.py` | Generate PDF report from analysis results |
| `run_jiras.py` | Create JIRA tasks from improvement plans |

## Config Helper Scripts

### `create_single_trace_config.py`

Generates a single-trace config JSON from command-line parameters. Automatically infers framework name and run command from the run log.

```bash
python -m llm.auto_analyze.create_single_trace_config --help
```

Key parameters: `--model`, `--gpu-type`, `--trace-file`, `--run-log-file`, `--clean-source-code-path`, `--commit-id`, `--analyze-output-dir`, `--output-config-file`

### `create_cross_trace_config.py`

Generates a cross-trace config JSON from single-trace result directories. Validates that required output files exist.

```bash
python -m llm.auto_analyze.create_cross_trace_config --help
```

Key parameters: `--trace-result-dir` (repeatable), `--target-trace-id`, `--analyze-output-dir`, `--output-config-file`, `--make-improvement-plan`

## Full Pipeline (`run_all.sh`)

For automated profiling + analysis across multiple frameworks:

```bash
./run_all.sh <run_config>
# Example: ./run_all.sh ./llm/auto_profile/test_configs/deepseek_r1_nvfp4/run_deepseek_r1_nvfp4.json
```

This orchestrates:
1. Run benchmarks and GPU profiling across frameworks
2. Parse results and generate per-test-case analysis configs
3. Run single-trace analysis for each framework
4. Run cross-trace analysis comparing frameworks
5. Generate PDF reports, Chrome traces, and JIRA tasks

Use `run_all_scheduled.sh` to schedule execution at a future time with GPU availability checks.

## Source Layout

```
llm/
├── auto_analyze/                     # Analysis pipeline
│   ├── run_single_trace.py           #   Single-trace analysis entry point
│   ├── run_cross_trace.py            #   Cross-trace analysis entry point
│   ├── run_chrome_trace.py           #   Chrome trace JSON generation
│   ├── run_summary_pdf.py            #   PDF report generation
│   ├── run_jiras.py                  #   JIRA task creation
│   ├── create_single_trace_config.py #   Helper: generate single-trace config JSON
│   ├── create_cross_trace_config.py  #   Helper: generate cross-trace config JSON
│   ├── configs/                      #   Config dataclasses and examples
│   ├── prompts/                      #   Prompt templates for all analysis steps
│   └── examples/                     #   Guides and example runs
└── auto_profile/                     # Profiling orchestration
    ├── run_profile_core.sh           #   Main profiling script
    ├── run_profile_summary.py        #   Parse results into analysis configs
    ├── parse_run_config.py           #   Config parser and validator
    └── test_configs/                 #   JSON configs (infra, run, docker, GPUs)
```

## Logging

All pipeline steps log to `logs/run_{step_name}.log` with simultaneous stdout output.

| Step | Log file |
|------|----------|
| Single-trace analysis | `logs/run_single_trace.log` |
| Cross-trace analysis | `logs/run_cross_trace.log` |
| Summary PDF | `logs/run_summary_pdf.log` |
| Chrome trace | `logs/run_chrome_trace.log` |
| JIRA creation | `logs/run_create_jiras.log` |
