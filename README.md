# AI-Based Automatic Performance Analysis

End-to-end automation for profiling, analyzing, and comparing GPU traces of LLM inference frameworks (vLLM, SGLang, TensorRT-LLM), and for porting bug fixes between branches of C/systems projects. Uses Claude as an AI agent.

## Overview

This repo contains two independent AI pipelines:

### GPU Performance Analysis (vLLM / SGLang / TRT-LLM)

Manual analysis of GPU profile traces is labor-intensive, particularly when correlating hundreds of low-level CUDA kernels with high-level Python/C++/CUDA code. This tool automates the entire workflow:

1. **Single-Trace Analysis** — Analyze a single framework's GPU trace to extract transformer block structure, correlate every GPU kernel to its high-level operation, and produce an annotated trace viewable in Perfetto
2. **Cross-Trace Analysis** — Compare two single-trace results (different commits or different frameworks) to identify all performance differences with root cause analysis and optional improvement plans
3. **Profiling** *(optional)* — Run inference benchmarks with GPU profiling and auto-generate analysis configs
4. **Code Generation** *(optional)* — Port optimizations from a faster framework to a slower one

### Bug Fix Porting (C/Systems Projects)

Port a bug fix from one branch of a C/systems project (gcc, openssl, glibc, etc.) to another using Claude as the AI engine. The pipeline traces what the fix touches, ports its tests, plans and generates the ported patch, then drives an autonomous build-test-fix loop until the fix is clean on the target branch — no manual copy-pasting of compiler errors required.

## Examples and Guides

### GPU Performance Analysis

Detailed guides with copy-pasteable commands are in `auto_analyze/examples/`:

| Guide | Description |
|-------|-------------|
| [Single-Trace Analysis and Annotation](https://github.com/neuralmagic/ai_auto_analyze/blob/main/auto_analyze/examples/single_trace_analysis_and_annotation_guide.md) | Analyze a GPU trace and produce an annotated Perfetto visualization |
| [Cross-Commit Comparison](https://github.com/neuralmagic/ai_auto_analyze/blob/main/auto_analyze/examples/cross_commit_comparison_guide.md) | Compare two commits of the same framework |
| [Cross-Framework Comparison](https://github.com/neuralmagic/ai_auto_analyze/blob/main/auto_analyze/examples/cross_framework_comparison_guide.md) | Compare different frameworks (e.g., vLLM vs SGLang) |
| [vLLM Trace Generation](https://github.com/neuralmagic/ai_auto_analyze/blob/main/auto_analyze/examples/vllm_generate_trace_example.md) | How to capture PyTorch profiling traces with vLLM |

Worked examples with full analysis results:

- [Cross-commit example (Kimi-K2.5)](https://github.com/neuralmagic/ai_auto_analyze/tree/main/auto_analyze/examples/cross_commit_cmp_example_kimi) — vLLM `main` vs `v0.16.0` on 8xB200, showing a 21.3% improvement from specialized CUDA kernels
- [Cross-framework example (Kimi-K2.5)](https://github.com/neuralmagic/ai_auto_analyze/tree/main/auto_analyze/examples/cross_framework_cmp_example_kimi) — vLLM vs SGLang on 8xB200, showing an 8.6% gap driven by MoE dual-stream design differences

### Bug Fix Porting

| Guide | Description |
|-------|-------------|
| [auto_code_gen README](auto_code_gen/README.md) | Unified code gen pipeline — covers both LLM framework and bug fix use cases, with config examples for gcc, openssl, and curl |

## Project Structure

```
ai_auto_analyze/
├── common/                             # Shared utilities
│   ├── claude_utils.py                 #   Claude Agent SDK wrapper
│   ├── utils.py                        #   Logging, directory cleanup, output dir helpers
│   └── convert_nsys_to_sqlite.sh       #   NSYS trace format converter
├── auto_analyze/                       # Analysis pipeline
│   ├── run_single_trace.py             #   Single-trace analysis entry point
│   ├── run_cross_trace.py              #   Cross-trace analysis entry point
│   ├── run_chrome_trace.py             #   Chrome trace JSON generation
│   ├── run_summary_pdf.py              #   PDF report generation
│   ├── run_jiras.py                    #   JIRA task creation
│   ├── create_single_trace_config.py   #   Helper: generate single-trace config JSON
│   ├── create_cross_trace_config.py    #   Helper: generate cross-trace config JSON
│   ├── configs/                        #   Config dataclasses and examples
│   │   ├── single_trace_config.py
│   │   ├── single_trace_config_example.json
│   │   ├── cross_trace_config.py
│   │   ├── cross_trace_config_example.json
│   │   └── claude_config.json
│   ├── prompts/                        #   Prompt templates for all analysis steps
│   │   ├── single_trace_prompts.py
│   │   ├── cross_trace_prompts.py
│   │   ├── chrome_trace_prompts.py
│   │   ├── jira_prompts.py
│   │   └── summary_pdf_prompts.py
│   └── examples/                       #   Guides and example runs
├── auto_profile/                       # Profiling orchestration
│   ├── run_profile_core.sh             #   Main profiling script
│   ├── run_profile_summary.py          #   Parse results into analysis configs
│   ├── parse_run_config.py             #   Config parser and validator
│   └── test_configs/                   #   JSON configs (infra, run, docker, GPUs)
├── auto_code_gen/                      # AI-based code generation pipeline
│   ├── run_code_gen.py                #   Pipeline orchestrator (all use cases)
│   ├── code_gen_prompts.py            #   Shared prompt classes
│   ├── code_gen_configs.py            #   PipelineConfig, CodeGenConfig, BugFixConfig
│   ├── use_cases/                     #   Pluggable use case definitions
│   │   ├── base.py                    #     UseCase ABC
│   │   ├── llm_framework.py           #     LLM framework optimization porting
│   │   └── bug_fix.py                 #     Bug fix branch porting
│   ├── configs/                       #   Example JSON configs
│   ├── run_runtime_iters.py           #   Standalone runtime iteration runner
│   ├── run_investigate_issue.py
│   ├── run_fix_issue.py
│   ├── run_work_items.py
│   └── run_summary.py
├── env.sh                              # Environment variables
├── run_all.sh                          # Full pipeline orchestrator
└── run_all_scheduled.sh                # Scheduled pipeline execution
```

## Prerequisites

- Python 3.10+
- [Claude Agent Python SDK](https://pypi.org/project/claude-agent-sdk/) (`pip install claude-agent-sdk`)
- Framework source code (clean git repos for each framework being analyzed)
- GPU trace files (PyTorch Chrome traces or NSYS traces)

## Quick Start

### Single-Trace Analysis

Analyze one framework's GPU trace to produce an annotated trace with high-level operation labels:

```bash
# 1. Create the analysis config
python -m auto_analyze.create_single_trace_config \
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
python -m auto_analyze.run_single_trace --config /path/to/config.json

# 3. Open the annotated trace in Perfetto
#    -> /path/to/output/single_trace_transformer_block.json
```

### Cross-Trace Analysis

Compare two single-trace results to identify performance differences:

```bash
# 1. Run single-trace analysis for each commit (see above)

# 2. Create the cross-trace config
python -m auto_analyze.create_cross_trace_config \
    --trace-result-dir /path/to/commit_a/analyze \
    --trace-result-dir /path/to/commit_b/analyze \
    --target-trace-id 0 \
    --analyze-output-dir /path/to/cross_output \
    --output-config-file /path/to/cross_config

# 3. Run the cross-trace analysis
python -m auto_analyze.run_cross_trace --config /path/to/cross_config.json
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
python -m auto_analyze.create_single_trace_config --help
```

Key parameters: `--model`, `--gpu-type`, `--trace-file`, `--run-log-file`, `--clean-source-code-path`, `--commit-id`, `--analyze-output-dir`, `--output-config-file`

### `create_cross_trace_config.py`

Generates a cross-trace config JSON from single-trace result directories. Validates that required output files exist.

```bash
python -m auto_analyze.create_cross_trace_config --help
```

Key parameters: `--trace-result-dir` (repeatable), `--target-trace-id`, `--analyze-output-dir`, `--output-config-file`, `--make-improvement-plan`

## Full Pipeline (`run_all.sh`)

For automated profiling + analysis across multiple frameworks:

```bash
./run_all.sh <run_config>
# Example: ./run_all.sh ./auto_profile/test_configs/deepseek_r1_nvfp4/run_deepseek_r1_nvfp4.json
```

This orchestrates:
1. Run benchmarks and GPU profiling across frameworks
2. Parse results and generate per-test-case analysis configs
3. Run single-trace analysis for each framework
4. Run cross-trace analysis comparing frameworks
5. Generate PDF reports, Chrome traces, and JIRA tasks

Use `run_all_scheduled.sh` to schedule execution at a future time with GPU availability checks.

## Claude Agent Integration

All AI-driven steps use the Claude Agent SDK via `common/claude_utils.py`:

| Parameter | Value |
|-----------|-------|
| Model | `claude-opus-4-6[1m]` (1M context) |
| Allowed tools | `Read`, `Write`, `Bash` |
| Permission mode | `acceptEdits` |
| Thinking mode | Adaptive |
| Effort | Max |

## Logging

All pipeline steps log to `logs/run_{step_name}.log` with simultaneous stdout output.

| Step | Log file |
|------|----------|
| Single-trace analysis | `logs/run_single_trace.log` |
| Cross-trace analysis | `logs/run_cross_trace.log` |
| Summary PDF | `logs/run_summary_pdf.log` |
| Chrome trace | `logs/run_chrome_trace.log` |
| JIRA creation | `logs/run_create_jiras.log` |

## Bug Fix Porting

Port a bug fix from one branch of a C/systems project to another using Claude as the AI engine. Supported for any project with a git repository and shell-invokable build and test commands (gcc, openssl, glibc, etc.).

See **[auto_code_gen/README.md](auto_code_gen/README.md)** for the full setup and usage guide (covers both LLM framework and bug fix use cases).

**Quick start:**

```bash
# 1. Create a JSON config (see auto_code_gen/configs/bug_fix_config_example.json)
# 2. Run
python -m auto_code_gen.run_code_gen --config <path_to_config.json>
```
