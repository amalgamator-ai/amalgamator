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

### Bug Fix Porting

Backport a bug fix from one branch to another using Claude as the AI engine. The [`auto_bug_fix`](auto_bug_fix/README.md) pipeline is git-first: it exhausts deterministic mechanisms (`git cherry-pick`, mechanical re-apply) before letting an LLM touch anything, audits every LLM edit against deterministic evidence, and escalates to an autonomous build-test-fix loop only as a last resort.

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
| [auto_bug_fix README](auto_bug_fix/README.md) | Git-first backport pipeline — phase-by-phase walkthrough, config reference, and example YAML configs |

## Project Structure

```
ai_auto_analyze/
├── common/                             # Shared utilities
│   ├── llm_client.py                   #   Model-agnostic LLM completion (LiteLLM)
│   ├── agent_runner.py                 #   Agent dispatch: Claude SDK or LiteLLM tool loop
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
│   ├── code_gen_configs.py            #   PipelineConfig, CodeGenConfig
│   ├── use_cases/                     #   Pluggable use case definitions
│   │   ├── base.py                    #     UseCase ABC
│   │   └── llm_framework.py           #     LLM framework optimization porting
│   ├── configs/                       #   Example JSON configs
│   ├── run_runtime_iters.py           #   Standalone runtime iteration runner
│   ├── run_investigate_issue.py
│   ├── run_fix_issue.py
│   ├── run_work_items.py
│   └── run_summary.py
├── auto_bug_fix/                       # Git-first bug fix backport pipeline
│   ├── run_bug_fix.py                 #   Phase 0-5 orchestrator, entry point
│   ├── bug_fix_config.py              #   Config dataclass
│   ├── config_loader.py               #   YAML/JSON loading and validation
│   ├── bug_fix_prompts.py             #   Agent prompt templates
│   ├── phase_4b.py                    #   Autonomous build-test-fix loop prompt
│   ├── git_tools.py                   #   git plumbing (ancestry, worktrees, status)
│   ├── mechanical.py                  #   Mechanical conflict resolution
│   ├── evidence.py                    #   Token/line-provenance audits
│   └── configs/                       #   Example YAML configs
├── env.sh                              # Environment variables
├── run_all.sh                          # Full pipeline orchestrator
└── run_all_scheduled.sh                # Scheduled pipeline execution
```

## Prerequisites

- Python 3.14+ (use [uv](https://docs.astral.sh/uv/) for version management)
- Dependencies: `uv sync` (installs `claude-agent-sdk`, `litellm`, `colorama`, `pyyaml`)
- An LLM endpoint — Claude via Vertex AI, a local Ollama instance, a
  self-hosted vLLM server, or an OpenAI API key
- For GPU analysis: framework source code and GPU trace files

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

## Model-Agnostic LLM Support

The backporting pipeline was originally built to work only with Claude
(Anthropic) models. It now supports **any LLM provider** — Claude,
open-weight models like Qwen and Llama (via Ollama or vLLM), OpenAI GPT,
and any OpenAI-compatible API server.

### What changed

The pipeline logic (phases 0–5, git operations, cherry-pick, build-test-fix)
is **unchanged**. What changed is the layer that talks to the LLM:

- A unified completion layer (`common/llm_client.py`) routes all LLM calls
  through [LiteLLM](https://github.com/BerriAI/litellm), which handles
  provider differences automatically.
- An agent runner (`common/agent_runner.py`) provides a tool-use loop for
  non-Claude models, giving them the same `read_file`, `write_file`, and
  `bash` capabilities that the Claude Agent SDK provides.
- The YAML config now accepts `model_params` (temperature, top_p, etc.)
  and `api_key_file` (path to an API key on disk), so you can tune model
  behavior and manage credentials without editing code or environment variables.

To switch models, you only change the config YAML — **no code changes required**.

### Before vs After

**Before** (Claude-only):

```yaml
model: "claude-opus-4-6"
# temperature, api keys, etc. were hardcoded or set via env vars
```

**After** (any model, with optional tuning):

```yaml
model: "openai/Qwen3-27B"
api_base: "http://your-server:8000/v1"
model_params:
  temperature: 0
  top_p: 0.95
api_key_file: "~/.keys/vllm"
```

Existing Claude configs continue to work with zero changes.

### Supported Providers

| Provider | YAML `model:` value | `api_base:` needed? | Cost tracking |
|----------|---------------------|---------------------|---------------|
| **Claude via Vertex AI** | `claude-opus-4-6` | No (uses `CLAUDE_CODE_USE_VERTEX` env var) | Per-token cloud pricing |
| **Claude direct API** | `claude-opus-4-6` | No (uses `ANTHROPIC_API_KEY` env var) | Per-token cloud pricing |
| **Ollama (local)** | `ollama/qwen3:14b` | No (default `localhost:11434`) | $0.00 (self-hosted) |
| **vLLM / TGI (remote)** | `openai/<model-name>` | **Yes** | $0.00 (self-hosted) |
| **OpenAI** | `gpt-4o` | No | Per-token cloud pricing |

### Config Examples

**Claude via Vertex AI** (default — no changes needed from existing configs):

```yaml
model: "claude-opus-4-6"
```

Set `CLAUDE_CODE_USE_VERTEX=1` and authenticate via `gcloud auth login`.

**Self-hosted vLLM / TGI server:**

```yaml
model: "openai/Inferact/Qwen3.8-27B-NVFP4"
api_base: "http://your-server:8000/v1"
```

Set `OPENAI_API_KEY=dummy` (required by the client library, but not
validated by self-hosted servers). The `api_base` can also be set via
the `OPENAI_API_BASE` environment variable.

**Self-hosted vLLM with tuned parameters:**

```yaml
model: "openai/Qwen3-27B"
api_base: "http://your-server:8000/v1"
model_params:
  temperature: 0
  max_tokens: 16384
  top_p: 0.95
api_key_file: "~/.keys/vllm"
```

**Local Ollama:**

```yaml
model: "ollama/qwen3:14b"
```

Ensure Ollama is running (`ollama serve`). Thinking-mode tags are
automatically stripped for models that emit them (Qwen 3, DeepSeek-R1).

### Config Reference

All fields below are set at the **top level** of the pipeline YAML config.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | No | LiteLLM model identifier (see provider table above). Defaults to `claude-sonnet-4-6`. |
| `api_base` | string | No | Base URL for OpenAI-compatible endpoints (vLLM, TGI). Can also be set via `OPENAI_API_BASE` env var. |
| `model_params` | mapping | No | Parameters merged into every LLM call. Supports any key the provider accepts: `temperature`, `top_p`, `max_tokens`, `reasoning_effort`, etc. |
| `api_key` | string | No | Explicit API key. Overrides the `OPENAI_API_KEY` environment variable. |
| `api_key_file` | string | No | Path to a file containing the API key. Supports `~` expansion (e.g. `~/.keys/vllm`). The file contents are read and trimmed at startup. |
| `claude.allowed_tools` | list | No | Tools available to the Claude Agent SDK (default: `["Read", "Write", "Bash"]`). |
| `claude.perm_mode` | string | No | Claude Agent SDK permission mode (default: `"acceptEdits"`). |
| `advanced.phase_4b_model` | string | No | Override model for the Phase 4b escalation loop. Defaults to the pipeline `model`. |

**Precedence rules:**
- **Model parameters:** Explicit arguments in code > `model_params` from YAML > hardcoded defaults.
- **API keys:** `api_key` / `api_key_file` from YAML > `OPENAI_API_KEY` env var.
- **API base:** `api_base` from YAML > `OPENAI_API_BASE` env var.

### How It Works

The pipeline routes models through two paths:

1. **Claude models** (`claude-*`, `vertex_ai/claude-*`) →
   Claude Agent SDK for autonomous multi-step tool use.
2. **Everything else** (`ollama/*`, `openai/*`, `gpt-*`) →
   LiteLLM-powered tool-use loop with `read_file`, `write_file`, and
   `bash` tools, loop detection, and malformed-argument normalization.

Dispatch is automatic based on the model name. All pipeline phases
(semantic triage, conflict resolution, test porting, Phase 4b
build-test-fix loop) use the same model unless overridden:

```yaml
advanced:
  phase_4b_model: "claude-sonnet-4-6"  # override just Phase 4b
```

If `phase_4b_model` is not set, it defaults to the pipeline `model`.

### Cost Tracking

Self-hosted models (Ollama, vLLM via `api_base`) report $0.00.
Cloud models use published per-token pricing. The cost summary
appears at the end of each pipeline run.

### Claude Agent SDK Integration (legacy)

For Claude models, the pipeline uses the Claude Agent SDK directly:

| Parameter | Value |
|-----------|-------|
| Model | Configurable via YAML |
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

Backport a bug fix from one branch to another using Claude as the AI engine. Works for any project with a git repository and shell-invokable build and test commands (tested on Go and C codebases).

See **[auto_bug_fix/README.md](auto_bug_fix/README.md)** for the full setup and usage guide.

**Quick start:**

```bash
# 1. Create a YAML config (see auto_bug_fix/configs/example_backport.yaml)
# 2. Run
python -m auto_bug_fix.run_bug_fix <path_to_config.yaml>
```
