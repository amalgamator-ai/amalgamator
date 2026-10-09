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

Backport a bug fix from one branch to another using Claude as the AI engine. The [`auto_bug_fix`](auto_bug_fix/README.md) pipeline is git-first: it exhausts deterministic mechanisms (`git cherry-pick`, mechanical re-apply) before letting an LLM touch anything, audits every LLM edit against deterministic evidence, and escalates to an autonomous build-test-fix loop only as a last resort. The vendored [`adapter/`](adapter/README.md) adds a hybrid mode on top: a fast conflict resolver does the bulk work, a deterministic gate (zero LLM) decides whether it is good enough to ship, and the amalgamator is invoked seeded — with the prior resolver's output and the gate's findings — only when the gate fires.

## Examples and Guides

### GPU Performance Analysis

Detailed guides with copy-pasteable commands are in `llm/auto_analyze/examples/`:

| Guide | Description |
|-------|-------------|
| [Single-Trace Analysis and Annotation](llm/auto_analyze/examples/single_trace_analysis_and_annotation_guide.md) | Analyze a GPU trace and produce an annotated Perfetto visualization |
| [Cross-Commit Comparison](llm/auto_analyze/examples/cross_commit_comparison_guide.md) | Compare two commits of the same framework |
| [Cross-Framework Comparison](llm/auto_analyze/examples/cross_framework_comparison_guide.md) | Compare different frameworks (e.g., vLLM vs SGLang) |
| [vLLM Trace Generation](llm/auto_analyze/examples/vllm_generate_trace_example.md) | How to capture PyTorch profiling traces with vLLM |

### Bug Fix Porting

| Guide | Description |
|-------|-------------|
| [auto_bug_fix README](auto_bug_fix/README.md) | Git-first backport pipeline — phase-by-phase walkthrough, config reference, and example YAML configs |
| [Hybrid adapter](adapter/README.md) | Fast-resolver → deterministic gate → seeded amalgamator → arbiter; the vendored hybrid backport flow |
| [External-context contract](docs/external-context.md) | How the hybrid seeds the amalgamator: the Markdown format, per-phase prompt injection, and design invariants |

## Project Structure

```
ai_auto_analyze/
├── common/                             # Shared utilities
│   ├── llm_client.py                   #   Model-agnostic LLM completion (LiteLLM)
│   ├── agent_runner.py                 #   Agent dispatch: Claude SDK or LiteLLM tool loop
│   ├── claude_utils.py                 #   Claude Agent SDK wrapper
│   ├── utils.py                        #   Logging, directory cleanup, output dir helpers
│   └── convert_nsys_to_sqlite.sh       #   NSYS trace format converter
├── llm/                                # LLM use-case subtree (GPU inference frameworks) — see llm/README.md
│   ├── auto_analyze/                       # Analysis pipeline
│   │   ├── run_single_trace.py             #   Single-trace analysis entry point
│   │   ├── run_cross_trace.py              #   Cross-trace analysis entry point
│   │   ├── run_chrome_trace.py             #   Chrome trace JSON generation
│   │   ├── run_summary_pdf.py              #   PDF report generation
│   │   ├── run_jiras.py                    #   JIRA task creation
│   │   ├── create_single_trace_config.py   #   Helper: generate single-trace config JSON
│   │   ├── create_cross_trace_config.py    #   Helper: generate cross-trace config JSON
│   │   ├── configs/                        #   Config dataclasses and examples
│   │   │   ├── single_trace_config.py
│   │   │   ├── single_trace_config_example.json
│   │   │   ├── cross_trace_config.py
│   │   │   ├── cross_trace_config_example.json
│   │   │   └── claude_config.json
│   │   ├── prompts/                        #   Prompt templates for all analysis steps
│   │   │   ├── single_trace_prompts.py
│   │   │   ├── cross_trace_prompts.py
│   │   │   ├── chrome_trace_prompts.py
│   │   │   ├── jira_prompts.py
│   │   │   └── summary_pdf_prompts.py
│   │   └── examples/                       #   Guides and example runs
│   ├── auto_profile/                       # Profiling orchestration
│   │   ├── run_profile_core.sh             #   Main profiling script
│   │   ├── run_profile_summary.py          #   Parse results into analysis configs
│   │   ├── parse_run_config.py             #   Config parser and validator
│   │   └── test_configs/                   #   JSON configs (infra, run, docker, GPUs)
│   ├── auto_code_gen/                      # AI-based code generation pipeline
│   │   ├── run_code_gen.py                #   Pipeline orchestrator (all use cases)
│   │   ├── code_gen_prompts.py            #   Shared prompt classes
│   │   ├── code_gen_configs.py            #   PipelineConfig, CodeGenConfig
│   │   ├── use_cases/                     #   Pluggable use case definitions
│   │   │   ├── base.py                    #     UseCase ABC
│   │   │   └── llm_framework.py           #     LLM framework optimization porting
│   │   ├── configs/                       #   Example JSON configs
│   │   ├── run_runtime_iters.py           #   Standalone runtime iteration runner
│   │   ├── run_investigate_issue.py
│   │   ├── run_fix_issue.py
│   │   ├── run_work_items.py
│   │   └── run_summary.py
│   ├── auto_profile_server_client/     #   Framework-aware profiling client (docker/K8s)
│   └── auto_config/                    #   vLLM/SGLang model inventory → run configs
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
├── adapter/                            # Hybrid backport adapter
│   ├── hybrid-backport.sh             #   Driver: resolver → gate → seeded amalgamator → arbiter
│   └── check-backport.py              #   Deterministic gate + external-context emitter
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

## Model-Agnostic LLM Support

The bug-fix pipeline supports Claude, OpenAI, Ollama, and OpenAI-compatible
servers such as vLLM through named model profiles:

- **`models.yaml`** defines reusable model IDs, runner selection, endpoints,
  credentials, and parameters.
- **`case.yaml`** describes the repository, fix, target branches, and commands,
  then selects a profile by ID. The case file can have any `.yaml`, `.yml`, or
  `.json` filename.

### Create the two configuration files

From the repository root:

```bash
mkdir -p my-backport
cp auto_bug_fix/configs/example_models.yaml my-backport/models.yaml
cp auto_bug_fix/configs/example_backport.yaml my-backport/case.yaml
# Edit models.yaml for your provider and case.yaml for your repository and fix.
python -m auto_bug_fix.run_bug_fix my-backport/case.yaml
```

The templates use `m1` for Claude Sonnet, `m2` for a local OpenAI-compatible
server, and `m3` for Claude Opus. Change the case selection to use another profile:

```yaml
model: m2
advanced:
  phase_4b_model: m3  # optional: use a different complete profile for recovery
```

See the **[complete models.yaml and case.yaml guide](auto_bug_fix/README.md#quickstart)**
for full examples, authentication, field references, patch input, and multiple targets.

### Discovery and defaults

The first registry selected is used; registries are not merged:

1. An explicit `--models PATH`.
2. A file named exactly `models.yaml` beside the case file.
3. `~/.config/amalgamator/models.yaml`.

```bash
python -m auto_bug_fix.run_bug_fix my-backport/case.yaml --models /path/to/shared-models.yaml
```

An explicit missing or invalid registry is an error. If the case omits `model`,
the registry's required `default` profile is selected. Without a registry, the
fallback is `claude-sonnet-4-6`. Phase 4b inherits the pipeline's complete profile
unless `advanced.phase_4b_model` overrides it.

Add top-level `strict: true` to `models.yaml` to reject unregistered model/profile
selections at load time, including Phase 4b overrides. By default, unregistered
strings remain supported as bare model names and produce a warning.

### Provider routing and compatibility

`provider: anthropic` selects the Claude Agent SDK. `provider: litellm` selects
the generic tool-use runner, including when calling an Anthropic model through
LiteLLM. Use a model that supports tool/function calling for the generic runner.
Credentials are specific to the selected provider; Anthropic credentials are
needed only when using that provider.

Profiles put generation options under **`params`**. These apply to LiteLLM calls,
including context summarization; they are not arbitrary Claude SDK options.
Keep `model`, `provider`, `api_base`, `api_key`, and `api_key_file` beside `params`.

Older cases with a bare model name and inline `api_base`, `api_key`/`api_key_file`,
and **`model_params`** remain supported. Named selections use the profile's own
settings and warn when inline model settings are ignored. See
[selection and compatibility rules](auto_bug_fix/README.md#model-selection-and-legacy-cases)
for precedence and Phase 4b overrides.

The [backport scanner](backport_scanner/README.md#model-selection) also supports
the shared registry and `--models`. Select its profile with `llm.model`; semantic
checks and command discovery use that profile, and generated cases retain the
selection with a command pointing to the same registry.

### Cost Tracking

Each query records its effective model and profile ID. A provider-reported cost,
including zero, takes precedence. Otherwise, the tracker estimates from token
counts: Ollama and `openai/` profiles with a custom endpoint are classified as
self-hosted with zero token cost, known cloud models use their pricing table,
and unknown cloud models use a fallback estimate. The summary appears at the
end of each run.

### Large-context diff limits

An explicit `advanced.context_length` scales diff limits above the historical
200,000/50,000-character floors. Full-diff growth begins around 400K configured
tokens; per-chunk growth begins around 416.7K. A 262K-token configuration retains
both existing limits. See [context and diff budgets](auto_bug_fix/README.md#context-and-diff-budgets)
for formulas and examples.

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

Backport a bug fix from one branch to another using a configured model profile. Works for any project with a git repository and shell-invokable build and test commands (tested on Go and C codebases).

See **[auto_bug_fix/README.md](auto_bug_fix/README.md)** for the full setup and usage guide.

**Quick start:**

```bash
# 1. Create models.yaml and case.yaml (see the paired templates above)
# 2. Run
python -m auto_bug_fix.run_bug_fix path/to/case.yaml
```

### Hybrid mode (fast resolver + gate + seeded amalgamator)

The repo also ships the **hybrid backport adapter** in
[`adapter/`](adapter/README.md), which wraps `auto_bug_fix` in a
faster, deterministic-first flow:

```
cherry-pick → conflict resolver (pluggable; tested: synthmerge)
           → deterministic gate (check-backport.py, zero LLM)
           → clean: ship            (~30 s typical)
           → findings: amalgamator, seeded with the prior resolver's
             output + the findings (external_context_file)
           → deterministic arbiter: keep or reject the escalation
```

```bash
# from this checkout (--amalgamator-root defaults to the repo root):
./adapter/hybrid-backport.sh \
    --repo /path/to/repo --upstream <fix-sha> \
    --synthmerge-config endpoints.yaml --synthmerge-cache /var/hybrid/lmdb \
    --python <python with this repo's deps> \
    --model openai/<model-id> --api-base https://... --api-key-file key
```

Validated on a 400-case kernel-backport evaluation (same model at every
stage): 42.5% combined exact/whitespace match against human results for
the hybrid vs 35.5% for the fast resolver alone. Contract and design:
**[docs/external-context.md](docs/external-context.md)**.
