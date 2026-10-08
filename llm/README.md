# llm/ — The LLM Use-Case Subtree

Everything specific to **GPU inference serving frameworks** (vLLM, SGLang,
TensorRT-LLM) lives in this directory: profiling them, analyzing their GPU
traces, and generating optimization patches for them. The repository's other
use-case — bug-fix backporting for C/systems projects — is deliberately kept
out of this subtree (see `auto_bug_fix/` and `backport_scanner/` at the
repository top level), and use-case-neutral code lives in `common/`.

## The Pipeline

The packages form one end-to-end flow, each stage producing the config or
artifact the next stage consumes:

```
auto_config                auto_profile / auto_profile_server_client
     │                            │
     │ model inventory            │ benchmarks + nsys/PyTorch traces
     ▼                            ▼
             campaign run configs  (auto_profile/test_configs/)
                            │
                            ▼
                       auto_analyze
                            │  single-trace kernel↔source correlation,
                            │  cross-trace comparison, improvement plans
                            ▼
                       auto_code_gen
                            │  port the planned optimization,
                            │  iterate until runtime gates pass
                            ▼
                       pr_flow / pr_review
                               upstream PR against vllm-project/vllm
```

Each stage is also usable standalone — every box has its own entry points
and configs.

## The Example Chain

One example campaign — DeepSeek-R1-NVFP4 on B200, vLLM vs SGLang — flows
through all three stages, and each stage's example config points at the
previous stage's output:

1. **Profile:** `auto_profile/test_configs/deepseek_r1_nvfp4/run_deepseek_r1_nvfp4.json`
   — benchmarks vLLM and SGLang with trace capture; `run_all.sh` drives
   profile → parse → analyze end to end
2. **Analyze:** `auto_analyze/configs/single_trace_config_example.json`
   (one per framework) and `cross_trace_config_example.json` (compares the
   two single-trace results)
3. **Codegen:** `auto_code_gen/configs/code_gen_config_example.json` —
   reads the cross-trace output and ports the chosen improvement;
   `agent_vllm_example.json` shows the model-agnostic agent backend

## Directory Guide

| Path | What it is |
|------|------------|
| `auto_config/` | Fetch vLLM/SGLang model and quantization inventories and generate run configs for the profiling pipeline (`python -m llm.auto_config`). |
| `auto_profile/` | Docker-based benchmark orchestration: stands up framework servers, drives load, captures nsys/PyTorch traces (`run_profile_core.sh`), then parses results into analysis configs (`run_profile_summary.py`). |
| `auto_profile_server_client/` | Framework-aware profiling client with docker and Kubernetes Job backends: launches vLLM/SGLang/TRT servers, runs benchmarks, collects profiles and metrics, analyzes run-to-run stability. |
| `auto_analyze/` | Trace analysis: single-trace (correlate GPU kernels to transformer-block operations, annotated Perfetto view), cross-trace (compare commits or frameworks, root-cause gaps, improvement plans), plus PDF reports and JIRA creation. Guides in `auto_analyze/examples/`. |
| `auto_code_gen/` | The code-generation pipeline (`use_case: llm_framework`): code trace → port plan → test plan → code gen → runtime iterations, with iterative review/convergence loops. See `auto_code_gen/README.md`. |
| `auto_code_gen/agents/` | Provider-neutral agent runtime: run the pipeline on Claude (Agent SDK) or any LiteLLM tool-calling endpoint via an `agent` config section. See `auto_code_gen/agents/README.md`. |
| `auto_code_gen/pr_flow/` | Human-in-the-loop workflow that turns a verified patch into an upstream PR and drives the review loop. The AI never touches git or GitHub — every remote operation is a script that announces its commands and waits for a human `yes`. See `auto_code_gen/pr_flow/README.md`. |
| `auto_code_gen/pr_review/` | Fixes review comments on an open PR through verified fix iterations. |
| `agents.py` | Stable facade for provider-neutral agent execution; re-exports from `auto_code_gen/agents/` so LLM-internal callers have a single import site. |

## Running

All entry points are `llm.`-prefixed modules, run from the repository root:

```bash
# Full code-generation pipeline (resume with --resume)
python -m llm.auto_code_gen.run_code_gen --config <config.json>

# Standalone runtime iterations
python -m llm.auto_code_gen.run_runtime_iters --config <config.json>

# Trace analysis
python -m llm.auto_analyze.run_single_trace --config <config.json>
python -m llm.auto_analyze.run_cross_trace --config <config.json>

# Profiling
./llm/auto_profile/run_profile_core.sh <run_config.json>

# Model inventory → run configs
python -m llm.auto_config
```

Claude Code skills (`/profile`, `/analyze`, `/codegen`, `/runtime`, ...) wrap
these entry points; see the repository `CLAUDE.md` for the full list.

## Configuration

- **Profiling:** `auto_profile/test_configs/` — shared infra files plus the
  `deepseek_r1_nvfp4` example campaign
- **Trace analysis:** `auto_analyze/configs/`
- **Code generation:** `auto_code_gen/configs/` (`use_case: llm_framework`;
  agent backends via an `agent` section — see `agent_vllm_example.json`)

## Outside this subtree

- `common/` — shared utilities (Claude SDK glue, LiteLLM completion, logging,
  config helpers); nothing in `common/` imports from `llm/`
- `auto_bug_fix/`, `backport_scanner/` — the bug-fix backport use-case
- `tests/llm/` — tests for the packages above
