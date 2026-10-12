# AI Auto Performance Analysis & Code Generation

## Project Overview

This project automates GPU performance analysis, AI-driven code generation, and bug fix backporting using the Claude Agent SDK (or any LiteLLM tool-calling endpoint for codegen). It consists of three interconnected pipelines.

## Pipelines

### 1. Performance Analysis (`auto_profile` + `auto_analyze`)

GPU profiling and trace analysis across inference frameworks (vLLM, SGLang, TRT-LLM).

**Steps:**
1. `auto_profile` — run benchmarks, capture GPU traces
2. `llm.auto_profile.run_profile_summary` — parse results, generate analysis configs
3. `llm.auto_analyze.run_single_trace` — analyze individual framework traces
4. `llm.auto_analyze.run_cross_trace` — compare traces across frameworks or commits
5. `llm.auto_analyze.run_chrome_trace` — generate Perfetto-viewable Chrome traces
6. `llm.auto_analyze.run_summary_pdf` — generate PDF reports
7. `llm.auto_analyze.run_jiras` — create JIRA tasks from improvement plans

### 2. Code Generation (`auto_code_gen`)

AI code generation with iterative generate+review convergence loops.

**One mode:**
- **LLM Framework** (`use_case: llm_framework`) — port GPU kernel optimizations between frameworks

Bug fix backporting lives in `auto_bug_fix` (pipeline 3), not here.

**Phases:** Code traces → Port plan → Test plan → Code gen → Runtime iterations

**Agent backends:** run on Claude (Agent SDK, default) or any LiteLLM tool-calling endpoint by adding an `agent` section to the pipeline JSON or passing `--agent-config`. See `llm/auto_code_gen/agents/README.md` and `llm/auto_code_gen/configs/agent_vllm_example.json`.

**Upstream PR workflow:** after runtime success, `llm/auto_code_gen/pr_flow/` turns the patch into a PR against vllm-project/vllm and drives the review loop (human confirms every remote operation); `llm/auto_code_gen/pr_review/` fixes review comments.

### 3. Bug Fix Backport (`auto_bug_fix`)

Git-first backporting pipeline: tries `git cherry-pick` first, falls back to LLM resolution.

**Phases:** Triage → Semantic triage → Prerequisites → Test port → Baseline → Cherry-pick → Resolve → Verify → Equivalence

## Skills (Slash Commands)

All pipeline entry points are available as Claude Code skills:

| Skill | Pipeline | Description |
|-------|----------|-------------|
| `/profile` | Perf | Run GPU profiling benchmarks |
| `/parse` | Perf | Parse profiling results, generate analysis configs |
| `/analyze` | Perf | Run trace analysis (single-trace or cross-trace) |
| `/perf-pipeline` | Perf | Full end-to-end performance pipeline |
| `/codegen` | CodeGen | AI code generation pipeline |
| `/runtime` | CodeGen | Standalone runtime iterations |
| `/investigate` | CodeGen | Post-pipeline issue investigation |
| `/fix` | CodeGen | Post-pipeline issue fixing |
| `/work-items` | CodeGen | Execute arbitrary work items |
| `/bugfix` | BugFix | Git-first bug fix backport pipeline |
| `/chrome-trace` | Util | Generate Chrome trace visualization |
| `/summary` | Util | Generate summary reports (PDF/PPTX) |
| `/jiras` | Util | Create JIRA tasks from analysis |
| `/create-config` | Util | Generate config files interactively |

## Project Structure

```
common/             Shared utilities (claude_utils.py, utils.py)
llm/                LLM use-case subtree (GPU inference frameworks) — see llm/README.md
  auto_profile/         Benchmark orchestration and GPU profiling
  auto_analyze/         GPU trace analysis pipeline
    configs/            Config dataclasses + examples
    prompts/            LLM prompt templates
  auto_code_gen/        AI code generation pipeline
    use_cases/          UseCase ABC + implementations (llm_framework)
    agents/             LiteLLM tool-calling agent runtime
    pr_flow/            Human-in-the-loop upstream PR workflow
    pr_review/          Review-comment fixing workflow
    configs/            Config examples
  auto_profile_server_client/  Framework-aware profiling client (docker/K8s)
  auto_config/          vLLM/SGLang model inventory → run config generation
auto_bug_fix/       Git-first bug fix backport pipeline
  configs/          YAML config examples
tests/              Test suite (pytest)
```

## Configuration

- **Performance analysis:** JSON configs in `llm/auto_profile/test_configs/` and `llm/auto_analyze/configs/` (one example chain: the `deepseek_r1_nvfp4` campaign flows profile → single/cross-trace analysis → codegen)
- **Code generation:** JSON configs in `llm/auto_code_gen/configs/` (`use_case: llm_framework`; agent backends via `agent` section, see `agent_vllm_example.json`)
- **Bug fix:** YAML configs in `auto_bug_fix/configs/`

## Dependencies

- Python >= 3.12
- claude-agent-sdk >= 0.2.87
- litellm >= 1.100.1, < 2 (agent backends)
- colorama, pyyaml

## Environment

Set `ANTHROPIC_API_KEY` in your environment.

## Running Tests

```bash
pytest
```
