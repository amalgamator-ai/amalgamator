# AI Auto Performance Analysis & Code Generation

## Project Overview

This project automates GPU performance analysis, AI-driven code generation, and bug fix backporting using the Claude Agent SDK. It consists of three interconnected pipelines.

## Pipelines

### 1. Performance Analysis (`auto_profile` + `auto_analyze`)

GPU profiling and trace analysis across inference frameworks (vLLM, SGLang, TRT-LLM).

**Steps:**
1. `auto_profile` — run benchmarks, capture GPU traces
2. `auto_profile.run_profile_summary` — parse results, generate analysis configs
3. `auto_analyze.run_single_trace` — analyze individual framework traces
4. `auto_analyze.run_cross_trace` — compare traces across frameworks or commits
5. `auto_analyze.run_chrome_trace` — generate Perfetto-viewable Chrome traces
6. `auto_analyze.run_summary_pdf` — generate PDF reports
7. `auto_analyze.run_jiras` — create JIRA tasks from improvement plans

### 2. Code Generation (`auto_code_gen`)

AI code generation with iterative generate+review convergence loops.

**Two modes:**
- **LLM Framework** (`use_case: llm_framework`) — port GPU kernel optimizations between frameworks
- **Bug Fix** (`use_case: bug_fix`) — port bug fixes between branches of C/systems projects

**Phases:** Code traces → Port plan → Test plan → Code gen → Runtime iterations

### 3. Gitted Bug Fix (`auto_bug_fix`)

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
| `/bugfix` | BugFix | Gitted bug fix pipeline |
| `/chrome-trace` | Util | Generate Chrome trace visualization |
| `/summary` | Util | Generate summary reports (PDF/PPTX) |
| `/jiras` | Util | Create JIRA tasks from analysis |
| `/create-config` | Util | Generate config files interactively |

## Project Structure

```
common/              Shared utilities (claude_utils.py, utils.py)
auto_profile/        GPU profiling orchestration
auto_analyze/        GPU trace analysis pipeline
  configs/           Config dataclasses + examples
  prompts/           LLM prompt templates
auto_code_gen/       AI code generation pipeline
  use_cases/         UseCase ABC + implementations (llm_framework, bug_fix)
  configs/           Config examples
auto_bug_fix/        Git-first bug fix pipeline
  configs/           YAML config examples
tests/               Test suite (pytest)
```

## Configuration

- **Performance analysis:** JSON configs in `auto_profile/test_configs/` and `auto_analyze/configs/`
- **Code generation:** JSON configs in `auto_code_gen/configs/` (set `use_case` to `llm_framework` or `bug_fix`)
- **Bug fix:** YAML configs in `auto_bug_fix/configs/`

## Dependencies

- Python >= 3.14
- claude-agent-sdk >= 0.2.87
- colorama, pyyaml

## Environment

Source `env.sh` before running shell scripts. Set `ANTHROPIC_API_KEY` in your environment.

## Running Tests

```bash
pytest
```
