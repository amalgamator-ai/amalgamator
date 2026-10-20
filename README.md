<img src="art/main-amalgamator-logo.png" alt="Amalgamator logo" width="50%">

# Amalgamator

Comparing and optimizing GPU inference frameworks, and keeping fixes in sync across branches and releases, are both repetitive, evidence-heavy tasks that are tedious to do by hand and easy to get wrong at scale. Amalgamator automates the mechanical parts with an LLM agent, while checking every step against deterministic evidence — git history, build output, test results — so it fails closed to a human instead of guessing. It covers two kinds of engineering work:

- **GPU performance analysis** — profile, analyze, and compare GPU traces of LLM inference frameworks (vLLM, SGLang, TensorRT-LLM), and port optimizations between them.
- **Bug-fix backporting** — port a fix from one branch to another. The pipeline is *git-first*: it exhausts deterministic mechanisms (`git cherry-pick`, mechanical re-apply) before letting an LLM touch anything, audits every LLM edit against deterministic evidence, and escalates to an autonomous build-test-fix loop only as a last resort. It works for any project with a git repository and shell-invokable build and test commands (tested on Go and C codebases).

**Who this is for:** any team comparing or optimizing GPU inference frameworks, and any team maintaining fixes across branches or releases with a shell-invokable build and test command. Nothing here assumes a specific codebase, language, or organization.

GPU analysis tools use Claude. Backporting runs on Claude, OpenAI, Ollama, or any OpenAI-compatible server such as vLLM, through named model profiles.

## Scope

Amalgamator covers the two use cases above, each backed by one or more pipelines:

1. **GPU performance analysis** — profiling, single/cross-trace analysis, and optimization porting between inference frameworks (`llm/auto_profile`, `llm/auto_analyze`, `llm/auto_code_gen`).
2. **Bug-fix backporting** — porting a fix commit to another branch, and finding which branches, tags, or releases are missing it (`auto_bug_fix`, `backport_scanner`).

### Out of scope

- **No autonomous merging without human review.** Every pipeline produces a reviewable commit, patch, or PR for a human to confirm — it does not push or merge on its own.
- **Does not generate net-new test cases.** The pipelines verify against the build and test commands you provide; they don't write new tests to cover gaps in your existing suite.
- **No cross-agent-platform capability porting.** Scope is strictly the two use cases above.

## Pipelines

Two use cases, four pipelines — GPU performance analysis spans profiling/analysis and code generation; bug-fix backporting spans the backport pipeline and the scanner that feeds it:

| Pipeline | Package | What it does | Docs |
|----------|---------|--------------|------|
| GPU performance analysis | `llm/auto_profile`, `llm/auto_analyze` | Profiles inference frameworks, analyzes single traces, and compares traces across commits or frameworks | [llm/auto_analyze/README.md](llm/auto_analyze/README.md) |
| Code generation | `llm/auto_code_gen` | Ports performance optimizations between inference frameworks with generate-and-review loops | [llm/auto_code_gen/README.md](llm/auto_code_gen/README.md) |
| Bug-fix backporting | `auto_bug_fix` | Backports a fix commit to another branch and produces a reviewable commit, a patch file, and a dossier of what it did and why | [auto_bug_fix/README.md](auto_bug_fix/README.md) |
| Backport scanner | `backport_scanner` | Finds which branches, tags, and releases are missing a fix, and generates backport configs for the pipeline above | [backport_scanner/README.md](backport_scanner/README.md) |

## Supported platforms

**Operating systems:** Linux and macOS — the full test suite runs on both (Linux in CI, macOS in development). Amalgamator itself is pure Python plus `git` and runs on an ordinary workstation; the GPU benchmarks it drives run on GPU machines. Windows is not tested and not supported.

**Python:** 3.12 or newer. CI tests the floor (3.12) and a current release (3.14). Raising the floor is a breaking change and, while the project is pre-1.0, lands in a minor release — see the [release policy](CONTRIBUTING.md#releases-and-versioning).

Per-pipeline requirements beyond that:

| Pipeline | Extra requirements |
|----------|--------------------|
| GPU performance analysis | Claude; traces from vLLM, SGLang, or TensorRT-LLM in `.nsys-rep`, `.sqlite`, `.json`, or `.json.gz` form |
| Code generation | The GitHub CLI (`gh`) for the PR flow; benchmark validation runs on GPU machines |
| Bug-fix backporting | Any git repository whose build and test commands run from a shell, and any LLM endpoint from [Prerequisites](#prerequisites) |
| Backport scanning | Only the git repository being scanned; generating a backport config from the scan results additionally uses an LLM |

Notes:

- Analyzing a `.nsys-rep` trace needs the `nsys` CLI (Nsight Systems) for conversion; the other trace formats do not.
- `git` is required everywhere. `git-lfs` is only needed to commit trace files — `*.nsys-rep` and `*.json.gz` are LFS-tracked — not to clone the repository.

## Getting Started

### Prerequisites

- Python 3.12+ (use [uv](https://docs.astral.sh/uv/) for version management)
- An LLM endpoint — Claude via Anthropic or Vertex AI, a local Ollama instance, a self-hosted vLLM server, or an OpenAI API key. The backporting pipeline works with any of these; the GPU analysis tools need Claude.
- For GPU analysis: framework source code and GPU trace files

### Install

Install from source:

```bash
git clone https://github.com/redhat-et/amalgamator.git
cd amalgamator
uv sync
```

## Quick Start: GPU Trace Analysis

Analyze one framework's GPU trace into an annotated trace you can open in [Perfetto](https://ui.perfetto.dev):

```bash
python -m llm.auto_analyze.run_single_trace --config /path/to/config.json
```

Configs are generated by helper scripts, and cross-commit and cross-framework comparison builds on single-trace results. See **[llm/auto_analyze/README.md](llm/auto_analyze/README.md)** for the complete workflow and the step-by-step [guides](llm/auto_analyze/README.md#guides).

## Quick Start: Backport a Fix

Create a model registry (`models.yaml`) and a case file (`case.yaml`) from the templates, then run the pipeline:

```bash
mkdir -p my-backport
cp auto_bug_fix/configs/example_models.yaml my-backport/models.yaml
cp auto_bug_fix/configs/example_backport.yaml my-backport/case.yaml
# Edit models.yaml for your provider and case.yaml for your repository and fix.
python -m auto_bug_fix.run_bug_fix my-backport/case.yaml
```

- **`models.yaml`** defines reusable model profiles: model IDs, runner selection, endpoints, credentials, and parameters.
- **`case.yaml`** describes the repository, the fix, the target branches, and the build and test commands, and selects a profile by ID. The case file can have any `.yaml`, `.yml`, or `.json` filename.

The templates define `m1` (Claude Sonnet), `m2` (a local OpenAI-compatible server), and `m3` (Claude Opus). Select another profile in the case file:

```yaml
model: m2
advanced:
  phase_4b_model: m3  # optional: use a different complete profile for recovery
```

The registry is discovered in this order, and registries are not merged: an explicit `--models PATH`, then a `models.yaml` beside the case file, then `~/.config/amalgamator/models.yaml`.

The full setup is in **[auto_bug_fix/README.md](auto_bug_fix/README.md)**:

- [Quickstart with complete `models.yaml` and `case.yaml` examples](auto_bug_fix/README.md#quickstart)
- [Model selection and legacy cases](auto_bug_fix/README.md#model-selection-and-legacy-cases)
- [Context and diff budgets](auto_bug_fix/README.md#context-and-diff-budgets)

To find out *which* branches and tags still need a fix before you backport it, use the [backport scanner](backport_scanner/README.md); it shares the same model registry via `--models` and can generate Amalgamator cases from its results:

```bash
python -m backport_scanner.run_scanner --config my_scan.yaml
```

## Cost Tracking

Each query records its effective model and profile ID. A provider-reported cost, including zero, takes precedence. Otherwise, the tracker estimates from token counts: Ollama and `openai/` profiles with a custom endpoint are classified as self-hosted with zero token cost, known cloud models use their pricing table, and unknown cloud models use a fallback estimate. The summary appears at the end of each run.

## Repository Layout

```
amalgamator/
├── llm/                   # GPU performance analysis, profiling, and code-gen pipelines (see llm/README.md)
├── auto_bug_fix/          # Git-first bug-fix backport pipeline
├── backport_scanner/      # Find branches/tags missing a fix; generate backport cases
├── common/                # Shared utilities: LLM client, agent runner, logging
├── tests/                 # pytest suite
├── run_all.sh             # Full profiling + analysis orchestrator
└── run_all_scheduled.sh   # Scheduled orchestration with GPU availability checks
```

## Project Background

**Sponsoring organization:** Red Hat.

**Strategic objective:** reduce the manual engineering toil of backporting fixes (including CVE fixes) across branches and releases, and of comparing and optimizing GPU inference frameworks — built as a reusable piece of a broader agentic software development lifecycle.

### Upstream Dependencies

- [git](https://git-scm.com/) — backporting is git-first; `git cherry-pick` and mechanical re-apply are tried before any LLM involvement.
- [Claude Agent SDK](https://docs.claude.com/en/api/agent-sdk/overview) — the GPU analysis tools, and the default backporting agent runner.
- [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python) — direct one-shot model calls, with optional Vertex support.
- [LiteLLM](https://docs.litellm.ai/) — the generic tool-use loop backporting uses for OpenAI, Ollama, and other OpenAI-compatible endpoints.
- [PyYAML](https://pyyaml.org/) — parsing of case, model, and scan configurations.
- [colorama](https://github.com/tartley/colorama) — terminal output coloring.
- Python 3.12+, managed via [uv](https://docs.astral.sh/uv/).

### Success Metrics

The health signal for any single run is the same gate the pipeline itself enforces: **the build and test commands you provide pass** against the backported fix or ported optimization (the pipeline's own verification gate) before anything is handed to a human for review. As the project grows in the open, we'll also track community health indicators (issue/PR velocity, external contributions) — no specific targets committed yet.

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).
