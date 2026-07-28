# auto_code_gen — AI Code Generation Pipeline

Unified pipeline for AI-driven code porting. Uses Claude (via the Agent SDK) to trace code, plan the port, generate patches, review them iteratively, and drive runtime validation — all without manual intervention.

Built on a pluggable use-case architecture. One use case ships today:
- **LLM Framework** — port performance optimizations between GPU inference frameworks (vLLM, SGLang)

Bug-fix backporting lives in its own package, [`auto_bug_fix`](../auto_bug_fix/README.md).

## Pipeline

Every use case follows the same five-phase pipeline. The orchestration (iterations, convergence detection, resume, timing) is generic; only the prompt content differs.

```
Phase 1: Code Trace
         Analyze what code is relevant and document it.

Phase 2: Code Port Plan + Review Iterations
         Plan how to port the code. Each iteration: generate plan -> review -> fix.
         Stops early when the review finds no issues (CONVERGED).

Phase 3: Test Plan + Review Iterations
         Plan the tests. Same generate -> review -> converge loop.
         (Skipped when the use case folds the test plan into Phase 2.)

Phase 4: Code Gen + Review Iterations
         Generate the code patch (with tests). Same generate -> review -> converge loop.

Phase 5: Runtime Iterations
         Validate the generated code at runtime.
         LLM framework: apply patch, run benchmark, investigate failures, fix, repeat.
```

Any phase's review+iterations can be skipped via `code_port_plan_skip_review`, `test_plan_skip_review`, or `code_gen_skip_review` flags for faster runs.

## Quick Start

### 1. Create a JSON config

LLM framework — see [`configs/code_gen_config_example.json`](configs/code_gen_config_example.json).

### 2. Run

```bash
python -m auto_code_gen.run_code_gen --config <path_to_config.json>
```

Resume from where a previous run stopped:

```bash
python -m auto_code_gen.run_code_gen --config <path_to_config.json> --resume
```

The `use_case` field in the JSON determines which pipeline variant runs. No separate entry points needed.

## Use Cases

### LLM Framework (`"use_case": "llm_framework"`)

Ports performance optimizations from a faster framework (source) to a slower one (target). Designed for GPU/CUDA workloads with execution mode awareness (decode-only, prefill-only, mixed).

**Code trace** analyzes both frameworks' code paths for the improvement plan step.

**Code gen** includes a feature toggle mechanism: an environment variable to disable the new feature and a startup log message to confirm activation.

**Runtime iterations** apply the patch, run benchmarks with GPU polling, compare performance against a baseline (with the feature disabled), and optionally run lm_eval correctness checks.

Config fields specific to this use case:

| Field | Description |
|---|---|
| `cross_trace_config` | Path to the cross-trace analysis config (from `auto_analyze`) |
| `improvement_id` | Which improvement step to implement |
| `source_code_dir` | Path to the target framework source code |
| `num_runtime_iterations` | Max runtime iteration attempts (default: 10) |
| `use_smaller_model_for_runtime` | Find a smaller model for faster runtime testing |
| `disable_new_feature_for_runtime` | Run baseline comparison with feature disabled |

## Common Config Fields

These fields are shared across all use cases:

| Field | Default | Description |
|---|---|---|
| `output_dir` | — | Directory for all generated artifacts |
| `num_code_port_plan_iterations` | 3 | Max code port plan iterations (generate + review each) |
| `num_test_plan_iterations` | 3 | Max test plan iterations |
| `num_code_gen_iterations` | 3 | Max code gen iterations |
| `disallowed_modules` | [] | Files/directories that must not be modified |
| `code_port_plan_skip_review` | false | Skip review step — run one generate pass, no iterations |
| `test_plan_skip_review` | false | Skip test plan review |
| `code_gen_skip_review` | false | Skip code gen review |
| `thinking-mode` | "deep" | `"deep"` (Opus, max effort) or `"normal"` (Sonnet, medium effort) |

## Architecture

```
auto_code_gen/
├── run_code_gen.py              # Pipeline orchestrator (all use cases)
├── run_runtime_iters.py         # Standalone runtime iteration runner
├── code_gen_prompts.py          # Shared prompt classes (generic, template-injected)
├── code_gen_configs.py          # PipelineConfig, CodeGenConfig
├── use_cases/
│   ├── base.py                  # UseCase ABC
│   └── llm_framework.py         # LLM framework templates + step generators
├── configs/
│   └── code_gen_config_example.json
├── run_investigate_issue.py     # Post-pipeline issue investigation
├── run_fix_issue.py             # Post-pipeline issue fixing
├── run_work_items.py            # Arbitrary work item execution
└── run_summary.py               # PowerPoint summary generation
```

**Design pattern:** The pipeline orchestrator (`run_code_gen.py`) defines the phase structure. Each `UseCase` subclass provides domain-specific prompt templates and step generators. The prompt classes in `code_gen_prompts.py` are generic — they accept the template text at construction time, so the same class works for any use case's content.

## Output Files

All output is written to `output_dir`. File names follow a consistent pattern:

| Pattern | Description |
|---|---|
| `*_code_trace.txt` | Code trace analysis |
| `code_port_plan_V{N}.txt` / `_summary_V{N}.txt` | Code port plan iteration N |
| `code_port_plan_review_V{N}.txt` / `_summary_V{N}.txt` | Review of iteration N |
| `test_plan_V{N}.txt` / `_summary_V{N}.txt` | Test plan iteration N |
| `code_gen_V{N}.patch` / `_summary_V{N}.txt` | Code gen patch iteration N |
| `code_gen_review_V{N}.patch` / `_summary_V{N}.txt` | Reviewed patch iteration N |
| `code_gen_runtime_V{N}.patch` | Fixed patch from runtime iteration N |
| `runtime_success.txt` | Written on successful runtime validation |

## On Failure

If the runtime phase exhausts its retries, it stops with a summary and exits.

Manual escalation:

```bash
# Deeper investigation of a specific issue
python -m auto_code_gen.run_investigate_issue --config <config.json> ...

# Direct fix attempt
python -m auto_code_gen.run_fix_issue --config <config.json> ...
```

Or increase `num_runtime_iterations` and re-run with `--resume`.
