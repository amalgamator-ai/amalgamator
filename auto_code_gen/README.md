# auto_code_gen — AI Code Generation Pipeline

Unified pipeline for AI-driven code porting. Uses Claude (via the Agent SDK) to trace code, plan the port, generate patches, review them iteratively, and drive runtime validation — all without manual intervention.

Supports multiple use cases through a pluggable architecture:
- **LLM Framework** — port performance optimizations between GPU inference frameworks (vLLM, SGLang)
- **Bug Fix** — port a bug fix from one branch of a C/systems project (gcc, openssl, glibc, curl) to another

## Pipeline

Every use case follows the same five-phase pipeline. The orchestration (iterations, convergence detection, resume, timing) is identical; only the prompt content differs.

```
Phase 1: Code Trace
         Analyze what code is relevant and document it.

Phase 2: Code Port Plan + Review Iterations
         Plan how to port the code. Each iteration: generate plan -> review -> fix.
         Stops early when the review finds no issues (CONVERGED).

Phase 3: Test Plan + Review Iterations
         Plan the tests. Same generate -> review -> converge loop.
         (Skipped when combined with Phase 2 via use_combined_code_and_test_port_plan.)

Phase 4: Code Gen + Review Iterations
         Generate the code patch (with tests). Same generate -> review -> converge loop.

Phase 5: Runtime Iterations
         Validate the generated code at runtime.
         LLM framework: apply patch, run benchmark, investigate failures, fix, repeat.
         Bug fix: autonomous build-test-fix loop until clean or retry limit.
```

Any phase's review+iterations can be skipped via `code_port_plan_skip_review`, `test_plan_skip_review`, or `code_gen_skip_review` flags for faster runs.

## Quick Start

### 1. Create a JSON config

LLM framework — see [`configs/code_gen_config_example.json`](configs/code_gen_config_example.json).

Bug fix — see [`configs/bug_fix_config_example.json`](configs/bug_fix_config_example.json).

Bug fix (fast — skips all reviews, uses combined plan, normal thinking mode) — see [`configs/bug_fix_config_fast_example.json`](configs/bug_fix_config_fast_example.json).

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

### Bug Fix (`"use_case": "bug_fix"`)

Ports a bug fix from one branch to another in a C/systems project. Given a source branch, target branch, and the commit SHA that introduced the fix, Claude traces what the fix touches, plans and generates the ported patch (including ported tests), then drives an autonomous build-test-fix loop.

**Code trace** analyzes the fix commit on the source branch and documents how the corresponding code on the target branch differs (renamed symbols, changed APIs, divergence points).

**Test planning** is included in the code port plan by default (`use_combined_code_and_test_port_plan: true`). Claude reads `git show <source_fix_commit>` to identify test files from the fix, plans how to port them to the target branch's test infrastructure, and plans supplemental tests for coverage gaps — all in one step. Set to `false` for separate test plan iterations.

**Code gen** generates the fix patch including ported tests, compiles using the configured build command, and runs all tests.

**Runtime iterations** run an autonomous build-test-fix loop: compile, run tests, investigate failures, fix, repeat — up to `max_build_test_retries` attempts.

Config fields specific to this use case:

| Field | Description |
|---|---|
| `repo_path` | Path to the cloned repository |
| `build_dir` | Working directory for build/test commands |
| `source_branch` | Branch that has the fix |
| `target_branch` | Branch that needs the fix |
| `source_fix_commit` | Commit SHA of the fix on the source branch |
| `bug_description` | Human-readable description of the bug and fix |
| `issue_id` | Tracker ID (CVE, GitHub issue, etc.) |
| `build_command` | Incremental build command (e.g., `make -j$(nproc)`) |
| `test_command` | Test suite command (e.g., `make check -j$(nproc)`) |
| `max_build_test_retries` | Retry limit for build-test-fix loop (default: 3) |
| `use_combined_code_and_test_port_plan` | Combine code port plan and test plan (default: true) |

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
├── code_gen_configs.py          # PipelineConfig, CodeGenConfig, BugFixConfig
├── use_cases/
│   ├── base.py                  # UseCase ABC
│   ├── llm_framework.py         # LLM framework templates + step generators
│   └── bug_fix.py               # Bug fix templates + step generators
├── configs/
│   ├── code_gen_config_example.json
│   └── bug_fix_config_example.json
├── run_investigate_issue.py     # Post-pipeline issue investigation
├── run_fix_issue.py             # Post-pipeline issue fixing
├── run_work_items.py            # Arbitrary work item execution
└── run_summary.py               # PowerPoint summary generation
```

**Design pattern:** The pipeline orchestrator (`run_code_gen.py`) defines the phase structure. Each `UseCase` subclass provides domain-specific prompt templates and step generators. The prompt classes in `code_gen_prompts.py` are generic — they accept the template text at construction time, so the same class works for both LLM framework and bug fix content.

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
| `run_and_fix_failure.txt` | Written when build-test-fix retries are exhausted (bug fix) |

## Bug Fix Examples

### gcc CVE

```json
{
    "use_case": "bug_fix",
    "repo_path": "/home/user/gcc",
    "build_dir": "/home/user/gcc/build",
    "source_branch": "gcc-14-branch",
    "target_branch": "gcc-13-branch",
    "source_fix_commit": "abc1234",
    "bug_description": "CVE-2024-XXXX: buffer overflow in fold_convert()",
    "issue_id": "CVE-2024-XXXX",
    "output_dir": "/tmp/gcc_cve_output",
    "build_command": "make -j$(nproc)",
    "test_command": "make check RUNTESTFLAGS='gcc.dg/CVE-2024-XXXX.c' -j$(nproc)",
    "disallowed_modules": ["gcc/config/arm/"],
    "max_build_test_retries": 3
}
```

### openssl CVE

```json
{
    "use_case": "bug_fix",
    "repo_path": "/home/user/openssl",
    "source_branch": "openssl-3.2",
    "target_branch": "openssl-3.1",
    "source_fix_commit": "def5678",
    "bug_description": "CVE-2024-YYYY: use-after-free in SSL_free()",
    "issue_id": "CVE-2024-YYYY",
    "output_dir": "/tmp/openssl_fix",
    "build_command": "make -j$(nproc)",
    "test_command": "make test TESTS=test_ssl_old -j$(nproc)",
    "disallowed_modules": ["apps/", "doc/"]
}
```

### curl (cmake, macOS)

```json
{
    "use_case": "bug_fix",
    "repo_path": "/path/to/curl",
    "build_dir": "/path/to/curl/build",
    "source_branch": "curl-8_19_0",
    "target_branch": "curl-8_18_0",
    "source_fix_commit": "b35e58b24c",
    "bug_description": "openssl: fix potential OOB read in debug/verbose logging",
    "issue_id": "20656",
    "output_dir": "/tmp/curl_fix",
    "build_command": "cmake --build /path/to/curl/build -j$(sysctl -n hw.logicalcpu)",
    "test_command": "ctest --test-dir /path/to/curl/build -j$(sysctl -n hw.logicalcpu)",
    "disallowed_modules": [],
    "code_port_plan_skip_review": true,
    "code_gen_skip_review": true
}
```

### Build Command Guidelines

- Always use incremental compilation. Claude never does a speculative clean rebuild.
- Use `-j$(nproc)` on Linux, `-j$(sysctl -n hw.logicalcpu)` on macOS.
- For cmake projects, configure the build directory once before running (`cmake ..`).
- Scope `test_command` to the relevant test file/suite when possible to reduce cycle time.

## On Failure

If the runtime phase exhausts its retries, it writes a failure log (`run_and_fix_failure.txt` for bug fix, or stops with a summary for LLM framework) and exits.

Manual escalation:

```bash
# Deeper investigation of a specific issue
python -m auto_code_gen.run_investigate_issue --config <config.json> ...

# Direct fix attempt
python -m auto_code_gen.run_fix_issue --config <config.json> ...
```

Or increase `max_build_test_retries` / `num_runtime_iterations` and re-run with `--resume`.
