# Code Review: Repurposing for Branch Bug-Fix Porting

## Context

This repo (`ai_auto_perf_analysis`) is an AI-driven pipeline for performance analysis of GPU-based LLM inference frameworks (vLLM, SGLang, TensorRT-LLM). It uses Claude (via the Agent SDK) to analyze GPU traces, compare framework versions, and port performance optimizations.

The question: which parts of this system could support **porting a bug fix from one branch of a C/systems project (e.g., gcc, openssl) to another**, where the porting work is not performance-related?

---

## Architecture Summary

```
common/          ← shared Claude SDK wrapper, logging, git utilities
auto_analyze/    ← GPU trace analysis pipelines (configs, prompts, runners)
auto_profile/    ← benchmark profiling orchestration and config parsing
auto_code_gen/   ← AI-driven code porting and generation pipeline
```

---

## RELEVANT Modules

### 1. `common/claude_utils.py` — **Core execution engine** (fully reusable)

The `claude_run()` async function and `ClaudeConfig` dataclass are the heart of every pipeline. They:
- Accept a list of prompts/commands and stream them through the Claude Agent SDK
- Support mixed prompt types (strings or `eval()`-based dict commands)
- Provide timing and logging

**For bug fix porting**: This is the execution substrate for every AI step. No changes needed — it is entirely generic.

---

### 2. `common/utils.py` — **Infrastructure utilities** (fully reusable)

- `clear_vllm_source_tree(path)` — runs `git reset --hard` + `git clean` on a repo. Directly applicable to any git repo; just needs renaming.
- `safe_clean_dir()` — safe recursive directory cleanup with guards against wiping system dirs, the home dir, or git repos. Useful during any multi-step pipeline.
- `Tee` class + `setup_logging()` — dual-output logging (stdout + file). Generic and reusable.

**For bug fix porting**: These utilities handle the filesystem and git hygiene that any branch-checkout workflow needs.

---

### 3. `auto_code_gen/code_gen_configs.py` — **Configuration dataclass** (needs adaptation)

`CodeGenConfig` currently holds:
- Frameworks (e.g., vLLM, SGLang), source paths, GPU type, batch sizes
- "Faster" and "slower" framework designation
- `framework_test_dirs` — test directories used for execution trace analysis
- Output directories, disallowed modules, improvement plan file + step

**For bug fix porting**: The structural pattern maps directly:
- "source framework" → branch with the fix
- "target framework" → branch needing the fix
- "disallowed modules" → files that must not be touched on the target branch
- The GPU/model/batch fields are irrelevant; drop them and add:
  - `source_branch`, `target_branch`, `bug_description`, `issue_id`
  - `source_fix_commit` — the specific commit SHA on the source branch that introduced the fix (used to extract the diff and associated test files)
  - `source_test_files` — list of test files added/modified in `source_fix_commit`, populated after extraction
  - `port_tests` — boolean flag; when true, extracted test cases are ported before code generation begins
  - `build_command` — shell command to compile the target branch (e.g., `make -j$(nproc)` or `cmake --build build -j$(nproc)`); always incremental, never a clean rebuild unless the previous attempt explicitly failed due to a stale artifact
  - `test_command` — shell command to run the relevant test suite (e.g., `make check -j$(nproc)`, `ctest --output-on-failure`, or a specific dejagnu/OpenSSL harness invocation)
  - `build_dir` — working directory from which `build_command` and `test_command` are invoked (defaults to repo root)

---

### 4. `auto_code_gen/code_gen_prompts.py` — **Most relevant prompt module** (adapt prompts)

This is the most directly applicable module. Each prompt class wraps a multi-step Claude task:

| Prompt Class | Purpose | Relevance to Bug Fix Porting |
|---|---|---|
| `CodeTracePrompt` | Trace code paths and call chains through a codebase | **High** — understand what code the fix touches on the source branch; also identifies test dirs via `framework_test_dirs` |
| `CodePortPlanPrompt` | Plan how to port code from one framework to another | **High** — plan how to apply the same fix on the diverged target branch |
| `ReviewCodePortPlanPrompt` | Review and refine the porting plan | **High** — iterative plan refinement |
| `TestPlanPrompt` | Generates a new test plan by analyzing existing test structures in both frameworks | **Medium** — useful for supplemental coverage, but **does not port existing tests from the fix commit**; see gap below |
| `CodeGenPrompt` | Generate code patch AND test implementations; triggers compilation and enforces build success before handing off to `RunAndFixPrompt` | **High** — writes ported fix; uses `config.build_command` for incremental compilation with all available CPUs; never hard-codes framework-specific build steps |
| `RunAndFixPrompt` | Run the compiled binary or test suite, capture stdout/stderr, and feed failures directly into `InvestigateIssuePrompt` without manual copy-paste; loops until all tests pass | **High — new** — replaces the manual run → copy-paste-error → re-run cycle; see gap note below |
| `ReviewCodeGenPrompt` | Review generated code and tests; can add new tests | **High** — verify generated patch; enforces test coverage |
| `InvestigateIssuePrompt` | Investigate test failures after patch is applied; iterates until resolved | **High** — debug porting problems in both fix code and ported tests |
| `FixIssuePrompt` | Generate fixes for identified issues | **High** — correct porting errors |
| `WorkItemsPrompt` | Execute a list of discrete work items | **High** — run any ordered sub-tasks including test execution |
| `create_context_str()` | Format config as a context header for prompts | **High** — needs editing to remove GPU/performance language |

**Only adaptation needed**: The context strings reference GPU types, batch sizes, and ML-specific language. Replace with branch names, repo path, and bug description.

#### Gap: Test Case Extraction from the Source Branch Fix Commit

**None of the existing prompt classes reads a `git diff` or `git show` output to identify test files that were committed alongside the fix on the source branch.** `TestPlanPrompt` generates a new test plan from scratch by analyzing the existing test structure in both branches — it does not automatically extract and port the actual test cases shipped with the fix.

For projects like gcc (dejagnu `.exp` files, `.c` regression files) or openssl (`test/` scripts, `make test` targets), the fix commit typically adds one or more test cases that directly exercise the patched code path. These are the highest-value tests to port because they were written specifically to catch the bug.

**Proposed addition: `TestPortPrompt`** — a new prompt class (or an extension of `CodePortPlanPrompt`) that:
1. Reads `git show <source_fix_commit>` to extract the diff
2. Identifies test files added or modified in the fix commit
3. Plans how to adapt those test files to the target branch's test infrastructure (e.g., different dejagnu suite layout, different OpenSSL test harness version)
4. Generates the ported test code as a separate patch, before the fix code is generated

This test patch is then fed into `CodeGenPrompt` alongside the fix patch, so `CodeGenPrompt`'s existing test-execution loop runs the ported tests and enforces that they pass.

#### Gap: Automatic Build and Test Execution with Error Feedback

**The existing pipeline requires manual intervention to run code and pass failure logs to Claude.** In the vLLM/SGLang context, GPU availability checks made automatic execution impractical, so the workflow was: run manually → copy-paste the error log into `run_investigate_issue.sh` → rerun manually after Claude's fix.

For C/systems projects (gcc, openssl, etc.) there is no GPU gating, so a fully automated compile → run → fix loop is feasible.

**Proposed addition: `RunAndFixPrompt`** — a new prompt class that:
1. Invokes `config.build_command` from `config.build_dir` using all available CPUs (incremental by default; full rebuild only if the error is a stale-artifact failure)
2. Captures stdout/stderr from the build step; if the build fails, immediately routes the captured log to `InvestigateIssuePrompt` and then `FixIssuePrompt`, then retries
3. On a successful build, invokes `config.test_command` and captures its output
4. If tests fail, routes the captured log to `InvestigateIssuePrompt` → `FixIssuePrompt`, then recompiles and reruns
5. Loops until both build and tests are clean, or until a configurable retry limit is hit

This eliminates the copy-paste step entirely and turns what was a human-in-the-loop debugging cycle into a fully autonomous build-test-fix loop.

**Compilation note**: The `build_command` in `CodeGenConfig` replaces the vLLM-specific incremental build sequence (`generate_cmake_presets.py` → `cmake --preset release` → `cmake --build --preset release --target install`). For a generic C project the equivalent is just `make -j$(nproc)` or `cmake --build <build_dir> -j$(nproc)`. Claude should never hard-code framework-specific build steps in prompts; it reads `config.build_command` instead.

---

### 5. `auto_code_gen/run_code_gen.py` — **Main orchestration** (adapt as entry point)

Orchestrates: `CodeTrace → PortPlan → Review → TestPlan → CodeGen → Review` with multi-iteration refinement loops. This is the pipeline structure that directly mirrors what bug fix porting requires. The ML-specific variable names and context strings need updating, but the control flow is sound.

---

### 6. `auto_code_gen/run_investigate_issue.py` and `run_fix_issue.py` — **Debugging sub-pipelines** (reuse as-is)

Used when generated code fails testing. These are generic enough — they take a plan, a test plan, and a failing code state, then reason about what's wrong and generate a fix. No GPU-specific logic.

---

### 7. `auto_code_gen/run_work_items.py` — **Task executor** (reuse as-is)

Executes a list of discrete work items (refactoring, applying patches, running tests). Fully generic.

---

### 8. `auto_analyze/configs/single_trace_config.py` — **Partial reuse**

- `prepare_source_code()` — clones a git repo and checks out a specific commit. The pattern (clone → reset → checkout) applies directly to checking out source/target branches.
- The rest of `SingleTraceConfig` and `SingleTraceParams` (trace files, GPU focus, transformer block files) is irrelevant.

---

## IRRELEVANT Modules

### `auto_analyze/prompts/` — **All prompt modules here are GPU/ML-specific**

| Module | Why irrelevant |
|---|---|
| `single_trace_prompts.py` | Analyzes CUDA kernel traces and transformer block execution — entirely GPU/ML |
| `cross_trace_prompts.py` | Compares GPU execution across runs, generates performance improvement plans |
| `chrome_trace_prompts.py` | Generates Perfetto Chrome trace JSON for GPU visualization |
| `summary_pdf_prompts.py` | Builds PDF reports from GPU profiling results |
| `jira_prompts.py` | Creates JIRA tasks from performance improvement proposals (could repurpose, but marginal) |

---

### `auto_analyze/run_*.py` — **All GPU trace entry points are irrelevant**

`run_single_trace.py`, `run_cross_trace.py`, `run_chrome_trace.py`, `run_summary_pdf.py` — all drive GPU/ML trace analysis pipelines. Not applicable.

---

### `auto_analyze/configs/cross_trace_config.py` — **Mostly irrelevant**

`TraceResult` and `CrossTraceConfig` manage pointers to GPU trace result directories (median block files, GPU ops files). The abstraction of "two versions to compare" maps conceptually to "source and target branch," but the fields are too trace-specific to reuse without a full rewrite.

---

### `auto_profile/` — **Entirely irrelevant**

This module orchestrates GPU benchmark profiling: Docker image configs, GPU group configs, inference benchmark execution modes, profiling result parsing. None of this has any applicability to porting a bug fix in a C project.

| Module | Why irrelevant |
|---|---|
| `parse_run_config.py` | Parses profiling benchmark configs, resolves GPU/exec-mode references |
| `parse_prompts.py` | Restructures benchmark results into test directories |
| `run_profile_summary.py` | Generates analysis configs from profiling results |
| `test_configs/*.json` | Docker images, GPU groups, execution modes |

---

### `auto_analyze/examples/` and `docs/` — **Irrelevant**

ML inference examples and documentation for the existing GPU analysis pipelines.

---

## Proposed Bug Fix Porting Workflow

Using the relevant modules, the workflow for porting a bug fix from `gcc-14-branch` → `gcc-13-branch` would be:

```
1. Config Setup
   ├─ Adapt CodeGenConfig:
   │    source_branch      = "gcc-14-branch"  (has the fix)
   │    target_branch      = "gcc-13-branch"  (needs the fix)
   │    repo_path          = "/path/to/gcc"
   │    source_fix_commit  = "abc1234"         (commit that introduced the fix)
   │    bug_description    = "CVE-2024-XXXX: buffer overflow in fold_convert()"
   │    disallowed_modules = ["gcc/config/arm/"]
   │    port_tests         = True
   │    build_command      = "make -j$(nproc)"          (incremental; project-specific)
   │    test_command       = "make check -j$(nproc)"    (or ctest, dejagnu, etc.)
   │    build_dir          = "/path/to/gcc/build"       (where build/test commands run)
   └─ Use common/utils.py for repo checkout/reset

2. Understand the Fix (CodeTracePrompt)
   ├─ Claude reads source_branch diff and traces affected call chains
   └─ Identifies source files and test files touched in source_fix_commit

2b. Extract and Port Test Cases (proposed TestPortPrompt — NEW)
   ├─ Run: git show <source_fix_commit> to get the full diff
   ├─ Identify test files added/modified (e.g., gcc/testsuite/gcc.dg/bug-12345.c)
   ├─ Plan adaptation to target_branch test infrastructure
   │    (different dejagnu suite layout, harness API differences, etc.)
   ├─ Generate ported test patch for target_branch
   └─ Populate config.source_test_files with ported test paths

3. Plan the Fix Port (CodePortPlanPrompt + ReviewCodePortPlanPrompt, multi-iteration)
   ├─ Claude identifies where same logic lives on target_branch
   ├─ Notes structural divergence between branches
   └─ Plans fix patches needed (separate from test patch)

4. Supplemental Test Plan (TestPlanPrompt — existing, reduced role)
   └─ Validates that ported tests from step 2b are sufficient;
      generates additional regression checks if coverage gaps exist

5. Generate Fix Code (CodeGenPrompt + ReviewCodeGenPrompt, multi-iteration)
   ├─ Receives both the fix patch plan AND the ported test files from step 2b
   ├─ Applies fix to target_branch
   ├─ Compiles using config.build_command (incremental, all CPUs)
   │    - Falls back to a full rebuild only if the error is a stale-artifact failure
   │    - Never uses framework-specific build steps; reads build_command from config
   └─ Hands off to RunAndFixPrompt once the patch is written

6. Run, Test, and Auto-Fix Loop (RunAndFixPrompt — NEW)
   ├─ Invoke config.build_command from config.build_dir
   ├─ If build fails:
   │    ├─ Capture stdout/stderr
   │    ├─ Pass captured log → InvestigateIssuePrompt → FixIssuePrompt
   │    └─ Retry build (no manual copy-paste required)
   ├─ On successful build, invoke config.test_command
   ├─ If tests fail:
   │    ├─ Capture stdout/stderr
   │    ├─ Pass captured log → InvestigateIssuePrompt → FixIssuePrompt
   │    └─ Recompile and rerun tests
   └─ Loop until build + tests are clean (or configurable retry limit reached)

7. Execute Work Items (run_work_items.py)
   └─ Apply combined patch (fix + tests), run full build, run test suite

8. If RunAndFixPrompt exhausts retries → run_investigate_issue.py → run_fix_issue.py
   ├─ Deeper investigation for stubborn failures
   └─ Handles cases where test infrastructure itself needs further adaptation

9. All steps run via claude_run() from common/claude_utils.py
```

### What Already Works for Test Verification

The iteration infrastructure handles investigation and fixing without changes:
- `InvestigateIssuePrompt` already iterates until failures are resolved, covering both fix code and test adaptation failures
- `ReviewCodeGenPrompt` already adds tests if coverage gaps are found

Two pieces are missing:

1. **Step 2b** — `TestPortPrompt`: reads the source branch fix commit's diff and ports its test additions to the target branch before code generation begins.

2. **Step 6** — `RunAndFixPrompt`: eliminates the manual run → copy-paste error log → re-run cycle. In the vLLM/SGLang case, GPU availability checks made automation impractical, so `run_investigate_issue.sh` required a human to paste the failure log. For C/systems projects there is no such gating: `RunAndFixPrompt` runs `config.test_command`, captures its output, and routes failures directly into `InvestigateIssuePrompt` without any manual step.

---

## Summary Table

| Module | Relevant? | Notes |
|---|---|---|
| `common/claude_utils.py` | **Yes — core** | Generic Claude execution, no changes needed |
| `common/utils.py` | **Yes** | Git reset, safe cleanup, logging |
| `auto_code_gen/code_gen_configs.py` | **Yes — adapt** | Drop GPU fields; add `source_fix_commit`, `source_test_files`, `port_tests` |
| `auto_code_gen/code_gen_prompts.py` | **Yes — adapt** | All prompt classes apply; update context strings; add `TestPortPrompt` |
| `auto_code_gen/run_code_gen.py` | **Yes — adapt** | Control flow directly applicable; insert test-port step before code gen |
| `auto_code_gen/run_investigate_issue.py` | **Yes** | Generic debugging pipeline; handles ported test failures without changes |
| `auto_code_gen/run_fix_issue.py` | **Yes** | Generic fix pipeline |
| `auto_code_gen/run_work_items.py` | **Yes** | Generic task executor |
| `auto_analyze/configs/single_trace_config.py` | **Partial** | Only `prepare_source_code()` git checkout logic |
| `auto_analyze/prompts/*` | **No** | GPU trace / ML-specific throughout |
| `auto_analyze/run_*.py` | **No** | GPU trace analysis entry points |
| `auto_analyze/configs/cross_trace_config.py` | **No** | Too trace-specific to reuse |
| `auto_profile/*` | **No** | Benchmark profiling infrastructure |
| Shell scripts (`run_all.sh`, etc.) | **No** | Orchestrate GPU profiling pipeline |

### New Components Required

| Component | Type | Purpose |
|---|---|---|
| `TestPortPrompt` | New prompt class in `code_gen_prompts.py` | Reads `git show <source_fix_commit>`, identifies test files in the diff, plans and generates ported test code for the target branch's test infrastructure |
| `RunAndFixPrompt` | New prompt class in `code_gen_prompts.py` | Invokes `config.build_command` then `config.test_command`; captures stdout/stderr; on failure routes the log directly to `InvestigateIssuePrompt` → `FixIssuePrompt` and retries; loops until clean — no manual copy-paste of error logs |
| `build_command`, `test_command`, `build_dir` | New fields on `CodeGenConfig` | Replace the hard-coded vLLM incremental build sequence (`generate_cmake_presets.py` / `cmake --preset` / `cmake --build`) with project-specific shell commands; prompts read these fields instead of embedding framework-specific steps |
