# Design: Bug Fix Branch Porting (`auto_bug_fix`)

**Date:** 2026-05-21
**Status:** Approved

## Context

`ai_auto_perf_analysis` is an AI-driven pipeline for GPU inference performance analysis. The existing `auto_code_gen/` module ports performance optimizations between ML frameworks (vLLM, SGLang) using Claude as the code generation engine.

This design extends the system to support a second use case: **porting a bug fix from one branch of a C/systems project (e.g., gcc, openssl) to another**, where the work is not performance-related. The extension lives in a new `auto_bug_fix/` module. The shared infrastructure in `common/` is already fully generic and requires no changes.

---

## Architecture & Module Structure

```
ai_auto_perf_analysis/
├── common/                        (unchanged)
│   ├── claude_utils.py            claude_run(), ClaudeConfig
│   └── utils.py                   git reset, safe_clean_dir, logging
│
├── auto_code_gen/                 (unchanged)
│
└── auto_bug_fix/                  (new)
    ├── __init__.py
    ├── bug_fix_config.py          BugFixConfig dataclass + example instance
    ├── bug_fix_prompts.py         All prompt classes, adapted for branch/bug context
    └── run_bug_fix.py             Main orchestrator
```

`auto_bug_fix/` has its own per-module prompt copies — no imports from `auto_code_gen/`. The separation means both modules can evolve independently. GPU-specific language stays in `auto_code_gen/`; bug-fix language lives in `auto_bug_fix/`.

---

## `BugFixConfig` Dataclass (`bug_fix_config.py`)

```python
@dataclass
class BugFixConfig:
    # Repository
    repo_path: str               # absolute path to the cloned repo
    build_dir: str               # working dir for build/test commands (often repo_path)

    # Branch identity
    source_branch: str           # branch that has the fix
    target_branch: str           # branch that needs the fix
    source_fix_commit: str       # SHA of the commit that introduced the fix

    # Bug context
    bug_description: str         # human-readable description / CVE / issue title
    issue_id: str                # tracker ID (e.g. "CVE-2024-1234" or "GH-98765")

    # Porting constraints
    disallowed_modules: list[str]  # files/dirs Claude must not touch on target branch
    port_tests: bool               # when True, TestPortPrompt runs before CodeGenPrompt

    # Build & test
    build_command: str           # e.g. "make -j$(nproc)"
    test_command: str            # e.g. "make check -j$(nproc)"
    max_build_test_retries: int  # retry limit for RunAndFixPrompt (default: 3)
```

The file also contains a concrete `claude_config: ClaudeConfig` instance and an example `bug_fix_config: BugFixConfig` instance that the user edits per project, following the same pattern as `code_gen_configs.py`.

---

## Prompt Classes (`bug_fix_prompts.py`)

All prompt classes use the existing dataclass + `ClassVar[str]` prompt template + `prompt()` method pattern. They are per-module copies — not shared with `auto_code_gen/`.

| Class | Status | Key change from `auto_code_gen` equivalent |
|---|---|---|
| `create_context_str()` | Adapted | Emits `repo_path`, `source_branch`, `target_branch`, `source_fix_commit`, `bug_description`; drops all GPU/ML fields |
| `CodeTracePrompt` | Adapted | Goal: trace what `source_fix_commit` touches on `source_branch`; drops CUDA/prefill/decode execution-mode analysis |
| `TestPortPrompt` | **New** | Runs `git show <source_fix_commit>`, identifies test files in the diff, plans and generates adapted test code for `target_branch`'s test infrastructure; writes a manifest of ported test paths to `test_port_manifest.txt` |
| `CodePortPlanPrompt` | Adapted | Source/target are branches, not frameworks; removes kernel vendoring, CUDA graph, and shape-verification language; adds structural divergence analysis between branches |
| `ReviewCodePortPlanPrompt` | Adapted | Same structural rewrite; removes GPU execution mode review |
| `TestPlanPrompt` | Adapted | Reduced role: validates coverage of ported tests from `TestPortPrompt`; adds supplemental regression checks if gaps exist |
| `CodeGenPrompt` | Adapted | Reads `config.build_command` instead of hard-coded vLLM cmake sequence; receives both fix patch plan and ported test files; compilation uses `config.build_dir` |
| `ReviewCodeGenPrompt` | Adapted | Same structural rewrite |
| `RunAndFixPrompt` | **New** | Single Claude prompt: (1) runs `build_command` via Bash, (2) on failure investigates and fixes, (3) runs `test_command` via Bash, (4) on failure investigates, fixes, recompiles, retests, (5) loops up to `max_build_test_retries`; on exhaustion writes `run_and_fix_failure.txt` and stops |
| `InvestigateIssuePrompt` | Adapted | Removes GPU/framework-specific investigation language; structurally identical |
| `ReviewInvestigatedIssuePrompt` | Adapted | Same |
| `WorkItemsPrompt` | Adapted | Context string update only |

### `TestPortPrompt` detail

Prompt instructs Claude to:
1. Run `git show <source_fix_commit>` and parse the diff
2. Identify test files added or modified in the commit (e.g., `gcc/testsuite/gcc.dg/bug-12345.c`, `test/recipes/70-test_foo.t`)
3. Inspect `target_branch`'s test infrastructure to understand harness conventions (dejagnu suite layout, OpenSSL test harness version, ctest structure, etc.)
4. Generate ported test code adapted to `target_branch`'s conventions
5. Write ported test files and record their paths in `test_port_manifest.txt`

`CodeGenPrompt` is given `test_port_manifest.txt` as an input (same pattern as `previous_code_port_plan_attempt_file` flowing between plan iterations), so the compilation/test loop enforces that the ported tests pass.

### `RunAndFixPrompt` detail

This is Approach A (Claude-owns-the-loop): a single prompt class whose template instructs Claude to drive the entire build-test-fix cycle using the Bash tool. Claude:

- Invokes `config.build_command` from `config.build_dir` using all available CPUs
- Falls back to a full rebuild only if the error is explicitly a stale-artifact failure; never does a full rebuild speculatively
- On build failure: investigates the compiler/linker output, applies targeted fixes, retries
- On successful build: invokes `config.test_command` and inspects output
- On test failure: investigates failures, applies fixes, recompiles, retests
- Loops until both build and tests are clean, or until `max_build_test_retries` is reached
- On exhaustion: writes the final failure log and error summary to `run_and_fix_failure.txt` and stops — the human reviews and can re-trigger `run_investigate_issue.py` manually

This matches the pattern already established in `CodeGenPrompt`, which tells Claude to recompile and iterate until tests pass within a single query.

---

## Pipeline Orchestrator (`run_bug_fix.py`)

The orchestrator builds a prompt list and passes it to `claude_run()`, identical in structure to `run_code_gen.py`. Default iteration counts: `N=4` plan review iterations, `M=3` code gen review iterations (matching `run_code_gen.py`).

```
Step 1: Reset target branch
        └─ eval cmd: utils.clear_repo(config.repo_path)

Step 2: CodeTracePrompt
        └─ Traces what source_fix_commit touches on source_branch

Step 3: TestPortPrompt                    [only if config.port_tests is True]
        ├─ git show source_fix_commit → identify test files in diff
        ├─ Generate ported tests for target_branch test infrastructure
        └─ Writes ported test file paths to output

Step 4: CodePortPlanPrompt × N
        └─ Each iteration: generate plan → ReviewCodePortPlanPrompt → fixed plan

Step 5: TestPlanPrompt
        └─ Validates test coverage from step 3; adds supplemental regression checks

Step 6: CodeGenPrompt × M
        ├─ Receives: final port plan + ported test files from step 3
        ├─ Applies fix to target_branch; compiles using config.build_command
        └─ Each iteration: codegen → ReviewCodeGenPrompt → fixed patch

Step 7: RunAndFixPrompt
        ├─ Claude runs config.build_command via Bash
        ├─ On build failure: investigate → fix → retry
        ├─ On build success: runs config.test_command via Bash
        ├─ On test failure: investigate → fix → recompile → retest
        ├─ Loops up to config.max_build_test_retries
        └─ On exhaustion: writes run_and_fix_failure.txt, stops

Step 8: WorkItemsPrompt                   [optional, user-defined follow-up tasks]
        └─ Apply combined patch, final verification run, etc.
```

### `clear_repo()` alias

`common/utils.py` gets a `clear_repo(path: str)` function as a generic alias for the same logic as `clear_vllm_source_tree()`. The original name is kept for backward compatibility with `run_code_gen.py`.

---

## Files to Create or Modify

| File | Action |
|---|---|
| `auto_bug_fix/__init__.py` | Create (empty) |
| `auto_bug_fix/bug_fix_config.py` | Create |
| `auto_bug_fix/bug_fix_prompts.py` | Create |
| `auto_bug_fix/run_bug_fix.py` | Create |
| `common/utils.py` | Add `clear_repo()` alias |

`auto_code_gen/` and all other existing files are untouched.

---

## Example Configuration (gcc use case)

```python
bug_fix_config = BugFixConfig(
    repo_path="/path/to/gcc",
    build_dir="/path/to/gcc/build",
    source_branch="gcc-14-branch",
    target_branch="gcc-13-branch",
    source_fix_commit="abc1234",
    bug_description="CVE-2024-XXXX: buffer overflow in fold_convert()",
    issue_id="CVE-2024-XXXX",
    disallowed_modules=["gcc/config/arm/"],
    port_tests=True,
    build_command="make -j$(nproc)",
    test_command="make check -j$(nproc)",
    max_build_test_retries=3,
)
```
