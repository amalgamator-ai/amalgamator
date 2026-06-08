# Bug Fix Porting Guide

This guide explains how to use the `auto_code_gen` pipeline to port a bug fix from one branch of a C/systems project (gcc, openssl, glibc, curl, etc.) to another, with automated test validation.

---

## Overview

Given a source commit that fixes a bug on one branch, the pipeline:

1. Traces the fix and compares code paths across branches
2. Ports the fix's test files to the target branch and validates they catch the bug (Red Check)
3. Plans and generates a ported patch
4. Drives an autonomous build-test-fix loop until the fix is clean
5. Confirms no regressions were introduced across the full test suite (Baseline Diff)

All steps are driven by Claude (Opus, with extended thinking). No manual compiler error copy-pasting required.

---

## Pipeline Steps

The pipeline runs in this order:

| Step | Name | Description |
|------|------|-------------|
| 0 | **Baseline Test** | Run the full test suite on the target branch *before* any changes. Captures a snapshot of pre-existing failures. |
| 1 | **Code Trace** | Analyze the source fix commit. Map which files, functions, and code paths are changed, and compare them against the target branch. |
| 1.5 | **Port Tests** | Extract test files added or modified in the source fix commit. Port them to the target branch's test infrastructure. |
| 1.7 | **Red Check** | Run the ported tests on the *unpatched* target branch. They should **fail** — this proves the tests are meaningful and catch the real bug. If they pass, the tests are not useful. |
| 2–4 | **Planning + CodeGen** | Generate a port plan, review it, generate the code patch, review it. Iterates until converged. |
| 5 | **Runtime Loop** | Apply the patch. Build. Run tests. If anything fails, Claude fixes it and retries — autonomously — up to `max_build_test_retries` times. |
| 6 | **Baseline Diff** | Run the full test suite again on the patched target branch. Compare against the Step 0 baseline to detect any regressions in tests unrelated to the bug fix. |

### What the validation steps check

```
Step 0  →  "What was already broken before we touched anything?"
Step 1.7 →  "Do our ported tests actually catch the bug? (Red = fail before fix)"
Step 6  →  "Did our fix break anything else? (Green = no new failures)"
```

Steps 1.5 and 1.7 run **before** Planning and CodeGen. This means you know the tests are valid before wasting time on a patch.

Step 6 runs **after** the Runtime Loop completes. The bug fix is always attempted regardless of the Baseline Diff result — Step 6 is a safety check, not a blocker.

---

## Quick Start

### 1. Create a config file

Copy [`configs/bug_fix_config_example.json`](../configs/bug_fix_config_example.json) and fill in your values:

```json
{
    "use_case": "bug_fix",
    "repo_path": "/path/to/your-project",
    "build_dir": "/path/to/your-project/build",
    "source_branch": "source-branch-name",
    "target_branch": "target-branch-name",
    "source_fix_commit": "<git-commit-sha>",
    "bug_description": "Short description of the bug being ported",
    "issue_id": "CVE-XXXX-YYYY or JIRA-123",
    "output_dir": "/path/to/output",
    "build_command": "make -j$(nproc)",
    "test_command": "make check",
    "max_build_test_retries": 5,
    "enable_test_port": true,
    "enable_baseline_diff": true,
    "thinking-mode": "deep"
}
```

### 2. Run

```bash
cd /path/to/ai_auto_analyze
source env.sh
python -m auto_code_gen.run_code_gen --config /path/to/config.json
```

Resume a run that was interrupted:

```bash
python -m auto_code_gen.run_code_gen --config /path/to/config.json --resume
```

---

## Config Reference

### Required fields

| Field | Description |
|-------|-------------|
| `use_case` | Must be `"bug_fix"` |
| `repo_path` | Absolute path to the git repository |
| `build_dir` | Directory where build commands are run (usually same as `repo_path`) |
| `source_branch` | Branch that contains the original fix commit |
| `target_branch` | Branch you want to port the fix to |
| `source_fix_commit` | Git commit SHA of the fix on the source branch |
| `bug_description` | Human-readable description of the bug (used as context for Claude) |
| `issue_id` | Identifier for this run (e.g. CVE number, ticket ID) |
| `output_dir` | Directory where all output files are written |
| `build_command` | Shell command to build the project |
| `test_command` | Shell command to run tests for the Runtime Loop |

### Validation flags

| Field | Default | Description |
|-------|---------|-------------|
| `enable_test_port` | `true` | Run Port Tests (Step 1.5) and Red Check (Step 1.7) |
| `enable_baseline_diff` | `false` | Run Baseline Test (Step 0) and Baseline Diff (Step 6). Set `test_command` to `make test` (all tests) for a meaningful diff. |

### Planning and CodeGen tuning

| Field | Default | Description |
|-------|---------|-------------|
| `num_code_port_plan_iterations` | 4 | Max planning iterations before stopping |
| `num_code_gen_iterations` | 3 | Max CodeGen iterations before stopping |
| `code_port_plan_skip_review` | `false` | Skip planning review (faster but less thorough) |
| `code_gen_skip_review` | `false` | Skip CodeGen review |
| `use_combined_code_and_test_port_plan` | `true` | Combine code + test planning into one phase |
| `max_build_test_retries` | 3 | Max attempts in the Runtime Loop |
| `thinking-mode` | `"deep"` | `"deep"` = Claude Opus (thorough), `"normal"` = Sonnet (faster) |
| `disallowed_modules` | `[]` | File paths Claude must not modify |

---

## Output Files

All files are written to `output_dir`:

| File | Written by | Description |
|------|-----------|-------------|
| `baseline_test_raw.txt` | Step 0 | Raw output of the full test suite before patching |
| `baseline_test_results.txt` | Step 0 | Parsed summary: total / passed / failed, with failure list |
| `{source_branch}_code_trace.txt` | Step 1 | Detailed code trace comparing source and target |
| `test_port_manifest.txt` | Step 1.5 | List of test files ported to the target branch |
| `test_validation_result.txt` | Step 1.7 | `RED_CHECK_PASSED` or `RED_CHECK_FAILED` with full analysis |
| `code_port_plan_V{N}.txt` | Planning | Port plan, one file per iteration |
| `code_port_plan_review_V{N}.txt` | Planning | Review of each plan iteration |
| `code_gen_V{N}.patch` | CodeGen | Generated patch, one file per iteration |
| `code_gen_review_V{N}.patch` | CodeGen | Reviewed/refined patch |
| `run_and_fix_success.txt` | Runtime Loop | Written on success: `SUCCESS`, retry count, build/test summary |
| `run_and_fix_failure.txt` | Runtime Loop | Written on failure: failure summary after retries exhausted |
| `baseline_diff_report.txt` | Step 6 | `NO_REGRESSION` or regression report with new failure list |

---

## Understanding the Validation Output

### Red Check (`test_validation_result.txt`)

```
RED_CHECK_PASSED           ← First line always one of these
                           RED_CHECK_FAILED
                           BUILD_FAILED
                           SKIPPED

[Detailed analysis of why tests fail on the unpatched target...]
```

- `RED_CHECK_PASSED` — tests fail without the fix. Proceed to Planning/CodeGen.
- `RED_CHECK_FAILED` — tests pass without the fix. The tests are not catching the bug. Investigate before proceeding.
- `BUILD_FAILED` — the ported test files have a compilation issue. Inspect `test_port_manifest.txt` and fix manually.

### Baseline Diff (`baseline_diff_report.txt`)

```
NO_REGRESSION              ← First line always one of these
                           REGRESSION_DETECTED

Pre-patch:  Total <N> | Pass <M> | Fail <K>
Post-patch: Total <N> | Pass <N> | Fail 0

[Details: new failures (regressions), fixed tests, unchanged failures...]
```

- `NO_REGRESSION` — the patch introduced no new test failures. 
- `REGRESSION_DETECTED` — one or more tests that passed before the patch now fail. Review whether these are caused by the patch or are pre-existing flakiness.

> **Note:** Pre-existing failures from Step 0 are excluded from the regression count — only *new* failures introduced by the patch are flagged.

---

## Example: OpenSSL CVE-2024-12797

**Setup:**
- Source branch: `openssl-3.6` (has the fix)
- Target branch: `openssl-3.4` (needs the fix ported)
- Fix commit: adds `ssl/ssl_sess.c` change + `test/tls12psk.c` + recipe + `build.info`

**Config:**
```json
{
    "use_case": "bug_fix",
    "repo_path": "/path/to/openssl",
    "build_dir": "/path/to/openssl",
    "source_branch": "openssl-3.6",
    "target_branch": "openssl-3.4",
    "source_fix_commit": "<commit-sha>",
    "bug_description": "CVE-2024-12797: ssl_get_prev_session() does not verify session ID returned by external cache matches ClientHello",
    "issue_id": "CVE-2024-12797",
    "output_dir": "/path/to/output",
    "build_command": "make -j4",
    "test_command": "make test TESTS=test_tls12_psk",
    "max_build_test_retries": 5,
    "enable_test_port": true,
    "enable_baseline_diff": true,
    "thinking-mode": "deep"
}
```

**Results:**
- Step 0: 3,809 tests, 13 pre-existing failures (environment issues, not caused by patch)
- Step 1.7: `RED_CHECK_PASSED` — `test_tls12_psk_resume_sessid_mismatch` failed across all 4 PSK cipher suites on the unpatched target
- Runtime Loop: `SUCCESS` on retry 0 — patch built and tested cleanly first attempt
- Step 6: `NO_REGRESSION` — 3,810/3,810 post-patch, the 13 pre-existing failures resolved after rebuild

---

## Tips

- **Set `test_command` to a targeted test** for the Runtime Loop (fast feedback). Use `make test` (all tests) for `enable_baseline_diff` to get a meaningful regression check.
- **Use `thinking-mode: "normal"`** (Sonnet) for quick exploratory runs. Switch to `"deep"` (Opus) for production runs.
- **Set `enable_baseline_diff: false`** if the full test suite is very slow and you only care about the targeted fix passing.
- **Check `test_port_manifest.txt` first** if the Red Check fails — the ported test files are listed there and can be inspected manually.
- **The fix is always attempted** even if the Red Check fails or the Baseline Diff detects regressions. Validation steps are informational, not blockers (except build failures).
