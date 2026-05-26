# Bug Fix Porting Guide

## Motivation

When a bug is fixed on a maintenance branch (e.g., `gcc-14-branch`), the same fix often
needs to be applied to an older branch (`gcc-13-branch`) that has diverged in structure,
APIs, and surrounding code. Manual porting is error-prone and slow: you read the diff,
hunt for the equivalent code on the target branch, re-implement the logic accounting for
divergence, adapt any test cases, compile, iterate on build errors, and repeat.

`auto_bug_fix` automates the entire cycle. Given the source branch, target branch, and
the commit SHA that introduced the fix, Claude:

1. Traces which code the fix touches on the source branch
2. Extracts and ports the fix's test cases to the target branch's test infrastructure
3. Plans how to apply the same fix on the diverged target branch
4. Generates the ported patch and compiles it
5. Drives an autonomous build-test-fix loop — no manual copy-pasting of compiler errors

**Supported projects:** Any git repository with shell-invokable build and test commands —
gcc (dejagnu), openssl, glibc, binutils, llvm, or any project using make/cmake/ctest.

---

## Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) (used to run the pipeline; installs `claude-agent-sdk` automatically via `pyproject.toml`)
- A cloned git repository with both the source and target branches available locally
- The commit SHA of the fix on the source branch (`git log` on the source branch to find it)
- Working `build_command` and `test_command` for the target branch
- For cmake-based projects: `cmake` installed (`brew install cmake` on macOS) and the build directory configured (`cmake ..` run from `build_dir`) before the pipeline starts

---

## Setup

### 1. Identify the fix commit

On the source branch, find the commit that introduced the fix:

```bash
git log --oneline gcc-14-branch | head -20
# e.g.: abc1234 Fix buffer overflow in fold_convert()
```

### 2. Edit `auto_bug_fix/bug_fix_config.py`

Open `auto_bug_fix/bug_fix_config.py` and fill in the `bug_fix_config` and `claude_config`
instances near the bottom of the file.

**`claude_config`** — controls Claude's runtime settings:

```python
claude_config = ClaudeConfig(
    model="claude-opus-4-6",
    allowed_tools=["Read", "Write", "Bash"],
    perm_mode="acceptEdits",
    cwd="/path/to/output/dir",   # ← where Claude writes its working files
)
```

Set `cwd` to a dedicated output directory (e.g., `/tmp/bug_fix_output/`). Claude writes
code traces, port plans, generated patches, and failure logs here.

**`bug_fix_config`** — describes the porting task:

```python
bug_fix_config = BugFixConfig(
    # Repository
    repo_path="/path/to/gcc",        # absolute path to the cloned repo
    build_dir="/path/to/gcc/build",  # directory from which build/test commands run

    # Branch identity
    source_branch="gcc-14-branch",   # branch that has the fix
    target_branch="gcc-13-branch",   # branch that needs the fix

    # The specific commit that introduced the fix on source_branch
    source_fix_commit="abc1234",

    # Human-readable description (used in prompts for context)
    bug_description="CVE-2024-XXXX: buffer overflow in fold_convert() when handling "
                    "integer conversions with mismatched types",
    issue_id="CVE-2024-XXXX",

    # Files/directories Claude must NOT modify on the target branch
    disallowed_modules=["gcc/config/arm/"],

    # When True, Claude extracts and ports test files from source_fix_commit
    port_tests=True,

    # Shell commands to compile and test the target branch
    build_command="make -j$(nproc)",
    test_command="make check RUNTESTFLAGS='gcc.dg/CVE-2024-XXXX.c' -j$(nproc)",

    # Retry limit for the autonomous build-test-fix loop
    max_build_test_retries=3,
)
```

**Field reference:**

| Field | Description |
|---|---|
| `repo_path` | Absolute path to the cloned repository |
| `build_dir` | Working directory for `build_command` and `test_command`; often `repo_path` or a `build/` subdirectory |
| `source_branch` | Branch that contains the fix |
| `target_branch` | Branch that needs the fix applied |
| `source_fix_commit` | SHA of the commit on `source_branch` that introduced the fix |
| `bug_description` | Plain-language description of the bug and fix (appears in all prompts) |
| `issue_id` | Tracker ID: CVE number, GitHub issue, Bugzilla ID, etc. |
| `disallowed_modules` | List of file/directory paths that must not be modified on the target branch |
| `port_tests` | `True`: extract and port test files from `source_fix_commit`; `False`: skip test extraction |
| `build_command` | Incremental build command. Always uses all available CPUs (`-j$(nproc)`). |
| `test_command` | Test suite command. Can be scoped to just the relevant test file/suite. |
| `max_build_test_retries` | Max iterations of the build-test-fix loop before stopping (default: 3) |

### 3. Create the output directory

```bash
mkdir -p /path/to/output/dir
```

### 4. Check out the target branch

The repository must be on the target branch before running:

```bash
cd /path/to/gcc
git checkout gcc-13-branch
```

The pipeline's first step resets the branch to a clean state (`git reset --hard` + `git clean -fd`),
so any local modifications are safe to leave — they will be removed.

---

## Running

```bash
cd /path/to/ai_auto_perf_analysis
uv run uv run python -m auto_bug_fix.run_bug_fix
```

All output is logged to stdout and to `__run_log_bug_fix.txt` in the current directory.

---

## Pipeline Steps

The pipeline runs sequentially. Each step is a Claude query; Claude writes its results
to files in `claude_config.cwd`.

```
Step 1: Reset target branch
        └─ git reset --hard + git clean -fd on repo_path

Step 2: CodeTracePrompt
        └─ Runs git show <source_fix_commit> and traces the full call chain
           of the fix on the source branch
           Output: <source_branch>_code_trace.txt

Step 3: TestPortPrompt  [only if port_tests=True]
        ├─ Identifies test files added/modified in source_fix_commit
        ├─ Inspects target_branch's test harness (dejagnu, ctest, OpenSSL runner, etc.)
        ├─ Generates ported test files adapted to target_branch's conventions
        └─ Output: test_port_manifest.txt (one ported test path per line)

Step 4: CodePortPlanPrompt × 4 iterations  (each followed by ReviewCodePortPlanPrompt)
        ├─ Plans how to apply the fix on the diverged target branch
        ├─ Each iteration refines the plan based on review feedback
        └─ Output: code_port_plan_V{N}_fixed_from_..._to_....txt

Step 5: TestPlanPrompt
        ├─ Validates coverage of ported tests from Step 3
        ├─ Plans supplemental regression tests if gaps exist
        └─ Output: test_plan_from_..._to_....txt

Step 6: CodeGenPrompt × 3 iterations  (each followed by ReviewCodeGenPrompt)
        ├─ Implements the porting plan — applies fix to target_branch source
        ├─ Compiles using build_command (incremental, never speculative clean rebuild)
        ├─ Runs ported tests
        └─ Output: code_gen_V{N}_PR_FIXED_from_..._to_....txt

Step 7: RunAndFixPrompt
        ├─ Autonomous build-test-fix loop (Claude uses Bash tool directly)
        ├─ Runs build_command → on failure: investigate + fix + retry
        ├─ Runs test_command → on failure: investigate + fix + recompile + retest
        ├─ Loops up to max_build_test_retries times
        └─ On success: reports clean build + tests
           On exhaustion: writes run_and_fix_failure.txt and stops
```

---

## Output Files

All files are written to `claude_config.cwd`.

| File | When written | Description |
|---|---|---|
| `<source_branch>_code_trace.txt` | Step 2 | Full call-chain trace of the fix on the source branch |
| `test_port_manifest.txt` | Step 3 | Paths of ported test files, one per line |
| `code_port_plan_V{N}_*.txt` | Step 4 | Plan iterations and review feedback |
| `test_plan_from_*.txt` | Step 5 | Supplemental test plan |
| `code_gen_V{N}_PR_*.txt` / `.patch` | Step 6 | Generated patch iterations and reviews |
| `run_and_fix_failure.txt` | Step 7 (on failure) | Final build/test output + summary of all attempted fixes |

---

## Build Command Guidelines

**Always incremental.** Claude never does a speculative `make clean` or full rebuild.
A full rebuild is only triggered if the error output explicitly indicates a stale artifact
(e.g., a missing symbol that clearly exists in a source file that wasn't recompiled).

**Parallelism flag:** Use `-j$(nproc)` on Linux. On macOS, `nproc` is not available — use
`-j$(sysctl -n hw.logicalcpu)` instead.

**cmake-based projects:** Use `cmake --build` and `ctest` rather than invoking `make`
directly. Specify the build directory explicitly so the commands work regardless of the
current working directory:

```bash
# cmake build
build_command="cmake --build /path/to/build -j$(sysctl -n hw.logicalcpu)"

# ctest — run all tests
test_command="ctest --test-dir /path/to/build -j$(sysctl -n hw.logicalcpu)"

# ctest — run a named test only
test_command="ctest --test-dir /path/to/build -R test_foo --output-on-failure"
```

The `build_dir` must be configured (i.e., `cmake ..` run from it) before the pipeline
starts — the pipeline does not run cmake configuration itself.

**Scope the test command** to the relevant test file or suite when possible — it reduces
cycle time in the build-test-fix loop significantly:

```bash
# gcc dejagnu — run only the specific regression test
test_command="make check RUNTESTFLAGS='gcc.dg/CVE-2024-XXXX.c' -j$(nproc)"

# openssl — run a specific recipe
test_command="make test TESTS=test_aes -j$(nproc)"

# cmake/ctest — run a named test
test_command="ctest --test-dir /path/to/build -R test_foo --output-on-failure"

# Full suite (slower, but gives full coverage)
test_command="make check -j$(nproc)"
```

---

## On Failure

If `RunAndFixPrompt` exhausts `max_build_test_retries`, it writes `run_and_fix_failure.txt`
to `claude_config.cwd` and stops. The file contains the final build/test output and a
summary of every fix attempted during the loop.

**Manual escalation path:**

1. Read `run_and_fix_failure.txt` to understand what was tried and where it's stuck.
2. If the failure is a complex porting issue (API divergence, missing symbol, test harness
   mismatch), run the deeper investigation pipeline manually:

   ```bash
   uv run python -m auto_code_gen.run_investigate_issue
   uv run python -m auto_code_gen.run_fix_issue
   ```

3. Alternatively, increase `max_build_test_retries` and re-run — Claude picks up where
   it left off using the previous iteration's files in `claude_config.cwd`.

---

## Example: gcc CVE

**Scenario:** A buffer overflow in `fold_convert()` was fixed on `gcc-14-branch` in commit
`abc1234`. The same fix is needed on `gcc-13-branch`, which has diverged significantly
in the surrounding integer conversion code.

```python
bug_fix_config = BugFixConfig(
    repo_path="/home/user/gcc",
    build_dir="/home/user/gcc/build",
    source_branch="gcc-14-branch",
    target_branch="gcc-13-branch",
    source_fix_commit="abc1234",
    bug_description="CVE-2024-XXXX: buffer overflow in fold_convert() when "
                    "converting between integer types with mismatched sign/width",
    issue_id="CVE-2024-XXXX",
    disallowed_modules=["gcc/config/arm/", "gcc/config/aarch64/"],
    port_tests=True,
    build_command="make -j$(nproc)",
    test_command="make check RUNTESTFLAGS='gcc.dg/CVE-2024-XXXX.c' -j$(nproc)",
    max_build_test_retries=3,
)

claude_config = ClaudeConfig(
    model="claude-opus-4-6",
    allowed_tools=["Read", "Write", "Bash"],
    perm_mode="acceptEdits",
    cwd="/tmp/gcc_cve_output",
)
```

```bash
mkdir -p /tmp/gcc_cve_output
cd /home/user/gcc && git checkout gcc-13-branch
cd /path/to/ai_auto_perf_analysis
uv run python -m auto_bug_fix.run_bug_fix
```

> **macOS note:** Replace `-j$(nproc)` with `-j$(sysctl -n hw.logicalcpu)` in all commands.

---

## Example: openssl

```python
bug_fix_config = BugFixConfig(
    repo_path="/home/user/openssl",
    build_dir="/home/user/openssl",
    source_branch="openssl-3.2",
    target_branch="openssl-3.1",
    source_fix_commit="def5678",
    bug_description="CVE-2024-YYYY: use-after-free in SSL_free() during "
                    "session renegotiation with NPN extension enabled",
    issue_id="CVE-2024-YYYY",
    disallowed_modules=["apps/", "doc/"],
    port_tests=True,
    build_command="make -j$(nproc)",
    test_command="make test TESTS=test_ssl_old -j$(nproc)",
    max_build_test_retries=3,
)
```

---

## Example: curl (cmake, macOS)

Prerequisites: `brew install cmake libpsl`, then configure the build directory once before
running the pipeline:

```bash
mkdir /path/to/curl/build
cd /path/to/curl/build && cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_DEBUG=ON -DBUILD_SHARED_LIBS=OFF
cd /path/to/curl && git checkout curl-8_18_0
```

```python
bug_fix_config = BugFixConfig(
    repo_path="/path/to/curl",
    build_dir="/path/to/curl/build",
    source_branch="curl-8_19_0",
    target_branch="curl-8_18_0",
    source_fix_commit="b35e58b24c",
    bug_description="openssl: fix potential OOB read in debug/verbose logging",
    issue_id="20656",
    disallowed_modules=[],
    port_tests=False,
    build_command="cmake --build /path/to/curl/build -j$(sysctl -n hw.logicalcpu)",
    test_command="ctest --test-dir /path/to/curl/build -j$(sysctl -n hw.logicalcpu)",
    max_build_test_retries=3,
)
```
