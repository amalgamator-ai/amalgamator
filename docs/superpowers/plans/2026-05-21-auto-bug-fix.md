# auto_bug_fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the `auto_bug_fix` module that ports a bug fix from one branch of a C/systems project to another using Claude as the AI engine.

**Architecture:** New `auto_bug_fix/` module sits alongside `auto_code_gen/` and `common/`. `common/` is the shared base (unchanged except for a `clear_repo()` alias). `auto_bug_fix/` has its own per-module prompt copies, config, and orchestrator. All prompt classes use Claude-owns-the-loop via a single `claude_run()` call.

**Tech Stack:** Python 3.11+, Claude Agent SDK (`claude_agent_sdk`), pytest, pytest-mock

---

## File Map

**Create:**
- `pytest.ini` — test runner config
- `tests/__init__.py`
- `tests/common/__init__.py`
- `tests/common/test_utils.py` — tests for `clear_repo()`
- `tests/auto_bug_fix/__init__.py`
- `tests/auto_bug_fix/conftest.py` — shared fixtures
- `tests/auto_bug_fix/test_bug_fix_config.py` — BugFixConfig unit tests
- `tests/auto_bug_fix/test_bug_fix_prompts.py` — all prompt class unit tests
- `tests/auto_bug_fix/test_run_bug_fix.py` — orchestrator unit tests
- `tests/auto_bug_fix/test_bug_fix_behavior.py` — BDD pipeline behavioral tests
- `auto_bug_fix/__init__.py`
- `auto_bug_fix/bug_fix_config.py` — BugFixConfig dataclass + example instance
- `auto_bug_fix/bug_fix_prompts.py` — all 12 prompt classes
- `auto_bug_fix/run_bug_fix.py` — orchestrator

**Modify:**
- `common/utils.py` — add `clear_repo()` alias after `clear_vllm_source_tree()`

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pytest.ini`
- Create: `tests/__init__.py`
- Create: `tests/common/__init__.py`
- Create: `tests/auto_bug_fix/__init__.py`
- Create: `auto_bug_fix/__init__.py`

- [ ] **Step 1: Install test dependencies**

```bash
pip install pytest pytest-mock
```

- [ ] **Step 2: Create pytest.ini**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v
```

- [ ] **Step 3: Create empty init files**

```bash
mkdir -p tests/common tests/auto_bug_fix auto_bug_fix
touch tests/__init__.py tests/common/__init__.py tests/auto_bug_fix/__init__.py auto_bug_fix/__init__.py
```

- [ ] **Step 4: Verify pytest collects zero tests without error**

```bash
pytest --collect-only
```
Expected: `no tests ran` — no errors.

- [ ] **Step 5: Commit**

```bash
git add pytest.ini tests/ auto_bug_fix/__init__.py
git commit -m "feat: scaffold auto_bug_fix module and test infrastructure"
```

---

## Task 2: `clear_repo()` alias in `common/utils.py`

**Files:**
- Test: `tests/common/test_utils.py`
- Modify: `common/utils.py`

- [ ] **Step 1: Write the failing test**

Create `tests/common/test_utils.py`:

```python
import pytest
from common.utils import clear_repo


def test_clear_repo_delegates_to_clear_vllm_source_tree(mocker):
    mock = mocker.patch("common.utils.clear_vllm_source_tree")
    clear_repo("/some/repo/path")
    mock.assert_called_once_with("/some/repo/path")


def test_clear_repo_raises_for_nonexistent_path():
    with pytest.raises(FileNotFoundError):
        clear_repo("/nonexistent/path/abc123")


def test_clear_repo_raises_for_non_git_repo(tmp_path):
    with pytest.raises(RuntimeError, match="Not a git repository"):
        clear_repo(str(tmp_path))
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/common/test_utils.py -v
```
Expected: `ImportError: cannot import name 'clear_repo'`

- [ ] **Step 3: Add `clear_repo()` to `common/utils.py`**

Add after the `clear_vllm_source_tree` function (after line 278):

```python
def clear_repo(path: str | Path) -> None:
    """Reset a git repository to a clean state. Generic alias for clear_vllm_source_tree."""
    clear_vllm_source_tree(path)
```

- [ ] **Step 4: Run to verify all pass**

```bash
pytest tests/common/test_utils.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add common/utils.py tests/common/test_utils.py
git commit -m "feat: add clear_repo() generic alias for clear_vllm_source_tree"
```

---

## Task 3: `BugFixConfig` dataclass

**Files:**
- Test: `tests/auto_bug_fix/test_bug_fix_config.py`
- Create: `tests/auto_bug_fix/conftest.py`
- Create: `auto_bug_fix/bug_fix_config.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/auto_bug_fix/conftest.py`:

```python
import pytest
from common.claude_utils import ClaudeConfig
from auto_bug_fix.bug_fix_config import BugFixConfig


@pytest.fixture
def claude_cfg():
    return ClaudeConfig(
        model="claude-opus-4-6",
        allowed_tools=["Read", "Write", "Bash"],
        perm_mode="acceptEdits",
        cwd="/tmp/test_cwd",
    )


@pytest.fixture
def bug_cfg():
    return BugFixConfig(
        repo_path="/path/to/gcc",
        build_dir="/path/to/gcc/build",
        source_branch="gcc-14-branch",
        target_branch="gcc-13-branch",
        source_fix_commit="abc1234",
        bug_description="CVE-2024-1234: buffer overflow in fold_convert()",
        issue_id="CVE-2024-1234",
        disallowed_modules=["gcc/config/arm/"],
        port_tests=True,
        build_command="make -j$(nproc)",
        test_command="make check -j$(nproc)",
        max_build_test_retries=3,
    )
```

Create `tests/auto_bug_fix/test_bug_fix_config.py`:

```python
import pytest
from auto_bug_fix.bug_fix_config import BugFixConfig


def test_construction_sets_branch_fields(bug_cfg):
    assert bug_cfg.source_branch == "gcc-14-branch"
    assert bug_cfg.target_branch == "gcc-13-branch"
    assert bug_cfg.source_fix_commit == "abc1234"


def test_construction_sets_build_fields(bug_cfg):
    assert bug_cfg.build_command == "make -j$(nproc)"
    assert bug_cfg.test_command == "make check -j$(nproc)"
    assert bug_cfg.build_dir == "/path/to/gcc/build"
    assert bug_cfg.max_build_test_retries == 3


def test_construction_sets_porting_fields(bug_cfg):
    assert bug_cfg.port_tests is True
    assert "gcc/config/arm/" in bug_cfg.disallowed_modules


def test_port_tests_false():
    config = BugFixConfig(
        repo_path="/path/to/gcc",
        build_dir="/path/to/gcc/build",
        source_branch="gcc-14-branch",
        target_branch="gcc-13-branch",
        source_fix_commit="abc1234",
        bug_description="test",
        issue_id="GH-1",
        disallowed_modules=[],
        port_tests=False,
        build_command="make -j$(nproc)",
        test_command="make check -j$(nproc)",
        max_build_test_retries=3,
    )
    assert config.port_tests is False
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/auto_bug_fix/test_bug_fix_config.py -v
```
Expected: `ModuleNotFoundError: No module named 'auto_bug_fix.bug_fix_config'`

- [ ] **Step 3: Create `auto_bug_fix/bug_fix_config.py`**

```python
from dataclasses import dataclass
from common.claude_utils import ClaudeConfig


@dataclass
class BugFixConfig:
    # Repository
    repo_path: str
    build_dir: str

    # Branch identity
    source_branch: str
    target_branch: str
    source_fix_commit: str

    # Bug context
    bug_description: str
    issue_id: str

    # Porting constraints
    disallowed_modules: list[str]
    port_tests: bool

    # Build & test
    build_command: str
    test_command: str
    max_build_test_retries: int


claude_config = ClaudeConfig(
    model="claude-opus-4-6[1m]",
    allowed_tools=["Read", "Write", "Bash"],
    perm_mode="acceptEdits",
    cwd="/path/to/output/dir",
)

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

- [ ] **Step 4: Run to verify all pass**

```bash
pytest tests/auto_bug_fix/test_bug_fix_config.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add auto_bug_fix/bug_fix_config.py tests/auto_bug_fix/conftest.py tests/auto_bug_fix/test_bug_fix_config.py
git commit -m "feat: add BugFixConfig dataclass"
```

---

## Task 4: `create_context_str()` and `CodeTracePrompt`

**Files:**
- Test: `tests/auto_bug_fix/test_bug_fix_prompts.py`
- Create: `auto_bug_fix/bug_fix_prompts.py` (partial — first two items)

- [ ] **Step 1: Write the failing tests**

Create `tests/auto_bug_fix/test_bug_fix_prompts.py`:

```python
import pytest
from auto_bug_fix.bug_fix_prompts import (
    create_context_str,
    gen_CodeTracePrompt,
)


class TestCreateContextStr:
    def test_contains_branch_info(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        assert "gcc-14-branch" in ctx
        assert "gcc-13-branch" in ctx
        assert "abc1234" in ctx

    def test_contains_bug_info(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        assert "CVE-2024-1234" in ctx

    def test_contains_build_info(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        assert "make -j$(nproc)" in ctx
        assert "make check -j$(nproc)" in ctx
        assert "/path/to/gcc/build" in ctx

    def test_no_gpu_language(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        assert "gpu_type" not in ctx
        assert "batch_size" not in ctx
        assert "precision" not in ctx

    def test_contains_cwd(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        assert "/tmp/test_cwd" in ctx


class TestCodeTracePrompt:
    def test_contains_source_branch(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        p = gen_CodeTracePrompt(ctx, bug_cfg.source_branch)
        assert "gcc-14-branch" in p.prompt()

    def test_references_fix_commit(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        p = gen_CodeTracePrompt(ctx, bug_cfg.source_branch)
        assert "source_fix_commit" in p.prompt()

    def test_no_cuda_language(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        p = gen_CodeTracePrompt(ctx, bug_cfg.source_branch)
        text = p.prompt()
        assert "cuda" not in text.lower()
        assert "prefill" not in text.lower()
        assert "decode" not in text.lower()

    def test_output_file_named_after_branch(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        p = gen_CodeTracePrompt(ctx, bug_cfg.source_branch)
        assert "gcc-14-branch" in p.output_file
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/auto_bug_fix/test_bug_fix_prompts.py -v
```
Expected: `ModuleNotFoundError: No module named 'auto_bug_fix.bug_fix_prompts'`

- [ ] **Step 3: Create `auto_bug_fix/bug_fix_prompts.py` with first two items**

```python
from dataclasses import dataclass
from typing import ClassVar

from auto_bug_fix.bug_fix_config import ClaudeConfig, BugFixConfig


def create_context_str(claude_config: ClaudeConfig, config: BugFixConfig) -> str:
    return """
<context>

<cwd>
{cwd}
</cwd>

<repo_path>
{repo_path}
</repo_path>
<build_dir>
{build_dir}
</build_dir>

<source_branch>
{source_branch}
</source_branch>
<target_branch>
{target_branch}
</target_branch>
<source_fix_commit>
{source_fix_commit}
</source_fix_commit>

<bug_description>
{bug_description}
</bug_description>
<issue_id>
{issue_id}
</issue_id>

<build_command>
{build_command}
</build_command>
<test_command>
{test_command}
</test_command>
<max_build_test_retries>
{max_build_test_retries}
</max_build_test_retries>

<disallowed_modules>
{disallowed_modules}
</disallowed_modules>

</context>

<definitions>
<code_trace>
code-paths, code-pieces, and their associated call-chains
</code_trace>
</definitions>

<context_explanations>
- <cwd> is the current working directory for Claude's output files.
- <repo_path> is the absolute path to the cloned repository.
- <build_dir> is the working directory from which <build_command> and <test_command> are invoked.
- <source_branch> is the branch that contains the bug fix to be ported.
- <target_branch> is the branch that needs the fix applied.
- <source_fix_commit> is the specific commit SHA on <source_branch> that introduced the fix.
- <bug_description> describes the bug and the fix.
- <issue_id> is the tracker identifier (CVE, GitHub issue, etc.).
- <build_command> is the incremental build command for <target_branch>. Always use all available CPUs.
- <test_command> is the command to run the relevant test suite on <target_branch>.
- <max_build_test_retries> is the maximum number of build-test-fix loop iterations.
- <disallowed_modules> lists files and directories that must not be modified on <target_branch>.
</context_explanations>

""".format(
        cwd=claude_config.cwd,
        repo_path=config.repo_path,
        build_dir=config.build_dir,
        source_branch=config.source_branch,
        target_branch=config.target_branch,
        source_fix_commit=config.source_fix_commit,
        bug_description=config.bug_description,
        issue_id=config.issue_id,
        build_command=config.build_command,
        test_command=config.test_command,
        max_build_test_retries=config.max_build_test_retries,
        disallowed_modules=config.disallowed_modules,
    )


@dataclass
class CodeTracePrompt:
    context: str
    source_branch: str
    output_file: str
    prompt_template: ClassVar[str] = """

{context}

<definitions>
<source_branch>
{source_branch}
</source_branch>
</definitions>

<instructions>
The goal of this task is to trace the <code_trace> that the fix in <source_fix_commit> touches on <source_branch>. Think hard and follow these guidelines:
- Run `git show <source_fix_commit>` and analyze the complete diff.
- Analyze and understand in detail what the fix does: which files it modifies, which functions it changes, and why.
- Trace the <code_trace> of the changed code: follow call chains from the highest-level entry point down to the lowest-level functions affected.
    - Go deep into C/C++ source if the fix touches compiled code.
    - Include all relevant wrappers, helpers, and data structures.
    - Do not miss any code piece that is part of the fix's scope.
- Document the full <code_trace> step-by-step from top to bottom:
    - For each function in the trace: goal, inputs, outputs, and why it is relevant to the fix.
    - For changed functions: document the before and after behavior.
    - Document classes/objects and their relationships where relevant.
    - Be professional, clear, and concise.
</instructions>

<output>
- Dump results to <cwd>/{output_file}
</output>

"""

    def prompt(self) -> str:
        return self.prompt_template.format(
            context=self.context,
            source_branch=self.source_branch,
            output_file=self.output_file,
        )


CODE_TRACE_FILE = "code_trace.txt"


def gen_CodeTracePrompt(context: str, source_branch: str) -> CodeTracePrompt:
    return CodeTracePrompt(
        context=context,
        source_branch=source_branch,
        output_file="{}_{}".format(source_branch, CODE_TRACE_FILE),
    )
```

- [ ] **Step 4: Run to verify all pass**

```bash
pytest tests/auto_bug_fix/test_bug_fix_prompts.py -v
```
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add auto_bug_fix/bug_fix_prompts.py tests/auto_bug_fix/test_bug_fix_prompts.py
git commit -m "feat: add create_context_str and CodeTracePrompt"
```

---

## Task 5: `TestPortPrompt` (new)

**Files:**
- Test: `tests/auto_bug_fix/test_bug_fix_prompts.py` (append)
- Modify: `auto_bug_fix/bug_fix_prompts.py` (append)

- [ ] **Step 1: Append failing tests**

Add to `tests/auto_bug_fix/test_bug_fix_prompts.py`:

```python
from auto_bug_fix.bug_fix_prompts import gen_TestPortPrompt, TEST_PORT_MANIFEST_FILE


class TestTestPortPrompt:
    def test_references_git_show(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        p = gen_TestPortPrompt(ctx, bug_cfg.source_fix_commit, bug_cfg.target_branch)
        assert "git show" in p.prompt()

    def test_contains_commit_sha(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        p = gen_TestPortPrompt(ctx, bug_cfg.source_fix_commit, bug_cfg.target_branch)
        assert "abc1234" in p.prompt()

    def test_references_manifest_file(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        p = gen_TestPortPrompt(ctx, bug_cfg.source_fix_commit, bug_cfg.target_branch)
        assert TEST_PORT_MANIFEST_FILE in p.prompt()

    def test_output_manifest_is_constant(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        p = gen_TestPortPrompt(ctx, bug_cfg.source_fix_commit, bug_cfg.target_branch)
        assert p.output_manifest_file == TEST_PORT_MANIFEST_FILE
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/auto_bug_fix/test_bug_fix_prompts.py::TestTestPortPrompt -v
```
Expected: `ImportError: cannot import name 'gen_TestPortPrompt'`

- [ ] **Step 3: Append to `auto_bug_fix/bug_fix_prompts.py`**

```python
@dataclass
class TestPortPrompt:
    context: str
    source_fix_commit: str
    target_branch: str
    output_manifest_file: str
    prompt_template: ClassVar[str] = """

{context}

<definitions>
<source_fix_commit>
{source_fix_commit}
</source_fix_commit>
<target_branch>
{target_branch}
</target_branch>
<output_manifest_file>
{output_manifest_file}
</output_manifest_file>
</definitions>

<definition_explanations>
- <source_fix_commit> is the commit SHA on the source branch that introduced the bug fix.
- <target_branch> is the branch that needs the fix ported to it.
- <output_manifest_file> is the file where the paths of all ported test files will be recorded, one path per line.
</definition_explanations>

<instructions>
The goal of this task is to extract test cases added in <source_fix_commit> and port them to <target_branch>'s test infrastructure. Think hard and follow these guidelines:
- Run `git show {source_fix_commit}` and analyze the complete diff.
- Identify all test files added or modified in <source_fix_commit>. Look for files under directories named test/, tests/, testsuite/, t/, or similar. These are the highest-value tests because they were written specifically to catch the bug.
- For each identified test file, inspect <target_branch>'s test infrastructure in detail:
    - Understand the test harness (dejagnu, ctest, pytest, OpenSSL test runner, or other).
    - Understand the suite directory layout, naming conventions, and registration patterns.
    - Understand how new tests are compiled and run.
- Generate a ported version of each test file, adapted to <target_branch>'s test infrastructure:
    - Preserve the test's intent and coverage exactly.
    - Adapt harness-specific syntax, suite registration, file layout, and include paths.
    - Make only the changes required for harness compatibility — do not change test logic.
- Write each ported test file to its correct location on <target_branch>.
- Record the absolute path of every ported test file in <cwd>/<output_manifest_file>, one path per line.
</instructions>

<output>
- Write ported test files to their correct locations.
- Write a manifest of all ported test file paths to <cwd>/{output_manifest_file}, one path per line.
</output>

"""

    def prompt(self) -> str:
        return self.prompt_template.format(
            context=self.context,
            source_fix_commit=self.source_fix_commit,
            target_branch=self.target_branch,
            output_manifest_file=self.output_manifest_file,
        )


TEST_PORT_MANIFEST_FILE = "test_port_manifest.txt"


def gen_TestPortPrompt(
    context: str, source_fix_commit: str, target_branch: str
) -> TestPortPrompt:
    return TestPortPrompt(
        context=context,
        source_fix_commit=source_fix_commit,
        target_branch=target_branch,
        output_manifest_file=TEST_PORT_MANIFEST_FILE,
    )
```

- [ ] **Step 4: Run to verify all pass**

```bash
pytest tests/auto_bug_fix/test_bug_fix_prompts.py -v
```
Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add auto_bug_fix/bug_fix_prompts.py tests/auto_bug_fix/test_bug_fix_prompts.py
git commit -m "feat: add TestPortPrompt (new) — extracts and ports tests from fix commit"
```

---

## Task 6: `CodePortPlanPrompt` + `ReviewCodePortPlanPrompt`

**Files:**
- Test: `tests/auto_bug_fix/test_bug_fix_prompts.py` (append)
- Modify: `auto_bug_fix/bug_fix_prompts.py` (append)

- [ ] **Step 1: Append failing tests**

```python
from auto_bug_fix.bug_fix_prompts import gen_CodePortPlanPrompt, gen_ReviewCodePortPlanPrompt


class TestCodePortPlanPrompt:
    def _make(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        return gen_CodePortPlanPrompt(
            context=ctx,
            branches=[bug_cfg.source_branch, bug_cfg.target_branch],
            branch_code_trace_files=["source_trace.txt"],
            disallowed_modules=bug_cfg.disallowed_modules,
            previous_attempt_file="",
            output_file="plan_V1.txt",
        )

    def test_contains_branches(self, claude_cfg, bug_cfg):
        p = self._make(claude_cfg, bug_cfg)
        assert "gcc-14-branch" in p.prompt()
        assert "gcc-13-branch" in p.prompt()

    def test_contains_disallowed_modules(self, claude_cfg, bug_cfg):
        p = self._make(claude_cfg, bug_cfg)
        assert "gcc/config/arm/" in p.prompt()

    def test_no_cuda_language(self, claude_cfg, bug_cfg):
        p = self._make(claude_cfg, bug_cfg)
        text = p.prompt()
        assert "cuda" not in text.lower()
        assert "kernel vendor" not in text.lower()

    def test_asserts_two_branches(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        with pytest.raises(AssertionError):
            gen_CodePortPlanPrompt(
                context=ctx,
                branches=["only-one"],
                branch_code_trace_files=["trace.txt"],
                disallowed_modules=[],
                previous_attempt_file="",
                output_file="plan.txt",
            )


class TestReviewCodePortPlanPrompt:
    def test_references_plan_and_output_files(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        p = gen_ReviewCodePortPlanPrompt(
            context=ctx,
            branches=[bug_cfg.source_branch, bug_cfg.target_branch],
            branch_code_trace_files=["trace.txt"],
            code_port_plan_file="plan.txt",
            output_review_file="review.txt",
            output_fixed_file="fixed.txt",
            output_total_review_summary_file="evolution.txt",
        )
        text = p.prompt()
        assert "plan.txt" in text
        assert "review.txt" in text
        assert "fixed.txt" in text
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/auto_bug_fix/test_bug_fix_prompts.py::TestCodePortPlanPrompt tests/auto_bug_fix/test_bug_fix_prompts.py::TestReviewCodePortPlanPrompt -v
```
Expected: `ImportError`

- [ ] **Step 3: Append to `auto_bug_fix/bug_fix_prompts.py`**

```python
@dataclass
class CodePortPlanPrompt:
    context: str
    branches: list[str]
    branch_code_trace_files: list[str]
    disallowed_modules: list[str]
    previous_attempt_file: str
    output_file: str
    prompt_template: ClassVar[str] = """

{context}

<definitions>
<branches>
{branches}
</branches>
<branch_code_trace_files>
{branch_code_trace_files}
</branch_code_trace_files>
<disallowed_modules>
{disallowed_modules}
</disallowed_modules>
<previous_attempt_file>
{previous_attempt_file}
</previous_attempt_file>
</definitions>

<definition_explanations>
- <branches> is a list of 2 branches: the first is "source" (has the fix), the second is "target" (needs the fix).
- <branch_code_trace_files> is a list of code trace files for <branches> respectively.
- <disallowed_modules> lists files and directories that must not be modified on the target branch.
- <previous_attempt_file> is the previous iteration's coding plan (if provided), used to improve this attempt.
</definition_explanations>

<instructions>
The goal of this task is to produce a high-level multi-step coding plan for applying the fix from "source" branch to "target" branch. Think hard and follow these guidelines:
- If <previous_attempt_file> exists, read it in full and use all learnings to improve this attempt.
- Analyze the <code_trace> of the fix on "source" branch from <branch_code_trace_files>.
- Analyze the corresponding code on "target" branch, noting where the same logic lives and where it has diverged.
- Compare the two branches step-by-step through the affected call chains. Identify:
    - Code that can be ported AS IS with no changes.
    - Code that requires adaptation due to branch divergence (different APIs, renamed symbols, refactored structure).
    - Code that does not exist on the target branch and must be introduced.
- Produce a step-by-step porting plan that:
    - Maximises the amount of code copied AS IS from "source" to "target".
    - Minimises integration points requiring changes to existing "target" code.
    - Does NOT modify any path under <disallowed_modules>.
    - For each step: state what is ported, what is changed, why the change is necessary, and how to integrate it.
- Verify the plan end-to-end:
    - Check all function APIs are used correctly after porting.
    - Identify hidden dependencies or initialization order constraints.
    - Provide a risk analysis and list of sensitive breaking points.
</instructions>

<output>
- Dump the high-level multi-step coding plan to <cwd>/{output_file}
</output>

"""

    def prompt(self) -> str:
        return self.prompt_template.format(
            context=self.context,
            branches=self.branches,
            branch_code_trace_files=self.branch_code_trace_files,
            disallowed_modules=self.disallowed_modules,
            previous_attempt_file=self.previous_attempt_file,
            output_file=self.output_file,
        )


CODE_PORT_PLAN_FILE_PREFIX = "code_port_plan"


def gen_CodePortPlanPrompt(
    context: str,
    branches: list[str],
    branch_code_trace_files: list[str],
    disallowed_modules: list[str],
    previous_attempt_file: str,
    output_file: str,
) -> CodePortPlanPrompt:
    assert len(branches) == 2
    assert len(branches) == len(branch_code_trace_files)
    return CodePortPlanPrompt(
        context=context,
        branches=branches,
        branch_code_trace_files=branch_code_trace_files,
        disallowed_modules=disallowed_modules,
        previous_attempt_file=previous_attempt_file,
        output_file=output_file,
    )


@dataclass
class ReviewCodePortPlanPrompt:
    context: str
    branches: list[str]
    branch_code_trace_files: list[str]
    code_port_plan_file: str
    output_review_file: str
    output_fixed_file: str
    output_total_review_summary_file: str
    prompt_template: ClassVar[str] = """

{context}

<definitions>
<branches>
{branches}
</branches>
<branch_code_trace_files>
{branch_code_trace_files}
</branch_code_trace_files>
<code_port_plan_file>
{code_port_plan_file}
</code_port_plan_file>
</definitions>

<definition_explanations>
- <branches> is a list of 2 branches: "source" (has the fix) and "target" (needs the fix).
- <branch_code_trace_files> is a list of code trace files for <branches> respectively.
- <code_port_plan_file> is the high-level multi-step coding plan to review.
</definition_explanations>

<instructions>
The goal of this task is to perform a critical, in-depth review of the coding plan in <code_port_plan_file>. Think hard and do the following:
- Understand the plan in detail and restate it step-by-step.
- Review the plan for correctness of the porting logic:
    - Are all changed APIs used correctly on "target" branch after porting?
    - Are all initialization order constraints respected?
    - Are all renamed or refactored symbols on "target" branch handled?
- Review for completeness: missing steps, incomplete porting of affected code paths.
- Review for risks: incorrect assumptions, hidden dependencies, ambiguous steps, missing edge cases.
- For each issue found: document the affected plan step, what is wrong, why it matters, and how to fix it.
- Produce a corrected plan with all issues resolved.
</instructions>

<output>
- Dump the documentation of issues found to <cwd>/{output_review_file}
- Dump the corrected plan to <cwd>/{output_fixed_file}
- If multiple plan => review iterations have been done, summarize the iteration evolution in <cwd>/{output_total_review_summary_file}
</output>

"""

    def prompt(self) -> str:
        return self.prompt_template.format(
            context=self.context,
            branches=self.branches,
            branch_code_trace_files=self.branch_code_trace_files,
            code_port_plan_file=self.code_port_plan_file,
            output_review_file=self.output_review_file,
            output_fixed_file=self.output_fixed_file,
            output_total_review_summary_file=self.output_total_review_summary_file,
        )


def gen_ReviewCodePortPlanPrompt(
    context: str,
    branches: list[str],
    branch_code_trace_files: list[str],
    code_port_plan_file: str,
    output_review_file: str,
    output_fixed_file: str,
    output_total_review_summary_file: str,
) -> ReviewCodePortPlanPrompt:
    assert len(branches) == 2
    assert len(branches) == len(branch_code_trace_files)
    return ReviewCodePortPlanPrompt(
        context=context,
        branches=branches,
        branch_code_trace_files=branch_code_trace_files,
        code_port_plan_file=code_port_plan_file,
        output_review_file=output_review_file,
        output_fixed_file=output_fixed_file,
        output_total_review_summary_file=output_total_review_summary_file,
    )
```

- [ ] **Step 4: Run to verify all pass**

```bash
pytest tests/auto_bug_fix/test_bug_fix_prompts.py -v
```
Expected: 18 passed.

- [ ] **Step 5: Commit**

```bash
git add auto_bug_fix/bug_fix_prompts.py tests/auto_bug_fix/test_bug_fix_prompts.py
git commit -m "feat: add CodePortPlanPrompt and ReviewCodePortPlanPrompt"
```

---

## Task 7: `TestPlanPrompt`, `CodeGenPrompt`, `ReviewCodeGenPrompt`

**Files:**
- Test: `tests/auto_bug_fix/test_bug_fix_prompts.py` (append)
- Modify: `auto_bug_fix/bug_fix_prompts.py` (append)

- [ ] **Step 1: Append failing tests**

```python
from auto_bug_fix.bug_fix_prompts import (
    gen_TestPlanPrompt,
    gen_CodeGenPrompt,
    gen_ReviewCodeGenPrompt,
)


class TestTestPlanPrompt:
    def test_references_manifest(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        p = gen_TestPlanPrompt(
            context=ctx,
            branches=[bug_cfg.source_branch, bug_cfg.target_branch],
            branch_code_trace_files=["trace.txt"],
            code_port_plan_file="plan.txt",
            test_port_manifest_file=TEST_PORT_MANIFEST_FILE,
        )
        assert TEST_PORT_MANIFEST_FILE in p.prompt()

    def test_no_performance_language(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        p = gen_TestPlanPrompt(
            context=ctx,
            branches=[bug_cfg.source_branch, bug_cfg.target_branch],
            branch_code_trace_files=["trace.txt"],
            code_port_plan_file="plan.txt",
            test_port_manifest_file=TEST_PORT_MANIFEST_FILE,
        )
        text = p.prompt()
        assert "speedup" not in text.lower()
        assert "throughput" not in text.lower()


class TestCodeGenPrompt:
    def _make(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        return gen_CodeGenPrompt(
            context=ctx,
            branches=[bug_cfg.source_branch, bug_cfg.target_branch],
            branch_code_trace_files=["trace.txt"],
            code_port_plan_file="plan.txt",
            test_plan_file="test_plan.txt",
            test_port_manifest_file=TEST_PORT_MANIFEST_FILE,
            previous_attempt_file="",
            output_info_file="info.txt",
            output_pr_file="patch.patch",
        )

    def test_references_build_command(self, claude_cfg, bug_cfg):
        p = self._make(claude_cfg, bug_cfg)
        assert "build_command" in p.prompt()

    def test_no_vllm_cmake_steps(self, claude_cfg, bug_cfg):
        p = self._make(claude_cfg, bug_cfg)
        text = p.prompt()
        assert "generate_cmake_presets" not in text
        assert "cmake --preset" not in text

    def test_references_manifest(self, claude_cfg, bug_cfg):
        p = self._make(claude_cfg, bug_cfg)
        assert TEST_PORT_MANIFEST_FILE in p.prompt()


class TestReviewCodeGenPrompt:
    def test_references_pr_and_output_files(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        p = gen_ReviewCodeGenPrompt(
            context=ctx,
            branches=[bug_cfg.source_branch, bug_cfg.target_branch],
            branch_code_trace_files=["trace.txt"],
            code_port_plan_file="plan.txt",
            test_plan_file="test_plan.txt",
            test_port_manifest_file=TEST_PORT_MANIFEST_FILE,
            code_pr_info_file="info.txt",
            code_pr_file="patch.patch",
            output_review_file="review.txt",
            output_fixed_file="fixed.txt",
            output_total_review_summary_file="evolution.txt",
        )
        text = p.prompt()
        assert "patch.patch" in text
        assert "review.txt" in text
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/auto_bug_fix/test_bug_fix_prompts.py::TestTestPlanPrompt tests/auto_bug_fix/test_bug_fix_prompts.py::TestCodeGenPrompt tests/auto_bug_fix/test_bug_fix_prompts.py::TestReviewCodeGenPrompt -v
```
Expected: `ImportError`

- [ ] **Step 3: Append to `auto_bug_fix/bug_fix_prompts.py`**

```python
@dataclass
class TestPlanPrompt:
    context: str
    branches: list[str]
    branch_code_trace_files: list[str]
    code_port_plan_file: str
    test_port_manifest_file: str
    output_file: str
    prompt_template: ClassVar[str] = """

{context}

<definitions>
<branches>
{branches}
</branches>
<branch_code_trace_files>
{branch_code_trace_files}
</branch_code_trace_files>
<code_port_plan_file>
{code_port_plan_file}
</code_port_plan_file>
<test_port_manifest_file>
{test_port_manifest_file}
</test_port_manifest_file>
</definitions>

<definition_explanations>
- <branches> is a list of 2 branches: "source" (has the fix) and "target" (needs the fix).
- <code_port_plan_file> describes the multi-step coding plan for applying the fix.
- <test_port_manifest_file> lists the paths of test files already ported from the fix commit. These are the primary tests.
</definition_explanations>

<instructions>
The goal of this task is to validate that the ported tests in <test_port_manifest_file> give sufficient coverage, and to plan any supplemental tests needed. Think hard and do the following:
- Read <test_port_manifest_file> and inspect each ported test file. Understand what each test covers.
- Analyze the coding plan in <code_port_plan_file> to understand all code paths that will change on "target" branch.
- Identify any code paths, edge cases, or integration points in the plan that are NOT covered by the ported tests.
- For each coverage gap, plan a supplemental test: unit tests for isolated changes, integration tests for multi-component interactions, regression tests for uncovered edge cases.
- If the ported tests already give full coverage, state that explicitly and add no supplemental tests.
- Do not plan performance benchmarks or speed-gain tests.
</instructions>

<output>
- Dump the supplemental test plan to <cwd>/{output_file}
</output>

"""

    def prompt(self) -> str:
        return self.prompt_template.format(
            context=self.context,
            branches=self.branches,
            branch_code_trace_files=self.branch_code_trace_files,
            code_port_plan_file=self.code_port_plan_file,
            test_port_manifest_file=self.test_port_manifest_file,
            output_file=self.output_file,
        )


TEST_PLAN_PREFIX = "test_plan"


def gen_TestPlanPrompt(
    context: str,
    branches: list[str],
    branch_code_trace_files: list[str],
    code_port_plan_file: str,
    test_port_manifest_file: str,
) -> TestPlanPrompt:
    assert len(branches) == 2
    assert len(branches) == len(branch_code_trace_files)
    return TestPlanPrompt(
        context=context,
        branches=branches,
        branch_code_trace_files=branch_code_trace_files,
        code_port_plan_file=code_port_plan_file,
        test_port_manifest_file=test_port_manifest_file,
        output_file="{}_from_{}_to_{}.txt".format(TEST_PLAN_PREFIX, branches[0], branches[1]),
    )


@dataclass
class CodeGenPrompt:
    context: str
    branches: list[str]
    branch_code_trace_files: list[str]
    code_port_plan_file: str
    test_plan_file: str
    test_port_manifest_file: str
    previous_attempt_file: str
    output_info_file: str
    output_pr_file: str
    prompt_template: ClassVar[str] = """

{context}

<definitions>
<branches>
{branches}
</branches>
<branch_code_trace_files>
{branch_code_trace_files}
</branch_code_trace_files>
<code_port_plan_file>
{code_port_plan_file}
</code_port_plan_file>
<test_plan_file>
{test_plan_file}
</test_plan_file>
<test_port_manifest_file>
{test_port_manifest_file}
</test_port_manifest_file>
<previous_attempt_file>
{previous_attempt_file}
</previous_attempt_file>
</definitions>

<definition_explanations>
- <branches> is a list of 2 branches: "source" (has the fix) and "target" (needs the fix).
- <code_port_plan_file> is the high-level multi-step coding plan to implement.
- <test_plan_file> is the supplemental test plan to implement alongside the fix.
- <test_port_manifest_file> lists the paths of test files already ported from the fix commit.
- <previous_attempt_file> is the previous code generation attempt (if provided), used to improve this attempt.
</definition_explanations>

<instructions>
The goal of this task is to generate a code patch for the "target" branch that implements the coding plan and passes all tests. Think hard and do the following:
- If <previous_attempt_file> exists, read it and use all learnings to improve this attempt.
- Analyze and understand the coding plan in <code_port_plan_file> in full detail.
- Read the ported test files listed in <test_port_manifest_file>.
- Implement the plans in <code_port_plan_file> and <test_plan_file> EXACTLY as described. Do not diverge.
- After applying the fix, compile "target" using <build_command> from <build_dir>.
    - Use all available CPUs.
    - Use incremental compilation. Do NOT do a full clean rebuild unless a stale artifact error explicitly requires it.
    - If compilation fails, investigate, fix, and recompile. Do not stop until compilation succeeds.
- Run all tests from <test_port_manifest_file> and <test_plan_file>. Do not skip any.
    - If tests fail, fix the issue and rerun. Do not stop until all tests pass.
- Ensure all code is written professionally and consistently with "target" branch conventions.
</instructions>

<output>
- Dump an explanation of the code patch to <cwd>/{output_info_file}
- Dump the code patch (fix code + all tests) to <cwd>/{output_pr_file}
</output>

"""

    def prompt(self) -> str:
        return self.prompt_template.format(
            context=self.context,
            branches=self.branches,
            branch_code_trace_files=self.branch_code_trace_files,
            code_port_plan_file=self.code_port_plan_file,
            test_plan_file=self.test_plan_file,
            test_port_manifest_file=self.test_port_manifest_file,
            previous_attempt_file=self.previous_attempt_file,
            output_info_file=self.output_info_file,
            output_pr_file=self.output_pr_file,
        )


CODE_GEN_FILE_PREFIX = "code_gen"


def gen_CodeGenPrompt(
    context: str,
    branches: list[str],
    branch_code_trace_files: list[str],
    code_port_plan_file: str,
    test_plan_file: str,
    test_port_manifest_file: str,
    previous_attempt_file: str,
    output_info_file: str,
    output_pr_file: str,
) -> CodeGenPrompt:
    assert len(branches) == 2
    assert len(branches) == len(branch_code_trace_files)
    return CodeGenPrompt(
        context=context,
        branches=branches,
        branch_code_trace_files=branch_code_trace_files,
        code_port_plan_file=code_port_plan_file,
        test_plan_file=test_plan_file,
        test_port_manifest_file=test_port_manifest_file,
        previous_attempt_file=previous_attempt_file,
        output_info_file=output_info_file,
        output_pr_file=output_pr_file,
    )


@dataclass
class ReviewCodeGenPrompt:
    context: str
    branches: list[str]
    branch_code_trace_files: list[str]
    code_port_plan_file: str
    test_plan_file: str
    test_port_manifest_file: str
    code_pr_info_file: str
    code_pr_file: str
    output_review_file: str
    output_fixed_file: str
    output_total_review_summary_file: str
    prompt_template: ClassVar[str] = """

{context}

<definitions>
<branches>
{branches}
</branches>
<branch_code_trace_files>
{branch_code_trace_files}
</branch_code_trace_files>
<code_port_plan_file>
{code_port_plan_file}
</code_port_plan_file>
<test_plan_file>
{test_plan_file}
</test_plan_file>
<test_port_manifest_file>
{test_port_manifest_file}
</test_port_manifest_file>
<code_pr_info_file>
{code_pr_info_file}
</code_pr_info_file>
<code_pr_file>
{code_pr_file}
</code_pr_file>
</definitions>

<definition_explanations>
- <branches> is a list of 2 branches: "source" (has the fix) and "target" (needs the fix).
- <code_port_plan_file> is the coding plan that was implemented.
- <test_plan_file> is the supplemental test plan that was implemented.
- <test_port_manifest_file> lists the ported test file paths.
- <code_pr_file> is the code patch to review.
- <code_pr_info_file> describes <code_pr_file>.
</definition_explanations>

<instructions>
The goal of this task is to perform a critical, in-depth review of the code patch in <code_pr_file>. Think hard and do the following:
- Understand the code patch in full detail. Restate the changes step-by-step.
- Review for correctness: does the patch fully implement the coding plan? Are all ported APIs used correctly?
- Review for completeness: missing steps, coverage gaps in the tests.
- Review for risks: incorrect assumptions, missing edge cases, hidden dependencies.
- For each issue: document the affected code, what is wrong, why it matters, how to fix it.
- Produce a corrected patch with all issues resolved.
- Add new tests to cover any newly found issues.
</instructions>

<output>
- Dump the documentation of issues found to <cwd>/{output_review_file}
- Dump the corrected code patch to <cwd>/{output_fixed_file}
- If multiple code gen => review iterations have been done, summarize the evolution in <cwd>/{output_total_review_summary_file}
</output>

"""

    def prompt(self) -> str:
        return self.prompt_template.format(
            context=self.context,
            branches=self.branches,
            branch_code_trace_files=self.branch_code_trace_files,
            code_port_plan_file=self.code_port_plan_file,
            test_plan_file=self.test_plan_file,
            test_port_manifest_file=self.test_port_manifest_file,
            code_pr_info_file=self.code_pr_info_file,
            code_pr_file=self.code_pr_file,
            output_review_file=self.output_review_file,
            output_fixed_file=self.output_fixed_file,
            output_total_review_summary_file=self.output_total_review_summary_file,
        )


def gen_ReviewCodeGenPrompt(
    context: str,
    branches: list[str],
    branch_code_trace_files: list[str],
    code_port_plan_file: str,
    test_plan_file: str,
    test_port_manifest_file: str,
    code_pr_info_file: str,
    code_pr_file: str,
    output_review_file: str,
    output_fixed_file: str,
    output_total_review_summary_file: str,
) -> ReviewCodeGenPrompt:
    assert len(branches) == 2
    assert len(branches) == len(branch_code_trace_files)
    return ReviewCodeGenPrompt(
        context=context,
        branches=branches,
        branch_code_trace_files=branch_code_trace_files,
        code_port_plan_file=code_port_plan_file,
        test_plan_file=test_plan_file,
        test_port_manifest_file=test_port_manifest_file,
        code_pr_info_file=code_pr_info_file,
        code_pr_file=code_pr_file,
        output_review_file=output_review_file,
        output_fixed_file=output_fixed_file,
        output_total_review_summary_file=output_total_review_summary_file,
    )
```

- [ ] **Step 4: Run to verify all pass**

```bash
pytest tests/auto_bug_fix/test_bug_fix_prompts.py -v
```
Expected: 25 passed.

- [ ] **Step 5: Commit**

```bash
git add auto_bug_fix/bug_fix_prompts.py tests/auto_bug_fix/test_bug_fix_prompts.py
git commit -m "feat: add TestPlanPrompt, CodeGenPrompt, ReviewCodeGenPrompt"
```

---

## Task 8: `RunAndFixPrompt` (new)

**Files:**
- Test: `tests/auto_bug_fix/test_bug_fix_prompts.py` (append)
- Modify: `auto_bug_fix/bug_fix_prompts.py` (append)

- [ ] **Step 1: Append failing tests**

```python
from auto_bug_fix.bug_fix_prompts import gen_RunAndFixPrompt, RUN_AND_FIX_FAILURE_FILE


class TestRunAndFixPrompt:
    def _make(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        return gen_RunAndFixPrompt(
            context=ctx,
            build_command=bug_cfg.build_command,
            test_command=bug_cfg.test_command,
            build_dir=bug_cfg.build_dir,
            max_build_test_retries=bug_cfg.max_build_test_retries,
        )

    def test_contains_build_command(self, claude_cfg, bug_cfg):
        p = self._make(claude_cfg, bug_cfg)
        assert "make -j$(nproc)" in p.prompt()

    def test_contains_test_command(self, claude_cfg, bug_cfg):
        p = self._make(claude_cfg, bug_cfg)
        assert "make check -j$(nproc)" in p.prompt()

    def test_references_failure_file(self, claude_cfg, bug_cfg):
        p = self._make(claude_cfg, bug_cfg)
        assert RUN_AND_FIX_FAILURE_FILE in p.prompt()

    def test_contains_retry_limit(self, claude_cfg, bug_cfg):
        p = self._make(claude_cfg, bug_cfg)
        assert "3" in p.prompt()

    def test_mentions_incremental_build(self, claude_cfg, bug_cfg):
        p = self._make(claude_cfg, bug_cfg)
        text = p.prompt()
        assert "incremental" in text.lower() or "stale artifact" in text.lower()

    def test_output_failure_file_is_constant(self, claude_cfg, bug_cfg):
        p = self._make(claude_cfg, bug_cfg)
        assert p.output_failure_file == RUN_AND_FIX_FAILURE_FILE
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/auto_bug_fix/test_bug_fix_prompts.py::TestRunAndFixPrompt -v
```
Expected: `ImportError`

- [ ] **Step 3: Append to `auto_bug_fix/bug_fix_prompts.py`**

```python
@dataclass
class RunAndFixPrompt:
    context: str
    build_command: str
    test_command: str
    build_dir: str
    max_build_test_retries: int
    output_failure_file: str
    prompt_template: ClassVar[str] = """

{context}

<definitions>
<build_command>
{build_command}
</build_command>
<test_command>
{test_command}
</test_command>
<build_dir>
{build_dir}
</build_dir>
<max_build_test_retries>
{max_build_test_retries}
</max_build_test_retries>
<output_failure_file>
{output_failure_file}
</output_failure_file>
</definitions>

<definition_explanations>
- <build_command> is the shell command to compile the target branch incrementally. Run from <build_dir>.
- <test_command> is the shell command to run the relevant test suite. Run from <build_dir>.
- <build_dir> is the working directory from which both commands are invoked.
- <max_build_test_retries> is the maximum number of build-test-fix iterations before stopping.
- <output_failure_file> is where to write the failure log if retries are exhausted.
</definition_explanations>

<instructions>
The goal of this task is to autonomously drive a build-test-fix loop until both the build and tests are clean. Think hard and follow these guidelines:
- Run <build_command> from <build_dir>.
    - Use all available CPUs (the command already specifies this).
    - If the build succeeds, proceed to running the tests.
    - If the build fails:
        - Inspect the compiler/linker output carefully to identify the root cause.
        - Apply a targeted fix to the source code.
        - Do NOT do a full clean rebuild unless the error is explicitly caused by a stale artifact. Never do a speculative clean rebuild.
        - Retry the build.
- On a successful build, run <test_command> from <build_dir>.
    - If all tests pass, the loop is complete. Report success.
    - If tests fail:
        - Inspect the test output carefully.
        - Identify the root cause — may be in the fix code or in the ported tests.
        - Apply a targeted fix.
        - Recompile using <build_command> (incremental).
        - Rerun <test_command>.
- Count each complete build+test attempt as one retry. Repeat until clean or <max_build_test_retries> is exhausted.
- If retries are exhausted:
    - Write the final build/test output and a summary of all attempted fixes to <cwd>/<output_failure_file>.
    - Stop. Do not attempt further fixes.
</instructions>

<output>
- If successful: report that both build and tests are clean, with a summary of fixes applied.
- If retries exhausted: write the failure summary to <cwd>/{output_failure_file} and stop.
</output>

"""

    def prompt(self) -> str:
        return self.prompt_template.format(
            context=self.context,
            build_command=self.build_command,
            test_command=self.test_command,
            build_dir=self.build_dir,
            max_build_test_retries=self.max_build_test_retries,
            output_failure_file=self.output_failure_file,
        )


RUN_AND_FIX_FAILURE_FILE = "run_and_fix_failure.txt"


def gen_RunAndFixPrompt(
    context: str,
    build_command: str,
    test_command: str,
    build_dir: str,
    max_build_test_retries: int,
) -> RunAndFixPrompt:
    return RunAndFixPrompt(
        context=context,
        build_command=build_command,
        test_command=test_command,
        build_dir=build_dir,
        max_build_test_retries=max_build_test_retries,
        output_failure_file=RUN_AND_FIX_FAILURE_FILE,
    )
```

- [ ] **Step 4: Run to verify all pass**

```bash
pytest tests/auto_bug_fix/test_bug_fix_prompts.py -v
```
Expected: 31 passed.

- [ ] **Step 5: Commit**

```bash
git add auto_bug_fix/bug_fix_prompts.py tests/auto_bug_fix/test_bug_fix_prompts.py
git commit -m "feat: add RunAndFixPrompt (new) — autonomous build-test-fix loop"
```

---

## Task 9: `InvestigateIssuePrompt`, `ReviewInvestigatedIssuePrompt`, `WorkItemsPrompt`

**Files:**
- Test: `tests/auto_bug_fix/test_bug_fix_prompts.py` (append)
- Modify: `auto_bug_fix/bug_fix_prompts.py` (append)

- [ ] **Step 1: Append failing tests**

```python
from auto_bug_fix.bug_fix_prompts import (
    gen_InvestigateIssuePrompt,
    gen_ReviewInvestigatedIssuePrompt,
    gen_WorkItemsPrompt,
)


def _investigate_prompt(claude_cfg, bug_cfg):
    ctx = create_context_str(claude_cfg, bug_cfg)
    branches = [bug_cfg.source_branch, bug_cfg.target_branch]
    return gen_InvestigateIssuePrompt(
        context=ctx,
        branches=branches,
        branch_code_trace_files=["trace.txt"],
        code_port_plan_file="plan.txt",
        test_plan_file="test_plan.txt",
        code_port_plan_review_evolution_file="plan_evolution.txt",
        code_pr_info_file="info.txt",
        code_pr_file="patch.patch",
        code_pr_review_evolution_file="code_evolution.txt",
        issue_desc_file="issue.txt",
        issue_fix_previous_attempt_file="",
        issue_fix_previous_attempt_review_evolution_file="",
        issue_fix_file="fix.txt",
        code_pr_fixed_file="fixed.patch",
    )


class TestInvestigateIssuePrompt:
    def test_references_issue_and_fix_files(self, claude_cfg, bug_cfg):
        p = _investigate_prompt(claude_cfg, bug_cfg)
        text = p.prompt()
        assert "issue.txt" in text
        assert "fix.txt" in text

    def test_no_transformer_block_language(self, claude_cfg, bug_cfg):
        p = _investigate_prompt(claude_cfg, bug_cfg)
        text = p.prompt()
        assert "transformer block" not in text.lower()
        assert "median block" not in text.lower()


class TestReviewInvestigatedIssuePrompt:
    def test_references_fix_and_review_files(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        branches = [bug_cfg.source_branch, bug_cfg.target_branch]
        p = gen_ReviewInvestigatedIssuePrompt(
            context=ctx,
            branches=branches,
            branch_code_trace_files=["trace.txt"],
            code_port_plan_file="plan.txt",
            test_plan_file="test_plan.txt",
            code_port_plan_review_evolution_file="plan_evolution.txt",
            code_pr_info_file="info.txt",
            code_pr_file="patch.patch",
            code_pr_review_evolution_file="code_evolution.txt",
            issue_desc_file="issue.txt",
            issue_fix_file="fix.txt",
            issue_fix_review_file="fix_review.txt",
            issue_fix_fixed_file="fix_fixed.txt",
            issue_fix_review_evolution_file="fix_evolution.txt",
            code_pr_review_fixed_file="pr_fixed.patch",
        )
        text = p.prompt()
        assert "fix.txt" in text
        assert "fix_review.txt" in text


class TestWorkItemsPrompt:
    def test_references_work_items_and_code_gen_dir(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        p = gen_WorkItemsPrompt(
            context=ctx,
            code_gen_dir="/tmp/code_gen",
            work_items_file="work_items.txt",
        )
        text = p.prompt()
        assert "work_items.txt" in text
        assert "/tmp/code_gen" in text
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/auto_bug_fix/test_bug_fix_prompts.py::TestInvestigateIssuePrompt tests/auto_bug_fix/test_bug_fix_prompts.py::TestReviewInvestigatedIssuePrompt tests/auto_bug_fix/test_bug_fix_prompts.py::TestWorkItemsPrompt -v
```
Expected: `ImportError`

- [ ] **Step 3: Append to `auto_bug_fix/bug_fix_prompts.py`**

```python
@dataclass
class InvestigateIssuePrompt:
    context: str
    branches: list[str]
    branch_code_trace_files: list[str]
    code_port_plan_file: str
    test_plan_file: str
    code_port_plan_review_evolution_file: str
    code_pr_info_file: str
    code_pr_file: str
    code_pr_review_evolution_file: str
    issue_desc_file: str
    issue_fix_previous_attempt_file: str
    issue_fix_previous_attempt_review_evolution_file: str
    issue_fix_file: str
    code_pr_fixed_file: str
    prompt_template: ClassVar[str] = """

{context}

<definitions>
<branches>
{branches}
</branches>
<branch_code_trace_files>
{branch_code_trace_files}
</branch_code_trace_files>
<code_port_plan_file>
{code_port_plan_file}
</code_port_plan_file>
<test_plan_file>
{test_plan_file}
</test_plan_file>
<code_port_plan_review_evolution_file>
{code_port_plan_review_evolution_file}
</code_port_plan_review_evolution_file>
<code_pr_info_file>
{code_pr_info_file}
</code_pr_info_file>
<code_pr_file>
{code_pr_file}
</code_pr_file>
<code_pr_review_evolution_file>
{code_pr_review_evolution_file}
</code_pr_review_evolution_file>
<issue_desc_file>
{issue_desc_file}
</issue_desc_file>
<issue_fix_previous_attempt_file>
{issue_fix_previous_attempt_file}
</issue_fix_previous_attempt_file>
<issue_fix_previous_attempt_review_evolution_file>
{issue_fix_previous_attempt_review_evolution_file}
</issue_fix_previous_attempt_review_evolution_file>
</definitions>

<definition_explanations>
- <branches> is a list of 2 branches: "source" (has the fix) and "target" (needs the fix).
- <issue_desc_file> describes the issue that needs investigation.
- <issue_fix_previous_attempt_file> and <issue_fix_previous_attempt_review_evolution_file> describe any previous fix attempt.
</definition_explanations>

<instructions>
The goal of this task is to investigate the issue in <issue_desc_file> and produce a fix. Think hard and do the following:
- If previous fix attempt files exist, read them and use all learnings. The current attempt is done from scratch but informed by them.
- Read all previous issue files in <cwd> to avoid repeating mistakes.
- Analyze the coding plan, review evolution, and code patch to understand the full context.
- Detect and analyze the root cause of the issue on "target" branch:
    - Dive deep into source code of both branches.
    - Trace all relevant call chains end-to-end.
    - Understand why the issue appears on "target" but not "source".
- Provide a detailed explanation: why it happens, key root causes with source references, and steps to fix.
- Apply the fix, add new tests to verify the issue is resolved, and run all tests.
- Do NOT stop until the issue is fixed and all tests pass.
</instructions>

<output>
- Dump the issue analysis and fix plan to <cwd>/{issue_fix_file}
- Add new tests to verify the fix.
- Dump the fixed code patch to <cwd>/{code_pr_fixed_file}
- Apply the new patch to "target" source code.
- Run all tests and ensure ALL PASS. If any fail, fix and rerun.
</output>

"""

    def prompt(self) -> str:
        return self.prompt_template.format(
            context=self.context,
            branches=self.branches,
            branch_code_trace_files=self.branch_code_trace_files,
            code_port_plan_file=self.code_port_plan_file,
            test_plan_file=self.test_plan_file,
            code_port_plan_review_evolution_file=self.code_port_plan_review_evolution_file,
            code_pr_info_file=self.code_pr_info_file,
            code_pr_file=self.code_pr_file,
            code_pr_review_evolution_file=self.code_pr_review_evolution_file,
            issue_desc_file=self.issue_desc_file,
            issue_fix_previous_attempt_file=self.issue_fix_previous_attempt_file,
            issue_fix_previous_attempt_review_evolution_file=self.issue_fix_previous_attempt_review_evolution_file,
            issue_fix_file=self.issue_fix_file,
            code_pr_fixed_file=self.code_pr_fixed_file,
        )


def gen_InvestigateIssuePrompt(
    context: str,
    branches: list[str],
    branch_code_trace_files: list[str],
    code_port_plan_file: str,
    test_plan_file: str,
    code_port_plan_review_evolution_file: str,
    code_pr_info_file: str,
    code_pr_file: str,
    code_pr_review_evolution_file: str,
    issue_desc_file: str,
    issue_fix_previous_attempt_file: str,
    issue_fix_previous_attempt_review_evolution_file: str,
    issue_fix_file: str,
    code_pr_fixed_file: str,
) -> InvestigateIssuePrompt:
    assert len(branches) == 2
    assert len(branches) == len(branch_code_trace_files)
    return InvestigateIssuePrompt(
        context=context,
        branches=branches,
        branch_code_trace_files=branch_code_trace_files,
        code_port_plan_file=code_port_plan_file,
        test_plan_file=test_plan_file,
        code_port_plan_review_evolution_file=code_port_plan_review_evolution_file,
        code_pr_info_file=code_pr_info_file,
        code_pr_file=code_pr_file,
        code_pr_review_evolution_file=code_pr_review_evolution_file,
        issue_desc_file=issue_desc_file,
        issue_fix_previous_attempt_file=issue_fix_previous_attempt_file,
        issue_fix_previous_attempt_review_evolution_file=issue_fix_previous_attempt_review_evolution_file,
        issue_fix_file=issue_fix_file,
        code_pr_fixed_file=code_pr_fixed_file,
    )


@dataclass
class ReviewInvestigatedIssuePrompt:
    context: str
    branches: list[str]
    branch_code_trace_files: list[str]
    code_port_plan_file: str
    test_plan_file: str
    code_port_plan_review_evolution_file: str
    code_pr_info_file: str
    code_pr_file: str
    code_pr_review_evolution_file: str
    issue_desc_file: str
    issue_fix_file: str
    issue_fix_review_file: str
    issue_fix_fixed_file: str
    issue_fix_review_evolution_file: str
    code_pr_review_fixed_file: str
    prompt_template: ClassVar[str] = """

{context}

<definitions>
<branches>
{branches}
</branches>
<issue_desc_file>
{issue_desc_file}
</issue_desc_file>
<issue_fix_file>
{issue_fix_file}
</issue_fix_file>
</definitions>

<instructions>
The goal of this task is to critically review the issue fix in <issue_fix_file>. Think hard and do the following:
- Analyze the issue in <issue_desc_file> and the fix in <issue_fix_file> in full detail.
- Review the fix for: incorrect root cause assumptions, missing steps, ambiguity, missing edge cases, hidden dependencies.
- For each problem found: document the affected part, what is wrong, why it matters, and how to fix it.
- Produce a corrected issue fix with all problems resolved.
- Apply the corrected patch, run all tests, and ensure they all pass.
</instructions>

<output>
- Dump the review of the fix to <cwd>/{issue_fix_review_file}
- Dump the corrected fix to <cwd>/{issue_fix_fixed_file}
- Dump the corrected code patch to <cwd>/{code_pr_review_fixed_file}. Add new tests if needed. Apply and run.
- Summarize the iteration evolution in <cwd>/{issue_fix_review_evolution_file}
</output>

"""

    def prompt(self) -> str:
        return self.prompt_template.format(
            context=self.context,
            branches=self.branches,
            branch_code_trace_files=self.branch_code_trace_files,
            code_port_plan_file=self.code_port_plan_file,
            test_plan_file=self.test_plan_file,
            code_port_plan_review_evolution_file=self.code_port_plan_review_evolution_file,
            code_pr_info_file=self.code_pr_info_file,
            code_pr_file=self.code_pr_file,
            code_pr_review_evolution_file=self.code_pr_review_evolution_file,
            issue_desc_file=self.issue_desc_file,
            issue_fix_file=self.issue_fix_file,
            issue_fix_review_file=self.issue_fix_review_file,
            issue_fix_fixed_file=self.issue_fix_fixed_file,
            issue_fix_review_evolution_file=self.issue_fix_review_evolution_file,
            code_pr_review_fixed_file=self.code_pr_review_fixed_file,
        )


def gen_ReviewInvestigatedIssuePrompt(
    context: str,
    branches: list[str],
    branch_code_trace_files: list[str],
    code_port_plan_file: str,
    test_plan_file: str,
    code_port_plan_review_evolution_file: str,
    code_pr_info_file: str,
    code_pr_file: str,
    code_pr_review_evolution_file: str,
    issue_desc_file: str,
    issue_fix_file: str,
    issue_fix_review_file: str,
    issue_fix_fixed_file: str,
    issue_fix_review_evolution_file: str,
    code_pr_review_fixed_file: str,
) -> ReviewInvestigatedIssuePrompt:
    assert len(branches) == 2
    assert len(branches) == len(branch_code_trace_files)
    return ReviewInvestigatedIssuePrompt(
        context=context,
        branches=branches,
        branch_code_trace_files=branch_code_trace_files,
        code_port_plan_file=code_port_plan_file,
        test_plan_file=test_plan_file,
        code_port_plan_review_evolution_file=code_port_plan_review_evolution_file,
        code_pr_info_file=code_pr_info_file,
        code_pr_file=code_pr_file,
        code_pr_review_evolution_file=code_pr_review_evolution_file,
        issue_desc_file=issue_desc_file,
        issue_fix_file=issue_fix_file,
        issue_fix_review_file=issue_fix_review_file,
        issue_fix_fixed_file=issue_fix_fixed_file,
        issue_fix_review_evolution_file=issue_fix_review_evolution_file,
        code_pr_review_fixed_file=code_pr_review_fixed_file,
    )


@dataclass
class WorkItemsPrompt:
    context: str
    code_gen_dir: str
    work_items_file: str
    prompt_template: ClassVar[str] = """

{context}

<definitions>
<code_gen_dir>
{code_gen_dir}
</code_gen_dir>
<work_items_file>
{work_items_file}
</work_items_file>
</definitions>

<definition_explanations>
- <code_gen_dir> is a directory holding the results of the bug fix porting process: code traces, port plans, code gen iterations, and issue fix sequences.
- <work_items_file> describes the work items to execute.
</definition_explanations>

<instructions>
The goal of this task is to execute the work items in <work_items_file>. Think hard and do the following:
- Read all result files in <code_gen_dir> to understand the full porting process. Take all learnings from review evolutions and issue fixing sequences.
- Read the "context section" and "work_items section" from <work_items_file>.
- Execute each work item one by one:
    - Find, analyze, and understand any relevant data needed to execute it.
    - For complex items, split into smaller steps, execute each, and verify before proceeding.
    - Do a final critical review and fix issues before marking each item done.
</instructions>

<output>
- Read the "output section" from <work_items_file> and generate the outputs described there.
</output>

"""

    def prompt(self) -> str:
        return self.prompt_template.format(
            context=self.context,
            code_gen_dir=self.code_gen_dir,
            work_items_file=self.work_items_file,
        )


def gen_WorkItemsPrompt(
    context: str,
    code_gen_dir: str,
    work_items_file: str,
) -> WorkItemsPrompt:
    return WorkItemsPrompt(
        context=context,
        code_gen_dir=code_gen_dir,
        work_items_file=work_items_file,
    )
```

- [ ] **Step 4: Run to verify all pass**

```bash
pytest tests/auto_bug_fix/test_bug_fix_prompts.py -v
```
Expected: 37 passed.

- [ ] **Step 5: Commit**

```bash
git add auto_bug_fix/bug_fix_prompts.py tests/auto_bug_fix/test_bug_fix_prompts.py
git commit -m "feat: add InvestigateIssuePrompt, ReviewInvestigatedIssuePrompt, WorkItemsPrompt"
```

---

## Task 10: `run_bug_fix.py` orchestrator

**Files:**
- Test: `tests/auto_bug_fix/test_run_bug_fix.py`
- Create: `auto_bug_fix/run_bug_fix.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/auto_bug_fix/test_run_bug_fix.py`:

```python
import pytest
from common.claude_utils import ClaudeConfig
from auto_bug_fix.bug_fix_config import BugFixConfig
from auto_bug_fix.run_bug_fix import gen_prompts, NUM_PLAN_REVIEWS, NUM_CODE_REVIEWS
from auto_bug_fix.bug_fix_prompts import RUN_AND_FIX_FAILURE_FILE, TEST_PORT_MANIFEST_FILE


def _claude_cfg():
    return ClaudeConfig(
        model="claude-opus-4-6",
        allowed_tools=["Read", "Write", "Bash"],
        perm_mode="acceptEdits",
        cwd="/tmp/test_cwd",
    )


def _config(**overrides):
    defaults = dict(
        repo_path="/path/to/gcc",
        build_dir="/path/to/gcc/build",
        source_branch="gcc-14-branch",
        target_branch="gcc-13-branch",
        source_fix_commit="abc1234",
        bug_description="CVE-2024-1234: test bug",
        issue_id="CVE-2024-1234",
        disallowed_modules=[],
        port_tests=True,
        build_command="make -j$(nproc)",
        test_command="make check -j$(nproc)",
        max_build_test_retries=3,
    )
    defaults.update(overrides)
    return BugFixConfig(**defaults)


def test_gen_prompts_returns_list():
    prompts = gen_prompts(_claude_cfg(), _config())
    assert isinstance(prompts, list)
    assert len(prompts) > 0


def test_first_entry_is_clear_repo_cmd():
    prompts = gen_prompts(_claude_cfg(), _config())
    assert isinstance(prompts[0], dict)
    assert "clear_repo" in prompts[0]["cmd"]


def test_run_and_fix_is_last_string_prompt():
    prompts = gen_prompts(_claude_cfg(), _config())
    string_prompts = [p for p in prompts if isinstance(p, str)]
    assert RUN_AND_FIX_FAILURE_FILE in string_prompts[-1]


def test_includes_git_show_when_port_tests_true():
    prompts = gen_prompts(_claude_cfg(), _config(port_tests=True))
    string_prompts = [p for p in prompts if isinstance(p, str)]
    assert any("git show" in p for p in string_prompts)


def test_excludes_git_show_when_port_tests_false():
    prompts = gen_prompts(_claude_cfg(), _config(port_tests=False))
    string_prompts = [p for p in prompts if isinstance(p, str)]
    assert not any("git show" in p for p in string_prompts)


def test_plan_review_iteration_count():
    prompts = gen_prompts(_claude_cfg(), _config())
    string_prompts = [p for p in prompts if isinstance(p, str)]
    # Each iteration produces plan + review prompt; identify by unique plan output filename
    plan_prompts = [p for p in string_prompts if "code_port_plan" in p and "output_file" not in p]
    # At minimum NUM_PLAN_REVIEWS plan prompts exist
    assert len([p for p in string_prompts if "high-level multi-step coding plan" in p]) >= NUM_PLAN_REVIEWS


def test_manifest_referenced_in_code_gen():
    prompts = gen_prompts(_claude_cfg(), _config())
    string_prompts = [p for p in prompts if isinstance(p, str)]
    code_gen_prompts = [p for p in string_prompts if "generate a code patch" in p.lower()]
    assert any(TEST_PORT_MANIFEST_FILE in p for p in code_gen_prompts)


def test_build_command_appears_in_run_and_fix():
    config = _config(
        build_command="cmake --build build -j$(nproc)",
        test_command="ctest --output-on-failure",
    )
    prompts = gen_prompts(_claude_cfg(), config)
    string_prompts = [p for p in prompts if isinstance(p, str)]
    run_and_fix = string_prompts[-1]
    assert "cmake --build build -j$(nproc)" in run_and_fix
    assert "ctest --output-on-failure" in run_and_fix
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/auto_bug_fix/test_run_bug_fix.py -v
```
Expected: `ModuleNotFoundError: No module named 'auto_bug_fix.run_bug_fix'`

- [ ] **Step 3: Create `auto_bug_fix/run_bug_fix.py`**

```python
import sys
import time
import asyncio

from common.utils import Tee, clear_repo
from common.claude_utils import ClaudeConfig, claude_run

from auto_bug_fix.bug_fix_config import BugFixConfig
from auto_bug_fix.bug_fix_prompts import (
    create_context_str,
    gen_CodeTracePrompt,
    gen_TestPortPrompt,
    gen_CodePortPlanPrompt,
    gen_ReviewCodePortPlanPrompt,
    gen_TestPlanPrompt,
    gen_CodeGenPrompt,
    gen_ReviewCodeGenPrompt,
    gen_RunAndFixPrompt,
    CODE_PORT_PLAN_FILE_PREFIX,
    CODE_GEN_FILE_PREFIX,
    TEST_PORT_MANIFEST_FILE,
    RUN_AND_FIX_FAILURE_FILE,
)

LOG_FILE = "__run_log_bug_fix.txt"
NUM_PLAN_REVIEWS = 4
NUM_CODE_REVIEWS = 3


def gen_prompts(
    claude_config: ClaudeConfig | None = None,
    config: BugFixConfig | None = None,
) -> list:
    if claude_config is None:
        from auto_bug_fix.bug_fix_config import claude_config as _default
        claude_config = _default
    if config is None:
        from auto_bug_fix.bug_fix_config import bug_fix_config as _default
        config = _default

    clear_repo_cmd = {"cmd": 'clear_repo("{}")'.format(config.repo_path)}
    context = create_context_str(claude_config, config)
    branches = [config.source_branch, config.target_branch]

    # Step 2: Code trace on source branch
    code_trace_prompt = gen_CodeTracePrompt(context=context, source_branch=config.source_branch)
    branch_code_trace_files = [code_trace_prompt.output_file]

    # Step 3: Port tests from fix commit
    test_port_prompt = gen_TestPortPrompt(
        context=context,
        source_fix_commit=config.source_fix_commit,
        target_branch=config.target_branch,
    )

    # Step 4: CodePortPlan iterations
    code_port_plan_and_review_prompts = []
    previous_attempt_file = ""
    for i in range(NUM_PLAN_REVIEWS):
        plan_file = "{}_V{}_from_{}_to_{}.txt".format(
            CODE_PORT_PLAN_FILE_PREFIX, i + 1, branches[0], branches[1]
        )
        review_file = "{}_V{}_review_from_{}_to_{}.txt".format(
            CODE_PORT_PLAN_FILE_PREFIX, i + 1, branches[0], branches[1]
        )
        fixed_file = "{}_V{}_fixed_from_{}_to_{}.txt".format(
            CODE_PORT_PLAN_FILE_PREFIX, i + 1, branches[0], branches[1]
        )
        evolution_file = "{}_V{}_total_review_evolution_from_{}_to_{}.txt".format(
            CODE_PORT_PLAN_FILE_PREFIX, i + 1, branches[0], branches[1]
        )

        plan_prompt = gen_CodePortPlanPrompt(
            context=context,
            branches=branches,
            branch_code_trace_files=branch_code_trace_files,
            disallowed_modules=config.disallowed_modules,
            previous_attempt_file=previous_attempt_file,
            output_file=plan_file,
        )
        review_prompt = gen_ReviewCodePortPlanPrompt(
            context=context,
            branches=branches,
            branch_code_trace_files=branch_code_trace_files,
            code_port_plan_file=plan_file,
            output_review_file=review_file,
            output_fixed_file=fixed_file,
            output_total_review_summary_file=evolution_file,
        )
        previous_attempt_file = fixed_file
        code_port_plan_and_review_prompts.append(plan_prompt.prompt())
        code_port_plan_and_review_prompts.append(review_prompt.prompt())

    final_plan_file = previous_attempt_file

    # Step 5: Test plan
    test_plan_prompt = gen_TestPlanPrompt(
        context=context,
        branches=branches,
        branch_code_trace_files=branch_code_trace_files,
        code_port_plan_file=final_plan_file,
        test_port_manifest_file=TEST_PORT_MANIFEST_FILE,
    )

    # Step 6: CodeGen iterations
    code_gen_and_review_prompts = []
    previous_code_gen_file = ""
    for i in range(NUM_CODE_REVIEWS):
        pr_info_file = "{}_V{}_PR_INFO_from_{}_to_{}.txt".format(
            CODE_GEN_FILE_PREFIX, i + 1, branches[0], branches[1]
        )
        pr_file = "{}_V{}_PR_from_{}_to_{}.patch".format(
            CODE_GEN_FILE_PREFIX, i + 1, branches[0], branches[1]
        )
        pr_review_file = "{}_V{}_PR_REVIEW_from_{}_to_{}.txt".format(
            CODE_GEN_FILE_PREFIX, i + 1, branches[0], branches[1]
        )
        pr_fixed_file = "{}_V{}_PR_FIXED_from_{}_to_{}.txt".format(
            CODE_GEN_FILE_PREFIX, i + 1, branches[0], branches[1]
        )
        pr_evolution_file = "{}_V{}_PR_TOTAL_REVIEW_EVOLUTION_from_{}_to_{}.txt".format(
            CODE_GEN_FILE_PREFIX, i + 1, branches[0], branches[1]
        )

        gen_prompt = gen_CodeGenPrompt(
            context=context,
            branches=branches,
            branch_code_trace_files=branch_code_trace_files,
            code_port_plan_file=final_plan_file,
            test_plan_file=test_plan_prompt.output_file,
            test_port_manifest_file=TEST_PORT_MANIFEST_FILE,
            previous_attempt_file=previous_code_gen_file,
            output_info_file=pr_info_file,
            output_pr_file=pr_file,
        )
        review_prompt = gen_ReviewCodeGenPrompt(
            context=context,
            branches=branches,
            branch_code_trace_files=branch_code_trace_files,
            code_port_plan_file=final_plan_file,
            test_plan_file=test_plan_prompt.output_file,
            test_port_manifest_file=TEST_PORT_MANIFEST_FILE,
            code_pr_info_file=pr_info_file,
            code_pr_file=pr_file,
            output_review_file=pr_review_file,
            output_fixed_file=pr_fixed_file,
            output_total_review_summary_file=pr_evolution_file,
        )
        previous_code_gen_file = pr_fixed_file
        code_gen_and_review_prompts.append(clear_repo_cmd)
        code_gen_and_review_prompts.append(gen_prompt.prompt())
        code_gen_and_review_prompts.append(review_prompt.prompt())

    # Step 7: RunAndFix
    run_and_fix_prompt = gen_RunAndFixPrompt(
        context=context,
        build_command=config.build_command,
        test_command=config.test_command,
        build_dir=config.build_dir,
        max_build_test_retries=config.max_build_test_retries,
    )

    # Assemble
    prompts: list = []
    prompts.append(clear_repo_cmd)
    prompts.append(code_trace_prompt.prompt())
    if config.port_tests:
        prompts.append(test_port_prompt.prompt())
    prompts.extend(code_port_plan_and_review_prompts)
    prompts.append(test_plan_prompt.prompt())
    prompts.extend(code_gen_and_review_prompts)
    prompts.append(run_and_fix_prompt.prompt())

    return prompts


if __name__ == "__main__":
    log_file = open(LOG_FILE, "w")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)

    from auto_bug_fix.bug_fix_config import claude_config, bug_fix_config

    start_time = time.time()
    asyncio.run(claude_run(claude_config, gen_prompts(claude_config, bug_fix_config)))
    duration_time = time.time() - start_time
    print("FINISHED ALL: total_duration = {}".format(duration_time))
```

- [ ] **Step 4: Run to verify all pass**

```bash
pytest tests/auto_bug_fix/test_run_bug_fix.py -v
```
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add auto_bug_fix/run_bug_fix.py tests/auto_bug_fix/test_run_bug_fix.py
git commit -m "feat: add run_bug_fix.py orchestrator"
```

---

## Task 11: BDD Behavioral Tests

**Files:**
- Create: `tests/auto_bug_fix/test_bug_fix_behavior.py`

- [ ] **Step 1: Create behavioral test file**

```python
"""BDD behavioral tests for the bug fix porting pipeline.

Each test is named as a scenario: Given / When / Then.
Tests verify pipeline-level behavior without running Claude.
"""
from common.claude_utils import ClaudeConfig
from auto_bug_fix.bug_fix_config import BugFixConfig
from auto_bug_fix.run_bug_fix import gen_prompts
from auto_bug_fix.bug_fix_prompts import RUN_AND_FIX_FAILURE_FILE, TEST_PORT_MANIFEST_FILE


def _claude_cfg():
    return ClaudeConfig(
        model="claude-opus-4-6",
        allowed_tools=["Read", "Write", "Bash"],
        perm_mode="acceptEdits",
        cwd="/tmp/test_cwd",
    )


def _config(**overrides):
    defaults = dict(
        repo_path="/path/to/gcc",
        build_dir="/path/to/gcc/build",
        source_branch="gcc-14-branch",
        target_branch="gcc-13-branch",
        source_fix_commit="abc1234",
        bug_description="CVE-2024-1234: test bug",
        issue_id="CVE-2024-1234",
        disallowed_modules=[],
        port_tests=True,
        build_command="make -j$(nproc)",
        test_command="make check -j$(nproc)",
        max_build_test_retries=3,
    )
    defaults.update(overrides)
    return BugFixConfig(**defaults)


def test_scenario_repo_is_reset_before_any_prompts():
    """
    Given a BugFixConfig,
    When the pipeline generates its prompt list,
    Then the first entry must be a clear_repo shell command.
    """
    prompts = gen_prompts(_claude_cfg(), _config())
    assert isinstance(prompts[0], dict), "First step must be a shell command"
    assert "clear_repo" in prompts[0]["cmd"]


def test_scenario_test_port_precedes_code_gen_when_enabled():
    """
    Given port_tests=True,
    When the pipeline generates its prompt list,
    Then TestPortPrompt (git show) must appear before any CodeGenPrompt.
    """
    prompts = gen_prompts(_claude_cfg(), _config(port_tests=True))
    string_prompts = [p for p in prompts if isinstance(p, str)]
    test_port_index = next(i for i, p in enumerate(string_prompts) if "git show" in p)
    code_gen_index = next(i for i, p in enumerate(string_prompts) if "generate a code patch" in p.lower())
    assert test_port_index < code_gen_index


def test_scenario_test_extraction_skipped_when_port_tests_false():
    """
    Given port_tests=False,
    When the pipeline generates its prompt list,
    Then no TestPortPrompt (git show) should appear.
    """
    prompts = gen_prompts(_claude_cfg(), _config(port_tests=False))
    string_prompts = [p for p in prompts if isinstance(p, str)]
    assert not any("git show" in p for p in string_prompts)


def test_scenario_run_and_fix_is_final_step():
    """
    Given a fully configured BugFixConfig,
    When the pipeline generates its prompt list,
    Then RunAndFixPrompt must be the last string prompt.
    """
    prompts = gen_prompts(_claude_cfg(), _config())
    string_prompts = [p for p in prompts if isinstance(p, str)]
    assert RUN_AND_FIX_FAILURE_FILE in string_prompts[-1]


def test_scenario_ported_tests_referenced_in_code_gen():
    """
    Given port_tests=True,
    When the pipeline generates its prompt list,
    Then CodeGenPrompt must reference test_port_manifest.txt.
    """
    prompts = gen_prompts(_claude_cfg(), _config(port_tests=True))
    string_prompts = [p for p in prompts if isinstance(p, str)]
    code_gen_prompts = [p for p in string_prompts if "generate a code patch" in p.lower()]
    assert any(TEST_PORT_MANIFEST_FILE in p for p in code_gen_prompts)


def test_scenario_config_values_flow_into_run_and_fix():
    """
    Given config with specific build and test commands,
    When the pipeline generates its prompt list,
    Then RunAndFixPrompt must embed those exact commands.
    """
    config = _config(
        build_command="cmake --build build -j$(nproc)",
        test_command="ctest --output-on-failure",
    )
    prompts = gen_prompts(_claude_cfg(), config)
    string_prompts = [p for p in prompts if isinstance(p, str)]
    run_and_fix = string_prompts[-1]
    assert "cmake --build build -j$(nproc)" in run_and_fix
    assert "ctest --output-on-failure" in run_and_fix
```

- [ ] **Step 2: Run to verify all pass**

```bash
pytest tests/auto_bug_fix/test_bug_fix_behavior.py -v
```
Expected: 6 passed.

- [ ] **Step 3: Run the full test suite**

```bash
pytest -v
```
Expected: all tests pass (no failures).

- [ ] **Step 4: Commit**

```bash
git add tests/auto_bug_fix/test_bug_fix_behavior.py
git commit -m "test: add BDD behavioral tests for bug fix porting pipeline"
```

---

## Task 12: Code Review

**Files:** None created — this is a checklist review pass.

- [ ] **Step 1: Verify no GPU language leaks into bug_fix_prompts.py**

```bash
grep -i "cuda\|prefill\|decode\|gpu_type\|batch_size\|generate_cmake_presets\|cmake --preset\|transformer block\|median block" auto_bug_fix/bug_fix_prompts.py
```
Expected: no output. If any matches appear, remove them from the offending prompt template.

- [ ] **Step 2: Verify context string contains all BugFixConfig fields**

```bash
grep -E "repo_path|build_dir|source_branch|target_branch|source_fix_commit|bug_description|issue_id|build_command|test_command|max_build_test_retries|disallowed_modules" auto_bug_fix/bug_fix_prompts.py | head -20
```
Expected: all 11 field names appear in `create_context_str()`.

- [ ] **Step 3: Verify factory function assertions cover all multi-branch prompts**

```bash
grep -n "assert len(branches)" auto_bug_fix/bug_fix_prompts.py
```
Expected: 6 lines — one per factory function that takes `branches`: `gen_CodePortPlanPrompt`, `gen_ReviewCodePortPlanPrompt`, `gen_TestPlanPrompt`, `gen_CodeGenPrompt`, `gen_ReviewCodeGenPrompt`, `gen_InvestigateIssuePrompt`, `gen_ReviewInvestigatedIssuePrompt`.

- [ ] **Step 4: Verify format keys match dataclass fields for each prompt class**

For each prompt class in `bug_fix_prompts.py`, confirm that every `{key}` in `prompt_template` has a matching field in the dataclass and a matching argument in `prompt_template.format(...)`.

```bash
python3 -c "
from auto_bug_fix.bug_fix_prompts import *
# Exercise every prompt() to catch KeyError / missing format keys
from common.claude_utils import ClaudeConfig
from auto_bug_fix.bug_fix_config import BugFixConfig
cfg = BugFixConfig('/r', '/b', 's', 't', 'c', 'd', 'i', [], True, 'mc', 'tc', 3)
cc = ClaudeConfig('m', [], 'a', '/cwd')
ctx = create_context_str(cc, cfg)
br = ['s', 't']
tr = ['t.txt']
gen_CodeTracePrompt(ctx, 's').prompt()
gen_TestPortPrompt(ctx, 'c', 't').prompt()
gen_CodePortPlanPrompt(ctx, br, tr, [], '', 'o.txt').prompt()
gen_ReviewCodePortPlanPrompt(ctx, br, tr, 'p', 'r', 'f', 'e').prompt()
gen_TestPlanPrompt(ctx, br, tr, 'p', 'manifest.txt').prompt()
gen_CodeGenPrompt(ctx, br, tr, 'p', 'tp', 'manifest.txt', '', 'i', 'pr').prompt()
gen_ReviewCodeGenPrompt(ctx, br, tr, 'p', 'tp', 'manifest.txt', 'i', 'pr', 'r', 'f', 'e').prompt()
gen_RunAndFixPrompt(ctx, 'mc', 'tc', '/b', 3).prompt()
gen_InvestigateIssuePrompt(ctx, br, tr, 'p', 'tp', 'pe', 'i', 'pr', 'ce', 'issue.txt', '', '', 'fix.txt', 'fixed.patch').prompt()
gen_ReviewInvestigatedIssuePrompt(ctx, br, tr, 'p', 'tp', 'pe', 'i', 'pr', 'ce', 'issue.txt', 'fix.txt', 'fr', 'ff', 'fe', 'prf').prompt()
gen_WorkItemsPrompt(ctx, '/dir', 'wi.txt').prompt()
print('All prompt() calls succeeded — no format key errors.')
"
```
Expected: `All prompt() calls succeeded — no format key errors.`

- [ ] **Step 5: Fix any issues found in steps 1–4, re-run full test suite**

```bash
pytest -v
```
Expected: all tests pass.

- [ ] **Step 6: Commit any fixes**

```bash
git add -p
git commit -m "fix: code review corrections in auto_bug_fix"
```
(Only commit if there were changes from the review.)

---

## Task 13: Documentation

**Files:**
- Modify: `auto_bug_fix/bug_fix_config.py` (module docstring)
- Modify: `auto_bug_fix/bug_fix_prompts.py` (module docstring)
- Modify: `auto_bug_fix/run_bug_fix.py` (module docstring)
- Modify: `README.md` (add `auto_bug_fix` section)

- [ ] **Step 1: Add module docstring to `bug_fix_config.py`**

Insert after the first line (`from dataclasses import dataclass`):

```python
"""Configuration dataclass and default instance for the auto_bug_fix pipeline.

Edit the ``bug_fix_config`` instance at the bottom of this file before running
``python -m auto_bug_fix.run_bug_fix``.
"""
```

- [ ] **Step 2: Add module docstring to `bug_fix_prompts.py`**

Insert at the top of the file:

```python
"""Prompt classes for the auto_bug_fix pipeline.

Each class wraps a single Claude query. All classes follow the same pattern:
a dataclass with a ``ClassVar[str]`` prompt template and a ``prompt()`` method
that formats the template with the instance's fields.

New classes in this module (not in auto_code_gen):
- ``TestPortPrompt``  — extracts and ports tests from the source fix commit.
- ``RunAndFixPrompt`` — autonomous build-test-fix loop (Claude uses Bash tool).
"""
```

- [ ] **Step 3: Add module docstring to `run_bug_fix.py`**

Insert at the top of the file:

```python
"""Main orchestrator for the auto_bug_fix pipeline.

Usage:
    Edit ``auto_bug_fix/bug_fix_config.py`` to configure your project, then run:

        python -m auto_bug_fix.run_bug_fix

The pipeline steps are:
    1. Reset target branch (git reset + clean)
    2. Trace what the fix commit touches on the source branch
    3. Port tests from the fix commit (if config.port_tests is True)
    4. Plan how to apply the fix on the target branch (N review iterations)
    5. Validate test coverage; plan supplemental tests
    6. Generate the fix patch and apply it (M review iterations)
    7. Autonomous build-test-fix loop until clean or retry limit hit
"""
```

- [ ] **Step 4: Add `auto_bug_fix` section to `README.md`**

Open `README.md` and append:

```markdown
## auto_bug_fix — Branch Bug Fix Porting

Port a bug fix from one branch of a C/systems project to another using Claude as the AI engine.

**Supported projects:** Any project with a `git` repository and shell-invokable build and test commands (gcc, openssl, glibc, etc.).

**Setup:**

1. Edit `auto_bug_fix/bug_fix_config.py`:
   - Set `repo_path`, `source_branch`, `target_branch`, `source_fix_commit`
   - Set `build_command`, `test_command`, `build_dir`
   - Set `bug_description`, `issue_id`, `disallowed_modules`
   - Set `port_tests=True` to auto-extract and port test files from the fix commit

2. Set `claude_config.cwd` to the directory where Claude should write output files.

3. Run:
   ```bash
   python -m auto_bug_fix.run_bug_fix
   ```

**On failure:** If `RunAndFixPrompt` exhausts its retry limit, inspect `run_and_fix_failure.txt` in `claude_config.cwd`, then run `auto_code_gen/run_investigate_issue.py` manually for a deeper diagnostic pass.
```

- [ ] **Step 5: Run tests to confirm docs changes don't break anything**

```bash
pytest -v
```
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add auto_bug_fix/bug_fix_config.py auto_bug_fix/bug_fix_prompts.py auto_bug_fix/run_bug_fix.py README.md
git commit -m "docs: add module docstrings and README section for auto_bug_fix"
```

---

## Self-Review Checklist

After all tasks complete, verify against the spec:

| Spec requirement | Covered by |
|---|---|
| `auto_bug_fix/` module with 4 files | Task 1, 3, 4–9, 10 |
| `BugFixConfig` with all 11 fields | Task 3 |
| `clear_repo()` alias in `common/utils.py` | Task 2 |
| `create_context_str()` — no GPU language | Task 4 |
| `CodeTracePrompt` — adapted, no CUDA | Task 4 |
| `TestPortPrompt` — new, git show + manifest | Task 5 |
| `CodePortPlanPrompt` + `ReviewCodePortPlanPrompt` | Task 6 |
| `TestPlanPrompt` — reduced role, references manifest | Task 7 |
| `CodeGenPrompt` — reads build_command, no cmake | Task 7 |
| `ReviewCodeGenPrompt` | Task 7 |
| `RunAndFixPrompt` — new, Claude-owns-loop | Task 8 |
| `InvestigateIssuePrompt` + `ReviewInvestigatedIssuePrompt` | Task 9 |
| `WorkItemsPrompt` | Task 9 |
| `run_bug_fix.py` — correct ordering, port_tests gate | Task 10 |
| `RunAndFixPrompt` is last string prompt | Task 10, 11 |
| BDD behavioral tests | Task 11 |
| Code review pass | Task 12 |
| Module docstrings + README | Task 13 |
