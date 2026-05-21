"""Prompt classes for the auto_bug_fix pipeline.

Each class wraps a single Claude query. All classes follow the same pattern:
a dataclass with a ``ClassVar[str]`` prompt template and a ``prompt()`` method
that formats the template with the instance's fields.

New classes in this module (not in auto_code_gen):
- ``TestPortPrompt``  — extracts and ports tests from the source fix commit.
- ``RunAndFixPrompt`` — autonomous build-test-fix loop (Claude uses Bash tool).
"""

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
    return ReviewCodePortPlanPrompt(
        context=context,
        branches=branches,
        branch_code_trace_files=branch_code_trace_files,
        code_port_plan_file=code_port_plan_file,
        output_review_file=output_review_file,
        output_fixed_file=output_fixed_file,
        output_total_review_summary_file=output_total_review_summary_file,
    )


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
