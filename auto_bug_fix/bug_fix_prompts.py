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
