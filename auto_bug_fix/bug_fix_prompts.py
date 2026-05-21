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
