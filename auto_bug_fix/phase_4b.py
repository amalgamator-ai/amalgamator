"""Autonomous build-test-fix loop prompt for Phase 4b.

Extracted from auto_code_gen/use_cases/bug_fix.py so auto_bug_fix is
standalone-portable. This is the last-resort escalation: a single
long-running agent that iterates build → test → targeted fix until clean
or retries are exhausted, reporting back through a success/failure file.
"""

from dataclasses import dataclass
from typing import ClassVar


@dataclass
class RunAndFixPrompt:
    context: str
    build_command: str
    test_command: str
    build_dir: str
    max_build_test_retries: int
    output_failure_file: str
    output_success_file: str
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
<output_success_file>
{output_success_file}
</output_success_file>
</definitions>

<definition_explanations>
- <build_command> is the shell command to compile the target branch incrementally. Run from <build_dir>.
- <test_command> is the shell command to run the relevant test suite. Run from <build_dir>.
- <build_dir> is the working directory from which both commands are invoked.
- <max_build_test_retries> is the maximum number of build-test-fix iterations before stopping.
- <output_failure_file> is where to write the failure log if retries are exhausted.
- <output_success_file> is where to write the success summary when all tests pass.
</definition_explanations>

<instructions>
The goal of this task is to autonomously drive a build-test-fix loop until both the build and tests are clean. Think hard and follow these guidelines:

IMPORTANT: If <target_branch> is provided in the context, ALL code analysis, reading, and modifications MUST be restricted to that branch only. Do NOT examine or reference code from other branches. Before starting, verify you are on <target_branch> by running `git -C <build_dir> branch --show-current`.

- Run <build_command> from <build_dir>.
    - Use all available CPUs (the command already specifies this).
    - If the build succeeds, proceed to running the tests.
    - If the build fails:
        - Inspect the compiler/linker output carefully to identify the root cause.
        - Apply a targeted fix to the source code (ON THE TARGET BRANCH ONLY).
        - Do NOT do a full clean rebuild unless the error is explicitly caused by a stale artifact. Never do a speculative clean rebuild.
        - Retry the build.
- On a successful build, run <test_command> from <build_dir>.
    - If all tests pass, the loop is complete.
        - Write a success summary to <output_dir>/{output_success_file} with:
            Line 1: "SUCCESS"
            Line 2: Number of retries used (e.g. "Retries: 1/5")
            Remaining lines: full test output and a summary of any fixes applied during the loop.
        - Stop.
    - If tests fail:
        - Inspect the test output carefully.
        - Identify the root cause -- may be in the fix code or in the ported tests.
        - Apply a targeted fix.
        - Recompile using <build_command> (incremental).
        - Rerun <test_command>.
- Count each complete build+test attempt as one retry. Repeat until clean or <max_build_test_retries> is exhausted.
- If retries are exhausted:
    - Write the final build/test output and a summary of all attempted fixes to <output_dir>/{output_failure_file}.
    - Stop. Do not attempt further fixes.
</instructions>

<output>
- If successful: write the success summary to <output_dir>/{output_success_file}. First line must be "SUCCESS".
- If retries exhausted: write the failure summary to <output_dir>/{output_failure_file}. Do not write the success file.
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
            output_success_file=self.output_success_file,
        )


RUN_AND_FIX_FAILURE_FILE = "run_and_fix_failure.txt"
RUN_AND_FIX_SUCCESS_FILE = "run_and_fix_success.txt"


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
        output_success_file=RUN_AND_FIX_SUCCESS_FILE,
    )
