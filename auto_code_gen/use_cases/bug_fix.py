"""Bug-fix porting use case for the unified code gen pipeline.

Implements the UseCase ABC to drive the pipeline for porting bug fixes
between branches of the same repository.  The pipeline phases are the same
as the LLM framework use case (code trace, code port plan, test plan,
code gen) but the prompt templates focus on branch-to-branch porting
rather than cross-framework GPU kernel porting.

One prompt class is defined here (not shared with other use cases):
- ``RunAndFixPrompt`` -- autonomous build-test-fix loop.
"""

from dataclasses import dataclass
from typing import ClassVar, Optional

from auto_code_gen.use_cases.base import UseCase
from common.claude_utils import PipelineStep
from common.utils import clear_repo

from auto_code_gen.code_gen_prompts import (
    _prev_iteration_section,
    gen_CodeTracePrompt,
    CodePortPlanPrompt,
    ReviewCodePortPlanPrompt,
    TestPlanPrompt,
    ReviewTestPlanPrompt,
    CodeGenPrompt,
    ReviewCodeGenPrompt,
    CODE_PORT_PLAN_FILE_PREFIX,
    CODE_GEN_FILE_PREFIX,
    TEST_PLAN_PREFIX,
)


# ---------------------------------------------------------------------------
# Bug-fix-specific prompt classes (NOT shared with other use cases)
# ---------------------------------------------------------------------------

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
        - Identify the root cause -- may be in the fix code or in the ported tests.
        - Apply a targeted fix.
        - Recompile using <build_command> (incremental).
        - Rerun <test_command>.
- Count each complete build+test attempt as one retry. Repeat until clean or <max_build_test_retries> is exhausted.
- If retries are exhausted:
    - Write the final build/test output and a summary of all attempted fixes to <output_dir>/<output_failure_file>.
    - Stop. Do not attempt further fixes.
</instructions>

<output>
- If successful: report that both build and tests are clean, with a summary of fixes applied.
- If retries exhausted: write the failure summary to <output_dir>/{output_failure_file} and stop.
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


# ---------------------------------------------------------------------------
# Template constants for the 7 shared prompt types
# ---------------------------------------------------------------------------

BUGFIX_CODE_TRACE_TEMPLATE = """

{context}

<instructions>
The goal of this task is to produce a complete code trace of the bug fix and a comparison against <target_branch>. The fix in <source_fix_commit> on <source_branch> will be ported to <target_branch>. Think hard and follow these guidelines:

Part 1 — Full code trace of the fix on <source_branch>:
- Run `git show <source_fix_commit>` on <source_branch> and analyze the complete diff.
- Analyze and understand in detail what the fix does: which files it modifies, which functions it changes, and why.
- Trace the <code_trace> of the changed code on <source_branch>: follow call chains from the highest-level entry point down to the lowest-level functions affected.
    - Go deep into C/C++ source if the fix touches compiled code.
    - Include all relevant wrappers, helpers, and data structures.
    - Do not miss any code piece that is part of the fix's scope.
- Document the full <code_trace> step-by-step from top to bottom:
    - For each function in the trace: goal, inputs, outputs, and why it is relevant to the fix.
    - For changed functions: document the before and after behavior.
    - Document classes/objects and their relationships where relevant.
    - Be professional, clear, and concise.

Part 2 — Target branch differences:
- For each code piece in the <code_trace> from Part 1, inspect the corresponding code on <target_branch>.
- Document how <target_branch> differs from <source_branch> for each traced code piece:
    - Does the same file/function/class exist on <target_branch>? If not, where does the equivalent logic live?
    - Are there renamed symbols, changed function signatures, refactored structures, or different APIs?
    - Are there missing dependencies or initialization differences?
    - What code on <target_branch> can receive the fix AS IS, and what requires adaptation?
- Summarize the key divergence points that the porting plan will need to address.
</instructions>

<output>
- Dump results to <output_dir>/{output_file} with two clearly separated sections:
    1. "Source Branch Code Trace" — the full <code_trace> of the fix on <source_branch> (Part 1)
    2. "Target Branch Differences" — the per-code-piece comparison against <target_branch> (Part 2)
</output>

"""

BUGFIX_CODE_PORT_PLAN_TEMPLATE = """

{context}

{prev_iteration_section}

<definitions>
<code_trace_files>
{code_trace_files}
</code_trace_files>
<disallowed_modules>
{disallowed_modules}
</disallowed_modules>
</definitions>

<definition_explanations>
- <code_trace_files> is the code trace file containing: (1) the full <code_trace> of the fix on <source_branch> — code paths, functions, and call chains that the fix commit touches, and (2) a comparison of how the corresponding code on <target_branch> differs, including renamed symbols, changed APIs, and divergence points relevant to porting.
- <disallowed_modules> lists files and directories that must not be modified on <target_branch>.
</definition_explanations>

<instructions>
The goal of this task is to produce a high-level multi-step coding plan for applying the fix from <source_branch> to <target_branch> (iteration {iteration}). Think hard and follow these guidelines:
- Analyze the <code_trace> of the fix on <source_branch> from <code_trace_files>.
- Analyze the corresponding code on <target_branch>, noting where the same logic lives and where it has diverged.
- Compare the two branches step-by-step through the affected call chains. Identify:
    - Code that can be ported AS IS with no changes.
    - Code that requires adaptation due to branch divergence (different APIs, renamed symbols, refactored structure).
    - Code that does not exist on <target_branch> and must be introduced.
- Produce a step-by-step porting plan that:
    - Maximises the amount of code copied AS IS from <source_branch> to <target_branch>.
    - Minimises integration points requiring changes to existing <target_branch> code.
    - Does NOT modify any path under <disallowed_modules>.
    - For each step: state what is ported, what is changed, why the change is necessary, and how to integrate it.
- Verify the plan end-to-end:
    - Check all function APIs are used correctly after porting.
    - Identify hidden dependencies or initialization order constraints.
    - Provide a risk analysis and list of sensitive breaking points.
</instructions>

<output>
- Dump the high-level multi-step coding plan to <output_dir>/{output_file}
- Dump a summary to <output_dir>/{output_summary_file} that includes:
    - An explanation of the work done to generate the coding plan in this iteration
    - A summary of the iteration evolution across all iterations up to and including iteration {iteration}, describing the progression of fixes and improvements
</output>

"""

BUGFIX_REVIEW_CODE_PORT_PLAN_TEMPLATE = """

{context}

<definitions>
<code_trace_files>
{code_trace_files}
</code_trace_files>
<input_file>
{input_file}
</input_file>
<input_summary_file>
{input_summary_file}
</input_summary_file>
</definitions>

<definition_explanations>
- <code_trace_files> is the code trace file containing: (1) the full <code_trace> of the fix on <source_branch>, and (2) a comparison of how the corresponding code on <target_branch> differs, including divergence points relevant to porting.
- <input_file> is the high-level multi-step coding plan for porting the fix from <source_branch> to <target_branch>.
- <input_summary_file> is a file that summarizes the work done to generate <input_file>.
</definition_explanations>

<instructions>
The goal of this task is to perform a critical, in-depth review of the coding plan in <input_file> (iteration {iteration}). Do the following and think hard:
- Understand the plan in detail and restate it step-by-step.
- Review the plan for correctness of the porting logic:
    - Are all changed APIs used correctly on <target_branch> after porting?
    - Are all initialization order constraints respected?
    - Are all renamed or refactored symbols on <target_branch> handled?
- Review for completeness: missing steps, incomplete porting of affected code paths.
- Review for risks: incorrect assumptions, hidden dependencies, ambiguous steps, missing edge cases.
- For each issue found: document the affected plan step, what is wrong, why it matters, and how to fix it.
- Produce a corrected plan with all issues resolved.
</instructions>

<output>
- Dump the corrected plan to <output_dir>/{output_file}
- Dump a summary to <output_dir>/{output_summary_file} that includes:
    - Documentation of the issues found and fixed in this review
    - A summary of the iteration evolution across all iterations up to and including iteration {iteration}
- IMPORTANT: If no changes were needed to the code port plan, write CONVERGED as the very first line of <output_dir>/{output_summary_file}. Otherwise, do NOT write CONVERGED.
</output>

"""

BUGFIX_COMBINED_CODE_AND_TEST_PORT_PLAN_TEMPLATE = """

{context}

{prev_iteration_section}

<definitions>
<code_trace_files>
{code_trace_files}
</code_trace_files>
<disallowed_modules>
{disallowed_modules}
</disallowed_modules>
</definitions>

<definition_explanations>
- <code_trace_files> is the code trace file containing: (1) the full <code_trace> of the fix on <source_branch>, and (2) a comparison of how the corresponding code on <target_branch> differs, including divergence points relevant to porting.
- <disallowed_modules> lists files and directories that must not be modified on <target_branch>.
</definition_explanations>

<instructions>
The goal of this task is to produce a combined code porting plan AND test plan for applying the fix from <source_branch> to <target_branch> (iteration {iteration}). Think hard and follow these guidelines:

Part 1 — Code porting plan:
- Analyze the <code_trace> of the fix on <source_branch> from <code_trace_files>.
- Analyze the corresponding code on <target_branch>, noting where the same logic lives and where it has diverged.
- Compare the two branches step-by-step through the affected call chains. Identify:
    - Code that can be ported AS IS with no changes.
    - Code that requires adaptation due to branch divergence (different APIs, renamed symbols, refactored structure).
    - Code that does not exist on <target_branch> and must be introduced.
- Produce a step-by-step porting plan that:
    - Maximises the amount of code copied AS IS from <source_branch> to <target_branch>.
    - Minimises integration points requiring changes to existing <target_branch> code.
    - Does NOT modify any path under <disallowed_modules>.
    - For each step: state what is ported, what is changed, why the change is necessary, and how to integrate it.
- Verify the plan end-to-end:
    - Check all function APIs are used correctly after porting.
    - Identify hidden dependencies or initialization order constraints.
    - Provide a risk analysis and list of sensitive breaking points.

Part 2 — Test plan:
- Run `git show <source_fix_commit>` and identify all test files added or modified in the fix commit. Look for files under directories named test/, tests/, testsuite/, t/, or similar.
- For each identified test, inspect <target_branch>'s test infrastructure in detail:
    - Understand the test harness (dejagnu, ctest, pytest, OpenSSL test runner, or other).
    - Understand the suite directory layout, naming conventions, and registration patterns.
- Plan how to port each test to <target_branch>'s test infrastructure:
    - Preserve the test's intent and coverage exactly.
    - Adapt harness-specific syntax, suite registration, file layout, and include paths.
    - Only change what is required for harness compatibility — do not change test logic.
- Document the planned ported tests: source file, target location, and adaptation details.
- Identify any code paths, edge cases, or integration points in the porting plan from Part 1 that are NOT covered by the ported tests, and plan supplemental tests for each gap.
- Ensure the test plan covers all ported/modified functions and code paths, edge cases in the fix logic, and interactions with surrounding code on <target_branch>.
- Do NOT plan performance benchmarks or speed-gain tests.
</instructions>

<output>
- Dump the combined code porting plan and test plan to <output_dir>/{output_file} with two clearly separated sections:
    1. "Code Porting Plan" — the step-by-step porting plan (Part 1)
    2. "Test Plan" — the ported tests and supplemental tests (Part 2)
- Dump a summary to <output_dir>/{output_summary_file} that includes:
    - An explanation of the work done in this iteration
    - A summary of the iteration evolution across all iterations up to and including iteration {iteration}, describing the progression of fixes and improvements
</output>

"""

BUGFIX_REVIEW_COMBINED_CODE_AND_TEST_PORT_PLAN_TEMPLATE = """

{context}

<definitions>
<code_trace_files>
{code_trace_files}
</code_trace_files>
<input_file>
{input_file}
</input_file>
<input_summary_file>
{input_summary_file}
</input_summary_file>
</definitions>

<definition_explanations>
- <code_trace_files> is the code trace file containing: (1) the full <code_trace> of the fix on <source_branch>, and (2) a comparison of how the corresponding code on <target_branch> differs, including divergence points relevant to porting.
- <input_file> is the combined code porting plan and test plan for porting the fix from <source_branch> to <target_branch>.
- <input_summary_file> is a file that summarizes the work done to generate <input_file>.
</definition_explanations>

<instructions>
The goal of this task is to perform a critical, in-depth review of the combined code porting plan and test plan in <input_file> (iteration {iteration}). Do the following and think hard:

Review the code porting plan (Part 1):
- Understand the plan in detail and restate it step-by-step.
- Review the plan for correctness of the porting logic:
    - Are all changed APIs used correctly on <target_branch> after porting?
    - Are all initialization order constraints respected?
    - Are all renamed or refactored symbols on <target_branch> handled?
- Review for completeness: missing steps, incomplete porting of affected code paths.
- Review for risks: incorrect assumptions, hidden dependencies, ambiguous steps, missing edge cases.

Review the test plan (Part 2):
- Verify the test plan covers all ported/modified functions and code paths.
- Verify proper test porting: are the ported tests adapted correctly to <target_branch>'s test infrastructure?
- Verify edge case coverage in the fix logic.
- Verify tests that would actually detect regressions.
- Review for missing test coverage, incorrect test assumptions, or tests that would not verify correctness.

- For each issue found in either part, document: the affected section, what is wrong, why it matters, and how to fix it.
- Produce a corrected combined plan with all issues resolved.
</instructions>

<output>
- Dump the corrected combined plan to <output_dir>/{output_file}
- Dump a summary to <output_dir>/{output_summary_file} that includes:
    - Documentation of the issues found and fixed in this review
    - A summary of the iteration evolution across all iterations up to and including iteration {iteration}
- IMPORTANT: If no changes were needed, write CONVERGED as the very first line of <output_dir>/{output_summary_file}. Otherwise, do NOT write CONVERGED.
</output>

"""

BUGFIX_TEST_PLAN_TEMPLATE = """

{context}

{prev_iteration_section}

<definitions>
<code_trace_files>
{code_trace_files}
</code_trace_files>
<code_port_plan_file>
{code_port_plan_file}
</code_port_plan_file>
</definitions>

<definition_explanations>
- <code_trace_files> is the code trace file containing: (1) the full <code_trace> of the fix on <source_branch>, and (2) a comparison of how the corresponding code on <target_branch> differs, including divergence points relevant to porting.
- <code_port_plan_file> is the high-level multi-step coding plan for porting the fix from <source_branch> to <target_branch>.
</definition_explanations>

<instructions>
The goal of this task is to plan the tests for the fix being ported from <source_branch> to <target_branch> (iteration {iteration}). Think hard and do the following:

Step 1 — Port existing tests from the fix commit:
- Run `git show <source_fix_commit>` and identify all test files added or modified in the fix commit. Look for files under directories named test/, tests/, testsuite/, t/, or similar.
- For each identified test, inspect <target_branch>'s test infrastructure in detail:
    - Understand the test harness (dejagnu, ctest, pytest, OpenSSL test runner, or other).
    - Understand the suite directory layout, naming conventions, and registration patterns.
- Plan how to port each test to <target_branch>'s test infrastructure:
    - Preserve the test's intent and coverage exactly.
    - Adapt harness-specific syntax, suite registration, file layout, and include paths.
    - Only change what is required for harness compatibility — do not change test logic.
- Document the planned ported tests: source file, target location, and adaptation details.

Step 2 — Plan supplemental tests:
- Analyze the coding plan in <code_port_plan_file> to understand all code paths that will change on <target_branch>.
- Identify any code paths, edge cases, or integration points in the plan that are NOT covered by the tests from Step 1.
- For each coverage gap, plan a supplemental test: unit tests for isolated changes, integration tests for multi-component interactions, regression tests for uncovered edge cases.
- If the ported tests already give full coverage, state that explicitly and add no supplemental tests.

General guidelines:
- Ensure the test plan covers all ported/modified functions and code paths, edge cases in the fix logic, and interactions with surrounding code on <target_branch>.
- Do NOT plan performance benchmarks or speed-gain tests.
</instructions>

<output>
- Dump the test plan (ported tests + supplemental tests) to <output_dir>/{output_file}
- Dump a summary to <output_dir>/{output_summary_file} that includes:
    - An explanation of the test plan generated in this iteration
    - A summary of the iteration evolution across all iterations up to and including iteration {iteration}, describing the progression of fixes and improvements
</output>

"""

BUGFIX_REVIEW_TEST_PLAN_TEMPLATE = """

{context}

<definitions>
<code_trace_files>
{code_trace_files}
</code_trace_files>
<code_port_plan_file>
{code_port_plan_file}
</code_port_plan_file>
<input_file>
{input_file}
</input_file>
<input_summary_file>
{input_summary_file}
</input_summary_file>
</definitions>

<definition_explanations>
- <code_trace_files> is the code trace file containing: (1) the full <code_trace> of the fix on <source_branch>, and (2) a comparison of how the corresponding code on <target_branch> differs, including divergence points relevant to porting.
- <code_port_plan_file> is the high-level multi-step coding plan for porting the fix from <source_branch> to <target_branch>.
- <input_file> is the test plan to review.
- <input_summary_file> is a file that summarizes the work done to generate <input_file>.
</definition_explanations>

<instructions>
The goal of this task is to perform a critical, in-depth review of the test plan in <input_file> (iteration {iteration}). Do the following and think hard:
- Understand in detail the test plan in <input_file>, and restate the plan step-by-step.
- Verify the test plan has proper unit tests for each ported/modified component.
- Verify the test plan covers edge cases in the fix logic.
- Verify the test plan has integration tests where the fix interacts with surrounding code.
- Verify test harness compatibility: do the planned tests follow <target_branch>'s testing conventions?
- Review misc issues:
    - missing test coverage
    - incorrect test assumptions
    - missing edge cases
    - tests that would not actually verify correctness
    - tests that would not detect regressions

- For each issue found, document:
    - The affected part of the test plan
    - What is wrong
    - Why it matters
    - How it should be improved
- Produce a corrected test plan with all issues fixed.
</instructions>

<output>
- Dump the corrected test plan to <output_dir>/{output_file}
- Dump a summary to <output_dir>/{output_summary_file} that includes:
    - Documentation of the issues found and fixed in this review
    - A summary of the iteration evolution across all iterations up to and including iteration {iteration}
- IMPORTANT: If no changes were needed, write CONVERGED as the very first line of <output_dir>/{output_summary_file}. Otherwise, do NOT write CONVERGED.
</output>

"""

BUGFIX_CODE_GEN_TEMPLATE = """

{context}

{prev_iteration_section}

<definitions>
<code_trace_files>
{code_trace_files}
</code_trace_files>
<code_port_plan_file>
{code_port_plan_file}
</code_port_plan_file>
<test_plan_file>
{test_plan_file}
</test_plan_file>
</definitions>

<definition_explanations>
- <code_trace_files> is the code trace file containing: (1) the full <code_trace> of the fix on <source_branch>, and (2) a comparison of how the corresponding code on <target_branch> differs, including divergence points relevant to porting.
- <code_port_plan_file> is the high-level multi-step coding plan for porting the fix from <source_branch> to <target_branch>.
- <test_plan_file> is the test plan for the ported fix.
</definition_explanations>

<instructions>
The goal of this task is to generate a code patch for <target_branch> that implements the coding plan in <code_port_plan_file> and the testing plan in <test_plan_file> (iteration {iteration}). Do the following and think hard:
- Analyze and understand in detail the coding plan in <code_port_plan_file>.
- Analyze and understand in detail the testing plan in <test_plan_file>. The test plan includes both tests ported from <source_fix_commit> and supplemental tests.
- Implement the plans in <code_port_plan_file> and <test_plan_file> EXACTLY as described:
    - Follow strictly the code and test plans as described, DO NOT DIVERGE.
    - Implement the fix code as described in <code_port_plan_file>.
    - Port the tests from <source_fix_commit> as described in <test_plan_file>: adapt them to <target_branch>'s test infrastructure (harness syntax, directory layout, registration patterns, include paths) while preserving test intent and coverage exactly.
    - Implement any supplemental tests described in <test_plan_file>.
    - After applying the fix and tests, compile <target_branch> using <build_command> from <build_dir>.
    - Use all available CPUs.
    - Use incremental compilation. Do NOT do a full clean rebuild unless a stale artifact error explicitly requires it.
    - If compilation fails, investigate, fix, and recompile. Do not stop until compilation succeeds.
- Run all tests. Do NOT skip anything. Make sure ALL TESTS ARE RUNNING AND PASSING (with the fully recompiled codebase if needed). If there are failures, then fix, revisit, rewrite, and re-run everything again.
- Ensure the code, both main code and test code, is written professionally, clearly, well-documented, and well-formatted, while taking into account how code is written on <target_branch>.
- Review your work critically and fix issues.
</instructions>

<output>
- Dump the code patch to <output_dir>/{output_patch_file} (it must have both code and its tests inside)
- Dump a summary to <output_dir>/{output_summary_file} that includes:
    - An explanation of the code patch and tests generated in this iteration
    - A summary of the iteration evolution across all iterations up to and including iteration {iteration}, describing the progression of fixes and improvements
</output>

"""

BUGFIX_REVIEW_CODE_GEN_TEMPLATE = """

{context}

<definitions>
<code_trace_files>
{code_trace_files}
</code_trace_files>
<code_port_plan_file>
{code_port_plan_file}
</code_port_plan_file>
<test_plan_file>
{test_plan_file}
</test_plan_file>
<input_patch_file>
{input_patch_file}
</input_patch_file>
<input_summary_file>
{input_summary_file}
</input_summary_file>
</definitions>

<definition_explanations>
- <code_trace_files> is the code trace file containing: (1) the full <code_trace> of the fix on <source_branch>, and (2) a comparison of how the corresponding code on <target_branch> differs, including divergence points relevant to porting.
- <code_port_plan_file> is the high-level multi-step coding plan for porting the fix from <source_branch> to <target_branch>.
- <test_plan_file> is the test plan for the ported fix.
- <input_patch_file> is the code patch to review.
- <input_summary_file> is a file that summarizes the work done to generate <input_patch_file>.
</definition_explanations>

<instructions>
The goal of this task is to perform a critical, in-depth review of the code patch in <input_patch_file> (iteration {iteration}). Do the following and think hard:
- Understand in detail the code in <input_patch_file>, and restate the coding process step-by-step.
- Review the code for correctness:
    - Does the patch fully implement the coding plan?
    - Are all ported APIs used correctly on <target_branch>?
    - Are all function signatures, parameter types, and return types correct?
- Review for completeness: missing steps, coverage gaps in the tests.
- Review for risks:
    - incorrect assumptions
    - missing steps
    - bad ordering or sequencing
    - ambiguity or vagueness
    - missing edge cases
    - architectural risks
    - hidden dependencies

- For each issue found, document:
    - The affected part of the patch
    - What is wrong
    - Why it matters
    - How it should be improved
- Produce a corrected code patch with all issues fixed.
- Add additional tests if needed to cover new issues, or fixed issues.
</instructions>

<output>
- Dump the corrected code patch to <output_dir>/{output_patch_file}
- Dump a summary to <output_dir>/{output_summary_file} that includes:
    - Documentation of the issues found and fixed in this review
    - A summary of the iteration evolution across all iterations up to and including iteration {iteration}, describing the progression of fixes and improvements
- IMPORTANT: If no changes were needed, write CONVERGED as the very first line of <output_dir>/{output_summary_file}. Otherwise, do NOT write CONVERGED.
</output>

"""


# ---------------------------------------------------------------------------
# Bug-fix context string builder
# ---------------------------------------------------------------------------

def _create_bug_fix_context_str(claude_config, config) -> str:
    return """
<context>

<output_dir>
{output_dir}
</output_dir>

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
- <output_dir> is the output directory where all generated artifacts (plans, summaries, patches, test plans) must be saved. ALL output files MUST be written to this directory.
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
        output_dir=claude_config.cwd,
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


# ---------------------------------------------------------------------------
# BugFixUseCase -- orchestrates the bug-fix porting pipeline
# ---------------------------------------------------------------------------

class BugFixUseCase(UseCase):
    """Use case for porting bug fixes between branches of a repository."""

    def create_context_str(self, claude_config, config):
        return _create_bug_fix_context_str(claude_config, config)

    def skip_test_plan_phase(self, config):
        return getattr(config, 'use_combined_code_and_test_port_plan', False)

    def clear_target_cmd(self, config):
        return {
            "fn": lambda: clear_repo(config.repo_path),
            "fn_name": 'clear_repo("{}")'.format(config.repo_path),
        }

    async def run_runtime_iterations(
        self, context, config, claude_config, code_trace_files,
        code_port_plan_file, test_plan_file, resume=False,
    ):
        import time
        from common.claude_utils import claude_run

        phase_start = time.time()
        prompt = gen_RunAndFixPrompt(
            context=context,
            build_command=config.build_command,
            test_command=config.test_command,
            build_dir=config.build_dir,
            max_build_test_retries=config.max_build_test_retries,
        )
        steps = [
            PipelineStep(
                name="run_and_fix",
                prompt=prompt.prompt(),
            ),
        ]
        timings = await claude_run(claude_config, steps)

        phase_results = [{
            "name": "Runtime",
            "iterations": 1,
            "max_iterations": 1,
            "converged": False,
            "duration": time.time() - phase_start,
        }]
        return phase_results, timings

    # -- Phase 1: code trace ---------------------------------------------------

    def gen_code_trace_steps(self, context, config):
        prompt = gen_CodeTracePrompt(
            context=context,
            framework=config.source_branch,
            prompt_template=BUGFIX_CODE_TRACE_TEMPLATE,
        )

        code_trace_files = [prompt.output_file]

        steps = [
            PipelineStep(
                name="clear_target_repo",
                prompt=self.clear_target_cmd(config),
            ),
            PipelineStep(
                name="code_trace_{}".format(config.source_branch),
                prompt=prompt.prompt(),
                output_files=[prompt.output_file],
            ),
        ]

        return steps, code_trace_files

    # -- Phase 2: code port plan iterations ------------------------------------

    def gen_code_port_plan_iter_steps(
        self, context, config, code_trace_files,
        prev_output_file, prev_output_summary_file, iteration,
    ):
        # Bug fix has only 1 trace file, so we construct the prompt
        # directly instead of using gen_CodePortPlanPrompt (which
        # asserts len == 2).
        combined = getattr(config, 'use_combined_code_and_test_port_plan', False)
        if combined:
            plan_template = BUGFIX_COMBINED_CODE_AND_TEST_PORT_PLAN_TEMPLATE
            review_template = BUGFIX_REVIEW_COMBINED_CODE_AND_TEST_PORT_PLAN_TEMPLATE
        else:
            plan_template = BUGFIX_CODE_PORT_PLAN_TEMPLATE
            review_template = BUGFIX_REVIEW_CODE_PORT_PLAN_TEMPLATE

        plan_prompt = CodePortPlanPrompt(
            context=context,
            code_trace_files=code_trace_files,
            disallowed_modules=config.disallowed_modules,
            output_file="{}_V{}.txt".format(CODE_PORT_PLAN_FILE_PREFIX, iteration),
            output_summary_file="{}_summary_V{}.txt".format(
                CODE_PORT_PLAN_FILE_PREFIX, iteration,
            ),
            prev_output_file=prev_output_file,
            prev_output_summary_file=prev_output_summary_file,
            iteration=iteration,
            prompt_template=plan_template,
        )

        review_prompt = ReviewCodePortPlanPrompt(
            context=context,
            code_trace_files=code_trace_files,
            input_file=plan_prompt.output_file,
            input_summary_file=plan_prompt.output_summary_file,
            output_file="{}_review_V{}.txt".format(
                CODE_PORT_PLAN_FILE_PREFIX, iteration,
            ),
            output_summary_file="{}_review_summary_V{}.txt".format(
                CODE_PORT_PLAN_FILE_PREFIX, iteration,
            ),
            iteration=iteration,
            prompt_template=review_template,
        )

        steps = [
            PipelineStep(
                name="code_port_plan_V{}".format(iteration),
                prompt=plan_prompt.prompt(),
                output_files=[plan_prompt.output_file, plan_prompt.output_summary_file],
            ),
            PipelineStep(
                name="code_port_plan_review_V{}".format(iteration),
                prompt=review_prompt.prompt(),
                output_files=[review_prompt.output_file, review_prompt.output_summary_file],
            ),
        ]

        return steps, review_prompt

    # -- Phase 3: test plan iterations -----------------------------------------

    def gen_test_plan_iter_steps(
        self, context, config, code_trace_files,
        code_port_plan_file,
        prev_output_file, prev_output_summary_file, iteration,
    ):
        test_plan_prompt = TestPlanPrompt(
            context=context,
            code_trace_files=code_trace_files,
            code_port_plan_file=code_port_plan_file,
            output_file="{}_V{}.txt".format(TEST_PLAN_PREFIX, iteration),
            output_summary_file="{}_summary_V{}.txt".format(
                TEST_PLAN_PREFIX, iteration,
            ),
            prev_output_file=prev_output_file,
            prev_output_summary_file=prev_output_summary_file,
            iteration=iteration,
            prompt_template=BUGFIX_TEST_PLAN_TEMPLATE,
        )

        review_prompt = ReviewTestPlanPrompt(
            context=context,
            code_trace_files=code_trace_files,
            code_port_plan_file=code_port_plan_file,
            input_file=test_plan_prompt.output_file,
            input_summary_file=test_plan_prompt.output_summary_file,
            output_file="{}_review_V{}.txt".format(TEST_PLAN_PREFIX, iteration),
            output_summary_file="{}_review_summary_V{}.txt".format(
                TEST_PLAN_PREFIX, iteration,
            ),
            iteration=iteration,
            prompt_template=BUGFIX_REVIEW_TEST_PLAN_TEMPLATE,
        )

        steps = [
            PipelineStep(
                name="test_plan_V{}".format(iteration),
                prompt=test_plan_prompt.prompt(),
                output_files=[test_plan_prompt.output_file, test_plan_prompt.output_summary_file],
            ),
            PipelineStep(
                name="test_plan_review_V{}".format(iteration),
                prompt=review_prompt.prompt(),
                output_files=[review_prompt.output_file, review_prompt.output_summary_file],
            ),
        ]

        return steps, review_prompt

    # -- Phase 4: code gen iterations ------------------------------------------

    def gen_code_gen_iter_steps(
        self, context, config, code_trace_files,
        code_port_plan_file, test_plan_file,
        prev_output_patch_file, prev_output_summary_file, iteration,
    ):
        code_gen_prompt = CodeGenPrompt(
            context=context,
            code_trace_files=code_trace_files,
            code_port_plan_file=code_port_plan_file,
            test_plan_file=test_plan_file,
            output_patch_file="{}_V{}.patch".format(
                CODE_GEN_FILE_PREFIX, iteration,
            ),
            output_summary_file="{}_summary_V{}.txt".format(
                CODE_GEN_FILE_PREFIX, iteration,
            ),
            prev_output_patch_file=prev_output_patch_file,
            prev_output_summary_file=prev_output_summary_file,
            iteration=iteration,
            prompt_template=BUGFIX_CODE_GEN_TEMPLATE,
        )

        code_review_prompt = ReviewCodeGenPrompt(
            context=context,
            code_trace_files=code_trace_files,
            code_port_plan_file=code_port_plan_file,
            test_plan_file=test_plan_file,
            input_patch_file=code_gen_prompt.output_patch_file,
            input_summary_file=code_gen_prompt.output_summary_file,
            output_patch_file="{}_review_V{}.patch".format(
                CODE_GEN_FILE_PREFIX, iteration,
            ),
            output_summary_file="{}_review_summary_V{}.txt".format(
                CODE_GEN_FILE_PREFIX, iteration,
            ),
            iteration=iteration,
            prompt_template=BUGFIX_REVIEW_CODE_GEN_TEMPLATE,
        )

        steps = [
            PipelineStep(
                name="clear_target_repo_code_gen_V{}".format(iteration),
                prompt=self.clear_target_cmd(config),
            ),
            PipelineStep(
                name="code_gen_V{}".format(iteration),
                prompt=code_gen_prompt.prompt(),
                output_files=[code_gen_prompt.output_patch_file, code_gen_prompt.output_summary_file],
            ),
            PipelineStep(
                name="code_gen_review_V{}".format(iteration),
                prompt=code_review_prompt.prompt(),
                output_files=[code_review_prompt.output_patch_file, code_review_prompt.output_summary_file],
            ),
        ]

        return steps, code_review_prompt

