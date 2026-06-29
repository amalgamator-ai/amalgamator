import glob
import os
import re
from dataclasses import dataclass
from typing import Optional

from openclaw_vs_hermes.agent_prompts import _prev_iteration_section


# ---------------------------------------------------------------------------
# File naming with round IDs
# ---------------------------------------------------------------------------

PR_COMMENTS_FILE_TEMPLATE = "PR_comments_{}.txt"
PR_FIX_FILE_PREFIX = "pr_fix"
PR_FIXES_SUMMARY_TEMPLATE = "PR_fixes_{}.txt"
CODE_GEN_FILE_PREFIX = "code_gen"


def pr_comments_filename(round_id: int) -> str:
    return PR_COMMENTS_FILE_TEMPLATE.format(round_id)


def pr_fix_patch_filename(round_id: int, iteration: int) -> str:
    return "{}_R{}_V{}.patch".format(PR_FIX_FILE_PREFIX, round_id, iteration)


def pr_fix_summary_filename(round_id: int, iteration: int) -> str:
    return "{}_summary_R{}_V{}.txt".format(PR_FIX_FILE_PREFIX, round_id, iteration)


def pr_fixes_report_filename(round_id: int) -> str:
    return PR_FIXES_SUMMARY_TEMPLATE.format(round_id)


def _detect_max_code_gen_version(output_dir: str) -> int:
    """Find the highest code_gen version number across all code_gen_*.patch files."""
    pattern = re.compile(r'^code_gen(?:_review)?_V(\d+)(?:_pr_r_\d+)?\.patch$')
    max_ver = 0
    for f in os.listdir(output_dir):
        m = pattern.match(f)
        if m:
            max_ver = max(max_ver, int(m.group(1)))
    runtime_pattern = re.compile(r'^code_gen_runtime_V(\d+)\.patch$')
    for f in os.listdir(output_dir):
        m = runtime_pattern.match(f)
        if m:
            max_ver = max(max_ver, int(m.group(1)))
    return max_ver


def code_gen_pr_patch_filename(version: int, round_id: int) -> str:
    return "{}_V{}_pr_r_{}.patch".format(CODE_GEN_FILE_PREFIX, version, round_id)


def code_gen_pr_summary_filename(version: int, round_id: int) -> str:
    return "{}_summary_V{}_pr_r_{}.txt".format(CODE_GEN_FILE_PREFIX, version, round_id)


def code_gen_pr_review_patch_filename(version: int, round_id: int) -> str:
    return "{}_review_V{}_pr_r_{}.patch".format(CODE_GEN_FILE_PREFIX, version, round_id)


def code_gen_pr_review_summary_filename(version: int, round_id: int) -> str:
    return "{}_review_summary_V{}_pr_r_{}.txt".format(CODE_GEN_FILE_PREFIX, version, round_id)


def detect_next_round_id(output_dir: str) -> int:
    """Scan output_dir for existing PR_comments_N.txt files and return the next ID."""
    pattern = re.compile(r'^PR_comments_(\d+)\.txt$')
    max_id = 0
    for f in os.listdir(output_dir):
        m = pattern.match(f)
        if m:
            max_id = max(max_id, int(m.group(1)))
    return max_id + 1


def detect_current_round_id(output_dir: str) -> int:
    """Return the highest existing round ID, or 0 if none exist."""
    pattern = re.compile(r'^PR_comments_(\d+)\.txt$')
    max_id = 0
    for f in os.listdir(output_dir):
        m = pattern.match(f)
        if m:
            max_id = max(max_id, int(m.group(1)))
    return max_id


# ---------------------------------------------------------------------------
# Context string for PR review prompts
# ---------------------------------------------------------------------------

def create_pr_review_context_str(config):
    return """
<context>

<output_dir>
{output_dir}
</output_dir>

<target_project>openclaw</target_project>
<target_project_full_name>OpenClaw</target_project_full_name>
<target_project_language>TypeScript</target_project_language>
<target_project_dir>
{openclaw_source_dir}
</target_project_dir>
<target_project_github>
{openclaw_github_url}
</target_project_github>

</context>

<context_explanations>
- <output_dir> is the directory containing all pipeline artifacts AND where new PR review files are written.
- <target_project_dir> is the local path to the OpenClaw source code (on the feature branch).
- ALL code modifications MUST be made inside <target_project_dir>.
- ALL output files MUST be written to <output_dir>.
</context_explanations>

""".format(
        output_dir=config.output_dir,
        openclaw_source_dir=config.openclaw_source_dir,
        openclaw_github_url=config.openclaw_github_url,
    )


# ---------------------------------------------------------------------------
# Artifact context builder
# ---------------------------------------------------------------------------

def _find_highest_version(output_dir, prefix, suffix):
    """Find the highest version file matching {prefix}_V{N}{suffix}."""
    pattern = re.compile(
        r'^{}(?:_review)?_V(\d+){}$'.format(re.escape(prefix), re.escape(suffix))
    )
    best_file = None
    best_ver = 0
    for f in os.listdir(output_dir):
        m = pattern.match(f)
        if m:
            ver = int(m.group(1))
            if ver > best_ver:
                best_ver = ver
                best_file = f
    return best_file


def _build_artifact_context_section(output_dir: str, current_round_id: int) -> str:
    """Build a prompt section listing all pipeline artifacts for Claude to read."""
    sections = []

    sections.append(
        "<pipeline_artifacts>\n"
        "Read and understand ALL of the following files in <output_dir>. They contain "
        "the complete history of how this feature was designed, planned, coded, tested, "
        "and debugged. This context is critical for making informed fixes.\n"
    )

    # Code traces
    traces = []
    for name in ["hermes_code_trace.txt", "openclaw_code_trace.txt"]:
        if os.path.isfile(os.path.join(output_dir, name)):
            traces.append(name)
    if traces:
        sections.append("<code_traces>\n{}\n</code_traces>\n".format(
            "\n".join(traces)
        ))

    # Code port plans — highest review version
    plan_files = []
    for f in sorted(os.listdir(output_dir)):
        if f.startswith("code_port_plan_review_V") and f.endswith(".txt"):
            plan_files.append(f)
        elif f.startswith("code_port_plan_review_summary_V") and f.endswith(".txt"):
            plan_files.append(f)
    if plan_files:
        highest = plan_files[-1]
        summary = highest.replace("_review_V", "_review_summary_V") if "summary" not in highest else highest
        review = highest.replace("_review_summary_V", "_review_V") if "summary" in highest else highest
        items = sorted(set([review, summary]))
        items = [i for i in items if os.path.isfile(os.path.join(output_dir, i))]
        if items:
            sections.append("<code_port_plans>\n{}\n</code_port_plans>\n".format(
                "\n".join(items)
            ))

    # Test plans — highest review version
    test_files = []
    for f in sorted(os.listdir(output_dir)):
        if f.startswith("test_plan_review_V") and f.endswith(".txt"):
            test_files.append(f)
        elif f.startswith("test_plan_review_summary_V") and f.endswith(".txt"):
            test_files.append(f)
    if test_files:
        highest = test_files[-1]
        summary = highest.replace("_review_V", "_review_summary_V") if "summary" not in highest else highest
        review = highest.replace("_review_summary_V", "_review_V") if "summary" in highest else highest
        items = sorted(set([review, summary]))
        items = [i for i in items if os.path.isfile(os.path.join(output_dir, i))]
        if items:
            sections.append("<test_plans>\n{}\n</test_plans>\n".format(
                "\n".join(items)
            ))

    # Code gen patches — highest review + runtime
    patch_files = []
    for f in sorted(os.listdir(output_dir)):
        if f.startswith("code_gen_review_V") and f.endswith(".patch"):
            patch_files.append(f)
    runtime_patches = []
    for f in sorted(os.listdir(output_dir)):
        if f.startswith("code_gen_runtime_V") and f.endswith(".patch"):
            runtime_patches.append(f)
    all_patches = []
    if patch_files:
        all_patches.append(patch_files[-1])
    if runtime_patches:
        all_patches.append(runtime_patches[-1])
    if all_patches:
        sections.append("<code_patches>\n{}\n</code_patches>\n".format(
            "\n".join(all_patches)
        ))

    # Code gen summaries + runtime summaries
    summary_files = []
    for f in sorted(os.listdir(output_dir)):
        if f.startswith("code_gen_review_summary_V") and f.endswith(".txt"):
            summary_files.append(f)
        elif f.startswith("code_gen_runtime_summary_V") and f.endswith(".txt"):
            summary_files.append(f)
    if os.path.isfile(os.path.join(output_dir, "runtime_success_result.txt")):
        summary_files.append("runtime_success_result.txt")
    if summary_files:
        items = []
        if summary_files:
            items.append(summary_files[-1] if summary_files[-1] != "runtime_success_result.txt" else summary_files[-1])
        if "runtime_success_result.txt" in summary_files:
            items.append("runtime_success_result.txt")
        items = sorted(set(items))
        sections.append("<runtime_summaries>\n{}\n</runtime_summaries>\n".format(
            "\n".join(items)
        ))

    # Previous PR fix rounds
    prev_fixes = []
    for rid in range(1, current_round_id):
        comments_f = pr_comments_filename(rid)
        fixes_f = pr_fixes_report_filename(rid)
        if os.path.isfile(os.path.join(output_dir, comments_f)):
            prev_fixes.append(comments_f)
        if os.path.isfile(os.path.join(output_dir, fixes_f)):
            prev_fixes.append(fixes_f)
        # Also include the highest fix patch from that round
        round_patches = sorted(glob.glob(
            os.path.join(output_dir, "{}_R{}_V*.patch".format(PR_FIX_FILE_PREFIX, rid))
        ))
        if round_patches:
            prev_fixes.append(os.path.basename(round_patches[-1]))
    if prev_fixes:
        sections.append(
            "<previous_pr_review_rounds>\n{}\n</previous_pr_review_rounds>\n".format(
                "\n".join(prev_fixes)
            )
        )

    sections.append("</pipeline_artifacts>")
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

FIX_PR_COMMENTS_TEMPLATE = """

{context}

{artifact_context}

<definitions>
<pr_comments_file>
{pr_comments_file}
</pr_comments_file>
<branch_name>
{branch_name}
</branch_name>
<round_id>
{round_id}
</round_id>
</definitions>

{prev_iteration_section}

<instructions>
The goal of this task is to address PR review comments (round {round_id}) for a feature that was ported from Hermes to OpenClaw. The feature code is already committed on branch <branch_name> in <target_project_dir>. Reviewers have left comments that need to be addressed. Think hard for this task.

Do the following:

1. Read and understand ALL files listed in <pipeline_artifacts>. These contain the complete history of how this feature was designed, planned, coded, tested, and debugged. You MUST understand this context to make informed fixes.

2. Read <pr_comments_file> from <output_dir>. Analyze each comment:
   - Categorize: bug fix, style issue, design concern, missing test, documentation, nit
   - Determine priority: must-fix, should-fix, optional
   - Identify the affected file(s) and code area

3. For each must-fix and should-fix comment:
   - Understand WHY the original code was written that way (from the pipeline artifacts)
   - Determine the correct fix that addresses the reviewer's concern while preserving the feature's correctness
   - Make the code change in <target_project_dir>
   - If a reviewer requests additional tests, add them

4. After all fixes are applied:
   - Run the build: cd <target_project_dir> && {build_command}
   - Run the tests: cd <target_project_dir> && {test_command}
   - Fix any build/test failures caused by the changes
   - Iterate until build and tests pass

5. Generate a patch of all changes since the last commit:
   cd <target_project_dir> && git diff HEAD > <output_dir>/{output_patch_file}

NOTE: This is PR review round {round_id}. The output patch file {output_patch_file}
continues the code_gen version sequence from the original pipeline to maintain
a continuous history of code evolution.

IMPORTANT:
- ALL code modifications MUST be made exclusively inside <target_project_dir>
- Do NOT change code that is unrelated to the PR comments
- Preserve the feature's correctness -- do not regress behavior while fixing style/design issues
- If a reviewer comment is unclear or seems wrong, document your reasoning for how you handled it in the summary
</instructions>

<output>
- Apply fixes directly to <target_project_dir>
- Dump the fix patch to <output_dir>/{output_patch_file}
- Dump a summary to <output_dir>/{output_summary_file} that includes:
    - A header noting this is PR review round {round_id}
    - For each PR comment: what was requested, what was done, and why
    - List of all files modified
    - Build and test results
    - Any comments that were intentionally not addressed, with reasoning
</output>

"""


SUMMARIZE_PR_FIXES_TEMPLATE = """

{context}

<definitions>
<pr_comments_file>
{pr_comments_file}
</pr_comments_file>
<fix_patch_file>
{fix_patch_file}
</fix_patch_file>
<fix_summary_file>
{fix_summary_file}
</fix_summary_file>
<round_id>
{round_id}
</round_id>
</definitions>

<instructions>
Generate a final summary of all PR review fixes applied in round {round_id}. This summary will be used to update the PR on GitHub.

Do the following:
1. Read the original PR comments from <pr_comments_file>
2. Read the fix patch from <fix_patch_file>
3. Read the fix summary from <fix_summary_file>

Write a structured summary suitable for posting as a PR update comment:
- Header indicating this is round {round_id} of PR review fixes
- For each reviewer comment: quote the original comment, describe what was changed
- List all files modified with a one-line description of each change
- Report build and test results
- Note any comments that were not addressed, with explanation
- Provide the git commands needed to push the updated branch
</instructions>

<output>
- Write the summary to <output_dir>/{output_file}
</output>

"""


# ---------------------------------------------------------------------------
# Prompt dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FixPRCommentsPrompt:
    context: str
    artifact_context: str
    pr_comments_file: str
    branch_name: str
    round_id: int
    build_command: str
    test_command: str
    output_patch_file: str
    output_summary_file: str
    prev_output_patch_file: Optional[str]
    prev_output_summary_file: Optional[str]
    iteration: int

    def prompt(self):
        return FIX_PR_COMMENTS_TEMPLATE.format(
            context=self.context,
            artifact_context=self.artifact_context,
            pr_comments_file=self.pr_comments_file,
            branch_name=self.branch_name,
            round_id=self.round_id,
            build_command=self.build_command,
            test_command=self.test_command,
            output_patch_file=self.output_patch_file,
            output_summary_file=self.output_summary_file,
            prev_iteration_section=_prev_iteration_section(
                self.prev_output_patch_file, self.prev_output_summary_file,
            ),
            iteration=self.iteration,
        )


@dataclass
class SummarizePRFixesPrompt:
    context: str
    pr_comments_file: str
    fix_patch_file: str
    fix_summary_file: str
    round_id: int
    output_file: str

    def prompt(self):
        return SUMMARIZE_PR_FIXES_TEMPLATE.format(
            context=self.context,
            pr_comments_file=self.pr_comments_file,
            fix_patch_file=self.fix_patch_file,
            fix_summary_file=self.fix_summary_file,
            round_id=self.round_id,
            output_file=self.output_file,
        )


# ---------------------------------------------------------------------------
# Review prompt for PR fix code (mirrors the code gen review pattern)
# ---------------------------------------------------------------------------

REVIEW_PR_FIX_TEMPLATE = """

{context}

{artifact_context}

<definitions>
<pr_comments_file>
{pr_comments_file}
</pr_comments_file>
<input_patch_file>
{input_patch_file}
</input_patch_file>
<input_summary_file>
{input_summary_file}
</input_summary_file>
<branch_name>
{branch_name}
</branch_name>
<round_id>
{round_id}
</round_id>
</definitions>

<instructions>
The goal of this task is to perform a critical, in-depth review of the PR fix code patch in <input_patch_file> (PR review round {round_id}). This patch was generated to address reviewer comments from <pr_comments_file>. Think hard.

Do the following:

1. Read and understand ALL files listed in <pipeline_artifacts>. These contain the complete history of the feature's design and implementation.

2. Read the original PR comments from <pr_comments_file>.

3. Read the fix patch from <input_patch_file> and its summary from <input_summary_file>.

4. Review the fix code for:
   - Correctness: do the fixes actually address the reviewer comments?
   - Completeness: are there any reviewer comments that were missed or inadequately addressed?
   - Regressions: do the fixes break any existing behavior or tests?
   - TypeScript quality: proper typing, no `any` types unless necessary, idiomatic patterns
   - Test coverage: are new/modified behaviors covered by tests?
   - Style consistency: do fixes follow OpenClaw's code conventions?
   - Over-engineering: are any fixes more complex than necessary?

5. For each issue found:
   - Describe what is wrong
   - Explain why it matters
   - Make the fix in <target_project_dir>

6. After all review fixes are applied:
   - Run the build: cd <target_project_dir> && {build_command}
   - Run the tests: cd <target_project_dir> && {test_command}
   - Fix any build/test failures

7. Generate the reviewed patch:
   cd <target_project_dir> && git diff HEAD > <output_dir>/{output_patch_file}

NOTE: This is the review step for PR review round {round_id}. The output files
continue the code_gen version sequence.

IMPORTANT:
- ALL code modifications MUST be made exclusively inside <target_project_dir>
- If no changes were needed to the fix code, write CONVERGED as the very first line of <output_dir>/{output_summary_file}
</instructions>

<output>
- Dump the reviewed patch to <output_dir>/{output_patch_file}
- Dump a summary to <output_dir>/{output_summary_file} that includes:
    - A header noting this is the review of PR review round {round_id}
    - Issues found in the fix code and how they were corrected
    - Build and test results
- IMPORTANT: If no changes were needed, write CONVERGED as the very first line of <output_dir>/{output_summary_file}. Otherwise, do NOT write CONVERGED.
</output>

"""


@dataclass
class ReviewPRFixPrompt:
    context: str
    artifact_context: str
    pr_comments_file: str
    input_patch_file: str
    input_summary_file: str
    branch_name: str
    round_id: int
    build_command: str
    test_command: str
    output_patch_file: str
    output_summary_file: str

    def prompt(self):
        return REVIEW_PR_FIX_TEMPLATE.format(
            context=self.context,
            artifact_context=self.artifact_context,
            pr_comments_file=self.pr_comments_file,
            input_patch_file=self.input_patch_file,
            input_summary_file=self.input_summary_file,
            branch_name=self.branch_name,
            round_id=self.round_id,
            build_command=self.build_command,
            test_command=self.test_command,
            output_patch_file=self.output_patch_file,
            output_summary_file=self.output_summary_file,
        )
