# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import ClassVar


@dataclass
class FixReviewCommentsPrompt:
    repo_dir: str
    review_comments_summary_file: str
    review_comments_json_file: str
    pr_metadata_file: str
    output_dir: str
    prompt_template: ClassVar[str] = """

<definitions>
<repo_dir>
{repo_dir}
</repo_dir>
<review_comments_summary_file>
{review_comments_summary_file}
</review_comments_summary_file>
<review_comments_json_file>
{review_comments_json_file}
</review_comments_json_file>
<pr_metadata_file>
{pr_metadata_file}
</pr_metadata_file>
<output_dir>
{output_dir}
</output_dir>
</definitions>

<definition_explanations>
- <repo_dir> is the path to the local git repository containing the PR branch. The branch is already checked out. ALL code modifications MUST be made exclusively inside this directory.
- <review_comments_summary_file> is a human-readable summary of all PR review comments, including inline code comments with diff hunk context, general review comments, and conversation threads.
- <review_comments_json_file> is the raw JSON of inline review comments with full metadata (file path, line numbers, diff hunks).
- <pr_metadata_file> is the PR metadata JSON (title, body, base/head branches, author).
- <output_dir> is where to save any analysis or summary output files.
</definition_explanations>

<instructions>
The goal of this task is to address all PR review comments by making code changes in <repo_dir>. Follow these steps carefully:

Step 1: Read and understand the full context
- Read <pr_metadata_file> to understand the PR title, description, and what the PR is implementing. Understand the full scope of the PR, its purpose, and design decisions.
- Read <review_comments_summary_file> thoroughly to understand every review comment, what changes reviewers are requesting, and the diff context for each comment.
- Read <review_comments_json_file> for exact file paths and line numbers referenced by each comment.
- Categorize each review comment as:
    a) Code change request — reviewer wants specific code changes
    b) Question — reviewer is asking for clarification (may or may not need code changes)
    c) Suggestion — reviewer suggests an improvement (decide if it should be adopted)
    d) Nit — minor style/formatting issue
    e) Approval/acknowledgment — no action needed

Step 2: Analyze the current code
- For each review comment that requires code changes, read the relevant source file in <repo_dir> to understand the current state of the code.
- Read the full file, not just the commented lines — understand the surrounding functions, classes, imports, and how the commented code fits into the larger module.
- Cross-reference comments that relate to each other (e.g., a reviewer may request a change in one place that logically implies a change in another).
- Check if the issue raised in a comment has already been fixed since the review was posted — compare the comment's diff hunk context with the current file content.

Step 3: Plan the fixes
- Group related review comments that should be fixed together. Examples of groupings:
    - Multiple comments about the same function or class
    - Comments about the same conceptual issue across different files (e.g., "add error handling" in multiple places)
    - A comment and its follow-up replies that refine the requested change
- Determine the order of fixes — fix foundational/structural issues first, then dependent ones, then nits.
- For each group, plan the specific code changes needed.
- If a review comment is purely a question with no code change needed, or an approval/acknowledgment, skip it but note it in the summary.
- If a suggestion conflicts with the PR's design or other review comments, note the conflict and make a judgment call.

Step 4: Implement fixes
- For each logical group of related review comments:
    a) Make the code changes in the relevant files in <repo_dir>.
    b) Ensure the changes are consistent with the rest of the codebase — follow existing code style, naming conventions, patterns, and idioms visible in the file.
    c) After making changes, read the modified file to verify the changes are correct and do not break surrounding code.
    d) Stage and commit the changes with a clear message. Use the format:
        git add <specific_files>
        git commit -m "Address review: <brief description of what was fixed>"
    e) Do NOT combine unrelated fixes into a single commit. Each commit should address one logical group of related comments.
    f) Do NOT use `git add -A` or `git add .` — only add the specific files you modified.
    g) If a fix requires changes across multiple files, include all of them in the same commit if they address the same review comment(s).

Step 5: Verify
- After all fixes are committed, review all changes:
    - Run `git log --oneline -20` to see the commit history
    - Run `git diff HEAD~<n>..HEAD --stat` (where n is the number of commits made) to see all files changed
    - For each modified file, read the final version to verify correctness
- Check for regressions:
    - Verify no imports are broken
    - Verify no function signatures were inadvertently changed
    - Verify no existing tests would be broken by the changes
- If any fix introduced new issues, fix them in an additional commit.

Step 6: Generate summary
- Write a summary file to <output_dir>/fix_summary.txt containing:
    - A header with the PR number and title
    - For each review comment addressed:
        - The reviewer's original comment text (quoted)
        - The file and line referenced
        - What was changed to address it
        - The git commit hash for the fix
    - A section listing review comments that were NOT addressed, with the reason:
        - "Question/discussion — no code change needed"
        - "Already fixed in the current code"
        - "Conflicts with other review feedback — chose X instead"
        - "Out of scope for this PR"
    - A final summary line: "Addressed X of Y review comments with Z commits"
</instructions>

<output>
- Modified source files in <repo_dir> with fixes committed as separate git commits
- <output_dir>/fix_summary.txt — summary of all changes made and review comments addressed
</output>

"""

    def prompt(self):
        return self.prompt_template.format(
            repo_dir=self.repo_dir,
            review_comments_summary_file=self.review_comments_summary_file,
            review_comments_json_file=self.review_comments_json_file,
            pr_metadata_file=self.pr_metadata_file,
            output_dir=self.output_dir,
        )


def gen_FixReviewCommentsPrompt(
    repo_dir: str,
    review_comments_summary_file: str,
    review_comments_json_file: str,
    pr_metadata_file: str,
    output_dir: str,
):
    return FixReviewCommentsPrompt(
        repo_dir=repo_dir,
        review_comments_summary_file=review_comments_summary_file,
        review_comments_json_file=review_comments_json_file,
        pr_metadata_file=pr_metadata_file,
        output_dir=output_dir,
    )
