# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prompts for the review-comment fix step of a PR-review runtime iteration.

The agent edits files in the PR worktree; the runner records the resulting diff,
syncs it to the cluster and measures it. The agent never runs git or any network
command -- see the hard rules in the template.
"""

from dataclasses import dataclass
from typing import ClassVar

COMMENT_RESOLUTION_FILE = "comment_resolution.md"
SUMMARY_DRAFT_FILE = "summary_draft_R0.txt"
FOLLOWUP_COMMIT_MSG_FILE = "followup_commit_msg.txt"
REBASE_RESOLUTION_FILE = "rebase_resolution.md"
REBASE_SUMMARY_DRAFT_FILE = "rebase_summary_draft.txt"
REBASE_FIX_REPORT_FILE = "rebase_fix_report.md"


@dataclass
class PRCommentFixPrompt:
    context: str
    repo_dir: str
    output_dir: str
    iteration_dir: str
    iteration: int
    round_no: int
    pr_number: object
    pr_url: str
    new_comments_file: str
    prev_summary_file: str
    prev_verdict_file: str
    gate_report: str
    feature_toggle: str
    allowed_files: str
    prompt_template: ClassVar[str] = """

{context}

<definitions>
<repo_dir>{repo_dir}</repo_dir>
<output_dir>{output_dir}</output_dir>
<iteration_dir>{iteration_dir}</iteration_dir>
<round>{round_no}</round>
<iteration>V{iteration}</iteration>
<pr>{pr_url} (#{pr_number})</pr>
<review_input>{new_comments_file}</review_input>
<prev_summary>{prev_summary_file}</prev_summary>
<prev_verdict>{prev_verdict_file}</prev_verdict>
<feature_toggle>{feature_toggle}</feature_toggle>
<allowed_files>{allowed_files}</allowed_files>
</definitions>

<current_gates>
{gate_report}
</current_gates>

<hard_rules>
1. Edit files under <repo_dir> ONLY. Do NOT run any git command that writes:
   no add, commit, amend, checkout, reset, stash, rebase, push, tag.
   Read-only git (status, diff, log, show) is allowed.
2. Do NOT access the network. No gh, no curl, no pip install, no git fetch.
   Review comments have already been fetched for you into <review_input>.
3. Touch only <allowed_files> unless a review comment explicitly demands another
   file. Never reformat, rename or "clean up" code the PR does not already change.
4. The feature toggle <feature_toggle> MUST keep working. The verification below
   measures ON vs OFF through it; if it stops disabling the feature, the whole
   measurement is meaningless.
5. Do not weaken the change that delivers the performance win, and do not add
   fallbacks that silently route around it in order to make a reviewer's case
   pass. If a comment conflicts with the win, implement what is asked and mark
   that comment STATUS: unresolved with the conflict explained in REPLY.
6. Never write developer paths, cluster namespaces, image registries, or tool
   attribution (AI/assistant/co-authored-by) into code, comments or commit text.
</hard_rules>

<instructions>
You are addressing upstream reviewer feedback on an already-merged-quality patch.
Work in this order:

Step 1 -- Understand the current state.
Read <prev_summary> and <prev_verdict> to see what the last measured iteration
proved (performance delta, accuracy, kernel-level evidence), then read the files
in <allowed_files> in full. Compare the current tree against what the reviewer
saw: comments may refer to lines that have already moved or been fixed.

Step 2 -- Enumerate the asks.
Read <review_input>. Every `## Comment <id>` block is one thread you owe an
answer to. Classify each: bug / API or style request / design objection / test
coverage ask / question / already fixed.

Step 3 -- Implement.
Apply the changes in <repo_dir>. Keep them minimal and idiomatic for this
codebase. Preserve the ON/OFF toggle, the startup log lines that prove the
feature is active, and the unit tests that cover the new path; extend the tests
when a reviewer asks for coverage. Re-read each file you edited.

Step 4 -- Write <iteration_dir>/{resolution}.
One block per comment id from <review_input>, no omissions, exact format:

    ## Comment <id>: <file>:<line>
    STATUS: fixed | explained | unresolved
    REPLY: <1-3 sentences addressed to the reviewer, describing what changed or
           why nothing changed>
    CHANGE: <file(s) and function(s) edited, or "none">

STATUS meanings: fixed = code changed as asked; explained = answered, no code
change (already fixed, works as intended, reviewer's premise incorrect);
unresolved = you could not address it safely (conflicts with the performance
gate, needs the PR author's decision, or requires information you do not have).
Anything you are not certain about is unresolved -- never optimistic here.

Step 5 -- Write <iteration_dir>/{commit_msg}.
A conventional-commit style subject (<= 72 chars) plus a short body describing
this round's changes, for example "fix(model): address review round {round_no}".
No attribution, no co-authors.

Step 6 -- Write <iteration_dir>/{summary}.
Report: what each reviewer asked and what you did; which files changed; the
expected effect on the measured gates; what the verification run must confirm;
and anything still open. Do not claim a performance or correctness result --
the harness measures it after you finish.
</instructions>

<output>
- Edited files under <repo_dir>, restricted to <allowed_files>.
- <iteration_dir>/{resolution}
- <iteration_dir>/{commit_msg}
- <iteration_dir>/{summary}
</output>
"""

    def prompt(self):
        return self.prompt_template.format(
            context=self.context,
            repo_dir=self.repo_dir,
            output_dir=self.output_dir,
            iteration_dir=self.iteration_dir,
            iteration=self.iteration,
            round_no=self.round_no,
            pr_number=self.pr_number,
            pr_url=self.pr_url,
            new_comments_file=self.new_comments_file,
            prev_summary_file=self.prev_summary_file,
            prev_verdict_file=self.prev_verdict_file,
            gate_report=self.gate_report,
            feature_toggle=self.feature_toggle,
            allowed_files=self.allowed_files,
            resolution=COMMENT_RESOLUTION_FILE,
            commit_msg=FOLLOWUP_COMMIT_MSG_FILE,
            summary=SUMMARY_DRAFT_FILE,
        )


def gen_PRCommentFixPrompt(
    context,
    repo_dir,
    output_dir,
    iteration_dir,
    iteration,
    round_no,
    pr_number,
    pr_url,
    new_comments_file,
    prev_summary_file,
    prev_verdict_file,
    gate_report,
    feature_toggle,
    allowed_files,
):
    return PRCommentFixPrompt(
        context=context,
        repo_dir=repo_dir,
        output_dir=output_dir,
        iteration_dir=iteration_dir,
        iteration=iteration,
        round_no=round_no,
        pr_number=pr_number,
        pr_url=pr_url,
        new_comments_file=new_comments_file,
        prev_summary_file=prev_summary_file,
        prev_verdict_file=prev_verdict_file,
        gate_report=gate_report,
        feature_toggle=feature_toggle,
        allowed_files=(
            "\n    ".join(allowed_files) if allowed_files else "(none listed)"
        ),
    )


@dataclass
class RebaseConflictFixPrompt:
    context: str
    repo_dir: str
    output_dir: str
    stage_dir: str
    attempt: int
    conflicts_file: str
    prev_attempt_file: str
    pr_description_file: str
    feature_toggle: str
    prompt_template: ClassVar[str] = """

{context}

<definitions>
<repo_dir>{repo_dir}</repo_dir>
<output_dir>{output_dir}</output_dir>
<stage_dir>{stage_dir}</stage_dir>
<attempt>R{attempt}</attempt>
<conflicts>{conflicts_file}</conflicts>
<previous_attempt>{prev_attempt_file}</previous_attempt>
<pr_intent>{pr_description_file}</pr_intent>
<feature_toggle>{feature_toggle}</feature_toggle>
</definitions>

<hard_rules>
1. The repository is paused in the middle of a rebase. Edit files under
   <repo_dir> ONLY. Do NOT run any git command that writes: no add, commit,
   amend, checkout, reset, stash, rebase, rebase --continue, push, tag. The
   human script stages and continues the rebase after you finish.
   Read-only git (status, diff, log, show, merge-base) is allowed and expected.
2. Do NOT access the network. No gh, no curl, no pip install, no git fetch.
3. No conflict markers may remain anywhere: no line starting with <<<<<<<,
   =======, or >>>>>>>.
4. Keep BOTH sides working. The upstream side is authoritative for everything it
   refactored (renamed modules, changed signatures, new helpers); this PR's
   behaviour is authoritative for the optimisation it introduces. Never resolve a
   conflict by silently dropping the PR's change, and never "resolve" it by
   reverting upstream work.
5. The feature toggle <feature_toggle> must still disable the feature exactly as
   before -- the verification run measures ON versus OFF through it.
6. Reconcile callers, imports, registrations and tests after any rename: an
   unresolved AttributeError in a lazy import is the usual way a semantic
   rebase conflict shows up long after the markers are gone.
7. Never write developer paths, cluster namespaces, image registries, or tool
   attribution into code, comments or commit text.
</hard_rules>

<instructions>
You are re-applying this PR onto a newer upstream base. Resolve the conflicts
so the PR keeps exactly its original intent on top of the new tree.

Step 1 -- Read <conflicts>. It lists the rebase plan, the conflicted files and
the marked-up regions. Read <pr_intent> to recall what the PR must still do.

Step 2 -- For each conflicted file, read enough of the surrounding code to know
what upstream now expects: the new function signatures, the moved symbols, the
surrounding conventions. Use read-only git to compare, e.g.
`git -C <repo_dir> log --oneline <base>..<target> -- <file>` and
`git -C <repo_dir> show <target>:<file>`.

Step 3 -- Resolve. Edit the file until it compiles conceptually on the NEW base
with the PR's behaviour intact. Delete every marker. Re-read the file afterwards.

Step 4 -- Follow the seams. Check callers, registrations, config/env entries and
tests that reference anything upstream renamed, and fix them inside the PR's
scope. If a fix requires touching a file the PR never touched, do it and record
it -- a broken import is not a resolution.

Step 5 -- Write <stage_dir>/{resolution}, one block per file you handled, exact
format:

    ## File: <path>
    STATUS: resolved | unresolved
    NOTES: <what upstream changed here and how the PR's change was re-applied>

Anything you are not fully certain about is unresolved -- the gates re-measure
after you, so a wrong resolution costs a cycle, and an optimistic one costs a
wrong conclusion.

Step 6 -- Write <stage_dir>/{summary}: per-file reasoning, what upstream
refactoring you conformed to, files you had to touch beyond the original scope,
and what the verification run must specifically confirm.
</instructions>

<output>
- Conflict-free files under <repo_dir>
- <stage_dir>/{resolution}
- <stage_dir>/{summary}
</output>
"""

    def prompt(self):
        return self.prompt_template.format(
            context=self.context,
            repo_dir=self.repo_dir,
            output_dir=self.output_dir,
            stage_dir=self.stage_dir,
            attempt=self.attempt,
            conflicts_file=self.conflicts_file,
            prev_attempt_file=self.prev_attempt_file,
            pr_description_file=self.pr_description_file,
            feature_toggle=self.feature_toggle,
            resolution=REBASE_RESOLUTION_FILE,
            summary=REBASE_SUMMARY_DRAFT_FILE,
        )


def gen_RebaseConflictFixPrompt(
    context,
    repo_dir,
    output_dir,
    stage_dir,
    attempt,
    conflicts_file,
    prev_attempt_file,
    pr_description_file,
    feature_toggle,
):
    return RebaseConflictFixPrompt(
        context=context,
        repo_dir=repo_dir,
        output_dir=output_dir,
        stage_dir=stage_dir,
        attempt=attempt,
        conflicts_file=conflicts_file,
        prev_attempt_file=prev_attempt_file or "(first attempt)",
        pr_description_file=pr_description_file,
        feature_toggle=feature_toggle,
    )


@dataclass
class RebaseRuntimeFixPrompt:
    """Fix + verify iteration after a rebase: make the patch work on the new base.

    Marker-level conflicts are already resolved and committed. This iteration
    hunts what git could not see -- semantic conflicts, broken seams, and any
    performance or accuracy regression the measurement just reported -- and fixes
    it, exactly like a normal runtime iteration.
    """

    context: str
    repo_dir: str
    output_dir: str
    iteration_dir: str
    iteration: int
    pr_number: object
    pr_url: str
    rebase_report: str
    drift_file: str
    conflicts_file: str
    resolution_file: str
    pr_description_file: str
    prev_summary_file: str
    prev_verdict_file: str
    gate_report: str
    feature_toggle: str
    allowed_files: str
    prompt_template: ClassVar[str] = """

{context}

<definitions>
<repo_dir>{repo_dir}</repo_dir>
<output_dir>{output_dir}</output_dir>
<iteration_dir>{iteration_dir}</iteration_dir>
<iteration>V{iteration}</iteration>
<pr>{pr_url} (#{pr_number})</pr>
<rebase_report>{rebase_report}</rebase_report>
<upstream_drift>{drift_file}</upstream_drift>
<conflicts>{conflicts_file}</conflicts>
<resolution>{resolution_file}</resolution>
<pr_intent>{pr_description_file}</pr_intent>
<prev_summary>{prev_summary_file}</prev_summary>
<prev_verdict>{prev_verdict_file}</prev_verdict>
<feature_toggle>{feature_toggle}</feature_toggle>
<allowed_files>{allowed_files}</allowed_files>
</definitions>

<current_gates>
{gate_report}
</current_gates>

<hard_rules>
1. Edit files under <repo_dir> ONLY. Do NOT run any git command that writes: no
   add, commit, amend, checkout, reset, stash, rebase, push, tag. Read-only git
   (status, diff, log, show, merge-base, grep) is allowed and expected.
2. Do NOT access the network. No gh, no curl, no pip install, no git fetch.
3. Upstream's new code is authoritative. A file that upstream changed and this PR
   never owned must stay byte-identical to the new base -- never "fix" a rebase by
   reverting, deleting or re-editing upstream work.
4. Keep BOTH intents alive: upstream's refactor (renames, new signatures, new
   helpers) and this PR's optimisation. Dropping the PR's change to make the tree
   import is a regression, not a fix.
5. The feature toggle <feature_toggle> MUST still disable the feature exactly as
   before. The measurement below is ON versus OFF through it; if it stops working,
   the result is meaningless.
6. Do not weaken or route around the optimisation to make a run succeed (no silent
   fallback to the old kernel, no disabled code path, no lowered precision that was
   not already there). If the win cannot survive the new base, say so in the report
   as unresolved instead of faking a pass.
7. Stay inside <allowed_files> where possible. If upstream renamed something and a
   caller outside that list must follow, you may edit it -- but every extra file
   must be named in the report, because it is recorded and reviewed.
8. Never write developer paths, cluster namespaces, image registries, or tool
   attribution (AI/assistant/co-authored-by) into code, comments or commit text.
9. Do not start servers, benchmarks or GPU work yourself. The harness measures the
   tree after you finish; cheap local reading and static checks are your evidence.
10. A failure inside code this PR never owned, caused by a third-party package
    being older or newer than requirements pin it, is ENVIRONMENT drift -- the
    image, not the rebase. Say so in the report and leave the tree alone. Do not
    add try/except imports, importlib loaders, sys.path tricks or vendored
    shims to route around it: that ships a hack upstream would reject, and it
    hides the reason the measurement could not run.
</hard_rules>

<instructions>
You are the fix step of a runtime iteration that runs right after this PR was
rebased onto a newer upstream base. <rebase_report> says what moved, what
conflicted and how it was resolved. Conflict markers are gone already -- what is
left to find is what git could not see.

Step 1 -- Reconstruct both intents.
Read <pr_intent>, then <resolution> and <conflicts> for the files that actually
collided, then <upstream_drift> for what upstream did to those same files. For a
file still in doubt, compare the three trees directly:
  git -C <repo_dir> show <old base>:<file>
  git -C <repo_dir> show <new base>:<file>
  git -C <repo_dir> diff <new base> HEAD -- <file>

Step 2 -- Audit the resolution for semantic damage.
For every conflicted file, read the whole file, not just the hunks. Check that
every symbol the PR uses still exists under the name upstream gave it; that
signatures, keyword-only arguments, decorators, registries/config registrations,
env-var wiring, imports (especially lazy/deferred ones), and the PR's own tests
all agree with the new tree. A rebase regression usually surfaces as an
AttributeError/ImportError at first use, or as a silently unused fast path.

Step 3 -- Diagnose whatever the last measurement reported.
If <prev_verdict> / <prev_summary> exist, they are the priority: a failed run
status, a traceback in the run logs under <output_dir>/runtime/, a perf delta
below the retention floor (read <iteration_dir>/diagnostic_profile/ if NSYS ran),
or an accuracy drop. Localise the cause in the rebased tree and fix that cause --
not the symptom.

Step 4 -- Fix, minimally and idiomatically.
Preserve the startup log lines that prove the fast path is active and the tests
that cover it; extend tests when upstream moved behaviour they assert. Re-read
every file you edited. Confirm the toggle path still compiles to the old code.

Step 5 -- Write <iteration_dir>/{report}. Report a block for EVERY file that
conflicted, EVERY file you edited, and -- when nothing conflicted -- one block for
each file in <allowed_files> confirming you audited it against the new base. An
empty report fails the stage, so a clean rebase is still reported, not skipped.
Exact format:

    ## File: <path>
    STATUS: resolved | unresolved
    NOTES: <what the new base changed here, what you verified or repaired, and how
           you know it is right>
    CHANGE: <file(s)/function(s) you edited, or "none">

STATUS: resolved = you have verified this file is correct on the new base;
unresolved = still broken, or you lack the information to decide. Anything you are
not certain about is unresolved -- the gates re-measure after you, so an optimistic
answer costs a wrong conclusion, not a shorter run.

Step 6 -- If you edited any file, write <iteration_dir>/{commit_msg}: a
conventional-commit subject (<= 72 chars) plus a short body, e.g.
"fix(model): repair fused gemm path after rebase". No attribution, no co-authors.
If you edited nothing, do not write it.

Step 7 -- Write <iteration_dir>/{summary}: what you audited, what upstream
refactoring you conformed to, every out-of-scope file you had to touch and why,
what you changed, and specifically what the measurement must confirm. Do not claim
a performance or correctness result -- the harness measures it after you.
</instructions>

<output>
- Repaired files under <repo_dir> (possibly none)
- <iteration_dir>/{report}
- <iteration_dir>/{commit_msg} (only if you edited a file)
- <iteration_dir>/{summary}
</output>
"""

    def prompt(self):
        return self.prompt_template.format(
            context=self.context,
            repo_dir=self.repo_dir,
            output_dir=self.output_dir,
            iteration_dir=self.iteration_dir,
            iteration=self.iteration,
            pr_number=self.pr_number,
            pr_url=self.pr_url,
            rebase_report=self.rebase_report,
            drift_file=self.drift_file,
            conflicts_file=self.conflicts_file,
            resolution_file=self.resolution_file,
            pr_description_file=self.pr_description_file,
            prev_summary_file=self.prev_summary_file,
            prev_verdict_file=self.prev_verdict_file,
            gate_report=self.gate_report,
            feature_toggle=self.feature_toggle,
            allowed_files=self.allowed_files,
            report=REBASE_FIX_REPORT_FILE,
            commit_msg=FOLLOWUP_COMMIT_MSG_FILE,
            summary=SUMMARY_DRAFT_FILE,
        )


def gen_RebaseRuntimeFixPrompt(
    context,
    repo_dir,
    output_dir,
    iteration_dir,
    iteration,
    pr_number,
    pr_url,
    rebase_report,
    drift_file,
    conflicts_file,
    resolution_file,
    pr_description_file,
    prev_summary_file,
    prev_verdict_file,
    gate_report,
    feature_toggle,
    allowed_files,
):
    return RebaseRuntimeFixPrompt(
        context=context,
        repo_dir=repo_dir,
        output_dir=output_dir,
        iteration_dir=iteration_dir,
        iteration=iteration,
        pr_number=pr_number,
        pr_url=pr_url,
        rebase_report=rebase_report,
        drift_file=drift_file,
        conflicts_file=conflicts_file or "(none -- the rebase was clean)",
        resolution_file=resolution_file or "(none -- the rebase was clean)",
        pr_description_file=pr_description_file,
        prev_summary_file=prev_summary_file or "(none)",
        prev_verdict_file=prev_verdict_file or "(none)",
        gate_report=gate_report,
        feature_toggle=feature_toggle,
        allowed_files=(
            "\n    ".join(allowed_files) if allowed_files else "(none listed)"
        ),
    )
