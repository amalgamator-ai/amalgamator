# PR flow: submit a code-gen result upstream and drive the review loop

Automates the part that comes *after* `auto_code_gen` passes its perf and
correctness gates: opening a PR against `vllm-project/vllm`, then repeatedly
turning reviewer comments into another verified runtime iteration until the PR
merges.

## The invariant

**The AI never touches git or GitHub.** Every command that reads from or writes
to a remote is a bash script that a human runs. Each script:

1. **announces** the exact commands it is about to run;
2. runs **local, read-only guardrails** (`llm.auto_code_gen.pr_flow.checks`) and
   refuses on failure;
3. asks for confirmation -- typing `yes` and nothing else proceeds;
4. **executes visibly**, so SSH passphrase prompts reach the human terminal;
5. **records** the outcome in `pr_state.json` and a timestamped log.

The AI side (the review-fix iteration) runs with git network access disabled
(`GIT_SSH_COMMAND=/bin/false`, `GH_TOKEN`/`GITHUB_TOKEN` cleared) and only
edits files inside the scope of the original patch.

## Flow

```
 step 0  pr_flow_step_0_prepare.sh    local   init pr_state.json
 step 1  pr_flow_step_1_commit.sh     local   commit the verified worktree
 -------------------------------------------------------------------------
 stage R -- only when upstream has moved (see "Rebasing"):
 step R1 pr_flow_step_r1_rebase.sh         NETWORK(opt) fetch + start the rebase
        run_rebase_conflict_fix.py          local        resolve conflicts, no git
 step R1 pr_flow_step_r1_rebase.sh --continue local     stage + replay the rest
        run_rebase_runtime_iters.py         local        repair, then re-prove both gates
 -------------------------------------------------------------------------
 step 2  pr_flow_step_2_push.sh       NETWORK push to fork + gh pr create
 step 2b pr_flow_step_2b_suggest_reviewers.sh NETWORK(read) rank reviewers -> pr_reviewers.txt draft
        (human reviews/edits the draft -- nothing is posted yet)
 step 2c pr_flow_step_2c_post_reviewers.sh    NETWORK post the draft as a PR comment
 ---------------------------------------------------------------------------
 round loop, repeated until the PR merges:
 step 3  pr_flow_step_3_fetch_reviews.sh  NETWORK(read)  gh api -> inbox/round_N
        run_pr_review_iters.py           local        fix + perf + correctness
           \-- while comments remain unresolved, another iteration (V n+1)
        (human reviews the diff)
 step 1  pr_flow_step_1_commit.sh           local      commit the fixes (follow-up commit)
 step 4  pr_flow_step_4_update_pr.sh        NETWORK    push + PR body/comments
        (a mid-review rebase is published with --force-with-lease)
 ---------------------------------------------------------------------------
 step 3 again reports state=MERGED -> done
```

A round is only ever pushed after `verify_verdict.json` says PASS, so reviewer
feedback can never degrade the measured improvement: the perf gate keeps a
retention floor at 80% of the accepted delta, correctness re-runs lm_eval
against the shared OFF baseline, and *closure* requires an answer to every
fetched comment (`fixed` or `explained`). NSYS profiling runs only when the
perf gate fails, which is the only case where the cause must be localized.

## Rebasing

Upstream moves. A PR whose base commit is weeks old is harder to review and
harder to merge, so the flow can move it onto a newer base. A rebase is the one
stage that rewrites history *and* invalidates every measurement, so it is a
stage of its own with the same human/AI split as everything else:

| part | who | what |
| --- | --- | --- |
| R1 start | human | `pr_flow_step_r1_rebase.sh [--fetch] [--target upstream/main]`: announces, creates the `pr-flow/pre-rebase-<short>` safety tag, runs `rebase --onto <target> <old-base> <branch>`; exits **30** when it pauses |
| R1b | AI | `run_rebase_conflict_fix.py`: edits the paused worktree until no markers remain -- **no git writes at all** |
| R1c | human | the same script with `--continue`: stages the resolutions and replays the remaining commits; `--abort` returns to `pre_rebase_head` |
| R2 | AI | `run_rebase_runtime_iters.py`: a **normal runtime iteration** -- the agent repairs whatever the rebase broke, then the tree is re-measured (OFF arms for the new base, bench ON, perf gate, lm_eval; NSYS only if perf regressed). Repeats until green or exhausted |

`rebase plan` answers *whether* to rebase before anything happens: how far
upstream has moved, which files both sides touched, and therefore where conflicts
are likely. `checks rebase-plan` refuses a target that is not present locally, a
target that is already the base, and a worktree with nothing committed to replay.

### Why R2 fixes, not just measures

A rebase that stops conflicting textually can still be wrong. Git merges text, so
what survives a clean rebase is the *text* of both sides, not their *meaning*: the
patch may call a symbol upstream renamed, register into a table that moved, or
take a path that upstream's refactor quietly stopped using. Nothing in a textual
rebase flags those, and none of them show up until the tree runs. So R2 is the
same shape as any other runtime iteration -- fix first, measure second -- driven
by the evidence a plain iteration cannot see:

| evidence the R2 fix step is given | file |
| --- | --- |
| what the rebase moved, and who owns which file | `V<n>/rebase_report.txt` |
| what upstream did to the files this PR touches | `V<n>/upstream_drift.patch` |
| what conflicted and how it was resolved | `rebase/R<n>/CONFLICTS.md`, `rebase_resolution.md` |
| the previous verdict / summary and gate floors | `verify_verdict.json`, `summary.txt`, the gate report |
| kernel-level evidence, when perf regressed | `V<n>/diagnostic_profile/` (NSYS) |

Its report is `V<n>/rebase_fix_report.md` -- one `## File:` block per conflicted
file, per edited file, and (when nothing conflicted) per file it audited --
`STATUS: resolved|unresolved`. That report *is* the stage's closure gate, so a file
the agent is unsure about fails the verdict instead of passing on optimism, and an
empty report fails too: a clean rebase is still an audited rebase.

Two mechanical safety nets run around the fix step, because no gate can catch
these later:

- **Upstream-owned files are restored.** A file upstream changed and this PR never
  owned must stay byte-identical to the new base; if the fix step edits one, it is
  checked out again and recorded in `V<n>/reverted_upstream_owned.txt`. Silently
  reverting upstream work is the one rebase error that would merge cleanly.
- **A repaired tree has to compile.** Every changed `.py` file is compiled locally
  and conflict markers are re-scanned, so a bad edit costs seconds
  (`V<n>/syntax_check.txt`) instead of a cluster run.
- **The measured bytes are fingerprinted.** `V<n>/measured_content.json` records a
  hash of every file the run measured. The fix step edits the worktree and the
  harness measures *that*; the human then commits the same bytes, which is by
  definition a different commit. Step 4 therefore proves content equality against
  that fingerprint instead of comparing commit ids -- which would make any
  commit-after-measurement round impossible to push.

Out-of-scope edits are *recorded and reported* rather than reverted, unlike a
review round: following an upstream rename legitimately reaches files the PR never
touched, and the human reviews that diff before pushing.

Two rules make this safe:

- **A rebase always re-measures.** The shared OFF baselines were taken against
  the old base, so new ON numbers against them would compare two different
  upstream trees. `gates.baseline_is_valid` keys each baseline on its
  `base_commit`, so the base move alone forces the re-measure; R2 performs it and
  re-stamps them.
- **Rewritten history is published under a lease only.** After a rebase
  `head_sha` is no longer a descendant of `remote_head_sha` (what the fork
  holds). `checks clean-tree` refuses that unless `--allow-rewritten`, which step
  4 passes only when given `--force-with-lease`, and then pushes with
  `--force-with-lease=<branch>:<published>`.

If the gates stay red after the fix iterations (`--max-iterations`, 3 by default),
that stops being a machine question: either the conflict resolution is wrong or the
win does not survive the move. The runner writes
`pr_p3/rebase/escalation_runtime_V<n>.md` with every failure it saw and says do not
publish. The human choice is to re-run R2 with better information, redo stage R1
with a different resolution, or return to the pre-rebase base -- which the
`pr-flow/pre-rebase-*` tag keeps available -- and let the PR merge on it.

When R2 does repair something, that repair is a real commit and goes through step 1
with `--message-file V<n>/followup_commit_msg.txt` before the push, so reviewers see
exactly what the rebase cost.

## Layout

Everything the flow writes lives under `<output_dir>/pr_p3/`:

| path | meaning |
| --- | --- |
| `pr_state.json` | single source of truth shared by scripts and the runner |
| `inbox/round_N/` | raw `gh` JSON + `NEW_COMMENTS.md`, `pending_comments.json`, `review_summary.txt` |
| `comments/round_N_comment.md` | rendered PR comment, posted by step 4 |
| `escalation_round_N.md` | written when a round cannot converge |
| `rebase/R<n>/` | per rebase: `CONFLICTS.md` (marked hunks), `rebase_plan.json`, `rebase_resolution.md` (agent), `rebase_summary.txt`, `resolution_summary.json` |
| `rebase/escalation_stage_R<n>.md` | written when conflicts survive every attempt |
| `rebase/escalation_runtime_V<n>.md` | written when R2 cannot reach green gates |
| `logs/` | one log per script execution |

Outside it, and read rather than owned:

| path | meaning |
| --- | --- |
| `<output_dir>/pr-generate/{commit_desc,pr_desc}.txt` | commit/PR text written by `run_generate_pr_commands --descs-only`, validated before use |
| `<output_dir>/runtime/V<n>/` | per-iteration fixes, benches, `verify_verdict.json`, `comment_resolution.md`, `thread_replies.json` |
| `<output_dir>/runtime/V<n>/` (rebase stage) | `rebase_report.txt`, `upstream_drift.patch`, `rebase_fix_report.md`, `syntax_check.txt`, `reverted_upstream_owned.txt` |
| `<output_dir>/runtime/V<n>/measured_content.json` | hashes of the bytes that iteration measured; what step 4 validates the push against |
| `<output_dir>/runtime/baselines/` | shared OFF arms (bench / lm_eval / nsys), reused until the base commit changes |

## Modules

- `pr_state.py` -- state document, atomic writes, field whitelist, gate history.
- `checks.py` -- the guardrails; the exit codes below are the script contract.
- `review_comments.py` -- raw GitHub JSON -> threads -> pending comments for the agent.
- `gates.py` -- perf retention, lm_eval tolerance, comment closure, verdict file.
- `run_pr_review_iters.py` -- the AI iteration: fix -> sync -> bench -> lm_eval -> verdict.
- `gh_publish.py` -- renders the round comment and per-thread replies; posts only with `--execute`.
- `prompts.py` -- the review-fix prompt, including the no-git / no-network rules.
- `rebase.py` -- rebase planning, conflict snapshots, resolution parsing (read-only).
- `run_rebase_conflict_fix.py` -- the AI half of a paused rebase (edits files, never git).
- `run_rebase_runtime_iters.py` -- repair what the rebase broke, then re-prove both gates.

Three state fields carry a rebase across the human/AI boundary: `pre_rebase_head`
(where `--abort` returns), `rebase_target` (what to land on), and
`remote_head_sha` (what the fork holds -- written only by steps 2 and 4, so a
rebase before the first push still looks like local work).

## Exit codes

| code | meaning | typical reaction |
| --- | --- | --- |
| 0 | OK | continue |
| 21 | state missing/unusable, or state and config point at different clones | re-run step 0 |
| 22 | change scope violation or internal string leak | fix the patch text |
| 23 | commit/PR description invalid (title too long, no `## Summary`, leak) | regenerate descs |
| 24 | tree dirty, HEAD not `state.head_sha`, or nothing new to push | commit, or rebase outside the flow |
| 25 | branch name unsafe/protected/colliding | choose another `--remote-branch` |
| 26 | remotes wrong (fork missing, `file://`, fork == upstream) | `git remote add fork ...` |
| 27 | verdict not green or measured for a different HEAD | re-run the iteration |
| 28 | nothing pending (no new review comments) | wait for reviewers |
| 29 | `gh` missing or unauthenticated | `gh auth login` |
| 30 | a rebase is paused, or rebase state/resolution is not what the step expects | finish with `--continue`, or `--abort` |
| 50 | round did not converge within `--max-iterations` | read `escalation_round_N.md` |
| 51 | cluster step failed before the gate could be judged | check the job log |
| 130 | operator declined the confirmation | nothing changed |

## Typical invocation

```bash
CFG=llm/auto_code_gen/configs/code_gen_config_example.json

bash llm/auto_code_gen/pr_flow/scripts/pr_flow_step_0_prepare.sh --config $CFG \
     --repo-dir /home/<you>/code/vllm_deepseek \
     --fork-url git@github.com:<you>/vllm.git --fork-owner <you>
bash llm/auto_code_gen/pr_flow/scripts/pr_flow_step_1_commit.sh --config $CFG

# optional rebase -- start, resolve, continue, re-prove
bash llm/auto_code_gen/pr_flow/scripts/pr_flow_step_r1_rebase.sh --config $CFG --fetch
python -m llm.auto_code_gen.pr_flow.run_rebase_conflict_fix --config $CFG
bash llm/auto_code_gen/pr_flow/scripts/pr_flow_step_r1_rebase.sh --config $CFG --continue
python -m llm.auto_code_gen.pr_flow.run_rebase_runtime_iters --config $CFG

bash llm/auto_code_gen/pr_flow/scripts/pr_flow_step_2_push.sh --config $CFG \
     --remote-branch perf/example-low-latency-gemm --unique-suffix

# per review round
bash llm/auto_code_gen/pr_flow/scripts/pr_flow_step_3_fetch_reviews.sh --config $CFG
python -m llm.auto_code_gen.pr_flow.run_pr_review_iters --config $CFG --round 1
bash llm/auto_code_gen/pr_flow/scripts/pr_flow_step_1_commit.sh --config $CFG \
     --message-file <output_dir>/runtime/V1/followup_commit_msg.txt
bash llm/auto_code_gen/pr_flow/scripts/pr_flow_step_4_update_pr.sh --config $CFG --reply-threads

# mid-review rebase, published under a lease (step R1 first, then step 4)
bash llm/auto_code_gen/pr_flow/scripts/pr_flow_step_4_update_pr.sh --config $CFG \
     --force-with-lease --reply-threads
```

Every script accepts `--dry-run`, which prints the commands and exits without
running them or asking for confirmation.

`run_pr_review_iters --dry-run` goes further and runs nothing that mutates
anything: no cluster jobs, and no cleanup of leftover cluster jobs/pods.

Round one is a single commit (`--squash-into-base` keeps it that way); later
rounds add a follow-up commit so reviewers can see exactly what their feedback
changed, and vLLM squash-merges anyway.

## Fail-safe details

- The runner refuses to start if `pr_state.repo_dir` and the config's
  `source_code_dir` name different clones -- fixes would land in the wrong tree.
- The agent runs with `GIT_SSH_COMMAND=/bin/false`, `GIT_TERMINAL_PROMPT=0` and
  no `GH_TOKEN`/`GITHUB_TOKEN`; `--allow-git-network` is the only override.
- Files edited outside the original patch scope are reverted by the runner, so a
  review fix cannot wander into unrelated code.
- Step 4 requires `verify_verdict.json` for the exact HEAD being pushed, and a
  commit-only push needs no `gh` token at all.
- A rebase is tagged before it starts (`pr-flow/pre-rebase-<short>`), so the old
  tip survives an `--abort` and can always be recovered by hand.
- The conflict resolver never runs git: while it works the repository stays
  paused mid-rebase, so a bad edit is a re-edit, never a lost commit.
- `--continue` refuses to run while conflict markers remain, and refuses to guess
  when git pauses with nothing unmerged -- it prints the state and hands the
  `--skip` / `--abort` choice back to the operator.
- `--dry-run` never mutates anything: no git, no `gh`, and no deletion of
  leftover cluster jobs/pods.

## Cost of a round

Comment fetch is seconds. A fix iteration costs roughly 45-60 min: fix + sync
(~10 min), 3x150-request benchmark ON (~18 min), lm_eval gsm8k (~14 min).
Add ~30 min when NSYS runs (perf regression only) and ~32 min when the shared
OFF baselines must be re-measured, which happens only after a rebase changes
the base commit -- `pr_state.baselines.*_base_commit` is what detects that, and
`run_pr_review_iters --adopt-existing-baselines` re-stamps them once re-measured.

A rebase therefore costs the conflict resolution (minutes, unless the upstream
refactor is deep) plus an R2 iteration: ~10-15 min repair + ~32 min of OFF
baselines + ~18 min bench ON + ~14 min lm_eval, so ~75-85 min when it passes first
time, ~115 min when NSYS has to run, and ~45-60 min for each extra fix iteration.
That is the price of proving the win still exists on the new base; skipping it is
how a rebased PR gets merged with a silently different measurement.

## Prerequisites

- `gh` installed **and authenticated** (`gh auth login`, git protocol `ssh`).
  Read-only guardrails refuse early, but step 2/3/4 need it.
- A real fork remote (`fork`) that is not the upstream repo and not a local
  path. `check_remotes` refuses `file://` and absolute-path remotes.
- A branch whose name is protected-free, prefix-conforming (`perf/` by
  default), distinct from the local branch, and unused on the fork.
- `git` on the operator's PATH; the SSH key/passphrase prompt is expected to be
  answered interactively -- this is by design.
