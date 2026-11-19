# Preserve the upstream fix as a commit series

## Why

Amalgamator currently takes **one commit** from `fix.commit` and cherry-picks it.
For a pull/merge request the adapter often supplies the **merge commit**, and
`git_cherry_pick()` applies it with `-m 1`. That turns the whole PR into one
combined diff.

On expat 2.5.0 this worked and produced the same tree as Ymir. On expat 2.2.5
the combined diff could not be mapped by any cherry-pick strategy:

```
PipelineEscalation: main commit (RHEL-114618): unmappable cherry-pick (all strategies failed)
```

Amalgamator then fell back to Phase 4b, which is a generic build-test-fix loop.
The Ymir backport skill, by contrast, applied the PR's 19 commits one at a time
onto the same prepared base and finished in 17 minutes.

The difference is therefore **input granularity**, not “squashing” the final
patch. A single merge-commit diff is harder to map than 19 small commits whose
conflicts are resolved against the progressively updated tree.

## Proposal

Add an optional `fix.commits` list:

```yaml
fix:
  commit: "ac7a27e1..."   # merge/topic head, used for analysis/diff
  commits:                # optional; applied in order
    - "0872c189..."
    - "8768dada..."
    - ...
```

- `fix.commit` stays the analysis anchor for triage, test port, and combined
  diff prompts.
- `fix.commits`, when present, is the ordered list Amalgamator actually
  cherry-picks.
- If absent, behavior is unchanged.

## Implementation sketch

1. `BugFixConfig`
   - add `source_fix_commits: list[str] = []`.

2. `config_loader`
   - accept `fix.commits` (non-empty list of strings).
   - mutually exclusive with `fix.patch`.
   - `fix.commit` is still required when `fix.commits` is present, because the
     triage/test-port phases need a single diff anchor.
   - validate each entry is a full/short SHA present in the repo.

3. Phase 2+3a
   - replace the single `cherry_pick_and_resolve(... commit=config.source_fix_commit)`
     call with a loop over `config.source_fix_commits or [config.source_fix_commit]`.
   - label each attempt `commit i/N (sha)`.
   - record each result in the dossier/tracker; `cherry_pick_path` becomes
     `clean` only if every commit is clean, otherwise `resolved`.
   - preserve `files_to_skip` handling per commit.

4. Prerequisite chain
   - skip Phase 0.5b in series mode. The ordered upstream series is the
     source of truth; porting triage-inferred prerequisites first can pull in
     a large feature the series does not need (e.g. billion-laughs accounting
     on expat 2.2.5) and changes the order the source branch presents.

   Also skip Phase 0.5 (semantic triage) itself. On the 2.2.5 series it cost
   315s of 786s agent time, most of it tracing prerequisite commits that
   Phase 0.5b then discards. Its remaining products are all derivable from
   git, so series mode builds the assessment deterministically:

   - `files_to_skip` — paths whose components are bookkeeping only
     (`.github/`, `doc/`, `docs/`, `fuzz/`, `fuzzing/`) plus release notes
     (`Changes`, `CHANGELOG`, `NEWS`, `HISTORY`); on the expat series this
     recovers 8 of the 12 files triage skipped and never skips source or
     tests.
   - `test_files` — `identify_test_files(seed)`; `_planned_test_files`
     unions this with the same function anyway.
   - `priority_files` — the allowlist, which is what triage's value falls
     back to for files outside the fix.
   - `prerequisites` — empty, since none are ported.

   Lost: `lint_commands`/`verification_commands` discovery (so a case must
   set `lint_command` itself if it wants lint), and the difficulty and
   recommendation fields, which are only recorded, never gating.

5. Failure paths
   - `_reapply_force_resolve` and Phase 4b currently re-apply
     `source_fix_commit`. In series mode they must resume from the first
     unfinished commit, not replay the merge commit. Initially, fail the run
     with a clear escalation if a series commit cannot be mapped; do not
     silently fall back to the merge commit.

6. Patch export
   - `git format-patch base..HEAD` already preserves commits. With the loop,
     `candidate.patch` will contain the per-commit series instead of one merge
     commit.
   - Phase 0.5c (test port) still creates its own commit before the series.
     For strict series parity, add a mode that folds the test port into the
     relevant commit or skips the separate test-port phase.
   - Phase 3a.1/3a.2 review corrections may add a fixup commit. Preserve it,
     or amend it into the last series commit after review.

## Experiment

On expat 2.2.5 / RHEL-114618, base `abf81a11`:

- `fix.commit`: `ac7a27e1...`
- `fix.commits`: the 19 PR #1034 commits in order
- Opus 5.5, `effort: high`
- same autotools build/test commands

Success criteria:

1. no Phase 4b escalation caused by the merge-commit mapping;
2. final tree comparable to Ymir's 14-commit result;
3. `candidate.patch` contains the individual ports, not a merge commit;
4. build/test and the attack reproducer pass;
5. wall time and cost recorded against the previous 2.2.5 attempt.

## Non-goals

- This does not make the final patch identical to Ymir's; conflict resolution
  and test porting still differ.
- This does not remove Amalgamator's test-port/post-resolution/blind review
  phases. Those remain part of the cost and may still be useful on harder
  cases. Only the semantic triage pass is replaced, and only in series mode.
- This does not replace the need for a dynamic ASan/oracle review.
