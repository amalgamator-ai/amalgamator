# Cluster the commit series, plan once per cluster

Supersedes the "skip Phase 0.5 entirely" shortcut added in `0f53cd9`.

## Why

Two series runs on expat 2.2.5 / RHEL-114618 produced the evidence.

| Run | Configuration | Result |
|---|---|---|
| A (stopped) | triage on, prerequisites skipped, reviews off | ~18 min, stopped at commit 9/19 |
| B (completed) | triage **skipped** | 38m52s, $9.98, 4 defects, Phase 4b rescue |

Run B's cost split:

```
Phase 2+3a  13 resolver sessions   922s  $4.05
Phase 4b    1 rescue session      1260s  $5.13
```

Three findings drive this design.

**1. Per-commit granularity overcharges enabler commits.** Commits 1–7 of the
series are one mechanical transformation (`MALLOC(ms, s)` → `MALLOC(parser, s)`)
applied to different functions: 55 insertions, 60 deletions, one file. They cost
336s of agent time — 14% of the run for 1.5% of the code.

**2. Decisions made per commit are inconsistent across commits.** Each resolver
saw one commit and no shared plan. Commit 9 copied upstream's `#if XML_GE == 1`
verbatim into a target that never defines `XML_GE`, leaving the fix compiled out;
a later phase commented out the `libexpat.def` exports. One plan, applied to
every commit in the group, would have prevented both.

**3. The semantic triage already produced the right plan.** Run A's
`semantic_triage.json`, written by `phase_0_5_semantic_triage`, said unprompted:

> the target has no XML_GE (use XML_DTD) … uses lib/libexpat.def instead of
> libexpat.def.cmake (add the two new exports with fresh ordinals after @69)

Both defects of run B are named there. The knowledge existed; it was neither
binding nor scoped to the work.

## Decision

Split the series into **clusters**, and make the cluster the unit of planning,
resolution, and static verification.

- **Clustering is proposed by the semantic triage**, because boundaries are
  semantic: only a model reading the whole PR knows that commits 1–7 exist only
  to enable commit 9. Commit subjects and directory names are not used.
- **Sizes are measured by git**, never estimated by the model. The model is
  never asked "how many lines is this cluster".
- **`fix.commits` stays a flat ordered list.** Clusters are derived; the YAML
  does not author them (an optional pinned form exists for reruns).
- **Plans are per cluster and generated lazily**, against the tree as it is when
  the cluster starts.
- **Plans are binding on the resolver; their static `verify` checks are
  advisory evidence**, not gates. Build and test remain the gates.

Division of labour:

| Question | Answer comes from |
|---|---|
| Where are the boundaries? | triage model |
| How big is each cluster? | `git diff` net numstat |
| Must this cluster be split? | code, from measured size and caps |
| What must this cluster do? | lazy per-cluster plan |
| Did it do it? | plan `verify` commands |

## What triage must receive

Today `phase_0_5_semantic_triage` passes only a diffstat for merges
(`run_bug_fix.py:899-927`), and `SemanticTriagePrompt` has no series input
(`bug_fix_prompts.py:44-49`). A model cannot propose commit boundaries from a
stat line.

In series mode the prompt gains:

- the ordered `fix.commits` list with short SHA, subject, and `--shortstat`;
- the instruction that these commits are applied in this order;
- the existing target-branch exploration tools.

`FusedPreRedPrompt` (`bug_fix_prompts.py:876-898`) receives the same inputs and
the same two output fields when `fused_pre_red` is enabled, so the two paths
cannot drift.

## Data model

Triage output gains three fields; everything else in `semantic_triage.json` is
unchanged.

```json
{
  "clusters": [
    {
      "id": "enablers",
      "commits": ["0872c189", "8768dada", "4fc6f1ee", "51487ad9",
                  "b3f0bda5", "53a3eda0", "4e7a5d03"],
      "intent": "convert allocation call sites to the MALLOC/REALLOC/FREE macros",
      "kind": "enabler",
      "expected_conflict_files": ["expat/lib/xmlparse.c"]
    }
  ],
  "dropped": [
    {"commit": "20b4c2ee", "reason": "unrelated CI change (mass-cppcheck.sh)"}
  ],
  "public_interface_files": ["expat/lib/expat.h", "expat/lib/libexpat.def"]
}
```

- `commits` are short or full SHAs. They are matched by prefix against
  `config.source_fix_commits` only. Anything unmatched invalidates the partition.
- `public_interface_files` feeds the mechanical/semantic decision and the export
  sanity check. It is a triage judgement, not a code heuristic.

### The per-cluster plan (lazy)

Written to `tracker.output_dir/cluster_plan_<cluster-id>.json`, never inside the
repository — numbered files at the repo root would otherwise be swept into the
candidate patch by `_pre_cherry_pick_cleanup` and `git add -A`.

```json
{
  "cluster": "tracker",
  "actions": [
    {
      "symbol": "XML_GE",
      "target_state": "undefined",
      "action": "synthesize",
      "from": "XML_DTD",
      "verify": "grep -q 'define XML_GE' expat/lib/internal.h"
    },
    {
      "symbol": "XML_SetAllocTrackerMaximumAmplification",
      "target_state": "new public API",
      "action": "export",
      "where": "expat/lib/libexpat.def",
      "note": "uncommented, ordinal after @69",
      "verify": "grep -v '^;' expat/lib/libexpat.def | grep -q AllocTracker"
    }
  ],
  "risks": ["parser struct now carries a size_t prefix"]
}
```

`action` ∈ `translate | synthesize | export | adapt_call_site | skip`.
`port` is deliberately absent: prerequisites are never ported.

Semantics:

| action | effect at execution |
|---|---|
| `translate`, `synthesize`, `export`, `adapt_call_site` | binding instruction in the resolver prompt |
| `skip` | path is added to `files_to_skip` for this cluster |

Only `verify` is executable. Every action must carry one; an action without a
`verify` is rejected at load.

Validation at load (this is new code, there is no plan loader today):

1. JSON parses and `cluster` matches the cluster being planned.
2. Every action has a non-empty `verify`.
3. Every `verify` passes `_validate_shell_command` (`run_bug_fix.py:458-498`) —
   triage-supplied commands are already filtered for this reason
   (`run_bug_fix.py:7620-7624`, `7711-7721`).
4. Commands run with `cwd=config.repo_path` (the examples are repo-relative) and
   `config.command_timeout`.
5. Each `verify` command inspects exactly one path. A multi-file `grep -L`
   prints only the files that lack the symbol, so it cannot prove that every
   path is clean.

A rejected plan is regenerated once with the rejection reason; a second failure
falls back to per-commit mode for that cluster.

## Validation and sizing

The model's partition is a proposal; code enforces the contract.

### Partition invariants

1. Clusters are **contiguous** in the original order and do not overlap.
2. Every non-dropped commit appears in exactly one cluster.
3. Every SHA resolves to an entry of `config.source_fix_commits`.
4. No cluster is empty.
5. Merge commits are not accepted inside a multi-commit cluster: `git_cherry_pick`
   applies merges with `-m1` (`git_tools.py:184-185`) while `git diff c1^ cn`
   measures first-parent history, so measurement and application would disagree.
   A merge commit is always its own single-commit cluster.
6. Any violation → discard the partition, fall back to one cluster per commit,
   and record a `clusters` gate with the reason.

### Dropping commits

`dropped` is a proposal, applied only under a deterministic condition: the
pipeline drops a commit only when **every path it touches** matches the skip
policy. Otherwise the commit is kept and moved into its own cluster.

The skip policy is now triage's `files_to_skip` only, and that list is
restricted to auto-generated build artifacts. The former deterministic path set
(`.github/`, `doc/`, `fuzz/`, release-note files) was removed: path shape is not
a reliable signal. A generator script is source, not generated output, and API
documentation is part of a public-API fix. Docs, CI, fuzz and release notes are
ported with the fix. With no triage list, no file is skipped by path, so a commit
is never dropped for path shape alone.

The same generated-only rule applies to single-commit and fused pre-RED triage;
series mode only changes how the skip list is applied.

This is stricter than it sounds, because `files_to_skip` is currently applied
only on the conflict path — a clean cherry-pick returns before the revert block
(`run_bug_fix.py:4306-4310` vs `4354-4368`). Two fixes are prerequisites for this
design:

- skipped paths must be reverted/removed on the clean path too;
- the revert must use `_matches_any` (glob/prefix) the way triage's skip set is
  interpreted elsewhere (`run_bug_fix.py:1219-1246`), not exact string equality
  (`4358`).

Dropped commits are recorded in the dossier with their reason.

### Measurement

Cluster size is the **net** diff between the parent of the first commit and the
last commit: `git diff --numstat <first>^ <last>`, summed. It is not the sum of
per-commit diffs — commits inside a cluster touch the same lines, so summing
overstates.

"Changed lines" means insertions + deletions from that numstat. This is the
number the caps and the mode decision use.

Measured on expat:

| Cluster | Commits | Insertions | Deletions | Changed | Files |
|---|---|---|---|---|---|
| enablers | 1–7 | 55 | 60 | **115** | 1 |
| tracker core | 9–13 | 408 | 24 | **432** | 9 |
| — commit 9 alone | 9 | 363 | 12 | 375 | 9 |
| — refinements | 10–13 | 45 | 12 | **57** | 1 |
| tests + xmlwf | 14–15 | 256 | 10 | **266** | 6 |

### Conflict prediction

Size is only a proxy for cost; sessions are spent per **conflicted** commit. For
each commit, `git merge-tree --write-tree --merge-base=<commit>^ <base> <commit>`
reproduces cherry-pick's 3-way merge without touching the worktree and lists the
conflicted files. Predicting every commit against a fixed base overestimates
later commits in a series (their predecessors have not landed), which is
acceptable for grouping and never underestimates.

`batch_groups()` then joins contiguous commits whose predicted conflict set is
**identical**, so batching only ever merges repetitions of one conflict shape.
A clean commit ends a run, because it costs no session anyway.

Measured on the expat series against `abf81a11`:

| Commit | Predicted conflict files |
|---|---|
| 1–7 (enablers) | `xmlparse.c` each |
| 8 (cppcheck) | `.travis.sh` |
| 9 (tracker core) | 9 files |
| 10, 11 | `xmlparse.c` each |
| 12 | none |
| 13 | `xmlparse.c` |
| 14 (tests) | 3 files |
| 15 (xmlwf) | 3 files |
| 17–19 (docs) | 1 file each |

Batches: enablers `[1..7]`, tracker `[9]` `[10,11]` `[13]`, tests `[14]` `[15]` —
13 resolution sessions become 6.

### Caps and splitting

```
max_net_lines   300
max_files         8
max_commits       8
```

Caps govern whether to **merge** commits into a cluster. They never split a
single commit — a commit is atomic, so commit 9 (375 lines, 9 files) stays whole
even though it alone exceeds `max_net_lines`. `max_commits` bounds the blast
radius of a failed cluster session, not the cost.

Split procedure, applied until every cluster fits or is a single commit:

1. Measure the cluster.
2. If within all caps, accept.
3. Otherwise peel off the largest single commit into its own cluster and
   re-measure the remainder.
4. Record the split in the dossier.

Applying this to the table above: the proposed 9–13 cluster (432 lines) peels
off commit 9, leaving 10–13 at 57 lines. Final partition: **4 clusters**
(enablers, commit 9, refinements, tests+xmlwf) from 19 commits.

### Mode

```
mechanical  changed <= 120 AND files == 1 AND touches no public_interface_files
semantic    otherwise
```

`mode` decides only whether a plan session is spent: a mechanical cluster's plan
is a short instruction plus verify commands. A cluster that resolves through the
existing mechanical rungs with no session at all is the ideal outcome.

Note that clean cherry-picks and mechanically-resolved conflicts already run no
session (`run_bug_fix.py:4306-4310`, `4451-4453`, `4489-4509`); `mode` does not
add a path, it only decides whether to plan.

## Execution

For each cluster, in order:

1. **Plan** — generate `cluster_plan_<id>.json` against the current tree, passing
   prior decisions from the dossier. Validate per the rules above.
2. **Apply** — cherry-pick the cluster's commits one at a time, preserving order.
   The existing strategy ladder and mechanical rungs run first.
3. **Resolve** — each conflicted commit is resolved in its own session
   (one session per conflicted commit, not one session spanning the cluster),
   with the cluster plan in context. Session-spanning is explicitly out of scope
   for v1: `_run_fused_resolution` assumes a single commit, a single `base_head`,
   and one resulting commit (`run_bug_fix.py:4668-4742`), and a cluster-wide
   session would have to bypass the strategy ladder and mechanical rungs.
4. **Close** — record the cluster's commits, mode, splits, and verify results as
   tracker gates and dossier entries.
5. **Verify** — run the plan's `verify` commands, then the configured **build**
   command. Tests are not run here: series mode defers the first test run to
   Phase 4 (commit `c940577`), because the test port lands before the series and
   may not compile until later commits arrive. A cluster verify failure is
   recorded as `cluster_<id>=advisory` and does not block the next cluster.

Repair commits created inside a cluster are left in place. Squashing them is
deferred: a post-hoc rebase invalidates the commit ranges already checked by
`_strip_committed_build_output` and `_agent_commit_guard`
(`run_bug_fix.py:2991-3054`).

## Failure handling

| Failure | Action |
|---|---|
| Partition invalid | one cluster per commit, record gate, continue |
| Plan unparseable or missing `verify` | regenerate once, then per-commit mode for that cluster |
| A commit cannot be applied | raise as today; the per-commit path is not currently a fallback and making it one is a control-flow change, listed as a follow-up |
| `verify` fails | record `cluster_<id>=advisory` with the failing entry and its output; continue. Build/test remain the gates |
| Cluster ends with zero commits | drop the cluster, skip its verify, record it |
| Agent commits during planning | `_agent_commit_guard` strips build output; the plan prompt forbids commits and any commit it makes is reported as a gate |

The per-commit loop is not a universal fallback in the current code: in series
mode a failed cherry-pick raises (`run_bug_fix.py:7632-7643`). Making per-commit
replay a fallback is a separate control-flow change and is not assumed here.
Clustering therefore adds no new correctness dependency: if the partition is
rejected, behaviour is exactly today's.

## Configuration

```yaml
fix:
  commits: [ ... ]        # unchanged, flat, ordered
  clusters: []            # optional; empty = derive from triage

advanced:
  cluster_max_net_lines: 300
  cluster_max_files: 8
  cluster_max_commits: 8
```

Pinned form: a list of lists of SHAs, each inner list one cluster in order. It is
validated by the same invariants and still subject to caps. Pinning exists for
reproducible reruns; the derived partition is recorded in the tracker either way.

## What this changes in the pipeline

1. **Phase 0.5 runs again in series mode.** The skip in `0f53cd9` is reverted
   (`run_bug_fix.py:7355-7363`). The deterministic assessment is computed
   *before* the LLM result and merged under it, so an unparseable triage JSON
   still leaves `test_files` populated. `files_to_skip` is triage-only: the
   deterministic path-based skip set was removed, so without a triage list no
   file is skipped by path shape.
2. **New input and output fields** in `SemanticTriagePrompt` and
   `FusedPreRedPrompt`.
3. **New lazy phase** between Phase 0.5 and Phase 2+3a: cluster planning.
4. **Phase 2+3a becomes a loop over clusters**, with the per-commit loop inside.
5. **New deterministic module** for measurement, validation, splitting, mode.
6. **Per-cluster verify** driven by plan commands plus the build command.

## Implementation order

| Step | Content | LLM needed | Status |
|---|---|---|---|
| 1 | `auto_bug_fix/series_clusters.py`: model, measurement, validation, caps, mode | no | done |
| 2 | Skip-path fixes: prune on the clean path and after resolution, glob matching, delete files the target never had | no | done |
| 3 | Triage prompt: series input, `clusters`, `dropped`, `public_interface_files` | prompt only | done |
| 4 | Merge deterministic assessment under LLM triage; skips are triage-only | no | done |
| 5 | Cluster plan phase, loader, `verify` enforcement | yes at run time | done |
| 6 | Phase 2+3a loop over clusters, gates, per-cluster verify | no | done |

All of the above applies **only when `fix.commits` is set**. Single-commit
porting is untouched: every prune path returns immediately without a series,
and the conflict-path revert keeps its previous exact-match behaviour.

## Test plan

Unit (temporary git repos, no network):

- partition invariants: gaps, overlaps, reordering, unknown SHA, empty cluster,
  merge commit inside a cluster;
- measurement equals `git diff --numstat` over the net range, and differs from
  the sum of per-commit numstat;
- caps split at a commit boundary and never split a single commit;
- mode boundaries, including a cluster that touches `public_interface_files`;
- pinned clusters pass the same validation;
- invalid partition falls back to one cluster per commit and records a gate;
- dropped commit whose paths exceed the skip policy is kept.

Integration (no LLM): feed the real 19-commit expat series into measurement and
splitting, assert 4 clusters with 115 / 375 / 57 / 266 changed lines.

End-to-end (live): rerun expat 2.2.5 against run B.

## Success criteria

| Metric | Run B | Target |
|---|---|---|
| Resolver sessions | 13 | ≤ 5 |
| Phase 2+3a | 922s / $4.05 | ≤ 350s / ≤ $1.50 |
| Phase 4b reached | yes (1260s) | no |
| Total | 2333s / $9.98 | ≤ 1000s / ≤ $3 |
| Defects (inert fix, dead exports, churn) | 4 | 0 |

## Non-goals

- Not preserving a 1:1 commit mapping. Upstream order is preserved; repair
  commits are not squashed in v1.
- Not porting prerequisites (`action: port` is not in the schema).
- Not caching knowledge across cases. A repo profile is out of scope until the
  same facts appear in several cluster plans.
- Not replacing the per-commit path.
- Phase 4.5 semantic equivalence is unchanged: it still compares the whole
  backport against the fix commit's diff. Clustering is invisible to it by
  design.

## Open questions

1. Whether repair-commit squashing is worth the range revalidation it implies.
2. Whether `dropped` should require operator opt-in once a case is tracked. v1
   drops only commits whose every path is in the skip policy.
3. Whether the mechanical mode deserves its own prompt or is the semantic prompt
   with a smaller budget.

## Appendix: corrections from review

| Original claim | Correction |
|---|---|
| Run A's triage JSON came from fused pre-RED | Wrong. The run log contains one `Semantic Triage Agent` prompt and no fused pre-RED; `fused_pre_red` defaults to false and was unset in the case. |
| Triage output "gains one field" | It also needs the series list as input, which it never received. |
| Dropped commits may only touch `files_to_skip` | `files_to_skip` is only applied on the conflict path today; the clean path commits skipped files. Two skip-path bugs must be fixed first. |
| Per-commit loop is the universal fallback | It is a hard stop in series mode; making it a fallback is a control-flow change, now out of scope. |
| Plan entries rejected "by the loader" | No loader exists; validation and `_validate_shell_command` filtering are new code. |
| Example `verify` for `translate` | Was inverted. Replaced with a `synthesize` action whose verify matches the shim. |
| 56 lines for commits 1–8 | 55/60 for 1–7; 56/60 for 1–8 including the unrelated CI commit. |
| One session spanning a cluster | Removed. One session per conflicted commit; spanning would bypass the strategy ladder and mechanical rungs. |
| Per-cluster build+test gate | Tests stay deferred to Phase 4. Cluster gate is plan `verify` + build only. |
| Journal as a new mechanism | Reuses the existing `Dossier` and tracker gates. |
| Plans written as repo files | Would leak into the patch via `_pre_cherry_pick_cleanup`; they go to `tracker.output_dir`. |
