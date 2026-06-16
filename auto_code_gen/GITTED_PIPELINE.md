# Gitted Bug Fix Pipeline — Git-First Backporting with LLM Fallback

A companion pipeline for backporting commits across branches using git cherry-pick as the primary mechanism, with LLM-powered conflict resolution, semantic triage, and automatic escalation to the main `auto_code_gen` pipeline when fixes require deeper iteration.

**Location**: `auto_bug_fix/`  
**Entry point**: `python -m auto_bug_fix.run_bug_fix <config.yaml>`

---

## How It Relates to auto_code_gen

The main `auto_code_gen` bug fix pipeline (Phase 1–5: code trace → plan → code gen → runtime) starts from scratch — it analyzes the fix, plans the port, generates code, then iterates. This works well for C/systems projects where the fix must be substantially rewritten.

The **gitted pipeline** takes a different approach: **try `git cherry-pick` first**, resolve conflicts with an LLM, and only fall back to the full LLM pipeline when that fails. This is faster and cheaper for commits that cherry-pick cleanly or with minor conflicts, which empirically covers ~70% of backports.

**When the gitted pipeline's Phase 4a can't fix remaining build/test failures, it escalates to `auto_code_gen`'s `RunAndFixPrompt`** — the autonomous build-test-fix loop — passing ALL accumulated context (dossier, triage, prerequisite analysis) so the LLM has the full picture.

---

## Pipeline Phases

```
Phase 0    ─ Deterministic triage (patch-ID, ancestry, seed files)
Phase 0.5  ─ Semantic triage (LLM: difficulty, prerequisites, generated files,
              regeneration commands, verification commands, lint commands)
Phase 0.5b ─ Port prerequisite commits (only if fix NEEDS the prerequisite's
              behavior — call-convention-only changes are adapted at call site)
Phase 0.5c ─ Port test files from fix commit (filtered by files_to_skip)
Phase 1    ─ Comprehensive baseline on clean target branch:
              ├─ Build command
              ├─ Test command (+ RED check for bugfixes)
              └─ All lint/verification commands from triage
Phase 1.5  ─ Failure-mode confirmation (signature comparison)
Phase 2+3a ─ Cherry-pick + conflict resolution
              ├─ Revert files_to_skip to target version (both conflicting
              │   and cleanly-applied)
              └─ LLM resolves remaining source conflicts
Phase 3b   ─ Regeneration (run project codegen commands, rollback on failure)
Phase 4    ─ Verify (build + test from YAML config)
Phase 4a   ─ Build error recovery (LLM single-shot fix)
Phase 4.1  ─ Extended verification (commands discovered by Phase 0.5)
              ├─ Filters pre-existing failures using Phase 1 baseline
              └─ Only escalates for NEW failures introduced by the backport
Phase 4b   ─ Handoff to auto_code_gen RunAndFixPrompt (autonomous loop)
              ├─ Saves phase_4b_input.json (full context)
              ├─ Saves phase_4b_log.txt (conversation log)
              └─ Copies run_and_fix_success/failure.txt to run directory
Phase 4.5  ─ Semantic equivalence (range-diff in dossier)
```

---

## Detailed Flowchart

```
                            ┌─────────────────────┐
                            │   YAML Config File   │
                            │  (issue, repo, fix,  │
                            │   branches, build)   │
                            └─────────┬───────────┘
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │  Create Git Worktree    │
                         │  on target branch       │
                         └────────────┬───────────┘
                                      │
          ╔═══════════════════════════════════════════════════════╗
          ║                  PHASE 0 — TRIAGE                    ║
          ╚═══════════════════════════════════════════════════════╝
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │  Derive seed files      │
                         │  from fix commit         │
                         │  (git diff ^1..merge     │
                         │   for merge commits)     │
                         └────────────┬───────────┘
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │  Forward patch-ID check │◄── Is fix already on target?
                         └────────────┬───────────┘
                              │              │
                           found          not found
                              │              │
                              ▼              ▼
                         ┌────────┐   ┌──────────────────┐
                         │  STOP  │   │ Backward patch-ID │
                         │ (dup)  │   │ + ancestry check  │
                         └────────┘   └────────┬─────────┘
                                               │
                                        ┌──────┴──────┐
                                    affected      not affected
                                        │              │
                                        │              ▼
                                        │     ┌─────────────────┐
                                        │     │ Seed files exist │
                                        │     │  on target?      │
                                        │     └───────┬─────────┘
                                        │        yes  │   no
                                        │         │   │    │
                                        │         │   │    ▼
                                        │         │   │  ┌──────┐
                                        │         │   │  │ STOP │
                                        │         │   │  └──────┘
                                        ▼         ▼   │
                         ┌────────────────────────┐   │
                         │  Resolve renames        │◄──┘
                         │  (fork point → target)  │
                         │  Build allowlist         │
                         └────────────┬───────────┘
                                      │
          ╔═══════════════════════════════════════════════════════╗
          ║             PHASE 0.5 — SEMANTIC TRIAGE              ║
          ║                    (LLM Agent)                        ║
          ╚═══════════════════════════════════════════════════════╝
                                      │
                                      ▼
                         ┌────────────────────────────────┐
                         │  LLM analyzes fix commit:       │
                         │  • commit_type (bugfix/feature)  │
                         │  • new packages, types           │
                         │  • dependency depth              │
                         │  • generated/binary files        │
                         │  • files_to_skip                 │
                         │  • estimated difficulty           │
                         │  • prerequisite commits          │
                         │    (portable = fix NEEDS the     │
                         │     behavior, not just calling   │
                         │     convention)                   │
                         │  • regeneration commands          │
                         │  • lint commands                  │
                         │  • verification commands          │
                         │  • priority files                 │
                         └────────────┬───────────────────┘
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │  Writes assessment to   │
                         │  semantic_triage.json    │
                         └────────────┬───────────┘
                                      │
                              ┌───────┴────────┐
                      portable               no portable
                    prerequisites?          prerequisites
                              │                     │
                              ▼                     │
          ╔═══════════════════════════╗              │
          ║  PHASE 0.5b — PREREQ     ║              │
          ║  CHAIN PORTING           ║              │
          ╚═══════════════════════════╝              │
                              │                     │
                              ▼                     │
                   ┌──────────────────────┐         │
                   │  For each prerequisite│         │
                   │  marked portable=true │         │
                   │  (max 5 depth):       │         │
                   │  cherry_pick_and_     │         │
                   │  resolve(prereq_sha)  │         │
                   └──────────┬───────────┘         │
                              │                     │
                   NOTE: portable=false means       │
                   adapt at call site instead.       │
                   Example: drop extra arg,          │
                   change import path.               │
                              │                     │
                              ◄─────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Test files in     │
                    │ fix commit?       │
                    │ (filtered by      │
                    │  files_to_skip)   │
                    └────────┬─────────┘
                        yes  │    no
                         │   │     │
                         ▼   │     │
          ╔══════════════════════╗  │
          ║  PHASE 0.5c — TEST  ║  │
          ║  PORT (LLM Agent)   ║  │
          ╚══════════════════════╝  │
                         │         │
                         ▼         │
              ┌────────────────┐   │
              │ LLM ports test │   │
              │ files to target│   │
              │ branch (commit)│   │
              └──────┬─────────┘   │
                     │             │
                     ◄─────────────┘
                     │
          ╔═══════════════════════════════════════════════════════╗
          ║   PHASE 1 — COMPREHENSIVE BASELINE + RED CHECK       ║
          ╚═══════════════════════════════════════════════════════╝
                     │
                     ▼
              ┌──────────────────────┐
              │ 1. Run build command │
              │    (capture exit +   │
              │     output)          │
              │ 2. Run test command  │
              │    (capture baseline │
              │     failures)        │
              │ 3. Run ALL lint +    │
              │    verification cmds │
              │    from triage       │
              │    (capture each     │
              │     exit + output)   │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Store baseline:       │
              │ • baseline_build_exit │
              │ • baseline (test set) │
              │ • baseline_           │
              │   verification[]     │
              │ Record in dossier    │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ RED check (bugfix    │
              │ + ported tests):     │
              │ Tests pass → STOP   │
              │ (already fixed)      │
              └──────────┬───────────┘
                         │
          ╔═══════════════════════════════════════════════════════╗
          ║    PHASE 1.5 — FAILURE-MODE CONFIRMATION             ║
          ╚═══════════════════════════════════════════════════════╝
                │
                ▼
         ┌──────────────┐
         │  Signature   │  (currently skips — needs parent
         │  comparison  │   signature source to be wired)
         └──────┬───────┘
                │
          ╔═══════════════════════════════════════════════════════╗
          ║   PHASE 2+3a — CHERRY-PICK + CONFLICT RESOLUTION     ║
          ╚═══════════════════════════════════════════════════════╝
                │
                ▼
         ┌──────────────────────┐
         │  Try 3 strategies:   │
         │  1. default           │
         │  2. patience          │
         │  3. ort               │
         │  Pick fewest conflicts│
         └──────────┬───────────┘
                    │
             ┌──────┴────────┐
           clean         conflicts
             │               │
             ▼               ▼
       ┌──────────┐  ┌──────────────────────┐
       │  Done!    │  │ Re-apply best strat   │
       │  Commit   │  │ (keep conflicts open) │
       └──────────┘  └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │  files_to_skip        │
                     │  processing:          │
                     │  1. Revert cleanly-   │
                     │     applied skip files│
                     │     to target version │
                     │  2. Auto-resolve      │
                     │     conflicting skip  │
                     │     files (--theirs)  │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │  Source conflicts     │
                     │  remain?              │
                     └──────────┬───────────┘
                           no   │   yes
                            │   │    │
                            ▼   │    ▼
                     ┌────────┐ │ ┌──────────────────────┐
                     │ Commit │ │ │  LLM Resolution Agent │
                     └────────┘ │ │  (NarrowResolution     │
                                │ │   AgentPrompt)         │
                                │ └──────────┬────────────┘
                                │            │
                                │    ┌───────┴────────┐
                                │  resolved      still conflicts
                                │    │               │
                                │    ▼               ▼
                                │ ┌────────┐  ┌──────────────┐
                                │ │ Commit │  │ Retry (up to │
                                │ │ with   │  │ max_retries) │
                                │ │summary │  └──────┬───────┘
                                │ └────────┘         │
                                │              exhausted
                                │                │
                                │                ▼
                                │          ┌───────────┐
                                │          │  ESCALATE  │
                                │          └───────────┘
                                │
          ╔═══════════════════════════════════════════════════════╗
          ║  PHASE 3b — REGENERATION (conditional)               ║
          ╚═══════════════════════════════════════════════════════╝
                                │
                     ┌──────────┴───────────┐
                  regen cmds           no regen cmds
                  + skipped files         │
                     │                    │
                     ▼                    │
              ┌──────────────────┐        │
              │ Save safe point  │        │
              │ (git rev-parse)  │        │
              └──────┬───────────┘        │
                     │                    │
                     ▼                    │
              ┌──────────────────┐        │
              │ For each command:│        │
              │ run codegen      │        │
              │ (e.g. hack/      │        │
              │  update-codegen) │        │
              └──────┬───────────┘        │
                     │                    │
              ┌──────┴──────┐             │
           success       failure          │
              │              │            │
              ▼              ▼            │
        ┌──────────┐  ┌──────────────┐    │
        │  Keep    │  │ Build still  │    │
        │  changes │  │ passes?      │    │
        └──────────┘  └──────┬───────┘    │
                        yes  │   no       │
                         │   │    │       │
                         ▼   │    ▼       │
                   ┌────────┐│┌─────────┐ │
                   │  Keep  │││Rollback │ │
                   │partial │││to safe  │ │
                   └────────┘││point    │ │
                             │└─────────┘ │
                             │            │
              ┌──────────────┴────────┐   │
              │ Commit regenerated    │   │
              │ files (if any changed)│   │
              └───────────┬───────────┘   │
                          │               │
                          ◄───────────────┘
                          │
          ╔═══════════════════════════════════════════════════════╗
          ║            PHASE 4 — VERIFY                          ║
          ╚═══════════════════════════════════════════════════════╝
                          │
                          ▼
               ┌───────────────────┐
               │  Run build_command │
               │  Run test_command  │
               │  Check regression  │
               │  (vs baseline)     │
               └─────────┬─────────┘
                    │           │
                  pass        fail
                    │           │
                    │           ▼
                    │  ╔══════════════════════════════════╗
                    │  ║  PHASE 4a — BUILD ERROR RECOVERY ║
                    │  ║         (LLM Agent)               ║
                    │  ╚══════════════════════════════════╝
                    │           │
                    │           ▼
                    │  ┌────────────────────────┐
                    │  │ LLM diagnoses errors   │
                    │  │ and edits source files  │
                    │  │ (up to max_retries)     │
                    │  └────────────┬───────────┘
                    │         │           │
                    │       fixed     still failing
                    │         │           │
                    │         ▼           ▼
                    │  ┌──────────┐  ┌───────────┐
                    │  │  Re-run  │  │  ESCALATE  │
                    │  │  Phase 4 │  └───────────┘
                    │  │  verify  │
                    │  └──────────┘
                    │
                    ▼
          ╔═══════════════════════════════════════════════════════╗
          ║  PHASE 4.1 — EXTENDED VERIFICATION (conditional)     ║
          ║  Uses Phase 1 baseline to filter pre-existing fails  ║
          ╚═══════════════════════════════════════════════════════╝
                    │
             ┌──────┴──────┐
          commands      no commands
          from 0.5         │
             │             │
             ▼             │
      ┌──────────────────┐ │
      │ Run each         │ │
      │ verification cmd │ │
      │                  │ │
      │ For each result: │ │
      │ • If cmd failed  │ │
      │   in baseline →  │ │
      │   log as         │ │
      │   "pre-existing" │ │
      │   and SKIP       │ │
      │ • If cmd is NEW  │ │
      │   failure →      │ │
      │   escalate       │ │
      └──────┬───────────┘ │
        │          │       │
      all pass   NEW       │
      or pre-    failures  │
      existing     │       │
        │          ▼       │
        │   ┌────────────────────────┐
        │   │  Phase 4a recovery     │
        │   │  (LLM fixes errors)    │
        │   └────────┬───────────────┘
        │       │           │
        │     fixed     still failing
        │       │           │
        │       ▼           ▼
        │ ┌──────────┐  ╔══════════════════════════╗
        │ │ Re-run   │  ║  PHASE 4b — EXTERNAL     ║
        │ │ ext.     │  ║  PIPELINE HANDOFF         ║
        │ │ verify   │  ║  (Opus + deep thinking)   ║
        │ └──────────┘  ╚═════════════╤════════════╝
        │                             │
        │                             ▼
        │                 ┌────────────────────────┐
        │                 │ Saves to run directory: │
        │                 │ • phase_4b_input.json   │
        │                 │   (full context)        │
        │                 │ • phase_4b_log.txt      │
        │                 │   (conversation log)    │
        │                 │ • run_and_fix_*.txt     │
        │                 │   (success/failure)     │
        │                 │                         │
        │                 │ auto_code_gen's          │
        │                 │ RunAndFixPrompt          │
        │                 │ (autonomous build-test-  │
        │                 │  fix loop with full      │
        │                 │  dossier context)        │
        │                 └────────────┬───────────┘
        │                              │
        ◄──────────────────────────────┘
        │
          ╔═══════════════════════════════════════════════════════╗
          ║      PHASE 4.5 — SEMANTIC EQUIVALENCE                ║
          ╚═══════════════════════════════════════════════════════╝
        │
        ▼
 ┌──────────────────────┐
 │  git range-diff       │
 │  (upstream vs port)   │
 │  → identical /        │
 │    modified /          │
 │    unmatched           │
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐
 │  PHASE 5 — OUTPUTS   │
 │                       │
 │  • dossier.md         │
 │  • tracker.json       │
 │  • run_log.txt        │
 │  • phase_4b_input.json│
 │  • phase_4b_log.txt   │
 │  • semantic_triage.json│
 │  • git format-patch   │
 │  • commit trailers    │
 └──────────────────────┘
```

---

## Integration Point: Phase 4b

When the gitted pipeline exhausts its own recovery (Phase 4a), it imports `auto_code_gen`'s `RunAndFixPrompt` and calls it with enriched context:

```python
from auto_code_gen.use_cases.bug_fix import gen_RunAndFixPrompt
from common.claude_utils import claude_run, PipelineStep, ClaudeConfig

prompt = gen_RunAndFixPrompt(
    context=enriched_context,  # includes dossier + triage + repo info
    build_command=config.build_command,
    test_command=config.test_command,
    build_dir=config.build_dir,
    max_build_test_retries=config.max_build_test_retries,
)

steps = [PipelineStep(name="phase_4b_external_fix", prompt=prompt.prompt())]
timings = await claude_run(claude_config, steps)
```

The enriched context includes:
- Full pipeline dossier (every phase's results, conflicts, resolutions)
- Semantic triage assessment (difficulty, prerequisites, priority files)
- Repository context (branches, build/test commands, fix description)

**Phase 4b artifacts** saved to the run directory:
- `phase_4b_input.json` — full input context (config, dossier, triage) + post-run results (outcome, duration, cost, result summary)
- `phase_4b_log.txt` — complete Opus conversation log (tool calls, thinking, results)
- `run_and_fix_success.txt` or `run_and_fix_failure.txt` — copied from worktree

**No changes to `auto_code_gen` are required** — the gitted pipeline imports and calls the existing functions.

---

## YAML Config Format

```yaml
issue_id: "crio-post-stop-hooks-9613-to-1.33"

bug_description: "Fix race condition in post-stop hook ordering..."

repository:
  source_path: "/path/to/cri-o"
  workdir: "/tmp/bugfix-workdir"

branches:
  source: "origin/main"
  target: "b9ac8350f"  # commit or branch name

fix:
  commit: "92c18a2e2673764cd10f89b1a5061e2b26f44209"

build:
  command: "go build $(go list ./... | grep -v /test)"
  test_command: "go vet $(go list ./... | grep -v /test)"

model: "claude-sonnet-4-6"
```

Config files should be placed in `auto_bug_fix/configs/` and are gitignored.

---

## Key Design Decisions

### Prerequisite portability

The triage prompt distinguishes between two kinds of prerequisites:

- **`portable: true`** — The fix depends on the prerequisite's actual behavior (e.g., a new type the fix stores data in). These are cherry-picked before the main fix.
- **`portable: false`** — The prerequisite only changes a calling convention (extra parameter, renamed function, moved package). The resolution agent adapts the call site to match the target branch's existing API instead of porting the entire refactor.

This prevents porting unnecessary refactors that bloat the diff without adding fix-relevant behavior.

### Comprehensive baseline (Phase 1)

Phase 1 runs build, test, and all lint/verification commands on the **clean target branch** before any changes. This baseline data is used by Phase 4.1 to distinguish regressions from pre-existing failures:

- If a verification command already failed on the clean branch → logged as "pre-existing" and skipped
- Only NEW failures introduced by the backport trigger Phase 4a/4b escalation

This eliminates false Phase 4b escalations for issues like missing lint tools or `//go:build test`-gated test failures.

### files_to_skip handling

Files identified by the triage as irrelevant (e.g., lint changes depending on unported refactors) are:
1. **Excluded from test port** (Phase 0.5c) — won't be ported as test files
2. **Reverted after cherry-pick** (Phase 2+3a) — both conflicting AND cleanly-applied skip files are reset to the target branch version

---

## Key Differences from auto_code_gen Bug Fix

| Aspect | auto_code_gen | Gitted pipeline |
|--------|---------------|-----------------|
| Primary mechanism | LLM generates patch from scratch | `git cherry-pick` + LLM conflict resolution |
| Cost for clean cherry-picks | ~$3-10 (full LLM pipeline) | $0 (no LLM needed) |
| Cost for conflicts | ~$5-15 | $0.50-5 (conflict resolution only) |
| Generated file handling | LLM writes them | Auto-resolve with `--theirs` + run codegen |
| Prerequisite detection | Manual | Automatic (git log -S + LLM analysis) |
| Test porting | Combined with code plan | Separate Phase 0.5c before fix |
| Baseline verification | None | Build + test + lint on clean branch |
| Pre-existing failure filtering | None | Phase 4.1 skips baseline failures |
| Escalation path | N/A | Falls back to auto_code_gen's RunAndFixPrompt |
| Language support | C/systems (gcc, openssl) | Any (tested on Go/CRI-O, Go/Kubernetes, C/OpenSSL) |

---

## Tested Backports

### CRI-O Post-Stop Hooks (PR #9613)

| Target | Conflicts | Model | Cost | Duration | Match vs Human |
|--------|-----------|-------|------|----------|---------------|
| release-1.34 | 1 | Sonnet + Opus 4b | $7.12 | 28 min | 7/7 files identical (PR #9699) |
| release-1.33 | 5 | Sonnet | $6.82 | 26 min | 6/7 identical, 1 cosmetic (PR #9741) |

### Kubernetes

| Commit | Files | Conflicts | Model | Cost | Outcome |
|--------|-------|-----------|-------|------|---------|
| ReplicaSet availability | 6 | 3 | Opus | $3.44 | Success |
| CRI gogo-protobuf removal | 28 | 5 | Sonnet | $2.20 | Success |
| KEP-5229 scheduler dispatcher | 34 | 11 | Sonnet | $10.85 | Success (Phase 4a) |
| PreFilterPreBind | 13 | 6 | Sonnet | $2.49 | Success |
| Workload scheduling cycle | 24 | 6 | Opus | $14.08 | Success |
| Workload API v1alpha2 | 248 | 106→44 | Sonnet | $3.43 | Partial (regen fails) |
| Cloud provider cleanup | 28 | 0 | — | $0.00 | Clean cherry-pick |
| Build tags removal | 412 | 0 | — | $0.16 | Clean cherry-pick |

---

## Running

```bash
# From the new_ai_auto_analyze directory
python -m auto_bug_fix.run_bug_fix auto_bug_fix/configs/your_config.yaml
```

Output (in `runs/<issue_id>_<timestamp>/`):
- `run_log.txt` — full pipeline log
- `dossier.md` — full audit trail
- `semantic_triage.json` — Phase 0.5 assessment
- `<issue_id>_<timestamp>.json` — machine-readable run metrics
- `phase_4b_input.json` — Phase 4b context (if triggered)
- `phase_4b_log.txt` — Phase 4b conversation (if triggered)
- `run_and_fix_success.txt` / `run_and_fix_failure.txt` — Phase 4b result (if triggered)
- `pipeline.diff` / `human_backport_*.diff` — comparison diffs (if generated)
