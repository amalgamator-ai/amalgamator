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
              regeneration commands, verification commands)
Phase 0.5b ─ Port prerequisite commits (if portable prerequisites found)
Phase 0.5c ─ Port test files from fix commit (for RED/GREEN checking)
Phase 1    ─ Baseline + RED check (run ported tests before fix — should fail)
Phase 1.5  ─ Failure-mode confirmation (signature comparison)
Phase 2+3a ─ Cherry-pick + conflict resolution
              ├─ Auto-resolve generated/binary files (--theirs)
              └─ LLM resolves remaining source conflicts
Phase 3b   ─ Regeneration (run project codegen commands, rollback on failure)
Phase 4    ─ Verify (build + test from YAML config)
Phase 4a   ─ Build error recovery (LLM single-shot fix)
Phase 4.1  ─ Extended verification (commands discovered by Phase 0.5)
              └─ If fails → Phase 4a → if still fails → Phase 4b
Phase 4b   ─ Handoff to auto_code_gen RunAndFixPrompt (autonomous loop)
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
                         │  (git show --name-only)  │
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
                         │  • estimated difficulty           │
                         │  • prerequisite commits          │
                         │  • regeneration commands          │
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
                         prerequisites?     no prerequisites
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
                   │  (max 5 depth):       │         │
                   │  cherry_pick_and_     │         │
                   │  resolve(prereq_sha)  │         │
                   └──────────┬───────────┘         │
                              │                     │
                              ▼                     │
                   ┌──────────────────────┐         │
                   │  Failed? Log warning  │         │
                   │  and continue without │         │
                   └──────────┬───────────┘         │
                              │                     │
                              ◄─────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Test files in     │
                    │ fix commit?       │
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
          ║       PHASE 1 — BASELINE + RED CHECK                 ║
          ╚═══════════════════════════════════════════════════════╝
                     │
                     ▼
              ┌──────────────────────┐
              │ Run test suite       │
              │ (capture baseline)   │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Tests pass?          │
              └──────────┬───────────┘
                    │           │
                  pass        fail
                    │           │
                    ▼           ▼
         ┌──────────────┐  ┌──────────────────┐
         │ commit_type   │  │ Capture failure  │
         │ is bugfix?    │  │ signature        │
         └──────┬───────┘  │ (s_target)       │
           yes  │   no     └────────┬─────────┘
            │   │    │              │
            ▼   │    │              │
     ┌──────────┐    │              │
     │ Ported   │    │              │
     │ tests?   │    │              │
     └────┬─────┘    │              │
      yes │  no      │              │
       │  │   │      │              │
       ▼  │   │      │              │
   ┌───────┐  │      │              │
   │ESCALATE│  │     │              │
   │(already│  │     │              │
   │ fixed) │  │     │              │
   └───────┘  │      │              │
              ▼      ▼              │
         ┌──────────────┐           │
         │  Continue    │◄──────────┘
         └──────┬───────┘
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
         └──────────┬───────────┘
                    │
             ┌──────┴────────┐
           clean         conflicts
             │               │
             ▼               ▼
       ┌──────────┐  ┌──────────────────────┐
       │  Done!    │  │ Re-apply last strat   │
       │  Commit   │  │ (keep conflicts open) │
       └──────────┘  └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │  Generated/binary     │
                     │  files in conflicts?  │
                     │  (from Phase 0.5      │
                     │   files_to_skip)      │
                     └──────────┬───────────┘
                           yes  │   no
                            │   │    │
                            ▼   │    │
                     ┌──────────────┐│
                     │ Auto-resolve ││
                     │ with --theirs││
                     │ + git add    ││
                     └──────┬──────┘│
                            │       │
                            ◄───────┘
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
          ╚═══════════════════════════════════════════════════════╝
                    │
             ┌──────┴──────┐
          commands      no commands
          from 0.5         │
             │             │
             ▼             │
      ┌──────────────┐     │
      │ Run each     │     │
      │ verification │     │
      │ command      │     │
      └──────┬───────┘     │
        │          │       │
      all pass   failures  │
        │          │       │
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
        │                 │ auto_code_gen's         │
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

**No changes to `auto_code_gen` are required** — the gitted pipeline imports and calls the existing functions.

---

## YAML Config Format

```yaml
issue_id: "k8s-replicaset-availability-backport"

bug_description: "ReplicaSet controller schedules pod availability
  checks at the wrong time"

repository:
  source_path: "/path/to/kubernetes"
  workdir: "/path/to/worktree-dir"

branches:
  source: "origin/release-1.34"
  target: "origin/release-1.33"

fix:
  commit: "2cb48f77f0f4"

build:
  command: "go build ./pkg/controller/..."
  test_command: "go test -count=1 -timeout=300s ./pkg/controller/..."

model: "claude-sonnet-4-6"
```

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
| Escalation path | N/A | Falls back to auto_code_gen's RunAndFixPrompt |
| Language support | C/systems (gcc, openssl) | Any (tested on Go/Kubernetes, C/OpenSSL) |

---

## Tested Backports (Kubernetes)

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
# From the ai_auto_perf_analysis directory
python -m auto_bug_fix.run_bug_fix configs/your_config.yaml
```

Output:
- `dossier.md` — full audit trail
- `runs/<issue_id>_<timestamp>.json` — machine-readable run record
- `semantic_triage.json` — Phase 0.5 assessment (in worktree)
- Patch file via `git format-patch` on the worktree
