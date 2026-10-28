# Backport pipeline

`auto_bug_fix` ports a fix commit (or patch file) onto one or more target
branches, producing a reviewable commit, `candidate.patch`, and `dossier.md`.

Full docs: [auto_bug_fix/README.md](../../auto_bug_fix/README.md).

## Quickstart

```bash
mkdir -p my-backport
cp auto_bug_fix/configs/example_models.yaml my-backport/models.yaml
cp auto_bug_fix/configs/example_backport.yaml my-backport/case.yaml
```

Edit `models.yaml` for your provider ([Authentication](../authentication.md)).
Edit `case.yaml` for the repository and fix:

```yaml
issue_id: "PROJ-1234"
bug_description: "Fix null-pointer dereference in the connection teardown path"

repository:
  source_path: "/path/to/your/repo"
  workdir: "/tmp/bugfix-workdir"

branches:
  source: "origin/main"
  target: "origin/release-1.0"
  # or: targets: ["origin/release-1.0", "origin/release-1.1"]

fix:
  commit: "abc123def456"
  # or: patch: "fix.patch"   # path relative to the case file

build:
  command: "make -j$(nproc)"
  test_command: "make check -j$(nproc)"

model: m1
# advanced:
#   phase_4b_model: m3
```

Run from the Amalgamator repository root:

```bash
uv run python -m auto_bug_fix.run_bug_fix my-backport/case.yaml
uv run python -m auto_bug_fix.run_bug_fix my-backport/case.yaml --models /path/to/models.yaml
uv run python -m auto_bug_fix.run_bug_fix --help
```

The case path is a required positional argument. Exit status is 0 only if every
target succeeded.

## What happens (phases)

High level:

0. Triage (already present? follow-up fixes?)
0.5 Semantic triage + prerequisite/test port
1. Baseline build/test + RED expectation for ported tests
2–3. Cherry-pick → mechanical apply → LLM conflict resolution + audits
3b. Regenerate skipped generated files if needed
4. Verify (+ recovery / Phase 4b escalation)
4.5 Semantic equivalence (`git range-diff`)
5. Dossier for human review

Guiding rule: lines the fix changes keep the fix's bytes; untouched lines keep
the target's bytes; adaptations need hard evidence from the target tree.

## Reviewing results

Artifacts land under `runs/<case-stem>_<timestamp>/` in the working directory:

```
runs/case_20260929_153000/
├── run_log.txt
└── release-1.0/
    ├── dossier.md           # start here
    ├── candidate.patch
    ├── semantic_triage.json
    └── <issue>_<ts>.json    # machine-readable tracker
```

Ported commits are also preserved on a branch in the worktree repo, typically
`amalgamator/<issue-id>`, so you can push or cherry-pick after cleanup.

## Case file essentials

Required: `issue_id`, `bug_description`, `repository.source_path`, source/target
branch refs, and either `fix.commit` or `fix.patch`. The `build` section must
exist; individual commands may be empty to skip.

Optional highlights: `repository.build_subdir`, lint/file-check commands,
`advanced` toggles for mechanical resolution / snapback / evidence checks /
blind review, and Phase 4b model override.

Complete field tables: [Case YAML reference](../../auto_bug_fix/README.md#case-yaml-reference).

## Tips

- Use disposable `workdir` paths; the pipeline creates git worktrees there.
- Provide real build and test commands when possible — they are the verification
  gate. Empty commands skip verification and weaken the safety story.
- Prefer named profiles + `strict: true` in shared registries to catch typos.
- Keep secrets out of committed case files.
