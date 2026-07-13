---
description: "Run the gitted bug fix pipeline — git cherry-pick with LLM-powered conflict resolution and fallback"
---

# Gitted Bug Fix Pipeline

Run the git-first bug fix backporting pipeline. This pipeline tries `git cherry-pick` as the primary mechanism and falls back to LLM-powered resolution when conflicts arise. For complex cases, it escalates to the full Opus-powered build-test-fix loop.

## Arguments

The user provides:
- A path to a YAML config file (required)

## Instructions

1. Parse the argument to get the config path. If not provided:
   - List available configs: `ls auto_bug_fix/configs/*.yaml`
   - Show the example: `auto_bug_fix/configs/example_backport.yaml`
   - Ask the user if they want to use an existing config or create a new one.

2. If the user wants to create a new config, help them build a YAML file with these required fields:
   ```yaml
   issue_id: "CVE-2024-XXXX"
   bug_description: "Brief description of the bug"

   repository:
     source_path: "/path/to/repo"
     workdir: "/path/to/worktree/dir"

   branches:
     source: "origin/main"
     target: "origin/release-1.0"

   fix:
     commit: "abc123def456"

   build:
     command: "make -j$(nproc)"
     test_command: "make check -j$(nproc)"

   model: "claude-sonnet-4-6"
   ```

   Optional fields:
   ```yaml
   model: "claude-opus-4-6"           # for harder backports
   allowed_modules: ["src/core/"]      # restrict file modifications
   lint_command: "make lint"            # additional verification
   output:
     directory: "runs/my-backport"
   ```

3. Validate the config:
   - Check the file exists
   - Verify it's valid YAML
   - Check that the repository path exists

4. Run the pipeline:
   ```bash
   source env.sh && python -m auto_bug_fix.run_bug_fix "<config_path>"
   ```

5. This is a long-running operation. Inform the user.

6. Report results:
   - Show the outcome (success, pipeline_stop, pipeline_escalation)
   - Show the dossier path (human-reviewable audit trail)
   - Show cost and duration from the tracker JSON
   - Summarize the cherry-pick resolution path (clean, resolved, escalated)
   - For multi-target configs, show per-target results

## Pipeline Phases

| Phase | Description |
|-------|-------------|
| Phase 0 | Deterministic triage (patch-ID, ancestry, seed files) |
| Phase 0.5 | Semantic triage (LLM: difficulty, prerequisites) |
| Phase 0.5b | Port prerequisite commits |
| Phase 0.5c | Port test files from fix commit |
| Phase 1 | Comprehensive baseline (build, test, lint) |
| Phase 1.5 | RED triage (when tests pass unexpectedly) |
| Phase 2+3a | Cherry-pick + conflict resolution |
| Phase 3b | Regeneration (rebuild generated files) |
| Phase 4 | Verify (build + test) |
| Phase 4a | Build error recovery (LLM fix) |
| Phase 4.1 | Extended verification (lint commands) |
| Phase 4b | Escalation to Opus build-test-fix loop |
| Phase 4.5 | Semantic equivalence (range-diff) |

## Multi-Target Support

YAML configs can specify multiple target branches:
```yaml
targets:
  - branch: "origin/release-1.0"
    issue_id: "CVE-2024-XXXX-1.0"
  - branch: "origin/release-0.9"
    issue_id: "CVE-2024-XXXX-0.9"
```

Each target is processed serially in its own worktree.

## Example Usage

```
/bugfix auto_bug_fix/configs/example_backport.yaml
/bugfix auto_bug_fix/configs/crio_post_stop_hooks.yaml
```
