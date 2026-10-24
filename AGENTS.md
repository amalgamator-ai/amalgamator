# AGENTS.md — Amalgamator

Repository-wide guidance for coding agents. Read the relevant package documentation
below before changing a pipeline. Keep this file focused on engineering contracts;
put detailed usage and configuration examples in the package READMEs.

Amalgamator automates GPU performance analysis and optimization porting, plus
git-first bug-fix backporting and discovery of releases missing a fix. Pipeline
results are supported by artifacts, deterministic checks, and human review.

User-facing guides (getting started, workflows, CLI map, FAQ) live in
[docs/](docs/README.md). Keep those guides aligned when commands or user-visible
workflows change; this file remains the engineering-contract guide for agents.

## Setup and verification

Run commands from the repository root. Python **3.12+** is required; use syntax
compatible with 3.12 even when developing on 3.14. `pyproject.toml`, `uv.lock`,
`.pre-commit-config.yaml`, and `.github/workflows/ci.yml` define the toolchain.

```bash
# Reproduce the minimum-version CI environment.
uv sync --extra test --extra dev --locked --python 3.12

# CI gates.
uv run pre-commit run --all-files --show-diff-on-failure
uv run pyrefly check
uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md#development-setup) for alternative setup
and hook invocations. Keep these CI-aligned commands consistent with its guidance
when the toolchain changes.

- For live Claude SDK runs, configure `ANTHROPIC_API_KEY` or the SDK's supported
  backend authentication. Backport/scanner profiles can supply `api_key` or
  `api_key_file` through `models.yaml`; GPU `claude_sdk` agents use SDK/environment
  authentication. Tests require no live credentials.
- CI runs formatting, lint, and type checking on 3.12; pytest runs on 3.12 and
  3.14. To exercise the other interpreter, sync with `--python 3.14`, then test.
- Black and Ruff versions come from the pinned pre-commit hooks. Ruff runs with
  `--fix`; inspect the resulting diff. Pyrefly needs the project environment and
  runs separately, not as a default commit-stage hook.
- For a focused edit, start with the affected tests; run all three gates before
  handing off code changes. For documentation-only edits, check referenced paths
  and commands, run `uv run pre-commit run --files <changed-files>`, and run
  `git diff --check`.
- Dependency or package-version changes require a regenerated `uv.lock` in the
  same change (`uv lock`, or `uv add` for a dependency). Routine setup uses
  `--locked` so it cannot silently change the resolution.

## Where to work

| Area | Responsibility and starting points |
| --- | --- |
| `common/` | Shared SDK glue, `PipelineStep`, model profiles, completion/agent runners, iteration utilities, logging. |
| `auto_bug_fix/` | Git-first backport phases, conflict resolution, evidence, verification, and dossiers. Start with [its README](auto_bug_fix/README.md), `run_bug_fix.py`, and `config_loader.py`. |
| `backport_scanner/` | Ref detection, semantic triage, and backport-case generation. Start with [its README](backport_scanner/README.md), `triage.py`, and `generate_config.py`. |
| `llm/auto_config/` | Fetch framework recipes and generate server/client profiling configs; `cli.py`, `generate.py`. |
| `llm/auto_profile/` | Shell-based profiling and result parsing; `parse_run_config.py`, `run_profile_core.sh`, `run_profile_summary.py`. |
| `llm/auto_profile_server_client/` | Python benchmark orchestration, Docker/cluster execution, metrics, and profiling; `pipeline.py`, `run_config.py`, `frameworks/`. |
| `llm/auto_analyze/` | Single/cross-trace analysis, improvement plans, trace/report generation. Start with [its README](llm/auto_analyze/README.md), `configs/`, and `prompts/`. |
| `llm/auto_code_gen/` | Optimization porting and runtime iterations. Start with [its README](llm/auto_code_gen/README.md), `code_gen_configs.py`, and `use_cases/`. |
| `llm/auto_code_gen/agents/` | GPU-pipeline agent configuration and sessions; consume through `llm.agents`. Read the [runtime contract](llm/auto_code_gen/agents/README.md). |
| `llm/auto_code_gen/pr_flow/`, `llm/auto_code_gen/pr_review/` | Upstream PR preparation, review fixes, and rebase verification. Read the [PR workflow](llm/auto_code_gen/pr_flow/README.md). |
| `tests/` | Tests mirror application packages: `tests/common`, `tests/auto_bug_fix`, `tests/backport_scanner`, and `tests/llm`. |

GPU code and module entry points use the `llm.` prefix. See [llm/README.md](llm/README.md)
for the profile → analyze → codegen → PR artifact flow. Backporting stays at the
repository root. Dependencies flow from pipelines into `common/`; `common/`
must not import the pipelines. The scanner also reuses `auto_bug_fix` git helpers.

## Tests that match the change

```bash
# Shared model infrastructure, backporting, and scanner integration.
uv run pytest tests/common tests/auto_bug_fix tests/backport_scanner

# Analysis, codegen, agent sessions, and PR gates.
uv run pytest tests/llm

# Example focused regression check.
uv run pytest tests/llm/auto_code_gen/test_code_gen_configs.py

# Required when changing the test extra's Claude SDK pin or SDK transport glue.
uv run pytest tests/auto_bug_fix/test_sdk_profiles.py -k subprocess_transport_smoke
```

- Tests use temporary git repositories, mocked providers, and local HTTP
  endpoints. They do not require GPU hardware or live model credentials; some
  agent tests need permission to bind a loopback port.
- `tests/auto_bug_fix/conftest.py` installs SDK import stubs. Keep the real-SDK
  transport canary in its separate interpreter; in-process mocks cannot verify
  the installed SDK's transport behavior.
- For behavior changes, test the observable result at the affected boundary:
  loaded config → rendered prompt, provider options → request, git operation →
  resulting patch, or artifacts → verdict. Use `tmp_path` and temporary repos
  instead of real campaign configs or framework checkouts.

## Contracts to preserve

### Shared infrastructure and model configuration

- Reuse `common.claude_utils` for canonical step/config types and SDK execution.
  Pipeline-specific `claude_utils.py` modules specialize personas and re-export
  shared types. Preserve public exports, including
  `common.llm_client.is_claude_model`; an explicit `name as name` may be intentional.
- Backporting and scanning use `common.model_registry.ModelProfile` and
  `models.yaml`. Selection is explicit `--models`, then a registry beside the
  case, then `~/.config/amalgamator/models.yaml`; registries are not merged.
  An invalid explicit registry must fail rather than fall back.
- A named profile supplies the complete model, provider, transport, and params.
  Keep options isolated per request/client, including Phase 4b and summarization;
  retain the legacy no-profile behavior without leaking settings between runs.
  Use `public_snapshot()` / `public_config_snapshot()` for profile-bearing logs
  and persisted config, rather than serializing credentials.
- GPU analysis/codegen use `llm.agents.AgentConfig`: embedded `agent` JSON or a
  replacing `--agent-config` file. `agent.model/backend` selects the coding
  agent; workload `model` and `runtime_backend` describe the GPU job. These are
  distinct from the backport registry schema.
- Path semantics differ: registry key-file paths are working-directory-relative;
  GPU agent key-file paths are relative to the declaring JSON file; backport
  `fix.patch` is case-file-relative. Preserve each loader's contract and test it.

### Backporting and scanning

- In conflict resolution, try cherry-pick and mechanical application before LLM
  editing. Triage claims and adaptations need deterministic target-tree evidence;
  unchanged target content and source-fix bytes must retain provenance.
- Keep baseline failures distinct from regressions, and carry successful recovery
  edits into both the candidate patch and preserved branch. Update tracker and
  dossier output when phase outcomes or evidence change.
- The scanner's exact, equivalent, heuristic, ambiguous, and error outcomes are
  different contracts. Preserve the detection cascade and generated cases'
  selected profile/registry; a heuristic match is not proof of equivalence.

### GPU analysis and code generation

- The orchestrator owns phase ordering and iterations; `UseCase` implementations
  supply domain-specific prompts and steps. Framework-specific benchmark behavior
  belongs in `llm/auto_profile_server_client/frameworks/` adapters.
- Treat prompt fields and artifact names as interfaces. Update config loading,
  prompt construction, output declarations, downstream readers, and examples
  together. In codegen, `target_trace_id` determines the target: test both 0 and 1
  rather than assuming trace order means faster/slower.
- Resume/convergence is artifact-based. `PipelineStep.output_files` declares
  completion evidence; preserve non-empty-file checks and setup-step handling.
  `common.iteration_utils` owns the exact first-line `CONVERGED` marker and
  versioned-output resolution, including preference for non-empty reviewed output.
- Runtime/PR verdicts depend on measured artifacts. Preserve missing/stale-evidence
  failures, feature-OFF baseline identity, measured-content fingerprints, and
  performance/correctness/review-closure checks when changing verification.
- `pr_flow` deliberately separates agent edits from human-run git/publication
  scripts. Preserve its existing confirmation, scope, and verdict checks.

## Style and change discipline

- Follow Black's 88-column layout and the configured Ruff rules. Keep Python 3.12
  exception syntax (`except (A, B):`). Module docstrings precede imports.
- Use type annotations at changed interfaces and existing dataclass/config
  patterns. Narrow dynamic SDK/provider values at their boundary; keep any
  necessary type-ignore local and explained rather than weakening global checks.
- Review lint autofixes for behavior changes: unused imports may be public
  re-exports, and removed assignments may leave useless expression statements.
- Keep mechanical formatting separate from behavioral changes. Only purely
  mechanical commits belong in `.git-blame-ignore-revs`.
- Preserve existing Apache-2.0/SPDX headers. Use focused `feat:`, `fix:`, `docs:`,
  `test:`, or `chore:` commit subjects when a commit is requested; contribution
  and release details live in [CONTRIBUTING.md](CONTRIBUTING.md).

### DCO sign-off

- When asked to create commits, sign off every new commit with `git commit -s`
  (`--signoff`). The `Signed-off-by: Full Name <email@example.com>` trailer
  certifies the [Developer Certificate of Origin](https://developercertificate.org/).
- Use the contributor's Git identity (`user.name` and `user.email`). Check the
  effective identity with `git var GIT_AUTHOR_IDENT` and
  `git var GIT_COMMITTER_IDENT`; for new commits, the sign-off name and email must
  match the author. Ask if the identity is missing or inconsistent; never invent
  a name/email or substitute an agent identity.
- Verify the trailer with `git log -1 --format=full`. Preserve existing authorship
  and sign-off trailers when amending or cherry-picking commits.
- DCO sign-off (`-s`) is separate from GPG/SSH commit signing (`-S`);
  a cryptographic signature does not replace the DCO trailer.

## Running pipelines and handling artifacts

Use module help to inspect the actual CLI before assembling a run:

```bash
uv run python -m auto_bug_fix.run_bug_fix --help
uv run python -m backport_scanner.run_scanner --help
uv run python -m llm.auto_analyze.run_single_trace --help
uv run python -m llm.auto_code_gen.run_code_gen --help
uv run python -m llm.auto_profile_server_client --help
uv run python -m llm.auto_config --help
```

- Backporting takes a positional case path; scanner, analysis, and codegen take
  `--config`. Copy the relevant checked-in example and use its package README.
  Claude-specific pipeline runbooks are in `.claude/skills/`.
- Full pipeline runs use real model services, external repositories, or GPU/
  Docker/cluster resources. Run them for an explicitly requested workload;
  use the tests above for ordinary repository validation. `run_all.sh` also
  includes report and JIRA stages.
- Config loading is not always read-only: `CodeGenConfig.from_json()` can create
  a target worktree. Workspace cleanup helpers can reset or remove their target.
  Keep tests and experiments in dedicated temporary repositories/output paths.
- Preserve existing local/untracked work. `runs/` and generated results are run
  artifacts, not application sources. Keep keys and local credential-bearing
  configs out of commits. Trace types `*.nsys-rep` and `*.json.gz` use Git LFS;
  retain those `.gitattributes` rules alongside the generated `uv.lock` rule.

Finish with a concise description of the change, exact checks run and their
results, and any validation blocked by the environment. Update this guide when
commands or architectural contracts change; avoid recording transient run counts,
local paths, or copied dependency versions here.
