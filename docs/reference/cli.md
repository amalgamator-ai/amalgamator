# CLI entry points

Run all commands from the **Amalgamator repository root** unless noted. Prefer
`uv run python -m ...` so the locked environment is used.

Inspect live flags with `--help` on each module; this page is a map, not a
duplicate of every option.

## Bug-fix backporting

| Command | Purpose |
|---------|---------|
| `python -m auto_bug_fix.run_bug_fix <case.yaml>` | Run the backport pipeline |
| `python -m auto_bug_fix.run_bug_fix <case.yaml> --models PATH` | Explicit model registry |
| `python -m backport_scanner.run_scanner --config <scan.yaml>` | Scan refs for a missing fix |
| `python -m backport_scanner.run_scanner --config <scan.yaml> --output report.json` | Also write JSON report |
| `python -m backport_scanner.generate_config --config <scan.yaml> --mode generate --output-dir DIR` | Emit backport cases |

## GPU — config and profiling

| Command | Purpose |
|---------|---------|
| `python -m llm.auto_config` | Model inventory → profiling configs |
| `./llm/auto_profile/run_profile_core.sh <run_config.json>` | Core profiling runner |
| `./run_step1_profile.sh <run_config.json>` | Step 1 wrapper |
| `./run_step2_parse.sh <run_config.json>` | Parse results → analysis configs |
| `./run_step3_analyze.sh <run_config.json>` | Analysis (+ optional PDF/trace/JIRA) |
| `./run_all.sh <run_config.json>` | Profile → parse → analyze |
| `./run_all_scheduled.sh ...` | Scheduled orchestration with GPU checks |

`run_all.sh` flags: `--skip-profile`, `--skip-parse`, `--skip-analyze`,
`--skip-pdf`, `--skip-trace`, `--skip-jiras`.

## GPU — analysis

| Command | Purpose |
|---------|---------|
| `python -m llm.auto_analyze.create_single_trace_config ...` | Build single-trace JSON |
| `python -m llm.auto_analyze.run_single_trace --config FILE` | Single-trace analysis |
| `python -m llm.auto_analyze.create_cross_trace_config ...` | Build cross-trace JSON |
| `python -m llm.auto_analyze.run_cross_trace --config FILE` | Cross-trace analysis |
| `python -m llm.auto_analyze.run_chrome_trace ...` | Chrome/Perfetto visualization |
| `python -m llm.auto_analyze.run_summary_pdf ...` | PDF summary |
| `python -m llm.auto_analyze.run_jiras ...` | Create JIRA tasks from plans |

## GPU — codegen and PR helpers

| Command | Purpose |
|---------|---------|
| `python -m llm.auto_code_gen.run_code_gen --config FILE` | Full codegen pipeline |
| `python -m llm.auto_code_gen.run_code_gen --config FILE --resume` | Resume |
| `python -m llm.auto_code_gen.run_runtime_iters --config FILE` | Runtime iterations only |
| `python -m llm.auto_code_gen.run_investigate_issue ...` | Post-pipeline investigation |
| `python -m llm.auto_code_gen.run_fix_issue ...` | Post-pipeline fixing |
| `python -m llm.auto_code_gen.run_work_items ...` | Arbitrary work items |
| `python -m llm.auto_code_gen.run_generate_pr_commands ...` | PR command generation |
| `python -m llm.auto_code_gen.run_rebase_pr ...` | Rebase assistance |
| `python -m llm.auto_code_gen.run_pr_descs_update ...` | Update PR title/body |
| `python -m llm.auto_code_gen.run_summary ...` | PowerPoint summary |

Human network steps for PRs are shell scripts under
`llm/auto_code_gen/pr_flow/` — see that package README.

## Development gates

Documented in [CONTRIBUTING.md](../../CONTRIBUTING.md) and [AGENTS.md](../../AGENTS.md):

```bash
uv sync --extra test --extra dev --locked --python 3.12
uv run pre-commit run --all-files --show-diff-on-failure
uv run pyrefly check
uv run pytest
```
