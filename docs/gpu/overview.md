# GPU performance analysis — overview

The `llm/` subtree implements an end-to-end flow for comparing and optimizing
GPU inference frameworks. Each stage is usable alone; together they form a
campaign from benchmarks to an upstream PR.

```
auto_config → auto_profile → auto_analyze → auto_code_gen → pr_flow / pr_review
```

| Stage | Package | User outcome |
|-------|---------|--------------|
| Config | `llm/auto_config` | Model inventory and profiling run configs |
| Profile | `llm/auto_profile`, `llm/auto_profile_server_client` | Benchmarks + nsys/PyTorch traces |
| Analyze | `llm/auto_analyze` | Annotated Perfetto traces, cross-trace diffs, improvement plans |
| Codegen | `llm/auto_code_gen` | Reviewed patches validated by runtime gates |
| PR | `llm/auto_code_gen/pr_flow` | Human-confirmed push/PR; agent never touches remotes |

Subtree map and example campaign chain:
[llm/README.md](../../llm/README.md).

## What you need

- Claude access for analysis and (by default) codegen agents
- Framework source checkouts for the frameworks you compare
- Either existing traces **or** GPU machines / Docker / cluster access to profile
- Optional: `gh` for the PR publish scripts

## Typical paths

### Path A — Analyze traces you already have

1. [Create a single-trace config and run analysis](analysis.md)
2. Optionally run cross-trace comparison and an improvement plan
3. Open annotated JSON in [Perfetto](https://ui.perfetto.dev)

### Path B — Full campaign from profiling

1. [Run profiling](profiling.md) (`run_all.sh` or step scripts)
2. Parse results into analysis configs
3. Analyze (and optionally PDF / Chrome traces / JIRAs)
4. [Codegen](codegen.md) against a chosen improvement
5. Human PR flow when ready to publish

### Path C — Port one known improvement

1. Start from cross-trace output + `code_gen_config_example.json`
2. Run `run_code_gen` (resume with `--resume`)
3. Use PR flow only after runtime gates pass

## Orchestration scripts

From the repository root:

```bash
./run_all.sh ./llm/auto_profile/test_configs/deepseek_r1_nvfp4/run_deepseek_r1_nvfp4.json
```

Useful flags: `--skip-profile`, `--skip-parse`, `--skip-analyze`, `--skip-pdf`,
`--skip-trace`, `--skip-jiras`. Individual steps:
`run_step1_profile.sh`, `run_step2_parse.sh`, `run_step3_analyze.sh`.

Scheduled variant with GPU availability checks: `./run_all_scheduled.sh`.

## Claude Code skills

If you use Claude Code in this repo, skills such as `/profile`, `/analyze`,
`/codegen`, `/perf-pipeline`, and `/runtime` wrap the same entry points. See
[CLAUDE.md](../../CLAUDE.md).
