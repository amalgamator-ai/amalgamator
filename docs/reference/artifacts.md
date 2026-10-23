# Artifacts and outputs

Amalgamator writes run products next to your working directory or into paths you
set in config. Treat `runs/` and campaign output dirs as **generated artifacts**,
not source.

## Backport pipeline

Default: `runs/<case-filename-without-extension>_<timestamp>/` under the
current working directory.

| Artifact | Description |
|----------|-------------|
| `run_log.txt` | Full console log for the run |
| `<target>/dossier.md` | Human review narrative — decisions, gates, diffs |
| `<target>/candidate.patch` | Ported change as a patch file |
| `<target>/semantic_triage.json` | Triage assessment |
| `<target>/<issue>_<ts>.json` | Machine-readable tracker |

Also: a preserved git branch in the worktree repo (`amalgamator/<issue-id>` or
similar) holding the ported commits.

Worktrees themselves live under `repository.workdir` from the case file
(default `/tmp/bugfix-workdir`).

## Backport scanner

Console report by default; optional `--output report.json`. Generated cases from
`generate_config` go to the `--output-dir` you pass.

## Trace analysis

Configured via `analyze-output-dir` / config fields. Typical single-trace files:

- `single_trace_transformer_block.json` / `.txt`
- `transformer_block_high_level_ops.txt`
- `gpu_ops.txt`, `gpu_ops_to_blocks.txt`, `median_block.txt`

Cross-trace: `cross_matching_blocks.txt`, `cross_compare_blocks.txt`, optional
`cross_improvement_plan.txt`.

## Codegen

Everything under the config's `output_dir`:

- Versioned plans and reviews (`code_port_plan_V{N}.txt`, …)
- Patches (`code_gen_V{N}.patch`, `code_gen_review_V{N}.patch`,
  `code_gen_runtime_V{N}.patch`)
- `runtime/success_result.txt` on successful validation
- `pr_p3/pr_state.json` during PR flow

Resume logic looks for declared `PipelineStep` output files and prefers
non-empty reviewed artifacts when choosing versions.

## Profiling campaigns

Layout depends on the run config and infra JSON (often under campaign-specific
result directories referenced by parse/analyze steps). Prefer the paths printed by
`run_step2_parse.sh` / analysis config helpers rather than hard-coding.

## Large traces and LFS

In this repository, `*.nsys-rep` and `*.json.gz` are Git LFS-tracked. You need
`git-lfs` only if you commit those files here — not to use Amalgamator against
external framework checkouts.
