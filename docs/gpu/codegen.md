# Code generation and PR flow

`llm/auto_code_gen` ports a planned performance improvement into a target
framework, reviews patches iteratively, and validates with runtime benchmarks.
Package README: [llm/auto_code_gen/README.md](../../llm/auto_code_gen/README.md).

## When to use it

You already have (typically from [cross-trace analysis](analysis.md)):

- A cross-trace config / improvement plan
- Source and target framework trees
- Claude (default) or a LiteLLM tool-calling agent backend
- GPU access for runtime iterations

## Pipeline phases

1. **Code trace** — document relevant code paths
2. **Code port plan** (+ review loop until `CONVERGED`)
3. **Test plan** (+ review; may combine with phase 2)
4. **Code gen** (+ review)
5. **Runtime iterations** — apply patch, benchmark, fix until gates pass or retries exhaust

Skip review loops with `*_skip_review` flags when you need a faster exploratory run.

## Quick start

```bash
# Copy and edit the example
cp llm/auto_code_gen/configs/code_gen_config_example.json my_codegen.json

uv run python -m llm.auto_code_gen.run_code_gen --config my_codegen.json

# Resume after interruption
uv run python -m llm.auto_code_gen.run_code_gen --config my_codegen.json --resume
```

Important LLM-framework fields:

| Field | Meaning |
|-------|---------|
| `use_case` | `"llm_framework"` |
| `cross_trace_config` | Path to cross-trace analysis config |
| `improvement_id` | Which improvement step to implement |
| `source_code_dir` | Target framework source (the tree being patched) |
| `num_runtime_iterations` | Max runtime fix attempts (default 10) |
| `output_dir` | All artifacts land here |

Agent backends: embed `agent` in the JSON or pass `--agent-config`. See
[agents/README.md](../../llm/auto_code_gen/agents/README.md) and
[agent_vllm_example.json](../../llm/auto_code_gen/configs/agent_vllm_example.json).

Standalone runtime iterations (without re-running earlier phases):

```bash
uv run python -m llm.auto_code_gen.run_runtime_iters --config my_codegen.json
```

## Outputs

Under `output_dir`, look for versioned plans/patches (`*_V{N}.*`), reviewed
patches, `code_gen_runtime_V{N}.patch`, and on success
`runtime/success_result.txt`. Resume prefers non-empty reviewed artifacts where
applicable.

On exhausted runtime retries, use investigation/fix helpers:

```bash
uv run python -m llm.auto_code_gen.run_investigate_issue --help
uv run python -m llm.auto_code_gen.run_fix_issue --help
```

## Upstream PR flow (human in the loop)

After runtime gates pass, [pr_flow](../../llm/auto_code_gen/pr_flow/README.md)
turns the result into an upstream PR (designed around `vllm-project/vllm`).

Invariant: **the AI never runs network git or `gh`**. Humans run announce-and-
confirm shell steps; the agent only edits files inside the patch scope with
remote git/`gh` credentials cleared.

High-level sequence:

1. Prepare / commit locally (`pr_flow_step_0_*`, `pr_flow_step_1_*`)
2. Optional rebase stage if upstream moved
3. Human push + `gh pr create`
4. Fetch review comments → agent fix iterations → human re-commit / update PR
5. Repeat until merged

Related modules: `run_generate_pr_commands`, `run_rebase_pr`,
`run_pr_descs_update`, and `pr_review/` for comment-driven fix loops.

## Related skills

Claude Code: `/codegen`, `/runtime`, `/investigate`, `/fix`, `/work-items`
(see [CLAUDE.md](../../CLAUDE.md)).
