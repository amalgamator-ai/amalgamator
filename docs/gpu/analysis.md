# Trace analysis

`llm/auto_analyze` turns GPU traces into annotated Perfetto views, cross-trace
comparisons, and optional improvement plans. Full package docs:
[llm/auto_analyze/README.md](../../llm/auto_analyze/README.md).

## Prerequisites

- Project install ([Getting started](../getting-started.md))
- Claude credentials ([Authentication](../authentication.md))
- Framework source tree(s) at the commit(s) you care about
- Trace file(s) and preferably a run log from the profiling job

## Single-trace analysis

Produces an annotated Chrome-trace JSON you can open in
[Perfetto](https://ui.perfetto.dev).

```bash
# 1. Generate config
uv run python -m llm.auto_analyze.create_single_trace_config \
    --model nvidia/Kimi-K2.5-NVFP4 \
    --gpu-type B200 \
    --batch-size-range 1 \
    --prefill-size-range 4 \
    --output-size-range 1024 \
    --trace-file /path/to/trace.json.gz \
    --run-log-file /path/to/run_log.txt \
    --clean-source-code-path /path/to/vllm \
    --commit-id HEAD \
    --analyze-output-dir /path/to/output \
    --output-config-file /path/to/config

# 2. Run
uv run python -m llm.auto_analyze.run_single_trace --config /path/to/config.json

# 3. Open
#    /path/to/output/single_trace_transformer_block.json  in Perfetto
```

Pipeline steps (automated): high-level ops from source → GPU ops from trace →
kernel↔operation correlation → annotated trace.

Key outputs:

| File | Purpose |
|------|---------|
| `single_trace_transformer_block.json` | Annotated Perfetto trace |
| `single_trace_transformer_block.txt` | Human-readable summary |
| `gpu_ops_to_blocks.txt` | Full correlation |
| `median_block.txt` | Selected median transformer block |

Optional: `--enable-single-trace-perf-analysis` for bottleneck notes.

Step-by-step guide:
[single_trace_analysis_and_annotation_guide.md](../../llm/auto_analyze/examples/single_trace_analysis_and_annotation_guide.md).

## Cross-trace analysis

Compare two (or more) single-trace result directories — same framework across
commits, or different frameworks on the same model.

```bash
uv run python -m llm.auto_analyze.create_cross_trace_config \
    --trace-result-dir /path/to/trace_a/analyze \
    --trace-result-dir /path/to/trace_b/analyze \
    --target-trace-id 0 \
    --analyze-output-dir /path/to/cross_output \
    --output-config-file /path/to/cross_config

uv run python -m llm.auto_analyze.run_cross_trace --config /path/to/cross_config.json
```

`target_trace_id` selects which trace is the improvement **target** (the side
you intend to change). Do not assume index 0/1 means faster/slower — set it
explicitly. Enable `make_improvement_plan` (helper flag or config field) when you
want ranked coding guides for [codegen](codegen.md).

Guides:

- [Cross-commit comparison](../../llm/auto_analyze/examples/cross_commit_comparison_guide.md)
- [Cross-framework comparison](../../llm/auto_analyze/examples/cross_framework_comparison_guide.md)

## Extra tools

```bash
uv run python -m llm.auto_analyze.run_chrome_trace --help
uv run python -m llm.auto_analyze.run_summary_pdf --help
uv run python -m llm.auto_analyze.run_jiras --help
```

Example configs: `llm/auto_analyze/configs/single_trace_config_example.json`,
`cross_trace_config_example.json`.
