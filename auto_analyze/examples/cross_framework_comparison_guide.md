# Cross-Framework Performance Comparison

## Motivation

When multiple LLM inference frameworks (vLLM, SGLang, TensorRT-LLM) run the same
model on the same hardware, they can differ significantly in per-layer GPU kernel
performance. Understanding *why* one framework is faster requires matching
operations across frameworks kernel-by-kernel and analyzing the source code
differences that explain each gap — which is extremely tedious to do manually.

The cross-framework analysis pipeline automates this. Given two single-trace
analysis results from different frameworks running the same model, it:

1. **Matches operations** across the two median transformer blocks one-by-one
2. **Identifies all performance differences** (both positive and negative) with
   root cause analysis grounded in source code
3. **Optionally generates an improvement plan** with step-by-step coding guides
   to port optimizations from the faster framework

This guide walks through the complete process using a concrete example:
[comparing vLLM (v0.19.2, May 13, 2026) against SGLang (May 5, 2026)](https://github.com/neuralmagic/ai_auto_analyze/tree/main/auto_analyze/examples/cross_framework_cmp_example_kimi)
running Kimi-K2.5-NVFP4 on 8xB200 GPUs in a low-latency pure-decode configuration
(BS=1, ISL=4, OSL=1024).

The analysis found that vLLM is **8.6% slower** per MoE decoder layer than SGLang
(97.9 us vs 90.1 us per block). The gap is driven primarily by a design difference
in how the MoE dual-stream pipeline is structured, partially offset by vLLM's
advantage in AllReduce+RMSNorm fusion.

```
Block wall time:  vLLM (TARGET) = 97,919 ns  |  SGLang = 90,144 ns  |  Delta = +7,775 ns (+8.6%)
```

**Key differences identified:**

| # | Root Cause | Impact | Framework |
|---|------------|-------:|-----------|
| 1 | MoE dual-stream design: vLLM puts routed MoE on critical path, SGLang puts shared expert | +7.2 us | SGLang faster |
| 2 | vLLM-only MoE overhead kernels (workspace zero-fill, routing bias copy) | +4.4 us | SGLang faster |
| 3 | Explicit Q concat+scale Triton kernel in vLLM | +1.7 us | SGLang faster |
| 4 | AllReduce+RMSNorm fusion (Lamport-synchronized) | -6.5 us | vLLM faster |

**Details on the largest difference:**

The biggest gap (+7.2 us) comes from a structural design decision in the MoE layer.
Both frameworks execute the shared expert and routed MoE pipelines in parallel on
two CUDA streams, but they differ in *which* pipeline runs on the critical path:

- **vLLM**: routed MoE on main stream (37.1 us including 10.7 us routing overhead),
  shared expert hidden on secondary stream (22.4 us)
- **SGLang**: shared expert on main stream (29.9 us), routed MoE hidden on alt
  stream (32.9 us — routing overhead is hidden)

SGLang achieves this by setting the shared expert to `tp_size=1` (no TP sharding),
making its GEMMs 8x larger per GPU and long enough to serve as a credible
main-stream workload that hides the routing-heavy routed MoE on the alternate stream.

**Details on the second largest difference — MoE overhead kernels (+4.4 us):**

vLLM launches two overhead kernels on its main stream before MoE expert computation
that SGLang does not require:

- **Workspace zero-fill** (+2.4 us) — vLLM's FlashInfer integration triggers an
  internal zero-fill of the MoE output accumulation buffer before expert computation.
  SGLang avoids this by using `torch.empty()` (no zeroing) and relying on the kernel
  to fully write all output elements.
- **Routing bias copy** (+2.0 us) — vLLM converts `e_score_correction_bias` to
  bfloat16 via `.to(torch.bfloat16)` on every forward pass, launching a copy kernel.
  SGLang handles the routing bias dtype internally within the routing kernel. This
  could be fixed by pre-converting the bias at model load time.

The analysis also generated an improvement plan showing that if vLLM adopted
SGLang's shared expert TP policy and eliminated the overhead kernels, it would be
~6.5% *faster* than SGLang (thanks to its AllReduce+RMSNorm fusion advantage).

## Prerequisites

- Two completed single-trace analyses (one per framework) with matching execution
  parameters (same model, GPU type, batch size, prefill size, output size)
- See [Single-Trace Analysis and High-Level Trace Annotation](single_trace_analysis_and_annotation_guide.md)
  for how to run single-trace analysis for each framework

## End-to-End Walkthrough

### Step 1: Run Single-Trace Analysis for Each Framework

Each framework needs its own trace capture and single-trace analysis. The execution
parameters must be identical so the comparison is meaningful.

```bash
# --- Shared parameters (must be identical for both frameworks) ---
MODEL="nvidia/Kimi-K2.5-NVFP4"
GPU_TYPE="B200"
BATCH_RANGE="1"
PREFILL_RANGE="4"
OUTPUT_RANGE="1024"
```

#### Framework A — vLLM

```bash
TRACE_FILE_A="/path/to/traces/vllm/trace.sqlite"
RUN_LOG_A="/path/to/traces/vllm/run_log.txt"
SOURCE_CODE_A="/path/to/vllm"
COMMIT_A="fe9c3d6c5"
ANALYZE_DIR_A="/path/to/results/analyze_vllm"
CONFIG_A="/path/to/results/single_trace_config_vllm"

python -m auto_analyze.create_single_trace_config \
    --model $MODEL \
    --gpu-type $GPU_TYPE \
    --batch-size-range $BATCH_RANGE \
    --prefill-size-range $PREFILL_RANGE \
    --output-size-range $OUTPUT_RANGE \
    --trace-file $TRACE_FILE_A \
    --run-log-file $RUN_LOG_A \
    --clean-source-code-path $SOURCE_CODE_A \
    --commit-id $COMMIT_A \
    --analyze-output-dir $ANALYZE_DIR_A \
    --output-config-file $CONFIG_A \
    --trace-gpu-focus 0

python -m auto_analyze.run_single_trace --config ${CONFIG_A}.json
```

#### Framework B — SGLang

```bash
TRACE_FILE_B="/path/to/traces/sgl/trace.sqlite"
RUN_LOG_B="/path/to/traces/sgl/run_log.txt"
SOURCE_CODE_B="/path/to/sglang"
COMMIT_B="612785ffdcaf35552f1ed433a981d596ca9fe900"
ANALYZE_DIR_B="/path/to/results/analyze_sgl"
CONFIG_B="/path/to/results/single_trace_config_sgl"

python -m auto_analyze.create_single_trace_config \
    --model $MODEL \
    --gpu-type $GPU_TYPE \
    --batch-size-range $BATCH_RANGE \
    --prefill-size-range $PREFILL_RANGE \
    --output-size-range $OUTPUT_RANGE \
    --trace-file $TRACE_FILE_B \
    --run-log-file $RUN_LOG_B \
    --clean-source-code-path $SOURCE_CODE_B \
    --commit-id $COMMIT_B \
    --analyze-output-dir $ANALYZE_DIR_B \
    --output-config-file $CONFIG_B \
    --trace-gpu-focus 0

python -m auto_analyze.run_single_trace --config ${CONFIG_B}.json
```

**Notes:**
- Each framework uses its **own source code directory** and commit ID (unlike
  cross-commit analysis which uses the same repo for both).
- `--trace-gpu-focus 0` is used here because NSYS traces contain all GPUs; for
  PyTorch traces (single-GPU), omit this flag.
- Both runs must use the same model, GPU type, and execution parameters.

### Step 2: Create the Cross-Trace Config

Once both single-trace analyses are complete, create the cross-trace config.
The `--target-trace-id` specifies which framework you want to optimize (index 0
below = vLLM):

```bash
CROSS_OUTPUT_DIR="/path/to/results/cross_trace_analysis"
CROSS_CONFIG="/path/to/results/cross_config"

python -m auto_analyze.create_cross_trace_config \
    --trace-result-dir $ANALYZE_DIR_A \
    --trace-result-dir $ANALYZE_DIR_B \
    --target-trace-id 0 \
    --analyze-output-dir $CROSS_OUTPUT_DIR \
    --output-config-file $CROSS_CONFIG \
    --make-improvement-plan
```

The `--make-improvement-plan` flag enables generation of a step-by-step coding
guide for porting optimizations from the faster framework (SGLang) to the target
(vLLM).

### Step 3: Run the Cross-Trace Analysis

```bash
python -m auto_analyze.run_cross_trace --config ${CROSS_CONFIG}.json
```

The analysis pipeline executes these steps automatically:

1. **Block matching** — Matches operations between the two median transformer
   blocks one-by-one based on high-level logical purpose, sequence order, and
   source code analysis across both frameworks.

2. **Performance comparison** — Analyzes every matched operation pair to identify
   all performance differences (both positive and negative) from the target
   framework's perspective, with root cause analysis grounded in both codebases.

3. **Improvement plan** *(when `make_improvement_plan` is enabled)* — Generates
   a ranked sequence of improvement proposals showing how to port code from the
   faster framework with minimal changes, each with step-by-step coding guides.

### Step 4: Review the Results

The output directory contains:

| File | Description |
|------|-------------|
| `cross_matching_blocks.txt` | Operation-by-operation matching table across frameworks |
| `cross_compare_blocks.txt` | Performance comparison with root causes and source references |
| `cross_improvement_plan.txt` | *(when enabled)* Improvement plan with coding guides |
| `run_params_cross.txt` | Cross-trace run parameters for downstream tools |

The comparison report (`cross_compare_blocks.txt`) includes:
- **Executive summary** — overall wall time delta, key findings, and per-category
  breakdown
- **Detailed differences** — sorted by impact, each with root cause analysis,
  code snippets from both frameworks, and source references

The improvement plan (`cross_improvement_plan.txt`) includes:
- **Ranked proposals** — from most impactful to least, each with estimated
  recovery in nanoseconds and percentage of block wall time
- **Step-by-step coding guides** — showing exactly what to change in the target
  framework, with code snippets ported from the faster framework

## Differences from Cross-Commit Analysis

While [cross-commit analysis](cross_commit_comparison_guide.md) compares different
versions of the *same* framework (same codebase, different commits), cross-framework
analysis compares *different* codebases running the same model:

| | Cross-Commit | Cross-Framework |
|--|-------------|----------------|
| Source code repos | Same repo, different commits | Different repos |
| Operation naming | Same high-level op names | Different naming conventions |
| Architecture | Same code paths, different versions | Different implementations |
| Improvement plan | Port code from older/newer version | Port code across frameworks |

The analysis mode is auto-detected from the traces — if the framework names match,
it's cross-commit; if they differ, it's cross-framework.
