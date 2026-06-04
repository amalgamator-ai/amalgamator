# Cross-Commit Performance Comparison

## Motivation

When a framework like vLLM ships a new version, performance can change in both
directions — some kernels get faster through new fusions or specialized kernels,
while others may regress due to library updates or code path changes. Identifying
exactly which operations improved and which regressed requires comparing GPU traces
from two commits kernel-by-kernel, which is extremely tedious to do manually.

The cross-commit analysis pipeline automates this. Given two single-trace analysis
results from different commits of the same framework, it:

1. **Matches operations** across the two median transformer blocks one-by-one
2. **Identifies all performance differences** (both positive and negative) with
   root cause analysis grounded in source code
3. **Optionally generates an improvement plan** with step-by-step coding guides
   to recover any regressions

This guide walks through the complete process using a concrete example: [comparing
vLLM `v0.16.0` (Feb 12, 2026) against the latest `main` branch (May 12, 2026)](https://github.com/neuralmagic/ai_auto_analyze/tree/main/auto_analyze/examples/cross_commit_cmp_example_kimi) running Kimi-K2.5-NVFP4 on B200
GPUs in a low-latency pure-decode configuration (BS=1, ISL=4, OSL=1024).

The analysis found that `main` is **21.3% faster** per MoE decoder layer than
`v0.16.0` (96.9 us vs 123.2 us per block). The improvement comes primarily from
three purpose-built CUDA kernels that replaced generic cuBLASLt GEMMs and fused
multiple PyTorch-dispatched operations into single monolithic kernels. These gains
are partially offset by new buffer initialization overhead and a cuBLAS library
regression.

```
Block wall time:  TARGET = 96,992 ns  |  Baseline = 123,232 ns  |  Delta = -26,240 ns (-21.3%)
```

**Improvements:**

| # | Root Cause | Impact | Origin/PR/Commit |
|---|------------|-------:|------------------|
| 1 | QKV-A projection GEMM → `dsv3_fused_a_gemm` | 41% | [#34758](https://github.com/vllm-project/vllm/pull/34758) |
| 2 | MoE gate routing GEMM → `dsv3_router_gemm` | 30% | [#34302](https://github.com/vllm-project/vllm/pull/34302) |
| 3 | MoE routing: 4 kernels → 1 monolithic TRT-LLM kernel | 29% | FlashInfer 0.6.3→0.6.8 upgrade |

**Improvement details:**

1. **QKV-A Projection GEMM** (saves 10.8 us) — The generic cuBLASLt CUTLASS BF16
   GEMM (`nvjet_tst_...TNN`, 18.1 us) was replaced with `dsv3_fused_a_gemm`
   (7.3 us), a hand-tuned warp-specialized CUDA kernel adapted from TRT-LLM's
   DeepSeek V3 minimum-latency kernels. It uses compile-time shape specialization
   for [M<=16, K=7168, N=2112], dual tile-N variants for optimal small-batch
   coverage, and PDL for kernel pipelining.

2. **MoE Gate Router GEMM** (saves 8.0 us) — The generic cuBLASLt GEMM
   (`nvjet_tst_...TNN`, 10.9 us) was replaced with `dsv3_router_gemm` (2.9 us),
   a one-block-per-expert kernel (grid=384) with compile-time template
   specialization for the extremely skinny [M<=16, N=384, K=7168] shape. Uses
   warp-level shuffle reduction instead of shared memory.

3. **MoE Routing Dispatch Consolidation** (saves 7.6 us) — Four separate
   PyTorch-dispatched kernels (sigmoid scoring, bias correction, top-k expert
   selection, index clustering — totaling 15.3 us) were consolidated into a single
   monolithic TRT-LLM `routingIndicesBlockKernel` (7.6 us). Data stays in shared
   memory between routing stages, eliminating 3 kernel launch overheads and global
   memory round-trips. Enabled by FlashInfer upgrade and vLLM backend selection
   routing to `Fp8MoeBackend.FLASHINFER_TRTLLM` on SM100.

**Regressions:**

| # | Root Cause | Impact | Origin/PR/Commit |
|---|------------|-------:|------------------|
| 1 | New buffer-init and dtype-cast ops for TRT-LLM MoE | +3.3 us | FlashInfer 0.6.3→0.6.8 upgrade TRT-LLM MoE pipeline |
| 2 | New shared expert buffer-init ops on aux stream | +2.8 us | FlashInfer 0.6.3→0.6.8 upgrade CUTLASS FP4 kernel |
| 3 | cuBLASLt nvjet GEMM heuristic change | +1.8 us | CUDA 13.0.2 cuBLASLt codegen |
| 4 | Post-MoE allreduce variance | +1.0 us | FlashInfer 0.6.3→0.6.8 upgrade unified allreduce API |

**Regression details (selected):**

1. **New TRT-LLM MoE buffer-init ops** (+3.3 us) — The monolithic TRT-LLM MoE
   kernel (which saved 7.6 us above) requires two new setup operations that didn't
   exist in the old non-monolithic path: `moe_route_buffer_init` (1.6 us) to
   zero-initialize routing workspace arrays, and `moe_input_bf16_cast` (1.7 us)
   to convert hidden states for the TRT-LLM routing kernel format. These could
   potentially be eliminated by pre-allocating persistent buffers or fusing the
   memset into the routing kernel's prologue.

2. **New shared expert buffer-init ops** (+2.8 us) — Two `FillFunctor` memset
   operations appeared on the auxiliary shared-expert CUDA stream, initializing
   output buffers for the CUTLASS FP4 GEMMs: `shared_expert_gateup_buf_init`
   (1.6 us) and `shared_expert_down_buf_init` (1.2 us). In v0.16.0, the shared
   expert path used pre-warmed memory regions. The FlashInfer 0.6.8 CUTLASS FP4
   kernel requires explicit buffer initialization for its epilogue. Since these
   ops run on the aux stream concurrently with MoE routing on the main stream,
   their wall-time impact is minimal — but they are visible in the trace.

The full example with detailed source code analysis is at
[cross_commit_cmp_example_kimi](https://github.com/neuralmagic/ai_auto_analyze/tree/main/auto_analyze/examples/cross_commit_cmp_example_kimi).

## Prerequisites

- Two completed single-trace analyses (one per commit) with matching execution
  parameters (same model, GPU type, batch size, prefill size, output size)
- See [Single-Trace Analysis and High-Level Trace Annotation](single_trace_analysis_and_annotation_guide.md)
  for how to run single-trace analysis for each commit

## End-to-End Walkthrough

### Step 1: Run Single-Trace Analysis for Each Commit

Each commit needs its own trace capture and single-trace analysis. The execution
parameters must be identical across both runs so the comparison is meaningful.

```bash
# --- Shared parameters (must be identical for both commits) ---
MODEL="nvidia/Kimi-K2.5-NVFP4"
GPU_TYPE="B200"
BATCH_RANGE="1"
PREFILL_RANGE="4"
OUTPUT_RANGE="1024"
SOURCE_CODE="/path/to/vllm"
```

#### Commit A — latest main

```bash
COMMIT_A="8f89381fc6b2d54591a7a560e20ee5211ce1ac33"
TRACE_FILE_A="/path/to/traces/commit_a/rank0.pt.trace.json.gz"
RUN_LOG_A="/path/to/traces/commit_a/run_log.txt"
ANALYZE_DIR_A="/path/to/results/vllm_latest/analyze"
CONFIG_A="/path/to/results/vllm_latest/single_trace_config"

python -m auto_analyze.create_single_trace_config \
    --model $MODEL \
    --gpu-type $GPU_TYPE \
    --batch-size-range $BATCH_RANGE \
    --prefill-size-range $PREFILL_RANGE \
    --output-size-range $OUTPUT_RANGE \
    --trace-file $TRACE_FILE_A \
    --run-log-file $RUN_LOG_A \
    --clean-source-code-path $SOURCE_CODE \
    --commit-id $COMMIT_A \
    --analyze-output-dir $ANALYZE_DIR_A \
    --output-config-file $CONFIG_A

python -m auto_analyze.run_single_trace --config ${CONFIG_A}.json
```

#### Commit B — v0.16.0 release

```bash
COMMIT_B="89a77b10846fd96273cce78d86d2556ea582d26e"
TRACE_FILE_B="/path/to/traces/commit_b/rank0.pt.trace.json.gz"
RUN_LOG_B="/path/to/traces/commit_b/run_log.txt"
ANALYZE_DIR_B="/path/to/results/vllm_v0.16.0/analyze"
CONFIG_B="/path/to/results/vllm_v0.16.0/single_trace_config"

python -m auto_analyze.create_single_trace_config \
    --model $MODEL \
    --gpu-type $GPU_TYPE \
    --batch-size-range $BATCH_RANGE \
    --prefill-size-range $PREFILL_RANGE \
    --output-size-range $OUTPUT_RANGE \
    --trace-file $TRACE_FILE_B \
    --run-log-file $RUN_LOG_B \
    --clean-source-code-path $SOURCE_CODE \
    --commit-id $COMMIT_B \
    --analyze-output-dir $ANALYZE_DIR_B \
    --output-config-file $CONFIG_B

python -m auto_analyze.run_single_trace --config ${CONFIG_B}.json
```

**Notes:**
- Both commits use the **same source code directory** (`$SOURCE_CODE`). The
  analysis automatically creates separate branches for each commit ID.
- The source code directory must be a clean git repo with no uncommitted changes.
- Both runs must use the same model, GPU type, and execution parameters — the
  cross-trace validation will reject mismatched configurations.

### Step 2: Create the Cross-Trace Config

Once both single-trace analyses are complete, create the cross-trace config.
The `--target-trace-id` specifies which trace you want to analyze from — typically
the newer commit (index 0 below):

```bash
CROSS_OUTPUT_DIR="/path/to/results/cross_analyze"
CROSS_CONFIG="/path/to/results/cross_config"

python -m auto_analyze.create_cross_trace_config \
    --trace-result-dir $ANALYZE_DIR_A \
    --trace-result-dir $ANALYZE_DIR_B \
    --target-trace-id 0 \
    --analyze-output-dir $CROSS_OUTPUT_DIR \
    --output-config-file $CROSS_CONFIG
```

The script validates that both result directories contain the required files
(`run_params.txt`, `median_block.txt`, `transformer_block_high_level_ops.txt`).

**Optional — include an improvement plan** with step-by-step coding guides for
recovering any regressions in the target trace:

```bash
python -m auto_analyze.create_cross_trace_config \
    --trace-result-dir $ANALYZE_DIR_A \
    --trace-result-dir $ANALYZE_DIR_B \
    --target-trace-id 0 \
    --analyze-output-dir $CROSS_OUTPUT_DIR \
    --output-config-file $CROSS_CONFIG \
    --make-improvement-plan
```

### Step 3: Run the Cross-Trace Analysis

```bash
python -m auto_analyze.run_cross_trace --config ${CROSS_CONFIG}.json
```

The analysis pipeline executes these steps automatically:

1. **Block matching** — Matches operations between the two median transformer
   blocks one-by-one based on high-level logical purpose, sequence order, and
   source code analysis. Handles differences in fusion, kernel naming, and
   stream layout.

2. **Performance comparison** — Analyzes every matched operation pair to identify
   all performance differences (both positive and negative) from the target
   trace's perspective. For each difference, provides root cause analysis with
   source code references and code snippets.

3. **Improvement plan** *(optional, when `make_improvement_plan` is enabled)* —
   Generates a ranked sequence of improvement proposals for recovering regressions,
   each with step-by-step coding guides showing exactly what to change and where.

### Step 4: Review the Results

The output directory contains:

| File | Description |
|------|-------------|
| `cross_matching_blocks.txt` | Operation-by-operation matching table showing each logical operation with kernel names, durations, and streams side by side |
| `cross_compare_blocks.txt` | Detailed comparison of all performance differences (positive and negative) with root causes, code snippets, and source references |
| `cross_improvement_plan.txt` | *(only when enabled)* Ranked improvement proposals with step-by-step coding guides |
| `run_params_cross.txt` | Cross-trace run parameters for downstream tools |

The comparison report (`cross_compare_blocks.txt`) includes:
- **Executive summary** — overall wall time delta, key findings, and gross
  improvement vs. regression breakdown
- **Detailed differences** — sorted from worst regression to best improvement,
  each with root cause analysis, code snippets, and source references
