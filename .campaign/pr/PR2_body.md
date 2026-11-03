# [ROCm] Turn on AITER fused shared experts by default on MI300

MoE models with a shared expert run that expert as a separate dense MLP next to
the routed MoE kernel: gate/up GEMM, activation, down GEMM, the shared-expert
gate GEMV, sigmoid, and a multiply, every layer.

AITER can run the same weights as one extra slot of the routed grouped GEMM,
with the gate folded into the kernel. vLLM already has that path; it is just
off by default. SGLang uses it by default for these models.

So this flips the default **on MI300 (gfx942)**, the only architecture the
change was measured on. Every other ROCm GPU keeps today's default unless
`VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS` is set explicitly, so the variable
still works as a kill switch in both directions.

The scoping is deliberately conservative: it is where we have numbers today, not
a claim that the fusion only pays off on MI300. There is nothing gfx942-specific
in the change, and we expect to relax it to all ROCm GPUs once we have
measurements on MI350 and the other parts.

The default only decides whether the fusion is *requested*:
`resolve_layer_fused_shared_expert` still checks quantization compatibility and
expert parallelism per layer, so checkpoints that cannot use it behave exactly
as before.

The resolver now also logs the enabled decision once. Previously only the skip
path said anything, so a log without a warning could mean "compatible but not
requested" or "requested and working" — you couldn't tell them apart.

## Numbers

Qwen3.5-35B-A3B-FP8, single MI300X, 8192 in / 1024 out, ROCm 10 image pinned by
digest. Each arm is a separate tree, so the patch is the only difference.
Measurement base is vLLM `76c6f3f2a9` (the 2026-11-02 nightly); two runs per
cell.

On top of the attention backend reordering from the companion PR
("try AITER attention backends before ROCM_ATTN"):

| | baseline | this PR |
|---|---|---|
| c1 | 4.644 / 4.611 ms | 4.478 / 4.478 ms (−3.2%) |
| c8 | 7.866 / 7.666 ms | 7.638 / 7.499 ms (−2.5%) |

With no other patches, on the same base:

| | baseline | this PR |
|---|---|---|
| c1 | 14.810 / 14.810 ms | 14.658 / 14.659 ms (−1.0%) |
| c8 | 18.188 / 18.047 ms | 17.909 / 17.717 ms (−1.7%) |

The saving is 150–300 µs per step. It is a bigger share of the step once the
attention change shortens it, which is why both are listed.

gsm8k, 5-shot, full split (1329 samples), two runs per arm, both arms on the
same attention backend so the comparison isolates this change. Flexible-extract:
baseline 0.7938 / 0.7870, this PR 0.7809 / 0.7892. The two runs of the same arm
differ by 0.007-0.008 absolute, which is larger than the 0.005 difference
between the arms, so there is no measurable accuracy change. Strict-match:
0.7748 / 0.7627 vs 0.7650 / 0.7703.

## Tests

`tests/model_executor/layers/test_fused_shared_expert_default.py`, ten cases:
the default is on for gfx942, other architectures keep the old default, an
explicit env value wins on any architecture, `0` / `false` still turn it off,
fused MoE being off disables it, the resolver still refuses an incompatible
quantization even with the default on, the EP opt-out still wins, and a
compatible checkpoint resolves to true.
