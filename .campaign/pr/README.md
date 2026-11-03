# Two ROCm PRs — rebased, tested, ready to publish

Both are rebased onto current `main` (`866bccf083`) and apply to a clean
checkout of it individually and together, so they can be opened in either
order. The performance numbers were taken on `e0d00e4b8`, the build inside the
pinned `nightly-rocm100` image used for both arms of every comparison.

## Branches

Both are **pushed to the fork** (`alexm-redhat/vllm`) and, per the GitHub
compare API, each is `ahead_by 1, behind_by 0` of `vllm-project/vllm:main` — no
rebase or conflict at PR time.

| branch (fork + local `/home/alexm/code/vllm`) | commit | subject |
|---|---|---|
| `pr/aiter-small-batch-priority` | `bbbc5e2b33` | platforms/rocm: try AITER attention backends before ROCM_ATTN |
| `pr/fse-shared-experts-default` | `2c89f54d25` | rocm: enable AITER fused shared experts by default on MI300 |

Both commits are authored and signed off as Alexander Matveev
`<amatveev@redhat.com>`, so DCO is satisfied.

**Both PRs are scoped to MI300 (gfx942)** — the only architecture they were
measured on. Every other ROCm GPU keeps today's behaviour: PR 1 returns before
touching the backend order, and PR 2 treats the env var as unset (an explicit
`VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1` still opts in anywhere).

The scoping is deliberately conservative, not a claim that either change is
MI300-specific — there is no gfx942-only mechanism in either patch. The intent
is to relax both to all ROCm GPUs once there are measurements on MI350 and the
other parts; both bodies say so.

Pushing again after review comments, from the local clone:

```bash
cd /home/alexm/code/vllm
git checkout pr/aiter-small-batch-priority
# edit, commit --amend (keep the Signed-off-by) or add a fixup commit
git push origin pr/aiter-small-batch-priority
```

## Publishing upstream (not done yet, needs your go-ahead)

Opening the two PRs is the only remaining step and it is deliberately not
done. When you want it:

```bash
cd /home/alexm/code/vllm
gh pr create --repo vllm-project/vllm \
  --head alexm-redhat:pr/aiter-small-batch-priority \
  --base main --title "[ROCm][Perf] Try AITER attention backends before ROCM_ATTN (limited to MI300)" \
  --body-file /home/alexm/code/amalgamator/.campaign/pr/PR1_body.md

gh pr create --repo vllm-project/vllm \
  --head alexm-redhat:pr/fse-shared-experts-default \
  --base main --title "[ROCm] Turn on AITER fused shared experts by default on MI300" \
  --body-file /home/alexm/code/amalgamator/.campaign/pr/PR2_body.md
```

Bodies are `PR1_body.md` and `PR2_body.md`; the `.patch` files here are
reference copies. Raw benchmark output for every number is in `evidence/`.

Order matters only for the wording: PR 2's first table is measured on top of
PR 1, so open PR 1 first and reference it in PR 2 if both are in flight. Both
apply standalone to `main` as well (verified), so either can land first.

## Verification on the rebased trees

Unit tests were re-run against the current base `866bccf083` (not just the
earlier one): PR 1's 12 cases and PR 2's 10 cases both pass
(`~/qwen38/logs/pr_tests_rebased.log`). The container needs `--device=/dev/kfd
--device=/dev/dri` for PR 1's tests — they touch platform init, and without the
devices torch reports "No CUDA GPUs are available" and every case fails.

Also done on the GPU box, in the pinned ROCm 10 image, `max_num_seqs=1`,
Qwen3.5-35B-A3B-FP8, on the earlier rebase:

- each branch boots a real engine and reaches the code path it changes — PR 1
  logs `Trying AITER attention backends before ROCM_ATTN (head_size=256 ...)`,
  PR 2 logs `AITER fused shared experts are ENABLED for <shared_expert>`;
- a 2048/128 bench cell completes on each (`~/qwen38/logs/rebased_boot.log`).

That bench is a start-up smoke test only — one arm per branch, no paired
baseline, and the image's prebuilt extensions are from the older build — so its
TPOT is not quoted anywhere as a result. The numbers in the PR bodies are the
paired measurements on `76c6f3f2a9`.

## What each PR does

**PR 1** reorders the ROCm attention backend list so the AITER backends are
tried before `ROCM_ATTN` when `ROCM_ATTN` has no custom paged-attention kernel
for the model's head size (i.e. anything other than 64/128): `ROCM_AITER_FA`,
then `ROCM_AITER_UNIFIED_ATTN`, then `ROCM_ATTN`. Head sizes 64/128 keep the
current order, and so does every ROCm architecture other than gfx942.
`VLLM_DISABLE_AITER_BACKEND_PRIORITY=1` restores the old order.

**PR 2** makes `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS` default-on **on
gfx942**, so MoE shared experts run as an extra slot of the routed grouped GEMM
instead of a separate dense MLP. The per-layer compatibility resolver still
gates it on quantization and EP, the env var still works as a kill switch in
both directions, and the MegaMoE shared-memory sizing asks the same scoped
question instead of reading the raw env var.

## Measured numbers

Environment for every number: single MI300X (gfx942), ROCm 10 image pinned by
digest, `vllm bench serve`, random dataset, `--ignore-eos`, TP=1. Each arm is a
separate git worktree with the same image and flags, so the patch is the only
difference between arms.

Base commit for all of these: vLLM `76c6f3f2a9`, the 2026-11-02 nightly, each
tree source-mounted into the image.
Two runs per cell. PR 1, 8192 in (c1/c8 at `--max-num-seqs 8`, 1024 out; c64 at
`--max-num-seqs 64`, 512 out, 128 prompts):

| model | cell | before | after |
|---|---|---|---|
| Qwen3.5-35B-A3B-FP8 | c1 | 14.810 / 14.810 ms | 4.644 / 4.611 ms (−68.8%) |
| Qwen3.5-35B-A3B-FP8 | c8 | 18.188 / 18.047 ms | 7.866 / 7.666 ms (−57.1%) |
| Qwen3.5-35B-A3B-FP8 | c64 | 49.24 / 47.85 ms | 34.72 / 33.84 ms (−29.4%) |
| Qwen3.8-27B-FP8 | c1 | 27.46 / 27.42 ms | 11.16 / 11.18 ms (−59.4%) |

PR 2 (see `PR2_body.md` for the full tables): −3.2% at c1 and −2.5% at c8 on top
of PR 1; −1.0% / −1.7% on stock.

Accuracy (gsm8k, 5-shot, full split, two runs per arm): PR 1 flexible-extract
0.7904 mean both sides (0.0015 apart, within the 0.011 spread); PR 2 0.7904
baseline vs 0.7851 patched (0.005 apart, within a 0.008 spread). No measurable
accuracy change in either PR.

## Working on review comments

Keep the branches as the review surface:

```bash
git checkout pr/aiter-small-batch-priority
# edit, then:
git commit --amend --no-edit --signoff
git push --force-with-lease fork pr/aiter-small-batch-priority
```

`--force-with-lease` rather than `--force` so a stale local view cannot clobber
a reviewer's rebase.

## Why these two

The residual analysis (vLLM vs SGLang on the same cell, after the attention fix
is applied) ranks enabling fused shared experts first. The attention ordering
change is what makes the second measurement visible at all: it removes a 10 ms
per-step attention cost that otherwise hides a 200 µs improvement.

## Evidence files

`evidence/` holds the raw `vllm bench serve` JSON for every number quoted in
the two bodies: `rb2_*` (the four-tree run on the `76c6f3f2a9` base, both PRs),
`x_q355_c64_*` (the c64 cell), `x_q38_c1_*` / `q38_base_*` (the Qwen3.8 cell),
`pr1f_*` / `pr1c64_*` (the earlier base-patch measurements), `pr3_*` (the
paired AITER FA vs unified comparison), and `pr2_*` (the earlier PR 2
comparison). `backend_selection.txt` records which attention
backend each arm actually selected at startup (stock `ROCM_ATTN`, PR 1
`ROCM_AITER_FA`). gsm8k output is in the logs on the box
(`~/qwen38/logs/pr1_lm_eval_fa*.log`, `pr_lm_eval_queue.log`).

`tools/` keeps the small scripts used to produce the patches, for provenance;
they are not part of either PR.
