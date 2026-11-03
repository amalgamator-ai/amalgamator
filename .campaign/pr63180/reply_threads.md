# Per-thread replies (optional — the main reply covers everything)

Use these only if you prefer to answer each inline thread instead of one
consolidated comment. Keys are `path:line` from the review.

**platforms/rocm.py:1139** — `VLLM_DISABLE_PREFILL_FUSION_ABOVE_MONO`
> Moved to `vllm/envs.py` in the next push.

**platforms/rocm.py:1121** — "turn `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS` default on"
> Agreed, and the guardrails you link are enough to justify it. I have that
> exact flip prepared as its own change; I'd rather land it separately so this
> PR stays about the heterogeneous fold, but I'm happy to open it right away if
> you want it in this series.

**platforms/rocm.py:1190** — "temporary, eventually globally enabled"
> Agreed. It is a window gate for now, and the intent is to drop it once the
> above-mono path is the default everywhere.

**models/deepseek_v41/quant_config.py:175** and **models/deepseek_v4/quant_config.py:171** — use `quant_utils.py`
> Switched to the helper in both files.

**models/deepseek_v4/amd/model.py:257** — env var to `envs.py`
> Moved.

**models/deepseek_v4/amd/model.py:265** — "not sure what armed means"
> Renamed to "engaged" — it means the fold is active for the current forward
> (prefill-shaped batch, above mono decode).

**models/deepseek_v4/amd/model.py:452** — logging
> Now a single short `info_once`.

**models/deepseek_v4/amd/model.py:647** — `torch.cuda.is_available()`
> Removed, redundant.

**models/deepseek_v4/amd/model.py:673** — duplicate warnings
> Same fix — one `info_once`, no warnings.

**models/deepseek_v4/amd/model.py:942** — dual-stream abstraction
> Agreed and out of scope here; noted as a follow-up on top of
> `runner/shared_experts.py`.

**model_executor/layers/fused_moe/router/fused_topk_bias_router.py:37** — `_DECODE_MAX_QUERY_LEN = 8`
> This is the prefill-vs-decode gate for the fold: the fold is appended only on
> prefill-shaped batches, so decode steps keep the stock 384/6 kernels. The
> value is a margin above `1 + num_speculative_tokens` (6 at DSpark k=5), which
> is why it is 8 and not 1.

**envs.py:313** — `VLLM_DISABLE_FUSED_SHARED_EXPERTS`
> Agreed, it is redundant with `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0`.
> I'll gate the fold on the existing variable and re-run the ON/OFF A/B to
> confirm the two arms stay identical.
