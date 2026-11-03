#!/usr/bin/env python3
"""P1, second attempt: prefer the AITER backends ahead of ROCM_ATTN at small batch.

Measured on Qwen3.5-35B (MI300X, 8K/1K, max_num_seqs=8):
    ROCM_ATTN            14.77 ms
    AITER unified         5.02 ms
    ROCM_AITER_FA         4.54 ms   <- best, and it supports head_dim=256
and at c8: ROCM_ATTN 18.78, unified 10.55, FA 8.15.

So the ordering to try first is FA, then unified, then ROCM_ATTN. Implemented by
reordering the entries the function already produced (no duplicate entries, all
fallbacks preserved) rather than by adding appends in several places.

usage: rewrite_p1_v2.py <vllm-worktree>
"""

import pathlib
import sys

WT = pathlib.Path(sys.argv[1])

OLD_HEAD = """    backends = []

    # ROCM_ATTN's custom paged-attention kernel only covers head_size 64/128
    # on CDNA, so models with a larger head_dim fall back to the generic
    # Triton kernel at every batch size. AITER unified covers those head
    # sizes and is faster at low concurrency, but slower at high concurrency
    # where ROCM_ATTN amortises better. Prefer it only for small-batch
    # engines.
    prefer_aiter_unified = _prefer_aiter_unified_small_batch(
        is_aiter_found_and_supported())

    if prefer_aiter_unified:
        backends.append(AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN)

    # Keep ROCM_ATTN disabled for KV connectors until connector transfer
    # semantics are validated for its asymmetric native K/V cache views.
    if not use_kv_connector:
        backends.append(AttentionBackendEnum.ROCM_ATTN)
    if rocm_aiter_ops.is_mha_enabled():
        backends.append(AttentionBackendEnum.ROCM_AITER_FA)
    if not prefer_aiter_unified and is_aiter_found_and_supported():
        backends.append(AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN)
    if rocm_aiter_ops.is_rdna_aiter_enabled():
        if on_rdna4():
            if not prefer_aiter_unified:
                backends.insert(0, AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN)
        elif not prefer_aiter_unified:
            backends.append(AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN)
"""

NEW_HEAD = """    backends = []

    # Keep ROCM_ATTN disabled for KV connectors until connector transfer
    # semantics are validated for its asymmetric native K/V cache views.
    if not use_kv_connector:
        backends.append(AttentionBackendEnum.ROCM_ATTN)
    if rocm_aiter_ops.is_mha_enabled():
        backends.append(AttentionBackendEnum.ROCM_AITER_FA)
    if is_aiter_found_and_supported():
        backends.append(AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN)
    if rocm_aiter_ops.is_rdna_aiter_enabled():
        if on_rdna4():
            backends.insert(0, AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN)
        else:
            backends.append(AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN)

    # ROCM_ATTN's custom paged-attention kernel only covers head_size 64/128 on
    # CDNA, so models with a larger head_dim fall back to the generic Triton
    # kernel at every batch size. The AITER backends cover those head sizes and
    # are much faster for small-batch decode, but lose at high concurrency
    # where ROCM_ATTN amortises better, so the preference is gated on the
    # configured batch size.
    if _prefer_aiter_small_batch():
        aiter_first = [
            backend
            for backend in (
                AttentionBackendEnum.ROCM_AITER_FA,
                AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN,
            )
            if backend in backends
        ]
        backends = aiter_first + [
            backend for backend in backends if backend not in aiter_first
        ]
"""

OLD_HELPER = '''_AITER_UNIFIED_SMALL_BATCH_MAX_SEQS = 8


def _prefer_aiter_unified_small_batch(aiter_available: bool) -> bool:
    """Whether to try AITER unified attention ahead of ROCM_ATTN.

    True only when AITER is available, the engine is configured for a small
    batch, and VLLM_DISABLE_AITER_UNIFIED_SMALL_BATCH is not set. With no
    vLLM config (e.g. backend selection outside a running engine) the
    previous ordering is kept.
    """
    if not aiter_available or envs.VLLM_DISABLE_AITER_UNIFIED_SMALL_BATCH:
        return False

    from vllm.config import get_current_vllm_config_or_none

    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        return False
    max_num_seqs = vllm_config.scheduler_config.max_num_seqs
    if max_num_seqs > _AITER_UNIFIED_SMALL_BATCH_MAX_SEQS:
        return False

    logger.info_once(
        "Preferring AITER unified attention for small batches "
        "(max_num_seqs=%d <= %d); set "
        "VLLM_DISABLE_AITER_UNIFIED_SMALL_BATCH=1 to disable.",
        max_num_seqs,
        _AITER_UNIFIED_SMALL_BATCH_MAX_SEQS,
    )
    return True
'''

NEW_HELPER = '''_AITER_SMALL_BATCH_MAX_SEQS = 8


def _prefer_aiter_small_batch() -> bool:
    """Whether to try the AITER attention backends ahead of ROCM_ATTN.

    True when the engine is configured for a small batch and
    VLLM_DISABLE_AITER_SMALL_BATCH_PRIORITY is not set. Without a vLLM config
    (backend selection outside a running engine) the previous order is kept.
    """
    if envs.VLLM_DISABLE_AITER_SMALL_BATCH_PRIORITY:
        return False

    from vllm.config import get_current_vllm_config_or_none

    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        return False
    max_num_seqs = vllm_config.scheduler_config.max_num_seqs
    if max_num_seqs > _AITER_SMALL_BATCH_MAX_SEQS:
        return False

    logger.info_once(
        "Trying AITER attention backends before ROCM_ATTN for small batches "
        "(max_num_seqs=%d <= %d); set "
        "VLLM_DISABLE_AITER_SMALL_BATCH_PRIORITY=1 to keep the old order.",
        max_num_seqs,
        _AITER_SMALL_BATCH_MAX_SEQS,
    )
    return True
'''

ENVS = [
    (
        "    VLLM_DISABLE_AITER_UNIFIED_SMALL_BATCH: bool = False",
        "    VLLM_DISABLE_AITER_SMALL_BATCH_PRIORITY: bool = False",
    ),
    (
        '    # Disable preferring AITER unified attention for small-batch decode\n'
        '    # on ROCm, restoring the previous ROCM_ATTN-first ordering.\n'
        '    "VLLM_DISABLE_AITER_UNIFIED_SMALL_BATCH": lambda: (\n'
        '        os.getenv("VLLM_DISABLE_AITER_UNIFIED_SMALL_BATCH", "False").lower()\n'
        '        in ("true", "1")\n'
        '    ),',
        '    # Keep the previous ROCM_ATTN-first ordering instead of trying the\n'
        '    # AITER attention backends first for small-batch decode on ROCm.\n'
        '    "VLLM_DISABLE_AITER_SMALL_BATCH_PRIORITY": lambda: (\n'
        '        os.getenv("VLLM_DISABLE_AITER_SMALL_BATCH_PRIORITY", "False").lower()\n'
        '        in ("true", "1")\n'
        '    ),',
    ),
]


def _sub(path: pathlib.Path, old: str, new: str) -> None:
    text = path.read_text()
    if text.count(old) != 1:
        raise SystemExit(f"{path}: expected one occurrence of {old[:70]!r}")
    path.write_text(text.replace(old, new))


def main() -> None:
    rocm = WT / "vllm/platforms/rocm.py"
    _sub(rocm, OLD_HEAD, NEW_HEAD)
    _sub(rocm, OLD_HELPER, NEW_HELPER)
    for old, new in ENVS:
        _sub(WT / "vllm/envs.py", old, new)
    print("rewrote for FA-first ordering")


if __name__ == "__main__":
    main()
