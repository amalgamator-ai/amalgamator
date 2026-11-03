#!/usr/bin/env python3
"""Tidy the P1 change for review.

The behaviour is unchanged; the shape is what a reviewer expects:
  * the kill switch is declared in vllm/envs.py instead of read with
    os.environ inside the platform module;
  * the decision moves into a named helper with a docstring, instead of ~25
    lines inline with repeated `not prefer` guards;
  * the explanatory comments say why, not how fast.

usage: rewrite_p1_code.py <vllm-worktree>
"""

import pathlib
import sys

WT = pathlib.Path(sys.argv[1])

OLD_BLOCK_START = "    backends = []\n\n    # ── AITER unified small-batch auto-selection ──"
NEW_BLOCK = '''    backends = []

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
    backends.append(AttentionBackendEnum.TRITON_ATTN)
    backends.append(AttentionBackendEnum.TURBOQUANT)
    backends.append(AttentionBackendEnum.ROCM_OCTAVE)
    backends.append(AttentionBackendEnum.ULTRAQUANT)

    return backends
'''

HELPER = '''_AITER_UNIFIED_SMALL_BATCH_MAX_SEQS = 8


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

ENVS_SCHEMA = (
    "    VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION: bool = False",
    "    VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION: bool = False\n"
    "    VLLM_DISABLE_AITER_UNIFIED_SMALL_BATCH: bool = False",
)

ENVS_GETTER = (
    '    "VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION": lambda: (\n'
    '        os.getenv("VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION", "False").lower()\n'
    '        in ("true", "1")\n'
    '    ),',
    '    "VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION": lambda: (\n'
    '        os.getenv("VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION", "False").lower()\n'
    '        in ("true", "1")\n'
    '    ),\n'
    '    # Disable preferring AITER unified attention for small-batch decode\n'
    '    # on ROCm, restoring the previous ROCM_ATTN-first ordering.\n'
    '    "VLLM_DISABLE_AITER_UNIFIED_SMALL_BATCH": lambda: (\n'
    '        os.getenv("VLLM_DISABLE_AITER_UNIFIED_SMALL_BATCH", "False").lower()\n'
    '        in ("true", "1")\n'
    '    ),',
)


def _replace_once(path: pathlib.Path, old: str, new: str) -> None:
    text = path.read_text()
    if text.count(old) != 1:
        raise SystemExit(f"{path}: expected one occurrence of {old[:80]!r}")
    path.write_text(text.replace(old, new))


def main() -> None:
    rocm = WT / "vllm/platforms/rocm.py"
    text = rocm.read_text()

    start = text.index(OLD_BLOCK_START)
    end_marker = "    return backends\n"
    end = text.index(end_marker, start) + len(end_marker)
    text = text[:start] + NEW_BLOCK + text[end:]
    rocm.write_text(text)

    # Insert the helper just before _get_backend_priorities.
    _replace_once(
        rocm,
        "def _get_backend_priorities(",
        HELPER + "def _get_backend_priorities(",
    )

    _replace_once(WT / "vllm/envs.py", *ENVS_SCHEMA)
    _replace_once(WT / "vllm/envs.py", *ENVS_GETTER)
    print("rewrote platforms/rocm.py and envs.py")


if __name__ == "__main__":
    main()
