#!/usr/bin/env python3
"""P1, third and final shape: gate on the kernel, not on the batch size.

The batch gate was chosen when the only alternative measured was AITER unified,
which does lose at high concurrency. FA does not: at c64 on Qwen3.5 it gives
35.14 ms against ROCM_ATTN's 49.59 ms. So the real condition is the one that
explains the whole effect — ROCM_ATTN's custom paged-attention kernel is only
eligible for head_size 64 and 128 on CDNA, and outside that it runs the generic
Triton kernel, which is what loses everywhere.

usage: rewrite_p1_v3.py <vllm-worktree>
"""

import pathlib
import sys

WT = pathlib.Path(sys.argv[1])

OLD_HELPER = '''_AITER_SMALL_BATCH_MAX_SEQS = 8


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

NEW_HELPER = '''# Head sizes for which ROCM_ATTN runs its custom paged-attention kernel on
# CDNA (see use_rocm_custom_paged_attention). Outside this set ROCM_ATTN falls
# back to a generic Triton kernel.
_ROCM_ATTN_CUSTOM_HEAD_SIZES = (64, 128)


def _prefer_aiter_backends(head_size: int | None) -> bool:
    """Whether to try the AITER backends ahead of ROCM_ATTN.

    ROCM_ATTN is only competitive when it can use its custom paged-attention
    kernel; otherwise the AITER backends are faster at every concurrency
    measured. Disabled with VLLM_DISABLE_AITER_BACKEND_PRIORITY=1.
    """
    if envs.VLLM_DISABLE_AITER_BACKEND_PRIORITY:
        return False
    if head_size in _ROCM_ATTN_CUSTOM_HEAD_SIZES:
        return False

    logger.info_once(
        "Trying AITER attention backends before ROCM_ATTN "
        "(head_size=%s has no ROCm custom paged-attention kernel); set "
        "VLLM_DISABLE_AITER_BACKEND_PRIORITY=1 to keep the old order.",
        head_size,
    )
    return True
'''

REPLACEMENTS = [
    (
        WT / "vllm/platforms/rocm.py",
        """def _get_backend_priorities(
    use_mla: bool,
    use_sparse: bool,
    use_kv_connector: bool = False,
) -> list[AttentionBackendEnum]:""",
        """def _get_backend_priorities(
    use_mla: bool,
    use_sparse: bool,
    use_kv_connector: bool = False,
    head_size: int | None = None,
) -> list[AttentionBackendEnum]:""",
    ),
    (
        WT / "vllm/platforms/rocm.py",
        """    # ROCM_ATTN's custom paged-attention kernel only covers head_size 64/128 on
    # CDNA, so models with a larger head_dim fall back to the generic Triton
    # kernel at every batch size. The AITER backends cover those head sizes and
    # are much faster for small-batch decode, but lose at high concurrency
    # where ROCM_ATTN amortises better, so the preference is gated on the
    # configured batch size.
    if _prefer_aiter_small_batch():""",
        """    # Try the AITER backends before ROCM_ATTN when ROCM_ATTN would run its
    # generic Triton kernel instead of the CDNA custom one.
    if _prefer_aiter_backends(head_size):""",
    ),
    (
        WT / "vllm/platforms/rocm.py",
        """        backend_priorities = _get_backend_priorities(
            attn_selector_config.use_mla,
            attn_selector_config.use_sparse,
            attn_selector_config.use_kv_connector,
        )""",
        """        backend_priorities = _get_backend_priorities(
            attn_selector_config.use_mla,
            attn_selector_config.use_sparse,
            attn_selector_config.use_kv_connector,
            attn_selector_config.head_size,
        )""",
    ),
    (
        WT / "vllm/envs.py",
        "    VLLM_DISABLE_AITER_SMALL_BATCH_PRIORITY: bool = False",
        "    VLLM_DISABLE_AITER_BACKEND_PRIORITY: bool = False",
    ),
    (
        WT / "vllm/envs.py",
        '    # Keep the previous ROCM_ATTN-first ordering instead of trying the\n'
        '    # AITER attention backends first for small-batch decode on ROCm.\n'
        '    "VLLM_DISABLE_AITER_SMALL_BATCH_PRIORITY": lambda: (\n'
        '        os.getenv("VLLM_DISABLE_AITER_SMALL_BATCH_PRIORITY", "False").lower()\n'
        '        in ("true", "1")\n'
        '    ),',
        '    # Keep the previous ROCM_ATTN-first ordering instead of trying the\n'
        '    # AITER attention backends first on ROCm.\n'
        '    "VLLM_DISABLE_AITER_BACKEND_PRIORITY": lambda: (\n'
        '        os.getenv("VLLM_DISABLE_AITER_BACKEND_PRIORITY", "False").lower()\n'
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
    _sub(WT / "vllm/platforms/rocm.py", OLD_HELPER, NEW_HELPER)
    for path, old, new in REPLACEMENTS:
        _sub(path, old, new)
    print("rewrote for head-size-based ordering")


if __name__ == "__main__":
    main()
