#!/usr/bin/env python3
"""Apply the PR-2 change in the vLLM worktree: fused shared experts on by default.

Three edits, each asserted so a silent no-op is impossible:
  1. env schema: default True
  2. env getter: default "True" + comment
  3. resolver: an info_once line so the resolved decision is visible in the log
     (the skip path already warns; the enabled path said nothing)

usage: apply_fse_patch.py <vllm-worktree>
"""

import pathlib
import sys

WT = pathlib.Path(sys.argv[1])

EDITS = [
    (
        WT / "vllm/envs.py",
        "    VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS: bool = False",
        "    VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS: bool = True",
    ),
    (
        WT / "vllm/envs.py",
        '    # Whether to use aiter fusion shared experts ops.\n'
        '    # By default is disabled.\n'
        '    "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS": lambda: (\n'
        '        os.getenv("VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS", "False").lower()\n'
        '        in ("true", "1")\n'
        '    ),',
        '    # Whether to use aiter fusion shared experts ops.\n'
        '    # Enabled by default: the shared expert then runs as one extra slot of\n'
        '    # the routed MoE kernel instead of a separate dense MLP. The per-layer\n'
        '    # resolver (fused_moe/utils.py) still gates it on quantization\n'
        '    # compatibility and expert parallelism, so checkpoints that cannot use\n'
        '    # it keep the previous behaviour. Set to 0 to restore it.\n'
        '    "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS": lambda: (\n'
        '        os.getenv("VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS", "True").lower()\n'
        '        in ("true", "1")\n'
        '    ),',
    ),
    (
        WT / "vllm/model_executor/layers/fused_moe/utils.py",
        "    if skip_reason is not None:",
        "    if is_fused_shared_expert_enabled:\n"
        "        # The skip path warns once per cause; state the enabled decision\n"
        "        # too, so an operator can tell \"not compatible\" from \"not requested\"\n"
        "        # without inferring it from the absence of a warning.\n"
        "        logger.info_once(\n"
        "            \"AITER fused shared experts are ENABLED for %s; \"\n"
        "            \"disable with VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0.\",\n"
        "            _normalize_skip_reason(prefix),\n"
        "        )\n"
        "    if skip_reason is not None:",
    ),
]


def main() -> None:
    for path, old, new in EDITS:
        text = path.read_text()
        if new in text:
            print(f"already applied: {path.name}")
            continue
        if text.count(old) != 1:
            raise SystemExit(
                f"expected exactly one occurrence in {path} for:\n{old[:120]}")
        path.write_text(text.replace(old, new))
        print(f"patched {path.name}")


if __name__ == "__main__":
    main()
