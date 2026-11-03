#!/usr/bin/env python3
"""Move PR 1's env declaration away from the fused-shared-experts entries.

Both PRs touch vllm/envs.py. With the new variable sitting directly above
VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS, the two patches share diff context
and stop applying together, which is avoidable noise for two independent PRs.
Moving it next to the MHA entries leaves several lines of separation.

usage: move_env_entries.py <vllm-worktree>
"""

import pathlib
import sys

p = pathlib.Path(sys.argv[1]) / "vllm/envs.py"
s = p.read_text()

SCHEMA = "    VLLM_DISABLE_AITER_BACKEND_PRIORITY: bool = False\n"
GETTER = (
    "    # Keep the previous ROCM_ATTN-first ordering instead of trying the\n"
    "    # AITER attention backends first on ROCm.\n"
    '    "VLLM_DISABLE_AITER_BACKEND_PRIORITY": lambda: (\n'
    '        os.getenv("VLLM_DISABLE_AITER_BACKEND_PRIORITY", "False").lower()\n'
    '        in ("true", "1")\n'
    "    ),\n"
)
SCHEMA_ANCHOR = "    VLLM_ROCM_USE_AITER_MLA: bool = True\n"
GETTER_ANCHOR = (
    "    # Whether to use aiter mha ops.\n"
    "    # By default is enabled.\n"
    '    "VLLM_ROCM_USE_AITER_MHA": lambda: (\n'
    '        os.getenv("VLLM_ROCM_USE_AITER_MHA", "True").lower() in ("true", "1")\n'
    "    ),\n"
)

for block in (SCHEMA, GETTER):
    assert s.count(block) == 1, f"expected one {block[:40]!r}, found {s.count(block)}"
    s = s.replace(block, "")

for anchor, block in ((SCHEMA_ANCHOR, SCHEMA), (GETTER_ANCHOR, GETTER)):
    assert s.count(anchor) == 1, f"anchor {anchor[:40]!r} found {s.count(anchor)}x"
    s = s.replace(anchor, anchor + block)

p.write_text(s)
print("moved env entries next to the MHA ones")
