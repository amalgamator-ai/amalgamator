#!/usr/bin/env bash
# Fail loudly unless an image's torch is built for ROCm 10.
#
# Tag names are not proof: `nightly-rocm100` is a floating tag, and the vLLM
# image has no /opt/rocm at all (ROCm ships as SDK wheels, so hipconfig reports
# TheRock's internal 7.15.x numbering). torch.version.hip is the build the
# kernels actually run against, so that is the check.
#
# usage: assert_rocm10.sh <image-ref>
set -uo pipefail
IMG="${1:?usage: assert_rocm10.sh <image-ref>}"

out=$(docker run --rm --entrypoint python3 "$IMG" -c '
import torch, sys
v = torch.__version__
hip = getattr(torch.version, "hip", None) or ""
print(f"{v}|{hip}")
' 2>&1 | tail -1)

ver="${out%%|*}"
hip="${out##*|}"

if [[ "$ver" == *"+rocm10."* || "$hip" == 10.* ]]; then
    echo "ROCm 10 OK: $IMG (torch $ver, hip ${hip:-n/a})"
    exit 0
fi
echo "NOT ROCm 10: $IMG (torch ${ver:-?}, hip ${hip:-?})" >&2
exit 1
