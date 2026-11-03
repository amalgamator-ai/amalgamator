#!/usr/bin/env bash
# Report the ROCm version of a pinned image, from several independent sources.
# The two campaign images number their ROCm installs differently, so a single
# file (/opt/rocm/.info/version) is not a reliable answer on its own; torch's
# HIP build tag is what actually determines the runtime ABI.
set -u
IMG="${1:?usage: check_rocm.sh <image-ref>}"
docker run --rm --entrypoint bash "$IMG" -c '
echo "--- /opt/rocm/.info/version"
cat /opt/rocm/.info/version 2>/dev/null || echo "(absent)"
echo "--- hipconfig --version"
hipconfig --version 2>/dev/null | head -2 || echo "(absent)"
echo "--- torch build"
python3 - <<PY 2>/dev/null || echo "(no torch)"
import torch
print("torch", torch.__version__)
print("hip", getattr(torch.version, "hip", None))
print("hip_version", getattr(torch.version, "hip_version", None))
PY
'
