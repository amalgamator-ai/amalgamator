#!/usr/bin/env bash
# Copy the compiled vLLM extensions out of the pinned image into a source
# worktree.
#
# A source-mounted profile serves the mounted tree via PYTHONPATH, so the whole
# `vllm` package must be importable from there — including the compiled modules
# (`_C`, `_rocm_C`, ...) that ship in the image but are not checked in. A fresh
# `git worktree` has none of them and the engine dies with
# "No module named 'vllm._rocm_C'". The worktree must sit on the image's commit
# for the ABI to match; this script refuses otherwise.
#
# usage: sync_vllm_ext.sh <worktree> [image-ref]
set -euo pipefail
WT="${1:?usage: sync_vllm_ext.sh <worktree> [image-ref]}"
IMG="${2:-vllm/vllm-openai-rocm@sha256:b1374c3814b6e7706c49968ba17ca3b35484382f7b9fc7115dd3e66068d53f2f}"

[ -d "$WT/vllm" ] || { echo "not a vllm tree: $WT" >&2; exit 1; }

docker run --rm --entrypoint bash -v "$WT:/dst" "$IMG" -c '
set -e
P=$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))")
img_commit=$(python3 -c "import vllm;print(vllm.__version__.rsplit(\"+\",1)[-1])")
echo "image vllm: $img_commit ($P)"
n=$(ls "$P"/*.so 2>/dev/null | wc -l)
echo "copying $n compiled modules into /dst/vllm"
cp "$P"/*.so /dst/vllm/
# _version.py is generated at build time; without it vllm reports version
# "dev" and skips the commit stamp, which some code paths read.
[ -f "$P/_version.py" ] && cp "$P/_version.py" /dst/vllm/_version.py && echo "copied _version.py"
ls /dst/vllm/*.so | wc -l
'
