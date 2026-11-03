#!/usr/bin/env bash
# Preflight for every campaign driver: the images exist locally, are pinned by
# digest, and are ROCm 10. Sourced after images.env so $VLLM_IMAGE and
# $SGLANG_IMAGE are set.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=images.env
source "$HERE/images.env"

for img in "$VLLM_IMAGE" "$SGLANG_IMAGE"; do
  case "$img" in
    *@sha256:*) ;;
    *) echo "REFUSING to run: image is not digest-pinned: $img" >&2; exit 1 ;;
  esac
  bash "$HERE/assert_rocm10.sh" "$img" || exit 1
done
