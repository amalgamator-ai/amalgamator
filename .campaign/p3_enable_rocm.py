#!/usr/bin/env python3
"""Enable vLLM's fused QK-norm+RoPE+gate Triton kernel on ROCm for qwen3_next.

The kernel (fused_qk_norm_rope.py, @triton.jit) is platform-neutral; the guard
lists only CUDA and XPU. Widen it to ROCm so Qwen3.5/3.8 full-attention layers
take the fused path instead of the eager split + four generated Triton kernels.
"""
import sys

path = sys.argv[1]
src = open(path).read()
old = """            and (current_platform.is_cuda() or current_platform.is_xpu())
"""
new = """            and (
                current_platform.is_cuda()
                or current_platform.is_xpu()
                or current_platform.is_rocm()
            )
"""
if src.count(old) != 1:
    raise SystemExit(f"anchor found {src.count(old)} times in {path}")
open(path, "w").write(src.replace(old, new))
print("patched", path)
