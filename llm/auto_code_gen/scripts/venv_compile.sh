# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

# venv_compile.sh — pod-local venv setup + cmake incremental build.
#
# The venv is created under /tmp so each pod gets its own clean copy.
# This prevents cross-contamination when parallel pods install different
# packages (e.g. lm_eval upgrading transformers).
#
# Expected env vars (set by caller before sourcing):
#   OUT             — per-job output directory
#   VLLM_DIR        — vllm source tree on the workspace PVC
#   BUILD_FAIL_STATUS — status string written on build failure
#   STATUS_FILE     — file path for writing status on failure
#
# On success, writes BUILD_OK to $OUT/build_status.txt and leaves
# the venv activated with PYTHONPATH set.

echo "[cluster_runner] compile..." | tee -a $OUT/runner.log
cd "$VLLM_DIR"

# A tree rsynced from a git worktree has a .git pointer file that dangles
# here, so setuptools-scm cannot derive the version and 'pip install -e .'
# fails. The sync step ships the locally-computed version in
# .ai_auto_analyze_scm_version; vllm's setup.py get_vllm_version() calls
# setuptools_scm.get_version() directly (which ignores the
# SETUPTOOLS_SCM_PRETEND_VERSION_FOR_* env vars), but it does honor
# VLLM_VERSION_OVERRIDE -- use that.
if ! git -C "$VLLM_DIR" rev-parse HEAD >/dev/null 2>&1; then
    if [ -f "$VLLM_DIR/.ai_auto_analyze_scm_version" ]; then
        PRETEND="$(cat "$VLLM_DIR/.ai_auto_analyze_scm_version")"
        export VLLM_VERSION_OVERRIDE="$PRETEND"
        echo "[cluster_runner] no usable git metadata -- VLLM_VERSION_OVERRIDE=$PRETEND" \
            | tee -a $OUT/runner.log
    fi
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "[cluster_runner] installing uv..." | tee -a $OUT/runner.log
    curl -LsSf https://astral.sh/uv/install.sh | sh 2>&1 | tail -3
    export PATH=$HOME/.local/bin:$PATH
fi

VENV="/tmp/vllm-venv"
SYS_SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
SYS_VLLM="$SYS_SITE/vllm"

echo "[cluster_runner] setting up pod-local venv..." | tee -a $OUT/runner.log

uv venv --system-site-packages --prompt vllm "$VENV" --allow-existing \
    2>&1 | tee -a $OUT/build.log

VIRTUAL_ENV="$VENV" PATH="$VENV/bin:$PATH" \
    uv pip install setuptools_rust cmake ninja \
    2>&1 | tee -a $OUT/build.log

# Image-drift guard. The pod image is built independently of this tree, so it
# lags requirements/cuda.txt, and an older image copy silently wins over the
# pin the tree was written against -- the server then dies at import inside
# files this PR never touched. That reads as SERVER_START_FAILED on the OFF arm
# (a poisoned baseline the gates would happily trust), or as a "regression" that
# the agent then patches around inside vllm. Honour the tree's pins instead.
#
# The pins are read from requirements/cuda.txt at run time, so a rebase that
# bumps one is honoured without touching this script, and a package the tree
# stops pinning is skipped.
#
# torch/torchvision/torchaudio are deliberately NOT in this list: they are ABI
# coupled to the image's CUDA build and to the .so files built below. An
# install that resolves their dependencies drops a fresh torch into the pod
# venv, where it shadows the image copy while torchvision stays behind in
# dist-packages -- and torchvision then dies at import with
# "operator torchvision::nms does not exist", an ABI mismatch that looks nothing
# like its cause. They are not pinned here, and the torch check at the end of
# this guard keeps exactly one torch: the image's.
DRIFT_PKGS="humming-kernels nvidia-cutlass-dsl quack-kernels tilelang apache-tvm-ffi"
DRIFT_APPLIED=0

install_pinned_stack() {
    local pkg spec ver s
    for pkg in $DRIFT_PKGS; do
        # Take the whole "name[extra]==version" spec this tree pinned, if any.
        spec=$(grep -m1 -oE "${pkg}(\[[^]]*\])?==[0-9][^ ;#]*" \
                requirements/cuda.txt 2>/dev/null || true)
        [ -n "$spec" ] || continue
        if [ "$pkg" = "nvidia-cutlass-dsl" ]; then
            # The DSL is the one entry that must resolve its dependencies.
            # Since 4.7 the cutlass python package is split across
            # -libs-core (cutlass/, cute/, cutlass_dsl/), -libs-base
            # (cutlass/_mlir/) and -libs-cu13 (the compiled _cutlass_ir that
            # injects the nvvm enums the tree reads). --no-deps installs only
            # the shim, which leaves cutlass/ a namespace package -- and a
            # namespace package loses to the image's regular one no matter how
            # sys.path is ordered. The DSL depends on nothing but its own
            # libraries, so this cannot drag a second torch in; the torch check
            # below still has the last word.
            echo "[cluster_runner] installing the pinned $spec with its libraries" \
                | tee -a $OUT/runner.log
            VIRTUAL_ENV="$VENV" PATH="$VENV/bin:$PATH" \
                uv pip install --reinstall "$spec" 2>&1 | tail -3 \
                | tee -a $OUT/build.log \
                || echo "[cluster_runner] WARNING: $spec did not install" \
                    | tee -a $OUT/runner.log
        else
            echo "[cluster_runner] installing the pinned $spec (image drift guard)" \
                | tee -a $OUT/runner.log
            VIRTUAL_ENV="$VENV" PATH="$VENV/bin:$PATH" \
                uv pip install --no-deps --reinstall "$spec" 2>&1 | tail -3 \
                | tee -a $OUT/build.log \
                || echo "[cluster_runner] WARNING: $spec did not install" \
                    | tee -a $OUT/runner.log
        fi
        DRIFT_APPLIED=1
    done
}

# Having the pinned package in the venv is not the same as using it.
#
# nvidia-cutlass-dsl ships its python inside the distribution (dsl_packages in
# 4.7.x, python_packages in 4.5.x) and publishes it through a .pth file, so the
# image's copy -- published by the image's own .pth -- keeps winning over the
# venv's. The pinned version is then installed and reported by
# importlib.metadata while "import cutlass" still loads the image's, and the run
# dies later on a name the pinned release renamed. Prepending the venv's copy
# settles it: PYTHONPATH always precedes dist-packages.
export_pinned_dsl_path() {
    local pp
    for pp in "$VENV"/lib/python*/site-packages/nvidia_cutlass_dsl/dsl_packages \
              "$VENV"/lib/python*/site-packages/nvidia_cutlass_dsl/python_packages; do
        # __init__.py, not just the directory: a namespace cutlass/ is what
        # Python needs to fall through to the image's regular package.
        if [ -f "$pp/cutlass/__init__.py" ]; then
            export PYTHONPATH="$pp:${PYTHONPATH:-}"
            echo "[cluster_runner] using the pinned CuTe DSL from $pp" \
                | tee -a $OUT/runner.log
            return 0
        fi
    done
    return 0
}

verify_pinned_stack() {
    "$VENV/bin/python" - "$VENV" <<'PYEOF'
import sys

import importlib.metadata as md

venv = sys.argv[1]
missing = []

import torch
print("[cluster_runner] torch %s from %s" % (torch.__version__, torch.__file__))
if torch.__file__.startswith(venv):
    missing.append("the pod venv shadowed the image's torch, so torchvision "
                   "would die on 'operator torchvision::nms does not exist'")

try:
    import humming.dtypes as dt
    if not hasattr(dt, "float4e0m3"):
        missing.append("humming.dtypes has no float4e0m3")
except Exception as exc:
    missing.append("humming cannot be imported (%s)" % exc)

# vllm/cute_utils/_tcgen05.py reads these names at import time, so a CuTe DSL
# older than the tree kills every worker in warmup with
#   AttributeError: module 'cutlass._mlir.dialects.nvvm' has no attribute 'CTAGroupKind'
try:
    import cutlass
    from cutlass._mlir.dialects import nvvm
    print("[cluster_runner] cutlass from %s" % cutlass.__file__)
    if not cutlass.__file__.startswith(venv):
        missing.append("the pinned CuTe DSL is installed but not effective: "
                       "'import cutlass' still resolves to the image's copy")
    for attr in ("CTAGroupKind", "Tcgen05LdStShape"):
        if not hasattr(nvvm, attr):
            missing.append("cutlass._mlir.dialects.nvvm has no %s" % attr)
except Exception as exc:
    missing.append("the cutlass nvvm dialect cannot be imported (%s)" % exc)

for dist in ("humming-kernels", "nvidia-cutlass-dsl", "quack-kernels",
             "tilelang", "apache-tvm-ffi", "nvidia-cutlass-dsl-libs-core",
             "nvidia-cutlass-dsl-libs-cu13"):
    try:
        print("[cluster_runner] %-22s == %s" % (dist, md.version(dist)))
    except Exception as exc:
        missing.append("%s is not installed (%s)" % (dist, exc))

for problem in missing:
    print("[cluster_runner] IMAGE DRIFT: %s" % problem)
sys.exit(1 if missing else 0)
PYEOF
}

install_pinned_stack
export_pinned_dsl_path
# PIPESTATUS, not pipefail, because this fragment is shared by five entrypoint
# scripts and only run_iteration.sh is known to set the pipe status through.
if [ "$DRIFT_APPLIED" = "1" ]; then
    verify_pinned_stack 2>&1 | tee -a $OUT/runner.log
    if [ "${PIPESTATUS[0]:-1}" != "0" ]; then
        echo "$BUILD_FAIL_STATUS" > "$STATUS_FILE"
        echo "[cluster_runner] the pod image cannot measure this base: the pins" \
            "in requirements/cuda.txt are not satisfied (see IMAGE DRIFT above)." \
            "Rebuild the image or overlay the packages -- do NOT patch vllm to" \
            "work around third-party version drift." | tee -a $OUT/runner.log
        exit 1
    fi
fi

VIRTUAL_ENV="$VENV" PATH="$VENV/bin:$PATH" \
    VLLM_TARGET_DEVICE=empty \
    uv pip install --no-build-isolation --no-deps -e . \
    2>&1 | tee -a $OUT/build.log

for sub in */pyproject.toml; do
    [ -f "$sub" ] || continue
    VIRTUAL_ENV="$VENV" PATH="$VENV/bin:$PATH" \
        uv pip install --no-build-isolation --no-deps -e "$(dirname "$sub")" 2>&1 \
        || echo "WARNING: sub-package $(dirname "$sub") failed" | tee -a $OUT/build.log
done

# A SKIP_CMAKE=1 pod trusts the .so files an earlier compile pod left on the
# workspace PVC. That trust is only sound when the tree on the PVC is the tree
# those .so files were built from, and a rebase breaks it silently: the PR diff
# can be pure Python while thousands of upstream commits rewrite csrc/, and
# upstream can even move the ops into a new extension (CUDA migrated from _C to
# _C_stable_libtorch), so the cached artifacts lack modules the tree now imports
# and the server dies at import as SERVER_START_FAILED -- which the OFF arm
# would otherwise report as a healthy baseline.
#
# This verdict must come first: the copy below and the cmake build further down
# both key off SKIP_CMAKE, and a pod that only rebuilds without restoring the
# image's pre-built extensions still fails -- this time on the flash-attention
# modules that no source build here produces.
BUILD_STAMP="$VLLM_DIR/.ai_auto_analyze_build_stamp"
if [ "${SKIP_CMAKE:-0}" = "1" ]; then
    STALE_REASON=""
    TREE_SHA=""
    if [ -d "$VLLM_DIR/.git" ] || [ -f "$VLLM_DIR/.git" ]; then
        # A worktree's .git pointer may dangle here (exit 128 under set -e);
        # a failed rev-parse is just "no git metadata", never fatal.
        TREE_SHA=$(git -C "$VLLM_DIR" rev-parse HEAD 2>/dev/null || true)
    fi
    if [ -n "$TREE_SHA" ] && [ -f "$BUILD_STAMP" ]; then
        if [ "$(tr -d '[:space:]' < "$BUILD_STAMP")" != "$TREE_SHA" ]; then
            STALE_REASON="artifacts built at $(cut -c1-12 "$BUILD_STAMP"), tree is ${TREE_SHA:0:12}"
        fi
    elif [ -n "$TREE_SHA" ]; then
        STALE_REASON="no build stamp for $TREE_SHA"
    else
        # No git metadata on the PVC: fall back to the module set the tree
        # declares versus the .so files actually present.
        for _ext in $(grep -ohE 'CMakeExtension\(name="vllm\.[A-Za-z0-9_]+"' setup.py 2>/dev/null \
                        | sed -E 's/.*name="vllm\.([A-Za-z0-9_]+)".*/\1/' | sort -u); do
            if ! compgen -G "vllm/${_ext}*.so" > /dev/null 2>&1; then
                STALE_REASON="setup.py declares vllm.${_ext} but vllm/${_ext}*.so is absent"
                break
            fi
        done
    fi
    if [ -n "$STALE_REASON" ]; then
        echo "[cluster_runner] cached .so artifacts do not match this tree ($STALE_REASON) -- building" \
            | tee -a $OUT/runner.log
        SKIP_CMAKE=0
    fi
fi

if [ -d "$SYS_VLLM" ] && [ ! -L "$SYS_VLLM" ] && [ "${SKIP_CMAKE:-0}" != "1" ]; then
    echo "[cluster_runner] copying pre-built .so artifacts..." | tee -a $OUT/runner.log
    find "$SYS_VLLM" -type f -not -path '*/__pycache__/*' -not -name '*.pyc' | while read -r _f; do
        _rel="${_f#$SYS_VLLM/}"
        _target="$VLLM_DIR/vllm/$_rel"
        if [[ "$_f" == *.so ]] || [ ! -f "$_target" ]; then
            mkdir -p "$(dirname "$_target")"
            cp -f "$_f" "$_target"
        fi
    done
    SO_COUNT=$(find "$VLLM_DIR/vllm" -name '*.so' | wc -l)
    echo "[cluster_runner] $SO_COUNT .so files in workspace" | tee -a $OUT/runner.log
elif [ "${SKIP_CMAKE:-0}" = "1" ]; then
    echo "[cluster_runner] skipping system .so copy (SKIP_CMAKE=1, using compile pod artifacts)" | tee -a $OUT/runner.log
fi

[ -L "$SYS_VLLM" ] && rm -f "$SYS_VLLM" 2>/dev/null || true
[ -d "$SYS_VLLM" ] && rm -rf "$SYS_VLLM" 2>/dev/null || true
ln -sf "$VLLM_DIR/vllm" "$SYS_VLLM" 2>/dev/null || echo "[cluster_runner] WARNING: could not symlink system vllm (permission denied, using venv)" | tee -a $OUT/runner.log

source "$VENV/bin/activate"
export PYTHONPATH="$VLLM_DIR:${PYTHONPATH:-}"

# Image-drift guard. The runtime image can lag the repo's pins: it ships
# flashinfer 0.6.13 while requirements/cuda.txt pins 0.6.16.post3 (the version
# the analyzed trace actually ran). The installs above use --no-deps, so the
# image copy would otherwise win and APIs that exist only in the pinned version
# would be silently missing -- a feature patch then degrades to a no-op instead
# of failing loudly. FI_PREBUILT_DIR holds the pinned packages, prepared once on
# the workspace PVC; PYTHONPATH precedes site-packages, so it shadows the image.
FI_PREBUILT_DIR="${FI_PREBUILT_DIR:-/workspace/fi-0616-lean}"
if [ -d "$FI_PREBUILT_DIR/flashinfer" ]; then
    export PYTHONPATH="$FI_PREBUILT_DIR:$PYTHONPATH"
    # Shadowing flashinfer alone is not enough: the prebuilt kernels live in a
    # separate package (flashinfer-jit-cache) whose .so files embed the FFI
    # signatures of the build they came from. Overlay only the Python side and
    # the image's stale jit-cache still supplies them, so the two disagree --
    # trtllm_paged_attention_decode takes 30 args in 0.6.13 and 32 in 0.6.16,
    # which kills sparse MLA in engine init with "Expected 30 but got 32
    # arguments". The overlay therefore carries a matching jit-cache, and
    # flashinfer's version-check bypass goes back OFF so it stays in check.
    if [ -d "$FI_PREBUILT_DIR/flashinfer_jit_cache" ]; then
        echo "[cluster_runner] pinned flashinfer overlay active (with jit-cache): $FI_PREBUILT_DIR" \
            | tee -a $OUT/runner.log
    else
        export FLASHINFER_DISABLE_VERSION_CHECK=1
        echo "[cluster_runner] WARNING: $FI_PREBUILT_DIR carries no flashinfer_jit_cache; " \
            "keeping the image's prebuilt kernels and bypassing flashinfer's version " \
            "check -- kernel-ABI mismatches are expected if the image lags the pin." \
            | tee -a $OUT/runner.log
    fi
    # Cheap preflight: an ABI mismatch costs ~20 GPU-minutes to surface as an
    # engine-init crash, and costs ~20 seconds to detect here.
    # flashinfer >= 0.7.0 replaced the monolithic jit-cache package with a shim
    # + per-arch provider wheels (flashinfer_jit_cache.get_jit_cache_providers);
    # get_jit_cache_dir() exists only in the 0.6.x line.
    if ! python3 - <<'PYEOF' | tee -a $OUT/runner.log
import os
import flashinfer
import flashinfer_jit_cache
if hasattr(flashinfer_jit_cache, "get_jit_cache_providers"):
    providers = flashinfer_jit_cache.get_jit_cache_providers()
    if not providers:
        raise SystemExit(
            "flashinfer_jit_cache found no AOT providers -- kernels will "
            "fall back to JIT and fail offline"
        )
    print("[cluster_runner] flashinfer=%s jit_cache=%s providers=%s" % (
        flashinfer.__version__, flashinfer_jit_cache.__version__,
        ",".join(p.provider_id for p in providers)))
else:
    print("[cluster_runner] flashinfer=%s jit_cache=%s aot=%s" % (
        flashinfer.__version__, flashinfer_jit_cache.__version__,
        flashinfer_jit_cache.get_jit_cache_dir()))
if not flashinfer_jit_cache.__version__.startswith(flashinfer.__version__) \
        and not os.getenv("FLASHINFER_DISABLE_VERSION_CHECK"):
    raise SystemExit(1)
PYEOF
    then
        echo "$BUILD_FAIL_STATUS" > "$STATUS_FILE"
        echo "[cluster_runner] flashinfer/jit-cache preflight FAILED" | tee -a $OUT/runner.log
        exit 1
    fi
else
    echo "[cluster_runner] NOTE: $FI_PREBUILT_DIR absent, using the image's flashinfer" \
        | tee -a $OUT/runner.log
fi

if [ "${SKIP_CMAKE:-0}" != "1" ]; then
    echo "[cluster_runner] cmake incremental build..." | tee -a $OUT/runner.log
    if [ -f cmake-build-release/CMakeCache.txt ]; then
        CACHED_SRC=$(grep "CMAKE_HOME_DIRECTORY:INTERNAL=" cmake-build-release/CMakeCache.txt 2>/dev/null | cut -d= -f2)
        if [ -n "$CACHED_SRC" ] && [ "$CACHED_SRC" != "$VLLM_DIR" ]; then
            echo "[cluster_runner] removing stale cmake cache (was: $CACHED_SRC, now: $VLLM_DIR)" | tee -a $OUT/runner.log
            rm -rf cmake-build-release
        fi
    fi
    if [ -f tools/generate_cmake_presets.py ]; then
        # Touch csrc/ files that are older than their cached .cu.o so ninja
        # rebuilds them. oc rsync preserves host timestamps, so patched .cu
        # files can appear older than .cu.o from a previous PVC build.
        if [ -d cmake-build-release/CMakeFiles ]; then
            TOUCHED=0
            while IFS= read -r obj; do
                src="${obj#cmake-build-release/CMakeFiles/_C_stable_libtorch.dir/}"
                src="${src%.o}"
                if [ -f "$src" ] && [ "$obj" -nt "$src" ]; then
                    touch "$src"
                    TOUCHED=$((TOUCHED + 1))
                fi
            done < <(find cmake-build-release/CMakeFiles -name '*.cu.o' -o -name '*.cpp.o' 2>/dev/null)
            if [ "$TOUCHED" -gt 0 ]; then
                echo "[cluster_runner] touched $TOUCHED source files newer than their .o cache" | tee -a $OUT/runner.log
            fi
        fi
        python tools/generate_cmake_presets.py --force-overwrite 2>&1 | tee -a $OUT/build.log
        cmake --preset release -DCUDA_nvrtc_LIBRARY=/usr/local/cuda/lib64/libnvrtc.so.13 2>&1 | tee -a $OUT/build.log
        cmake --build --preset release --target _C_stable_libtorch -j 8 2>&1 | tee -a $OUT/build.log
        find cmake-build-release -name "_C_stable_libtorch*.so" -exec cp -v {} vllm/ \; 2>&1 | tee -a $OUT/build.log
        BUILD_EXIT=${PIPESTATUS[0]}
        if [ "$BUILD_EXIT" -ne 0 ]; then
            echo "$BUILD_FAIL_STATUS" > "$STATUS_FILE"
            echo "[cluster_runner] BUILD FAILED" | tee -a $OUT/runner.log
            exit 1
        fi
    else
        echo "[cluster_runner] no cmake presets found, relying on pre-built .so" | tee -a $OUT/runner.log
    fi
else
    echo "[cluster_runner] cmake skipped (SKIP_CMAKE=1)" | tee -a $OUT/runner.log
fi

# Record which commit these artifacts were built from, so the next SKIP_CMAKE=1
# pod can tell a matching tree from a rebased one.
if [ "${SKIP_CMAKE:-0}" != "1" ]; then
    # No .git on the PVC (worktree pointer was dropped): rev-parse would exit
    # 128 and, under 'set -e', kill this script right after a good build.
    # The host-side sync shipped the SHA it should stamp instead.
    STAMP_SHA=$(git -C "$VLLM_DIR" rev-parse HEAD 2>/dev/null || true)
    if [ -z "$STAMP_SHA" ] && [ -f "$VLLM_DIR/.ai_auto_analyze_scm_sha" ]; then
        STAMP_SHA=$(tr -d '[:space:]' < "$VLLM_DIR/.ai_auto_analyze_scm_sha")
    fi
    if [ -n "$STAMP_SHA" ]; then
        echo "$STAMP_SHA" > "$BUILD_STAMP" 2>/dev/null \
            || echo "[cluster_runner] WARNING: could not write $BUILD_STAMP" | tee -a $OUT/runner.log
    fi
fi
echo "BUILD_OK" > $OUT/build_status.txt
echo "[cluster_runner] build OK" | tee -a $OUT/runner.log
