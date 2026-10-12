import os as _os
import sys as _sys

_project_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import argparse
import asyncio
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from datetime import UTC

from common.claude_utils import PipelineStep, _all_steps_complete
from common.iteration_utils import check_converged as _check_converged
from common.utils import safe_clean_dir, setup_logging
from llm.auto_code_gen.agents import agent_run
from llm.auto_code_gen.code_gen_configs import load_config_and_use_case
from llm.auto_code_gen.code_gen_prompts import (
    CODE_GEN_DIR,
    DIAGNOSTIC_PROFILE_DIR,
    DIAGNOSTIC_REPORT_FILE,
    RUNTIME_DIR,
    RUNTIME_EXPECTED_IMPROVEMENT_FILE,
    RUNTIME_SUCCESS_FILE,
    gen_ApplyCodeAndCompilePrompt,
    gen_DiagnosticProfileAndAnalyzePrompt,
    gen_InvestigateRuntimeOutputAndFixCodePrompt,
    gen_RecordLearningsPrompt,
    gen_ReviewInvestigationPrompt,
    gen_RunAndLogPrompt,
    gen_RunLMEvalPrompt,
)
from llm.auto_code_gen.remote_prompts import (
    gen_RemoteApplyCodePrompt,
    gen_RemoteRunDiagnosticPrompt,
    gen_RemoteRunIterationPrompt,
)
from llm.auto_code_gen.workspace import clear_vllm_source_tree

# ── shared baselines (reuse across iterations) ──────────────────────────────

SHARED_BASELINES_DIR = "baselines"


def _mechanical_apply_and_sync(config, output_dir, patch_file, iteration):
    """Apply patch + sync to PVC without a Claude agent. Pure Python.

    Returns True on success, False on failure.
    """
    source_dir = config.source_code_dir
    patch_abs = (
        os.path.join(output_dir, patch_file)
        if not os.path.isabs(patch_file)
        else patch_file
    )

    # Clear local source tree (tracked + untracked files from previous patches)
    print("  [mechanical] git reset --hard + clean...")
    r = subprocess.run(
        ["git", "-C", source_dir, "reset", "--hard"], capture_output=True, text=True
    )
    if r.returncode != 0:
        print(f"  [mechanical] git reset failed: {r.stderr.strip()}")
        return False
    subprocess.run(
        ["git", "-C", source_dir, "clean", "-fd"], capture_output=True, text=True
    )

    # Apply patch
    print(f"  [mechanical] git apply {os.path.basename(patch_abs)}...")
    r = subprocess.run(
        ["git", "-C", source_dir, "apply", patch_abs], capture_output=True, text=True
    )
    if r.returncode != 0:
        # Try with --3way fallback
        r = subprocess.run(
            ["git", "-C", source_dir, "apply", "--3way", patch_abs],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print(f"  [mechanical] git apply failed: {r.stderr.strip()}")
            return False

    # Verify diff
    r = subprocess.run(
        ["git", "-C", source_dir, "diff", "--stat"], capture_output=True, text=True
    )
    print("  [mechanical] {}".format(r.stdout.strip().split("\n")[-1]))

    if not _sync_tree_to_pvc(config, source_dir):
        return False

    print("  [mechanical] apply + sync OK")
    return True


def _pvc_vllm_dir(config):
    """This run's vllm dir on the workspace PVC: /workspace/pvc_<run name>.

    The name defaults to "pvc_" + the output_dir basename, so the directory
    says both what it is (a run's PVC tree) and which run it belongs to. Each
    run owns a run-named tree on the PVC instead of sharing /workspace/vllm --
    concurrent runs (e.g. a new plan's codegen while review comments are being
    fixed on another) never rsync over each other's tree.
    """
    subdir = (getattr(config, "remote_vllm_subdir", "") or "").strip("/")
    if not subdir:
        run_name = os.path.basename(os.path.normpath(config.output_dir)) or "vllm"
        subdir = "pvc_" + run_name
    else:
        # Defensive: the loader validates a single segment, but never let a
        # stray "a/b" escape the /workspace prefix or point at shared dirs.
        subdir = os.path.basename(subdir)
    return f"/workspace/{subdir}"


def _scm_pretend_version(source_dir):
    """The setuptools-scm version of a source tree, as a plain string.

    The PVC copy of a git worktree has no usable .git (worktrees keep a
    pointer file to a local path that does not exist inside the pod), so
    setuptools-scm cannot derive a version there and 'pip install -e .'
    fails. Computed here -- where git works -- and shipped to the PVC as
    .ai_auto_analyze_scm_version; venv_compile.sh feeds it to setuptools-scm
    via SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM when the tree has no git.

    Returns "" when the tree has no describable version (e.g. no tags).
    """
    import re

    describe = subprocess.run(
        [
            "git",
            "-C",
            source_dir,
            "describe",
            "--tags",
            "--long",
            "--abbrev=40",
            "--match",
            "v[0-9]*",
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
    # "v0.30.1rc0-606-g<sha>" -> "0.30.1rc0.dev606+g<sha>" (PEP 440)
    match = re.match(r"^v?(\S+)-(\d+)-g([0-9a-f]+)$", describe)
    if not match:
        return ""
    return f"{match.group(1)}.dev{match.group(2)}+g{match.group(3)}"


def _sync_tree_to_pvc(config, source_dir):
    """rsync a local source tree to /workspace/<run subdir> on the PVC.

    Split out of _mechanical_apply_and_sync so a caller that already has the
    tree it wants measured (the PR-review flow, which edits in place) can sync
    without a reset --hard + git apply round trip.

    cmake-build-release/ lives inside the synced tree but is a build cache,
    not source. Excluding it also protects it from --delete, so iterations keep
    an incremental build instead of recompiling from scratch.
    """
    ns = config.remote_namespace
    pvc = config.remote_pvc_name
    pvc_vllm = _pvc_vllm_dir(config)
    helper = f"ai-auto-analyze-apply-{uuid.uuid4().hex[:6]}"
    overrides = json.dumps(
        {
            "spec": {
                "containers": [
                    {
                        "name": helper,
                        "image": "registry.access.redhat.com/ubi9/ubi:latest",
                        "command": ["sleep", "300"],
                        "volumeMounts": [{"name": "ws", "mountPath": "/workspace"}],
                    }
                ],
                "volumes": [
                    {"name": "ws", "persistentVolumeClaim": {"claimName": pvc}}
                ],
            }
        }
    )

    try:
        print("  [mechanical] creating helper pod for rsync...")
        subprocess.run(
            [
                "oc",
                "run",
                helper,
                "-n",
                ns,
                "--image=registry.access.redhat.com/ubi9/ubi:latest",
                "--overrides=" + overrides,
                "--restart=Never",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(
            [
                "oc",
                "wait",
                "-n",
                ns,
                "pod/" + helper,
                "--for=condition=Ready",
                "--timeout=120s",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        # oc rsync falls back to tar when the pod has no rsync binary (ubi9),
        # and the tar fallback cannot create the destination directory -- it
        # fails with "Cannot open: No such file or directory" on a fresh
        # run-named PVC subdir. Create it first.
        subprocess.run(
            ["oc", "exec", "-n", ns, helper, "--", "mkdir", "-p", pvc_vllm],
            capture_output=True,
            text=True,
            check=True,
        )

        print(f"  [mechanical] rsync to PVC ({helper}:{pvc_vllm})...")
        r = subprocess.run(
            [
                "oc",
                "rsync",
                "-n",
                ns,
                source_dir + "/",
                f"{helper}:{pvc_vllm}/",
                "--no-perms",
                "--delete",
                "--exclude=cmake-build-release",
            ],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print(f"  [mechanical] rsync failed: {r.stderr.strip()}")
            return False

        # Worktree .git is a pointer file that dangles inside the pod: git
        # there exits 128 and, under 'set -e', kills venv_compile.sh's build
        # stamp check. The pod never needs it (the version ships in
        # .ai_auto_analyze_scm_version), so drop the pointer from the PVC.
        subprocess.run(
            ["oc", "exec", "-n", ns, helper, "--", "rm", "-f", f"{pvc_vllm}/.git"],
            capture_output=True,
            text=True,
            check=False,
        )

        # The build there cannot ask setuptools-scm for a version without git.
        # Ship it instead.
        version = _scm_pretend_version(source_dir)
        if version:
            subprocess.run(
                [
                    "oc",
                    "exec",
                    "-n",
                    ns,
                    helper,
                    "--",
                    "sh",
                    "-c",
                    f"printf %s '{version}' > {pvc_vllm}/.ai_auto_analyze_scm_version",
                ],
                capture_output=True,
                text=True,
                check=False,
            )

        # Same for the build stamp: venv_compile.sh records the commit the
        # artifacts were built from, so later pods can detect a rebase. With
        # no .git the script cannot rev-parse; ship the SHA it should stamp.
        head_sha = subprocess.run(
            ["git", "-C", source_dir, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if re.fullmatch(r"[0-9a-f]{40}", head_sha):
            subprocess.run(
                [
                    "oc",
                    "exec",
                    "-n",
                    ns,
                    helper,
                    "--",
                    "sh",
                    "-c",
                    f"printf %s '{head_sha}' > {pvc_vllm}/.ai_auto_analyze_scm_sha",
                ],
                capture_output=True,
                text=True,
                check=False,
            )

        return True
    except subprocess.CalledProcessError as e:
        print(f"  [mechanical] helper pod failed: {e.stderr or e}")
        return False
    finally:
        subprocess.run(
            ["oc", "delete", "pod", helper, "-n", ns, "--wait=false"],
            capture_output=True,
            text=True,
        )


def _recompute_lm_eval_verdict(runtime_logs_dir, shared_baseline_dir):
    """Compute lm_eval.txt from this iteration's enabled results + shared baseline.

    Called when the baseline lm_eval run was skipped. Reads enabled results
    from runtime_logs_dir and disabled results from shared_baseline_dir.
    """
    import glob as _glob
    import json as _json

    def _extract(result_dir):
        results = sorted(
            _glob.glob(os.path.join(result_dir, "**", "*.json"), recursive=True)
        )
        if not results:
            return None, None
        r = _json.load(open(results[-1])).get("results", {}).get("gsm8k", {})
        return r.get("exact_match,flexible-extract"), r.get("exact_match,strict-match")

    def _stderr(result_dir):
        """flexible-extract stderr of this arm's lm_eval run, or None."""
        results = sorted(
            _glob.glob(os.path.join(result_dir, "**", "*.json"), recursive=True)
        )
        for path in reversed(results):
            try:
                r = _json.load(open(path)).get("results", {}).get("gsm8k", {})
            except OSError, ValueError:
                continue
            se = r.get("exact_match_stderr,flexible-extract")
            if isinstance(se, (int, float)) and se > 0:
                return float(se)
        return None

    en_exit_f = os.path.join(runtime_logs_dir, "lm_eval_enabled_exit_code.txt")
    dis_exit_f = os.path.join(shared_baseline_dir, "lm_eval_disabled_exit_code.txt")
    if not os.path.isfile(en_exit_f) or not os.path.isfile(dis_exit_f):
        return

    en_exit = int(open(en_exit_f).read().strip())
    dis_exit = int(open(dis_exit_f).read().strip())

    flex_en, strict_en = _extract(os.path.join(runtime_logs_dir, "lm_eval_output"))
    # Delta noise: two independent single samples -> sqrt(se_on^2 + se_off^2).
    se_en = _stderr(os.path.join(runtime_logs_dir, "lm_eval_output"))
    se_dis = _stderr(os.path.join(shared_baseline_dir, "lm_eval_output_disabled"))
    delta_se = None
    if se_en is not None or se_dis is not None:
        known = [se for se in (se_en, se_dis) if se is not None]
        if len(known) == 1:
            # One arm's stderr known: assume equal noise and scale by sqrt(2).
            delta_se = (2 * known[0] ** 2) ** 0.5
        else:
            delta_se = (se_en**2 + se_dis**2) ** 0.5
    # The fixed -0.01 tolerance sits inside the sampling noise (with the
    # default 200-question slice, one question is 0.005 and the delta's own
    # stderr is ~0.024), failing identical code a third of the time. When
    # stderrs are recorded, only a degradation beyond 1.65 combined sigmas
    # reads as FAIL. Kept in sync with gates.LM_EVAL_DELTA_SIGMAS.
    tolerance = -0.01 if delta_se is None else -1.65 * delta_se
    lines = [
        f"Feature ENABLED results: flexible-extract={flex_en}, strict-match={strict_en}",
        f"Feature ENABLED exit code: {en_exit}",
    ]

    if dis_exit != 0:
        lines.append(f"Feature DISABLED results: FAILED (exit {dis_exit})")
        lines.append("Verdict: INCONCLUSIVE")
    else:
        flex_dis, strict_dis = _extract(
            os.path.join(shared_baseline_dir, "lm_eval_output_disabled")
        )
        lines.append(
            f"Feature DISABLED results: flexible-extract={flex_dis}, strict-match={strict_dis}"
        )
        lines.append(f"Feature DISABLED exit code: {dis_exit} (shared baseline)")
        deltas = []
        if flex_en is not None and flex_dis is not None:
            fd = flex_en - flex_dis
            lines.append(f"Delta flexible-extract: {fd:+.4f}")
            deltas.append(fd)
        if strict_en is not None and strict_dis is not None:
            sd = strict_en - strict_dis
            lines.append(f"Delta strict-match: {sd:+.4f}")
            deltas.append(sd)
        if not deltas:
            lines.append("Verdict: INCONCLUSIVE")
        elif any(d < tolerance for d in deltas):
            if delta_se is not None:
                lines.append(
                    "Verdict: FAIL (degradation beyond sampling noise: "
                    f"delta {min(deltas):.4f} < tolerance {tolerance:.4f}, delta stderr {delta_se:.4f})"
                )
            else:
                lines.append(
                    "Verdict: FAIL (significant accuracy degradation detected)"
                )
        else:
            if delta_se is not None:
                lines.append(
                    f"Verdict: PASS (delta {min(deltas):.4f} within tolerance {tolerance:.4f}, "
                    f"delta stderr {delta_se:.4f})"
                )
            else:
                lines.append("Verdict: PASS (scores within tolerance)")

    with open(os.path.join(runtime_logs_dir, "lm_eval.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ── cluster subprocess helper ─────────────────────────────────────────────────

_CLUSTER_RUNNER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "cluster_runner.py"
)


def _run_cluster_job(
    subcommand,
    config,
    out_dir,
    bench_command,
    disable_env_var="",
    run_lm_eval=False,
    delay_iterations=None,
    log_name="cluster_runner.log",
    skip_baseline=False,
    skip_lm_eval_baseline=False,
    stage=None,
    force_feature_off=False,
):
    """Run cluster_runner.py as a subprocess with output redirected to a log file.

    Returns (exit_code, status) where status is from run_status.txt or
    diagnostic_status.txt depending on subcommand.
    """
    cmd = [
        sys.executable,
        _CLUSTER_RUNNER_PATH,
        subcommand,
        "--namespace",
        config.remote_namespace,
        "--pvc-name",
        config.remote_pvc_name,
        "--model-pvc-name",
        config.remote_model_pvc_name,
        "--image",
        config.remote_image,
        "--gpu-count",
        str(config.remote_gpu_count),
        "--gpu-product",
        config.remote_gpu_product,
        "--out-dir",
        out_dir,
        "--gpu-wait-timeout-minutes",
        str(config.gpu_wait_timeout_minutes),
        "--vllm-dir",
        _pvc_vllm_dir(config),
    ]
    for entry in getattr(config, "remote_extra_env", []) or []:
        cmd.extend(["--extra-env", entry])

    if subcommand in ("run-iteration", "run-iteration-parallel"):
        cmd.extend(["--bench-command", bench_command])
        cmd.extend(["--server-command", config.remote_server_command])
        cmd.extend(["--disable-env-var", disable_env_var])
        if run_lm_eval:
            cmd.append("--run-lm-eval")
        if skip_lm_eval_baseline:
            cmd.append("--skip-lm-eval-baseline")
        if stage:
            cmd.extend(["--stage", stage])
        if force_feature_off:
            cmd.append("--force-feature-off")
    elif subcommand == "run-diagnostic":
        cmd.extend(["--bench-command", bench_command])
        cmd.extend(["--disable-env-var", disable_env_var])
        if getattr(config, "remote_server_command", ""):
            # A client-style bench (vllm bench serve) profiles nothing on its
            # own -- the engine runs in the server process -- so the diagnostic
            # job needs the server command to know what to put under nsys.
            cmd.extend(["--server-command", config.remote_server_command])
        if delay_iterations is not None:
            cmd.extend(["--delay-iterations", str(delay_iterations)])
        if skip_baseline:
            cmd.append("--skip-baseline")

    log_path = os.path.join(out_dir, log_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[cluster] {subcommand} → {out_dir}")

    with open(log_path, "w") as log_f:
        result = subprocess.run(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )

    if subcommand == "run-diagnostic":
        status_file = os.path.join(out_dir, "diagnostic_status.txt")
    else:
        status_file = os.path.join(out_dir, "run_status.txt")

    status = ""
    if os.path.isfile(status_file):
        with open(status_file) as f:
            status = f.read().strip()

    print(f"[cluster] {subcommand} exit={result.returncode} status={status}")

    return result.returncode, status


# ── backend selectors ─────────────────────────────────────────────────────────
# The four GPU-touching phases are the only ones that vary with runtime_backend.
# "local"  → original prompts, unchanged behavior.
# "remote" → cluster Job prompts; same per-iteration file layout so all
#            downstream phases (perf gate, investigate, fix, summaries) are shared.


def _apply_prompt_gen(config):
    """Select the apply+compile prompt generator."""
    if config.runtime_backend == "remote":

        def _remote(context, patch_file, iteration):
            return gen_RemoteApplyCodePrompt(
                context=context,
                patch_file=patch_file,
                iteration=iteration,
                remote_namespace=config.remote_namespace,
                remote_pvc_name=config.remote_pvc_name,
                remote_model_pvc_name=config.remote_model_pvc_name,
            )

        return _remote
    return gen_ApplyCodeAndCompilePrompt


def _run_prompt_gen(config):
    """Select the benchmark prompt generator (bench only, no lm_eval).

    Used by _benchmark_candidates for per-candidate benchmarking.
    """
    if config.runtime_backend == "remote":

        def _remote(
            context,
            execution_command,
            runtime_logs_dir,
            iteration,
            gpu_wait_timeout_minutes=60,
            smaller_model_file=None,
            disable_new_feature=False,
        ):
            return gen_RemoteRunIterationPrompt(
                context=context,
                execution_command=execution_command,
                runtime_logs_dir=runtime_logs_dir,
                iteration=iteration,
                remote_namespace=config.remote_namespace,
                remote_pvc_name=config.remote_pvc_name,
                remote_model_pvc_name=config.remote_model_pvc_name,
                remote_image=config.remote_image,
                remote_gpu_count=config.remote_gpu_count,
                remote_gpu_product=config.remote_gpu_product,
                disable_env_var=_read_feature_disable_env_var(config.output_dir) or "",
                run_lm_eval=False,
                disable_new_feature=disable_new_feature,
                gpu_wait_timeout_minutes=gpu_wait_timeout_minutes,
            )

        return _remote
    return gen_RunAndLogPrompt


def _lm_eval_prompt_gen(config):
    """Select the lm_eval prompt generator.

    For the remote backend, generates a single-pod run-iteration that does
    compile + bench + baseline + lm_eval sequentially in one GPU pod.
    """
    if config.runtime_backend == "remote":

        def _remote(
            context,
            execution_command,
            runtime_logs_dir,
            iteration,
            gpu_wait_timeout_minutes=60,
            smaller_model_file=None,
        ):
            return gen_RemoteRunIterationPrompt(
                context=context,
                execution_command=execution_command,
                runtime_logs_dir=runtime_logs_dir,
                iteration=iteration,
                remote_namespace=config.remote_namespace,
                remote_pvc_name=config.remote_pvc_name,
                remote_model_pvc_name=config.remote_model_pvc_name,
                remote_image=config.remote_image,
                remote_gpu_count=config.remote_gpu_count,
                remote_gpu_product=config.remote_gpu_product,
                disable_env_var=_read_feature_disable_env_var(config.output_dir) or "",
                run_lm_eval=True,
                disable_new_feature=False,
                gpu_wait_timeout_minutes=gpu_wait_timeout_minutes,
            )

        return _remote
    return gen_RunLMEvalPrompt


def _diagnostic_prompt_gen(config):
    """Select the diagnostic profiling prompt generator.

    Remote: the GPU Job runs only nsys (Steps 0-1). Steps 2-5 (extraction,
    correlation, comparison, report) run locally in the same Claude session
    so the full reasoning history is preserved.
    Local: the original prompt, unchanged.
    """
    if config.runtime_backend == "remote":

        def _remote(
            context,
            execution_command,
            delay_iterations,
            feature_disable_env_var,
            original_analysis_dir,
            runtime_logs_dir,
            iteration,
            gpu_wait_timeout_minutes=60,
        ):
            return gen_RemoteRunDiagnosticPrompt(
                context=context,
                execution_command=execution_command,
                delay_iterations=delay_iterations,
                feature_disable_env_var=feature_disable_env_var,
                original_analysis_dir=original_analysis_dir,
                runtime_logs_dir=runtime_logs_dir,
                iteration=iteration,
                gpu_wait_timeout_minutes=gpu_wait_timeout_minutes,
                remote_namespace=config.remote_namespace,
                remote_pvc_name=config.remote_pvc_name,
                remote_model_pvc_name=config.remote_model_pvc_name,
                remote_image=config.remote_image,
                remote_gpu_count=config.remote_gpu_count,
                remote_gpu_product=config.remote_gpu_product,
            )

        return _remote
    return gen_DiagnosticProfileAndAnalyzePrompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AI code generation pipeline")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the code generation JSON config file.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last incomplete step instead of starting from scratch.",
    )
    from llm.auto_code_gen.agents import add_agent_argument

    add_agent_argument(parser)
    return parser.parse_args()


def _find_latest_patch(output_dir):
    """Find the latest code patch file from code gen or runtime iterations.

    All patches are standalone (applicable on a clean base commit).
    Returns the most recently modified patch path relative to output_dir
    (e.g., ``code_gen/V2_review.patch`` or ``runtime/V3/patch.patch``).
    """
    candidates = []

    # Scan code_gen/ for V{N}.patch and V{N}_review.patch
    code_gen_path = os.path.join(output_dir, CODE_GEN_DIR)
    if os.path.isdir(code_gen_path):
        pat = re.compile(r"^V(\d+)(?:_review)?\.patch$")
        for f in os.listdir(code_gen_path):
            if pat.match(f):
                path = os.path.join(code_gen_path, f)
                rel = os.path.join(CODE_GEN_DIR, f)
                candidates.append((os.path.getmtime(path), rel))

    # Scan runtime/V*/ for patch.patch
    runtime_path = os.path.join(output_dir, RUNTIME_DIR)
    if os.path.isdir(runtime_path):
        dir_pat = re.compile(r"^V(\d+)$")
        for d in os.listdir(runtime_path):
            if dir_pat.match(d):
                patch_file = os.path.join(runtime_path, d, "patch.patch")
                if os.path.isfile(patch_file):
                    rel = os.path.join(RUNTIME_DIR, d, "patch.patch")
                    candidates.append((os.path.getmtime(patch_file), rel))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _find_latest_runtime_iteration(output_dir):
    """Find the latest completed runtime iteration number (0 if none exist).

    An iteration is considered complete when it has a summary file
    (summary.txt or summary_draft_R*.txt), indicating the investigate
    step finished. patch.patch alone is not sufficient — it is copied
    at the START of each iteration.
    """
    runtime_path = os.path.join(output_dir, RUNTIME_DIR)
    if not os.path.isdir(runtime_path):
        return 0
    pattern = re.compile(r"^V(\d+)$")
    max_ver = 0
    for d in os.listdir(runtime_path):
        m = pattern.match(d)
        if m:
            iter_dir = os.path.join(runtime_path, d)
            has_summary = os.path.isfile(os.path.join(iter_dir, "summary.txt"))
            if not has_summary:
                import glob as _glob

                has_summary = bool(
                    _glob.glob(os.path.join(iter_dir, "summary_draft_R*.txt"))
                )
            if has_summary:
                ver = int(m.group(1))
                if ver > max_ver:
                    max_ver = ver
    return max_ver


def _check_apply_succeeded(output_dir, iteration):
    """Check if apply and compile succeeded for the given runtime iteration."""
    status_file = os.path.join(
        output_dir, RUNTIME_DIR, f"V{iteration}", "apply_status.txt"
    )
    if not os.path.isfile(status_file):
        return False
    with open(status_file) as f:
        first_line = f.readline().strip()
    return first_line == "SUCCESS"


def _check_runtime_success(output_dir, current_patch_file=None):
    """Check if runtime iteration resulted in a successful benchmark run.

    If current_patch_file is provided, also verifies the success file was
    generated for this specific patch version (not an outdated one).
    """
    success_path = os.path.join(output_dir, RUNTIME_DIR, RUNTIME_SUCCESS_FILE)
    if not os.path.isfile(success_path):
        return False
    if current_patch_file is None:
        return True
    with open(success_path) as f:
        content = f.read()
    if current_patch_file in content:
        return True
    print("  Note: existing success file is for a different patch version, re-running.")
    return False


def _check_run_succeeded(runtime_logs_dir, iteration=None):
    """Check if the RunAndLog patched run completed successfully.

    runtime_logs_dir is the per-iteration directory (e.g. runtime/V3/).
    The iteration parameter is accepted for call-site compatibility but unused.
    """
    status_file = os.path.join(runtime_logs_dir, "run_status.txt")
    if not os.path.isfile(status_file):
        return False
    with open(status_file) as f:
        return f.readline().strip() == "COMPLETED"


def _read_expected_improvement_pct(output_dir):
    """Read expected improvement percentage from the LLM-extracted JSON file.

    Returns None if the file doesn't exist or can't be parsed.
    """
    filepath = os.path.join(output_dir, RUNTIME_DIR, RUNTIME_EXPECTED_IMPROVEMENT_FILE)
    if not os.path.isfile(filepath):
        return None
    try:
        with open(filepath) as f:
            data = json.load(f)
        return data.get("expected_improvement_pct")
    except OSError, json.JSONDecodeError, KeyError:
        return None


# Supported `vllm bench` output dialects, most preferred first. `vllm bench
# serve --result-dir/--result-filename` writes output_throughput; the older
# `vllm bench throughput --output-json` wrote tokens_per_second. Reading only
# one of the two names yields 0 from the other, and a gate built on that
# silently passes whatever it measured -- including a 24% regression.
BENCH_THROUGHPUT_KEYS = (
    "output_throughput",
    "total_token_throughput",
    "tokens_per_second",
    "request_throughput",
)


def _bench_throughput_pair(patched, baseline):
    """Return (patched_tps, baseline_tps, key) for the first metric both share.

    Both sides must report the same key, otherwise the comparison is between
    two different quantities. Returns (None, None, None) when no usable metric
    exists in both files.
    """
    for key in BENCH_THROUGHPUT_KEYS:
        patched_tps = patched.get(key)
        baseline_tps = baseline.get(key)
        if (
            isinstance(patched_tps, (int, float))
            and isinstance(baseline_tps, (int, float))
            and patched_tps > 0
            and baseline_tps > 0
        ):
            return float(patched_tps), float(baseline_tps), key
    return None, None, None


def _check_performance_meets_target(runtime_logs_dir, iteration, expected_pct):
    """Check if benchmark throughput improvement meets the performance target.

    Reads patched and baseline bench result JSONs, computes throughput delta,
    and compares against threshold (PERF_GATE_DILUTION_FACTOR * expected_pct).
    Returns (passed, actual_pct, threshold_pct).

    Fails closed: a gate that cannot measure cannot certify an improvement, so a
    missing/unreadable/unknown-metric result is reported as not meeting the
    target and the pipeline investigates instead of declaring success.
    """
    if expected_pct is None:
        # No expectation was extracted for this plan, so there is nothing to
        # gate against -- this is the only case allowed to pass unmeasured.
        return (True, 0.0, 0.0)

    threshold_pct = expected_pct * PERF_GATE_DILUTION_FACTOR

    patched_file = os.path.join(runtime_logs_dir, "bench_result.json")
    baseline_file = os.path.join(runtime_logs_dir, "baseline_bench_result.json")

    try:
        with open(patched_file) as f:
            patched = json.load(f)
        with open(baseline_file) as f:
            baseline = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"  Perf gate: bench results unreadable ({e}), failing closed")
        return (False, 0.0, threshold_pct)

    patched_tps, baseline_tps, key = _bench_throughput_pair(patched, baseline)
    if patched_tps is None:
        print(
            f"  Perf gate: no shared throughput metric in {os.path.basename(patched_file)} / {os.path.basename(baseline_file)}, failing closed"
        )
        return (False, 0.0, threshold_pct)

    actual_pct = (patched_tps - baseline_tps) / baseline_tps * 100
    print(f"  Perf gate: {key} {patched_tps:.2f} vs {baseline_tps:.2f} tok/s")
    return (actual_pct >= threshold_pct, actual_pct, threshold_pct)


def _collect_previous_perf_results(results_dir, current_iteration):
    """Collect throughput deltas from all previous perf gate and comparison files.

    results_dir is the top-level runtime directory (e.g. ``runtime/``).
    """
    results = []
    for i in range(1, current_iteration):
        iter_dir = os.path.join(results_dir, f"V{i}")
        gate_file = os.path.join(iter_dir, "perf_gate.txt")
        bench_file = os.path.join(iter_dir, "bench_result.json")
        baseline_file = os.path.join(iter_dir, "baseline_bench_result.json")

        delta_str = "unknown"
        try:
            with open(bench_file) as f:
                patched = json.load(f)
            with open(baseline_file) as f:
                baseline = json.load(f)
            patched_tps, baseline_tps, _ = _bench_throughput_pair(patched, baseline)
            if patched_tps is not None:
                delta = (patched_tps - baseline_tps) / baseline_tps * 100
                delta_str = f"{delta:+.2f}%"
        except OSError, json.JSONDecodeError:
            pass

        gate_rejected = os.path.isfile(gate_file)
        results.append(
            "  V{}: {} {}".format(
                i,
                delta_str,
                "(REJECTED by perf gate)" if gate_rejected else "(REGRESSION)",
            )
        )
    return results


def _write_perf_gate_feedback(
    runtime_logs_dir, iteration, actual_pct, threshold_pct, expected_pct
):
    """Write performance gate feedback file for the investigate prompt.

    runtime_logs_dir is the per-iteration directory (e.g. ``runtime/V3/``).
    """
    feedback_file = os.path.join(runtime_logs_dir, "perf_gate.txt")

    # Derive the parent runtime dir for collecting previous iteration results
    results_dir = os.path.dirname(runtime_logs_dir)
    previous_results = _collect_previous_perf_results(results_dir, iteration)

    with open(feedback_file, "w") as f:
        f.write(f"PERFORMANCE GATE FAILED — Iteration V{iteration}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Actual throughput improvement:   {actual_pct:+.2f}%\n")
        f.write(f"Required threshold:              +{threshold_pct:.2f}%\n")
        f.write(
            f"Expected kernel-level impact:    +{expected_pct:.1f}% (from improvement plan)\n"
        )
        f.write(
            f"Dilution factor applied:         {PERF_GATE_DILUTION_FACTOR}x (kernel → end-to-end)\n\n"
        )

        if previous_results:
            f.write("PREVIOUS ITERATION RESULTS:\n")
            for line in previous_results:
                f.write(line + "\n")
            f.write(
                f"  V{iteration}: +{actual_pct:.2f}% (REJECTED by perf gate)  ← current\n\n"
            )

        f.write("This is a PERFORMANCE INVESTIGATION, not a bug fix.\n")
        f.write("The code compiles, runs, and produces correct results.\n")
        f.write("The problem is insufficient performance improvement.\n\n")
        f.write("A kernel-level diagnostic profile will be generated next.\n")
        f.write("It will identify the KEY REASONS for the insufficient performance.\n")
        f.write("The next fix attempt must address those specific reasons.\n")


def _generate_fallback_summary(output_dir, runtime_logs_dir, iteration):
    """Generate a basic summary.txt when the investigation agent doesn't produce one."""
    summary_path = os.path.join(runtime_logs_dir, "summary.txt")
    lines = [
        f"Runtime Iteration V{iteration} Summary (auto-generated)",
        "=" * 60,
        "",
    ]

    status_file = os.path.join(runtime_logs_dir, "run_status.txt")
    if os.path.isfile(status_file):
        with open(status_file) as f:
            lines.append(f"Status: {f.read().strip()}")
    else:
        lines.append("Status: UNKNOWN")

    perf_file = os.path.join(runtime_logs_dir, "perf_comparison.txt")
    if os.path.isfile(perf_file):
        lines.append("")
        lines.append("Performance Comparison:")
        with open(perf_file) as f:
            lines.append(f.read().strip())

    lm_eval_file = os.path.join(runtime_logs_dir, "lm_eval.txt")
    if os.path.isfile(lm_eval_file):
        lines.append("")
        lines.append(f"Correctness (lm_eval): {open(lm_eval_file).read().strip()}")

    diag_report = os.path.join(
        runtime_logs_dir, "diagnostic_profile", "diagnostic_report.txt"
    )
    if os.path.isfile(diag_report):
        lines.append("")
        lines.append(
            "Diagnostic profile: available at diagnostic_profile/diagnostic_report.txt"
        )

    patch_file = os.path.join(runtime_logs_dir, "patch.patch")
    if os.path.isfile(patch_file):
        with open(patch_file) as f:
            patch_lines = f.readlines()
        lines.append("")
        lines.append(f"Patch: {len(patch_lines)} lines")

    lines.append("")
    lines.append(
        "(This summary was auto-generated because the investigation agent "
        "did not produce one.)"
    )

    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Auto-generated fallback summary: {summary_path}")


def _format_duration(seconds):
    mins, secs = divmod(int(seconds), 60)
    hrs, mins = divmod(mins, 60)
    raw = f"{seconds:.1f}s"
    if hrs:
        human = f"{hrs}h {mins}m {secs}s"
    else:
        human = f"{mins}m {secs}s"
    return raw, human


def _read_feature_disable_env_var(output_dir):
    """Read the feature disable env var name from code gen summaries."""
    import glob
    import re as _re

    pattern = _re.compile(r"(VLLM_\w+)", _re.IGNORECASE)

    # Scan code_gen/ and runtime/V*/ for summary files
    summary_files = sorted(
        glob.glob(os.path.join(output_dir, CODE_GEN_DIR, "*summary*.txt"))
        + glob.glob(os.path.join(output_dir, RUNTIME_DIR, "V*", "summary.txt")),
        key=os.path.getmtime,
        reverse=True,
    )
    for summary in summary_files:
        try:
            with open(summary) as f:
                content = f.read()
            for line in content.splitlines():
                lower = line.lower()
                if "disable" in lower and (
                    "env" in lower or "variable" in lower or "toggle" in lower
                ):
                    match = pattern.search(line)
                    if match:
                        return match.group(1)
        except OSError:
            continue

    # Scan code_gen/ and runtime/V*/ for patch files
    patch_files = sorted(
        glob.glob(os.path.join(output_dir, CODE_GEN_DIR, "*.patch"))
        + glob.glob(os.path.join(output_dir, RUNTIME_DIR, "V*", "patch.patch")),
        key=os.path.getmtime,
        reverse=True,
    )
    for patch_file in patch_files:
        try:
            with open(patch_file) as f:
                for line in f:
                    if "VLLM_DISABLE_" in line:
                        match = _re.search(r"(VLLM_DISABLE_\w+)", line)
                        if match:
                            return match.group(1)
        except OSError:
            continue
    return None


def _should_run_diagnostic(iteration, runtime_logs_dir, original_analysis_dir):
    """Check if diagnostic profiling should run for this iteration.

    runtime_logs_dir is the per-iteration directory (e.g. ``runtime/V3/``).
    """
    gate_file = os.path.join(runtime_logs_dir, "perf_gate.txt")
    if not os.path.isfile(gate_file):
        return False
    diagnostic_dir = os.path.join(runtime_logs_dir, DIAGNOSTIC_PROFILE_DIR)
    report_file = os.path.join(diagnostic_dir, DIAGNOSTIC_REPORT_FILE)
    if os.path.isfile(report_file):
        return False
    if not original_analysis_dir or not os.path.isdir(original_analysis_dir):
        return False
    return True


NSYS_MAX_TRACE_ITERATIONS = 50
# Headroom so the capture always closes inside the measured workload: a window
# that starts too late is truncated, or never opens at all, which reads as
# "no kernels captured" in the analysis step.
NSYS_CAPTURE_MARGIN = 50


def _bench_serve_step_budget(command):
    """Engine steps a `vllm bench serve` run generates, or 0 if unknowable.

    Under --max-num-seqs 1 requests run one at a time and every generated
    token is one engine step, so num_prompts * output_len is the step count
    the profiler window has to fit inside.
    """
    tokens = shlex.split(command or "")
    if not any(tokens[i : i + 2] == ["bench", "serve"] for i in range(len(tokens) - 1)):
        return 0

    def _int_flag(*names):
        for i, token in enumerate(tokens):
            for name in names:
                if token == name and i + 1 < len(tokens):
                    try:
                        return int(tokens[i + 1])
                    except ValueError:
                        return 0
                if token.startswith(name + "="):
                    try:
                        return int(token.split("=", 1)[1])
                    except ValueError:
                        return 0
        return 0

    return _int_flag("--num-prompts") * _int_flag("--random-output-len", "--output-len")


def _compute_delay_iterations(config, execution_command=None):
    """Choose a profiling delay within the workload's active capture window."""
    command = (
        execution_command
        if execution_command is not None
        else getattr(config, "target_run_command", "")
    )
    tokens = shlex.split(command or "")
    if any(tokens[i : i + 2] == ["bench", "latency"] for i in range(len(tokens) - 1)):
        # vLLM latency warms up before start_profile(), then profiles ONE batch.
        # Server-style multi-wave delays can exceed this entire generation.
        output_len = 128  # vllm bench latency default
        for i, token in enumerate(tokens):
            if token == "--output-len" and i + 1 < len(tokens):
                output_len = int(tokens[i + 1])
            elif token.startswith("--output-len="):
                output_len = int(token.split("=", 1)[1])
        if output_len <= 0:
            raise ValueError("Profiling requires a positive --output-len")
        # Keep the runner's capture inside the generated sequence.
        return min(output_len // 2, max(0, output_len - NSYS_MAX_TRACE_ITERATIONS))

    output_len = getattr(config, "output_size", None)
    if output_len is None:
        try:
            output_len = config.traces[config.target_trace_id].output_size_range[1]
        except AttributeError, IndexError, TypeError:
            output_len = 1024
    num_warmups = getattr(config, "num_warmups", 0)
    num_waves = getattr(config, "num_waves", 4)
    delay = (
        (num_warmups * output_len) + ((num_waves // 2) * output_len) + (output_len // 2)
    )

    # The wave estimate comes from the analysis config, which need not match
    # the bench command actually submitted. Clamp it to the real workload so
    # the capture window can always open and close.
    budget = _bench_serve_step_budget(command)
    if budget > 0:
        latest_start = max(0, budget - NSYS_MAX_TRACE_ITERATIONS - NSYS_CAPTURE_MARGIN)
        delay = min(delay, latest_start)
    return delay


def _resolve_original_analysis_dir(config):
    """Resolve the original single-trace analysis directory for the target framework.

    The NSYS diagnostic analysis needs the reference artifacts (gpu_ops.txt,
    median_block_V*.txt, transformer_block_high_level_ops_V*.txt) that the
    original trace analysis wrote. For a cross-trace run those live in the
    target trace's own result directory, and the config already names a file in
    it -- the target framework's transformer-block review. The older
    ``<cross config dir>/analyze_<framework>`` layout is kept as a fallback.
    Returning None used to be common for cross-trace configs, which silently
    skipped the profiling step and left the investigate agent without any
    kernel-level evidence.
    """
    target_fw = getattr(config, "target_framework", None)
    ops_files = getattr(config, "transformer_block_high_level_ops_files", None) or []
    names = getattr(config, "framework_names", None) or []
    for index, ops_file in enumerate(ops_files):
        if target_fw and index < len(names) and names[index] != target_fw:
            continue
        analysis_dir = os.path.dirname(str(ops_file))
        if analysis_dir and os.path.isdir(analysis_dir):
            return analysis_dir

    cross_config_path = getattr(config, "cross_trace_config_path", None)
    if not cross_config_path:
        return None
    try:
        cross_dir = os.path.dirname(os.path.abspath(cross_config_path))
        analysis_dir = os.path.join(cross_dir, f"analyze_{target_fw}")
        if os.path.isdir(analysis_dir):
            return analysis_dir
    except AttributeError, TypeError:
        pass
    return None


def _format_tokens(count):
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count / 1_000:.1f}K"
    return str(count)


def _sum_costs(costs):
    """A total is unknown when any constituent provider cost is unreported."""
    values = list(costs)
    return None if any(cost is None for cost in values) else sum(values)


def _format_cost(cost):
    return "unknown" if cost is None else f"${cost:.4f}"


def _normalize_timing(s):
    """Convert a timing entry to a dict. Handles both dict and (name, duration) tuple formats."""
    if isinstance(s, dict):
        return s
    return {
        "name": s[0],
        "duration": s[1],
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
    }


def _print_detailed_table(all_step_timings, total_duration, output_dir=None):
    all_step_timings = [_normalize_timing(s) for s in all_step_timings]
    total_input = sum(s["input_tokens"] for s in all_step_timings)
    total_output = sum(s["output_tokens"] for s in all_step_timings)
    total_cost = _sum_costs(s["cost_usd"] for s in all_step_timings)

    rows = []
    for s in all_step_timings:
        raw, human = _format_duration(s["duration"])
        rows.append(
            {
                "name": s["name"],
                "dur_raw": raw,
                "dur_human": human,
                "input": _format_tokens(s["input_tokens"]),
                "output": _format_tokens(s["output_tokens"]),
                "cost": _format_cost(s["cost_usd"]),
            }
        )

    total_raw, total_human = _format_duration(total_duration)
    rows.append(
        {
            "name": "TOTAL",
            "dur_raw": total_raw,
            "dur_human": total_human,
            "input": _format_tokens(total_input),
            "output": _format_tokens(total_output),
            "cost": _format_cost(total_cost),
        }
    )

    max_dur_len = max(len(r["dur_raw"]) for r in rows)
    max_in_len = max(len(r["input"]) for r in rows)
    max_out_len = max(len(r["output"]) for r in rows)
    w = 120

    lines = []
    lines.append("\n" + "=" * w)
    lines.append("DETAILED STEP TIMINGS")
    if any(t.get("cost_unknown") for t in all_step_timings):
        lines.append(
            "COST INCOMPLETE: USD values include known charges only; some provider usage is unpriced."
        )
    if any(t.get("usage_incomplete") for t in all_step_timings):
        lines.append(
            "USAGE INCOMPLETE: token counts/cost estimates may omit failed requests or use estimated streaming usage."
        )
    lines.append("=" * w)
    lines.append(
        "{:<35s} {:<25s} {:<12s} {:<12s} {}".format(
            "Step", "Duration", "In Tokens", "Out Tokens", "Cost"
        )
    )
    lines.append("-" * w)

    for i, r in enumerate(rows):
        if i == len(rows) - 1:
            lines.append("-" * w)
        dur_str = "{:<{}}  ({})".format(r["dur_raw"], max_dur_len, r["dur_human"])
        lines.append(
            "{:<35s} {:<25s} {:<12s} {:<12s} {}".format(
                r["name"],
                dur_str,
                r["input"],
                r["output"],
                r["cost"],
            )
        )

    lines.append("=" * w)

    # Print to stdout
    for line in lines:
        print(line)

    # Write to file if output_dir is provided
    if output_dir:
        output_file = os.path.join(output_dir, "output.txt")
        with open(output_file, "w") as f:
            f.write("\n".join(lines) + "\n")


def _print_phase_summary(
    phase_results, all_step_timings, total_duration, output_dir=None
):
    all_step_timings = [_normalize_timing(s) for s in all_step_timings]
    phase_usage = {}
    for s in all_step_timings:
        name = s["name"]
        _PHASE_STEP_PREFIXES = {
            "Code traces": ["code_trace_", "clear_target_repo"],
            "Port Tests": ["test_port"],
            "Test Validation": ["test_validation"],
            "Baseline Test": ["baseline_test"],
            "Code port plan": ["code_port_plan"],
            "Test plan": ["test_plan"],
            "Code gen": [
                "code_gen_V",
                "code_gen_review",
                "clear_target_dir_code_gen",
                "clear_target_repo_code_gen",
            ],
            "Runtime": [
                "apply_compile_",
                "apply_V",
                "clear_V",
                "bench_on_V",
                "bench_off_V",
                "bench_V",
                "lm_eval_on_V",
                "lm_eval_off_V",
                "lm_eval_V",
                "nsys_V",
                "nsys_analysis_V",
                "run_log_",
                "investigate_runtime_",
                "investigate_V",
                "review_V",
                "cumulative_summary_V",
                "clear_runtime_",
                "run_and_fix",
            ],
            "Baseline Diff": ["baseline_diff"],
        }
        for phase_name, prefixes in _PHASE_STEP_PREFIXES.items():
            if any(name.startswith(p) for p in prefixes):
                phase_usage.setdefault(
                    phase_name, {"input": 0, "output": 0, "cost": 0.0}
                )
                phase_usage[phase_name]["input"] += s["input_tokens"]
                phase_usage[phase_name]["output"] += s["output_tokens"]
                phase_usage[phase_name]["cost"] = _sum_costs(
                    [phase_usage[phase_name]["cost"], s["cost_usd"]]
                )
                break

    total_input = sum(s["input_tokens"] for s in all_step_timings)
    total_output = sum(s["output_tokens"] for s in all_step_timings)
    total_cost = _sum_costs(s["cost_usd"] for s in all_step_timings)

    rows = []
    for phase in phase_results:
        if phase["max_iterations"] <= 1:
            converged_str = "N/A"
        elif phase["converged"]:
            converged_str = "Yes (iter {})".format(phase["iterations"])
        else:
            converged_str = "No"
        raw, human = _format_duration(phase["duration"])
        usage = phase_usage.get(phase["name"], {"input": 0, "output": 0, "cost": 0.0})
        rows.append(
            {
                "name": phase["name"],
                "iters": "{}/{}".format(phase["iterations"], phase["max_iterations"]),
                "conv": converged_str,
                "dur_raw": raw,
                "dur_human": human,
                "input": _format_tokens(usage["input"]),
                "output": _format_tokens(usage["output"]),
                "cost": _format_cost(usage["cost"]),
            }
        )

    total_raw, total_human = _format_duration(total_duration)
    rows.append(
        {
            "name": "TOTAL",
            "iters": "",
            "conv": "",
            "dur_raw": total_raw,
            "dur_human": total_human,
            "input": _format_tokens(total_input),
            "output": _format_tokens(total_output),
            "cost": _format_cost(total_cost),
        }
    )

    max_dur_len = max(len(r["dur_raw"]) for r in rows)
    w = 120

    lines = []
    lines.append("\n" + "=" * w)
    lines.append("PHASE SUMMARY")
    if any(t.get("cost_unknown") for t in all_step_timings):
        lines.append(
            "COST INCOMPLETE: USD values include known charges only; some provider usage is unpriced."
        )
    if any(t.get("usage_incomplete") for t in all_step_timings):
        lines.append(
            "USAGE INCOMPLETE: token counts/cost estimates may omit failed requests or use estimated streaming usage."
        )
    lines.append("=" * w)
    lines.append(
        "{:<20s} {:<10s} {:<14s} {:<25s} {:<12s} {:<12s} {}".format(
            "Phase", "Iters", "Converged", "Duration", "In Tokens", "Out Tokens", "Cost"
        )
    )
    lines.append("-" * w)

    for i, r in enumerate(rows):
        if i == len(rows) - 1:
            lines.append("-" * w)
        dur_str = "{:<{}}  ({})".format(r["dur_raw"], max_dur_len, r["dur_human"])
        lines.append(
            "{:<20s} {:<10s} {:<14s} {:<25s} {:<12s} {:<12s} {}".format(
                r["name"],
                r["iters"],
                r["conv"],
                dur_str,
                r["input"],
                r["output"],
                r["cost"],
            )
        )

    lines.append("=" * w)

    # Print to stdout
    for line in lines:
        print(line)

    # Append to file if output_dir is provided
    if output_dir:
        output_file = os.path.join(output_dir, "output.txt")
        with open(output_file, "a") as f:
            f.write("\n".join(lines) + "\n")


def _print_timeline_and_costs(
    all_step_timings, total_duration, pipeline_start_time, output_dir=None
):
    """Print a comprehensive timeline showing every step with timestamps, costs, and overhead."""
    from datetime import datetime

    all_step_timings = [_normalize_timing(s) for s in all_step_timings]
    total_step_time = sum(s["duration"] for s in all_step_timings)
    total_overhead = total_duration - total_step_time
    total_cost = _sum_costs(s.get("cost_usd") for s in all_step_timings)
    total_input = sum(s.get("input_tokens", 0) for s in all_step_timings)
    total_output = sum(s.get("output_tokens", 0) for s in all_step_timings)

    def _ts(epoch):
        return datetime.fromtimestamp(epoch, tz=UTC).strftime("%H:%M:%S")

    w = 130
    lines = []
    lines.append("\n" + "=" * w)
    lines.append("TIMELINE")
    if any(t.get("cost_unknown") for t in all_step_timings):
        lines.append(
            "COST INCOMPLETE: USD values include known charges only; some provider usage is unpriced."
        )
    if any(t.get("usage_incomplete") for t in all_step_timings):
        lines.append(
            "USAGE INCOMPLETE: token counts/cost estimates may omit failed requests or use estimated streaming usage."
        )
    lines.append("=" * w)
    lines.append(
        "{:<12s} {:<12s} {:<40s} {:<15s} {:<12s} {:<12s} {}".format(
            "Start", "End", "Step", "Duration", "In Tokens", "Out Tokens", "Cost"
        )
    )
    lines.append("-" * w)

    prev_end = pipeline_start_time
    for s in all_step_timings:
        start = s.get("start_time", prev_end)
        end = s.get("end_time", start + s["duration"])

        gap = start - prev_end
        if gap > 1.0:
            gap_raw, gap_human = _format_duration(gap)
            lines.append(
                "{:<12s} {:<12s} {:<40s} {:<15s}".format(
                    _ts(prev_end),
                    _ts(start),
                    "  [overhead]",
                    f"({gap_human})",
                )
            )

        dur_raw, dur_human = _format_duration(s["duration"])
        lines.append(
            "{:<12s} {:<12s} {:<40s} {:<15s} {:<12s} {:<12s} {}".format(
                _ts(start),
                _ts(end),
                s["name"],
                dur_human,
                _format_tokens(s.get("input_tokens", 0)),
                _format_tokens(s.get("output_tokens", 0)),
                _format_cost(s.get("cost_usd")),
            )
        )
        prev_end = end

    lines.append("-" * w)

    total_raw, total_human = _format_duration(total_duration)
    step_raw, step_human = _format_duration(total_step_time)
    overhead_raw, overhead_human = _format_duration(total_overhead)

    lines.append("")
    lines.append("COST & TIME SUMMARY")
    lines.append("-" * 60)
    lines.append(f"  Wall clock time:      {total_human}")
    lines.append(f"  Step execution time:  {step_human}")
    lines.append(
        f"  Overhead time:        {overhead_human} ({(total_overhead / total_duration * 100) if total_duration > 0 else 0:.1f}% of wall)"
    )
    lines.append(f"  Number of steps:      {len(all_step_timings)}")
    lines.append("")
    lines.append(f"  Input tokens:         {_format_tokens(total_input)}")
    lines.append(f"  Output tokens:        {_format_tokens(total_output)}")
    lines.append(
        "  Total cost:           {}".format(
            _format_cost(total_cost) if total_cost is None else f"${total_cost:.2f}"
        )
    )
    lines.append("")

    if total_cost > 0 and total_duration > 0:
        lines.append(
            f"  Cost per hour:        ${total_cost / (total_duration / 3600):.2f}/hr"
        )
    lines.append("=" * w)

    for line in lines:
        print(line)

    if output_dir:
        output_file = os.path.join(output_dir, "timing_summary.txt")
        with open(output_file, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nTiming summary saved to: {output_file}")


async def run_pipeline(config, claude_config, use_case, resume=False, client=None):
    output_dir = config.output_dir
    context = use_case.create_context_str(claude_config, config)
    phase_results = []
    all_step_timings = []

    enable_test_port = getattr(config, "enable_test_port", False)
    enable_baseline_diff = getattr(config, "enable_baseline_diff", False)

    # Phase 0: Baseline test snapshot (before any patch — do-no-harm check)
    if enable_baseline_diff and hasattr(use_case, "gen_baseline_test_steps"):
        phase_start = time.time()
        print("\n" + "=" * 80)
        print("PHASE 0: BASELINE TEST SNAPSHOT")
        print("=" * 80)
        baseline_steps, baseline_file = use_case.gen_baseline_test_steps(
            context, config
        )
        if resume and _all_steps_complete(baseline_steps, output_dir):
            print("Resume: skipping completed phase 'Baseline Test'")
        else:
            timings = await agent_run(
                claude_config, baseline_steps, resume=resume, client=client
            )
            all_step_timings.extend(timings)
        phase_results.append(
            {
                "name": "Baseline Test",
                "iterations": 1,
                "max_iterations": 1,
                "converged": False,
                "duration": time.time() - phase_start,
            }
        )

    # Phase 1: Code traces
    phase_start = time.time()
    code_trace_steps, code_trace_files = use_case.gen_code_trace_steps(context, config)
    if resume and _all_steps_complete(code_trace_steps, output_dir):
        print("Resume: skipping completed phase 'Code traces'")
    else:
        timings = await agent_run(
            claude_config, code_trace_steps, resume=resume, client=client
        )
        all_step_timings.extend(timings)
    phase_results.append(
        {
            "name": "Code traces",
            "iterations": 1,
            "max_iterations": 1,
            "converged": False,
            "duration": time.time() - phase_start,
        }
    )

    # Phase 1.5: Port tests (dedicated step before planning)
    if enable_test_port and hasattr(use_case, "gen_test_port_steps"):
        phase_start = time.time()
        print("\n" + "=" * 80)
        print("PHASE 1.5: PORT TESTS FROM SOURCE COMMIT")
        print("=" * 80)
        test_port_steps, test_port_manifest = use_case.gen_test_port_steps(
            context, config
        )
        if resume and _all_steps_complete(test_port_steps, output_dir):
            print("Resume: skipping completed phase 'Port Tests'")
        else:
            timings = await agent_run(
                claude_config, test_port_steps, resume=resume, client=client
            )
            all_step_timings.extend(timings)
        phase_results.append(
            {
                "name": "Port Tests",
                "iterations": 1,
                "max_iterations": 1,
                "converged": False,
                "duration": time.time() - phase_start,
            }
        )

        # Phase 1.7: Test validation — red check (tests must FAIL without fix)
        if hasattr(use_case, "gen_test_validation_steps"):
            phase_start = time.time()
            print("\n" + "=" * 80)
            print("PHASE 1.7: TEST VALIDATION (RED CHECK)")
            print("=" * 80)
            test_val_steps, test_val_file = use_case.gen_test_validation_steps(
                context, config
            )
            if resume and _all_steps_complete(test_val_steps, output_dir):
                print("Resume: skipping completed phase 'Test Validation'")
            else:
                timings = await agent_run(
                    claude_config, test_val_steps, resume=resume, client=client
                )
                all_step_timings.extend(timings)
            phase_results.append(
                {
                    "name": "Test Validation",
                    "iterations": 1,
                    "max_iterations": 1,
                    "converged": False,
                    "duration": time.time() - phase_start,
                }
            )

    # Phase 2: Code port plan iterations
    phase_start = time.time()
    skip_code_port_review = config.code_port_plan_skip_review
    max_code_port_plan_iterations = (
        1 if skip_code_port_review else config.num_code_port_plan_iterations
    )
    prev_output_file = None
    prev_output_summary_file = None
    code_port_converged = False
    code_port_iterations = 0
    for i in range(max_code_port_plan_iterations):
        iteration = i + 1
        code_port_iterations = iteration
        steps, review_prompt = use_case.gen_code_port_plan_iter_steps(
            context,
            config,
            code_trace_files,
            prev_output_file,
            prev_output_summary_file,
            iteration,
        )
        run_steps = steps[:-1] if skip_code_port_review else steps

        if resume and _all_steps_complete(run_steps, output_dir):
            print(f"Resume: skipping completed iteration 'Code port plan V{iteration}'")
            if skip_code_port_review:
                prev_output_file = run_steps[-1].output_files[0]
                prev_output_summary_file = (
                    run_steps[-1].output_files[1]
                    if len(run_steps[-1].output_files) > 1
                    else None
                )
            else:
                prev_output_file = review_prompt.output_file
                prev_output_summary_file = review_prompt.output_summary_file
                if _check_converged(output_dir, review_prompt.output_summary_file):
                    print(f"Code port plan CONVERGED at iteration {iteration}")
                    code_port_converged = True
                    break
            continue

        timings = await agent_run(
            claude_config, run_steps, resume=resume, client=client
        )
        all_step_timings.extend(timings)

        if skip_code_port_review:
            prev_output_file = run_steps[-1].output_files[0]
            prev_output_summary_file = (
                run_steps[-1].output_files[1]
                if len(run_steps[-1].output_files) > 1
                else None
            )
        else:
            prev_output_file = review_prompt.output_file
            prev_output_summary_file = review_prompt.output_summary_file
            if _check_converged(output_dir, review_prompt.output_summary_file):
                print(f"Code port plan CONVERGED at iteration {iteration}")
                code_port_converged = True
                break

    final_code_port_plan_file = prev_output_file
    phase_results.append(
        {
            "name": "Code port plan",
            "iterations": code_port_iterations,
            "max_iterations": max_code_port_plan_iterations,
            "converged": code_port_converged,
            "duration": time.time() - phase_start,
        }
    )

    # Phase 3: Test plan iterations (skipped when code port plan already includes the test plan)
    if use_case.skip_test_plan_phase(config):
        print("Skipping test plan phase (combined with code port plan)")
        final_test_plan_file = final_code_port_plan_file
    else:
        phase_start = time.time()
        skip_test_review = config.test_plan_skip_review
        max_test_plan_iterations = (
            1 if skip_test_review else config.num_test_plan_iterations
        )
        prev_output_file = None
        prev_output_summary_file = None
        test_plan_converged = False
        test_plan_iterations = 0
        for i in range(max_test_plan_iterations):
            iteration = i + 1
            test_plan_iterations = iteration
            steps, review_prompt = use_case.gen_test_plan_iter_steps(
                context,
                config,
                code_trace_files,
                final_code_port_plan_file,
                prev_output_file,
                prev_output_summary_file,
                iteration,
            )
            run_steps = steps[:-1] if skip_test_review else steps

            if resume and _all_steps_complete(run_steps, output_dir):
                print(f"Resume: skipping completed iteration 'Test plan V{iteration}'")
                if skip_test_review:
                    prev_output_file = run_steps[-1].output_files[0]
                    prev_output_summary_file = (
                        run_steps[-1].output_files[1]
                        if len(run_steps[-1].output_files) > 1
                        else None
                    )
                else:
                    prev_output_file = review_prompt.output_file
                    prev_output_summary_file = review_prompt.output_summary_file
                    if _check_converged(output_dir, review_prompt.output_summary_file):
                        print(f"Test plan CONVERGED at iteration {iteration}")
                        test_plan_converged = True
                        break
                continue

            timings = await agent_run(
                claude_config, run_steps, resume=resume, client=client
            )
            all_step_timings.extend(timings)

            if skip_test_review:
                prev_output_file = run_steps[-1].output_files[0]
                prev_output_summary_file = (
                    run_steps[-1].output_files[1]
                    if len(run_steps[-1].output_files) > 1
                    else None
                )
            else:
                prev_output_file = review_prompt.output_file
                prev_output_summary_file = review_prompt.output_summary_file
                if _check_converged(output_dir, review_prompt.output_summary_file):
                    print(f"Test plan CONVERGED at iteration {iteration}")
                    test_plan_converged = True
                    break

        final_test_plan_file = prev_output_file
        phase_results.append(
            {
                "name": "Test plan",
                "iterations": test_plan_iterations,
                "max_iterations": max_test_plan_iterations,
                "converged": test_plan_converged,
                "duration": time.time() - phase_start,
            }
        )

    # Phase 4: Code gen iterations
    phase_start = time.time()
    skip_code_gen_review = config.code_gen_skip_review
    max_code_gen_iterations = (
        1 if skip_code_gen_review else config.num_code_gen_iterations
    )
    prev_output_patch_file = None
    prev_output_summary_file = None
    code_gen_converged = False
    code_gen_iterations = 0
    for i in range(max_code_gen_iterations):
        iteration = i + 1
        code_gen_iterations = iteration
        steps, code_review_prompt = use_case.gen_code_gen_iter_steps(
            context,
            config,
            code_trace_files,
            final_code_port_plan_file,
            final_test_plan_file,
            prev_output_patch_file,
            prev_output_summary_file,
            iteration,
        )
        run_steps = steps[:-1] if skip_code_gen_review else steps

        if resume and _all_steps_complete(run_steps, output_dir):
            print(f"Resume: skipping completed iteration 'Code gen V{iteration}'")
            if skip_code_gen_review:
                prev_output_patch_file = run_steps[-1].output_files[0]
                prev_output_summary_file = (
                    run_steps[-1].output_files[1]
                    if len(run_steps[-1].output_files) > 1
                    else None
                )
            else:
                prev_output_patch_file = code_review_prompt.output_patch_file
                prev_output_summary_file = code_review_prompt.output_summary_file
                if _check_converged(output_dir, code_review_prompt.output_summary_file):
                    print(f"Code gen CONVERGED at iteration {iteration}")
                    code_gen_converged = True
                    break
            continue

        timings = await agent_run(
            claude_config, run_steps, resume=resume, client=client
        )
        all_step_timings.extend(timings)

        if skip_code_gen_review:
            prev_output_patch_file = run_steps[-1].output_files[0]
            prev_output_summary_file = (
                run_steps[-1].output_files[1]
                if len(run_steps[-1].output_files) > 1
                else None
            )
        else:
            prev_output_patch_file = code_review_prompt.output_patch_file
            prev_output_summary_file = code_review_prompt.output_summary_file
            if _check_converged(output_dir, code_review_prompt.output_summary_file):
                print(f"Code gen CONVERGED at iteration {iteration}")
                code_gen_converged = True
                break

    phase_results.append(
        {
            "name": "Code gen",
            "iterations": code_gen_iterations,
            "max_iterations": max_code_gen_iterations,
            "converged": code_gen_converged,
            "duration": time.time() - phase_start,
        }
    )

    # Phase 5: Runtime iterations
    runtime_phase_results, runtime_timings = await use_case.run_runtime_iterations(
        context,
        config,
        claude_config,
        code_trace_files,
        final_code_port_plan_file,
        final_test_plan_file,
        resume=resume,
        client=client,
    )
    phase_results.extend(runtime_phase_results)
    all_step_timings.extend(runtime_timings)

    # Phase 6: Baseline diff — compare full test suite before vs after patch
    if enable_baseline_diff and hasattr(use_case, "gen_baseline_diff_steps"):
        phase_start = time.time()
        print("\n" + "=" * 80)
        print("PHASE 6: BASELINE DIFF (DO-NO-HARM CHECK)")
        print("=" * 80)
        diff_steps, diff_file = use_case.gen_baseline_diff_steps(context, config)
        if resume and _all_steps_complete(diff_steps, output_dir):
            print("Resume: skipping completed phase 'Baseline Diff'")
        else:
            timings = await agent_run(
                claude_config, diff_steps, resume=resume, client=client
            )
            all_step_timings.extend(timings)
        phase_results.append(
            {
                "name": "Baseline Diff",
                "iterations": 1,
                "max_iterations": 1,
                "converged": False,
                "duration": time.time() - phase_start,
            }
        )

    # Phase 7: Generate PR commands (optional)
    if config.generate_pr_commands:
        any_success = os.path.isfile(
            os.path.join(output_dir, RUNTIME_DIR, RUNTIME_SUCCESS_FILE)
        )
        if any_success:
            pr_results, pr_timings = await use_case.run_generate_pr_commands(
                context,
                config,
                claude_config,
                client=client,
            )
            phase_results.extend(pr_results)
            all_step_timings.extend(pr_timings)
        else:
            print("Skipping PR commands generation (no runtime succeeded)")

    return phase_results, all_step_timings


PERF_GATE_DILUTION_FACTOR = 0.3


def _cleanup_stale_cluster_resources(namespace):
    """Delete any leftover ai-auto-analyze jobs/pods/configmaps from prior runs."""
    for kind in ("jobs", "pods"):
        r = subprocess.run(
            [
                "oc",
                "get",
                kind,
                "-n",
                namespace,
                "--no-headers",
                "-o",
                "custom-columns=NAME:.metadata.name",
            ],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            continue
        for name in r.stdout.strip().splitlines():
            name = name.strip()
            if "ai-auto-analyze" in name or "pvc-helper" in name:
                print(f"  Cleanup: deleting {kind[:-1]} {name}")
                subprocess.run(
                    ["oc", "delete", kind[:-1], name, "-n", namespace, "--wait=false"],
                    capture_output=True,
                    text=True,
                )

    r = subprocess.run(
        [
            "oc",
            "get",
            "configmap",
            "-n",
            namespace,
            "--no-headers",
            "-o",
            "custom-columns=NAME:.metadata.name",
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode == 0:
        for name in r.stdout.strip().splitlines():
            name = name.strip()
            if "ai-auto-analyze" in name:
                print(f"  Cleanup: deleting configmap {name}")
                subprocess.run(
                    [
                        "oc",
                        "delete",
                        "configmap",
                        name,
                        "-n",
                        namespace,
                        "--wait=false",
                    ],
                    capture_output=True,
                    text=True,
                )
    print("  Cleanup done.")


def _reset_pvc_workspace(config):
    """Reset this run's vllm dir on the PVC to a clean state via cluster_runner."""
    commit = getattr(config, "target_commit_id", "HEAD")
    cmd = [
        sys.executable,
        _CLUSTER_RUNNER_PATH,
        "reset-workspace",
        "--namespace",
        config.remote_namespace,
        "--pvc-name",
        config.remote_pvc_name,
        "--vllm-dir",
        _pvc_vllm_dir(config),
        "--commit",
        commit,
        "--keep-cmake",
    ]
    print(f"  Resetting PVC workspace to {commit}...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  WARNING: reset-workspace failed: {r.stderr.strip()}")
    else:
        print("  PVC workspace reset OK")


async def _run_cumulative_summary(results_dir, iteration, claude_config, client=None):
    """Run LLM agent to generate/update summary_all.txt."""
    from llm.auto_code_gen.code_gen_prompts import gen_CumulativeSummaryPrompt

    prompt = gen_CumulativeSummaryPrompt(results_dir, iteration)
    timings = await agent_run(
        claude_config,
        [
            PipelineStep(
                name=f"cumulative_summary_V{iteration}", prompt=prompt.prompt()
            ),
        ],
        client=client,
    )
    return timings


async def run_runtime_iterations(
    context,
    code_trace_files,
    config,
    claude_config,
    code_port_plan_file,
    test_plan_file,
    iteration_history_summary_file,
    resume=False,
    start_iteration=None,
    smaller_model_file=None,
    disable_new_feature=False,
    execution_command=None,
    phase_label="Runtime",
    client=None,
):
    """Run the runtime iteration loop: apply → bench → check → lm_eval → nsys → investigate.

    Shared baselines (bench OFF, lm_eval OFF, nsys OFF) are captured on first need
    and reused for all subsequent iterations.

    Returns (phase_results, all_step_timings).
    """
    output_dir = config.output_dir
    results_dir = os.path.join(output_dir, RUNTIME_DIR)
    max_runtime_iterations = config.num_runtime_iterations
    os.makedirs(results_dir, exist_ok=True)

    if execution_command is None:
        execution_command = config.target_run_command

    phase_start = time.time()
    all_step_timings = []
    runtime_succeeded = False
    runtime_iterations = 0

    latest_patch = _find_latest_patch(output_dir)
    if latest_patch is None:
        print(f"ERROR: No code patch found in {output_dir}")
        return [
            {
                "name": phase_label,
                "iterations": 0,
                "max_iterations": max_runtime_iterations,
                "converged": False,
                "duration": time.time() - phase_start,
            }
        ], []

    if resume and _check_runtime_success(output_dir, latest_patch):
        print("Resume: runtime iterations already succeeded")
        return [
            {
                "name": phase_label,
                "iterations": 0,
                "max_iterations": max_runtime_iterations,
                "converged": True,
                "duration": 0.0,
            }
        ], []

    if start_iteration is not None:
        iter_start = start_iteration
    else:
        iter_start = _find_latest_runtime_iteration(output_dir) + 1
        if resume and iter_start > 1:
            print(f"Resume: continuing from V{iter_start}")

    prev_patch_file = latest_patch
    prev_summary_file = None
    if iter_start > 1:
        prev_summary = os.path.join(RUNTIME_DIR, f"V{iter_start - 1}", "summary.txt")
        if os.path.isfile(os.path.join(output_dir, prev_summary)):
            prev_summary_file = prev_summary

    if not execution_command:
        print("ERROR: No execution command available.")
        return [
            {
                "name": phase_label,
                "iterations": 0,
                "max_iterations": max_runtime_iterations,
                "converged": False,
                "duration": time.time() - phase_start,
            }
        ], []

    expected_improvement_pct = _read_expected_improvement_pct(output_dir)

    print("\n" + "=" * 80)
    print(f"RUNTIME ITERATIONS: {phase_label}")
    print("=" * 80)
    print(f"  Max iterations: {max_runtime_iterations}")
    print(f"  Starting from: V{iter_start}")
    print(f"  Patch: {prev_patch_file}")
    print(f"  Command: {execution_command}")
    if expected_improvement_pct is not None:
        print(f"  Expected: {expected_improvement_pct:.1f}%")
    print("=" * 80 + "\n")

    # Pre-loop cleanup
    if config.runtime_backend == "remote":
        _cleanup_stale_cluster_resources(config.remote_namespace)
        if iter_start == 1:
            _reset_pvc_workspace(config)

    _disable_var = ""
    if config.runtime_backend == "remote":
        _disable_var = _read_feature_disable_env_var(output_dir) or ""

    # ── Main iteration loop ──────────────────────────────────────────────────
    for i in range(iter_start - 1, max_runtime_iterations):
        iteration = i + 1
        runtime_iterations = iteration
        runtime_logs_dir = os.path.join(results_dir, f"V{iteration}")
        os.makedirs(runtime_logs_dir, exist_ok=True)

        # Save started marker + patch copy
        started_file = os.path.join(runtime_logs_dir, "started.txt")
        if not os.path.isfile(started_file):
            with open(started_file, "w") as sf:
                sf.write(
                    "STARTED V{} at {}\n".format(
                        iteration, time.strftime("%Y-%m-%d %H:%M:%S")
                    )
                )
        iter_patch_path = os.path.join(runtime_logs_dir, "patch.patch")
        prev_patch_abs = (
            os.path.join(output_dir, prev_patch_file)
            if not os.path.isabs(prev_patch_file)
            else prev_patch_file
        )
        if not os.path.isfile(iter_patch_path) and os.path.isfile(prev_patch_abs):
            shutil.copy2(prev_patch_abs, iter_patch_path)

        print(f"\n--- Runtime Iteration {iteration} / {max_runtime_iterations} ---")

        # ── STEP 1: Apply patch ──────────────────────────────────────────────
        if config.runtime_backend == "remote":
            start_t = time.time()
            apply_ok = _mechanical_apply_and_sync(
                config, output_dir, prev_patch_file, iteration
            )
            all_step_timings.append((f"apply_V{iteration}", time.time() - start_t))
            with open(os.path.join(runtime_logs_dir, "apply_status.txt"), "w") as af:
                af.write("SUCCESS\n" if apply_ok else "FAILED\n")
            if not apply_ok:
                print("  Apply FAILED, stopping.")
                break
        else:
            clear_cmd = {
                "fn": lambda: clear_vllm_source_tree(config.source_code_dir),
                "fn_name": f'clear_vllm_source_tree("{config.source_code_dir}")',
            }
            apply_prompt = _apply_prompt_gen(config)(
                context=context, patch_file=prev_patch_file, iteration=iteration
            )
            timings = await agent_run(
                claude_config,
                [
                    PipelineStep(name=f"clear_V{iteration}", prompt=clear_cmd),
                    PipelineStep(
                        name=f"apply_V{iteration}",
                        prompt=apply_prompt.prompt(),
                        output_files=[apply_prompt.output_status_file],
                    ),
                ],
                client=client,
            )
            all_step_timings.extend(timings)

        if not _check_apply_succeeded(output_dir, iteration):
            print("  Apply FAILED, stopping.")
            break

        # ── STEP 2: Compile + Bench ON (Pod 1), then Bench OFF (Pod 2) ────
        has_bench_off = _has_shared_baseline(output_dir, "bench_off")
        print(
            "  Bench ON (Pod 1){}...".format(
                "" if has_bench_off else " + OFF (Pod 2, first run)"
            )
        )

        if config.runtime_backend == "remote":
            # Pod 1: compile + warmup ON + bench ON
            start_t = time.time()
            rc, status = _run_cluster_job(
                "run-iteration",
                config,
                runtime_logs_dir,
                bench_command=execution_command,
                disable_env_var=_disable_var,
                run_lm_eval=False,
                stage="bench",
            )
            all_step_timings.append((f"bench_on_V{iteration}", time.time() - start_t))

            # Pod 2: warmup OFF + bench OFF (separate pod = full cache isolation)
            if not has_bench_off and _check_run_succeeded(runtime_logs_dir, iteration):
                print("  Bench OFF (Pod 2)...")
                baseline_dir = os.path.join(runtime_logs_dir, "baseline_run")
                start_t = time.time()
                rc_b, status_b = _run_cluster_job(
                    "run-iteration",
                    config,
                    baseline_dir,
                    bench_command=execution_command,
                    disable_env_var=_disable_var,
                    run_lm_eval=False,
                    stage="baseline",
                )
                all_step_timings.append(
                    (f"bench_off_V{iteration}", time.time() - start_t)
                )

                # Copy baseline results to iteration dir
                _link_baseline_results(baseline_dir, runtime_logs_dir)

                _save_shared_baseline(output_dir, runtime_logs_dir, "bench_off")
            else:
                # The OFF arm runs only for the first iteration and is reused
                # from the shared baselines afterwards. Materialise it here, or
                # this iteration has no baseline_bench_result.json and the perf
                # gate has nothing to compare against -- from V2 on it would
                # fail closed no matter how well the patch performs.
                _link_baseline_results(
                    _shared_baseline_dir(output_dir, "bench_off"), runtime_logs_dir
                )

            _compute_perf_comparison(runtime_logs_dir, output_dir)
        else:
            run_prompt = _run_prompt_gen(config)(
                context=context,
                execution_command=execution_command,
                runtime_logs_dir=runtime_logs_dir,
                iteration=iteration,
                gpu_wait_timeout_minutes=config.gpu_wait_timeout_minutes,
                smaller_model_file=smaller_model_file,
                disable_new_feature=disable_new_feature,
            )
            timings = await agent_run(
                claude_config,
                [
                    PipelineStep(
                        name=f"bench_V{iteration}", prompt=run_prompt.prompt()
                    ),
                ],
                client=client,
            )
            all_step_timings.extend(timings)

        if not _check_run_succeeded(runtime_logs_dir, iteration):
            print(f"  Bench FAILED (status={_read_status(runtime_logs_dir)})")
            # Still investigate to fix the error
            improved = False
        else:
            # ── STEP 3: Check improvement ────────────────────────────────────
            perf_ok, actual_pct, threshold_pct = _check_performance_meets_target(
                runtime_logs_dir, iteration, expected_improvement_pct
            )
            print(
                f"  Perf: {actual_pct:+.2f}% (threshold: {threshold_pct:+.2f}%)"
                if expected_improvement_pct
                else f"  Perf: {actual_pct:+.2f}%"
            )
            improved = perf_ok

        lm_eval_result_file = None

        if not improved and _check_run_succeeded(runtime_logs_dir, iteration):
            # ── STEP 3a: No improvement → NSYS → jump to investigate ─────────
            print("  No improvement → running NSYS diagnostic...")
            await _run_nsys_step(
                config,
                output_dir,
                runtime_logs_dir,
                execution_command,
                _disable_var,
                iteration,
                context,
                claude_config,
                client,
                all_step_timings,
            )
        elif improved:
            # ── STEP 4: lm_eval ON (Pod) + OFF (Pod, if needed) ──────────
            if not disable_new_feature and config.runtime_backend == "remote":
                has_lm_off = _has_shared_baseline(output_dir, "lm_eval_off")
                print(
                    "  lm_eval ON (Pod){}...".format(
                        "" if has_lm_off else " + OFF (Pod, first run)"
                    )
                )

                # Pod: lm_eval with feature ON
                lm_eval_on_dir = os.path.join(runtime_logs_dir, "lm_eval_run")
                start_t = time.time()
                rc, status = _run_cluster_job(
                    "run-iteration",
                    config,
                    lm_eval_on_dir,
                    bench_command=execution_command,
                    disable_env_var=_disable_var,
                    run_lm_eval=True,
                    skip_lm_eval_baseline=True,
                    stage="lm_eval",
                )
                all_step_timings.append(
                    (f"lm_eval_on_V{iteration}", time.time() - start_t)
                )

                # Copy lm_eval ON results to iteration dir
                for name in (
                    "lm_eval_enabled_exit_code.txt",
                    "lm_eval_enabled_stdout.log",
                    "lm_eval_enabled_stderr.log",
                    "lm_eval_output",
                ):
                    src = os.path.join(lm_eval_on_dir, name)
                    dst = os.path.join(runtime_logs_dir, name)
                    if os.path.isfile(src) and not os.path.exists(dst):
                        shutil.copy2(src, dst)
                    elif os.path.isdir(src) and not os.path.exists(dst):
                        shutil.copytree(src, dst)

                # Pod: lm_eval with feature OFF (separate pod = cache isolation)
                if not has_lm_off:
                    print("  lm_eval OFF (Pod)...")
                    lm_eval_off_dir = os.path.join(runtime_logs_dir, "lm_eval_off_run")
                    start_t = time.time()
                    rc_b, status_b = _run_cluster_job(
                        "run-iteration",
                        config,
                        lm_eval_off_dir,
                        bench_command=execution_command,
                        disable_env_var=_disable_var,
                        run_lm_eval=True,
                        skip_lm_eval_baseline=True,
                        stage="lm_eval",
                        force_feature_off=True,
                    )
                    all_step_timings.append(
                        (f"lm_eval_off_V{iteration}", time.time() - start_t)
                    )

                    # Copy disabled results to iteration dir
                    for src_name, dst_name in [
                        (
                            "lm_eval_enabled_exit_code.txt",
                            "lm_eval_disabled_exit_code.txt",
                        ),
                        ("lm_eval_enabled_stdout.log", "lm_eval_disabled_stdout.log"),
                        ("lm_eval_enabled_stderr.log", "lm_eval_disabled_stderr.log"),
                        ("lm_eval_output", "lm_eval_output_disabled"),
                    ]:
                        src = os.path.join(lm_eval_off_dir, src_name)
                        dst = os.path.join(runtime_logs_dir, dst_name)
                        if os.path.isfile(src) and not os.path.exists(dst):
                            shutil.copy2(src, dst)
                        elif os.path.isdir(src) and not os.path.exists(dst):
                            shutil.copytree(src, dst)

                    _save_shared_baseline(output_dir, runtime_logs_dir, "lm_eval_off")

                # Compute verdict
                if has_lm_off:
                    shared_lm_dir = _shared_baseline_dir(output_dir, "lm_eval_off")
                    _recompute_lm_eval_verdict(runtime_logs_dir, shared_lm_dir)
                else:
                    _recompute_lm_eval_verdict(runtime_logs_dir, runtime_logs_dir)

                lm_eval_result_file = os.path.join(runtime_logs_dir, "lm_eval.txt")
                if os.path.isfile(lm_eval_result_file):
                    with open(lm_eval_result_file) as f:
                        lm_content = f.read().strip()
                    if "FAIL" in lm_content:
                        print("  lm_eval: FAIL")
                    else:
                        print("  lm_eval: PASS")

            # ── STEP 5: NSYS ON (+OFF if needed) ────────────────────────────
            print("  Running NSYS diagnostic (improvement path)...")
            await _run_nsys_step(
                config,
                output_dir,
                runtime_logs_dir,
                execution_command,
                _disable_var,
                iteration,
                context,
                claude_config,
                client,
                all_step_timings,
            )

            # Check success (perf + lm_eval)
            if perf_ok and lm_eval_result_file and os.path.isfile(lm_eval_result_file):
                with open(lm_eval_result_file) as f:
                    if "FAIL" not in f.read():
                        print(f"\n  SUCCESS! Perf: {actual_pct:+.2f}%, lm_eval: PASS")
                        runtime_succeeded = True
                        # Write success file
                        success_path = os.path.join(results_dir, RUNTIME_SUCCESS_FILE)
                        with open(success_path, "w") as sf:
                            sf.write(
                                f"SUCCESS\npatch={prev_patch_file}\ndelta={actual_pct:+.2f}%\n"
                            )

        # ── STEP 6: Investigate → Review loop ───────────────────────────────
        max_revisions = config.max_investigation_revisions
        accepted_round = None
        iter_dir_rel = os.path.join(RUNTIME_DIR, f"V{iteration}")

        for revision in range(max_revisions + 1):
            review_feedback_text = None
            if revision > 0:
                prev_review = os.path.join(
                    output_dir, iter_dir_rel, f"review_R{revision - 1}.txt"
                )
                if os.path.isfile(prev_review):
                    with open(prev_review) as rf:
                        review_feedback_text = rf.read()

            draft_patch = os.path.join(iter_dir_rel, f"patch_draft_R{revision}.patch")
            draft_summary = os.path.join(iter_dir_rel, f"summary_draft_R{revision}.txt")

            print(f"  Investigating (R{revision})...")
            investigate_prompt = gen_InvestigateRuntimeOutputAndFixCodePrompt(
                context=context,
                code_trace_files=code_trace_files,
                code_port_plan_file=code_port_plan_file,
                test_plan_file=test_plan_file,
                runtime_logs_dir=runtime_logs_dir,
                iteration=iteration,
                prev_patch_file=prev_patch_file,
                prev_summary_file=prev_summary_file,
                iteration_history_summary_file=iteration_history_summary_file,
                smaller_model_file=smaller_model_file,
                lm_eval_result_file=lm_eval_result_file,
                results_output_dir=results_dir,
                output_patch_file=draft_patch,
                output_summary_file=draft_summary,
                review_feedback=review_feedback_text,
            )
            timings = await agent_run(
                claude_config,
                [
                    PipelineStep(
                        name=f"investigate_V{iteration}_R{revision}",
                        prompt=investigate_prompt.prompt(),
                    ),
                ],
                client=client,
            )
            all_step_timings.extend(timings)

            # On success or last allowed revision, skip review
            if runtime_succeeded or revision >= max_revisions:
                accepted_round = revision
                break

            # Review the draft
            prior_reviews = [
                os.path.join(iter_dir_rel, f"review_R{r}.txt") for r in range(revision)
            ]
            print(f"  Reviewing (R{revision})...")
            review_prompt = gen_ReviewInvestigationPrompt(
                runtime_logs_dir=runtime_logs_dir,
                iteration=iteration,
                revision=revision,
                prior_review_files=prior_reviews,
            )
            timings = await agent_run(
                claude_config,
                [
                    PipelineStep(
                        name=f"review_V{iteration}_R{revision}",
                        prompt=review_prompt.prompt(),
                    ),
                ],
                client=client,
            )
            all_step_timings.extend(timings)

            # Check verdict
            review_file = os.path.join(
                output_dir, iter_dir_rel, f"review_R{revision}.txt"
            )
            if os.path.isfile(review_file):
                with open(review_file) as rf:
                    first_line = rf.readline().strip()
                if "APPROVED" in first_line:
                    print(f"  Review R{revision}: APPROVED")
                    accepted_round = revision
                    break
                else:
                    print(f"  Review R{revision}: NEEDS_REVISION")
            else:
                accepted_round = revision
                break

        if accepted_round is None:
            accepted_round = max_revisions

        # Copy accepted draft → final
        for src_name, dst_name in [
            (f"patch_draft_R{accepted_round}.patch", "patch.patch"),
            (f"summary_draft_R{accepted_round}.txt", "summary.txt"),
        ]:
            src = os.path.join(output_dir, iter_dir_rel, src_name)
            dst = os.path.join(output_dir, iter_dir_rel, dst_name)
            if os.path.isfile(src):
                shutil.copy2(src, dst)

        # If succeeded, stop
        if runtime_succeeded:
            # Write cumulative summary
            summary_timings = await _run_cumulative_summary(
                results_dir, iteration, claude_config, client=client
            )
            all_step_timings.extend(summary_timings)
            break

        # Check for new patch
        new_patch = os.path.join(RUNTIME_DIR, f"V{iteration}", "patch.patch")
        new_patch_path = os.path.join(output_dir, new_patch)
        new_summary = os.path.join(RUNTIME_DIR, f"V{iteration}", "summary.txt")
        new_summary_path = os.path.join(output_dir, new_summary)

        if not os.path.isfile(new_summary_path):
            _generate_fallback_summary(output_dir, runtime_logs_dir, iteration)

        if not os.path.isfile(new_patch_path):
            print("  No new patch generated, stopping.")
            break

        # Check patch unchanged
        if os.path.isfile(os.path.join(output_dir, prev_patch_file)):
            with open(new_patch_path) as f:
                new_content = f.read()
            with open(os.path.join(output_dir, prev_patch_file)) as f:
                prev_content = f.read()
            if new_content == prev_content:
                print("  Patch unchanged, stopping.")
                break

        prev_patch_file = new_patch
        prev_summary_file = new_summary

        # Write cumulative summary
        summary_timings = await _run_cumulative_summary(
            results_dir, iteration, claude_config, client=client
        )
        all_step_timings.extend(summary_timings)

        print(f"  New patch: {new_patch} (accepted R{accepted_round})")

    # ── End of loop ──────────────────────────────────────────────────────────
    return [
        {
            "name": phase_label,
            "iterations": runtime_iterations,
            "max_iterations": max_runtime_iterations,
            "converged": runtime_succeeded,
            "duration": time.time() - phase_start,
        }
    ], all_step_timings


async def _run_nsys_step(
    config,
    output_dir,
    runtime_logs_dir,
    execution_command,
    disable_var,
    iteration,
    context,
    claude_config,
    client,
    all_step_timings,
):
    """Run NSYS diagnostic: patched + baseline (if shared doesn't exist) + analysis."""
    diagnostic_dir = os.path.join(runtime_logs_dir, DIAGNOSTIC_PROFILE_DIR)
    report_file = os.path.join(diagnostic_dir, DIAGNOSTIC_REPORT_FILE)

    if os.path.isfile(report_file):
        print("  NSYS diagnostic already exists, skipping.")
        return

    original_analysis_dir = _resolve_original_analysis_dir(config)
    if not original_analysis_dir or not os.path.isdir(original_analysis_dir):
        print("  SKIP NSYS: could not resolve original analysis dir")
        return

    if not disable_var:
        disable_var = _read_feature_disable_env_var(output_dir) or ""
    if not disable_var:
        print("  SKIP NSYS: could not determine feature disable env var")
        return

    delay_iters = _compute_delay_iterations(config, execution_command)
    has_nsys_off = _has_shared_baseline(output_dir, "nsys_off")

    if config.runtime_backend == "remote":
        start_t = time.time()
        rc, diag_status = _run_cluster_job(
            "run-diagnostic",
            config,
            diagnostic_dir,
            bench_command=execution_command,
            disable_env_var=disable_var,
            delay_iterations=delay_iters,
            skip_baseline=has_nsys_off,
        )
        all_step_timings.append((f"nsys_V{iteration}", time.time() - start_t))

        if "FAILED" in diag_status or rc != 0:
            print(f"  NSYS FAILED (status={diag_status})")
            return

        if not has_nsys_off:
            _save_shared_baseline(output_dir, diagnostic_dir, "nsys_off")

        # Copy shared baseline traces into diagnostic dir for analysis
        if has_nsys_off:
            shared_nsys = _shared_baseline_dir(output_dir, "nsys_off")
            for name in ("trace_baseline.nsys-rep", "trace_baseline.sqlite"):
                src = os.path.join(shared_nsys, name)
                dst = os.path.join(diagnostic_dir, name)
                if os.path.isfile(src) and not os.path.exists(dst):
                    os.symlink(os.path.abspath(src), dst)

        # Local analysis
        from llm.auto_code_gen.code_gen_prompts import gen_LocalDiagnosticAnalysisPrompt

        local_diag_prompt = gen_LocalDiagnosticAnalysisPrompt(
            context=context,
            original_analysis_dir=original_analysis_dir,
            runtime_logs_dir=runtime_logs_dir,
            iteration=iteration,
        )
        timings = await agent_run(
            claude_config,
            [
                PipelineStep(
                    name=f"nsys_analysis_V{iteration}",
                    prompt=local_diag_prompt.prompt(),
                ),
            ],
            client=client,
        )
        all_step_timings.extend(timings)
    else:
        # Local backend: single agent does everything
        diagnostic_prompt = _diagnostic_prompt_gen(config)(
            context=context,
            execution_command=execution_command,
            delay_iterations=delay_iters,
            feature_disable_env_var=disable_var,
            original_analysis_dir=original_analysis_dir,
            runtime_logs_dir=runtime_logs_dir,
            iteration=iteration,
            gpu_wait_timeout_minutes=config.gpu_wait_timeout_minutes,
        )
        timings = await agent_run(
            claude_config,
            [
                PipelineStep(
                    name=f"nsys_V{iteration}", prompt=diagnostic_prompt.prompt()
                ),
            ],
            client=client,
        )
        all_step_timings.extend(timings)


# The artifact that proves a shared baseline can still be compared against.
_BASELINE_PRIMARY_ARTIFACT = {
    "bench_off": "baseline_bench_result.json",
    "lm_eval_off": "lm_eval_disabled_exit_code.txt",
    "nsys_off": "trace_baseline.sqlite",
}


def _has_shared_baseline(output_dir, baseline_type):
    """Check that a shared baseline exists AND still resolves.

    The .done marker alone is not enough: baseline entries used to be symlinks
    into a per-iteration directory, so they dangle as soon as that iteration is
    moved or cleaned. A dangling baseline makes the harness skip the OFF
    measurement and then be unable to compare, which reads as "no improvement".
    os.path.isfile follows symlinks, so a broken link reads as absent here and
    the OFF arm is measured again instead.
    """
    base = os.path.join(output_dir, RUNTIME_DIR, SHARED_BASELINES_DIR, baseline_type)
    if not os.path.isfile(os.path.join(base, ".done")):
        return False
    primary = _BASELINE_PRIMARY_ARTIFACT.get(baseline_type)
    if primary is None:
        return True
    if os.path.isfile(os.path.join(base, primary)):
        return True
    print(
        f"  Shared {baseline_type} baseline is broken ({primary} missing), re-measuring it."
    )
    return False


def _save_shared_baseline(output_dir, source_dir, baseline_type):
    """Copy source files into the shared baseline.

    Copies, not symlinks: the sources live in a per-iteration directory that
    later cleanup or a resumed run may move away, and a baseline that points
    into it dies with it. These are small result files.
    """
    dst = os.path.join(output_dir, RUNTIME_DIR, SHARED_BASELINES_DIR, baseline_type)
    os.makedirs(dst, exist_ok=True)

    file_patterns = {
        "bench_off": [
            "baseline_bench_result.json",
            "baseline_combined.log",
            "baseline_command.txt",
            "baseline_exit_code.txt",
            "baseline_stderr.log",
            "baseline_stdout.log",
        ],
        "lm_eval_off": [
            "lm_eval_disabled_exit_code.txt",
            "lm_eval_disabled_stdout.log",
            "lm_eval_disabled_stderr.log",
        ],
        "nsys_off": ["trace_baseline.nsys-rep", "trace_baseline.sqlite"],
    }

    for name in file_patterns.get(baseline_type, []):
        src = os.path.join(source_dir, name)
        link = os.path.join(dst, name)
        if os.path.isfile(src) and not os.path.exists(link):
            shutil.copy2(src, link)

    if baseline_type == "lm_eval_off":
        src_dir = os.path.join(source_dir, "lm_eval_output_disabled")
        link_dir = os.path.join(dst, "lm_eval_output_disabled")
        if os.path.isdir(src_dir) and not os.path.exists(link_dir):
            shutil.copytree(src_dir, link_dir)

    # Write done marker
    with open(os.path.join(dst, ".done"), "w") as f:
        f.write("1\n")
    print(f"  Saved shared {baseline_type} baseline")


def _shared_baseline_dir(output_dir, baseline_type):
    """Return path to shared baseline directory."""
    return os.path.join(output_dir, RUNTIME_DIR, SHARED_BASELINES_DIR, baseline_type)


def _retire_shared_baseline(output_dir, baseline_type, reason=""):
    """Move an obsolete shared baseline aside before a re-measurement.

    _save_shared_baseline deliberately never overwrites an existing file, so
    without retiring first a fresh OFF measurement taken on a new base would be
    dropped and the perf gate would silently compare new ON numbers against the
    previous base's OFF numbers. Moving rather than deleting keeps the audit
    trail for whichever base a verdict was computed against.
    """
    dst = os.path.join(output_dir, RUNTIME_DIR, SHARED_BASELINES_DIR, baseline_type)
    if not os.path.isdir(dst):
        return None
    aside = "{}.retired-{}".format(dst, time.strftime("%Y%m%d-%H%M%S"))
    shutil.move(dst, aside)
    print(
        "  Retired the previous {} baseline ({})".format(
            baseline_type, reason or "re-measured"
        )
    )
    return aside


BASELINE_RESULT_FILES = (
    "baseline_bench_result.json",
    "baseline_combined.log",
    "baseline_command.txt",
    "baseline_exit_code.txt",
    "baseline_stderr.log",
    "baseline_stdout.log",
)


def _link_baseline_results(src_dir, dst_dir):
    """Expose OFF-arm result files in dst_dir.

    Symlinks rather than copies: these files are already on the results volume
    and several iterations point at the same shared baseline.
    """
    for name in BASELINE_RESULT_FILES:
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if os.path.isfile(src) and not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)


def _compute_perf_comparison(runtime_logs_dir, output_dir):
    """Compute perf_comparison.txt from bench ON result + shared bench OFF baseline."""
    bench_on = os.path.join(runtime_logs_dir, "bench_result.json")
    shared_off = os.path.join(
        output_dir,
        RUNTIME_DIR,
        SHARED_BASELINES_DIR,
        "bench_off",
        "baseline_bench_result.json",
    )

    # If both are in the same dir (V1 ran both), perf_comparison.txt already exists
    if os.path.isfile(os.path.join(runtime_logs_dir, "perf_comparison.txt")):
        return

    if not os.path.isfile(bench_on) or not os.path.isfile(shared_off):
        return

    import json as _json

    try:
        p = _json.load(open(bench_on))
        b = _json.load(open(shared_off))
        lines = ["Patched (feature ON) vs Baseline (feature OFF)\n"]
        for key in sorted(set(list(p) + list(b))):
            pv, bv = p.get(key, "N/A"), b.get(key, "N/A")
            if (
                isinstance(pv, (int, float))
                and isinstance(bv, (int, float))
                and bv != 0
            ):
                d = (pv - bv) / abs(bv) * 100
                lines.append(
                    f"  {key}: patched={pv:.4g} baseline={bv:.4g} delta={d:+.2f}%"
                )
            else:
                lines.append(f"  {key}: patched={pv} baseline={bv}")
        with open(os.path.join(runtime_logs_dir, "perf_comparison.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
    except Exception:
        pass


def _read_status(runtime_logs_dir):
    """Read run_status.txt content."""
    status_file = os.path.join(runtime_logs_dir, "run_status.txt")
    if os.path.isfile(status_file):
        with open(status_file) as f:
            return f.read().strip()
    return "NO_STATUS_FILE"


async def run_record_learnings(output_dir, claude_config, client=None):
    """Run record-learnings to update the global knowledge base."""
    print("\n" + "=" * 60)
    print("  Recording learnings to knowledge base...")
    print("=" * 60)
    prompt = gen_RecordLearningsPrompt(output_dir)
    steps = [
        PipelineStep(
            name="record_learnings",
            prompt=prompt.prompt(),
        ),
    ]
    timings = await agent_run(claude_config, steps, client=client)
    return timings


if __name__ == "__main__":
    args = parse_args()

    config, use_case = load_config_and_use_case(args.config)
    claude_config = config.make_agent_config(args.agent_config)

    setup_logging("code_gen")

    os.makedirs(config.output_dir, exist_ok=True)
    if not args.resume:
        safe_clean_dir(config.output_dir)

    if hasattr(config, "prepare_branches"):
        print("Preparing source code branches...")
        config.prepare_branches()

    async def _main_with_session():
        from llm.auto_code_gen.agents import agent_session

        async with agent_session(claude_config) as client:
            results = await run_pipeline(
                config, claude_config, use_case, resume=args.resume, client=client
            )
            return results, client

    start_time = time.time()
    (phase_results, all_step_timings), _ = asyncio.run(_main_with_session())
    total_duration = time.time() - start_time

    _print_detailed_table(all_step_timings, total_duration, config.output_dir)
    _print_phase_summary(
        phase_results, all_step_timings, total_duration, config.output_dir
    )
    _print_timeline_and_costs(
        all_step_timings, total_duration, start_time, config.output_dir
    )

    # Save git diff to changes.diff
    try:
        diff_output = subprocess.run(
            ["git", "diff", "HEAD"],
            cwd=config.source_code_dir if hasattr(config, "source_code_dir") else None,
            capture_output=True,
            text=True,
            check=False,
        )
        changes_file = os.path.join(config.output_dir, "changes.diff")
        with open(changes_file, "w") as f:
            f.write(diff_output.stdout)
        print(f"\nGit diff saved to: {changes_file}")
    except Exception as e:
        print(f"\nWarning: Could not save git diff: {e}")

    # Record learnings to knowledge base
    if config.use_knowledge_base:
        try:
            asyncio.run(run_record_learnings(config.output_dir, claude_config))
        except Exception as e:
            print(f"\nWarning: Could not record learnings: {e}")
    else:
        print("\nSkipping record-learnings (use_knowledge_base=false).")
