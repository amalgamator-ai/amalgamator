"""cluster_runner.py — submit GPU Jobs to OpenShift for the remote runtime backend.

Subcommands
-----------
run-iteration
    One Job: compile → bench patched → bench baseline → lm_eval (optional).
    Used for candidate benchmarking (_benchmark_candidates).

run-iteration-parallel
    Two-phase execution: first a compile Job (venv + cmake on the PVC), then
    bench + baseline (+ optional lm_eval) as parallel GPU Jobs with cmake
    skipped.  Results are merged into --out-dir.
    Used for the main iteration path.

run-diagnostic
    One Job: nsys profile patched → nsys profile baseline → nsys export to SQLite.
    The agent then runs Steps 2-5 (extraction, correlation, comparison, report)
    locally in the same Claude session so the full reasoning history is preserved.

PVC layout — two PVCs mounted in every Job pod:
  workspace PVC (vllm-workspace, RWX NFS) at /workspace:
    /workspace/pvc_<run-name>/ patched vllm source (rsynced by agent before job);
                                run-named (--vllm-dir, defaults to the pipeline's
                                "pvc_<output_dir basename>") so concurrent runs
                                never share a tree; /workspace/vllm only when
                                invoked standalone without --vllm-dir
    /workspace/output/<nonce>/  per-job results; copied back to local after job completes
  pod-local (not on PVC):
    /tmp/vllm-venv/             editable-install venv (created fresh per pod)
  model PVC at /model-cache (required):
    HF_HOME is set to /model-cache so vllm/lm_eval use it as the HF cache.
    If the model is not yet cached there, vllm will download it on first run.
    Subsequent runs hit the cache. Use a pre-populated PVC to skip the download.

GPU allocation: the Job requests nvidia.com/gpu: N with a nodeSelector. NFS PVCs have
no node affinity so the scheduler picks whichever GPU node has free slots.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
from pathlib import Path

# ── oc helpers ────────────────────────────────────────────────────────────────


def _oc(*args, check=True, capture=True, input=None):
    return subprocess.run(
        ["oc"] + list(args),
        check=check,
        capture_output=capture,
        text=True,
        input=input,
    )


def _run_or_die(*args, **kw):
    try:
        return _oc(*args, **kw)
    except subprocess.CalledProcessError as e:
        print(
            "ERROR: oc {} failed\n{}".format(
                " ".join(str(a) for a in args[:3]), e.stderr
            ),
            file=sys.stderr,
        )
        sys.exit(1)


def _wait_for_pod(namespace, job_name, timeout_s, poll_s=10):
    """Return (pod_name, phase) once the Job pod leaves Pending, or raise on timeout."""
    deadline = time.time() + timeout_s
    last_label = ""
    while time.time() < deadline:
        r = _oc(
            "get",
            "pods",
            "-n",
            namespace,
            "-l",
            f"job-name={job_name}",
            "-o",
            "json",
            check=False,
        )
        if r.returncode == 0:
            items = json.loads(r.stdout).get("items", [])
            if items:
                pod = items[0]
                name = pod["metadata"]["name"]
                phase = pod["status"].get("phase", "Unknown")
                conds = pod["status"].get("conditions") or []
                reason = conds[-1].get("reason", "") if conds else ""
                msg = (conds[-1].get("message", "") or "")[:120] if conds else ""
                label = f"{reason} — {msg}" if msg else reason
                if phase not in ("Pending", "Unknown"):
                    return name, phase
                if label != last_label:
                    print(f"  pod {name}: {phase} — {label}", file=sys.stderr)
                    last_label = label
        time.sleep(poll_s)
    raise TimeoutError(f"Timed out waiting for pod of Job {job_name}")


def _poll_job_until_done(namespace, job_name, pod_name, timeout_s=7200, poll_s=30):
    """Poll job status and print log snapshots until the job reaches a terminal state."""
    deadline = time.time() + timeout_s
    terminal = None
    while time.time() < deadline:
        r = _oc("get", "job", job_name, "-n", namespace, "-o", "json", check=False)
        if r.returncode == 0:
            for c in json.loads(r.stdout).get("status", {}).get("conditions", []):
                if (
                    c.get("type") in ("Complete", "Failed")
                    and c.get("status") == "True"
                ):
                    terminal = c["type"]
                    break
        if terminal:
            break
        r_tail = _oc("logs", "-n", namespace, pod_name, "--tail=20", check=False)
        if r_tail.returncode == 0 and r_tail.stdout.strip():
            print("--- log snapshot ---", file=sys.stderr)
            print(r_tail.stdout, file=sys.stderr)
        time.sleep(poll_s)

    if terminal is None:
        raise TimeoutError(f"Timed out waiting for Job {job_name}")

    r_final = _oc("logs", "-n", namespace, pod_name, "--tail=80", check=False)
    if r_final.returncode == 0:
        print("--- final logs ---", file=sys.stderr)
        print(r_final.stdout, file=sys.stderr)

    return terminal


def _copy_results(namespace, pvc_name, remote_dir, local_dir):
    """Copy job results from PVC to local via a short-lived helper pod.

    oc cp execs ``tar`` inside the target container.  Once a Job pod
    reaches Completed/Failed the container has exited and tar cannot
    run, so we spin up a lightweight helper that mounts the same PVC,
    wait for ``run_status.txt`` to confirm the job flushed all output,
    then copy and tear down.
    """
    helper = f"ai-auto-analyze-cp-{uuid.uuid4().hex[:6]}"
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
                    {
                        "name": "ws",
                        "persistentVolumeClaim": {"claimName": pvc_name},
                    }
                ],
            }
        }
    )
    try:
        _oc(
            "run",
            helper,
            "-n",
            namespace,
            "--image=registry.access.redhat.com/ubi9/ubi:latest",
            "--overrides=" + overrides,
            "--restart=Never",
            check=True,
        )
        _oc(
            "wait",
            "-n",
            namespace,
            "pod/" + helper,
            "--for=condition=Ready",
            "--timeout=120s",
            check=True,
        )

        # Wait for run_status.txt — written as the last step by the job
        status_path = f"{remote_dir}/run_status.txt"
        for _ in range(30):
            r = _oc(
                "exec",
                "-n",
                namespace,
                helper,
                "--",
                "test",
                "-f",
                status_path,
                check=False,
            )
            if r.returncode == 0:
                break
            time.sleep(2)

        os.makedirs(local_dir, exist_ok=True)
        src = f"{namespace}/{helper}:{remote_dir}/"
        r = _oc("cp", src, local_dir + "/", check=False)
        if r.returncode != 0:
            print(
                f"WARNING: oc cp via helper failed ({r.returncode}): {r.stderr.strip()}",
                file=sys.stderr,
            )
            return False
        return True
    except subprocess.CalledProcessError as e:
        print(f"WARNING: helper pod failed: {e.stderr or e}", file=sys.stderr)
        return False
    finally:
        _oc("delete", "pod", helper, "-n", namespace, "--wait=false", check=False)


def _upload_dir(namespace, pod_name, local_dir, remote_dir):
    """oc cp local dir → pod. Used to upload diagnostic reference files."""
    r = _oc(
        "exec", "-n", namespace, pod_name, "--", "mkdir", "-p", remote_dir, check=False
    )
    r = _oc(
        "cp",
        local_dir + "/",
        f"{namespace}/{pod_name}:{remote_dir}/",
        check=False,
    )
    if r.returncode != 0:
        print(
            f"WARNING: upload to pod failed: {r.stderr.strip()}",
            file=sys.stderr,
        )
        return False
    return True


# ── bench command sanitization ───────────────────────────────────────────────


def _sanitize_bench_command(cmd):
    """Clean a raw run_command from run_params.txt for use inside the GPU Job.

    Three transforms:
    1. Strip leading `env KEY=VALUE ...` prefix — GPU assignment is handled by
       the Job spec (nvidia.com/gpu resource), not CUDA_VISIBLE_DEVICES.
    2. Rewrite either supported result-output dialect so results land in the
       per-run output directory the entrypoint manages.
    3. Unescape comma-separated values (e.g. 4\\,5\\,6\\,7 → 4,5,6,7) left over
       from run_params.txt shell escaping.
    """
    # Unescape backslash-escaped commas from run_params.txt
    cmd = cmd.replace("\\,", ",")

    # Strip leading `env VAR=VALUE VAR=VALUE ...` prefix (stops at first non-VAR= token)
    tokens = cmd.split()
    if tokens and tokens[0] == "env":
        tokens = tokens[1:]
    # Drop any remaining KEY=VALUE env tokens at the start
    while tokens and "=" in tokens[0] and not tokens[0].startswith("-"):
        tokens = tokens[1:]
    cmd = " ".join(tokens)

    # Replace --output-json <path> with the Job's output dir variable
    cmd = re.sub(r"--output-json\s+\S+", "--output-json $OUT/bench_result.json", cmd)

    # Current vLLM uses the pair --result-dir/--result-filename instead.
    cmd = re.sub(r"--result-dir\s+\S+", "--result-dir $OUT", cmd)
    cmd = re.sub(r"--result-filename\s+\S+", "--result-filename bench_result.json", cmd)

    return cmd


# ── lm_eval command derivation ────────────────────────────────────────────────


def _derive_lm_eval_command(bench_command, hf_cache_dir, server_command=""):
    """Build a standard gsm8k lm_eval command from the benchmark command.

    Extracts --model, --tensor-parallel-size and --kv-cache-dtype. lm_eval runs
    its own in-process vLLM engine, so it must reproduce the serving topology.
    A client-style bench command (`vllm bench serve --port 8123 ...`) talks to
    an already-running server and carries neither --tensor-parallel-size nor
    --kv-cache-dtype, so those are taken from the server command when absent;
    deriving tp=1 for a TP=4 deployment would fail to load the model.

    The lm_eval writes its JSON output to $OUT/lm_eval_output (set by the
    entrypoint).
    """

    def _first(pattern):
        for source in (bench_command or "", server_command or ""):
            match = re.search(pattern, source)
            if match:
                return match.group(1)
        return None

    model = _first(r"--model\s+(\S+)") or "UNKNOWN_MODEL"
    tp = _first(r"--tensor-parallel-size\s+(\d+)") or "1"
    kv = _first(r"--kv-cache-dtype\s+(\S+)") or ""

    model_args = (
        f"pretrained={model},tensor_parallel_size={tp},"
        "trust_remote_code=True,max_model_len=4096"
    )
    if kv:
        model_args += f",kv_cache_dtype={kv}"

    return (
        f"HF_HOME={hf_cache_dir} python3 -m lm_eval"
        " --model vllm"
        f" --model_args {model_args}"
        " --tasks gsm8k"
        " --limit 200"
        " --batch_size auto"
        " --output_path $OUT/lm_eval_output"
    )


# ── entrypoint script assembly ───────────────────────────────────────────────

_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")

_INCLUDE_MARKER = "# === INCLUDE: venv_compile.sh ==="


def _assemble_script(script_name):
    """Read a shell script from scripts/ and inline the shared compile fragment."""
    with open(os.path.join(_SCRIPTS_DIR, script_name)) as f:
        script = f.read()
    with open(os.path.join(_SCRIPTS_DIR, "venv_compile.sh")) as f:
        compile_block = f.read()
    return script.replace(_INCLUDE_MARKER, compile_block)


# ── ConfigMap helpers ─────────────────────────────────────────────────────────


def _create_configmap(namespace, name, script_content):
    """Create a ConfigMap containing the entrypoint script."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(script_content)
        tmp_path = f.name
    try:
        _run_or_die(
            "create",
            "configmap",
            name,
            "-n",
            namespace,
            f"--from-file=entrypoint.sh={tmp_path}",
        )
    finally:
        os.unlink(tmp_path)
    print(f"[cluster_runner] configmap {name} created", file=sys.stderr)


# ── Job manifest ──────────────────────────────────────────────────────────────

# Extra environment for Job containers, parsed from --extra-env in main().
_EXTRA_ENV = {}


def _parse_extra_env(values):
    env = {}
    for item in values or []:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--extra-env expects KEY=VALUE, got: {item}")
        env[key] = value
    return env


def _build_job_manifest(
    job_name,
    namespace,
    image,
    gpu_count,
    gpu_product,
    pvc_name,
    configmap_name,
    model_pvc_name="",
    ttl_seconds=600,
    env_vars=None,
    pull_secret=None,
):
    model_volume_mount = (
        "\n            - name: model-cache\n" "              mountPath: /model-cache"
        if model_pvc_name
        else ""
    )
    model_volume = (
        "\n        - name: model-cache\n"
        "          persistentVolumeClaim:\n"
        f"            claimName: {model_pvc_name}"
        if model_pvc_name
        else ""
    )
    pull_secret_section = (
        "imagePullSecrets:\n" f"                - name: {pull_secret}"
        if pull_secret
        else ""
    )

    all_env = [
        ("USER", "vllm"),
        ("HOME", "/tmp"),
    ]
    all_env.extend(sorted(_EXTRA_ENV.items()))
    if env_vars:
        all_env.extend(env_vars.items())
    env_lines = []
    for name, value in all_env:
        env_lines.append(f"            - name: {name}")
        escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
        env_lines.append(f'              value: "{escaped}"')
    env_section = "\n".join(env_lines)
    env_section = env_section.replace("{", "{{").replace("}", "}}")

    return (
        textwrap.dedent("""
        apiVersion: batch/v1
        kind: Job
        metadata:
          name: {job_name}
          namespace: {namespace}
          labels:
            app: ai-auto-analyze
        spec:
          backoffLimit: 0
          ttlSecondsAfterFinished: {ttl}
          template:
            spec:
              restartPolicy: Never
              {pull_secret_section}
              nodeSelector:
                nvidia.com/gpu.product: {gpu_product}
              containers:
                - name: runner
                  image: {image}
                  command: ["/bin/bash", "/scripts/entrypoint.sh"]
                  resources:
                    requests:
                      cpu: "8"
                      memory: "32Gi"
                      nvidia.com/gpu: "{gpu_count}"
                    limits:
                      memory: "256Gi"
                      nvidia.com/gpu: "{gpu_count}"
                  env:
        {env_section}
                  volumeMounts:
                    - name: entrypoint
                      mountPath: /scripts
                      readOnly: true
                    - name: workspace
                      mountPath: /workspace
                    - name: shm
                      mountPath: /dev/shm
                    - name: flashinfer-cubins
                      mountPath: /usr/local/lib/python3.12/dist-packages/flashinfer_cubin/cubins{model_volume_mount}
              volumes:
                - name: entrypoint
                  configMap:
                    name: {configmap_name}
                    defaultMode: 0755
                - name: workspace
                  persistentVolumeClaim:
                    claimName: {pvc_name}
                - name: shm
                  emptyDir:
                    medium: Memory
                    sizeLimit: 64Gi
                - name: flashinfer-cubins
                  emptyDir: {{}}{model_volume}
    """)
        .format(
            job_name=job_name,
            namespace=namespace,
            image=image,
            ttl=ttl_seconds,
            gpu_product=gpu_product,
            gpu_count=gpu_count,
            pvc_name=pvc_name,
            configmap_name=configmap_name,
            env_section=env_section,
            model_volume_mount=model_volume_mount,
            model_volume=model_volume,
            pull_secret_section=pull_secret_section,
        )
        .strip()
    )


# ── shared job runner ─────────────────────────────────────────────────────────


def _run_job(
    namespace,
    job_name,
    manifest,
    gpu_count,
    gpu_product,
    pvc_out_dir,
    local_out_dir,
    gpu_wait_timeout_minutes,
    pvc_name,
    yaml_name="job_manifest.yaml",
    entrypoint_script=None,
):
    """Submit the Job, wait for it, copy results back. Returns pod_name.

    If entrypoint_script is provided, its content is saved alongside the
    YAML for debugging.  ConfigMap cleanup is the caller's responsibility.
    """
    # Dump YAML for review and documentation
    yaml_dir = os.path.join(local_out_dir, "cluster_yamls")
    os.makedirs(yaml_dir, exist_ok=True)
    yaml_path = os.path.join(yaml_dir, yaml_name)
    with open(yaml_path, "w") as f:
        f.write(manifest)
    if entrypoint_script:
        script_path = os.path.join(
            yaml_dir, yaml_name.replace(".yaml", "_entrypoint.sh")
        )
        with open(script_path, "w") as f:
            f.write(entrypoint_script)
    print(f"[cluster_runner] job manifest written to {yaml_path}", file=sys.stderr)
    print(manifest, file=sys.stderr)

    print(
        f"[cluster_runner] submitting {job_name} namespace={namespace}",
        file=sys.stderr,
    )
    _run_or_die("apply", "-f", "-", "-n", namespace, capture=False, input=manifest)

    print(
        f"[cluster_runner] waiting for pod ({gpu_count} × {gpu_product} GPUs)...",
        file=sys.stderr,
    )
    try:
        pod_name, phase = _wait_for_pod(
            namespace,
            job_name,
            timeout_s=gpu_wait_timeout_minutes * 60,
        )
    except TimeoutError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        _oc("delete", "job", job_name, "-n", namespace, check=False)
        sys.exit(1)

    print(
        f"[cluster_runner] pod {pod_name} started (phase={phase})",
        file=sys.stderr,
    )

    try:
        terminal = _poll_job_until_done(namespace, job_name, pod_name)
    except TimeoutError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"[cluster_runner] Job state: {terminal}", file=sys.stderr)

    # Always copy results so the agent can inspect logs even on failure.
    print(
        f"[cluster_runner] copying results to {local_out_dir}...",
        file=sys.stderr,
    )
    _copy_results(namespace, pvc_name, pvc_out_dir, local_out_dir)

    return pod_name


# ── subcommand: run-iteration ─────────────────────────────────────────────────


def cmd_run_iteration(args):
    nonce = uuid.uuid4().hex[:8]
    job_name = f"ai-auto-analyze-iter-{nonce}"

    workspace = "/workspace"
    vllm_dir = getattr(args, "vllm_dir", None) or f"{workspace}/vllm"
    pvc_out_dir = f"{workspace}/output/{nonce}"
    hf_home = "/model-cache"
    bench_command = _sanitize_bench_command(args.bench_command)
    server_command = args.server_command.strip()
    print(f"[cluster_runner] bench command: {bench_command}", file=sys.stderr)
    print(f"[cluster_runner] server command: {server_command}", file=sys.stderr)

    lm_eval_command = args.lm_eval_command
    if args.run_lm_eval and not lm_eval_command:
        lm_eval_command = _derive_lm_eval_command(
            bench_command, hf_home, server_command
        )

    stage = getattr(args, "stage", "") or ""
    env = {
        "OUT": pvc_out_dir,
        "VLLM_DIR": vllm_dir,
        "HF_HOME": hf_home,
        "BENCH_CMD": bench_command,
        "SERVER_CMD": server_command,
        "DISABLE_ENV_VAR": args.disable_env_var,
        "RUN_LM_EVAL": "1" if args.run_lm_eval else "0",
        "LM_EVAL_CMD": lm_eval_command or "",
        "SKIP_LM_EVAL_BASELINE": (
            "1" if getattr(args, "skip_lm_eval_baseline", False) else "0"
        ),
    }
    if stage:
        env["STAGE"] = stage
        if stage in ("baseline", "lm_eval"):
            env["SKIP_CMAKE"] = "1"
    if getattr(args, "force_feature_off", False):
        env["FORCE_FEATURE_OFF"] = "1"
    if args.patch_file:
        env["PATCH_FILE"] = args.patch_file
        # Candidate isolation: the patch applies onto a copy of the base tree.
        # With a run-named vllm dir the base and the candidate are the same
        # tree (the pipeline rsyncs the exact candidate); a separate
        # --vllm-base-dir is still honored when given.
        env["VLLM_BASE_DIR"] = getattr(args, "vllm_base_dir", None) or vllm_dir

    entrypoint = _assemble_script("run_iteration.sh")
    cm_name = f"ai-auto-analyze-iter-{nonce}"
    _create_configmap(args.namespace, cm_name, entrypoint)

    manifest = _build_job_manifest(
        job_name=job_name,
        namespace=args.namespace,
        image=args.image,
        gpu_count=args.gpu_count,
        gpu_product=args.gpu_product,
        pvc_name=args.pvc_name,
        model_pvc_name=args.model_pvc_name,
        configmap_name=cm_name,
        ttl_seconds=args.ttl_seconds,
        env_vars=env,
    )

    _run_job(
        namespace=args.namespace,
        job_name=job_name,
        manifest=manifest,
        gpu_count=args.gpu_count,
        gpu_product=args.gpu_product,
        pvc_out_dir=pvc_out_dir,
        local_out_dir=args.out_dir,
        gpu_wait_timeout_minutes=args.gpu_wait_timeout_minutes,
        pvc_name=args.pvc_name,
        yaml_name="run_iteration.yaml",
        entrypoint_script=entrypoint,
    )

    status_file = os.path.join(args.out_dir, "run_status.txt")
    status = (
        Path(status_file).read_text().strip()
        if os.path.isfile(status_file)
        else "NO_STATUS_FILE"
    )
    print(f"[cluster_runner] run_status={status}", file=sys.stderr)

    if status != "COMPLETED":
        print(
            f"ERROR: iteration did not complete (status={status})",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[cluster_runner] success — results in {args.out_dir}", file=sys.stderr)


# ── perf comparison (shared by sequential and parallel paths) ────────────────


def _generate_perf_comparison(out_dir):
    """Generate perf_comparison.txt from the two bench result JSONs."""
    bench_file = os.path.join(out_dir, "bench_result.json")
    baseline_file = os.path.join(out_dir, "baseline_bench_result.json")
    if not (os.path.isfile(bench_file) and os.path.isfile(baseline_file)):
        return
    with open(bench_file) as f:
        p = json.load(f)
    with open(baseline_file) as f:
        b = json.load(f)
    lines = ["Patched (feature ON) vs Baseline (feature OFF)\n"]
    for key in sorted(set(list(p) + list(b))):
        pv, bv = p.get(key, "N/A"), b.get(key, "N/A")
        if isinstance(pv, (int, float)) and isinstance(bv, (int, float)) and bv != 0:
            d = (pv - bv) / abs(bv) * 100
            lines.append(f"  {key}: patched={pv:.4g} baseline={bv:.4g} delta={d:+.2f}%")
        else:
            lines.append(f"  {key}: patched={pv} baseline={bv}")
    with open(os.path.join(out_dir, "perf_comparison.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ── subcommand: run-iteration-parallel ───────────────────────────────────────


def cmd_run_iteration_parallel(args):
    """Compile once, then submit bench + baseline (+ optional lm_eval) in parallel.

    Phase 1: a single GPU Job runs STAGE=compile (venv + cmake) on the PVC.
    Phase 2: N parallel Jobs (SKIP_CMAKE=1) run bench/baseline/lm_eval.
    Results are merged into --out-dir after all Jobs complete.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    nonce = uuid.uuid4().hex[:8]
    workspace = "/workspace"
    vllm_dir = getattr(args, "vllm_dir", None) or f"{workspace}/vllm"
    hf_home = "/model-cache"
    bench_command = _sanitize_bench_command(args.bench_command)
    server_command = args.server_command.strip()
    print(f"[cluster_runner] bench command: {bench_command}", file=sys.stderr)
    print(f"[cluster_runner] server command: {server_command}", file=sys.stderr)

    lm_eval_command = args.lm_eval_command
    if args.run_lm_eval and not lm_eval_command:
        lm_eval_command = _derive_lm_eval_command(
            bench_command, hf_home, server_command
        )

    entrypoint = _assemble_script("run_iteration.sh")
    cm_name = f"ai-auto-analyze-iter-{nonce}"
    _create_configmap(args.namespace, cm_name, entrypoint)

    yaml_dir = os.path.join(args.out_dir, "cluster_yamls")
    os.makedirs(yaml_dir, exist_ok=True)
    script_path = os.path.join(yaml_dir, "run_iteration_entrypoint.sh")
    with open(script_path, "w") as f:
        f.write(entrypoint)

    # ── Phase 1: compile-only job ────────────────────────────────────────────
    compile_job_name = f"ai-auto-analyze-compile-{nonce}"
    compile_pvc_out = f"{workspace}/output/{nonce}-compile"
    compile_env = {
        "OUT": compile_pvc_out,
        "VLLM_DIR": vllm_dir,
        "HF_HOME": hf_home,
        "BENCH_CMD": bench_command,
        "SERVER_CMD": server_command,
        "DISABLE_ENV_VAR": args.disable_env_var,
        "STAGE": "compile",
        "RUN_LM_EVAL": "0",
        "LM_EVAL_CMD": "",
    }
    if args.patch_file:
        compile_env["PATCH_FILE"] = args.patch_file
        compile_env["VLLM_BASE_DIR"] = f"{workspace}/vllm"
    compile_manifest = _build_job_manifest(
        job_name=compile_job_name,
        namespace=args.namespace,
        image=args.image,
        gpu_count=args.gpu_count,
        gpu_product=args.gpu_product,
        pvc_name=args.pvc_name,
        model_pvc_name=args.model_pvc_name,
        configmap_name=cm_name,
        ttl_seconds=args.ttl_seconds,
        env_vars=compile_env,
    )

    print("[cluster_runner] Phase 1: compile", file=sys.stderr)
    _run_job(
        namespace=args.namespace,
        job_name=compile_job_name,
        manifest=compile_manifest,
        gpu_count=args.gpu_count,
        gpu_product=args.gpu_product,
        pvc_out_dir=compile_pvc_out,
        local_out_dir=args.out_dir,
        gpu_wait_timeout_minutes=args.gpu_wait_timeout_minutes,
        pvc_name=args.pvc_name,
        yaml_name="run_compile.yaml",
        entrypoint_script=entrypoint,
    )

    compile_status_file = os.path.join(args.out_dir, "run_status.txt")
    compile_status = (
        Path(compile_status_file).read_text().strip()
        if os.path.isfile(compile_status_file)
        else "NO_STATUS_FILE"
    )
    if compile_status != "COMPLETED":
        print(
            f"ERROR: compile job did not complete (status={compile_status})",
            file=sys.stderr,
        )
        sys.exit(1)

    print("[cluster_runner] Phase 1: compile OK", file=sys.stderr)

    # ── Phase 2: parallel benchmark jobs ─────────────────────────────────────
    stages = ["bench", "baseline"]
    if args.run_lm_eval:
        stages.append("lm_eval")

    stage_jobs = {}
    for stage in stages:
        job_name = f"ai-auto-analyze-{stage}-{nonce}"
        pvc_out = f"{workspace}/output/{nonce}-{stage}"
        env = {
            "OUT": pvc_out,
            "VLLM_DIR": vllm_dir,
            "HF_HOME": hf_home,
            "BENCH_CMD": bench_command,
            "SERVER_CMD": server_command,
            "DISABLE_ENV_VAR": args.disable_env_var,
            "STAGE": stage,
            "SKIP_CMAKE": "1",
            "RUN_LM_EVAL": "1" if stage == "lm_eval" else "0",
            "LM_EVAL_CMD": lm_eval_command if stage == "lm_eval" else "",
        }
        if args.patch_file:
            env["PATCH_FILE"] = args.patch_file
            env["VLLM_BASE_DIR"] = f"{workspace}/vllm"
        manifest = _build_job_manifest(
            job_name=job_name,
            namespace=args.namespace,
            image=args.image,
            gpu_count=args.gpu_count,
            gpu_product=args.gpu_product,
            pvc_name=args.pvc_name,
            model_pvc_name=args.model_pvc_name,
            configmap_name=cm_name,
            ttl_seconds=args.ttl_seconds,
            env_vars=env,
        )
        stage_jobs[stage] = {
            "job_name": job_name,
            "manifest": manifest,
            "pvc_out": pvc_out,
        }

    print(
        f"[cluster_runner] Phase 2: submitting {len(stages)} parallel jobs",
        file=sys.stderr,
    )

    try:
        for stage, info in stage_jobs.items():
            yaml_path = os.path.join(yaml_dir, f"run_{stage}.yaml")
            with open(yaml_path, "w") as f:
                f.write(info["manifest"])
            print(
                f"[cluster_runner] {stage} manifest → {yaml_path}",
                file=sys.stderr,
            )
            print(info["manifest"], file=sys.stderr)
            _run_or_die(
                "apply",
                "-f",
                "-",
                "-n",
                args.namespace,
                capture=False,
                input=info["manifest"],
            )

        print(
            f"[cluster_runner] submitted {len(stages)} parallel jobs ({args.gpu_count} × {args.gpu_product} GPUs each)",
            file=sys.stderr,
        )

        # ── monitor all jobs in parallel ──────────────────────────────────────
        def _monitor_stage(stage):
            info = stage_jobs[stage]
            try:
                pod_name, _ = _wait_for_pod(
                    args.namespace,
                    info["job_name"],
                    timeout_s=args.gpu_wait_timeout_minutes * 60,
                )
                print(
                    f"[cluster_runner] [{stage}] pod {pod_name} started",
                    file=sys.stderr,
                )
                terminal = _poll_job_until_done(
                    args.namespace,
                    info["job_name"],
                    pod_name,
                )
                print(
                    f"[cluster_runner] [{stage}] job status: {terminal}",
                    file=sys.stderr,
                )
                return stage, terminal, pod_name, None
            except Exception as e:
                return stage, "Failed", None, str(e)

        results = {}
        with ThreadPoolExecutor(max_workers=len(stages)) as executor:
            futures = {executor.submit(_monitor_stage, s): s for s in stages}
            for future in as_completed(futures):
                stage, terminal, pod_name, error = future.result()
                results[stage] = {
                    "terminal": terminal,
                    "pod_name": pod_name,
                    "error": error,
                }
                if error:
                    print(
                        f"[cluster_runner] [{stage}] ERROR: {error}",
                        file=sys.stderr,
                    )

        # ── copy results from PVC (sequentially to avoid overwrites) ──────────
        for stage in stages:
            if results[stage]["pod_name"]:
                print(
                    f"[cluster_runner] copying {stage} results...",
                    file=sys.stderr,
                )
                _copy_results(
                    args.namespace,
                    args.pvc_name,
                    stage_jobs[stage]["pvc_out"],
                    args.out_dir,
                )
    except Exception:

        raise

    # ── generate perf_comparison.txt and final status ─────────────────────────
    _generate_perf_comparison(args.out_dir)

    all_ok = all(r["terminal"] == "Complete" for r in results.values())
    if all_ok:
        final_status = "COMPLETED"
    else:
        failed = [s for s in stages if results[s]["terminal"] != "Complete"]
        final_status = f"{failed[0].upper()}_FAILED"

    status_file = os.path.join(args.out_dir, "run_status.txt")
    with open(status_file, "w") as f:
        f.write(final_status + "\n")

    print(f"[cluster_runner] run_status={final_status}", file=sys.stderr)

    if not all_ok:
        for stage in stages:
            r = results[stage]
            print(
                "  {}: {} {}".format(stage, r["terminal"], r["error"] or ""),
                file=sys.stderr,
            )
        sys.exit(1)

    print(f"[cluster_runner] success — results in {args.out_dir}", file=sys.stderr)


# ── subcommand: run-diagnostic ────────────────────────────────────────────────


def cmd_run_diagnostic(args):
    """Submit two Jobs for nsys profiling (Steps 0-1 of the diagnostic).

    Pod 1 profiles with the feature ON (patched), Pod 2 profiles with the
    feature OFF (baseline).  Using separate pods guarantees each gets a clean
    GPU state — running both in a single pod caused GPU OOM on the baseline
    because the patched run's memory was not released.

    After this returns, the local Claude agent runs Steps 2-5 (extraction,
    correlation, comparison, diagnostic_report.txt) in the same session so the
    full reasoning history is preserved.
    """
    workspace = "/workspace"
    vllm_dir = getattr(args, "vllm_dir", None) or f"{workspace}/vllm"
    hf_home = "/model-cache"
    bench_command = _sanitize_bench_command(args.bench_command)
    print(f"[cluster_runner] bench command: {bench_command}", file=sys.stderr)

    # ── Pod 1: patched (feature ON) ──────────────────────────────────
    nonce_p = uuid.uuid4().hex[:8]
    job_name_p = f"ai-auto-analyze-diag-p-{nonce_p}"
    pvc_out_dir_p = f"{workspace}/output/{nonce_p}"

    entrypoint_p = _assemble_script("run_diagnostic_patched.sh")
    cm_name_p = f"ai-auto-analyze-diag-p-{nonce_p}"
    _create_configmap(args.namespace, cm_name_p, entrypoint_p)

    manifest_p = _build_job_manifest(
        job_name=job_name_p,
        namespace=args.namespace,
        image=args.image,
        gpu_count=args.gpu_count,
        gpu_product=args.gpu_product,
        pvc_name=args.pvc_name,
        model_pvc_name=args.model_pvc_name,
        configmap_name=cm_name_p,
        ttl_seconds=args.ttl_seconds,
        env_vars={
            "OUT": pvc_out_dir_p,
            "VLLM_DIR": vllm_dir,
            "HF_HOME": hf_home,
            "BENCH_CMD": bench_command,
            "SERVER_CMD": args.server_command.strip(),
            "DISABLE_ENV_VAR": args.disable_env_var,
            "DELAY_ITERATIONS": str(args.delay_iterations),
        },
    )

    print("[cluster_runner] submitting PATCHED nsys pod...", file=sys.stderr)
    _run_job(
        namespace=args.namespace,
        job_name=job_name_p,
        manifest=manifest_p,
        gpu_count=args.gpu_count,
        gpu_product=args.gpu_product,
        pvc_out_dir=pvc_out_dir_p,
        local_out_dir=args.out_dir,
        gpu_wait_timeout_minutes=args.gpu_wait_timeout_minutes,
        pvc_name=args.pvc_name,
        yaml_name="run_diagnostic_patched.yaml",
        entrypoint_script=entrypoint_p,
    )

    status_file = os.path.join(args.out_dir, "diagnostic_status.txt")
    status_p = (
        Path(status_file).read_text().strip()
        if os.path.isfile(status_file)
        else "NO_STATUS_FILE"
    )
    print(
        f"[cluster_runner] patched diagnostic_status={status_p}",
        file=sys.stderr,
    )

    if status_p != "NSYS_PATCHED_DONE":
        print(
            f"ERROR: patched diagnostic did not complete (status={status_p})",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Pod 2: baseline (feature OFF) — skip if --skip-baseline ─────
    if getattr(args, "skip_baseline", False):
        print(
            "[cluster_runner] SKIP baseline nsys (--skip-baseline, reusing shared baseline)",
            file=sys.stderr,
        )
        # Write final status so the pipeline knows patched-only succeeded
        with open(status_file, "w") as f:
            f.write("NSYS_DONE\n")
    else:
        nonce_b = uuid.uuid4().hex[:8]
        job_name_b = f"ai-auto-analyze-diag-b-{nonce_b}"
        pvc_out_dir_b = f"{workspace}/output/{nonce_b}"

        entrypoint_b = _assemble_script("run_diagnostic_baseline.sh")
        cm_name_b = f"ai-auto-analyze-diag-b-{nonce_b}"
        _create_configmap(args.namespace, cm_name_b, entrypoint_b)

        manifest_b = _build_job_manifest(
            job_name=job_name_b,
            namespace=args.namespace,
            image=args.image,
            gpu_count=args.gpu_count,
            gpu_product=args.gpu_product,
            pvc_name=args.pvc_name,
            model_pvc_name=args.model_pvc_name,
            configmap_name=cm_name_b,
            ttl_seconds=args.ttl_seconds,
            env_vars={
                "OUT": pvc_out_dir_b,
                "VLLM_DIR": vllm_dir,
                "HF_HOME": hf_home,
                "BENCH_CMD": bench_command,
                "SERVER_CMD": args.server_command.strip(),
                "DISABLE_ENV_VAR": args.disable_env_var,
                "DELAY_ITERATIONS": str(args.delay_iterations),
            },
        )

        print("[cluster_runner] submitting BASELINE nsys pod...", file=sys.stderr)
        _run_job(
            namespace=args.namespace,
            job_name=job_name_b,
            manifest=manifest_b,
            gpu_count=args.gpu_count,
            gpu_product=args.gpu_product,
            pvc_out_dir=pvc_out_dir_b,
            local_out_dir=args.out_dir,
            gpu_wait_timeout_minutes=args.gpu_wait_timeout_minutes,
            pvc_name=args.pvc_name,
            yaml_name="run_diagnostic_baseline.yaml",
            entrypoint_script=entrypoint_b,
        )

        status = (
            Path(status_file).read_text().strip()
            if os.path.isfile(status_file)
            else "NO_STATUS_FILE"
        )
        print(f"[cluster_runner] diagnostic_status={status}", file=sys.stderr)

        if status != "NSYS_DONE":
            print(
                f"ERROR: diagnostic did not complete (status={status})",
                file=sys.stderr,
            )
            sys.exit(1)

    print(
        f"[cluster_runner] nsys done — SQLite traces in {args.out_dir}",
        file=sys.stderr,
    )
    print("[cluster_runner] agent will now run Steps 2-5 locally.", file=sys.stderr)


# ── subcommand: smoke-compile ────────────────────────────────────────────────


def cmd_smoke_compile(args):
    """Smoke test: venv setup + cmake build + import vllm.

    Uses the same venv_compile.sh code path as run_iteration.sh and
    run_diagnostic.sh, then verifies that `import vllm` succeeds.
    """
    nonce = uuid.uuid4().hex[:8]
    job_name = f"ai-auto-analyze-smoke-{nonce}"

    workspace = "/workspace"
    vllm_dir = getattr(args, "vllm_dir", None) or f"{workspace}/vllm"
    pvc_out_dir = f"{workspace}/output/{nonce}"
    hf_home = "/model-cache"

    entrypoint = _assemble_script("smoke_compile.sh")
    cm_name = f"ai-auto-analyze-smoke-{nonce}"
    _create_configmap(args.namespace, cm_name, entrypoint)

    manifest = _build_job_manifest(
        job_name=job_name,
        namespace=args.namespace,
        image=args.image,
        gpu_count=args.gpu_count,
        gpu_product=args.gpu_product,
        pvc_name=args.pvc_name,
        model_pvc_name=args.model_pvc_name,
        configmap_name=cm_name,
        ttl_seconds=args.ttl_seconds,
        env_vars={
            "OUT": pvc_out_dir,
            "VLLM_DIR": vllm_dir,
            "HF_HOME": hf_home,
        },
    )

    _run_job(
        namespace=args.namespace,
        job_name=job_name,
        manifest=manifest,
        gpu_count=args.gpu_count,
        gpu_product=args.gpu_product,
        pvc_out_dir=pvc_out_dir,
        local_out_dir=args.out_dir,
        gpu_wait_timeout_minutes=args.gpu_wait_timeout_minutes,
        pvc_name=args.pvc_name,
        yaml_name="smoke_compile.yaml",
        entrypoint_script=entrypoint,
    )

    status_file = os.path.join(args.out_dir, "smoke_status.txt")
    status = (
        Path(status_file).read_text().strip()
        if os.path.isfile(status_file)
        else "NO_STATUS_FILE"
    )
    print(f"[cluster_runner] smoke_status={status}", file=sys.stderr)

    if status != "SMOKE_OK":
        print(f"ERROR: smoke test failed (status={status})", file=sys.stderr)
        sys.exit(1)

    print(
        f"[cluster_runner] smoke test passed — results in {args.out_dir}",
        file=sys.stderr,
    )


# ── subcommand: smoke-lm-eval ───────────────────────────────────────────────


def cmd_smoke_lm_eval(args):
    """Smoke test: venv + cmake + install lm-eval[api] + verify import chain.

    Catches torch/torchvision ABI mismatches, missing operators, and other
    dependency conflicts before a real benchmark wastes GPU time.
    """
    nonce = uuid.uuid4().hex[:8]
    job_name = f"ai-auto-analyze-smoke-lmeval-{nonce}"

    workspace = "/workspace"
    vllm_dir = getattr(args, "vllm_dir", None) or f"{workspace}/vllm"
    pvc_out_dir = f"{workspace}/output/{nonce}"
    hf_home = "/model-cache"

    entrypoint = _assemble_script("smoke_lm_eval.sh")
    cm_name = f"ai-auto-analyze-smoke-lmeval-{nonce}"
    _create_configmap(args.namespace, cm_name, entrypoint)

    manifest = _build_job_manifest(
        job_name=job_name,
        namespace=args.namespace,
        image=args.image,
        gpu_count=args.gpu_count,
        gpu_product=args.gpu_product,
        pvc_name=args.pvc_name,
        model_pvc_name=args.model_pvc_name,
        configmap_name=cm_name,
        ttl_seconds=args.ttl_seconds,
        env_vars={
            "OUT": pvc_out_dir,
            "VLLM_DIR": vllm_dir,
            "HF_HOME": hf_home,
        },
    )

    _run_job(
        namespace=args.namespace,
        job_name=job_name,
        manifest=manifest,
        gpu_count=args.gpu_count,
        gpu_product=args.gpu_product,
        pvc_out_dir=pvc_out_dir,
        local_out_dir=args.out_dir,
        gpu_wait_timeout_minutes=args.gpu_wait_timeout_minutes,
        pvc_name=args.pvc_name,
        yaml_name="smoke_lm_eval.yaml",
        entrypoint_script=entrypoint,
    )

    status_file = os.path.join(args.out_dir, "smoke_status.txt")
    status = (
        Path(status_file).read_text().strip()
        if os.path.isfile(status_file)
        else "NO_STATUS_FILE"
    )
    print(f"[cluster_runner] smoke_status={status}", file=sys.stderr)

    if status != "SMOKE_OK":
        runner_log = os.path.join(args.out_dir, "runner.log")
        if os.path.isfile(runner_log):
            print("--- runner.log (last 40 lines) ---", file=sys.stderr)
            with open(runner_log) as f:
                lines = f.readlines()
                for line in lines[-40:]:
                    print(line, end="", file=sys.stderr)
        print(
            f"ERROR: smoke-lm-eval test failed (status={status})",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"[cluster_runner] smoke-lm-eval test passed — results in {args.out_dir}",
        file=sys.stderr,
    )


# ── subcommand: reset-workspace ──────────────────────────────────────────────


def cmd_reset_workspace(args):
    """Reset /workspace/vllm on the PVC to a clean state at a given commit.

    Spins up a helper pod, runs git reset + clean, then tears down the pod.

    With --keep-cmake (the default), cmake-build-release/ is preserved so
    the next build is incremental (~2 min for patched files only).
    Without --keep-cmake, cmake-build-release/ is deleted for a full
    from-scratch compilation (~13 min).
    """
    helper = f"ai-auto-analyze-reset-{uuid.uuid4().hex[:6]}"
    overrides = json.dumps(
        {
            "spec": {
                "containers": [
                    {
                        "name": helper,
                        "image": args.image,
                        "command": ["sleep", "600"],
                        "volumeMounts": [{"name": "ws", "mountPath": "/workspace"}],
                    }
                ],
                "volumes": [
                    {
                        "name": "ws",
                        "persistentVolumeClaim": {"claimName": args.pvc_name},
                    }
                ],
            }
        }
    )

    vllm_dir = args.vllm_dir
    commit = args.commit
    keep_cmake = args.keep_cmake

    try:
        print(
            f"[reset-workspace] creating helper pod {helper}...",
            file=sys.stderr,
        )
        _run_or_die(
            "run",
            helper,
            "-n",
            args.namespace,
            "--image=" + args.image,
            "--overrides=" + overrides,
            "--restart=Never",
        )
        _run_or_die(
            "wait",
            "-n",
            args.namespace,
            "pod/" + helper,
            "--for=condition=Ready",
            "--timeout=120s",
        )

        def _exec(*cmd):
            r = _oc("exec", "-n", args.namespace, helper, "--", *cmd, check=False)
            if r.stdout:
                print(r.stdout.rstrip(), file=sys.stderr)
            if r.stderr:
                print(r.stderr.rstrip(), file=sys.stderr)
            return r.returncode

        print(f"[reset-workspace] checking {vllm_dir}...", file=sys.stderr)
        rc = _exec("test", "-d", vllm_dir)
        if rc != 0:
            print(f"ERROR: {vllm_dir} does not exist on PVC", file=sys.stderr)
            sys.exit(1)

        print(f"[reset-workspace] git reset --hard {commit}...", file=sys.stderr)
        rc = _exec("git", "-C", vllm_dir, "reset", "--hard", commit)
        if rc != 0:
            print(f"ERROR: git reset --hard failed (exit {rc})", file=sys.stderr)
            sys.exit(1)

        clean_excludes = ["-e", "cmake-build-release/"]
        if keep_cmake:
            print(
                "[reset-workspace] git clean -fd (keeping cmake cache)...",
                file=sys.stderr,
            )
            rc = _exec("git", "-C", vllm_dir, "clean", "-fd", *clean_excludes)
        else:
            print(
                "[reset-workspace] git clean -fdx + removing cmake cache...",
                file=sys.stderr,
            )
            rc = _exec(
                "git",
                "-C",
                vllm_dir,
                "clean",
                "-fdx",
                "-e",
                "cmake-build-release/_deps/",
            )
            cmake_dir = f"{vllm_dir}/cmake-build-release"
            _exec("rm", "-rf", cmake_dir)

        if rc != 0:
            print(f"WARNING: git clean failed (exit {rc})", file=sys.stderr)

        rc = _exec("git", "-C", vllm_dir, "log", "--oneline", "-1")
        _exec("git", "-C", vllm_dir, "status", "--short")

        mode = "incremental" if keep_cmake else "full rebuild"
        print(
            f"[reset-workspace] done — workspace is clean at {commit} ({mode})",
            file=sys.stderr,
        )

    except subprocess.CalledProcessError as e:
        print(f"ERROR: reset-workspace failed: {e.stderr or e}", file=sys.stderr)
        sys.exit(1)
    finally:
        _oc("delete", "pod", helper, "-n", args.namespace, "--wait=false", check=False)


# ── CLI ───────────────────────────────────────────────────────────────────────

_DEFAULT_IMAGE = "quay.io/example/vllm-sandbox-nsys:latest"

_COMMON_ARGS = [
    ("--namespace", dict(default=None)),
    (
        "--pvc-name",
        dict(
            default="vllm-workspace",
            help="Workspace PVC (vllm source, pip-cache, output)",
        ),
    ),
    (
        "--model-pvc-name",
        dict(
            required=True,
            help="PVC mounted at /model-cache; sets HF_HOME=/model-cache. "
            "Model is downloaded there on first run if not cached.",
        ),
    ),
    ("--image", dict(default=_DEFAULT_IMAGE)),
    ("--gpu-count", dict(type=int, default=4)),
    ("--gpu-product", dict(default="NVIDIA-B200")),
    ("--out-dir", dict(required=True, help="Local directory for result files")),
    (
        "--vllm-dir",
        dict(
            default="/workspace/vllm",
            help="This run's vllm source dir on the PVC. "
            "The pipeline passes a run-named subdir "
            "(/workspace/<output_dir basename>) so "
            "concurrent runs never share a tree.",
        ),
    ),
    ("--gpu-wait-timeout-minutes", dict(type=int, default=60)),
    ("--ttl-seconds", dict(type=int, default=600)),
    (
        "--extra-env",
        dict(
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="Extra environment variable for the Job "
            "container; repeatable. Pre-seeded model "
            "caches are often read-only, so pass "
            "HF_HUB_OFFLINE=1 in that case.",
        ),
    ),
]


def _add_common(parser):
    for flag, kw in _COMMON_ARGS:
        parser.add_argument(flag, **kw)


def main():
    parser = argparse.ArgumentParser(
        description="Submit GPU Jobs to OpenShift for the remote runtime backend"
    )
    sub = parser.add_subparsers(dest="command")

    # ── run-iteration ──────────────────────────────────────────────────────────
    _ITERATION_ARGS = [
        (
            "--bench-command",
            dict(
                required=True,
                help="Frozen benchmark command (must write --output-json to bench_result.json)",
            ),
        ),
        (
            "--server-command",
            dict(
                required=True,
                help="Command that launches the vLLM OpenAI-compatible server",
            ),
        ),
        (
            "--disable-env-var",
            dict(required=True, help="Env var whose value=1 disables the new feature"),
        ),
        (
            "--lm-eval-command",
            dict(
                default="",
                help="Override lm_eval command (auto-derived from --bench-command if omitted)",
            ),
        ),
        ("--run-lm-eval", dict(action="store_true", default=False)),
        (
            "--patch-file",
            dict(
                default="",
                help="Patch file path on the PVC. When set, the job copies "
                "/workspace/vllm to --vllm-dir and applies this patch before building.",
            ),
        ),
        (
            "--skip-lm-eval-baseline",
            dict(
                action="store_true",
                default=False,
                help="Skip lm_eval baseline (feature OFF) run — reuse shared baseline from a previous iteration",
            ),
        ),
        (
            "--stage",
            dict(
                default="",
                help="Run only this stage: 'bench', 'baseline', 'lm_eval', or '' for all",
            ),
        ),
        (
            "--force-feature-off",
            dict(
                action="store_true",
                default=False,
                help="Force the feature OFF for this pod (sets DISABLE_ENV_VAR=1 + FORCE_FEATURE_OFF=1)",
            ),
        ),
    ]

    ri = sub.add_parser(
        "run-iteration",
        help="Compile + bench patched + bench baseline + optional lm_eval (sequential)",
    )
    _add_common(ri)
    for flag, kw in _ITERATION_ARGS:
        ri.add_argument(
            flag,
            **kw,  # pyrefly: ignore [bad-argument-type] (runtime-defined argparse options)
        )

    rip = sub.add_parser(
        "run-iteration-parallel",
        help="Bench + baseline + optional lm_eval as parallel GPU pods",
    )
    _add_common(rip)
    for flag, kw in _ITERATION_ARGS:
        rip.add_argument(
            flag,
            **kw,  # pyrefly: ignore [bad-argument-type] (runtime-defined argparse options)
        )

    # ── run-diagnostic ─────────────────────────────────────────────────────────
    rd = sub.add_parser(
        "run-diagnostic",
        help="Compile + nsys profile patched + nsys profile baseline + SQLite export",
    )
    _add_common(rd)
    rd.add_argument("--bench-command", required=True)
    rd.add_argument("--disable-env-var", required=True)
    rd.add_argument(
        "--server-command",
        default="",
        help="Command that launches the server. Required to profile a "
        "client-style bench (`vllm bench serve`): the engine lives "
        "in the server process, so nsys must wrap the server.",
    )
    rd.add_argument(
        "--delay-iterations",
        type=int,
        default=512,
        help="NSYS delay_iterations (warmup offset before tracing starts)",
    )
    rd.add_argument(
        "--skip-baseline",
        action="store_true",
        default=False,
        help="Skip baseline nsys profile (reuse shared baseline from a previous iteration)",
    )

    # ── smoke-compile ──────────────────────────────────────────────────────────
    sc = sub.add_parser(
        "smoke-compile", help="Smoke test: venv setup + cmake build + verification only"
    )
    _add_common(sc)
    sc.set_defaults(gpu_count=1)

    # ── smoke-lm-eval ─────────────────────────────────────────────────────────
    sle = sub.add_parser(
        "smoke-lm-eval",
        help="Smoke test: venv + build + lm-eval install + import chain verification",
    )
    _add_common(sle)
    sle.set_defaults(gpu_count=1)

    # ── reset-workspace ───────────────────────────────────────────────────────
    rw = sub.add_parser(
        "reset-workspace",
        help="Reset /workspace/vllm on PVC to clean state at a commit",
    )
    rw.add_argument("--namespace", default=None)
    rw.add_argument("--pvc-name", default="vllm-workspace", help="Workspace PVC name")
    rw.add_argument(
        "--vllm-dir", default="/workspace/vllm", help="Path to vllm source on the PVC"
    )
    rw.add_argument(
        "--commit",
        required=True,
        help="Git commit/ref to reset to (e.g. HEAD, a branch, or a SHA)",
    )
    rw.add_argument(
        "--image",
        default=_DEFAULT_IMAGE,
        help="Helper image; must provide git (ubi9 base does not)",
    )
    rw.add_argument(
        "--keep-cmake",
        action="store_true",
        default=True,
        help="Preserve cmake-build-release/ for incremental builds (default)",
    )
    rw.add_argument(
        "--no-keep-cmake",
        dest="keep_cmake",
        action="store_false",
        help="Delete cmake-build-release/ for a full from-scratch compilation",
    )

    args = parser.parse_args()
    _EXTRA_ENV.update(_parse_extra_env(getattr(args, "extra_env", None)))

    if args.command == "run-iteration":
        cmd_run_iteration(args)
    elif args.command == "run-iteration-parallel":
        cmd_run_iteration_parallel(args)
    elif args.command == "run-diagnostic":
        cmd_run_diagnostic(args)
    elif args.command == "smoke-compile":
        cmd_smoke_compile(args)
    elif args.command == "smoke-lm-eval":
        cmd_smoke_lm_eval(args)
    elif args.command == "reset-workspace":
        cmd_reset_workspace(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
