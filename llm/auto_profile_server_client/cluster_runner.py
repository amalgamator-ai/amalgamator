# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Cluster execution for the server-client benchmark pipeline.

Submits OpenShift Jobs that run the benchmark/profiling pipeline inside
GPU pods. Reads model, frameworks, GPU count, and docker images from the
run config JSON — only cluster-specific options (namespace, PVCs, GPU
product label) are CLI flags.

Logs and results are synced live to the local machine every poll interval
so you can ``tail -f`` server/client logs while the benchmark runs.

Usage:
  python -m llm.auto_profile_server_client.cluster_runner run-benchmark \
    --config llm/auto_profile/test_configs/deepseek_r1_nvfp4/run_deepseek_r1_nvfp4.json \
    --gpu-product NVIDIA-B200
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import textwrap
import time
import uuid
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


# ── oc helpers ────────────────────────────────────────────────────────────────


def _oc(*args, check=True, capture=True, input_text=None):
    return subprocess.run(
        ["oc"] + list(args),
        check=check,
        capture_output=capture,
        text=True,
        input=input_text,
    )


def _wait_for_pod(namespace, job_name, timeout_s=600, poll_s=10):
    """Wait until Job pod leaves Pending. Returns (pod_name, phase)."""
    deadline = time.time() + timeout_s
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
                if phase not in ("Pending", "Unknown"):
                    return name, phase
        time.sleep(poll_s)
    raise TimeoutError(f"Timed out waiting for pod of Job {job_name}")


# ── Live sync ────────────────────────────────────────────────────────────────


def _sync_pod_log(namespace, pod_name, local_dir):
    """Fetch the full pod log and write it to local_dir/pod.log."""
    r = _oc("logs", "-n", namespace, pod_name, check=False)
    if r.returncode == 0 and r.stdout:
        Path(local_dir, "pod.log").write_text(r.stdout)


def _sync_results(namespace, pod_name, remote_dir, local_dir):
    """Rsync output files from the running pod to local."""
    r = _oc(
        "rsync",
        f"{pod_name}:{remote_dir}/",
        f"{local_dir}/",
        "-n",
        namespace,
        "--no-perms",
        check=False,
    )
    if r.returncode != 0:
        log.debug("rsync interim: %s", (r.stderr or "")[:200])


def _poll_until_done(
    namespace, job_name, pod_name, remote_dir, local_dir, timeout_s=7200, poll_s=30
):
    """Poll job status, sync logs and results to local every interval."""
    deadline = time.time() + timeout_s
    terminal = None

    while time.time() < deadline:
        # Check job status
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

        # Print log snapshot to terminal
        r_tail = _oc("logs", "-n", namespace, pod_name, "--tail=20", check=False)
        if r_tail.returncode == 0 and r_tail.stdout.strip():
            print("--- log snapshot ---", file=sys.stderr)
            print(r_tail.stdout, file=sys.stderr)

        # Sync pod log + output files to local
        _sync_pod_log(namespace, pod_name, local_dir)
        _sync_results(namespace, pod_name, remote_dir, local_dir)

        time.sleep(poll_s)

    if terminal is None:
        raise TimeoutError(f"Timed out waiting for Job {job_name}")

    # Final pod log (oc logs works on Completed pods)
    _sync_pod_log(namespace, pod_name, local_dir)

    r_final = _oc("logs", "-n", namespace, pod_name, "--tail=80", check=False)
    if r_final.returncode == 0:
        print("--- final logs ---", file=sys.stderr)
        print(r_final.stdout, file=sys.stderr)

    return terminal


# ── Final result pull (via sync pod, works after job completion) ──────────────


SYNC_IMAGE = "registry.access.redhat.com/ubi9/ubi:latest"


def _pull_results_via_pvc(namespace, workspace_pvc, remote_dir, local_dir):
    """Pull results from PVC via a temporary sync pod.

    Unlike oc rsync from a running pod, this works after the job pod has
    completed because the sync pod mounts the PVC directly.
    """
    pod_name = f"results-sync-{uuid.uuid4().hex[:6]}"
    log.info("Final pull: %s → %s (via sync pod)", remote_dir, local_dir)

    pod_yaml = textwrap.dedent(f"""\
        apiVersion: v1
        kind: Pod
        metadata:
          name: {pod_name}
        spec:
          restartPolicy: Never
          containers:
          - name: sync
            image: {SYNC_IMAGE}
            command: ["sleep", "300"]
            volumeMounts:
            - name: workspace
              mountPath: /workspace
          volumes:
          - name: workspace
            persistentVolumeClaim:
              claimName: {workspace_pvc}
    """)

    try:
        _oc("apply", "-n", namespace, "-f", "-", input_text=pod_yaml)
        _oc(
            "wait",
            f"pod/{pod_name}",
            "-n",
            namespace,
            "--for=condition=Ready",
            "--timeout=120s",
        )

        os.makedirs(local_dir, exist_ok=True)
        _oc(
            "rsync",
            f"{pod_name}:{remote_dir}/",
            f"{local_dir}/",
            "-n",
            namespace,
            "--no-perms",
            check=True,
        )
        log.info("Results saved to %s", local_dir)
    except subprocess.CalledProcessError as e:
        log.warning("Failed to pull results: %s", e)
    finally:
        _oc("delete", "pod", pod_name, "-n", namespace, "--wait=false", check=False)


# ── Config loading ────────────────────────────────────────────────────────────


def _load_run_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


def _extract_from_config(raw: dict) -> tuple[str, int, list[str], dict[str, str]]:
    """Extract model, num_gpus, framework names, and docker_images from config."""
    profile = raw["profiles"][0]
    model = profile["model"]
    num_gpus = profile["num_gpus"]
    frameworks = [fw["name"] for fw in profile["frameworks"]]
    docker_images = raw.get("docker_images", {})
    return model, num_gpus, frameworks, docker_images


def _model_slug(model: str) -> str:
    """Convert model ID to a filesystem-safe slug."""
    return re.sub(r"[^a-zA-Z0-9._-]", "_", model.replace("/", "_")).lower()


def _make_local_out_dir(base_dir: str, fw: str) -> str:
    """Create a timestamped local output directory for a framework run."""
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = os.path.join(base_dir, f"{fw}_{ts}")
    os.makedirs(path, exist_ok=True)
    return path


# ── Job manifest ─────────────────────────────────────────────────────────────


def _build_benchmark_job_yaml(
    job_name: str,
    namespace: str,
    image: str,
    gpu_count: int,
    gpu_product: str,
    workspace_pvc: str,
    model_pvc: str,
    config_path: str,
    framework: str,
    model: str,
    output_nonce: str,
) -> str:

    return textwrap.dedent(f"""\
        apiVersion: batch/v1
        kind: Job
        metadata:
          name: {job_name}
          namespace: {namespace}
        spec:
          backoffLimit: 0
          ttlSecondsAfterFinished: 600
          template:
            spec:
              restartPolicy: Never
              nodeSelector:
                nvidia.com/gpu.product: {gpu_product}
              containers:
              - name: bench
                image: {image}
                command: ["bash", "-c"]
                args:
                - |
                  set -uo pipefail
                  mkdir -p /workspace/output/{output_nonce}
                  cd /workspace
                  pip install colorama 2>/dev/null || pip3 install colorama 2>/dev/null || true
                  export PATH="/workspace/tools/nsys/opt/nvidia/nsight-systems/2026.1.3/bin:${{PATH}}"
                  export PYTHONPATH=/workspace:${{PYTHONPATH:-}}
                  python3 -m auto_profile_server_client \\
                    --config {config_path} \\
                    --framework {framework} \\
                    --model '{model}' \\
                    --output-dir /workspace/output/{output_nonce}
                  if [ $? -eq 0 ]; then
                    echo "done" > /workspace/output/{output_nonce}/run_status.txt
                  fi
                resources:
                  limits:
                    nvidia.com/gpu: "{gpu_count}"
                env:
                - name: HF_HOME
                  value: /model-cache
                - name: USER
                  value: benchmark
                - name: HOME
                  value: /tmp
                volumeMounts:
                - name: workspace
                  mountPath: /workspace
                - name: model-cache
                  mountPath: /model-cache
                - name: dshm
                  mountPath: /dev/shm
              volumes:
              - name: workspace
                persistentVolumeClaim:
                  claimName: {workspace_pvc}
              - name: model-cache
                persistentVolumeClaim:
                  claimName: {model_pvc}
              - name: dshm
                emptyDir:
                  medium: Memory
                  sizeLimit: "32Gi"
    """)


# ── Subcommands ──────────────────────────────────────────────────────────────


def _submit_and_wait(args, job_name, yaml, local_dir, nonce):
    """Submit a job, poll until done, pull results."""
    os.makedirs(f"{local_dir}/cluster_yamls", exist_ok=True)
    yaml_path = f"{local_dir}/cluster_yamls/{job_name}.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml)

    print(yaml, file=sys.stderr)
    _oc("apply", "-f", "-", input_text=yaml)
    log.info("Submitted Job %s", job_name)
    log.info("Live logs: tail -f %s/pod.log", local_dir)

    pod_name, phase = _wait_for_pod(args.namespace, job_name, timeout_s=1800)
    log.info("Pod %s is %s", pod_name, phase)

    remote_dir = f"/workspace/output/{nonce}"
    terminal = _poll_until_done(
        args.namespace,
        job_name,
        pod_name,
        remote_dir=remote_dir,
        local_dir=local_dir,
    )
    log.info("Job %s finished: %s", job_name, terminal)

    _pull_results_via_pvc(
        namespace=args.namespace,
        workspace_pvc=args.workspace_pvc,
        remote_dir=remote_dir,
        local_dir=local_dir,
    )

    if terminal != "Complete":
        log.error("Job %s failed — check %s/pod.log", job_name, local_dir)
        sys.exit(1)

    return pod_name


def run_benchmark(args):
    """Submit benchmark + profiling jobs to the cluster.

    Single job: benchmark (stability runs + metrics) then profiling
    (nsys capture with iteration-based decode targeting) in the same pod.
    """
    raw = _load_run_config(args.config)
    model, num_gpus, frameworks, docker_images = _extract_from_config(raw)

    fw_filter = args.framework
    if fw_filter:
        if fw_filter not in frameworks:
            log.error(
                "Framework %r not in config (available: %s)", fw_filter, frameworks
            )
            sys.exit(1)
        frameworks = [fw_filter]

    gpu_count = args.gpu_count or num_gpus

    base_out_dir = args.out_dir or os.path.join(
        "auto_profile_server_client",
        "results",
        _model_slug(model),
    )

    log.info("Config: model=%s gpus=%d frameworks=%s", model, gpu_count, frameworks)
    log.info("Local output: %s", base_out_dir)

    for fw in frameworks:
        image = args.image or docker_images.get(fw)
        if not image:
            log.error("No docker image for framework %r (set in config or --image)", fw)
            sys.exit(1)

        nonce = uuid.uuid4().hex[:8]
        local_fw_dir = _make_local_out_dir(base_out_dir, fw)

        # Add a file handler so the full run log is saved alongside results
        run_log_path = os.path.join(local_fw_dir, "run.log")
        file_handler = logging.FileHandler(run_log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logging.getLogger().addHandler(file_handler)
        log.info("Run log: %s", run_log_path)

        log.info("=== Submitting job: %s (%s) ===", fw, nonce)
        job_name = f"bench-{fw}-{nonce}"
        yaml = _build_benchmark_job_yaml(
            job_name=job_name,
            namespace=args.namespace,
            image=image,
            gpu_count=gpu_count,
            gpu_product=args.gpu_product,
            workspace_pvc=args.workspace_pvc,
            model_pvc=args.model_pvc,
            config_path=args.config,
            framework=fw,
            model=model,
            output_nonce=nonce,
        )
        _submit_and_wait(args, job_name, yaml, local_fw_dir, nonce)
        log.info("Job complete — results at %s", local_fw_dir)
        log.info("Run log: %s", run_log_path)
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()


def main():
    parser = argparse.ArgumentParser(
        description="Cluster runner for auto_profile benchmark/profiling"
    )
    sub = parser.add_subparsers(dest="command")

    bench_parser = sub.add_parser("run-benchmark", help="Submit benchmark Job(s)")
    bench_parser.add_argument(
        "--config",
        required=True,
        help="Run config JSON (model, frameworks, images read from here)",
    )
    bench_parser.add_argument("--namespace", default="", help="OpenShift namespace")
    bench_parser.add_argument(
        "--gpu-product", default="NVIDIA-H100-80GB-HBM3", help="GPU node selector label"
    )
    bench_parser.add_argument(
        "--workspace-pvc", default="vllm-workspace", help="PVC for workspace"
    )
    bench_parser.add_argument(
        "--model-pvc", default="models", help="PVC for HF model cache"
    )
    bench_parser.add_argument("--out-dir", help="Local output base directory")
    bench_parser.add_argument(
        "--framework", help="Run only this framework (default: all in config)"
    )
    bench_parser.add_argument(
        "--image", help="Override docker image (default: from config)"
    )
    bench_parser.add_argument(
        "--gpu-count", type=int, help="Override GPU count (default: from config)"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.command == "run-benchmark":
        run_benchmark(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
