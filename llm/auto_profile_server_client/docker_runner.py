"""Docker container lifecycle management for benchmark/profiling runs."""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from llm.auto_profile_server_client.config import (
    DOCKER_BASE_DIR,
    DOCKER_HF_HUB_CACHE,
    DOCKER_SHM_SIZE,
)

log = logging.getLogger(__name__)


def _gpu_vendor() -> str:
    """Detect the host GPU vendor from its management CLI."""
    if shutil.which("nvidia-smi"):
        return "nvidia"
    if shutil.which("rocm-smi"):
        return "amd"
    return "unknown"


@dataclass
class DockerContainer:
    name: str
    process: subprocess.Popen | None = None
    image: str = ""

    def logs(self, tail: int = 50) -> str:
        r = subprocess.run(
            ["docker", "logs", "--tail", str(tail), self.name],
            capture_output=True, text=True, timeout=30,
        )
        return r.stdout + r.stderr

    def stop(self) -> None:
        log.info("Stopping container: %s", self.name)
        subprocess.run(["docker", "stop", self.name],
                        capture_output=True, timeout=60)
        subprocess.run(["docker", "rm", "-f", self.name],
                        capture_output=True, timeout=30)
        if self.process:
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def is_running(self) -> bool:
        r = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", self.name],
            capture_output=True, text=True, timeout=10,
        )
        return r.returncode == 0 and r.stdout.strip() == "true"


class DockerRunner:
    def __init__(
        self,
        base_dir: str | None = None,
        hf_hub_cache: str | None = None,
    ):
        self.base_dir = base_dir or os.getcwd()
        self.hf_hub_cache = hf_hub_cache or os.environ.get(
            "HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub")
        )

    def _base_docker_args(
        self,
        image: str,
        name: str,
        gpu_ids: list[int] | None = None,
        ports: dict[int, int] | None = None,
        env: dict[str, str] | None = None,
        extra_volumes: dict[str, str] | None = None,
    ) -> list[str]:
        args = [
            "docker", "run",
            "--rm",
            "--ulimit", "memlock=-1",
            "--ulimit", "stack=67108864",
            "--ipc=host",
            "--shm-size", DOCKER_SHM_SIZE,
        ]

        if _gpu_vendor() == "amd":
            # ROCm: expose the compute/render devices and loosen ptrace/seccomp
            # so profilers inside the container can attach (matches the docker
            # invocations in the vLLM/SGLang ROCm docs).
            args += [
                "--device", "/dev/kfd",
                "--device", "/dev/dri",
                "--group-add", "video",
                "--cap-add", "SYS_PTRACE",
                "--security-opt", "seccomp=unconfined",
                # ROCR's auto-probe picks the legacy IPC mode inside
                # containers, which exhausts KFD event slots and fails RCCL
                # init with "hipIpcGetMemHandle: invalid argument". Forcing
                # the dmabuf IPC path fixes multi-GPU comms. Listed before
                # the framework extra_env so a config can still override it.
                "--env", "HSA_ENABLE_IPC_MODE_LEGACY=0",
            ]
        else:
            args += ["--runtime", "nvidia", "--gpus=all"]

        args += [
            "-v", f"{self.base_dir}:{DOCKER_BASE_DIR}",
            "-v", f"{self.hf_hub_cache}:{DOCKER_HF_HUB_CACHE}",
            "--env", f"HF_HUB_CACHE={DOCKER_HF_HUB_CACHE}",
        ]

        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            args += ["--env", f"HF_TOKEN={hf_token}"]

        if gpu_ids:
            args += ["--env", f"CUDA_VISIBLE_DEVICES={','.join(str(g) for g in gpu_ids)}"]

        if ports:
            for host_port, container_port in ports.items():
                args += ["-p", f"{host_port}:{container_port}"]

        if env:
            for key, val in env.items():
                args += ["--env", f"{key}={val}"]

        if extra_volumes:
            for host_path, container_path in extra_volumes.items():
                args += ["-v", f"{host_path}:{container_path}"]

        nsys_dir = self._find_nsys_dir()
        if nsys_dir:
            args += ["-v", f"{nsys_dir}:/nsys:ro"]
            args += ["--env", "PATH=/nsys/bin:$PATH"]

        args += ["--name", name, "--entrypoint", "/bin/bash"]
        args.append(image)
        return args

    def _find_nsys_dir(self) -> str | None:
        candidates = sorted(
            Path("/opt/nvidia/nsight-systems").glob("*/bin/nsys"),
            reverse=True,
        ) if Path("/opt/nvidia/nsight-systems").is_dir() else []
        if candidates:
            return str(candidates[0].parent.parent.resolve())
        return None

    def _remove_if_exists(self, name: str) -> None:
        r = subprocess.run(
            ["docker", "container", "inspect", name],
            capture_output=True, timeout=10,
        )
        if r.returncode == 0:
            log.warning("Container %s exists, removing", name)
            subprocess.run(["docker", "stop", name], capture_output=True, timeout=60)
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)

    def run(
        self,
        image: str,
        name: str,
        cmd: str,
        gpu_ids: list[int] | None = None,
        ports: dict[int, int] | None = None,
        env: dict[str, str] | None = None,
        extra_volumes: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess:
        """Run a command inside a docker container (blocking)."""
        self._remove_if_exists(name)
        args = self._base_docker_args(image, name, gpu_ids, ports, env, extra_volumes)
        args += ["-c", f"cd {DOCKER_BASE_DIR} && {cmd}"]

        log.info("docker run [%s]: %s", name, cmd[:200])
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            log.error("Container %s exited with code %d:\n%s",
                      name, result.returncode, result.stderr[-2000:])
        return result

    def run_background(
        self,
        image: str,
        name: str,
        cmd: str,
        gpu_ids: list[int] | None = None,
        ports: dict[int, int] | None = None,
        env: dict[str, str] | None = None,
        extra_volumes: dict[str, str] | None = None,
    ) -> DockerContainer:
        """Start a container in the background."""
        self._remove_if_exists(name)
        args = self._base_docker_args(image, name, gpu_ids, ports, env, extra_volumes)
        args += ["-c", f"cd {DOCKER_BASE_DIR} && {cmd}"]

        log.info("docker run (background) [%s]: %s", name, cmd[:200])
        proc = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        return DockerContainer(name=name, process=proc, image=image)

    def ensure_image_updated(self, image: str) -> None:
        """Pull latest if remote digest differs from local."""
        if not image or image == "none":
            return

        log.info("Checking for updates: %s", image)
        try:
            r = subprocess.run(
                ["docker", "manifest", "inspect", image],
                capture_output=True, text=True, timeout=60,
            )
            if r.returncode != 0:
                log.info("Could not fetch remote manifest for %s, skipping", image)
                return

            manifest = json.loads(r.stdout)
            remote_digest = ""
            for entry in manifest.get("manifests", []):
                plat = entry.get("platform", {})
                if plat.get("architecture") == "amd64" and plat.get("os") == "linux":
                    remote_digest = entry.get("digest", "")
                    break
            if not remote_digest:
                remote_digest = manifest.get("config", {}).get("digest", "")

            r_local = subprocess.run(
                ["docker", "inspect", image, "--format", "{{index .RepoDigests 0}}"],
                capture_output=True, text=True, timeout=30,
            )
            local_digest = ""
            if r_local.returncode == 0 and "@" in r_local.stdout:
                local_digest = r_local.stdout.strip().split("@", 1)[1]

            if not local_digest:
                log.info("Image not found locally, pulling: %s", image)
                subprocess.run(["docker", "pull", image], timeout=600)
            elif local_digest != remote_digest:
                log.info("Newer version available, pulling: %s", image)
                subprocess.run(["docker", "pull", image], timeout=600)
            else:
                log.info("Up to date: %s", image)

        except Exception:
            log.warning("Failed to check for updates: %s", image, exc_info=True)


def _query_gpu_memory_used_mib(vendor: str) -> int | None:
    """Max per-GPU memory in use (MiB), or None if it cannot be read."""
    try:
        if vendor == "amd":
            r = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram", "--json"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode != 0:
                return None
            data = json.loads(r.stdout)
            return max(
                int(gpu["VRAM Total Used Memory (B)"])
                for gpu in data.values()
            ) // (1024 * 1024)
        if vendor == "nvidia":
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode != 0:
                return None
            return max(
                int(line.strip())
                for line in r.stdout.strip().split("\n") if line.strip()
            )
    except Exception:
        pass
    return None


def wait_for_gpu_memory_release(max_wait: int = 120, threshold_mib: int = 2000) -> bool:
    """Wait until GPU memory usage drops below threshold."""
    vendor = _gpu_vendor()
    if vendor == "unknown":
        log.warning("No GPU management CLI found (nvidia-smi/rocm-smi); "
                    "skipping GPU memory wait")
        return False

    interval = 5
    elapsed = 0
    log.info("Waiting for GPU memory release (vendor=%s, threshold=%d MiB, timeout=%ds)",
             vendor, threshold_mib, max_wait)

    while elapsed < max_wait:
        used_mib = _query_gpu_memory_used_mib(vendor)
        if used_mib is not None:
            if used_mib <= threshold_mib:
                log.info("GPU memory released (max used: %d MiB, elapsed: %ds)",
                         used_mib, elapsed)
                return True
            log.info("  GPU memory in use: %d MiB (%d/%ds)",
                     used_mib, elapsed, max_wait)

        time.sleep(interval)
        elapsed += interval

    log.warning("GPU memory not released after %ds", max_wait)
    return False
