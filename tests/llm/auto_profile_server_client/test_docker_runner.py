"""Tests for vendor-aware docker args and GPU memory polling."""

import json
import subprocess

import pytest

from llm.auto_profile_server_client import docker_runner
from llm.auto_profile_server_client.docker_runner import (
    DockerRunner,
    _gpu_vendor,
    _query_gpu_memory_used_mib,
    wait_for_gpu_memory_release,
)


def _completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode,
                                       stdout=stdout, stderr="")


class TestGpuVendor:
    def test_nvidia(self, monkeypatch):
        monkeypatch.setattr(docker_runner.shutil, "which",
                            lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None)
        assert _gpu_vendor() == "nvidia"

    def test_amd(self, monkeypatch):
        monkeypatch.setattr(docker_runner.shutil, "which",
                            lambda name: "/usr/bin/rocm-smi" if name == "rocm-smi" else None)
        assert _gpu_vendor() == "amd"

    def test_unknown(self, monkeypatch):
        monkeypatch.setattr(docker_runner.shutil, "which", lambda name: None)
        assert _gpu_vendor() == "unknown"

    def test_nvidia_wins_when_both_present(self, monkeypatch):
        monkeypatch.setattr(docker_runner.shutil, "which", lambda name: f"/usr/bin/{name}")
        assert _gpu_vendor() == "nvidia"


class TestBaseDockerArgs:
    def _args(self, monkeypatch, vendor):
        monkeypatch.setattr(docker_runner, "_gpu_vendor", lambda: vendor)
        runner = DockerRunner(base_dir="/repo", hf_hub_cache="/hf")
        return runner._base_docker_args("img:tag", "cont")

    def test_amd_uses_rocm_devices(self, monkeypatch):
        args = self._args(monkeypatch, "amd")
        joined = " ".join(args)
        assert "--device /dev/kfd" in joined
        assert "--device /dev/dri" in joined
        assert "--group-add video" in joined
        assert "--cap-add SYS_PTRACE" in joined
        assert "--security-opt seccomp=unconfined" in joined
        # Force the dmabuf IPC path: ROCR's auto-probe picks the legacy mode
        # inside containers, which breaks RCCL multi-GPU init.
        assert "--env HSA_ENABLE_IPC_MODE_LEGACY=0" in joined
        # No nvidia runtime on AMD hosts
        assert "--runtime nvidia" not in joined
        assert "--gpus=all" not in joined

    def test_amd_keeps_common_args(self, monkeypatch):
        args = self._args(monkeypatch, "amd")
        joined = " ".join(args)
        assert "--ipc=host" in joined
        assert "--shm-size 32g" in joined
        assert "-v /repo:/app" in joined
        assert "-v /hf:/app/hf_hub_cache" in joined
        assert "HF_HUB_CACHE=/app/hf_hub_cache" in joined

    def test_nvidia_unchanged(self, monkeypatch):
        args = self._args(monkeypatch, "nvidia")
        joined = " ".join(args)
        assert "--runtime nvidia" in joined
        assert "--gpus=all" in joined
        assert "--device /dev/kfd" not in joined
        assert "--security-opt seccomp=unconfined" not in joined

    def test_unknown_falls_back_to_nvidia(self, monkeypatch):
        # Conservative default: the pre-AMD behavior
        args = self._args(monkeypatch, "unknown")
        assert "--runtime" in args and "nvidia" in args


ROCM_SMI_JSON = json.dumps({
    "card0": {"VRAM Total Memory (B)": "309220868096",
              "VRAM Total Used Memory (B)": "297766912"},
    "card1": {"VRAM Total Memory (B)": "309220868096",
              "VRAM Total Used Memory (B)": "1073741824"},
})

NVIDIA_SMI_CSV = "1024\n2048\n512\n"


class TestQueryGpuMemoryUsed:
    def test_amd_parses_rocm_smi_json(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            assert cmd == ["rocm-smi", "--showmeminfo", "vram", "--json"]
            return _completed(ROCM_SMI_JSON)

        monkeypatch.setattr(docker_runner.subprocess, "run", fake_run)
        # max card is 1073741824 B = 1024 MiB
        assert _query_gpu_memory_used_mib("amd") == 1024

    def test_nvidia_parses_csv(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            assert cmd[0] == "nvidia-smi"
            return _completed(NVIDIA_SMI_CSV)

        monkeypatch.setattr(docker_runner.subprocess, "run", fake_run)
        assert _query_gpu_memory_used_mib("nvidia") == 2048

    def test_command_failure_returns_none(self, monkeypatch):
        monkeypatch.setattr(docker_runner.subprocess, "run",
                            lambda cmd, **kwargs: _completed("", returncode=1))
        assert _query_gpu_memory_used_mib("amd") is None
        assert _query_gpu_memory_used_mib("nvidia") is None

    def test_malformed_output_returns_none(self, monkeypatch):
        monkeypatch.setattr(docker_runner.subprocess, "run",
                            lambda cmd, **kwargs: _completed("not json"))
        assert _query_gpu_memory_used_mib("amd") is None


class TestWaitForGpuMemoryRelease:
    def test_amd_below_threshold(self, monkeypatch):
        monkeypatch.setattr(docker_runner, "_gpu_vendor", lambda: "amd")
        monkeypatch.setattr(docker_runner.subprocess, "run",
                            lambda cmd, **kwargs: _completed(ROCM_SMI_JSON))
        monkeypatch.setattr(docker_runner.time, "sleep", lambda s: None)
        # 1024 MiB <= 2000 MiB threshold
        assert wait_for_gpu_memory_release() is True

    def test_amd_above_threshold_times_out(self, monkeypatch):
        monkeypatch.setattr(docker_runner, "_gpu_vendor", lambda: "amd")
        monkeypatch.setattr(docker_runner.subprocess, "run",
                            lambda cmd, **kwargs: _completed(ROCM_SMI_JSON))
        monkeypatch.setattr(docker_runner.time, "sleep", lambda s: None)
        # 1024 MiB > 512 MiB threshold → polls once, then times out
        assert wait_for_gpu_memory_release(max_wait=1, threshold_mib=512) is False

    def test_unknown_vendor_skips_wait(self, monkeypatch):
        monkeypatch.setattr(docker_runner, "_gpu_vendor", lambda: "unknown")

        def boom(*a, **kw):
            raise AssertionError("should not poll on unknown vendor")

        monkeypatch.setattr(docker_runner.time, "sleep", boom)
        assert wait_for_gpu_memory_release() is False
