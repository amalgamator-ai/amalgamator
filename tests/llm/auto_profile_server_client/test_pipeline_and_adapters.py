"""Tests for the vLLM adapter's --profiler-config flags and the docker-mode
config round-trip."""

import json
import shlex

from llm.auto_profile_server_client.frameworks.vllm import VllmAdapter
from llm.auto_profile_server_client.pipeline import _serialize_config
from llm.auto_profile_server_client.run_config import load_config


class TestProfilerConfigFlags:
    def test_torch_produces_single_json_flag(self):
        flags = VllmAdapter().profiling_server_extra_flags(500, 50, "torch")
        parts = shlex.split(flags)
        assert parts[0] == "--profiler-config"
        assert len(parts) == 2
        config = json.loads(parts[1])
        assert config == {
            "profiler": "torch",
            "torch_profiler_dir": "/tmp/vllm_traces",
            "delay_iterations": 500,
            "max_iterations": 50,
            "ignore_frontend": True,
        }

    def test_nsys_uses_cuda_backend_without_torch_dir(self):
        flags = VllmAdapter().profiling_server_extra_flags(500, 50, "nsys")
        parts = shlex.split(flags)
        assert parts[0] == "--profiler-config"
        config = json.loads(parts[1])
        assert config["profiler"] == "cuda"
        # vLLM rejects torch_profiler_dir unless profiler == "torch"
        assert "torch_profiler_dir" not in config
        assert config["delay_iterations"] == 500
        assert config["max_iterations"] == 50

    def test_flags_survive_shell_roundtrip_in_server_cmd(self):
        adapter = VllmAdapter()
        prof = adapter.profiling_server_extra_flags(10, 5, "torch")
        cmd = adapter.build_server_cmd(
            model="deepseek-ai/DeepSeek-V4.1-Flash", tp=4, port=8123,
            extra_flags=prof, extra_env={"VLLM_ROCM_USE_AITER": "1"})
        # The server cmd runs via bash -c; the JSON blob must stay one token
        parsed = shlex.split(cmd)
        cfg_idx = parsed.index("--profiler-config")
        assert json.loads(parsed[cfg_idx + 1])["profiler"] == "torch"
        assert cmd.startswith("VLLM_ROCM_USE_AITER=1 vllm serve")


class TestSerializeConfigRoundTrip:
    CONFIG = {
        "profiles": [
            {
                "model": "deepseek-ai/DeepSeek-V4.1-Flash",
                "num_gpus": 4,
                "exec_mode": "pure_decode",
                "frameworks": [
                    {
                        "name": "vllm",
                        "enable_trace": True,
                        "profiler_type": "torch",
                        "server_args": {
                            "extra_flags": "--max-model-len 65536",
                            "extra_env": {"VLLM_ROCM_USE_AITER": "1"},
                        },
                        "client_args": {"extra_flags": ""},
                    },
                ],
            },
        ],
        "concurrencies": [1],
        "docker_images": {"vllm": "vllm/vllm-openai-rocm:v0.31.0"},
        "exec_modes": {
            "pure_decode": {"input_len": 4, "output_len": 1024},
            "kv_fill": {"input_len": 512, "kv_cache_fill": 30000},
        },
        "output_dir": "results/campaign",
    }

    def test_round_trip_preserves_values(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(self.CONFIG))
        config = load_config(config_path)

        serialized = _serialize_config(config)
        round_path = tmp_path / "round.json"
        round_path.write_text(json.dumps(serialized))
        round = load_config(round_path)

        assert round.profiles == config.profiles
        assert round.concurrencies == config.concurrencies
        assert round.docker_images == config.docker_images
        assert round.exec_modes == config.exec_modes
        assert round.output_dir == config.output_dir

    def test_zero_kv_cache_fill_not_serialized(self, tmp_path):
        # A zero kv_cache_fill key would fail load_config validation
        # (must be > input_len), so it must be dropped on re-serialize.
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(self.CONFIG))
        config = load_config(config_path)

        raw = _serialize_config(config)
        assert "kv_cache_fill" not in raw["profiles"][0]
        assert raw["profiles"][0]["exec_mode"] == "pure_decode"
