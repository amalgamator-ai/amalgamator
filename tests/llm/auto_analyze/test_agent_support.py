import argparse
import json

from llm.auto_analyze.agent_support import add_agent_arguments, make_agent_config
from llm.auto_analyze.configs.cross_trace_config import CrossTraceConfig
from llm.auto_analyze.configs.single_trace_config import SingleTraceConfig
from llm.agents import agent_run as shared_agent_run
from llm.auto_code_gen.agents import agent_run as codegen_agent_run


def _single_trace_data(tmp_path):
    return {
        "framework_name": "vllm",
        "model": "example/model",
        "gpu_type": "B200",
        "batch_size_range": "1",
        "prefill_size_range": "4",
        "output_size_range": "2048",
        "trace_file": str(tmp_path / "trace.json"),
        "framework_source_code": str(tmp_path),
        "output_dir": str(tmp_path / "output"),
    }


def test_single_trace_embedded_qwen_agent(tmp_path):
    path = tmp_path / "single.json"
    data = _single_trace_data(tmp_path)
    data.update({
        "thinking-mode": "normal",
        "agent": {
            "backend": "litellm",
            "model": "openai/example-qwen",
            "api_base": "https://model.example/v1",
            "api_key": "dummy",
            "billing": "self_hosted",
        },
    })
    path.write_text(json.dumps(data))

    config = SingleTraceConfig.from_json(path)
    agent = make_agent_config(config, config.output_dir)

    assert config.thinking_mode == "normal"
    assert config.config_json_path == str(path.resolve())
    assert agent.backend == "litellm"
    assert agent.model == "openai/example-qwen"
    assert agent.api_base == "https://model.example/v1"
    assert agent.cwd == str((tmp_path / "output").resolve())


def test_cross_trace_external_agent_overrides_embedded(tmp_path):
    config_path = tmp_path / "cross.json"
    config_path.write_text(json.dumps({
        "traces": [],
        "target_trace_id": 0,
        "output_dir": str(tmp_path / "output"),
        "agent": {"model": "openai/embedded"},
    }))
    agent_path = tmp_path / "agent.json"
    agent_path.write_text(json.dumps({
        "backend": "litellm",
        "model": "openai/external-qwen",
        "api_base": "https://external.example/v1",
        "api_key": "dummy",
        "billing": "self_hosted",
    }))

    config = CrossTraceConfig.from_json(config_path)
    agent = make_agent_config(config, config.output_dir, str(agent_path))

    assert agent.model == "openai/external-qwen"
    assert agent.api_base == "https://external.example/v1"


def test_legacy_config_defaults_to_claude_sdk(tmp_path):
    config = SingleTraceConfig(**_single_trace_data(tmp_path))
    agent = make_agent_config(config, config.output_dir)

    assert agent.backend == "claude_sdk"
    assert agent.model.startswith("claude-")


def test_legacy_cli_flag_aliases_agent_config():
    parser = argparse.ArgumentParser()
    add_agent_arguments(parser)

    assert parser.parse_args(["--agent-config", "new.json"]).agent_config == "new.json"
    assert parser.parse_args(["--claude-config", "old.json"]).agent_config == "old.json"


def test_external_agent_config_resolves(tmp_path):
    agent_path = tmp_path / "agent.json"
    agent_path.write_text(json.dumps({
        "backend": "litellm",
        "model": "openai/example-model",
        "api_base": "https://model.example/v1",
        "api_key": "dummy",
        "billing": "self_hosted",
    }))

    config = SingleTraceConfig(**_single_trace_data(tmp_path))
    agent = make_agent_config(config, config.output_dir, str(agent_path))

    assert agent.backend == "litellm"
    assert agent.model == "openai/example-model"
    assert agent.billing == "self_hosted"


def test_analysis_and_codegen_share_the_same_agent_runtime():
    assert shared_agent_run is codegen_agent_run
