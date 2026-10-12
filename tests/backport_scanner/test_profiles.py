"""Scanner profiles must survive completion calls and generated backport cases."""

import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from auto_bug_fix.config_loader import load_pipeline_config
from backport_scanner import generate_config, run_scanner, scanner
from backport_scanner.scanner import ScanConfig, ScanReport
from backport_scanner.triage import BranchCheckResult, FixStatus
from common import llm_client
from common.model_registry import ModelProfile, RegistryError


def write_registry(directory, *, model="openai/review", strict=True):
    directory.mkdir(parents=True, exist_ok=True)
    key_file = directory / "model.key"
    key_file.write_text("  FAKE_PROFILE_KEY\n")
    path = directory / "models.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "review": {
                        "provider": "litellm",
                        "model": model,
                        "api_base": "https://profile.invalid/v1",
                        "api_key_file": str(key_file),
                        "params": {
                            "temperature": 0.2,
                            "extra_body": {
                                "chat_template_kwargs": {"enable_thinking": False}
                            },
                        },
                    }
                },
                "default": "review",
                "strict": strict,
            }
        )
    )
    return path


def write_scan(directory, *, model=None, discover_commands=False):
    repo = directory / "repo"
    repo.mkdir(exist_ok=True)
    path = directory / "scan.yaml"
    llm = {"discover_commands": discover_commands}
    if model is not None:
        llm["model"] = model
    path.write_text(
        yaml.safe_dump(
            {
                "repository": {"local_path": str(repo)},
                "fix": {"commit": "abc123", "description": "Fix bug"},
                "scan": {"branches": ["release"]},
                "llm": llm,
            }
        )
    )
    return path


def response(text):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))], usage=None
    )


def result(status):
    return BranchCheckResult("release", status, "test", "test result", ["file.c"])


@pytest.mark.parametrize("selection", [None, "review"])
def test_named_profile_reaches_both_scanner_completion_paths(
    tmp_path, monkeypatch, selection
):
    registry = write_registry(tmp_path)
    scan = write_scan(tmp_path, model=selection)
    monkeypatch.setattr(llm_client, "_api_base", "http://stale/v1")
    monkeypatch.setattr(llm_client, "_api_key", "STALE_KEY")
    monkeypatch.setattr(llm_client, "_model_params", {"temperature": 0.9})
    config = run_scanner.load_config(scan)
    assert config.model_ref == "review"
    assert config.models_path == registry.resolve()
    assert config.llm_model == "openai/review"
    assert llm_client.get_api_key() == "STALE_KEY"  # Loading did not seed globals.

    request = Mock(
        side_effect=[
            response("NEEDS_FIX: fix is absent"),
            response("BUILD: make\nTEST: make test"),
        ]
    )
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=request))
    monkeypatch.setattr(
        scanner, "check_ref", Mock(return_value=result(FixStatus.AMBIGUOUS))
    )
    monkeypatch.setattr(scanner, "_run", Mock(return_value=(0, "+fixed code\n", "")))
    assert len(scanner.run_scan(config).needs_fix) == 1
    (config.repo_path / "README.md").write_text("Build with make, test with make test.")
    assert generate_config._discover_commands(
        config.repo_path, config.llm_model, profile=config.profile
    ) == ("make", "make test")
    assert request.call_count == 2
    for call in request.call_args_list:
        sent = call.kwargs
        assert sent["model"] == config.profile.model
        assert sent["api_base"] == config.profile.api_base
        assert sent["api_key"] == "FAKE_PROFILE_KEY"
        assert sent["temperature"] == 0.2
        assert sent["extra_body"] == config.profile.params["extra_body"]


def test_explicit_registry_overrides_discovery(tmp_path):
    write_registry(tmp_path, model="openai/sibling")
    explicit = write_registry(tmp_path / "shared models", model="openai/explicit")
    config = run_scanner.load_config(write_scan(tmp_path), models_path=explicit)
    assert config.llm_model == "openai/explicit"
    assert config.models_path == explicit.resolve()


def test_home_registry_default_is_discovered(tmp_path):
    registry = write_registry(tmp_path / ".config" / "amalgamator")
    config = run_scanner.load_config(write_scan(tmp_path))
    assert config.model_ref == "review"
    assert config.profile.api_key == "FAKE_PROFILE_KEY"
    assert config.models_path == registry.resolve()


def test_bare_model_preserves_legacy_transport_with_unrelated_registry(
    tmp_path, monkeypatch, caplog
):
    scan = write_scan(tmp_path, model="openai/bare")
    monkeypatch.setattr(llm_client, "_api_base", "http://legacy/v1")
    monkeypatch.setattr(llm_client, "_api_key", "LEGACY_KEY")
    request = Mock(return_value=response("done"))
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=request))
    before = run_scanner.load_config(scan)
    write_registry(tmp_path, strict=False)
    after = run_scanner.load_config(scan)
    for config in (before, after):
        assert config.profile is None
        llm_client.completion(config.llm_model, [], profile=config.profile)
        assert request.call_args.kwargs["api_base"] == "http://legacy/v1"
        assert request.call_args.kwargs["api_key"] == "LEGACY_KEY"
    assert "not a registered profile id" in caplog.text


def test_without_registry_retains_scanner_default(tmp_path):
    config = run_scanner.load_config(write_scan(tmp_path))
    assert config.profile is None
    assert config.llm_model == "claude-opus-4-6"


@pytest.mark.parametrize("selection", [None, "review"])
def test_generated_cases_round_trip_with_the_selected_registry(
    tmp_path, monkeypatch, capsys, selection
):
    registry = write_registry(tmp_path / "shared models")
    config = run_scanner.load_config(
        write_scan(tmp_path, model=selection, discover_commands=True),
        models_path=registry,
    )
    monkeypatch.setattr(
        generate_config,
        "run_scan",
        Mock(
            return_value=ScanReport(
                str(config.repo_path),
                config.fix_commit,
                config.fix_description,
                [result(FixStatus.NEEDS_FIX)],
            )
        ),
    )
    monkeypatch.setattr(
        generate_config, "git_show_files", Mock(return_value=["file.c"])
    )
    monkeypatch.setattr(
        generate_config, "_get_default_branch", Mock(return_value="main")
    )
    (config.repo_path / "README.md").write_text("Build/test instructions")
    request = Mock(return_value=response("BUILD: make\nTEST: make test"))
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=request))
    output = tmp_path / "generated cases"
    generate_config.mode_generate(config, output)
    request.assert_called_once()
    assert request.call_args.kwargs["api_key"] == "FAKE_PROFILE_KEY"

    case = next(output.glob("*.yaml"))
    text = case.read_text()
    data = yaml.safe_load(text)
    assert data["model"] == "review"
    assert data["build"] == {"command": "make", "test_command": "make test"}
    assert "FAKE_PROFILE_KEY" not in text and "profile.invalid" not in text
    assert "api_key" not in text and "params" not in data
    command = next(
        line.removeprefix("# ")
        for line in text.splitlines()
        if line.startswith("# python ")
    )
    assert command in capsys.readouterr().out
    args = shlex.split(command)
    assert args[:3] == ["python", "-m", "auto_bug_fix.run_bug_fix"]
    assert "--config" not in args
    bug, agent, *_ = load_pipeline_config(
        args[3], models_path=args[args.index("--models") + 1]
    )
    assert agent.profile == config.profile
    assert bug.phase_4b_profile is agent.profile


def test_generated_legacy_case_uses_the_scanner_model(tmp_path, monkeypatch):
    config = run_scanner.load_config(write_scan(tmp_path, model="gpt-4o"))
    monkeypatch.setattr(
        generate_config,
        "run_scan",
        Mock(
            return_value=ScanReport(
                str(config.repo_path),
                config.fix_commit,
                "test",
                [result(FixStatus.NEEDS_FIX)],
            )
        ),
    )
    monkeypatch.setattr(
        generate_config, "git_show_files", Mock(return_value=["file.c"])
    )
    monkeypatch.setattr(
        generate_config, "_get_default_branch", Mock(return_value="main")
    )
    output = tmp_path / "generated"
    generate_config.mode_generate(config, output)
    case = next(output.glob("*.yaml"))
    assert yaml.safe_load(case.read_text())["model"] == "gpt-4o"
    assert "--models" not in case.read_text()


def test_generation_requires_registry_provenance_for_programmatic_profile(
    tmp_path, monkeypatch
):
    config = ScanConfig(
        tmp_path,
        "abc",
        profile=ModelProfile("local", "litellm", "openai/local", api_key="KEY"),
    )
    scan = Mock()
    monkeypatch.setattr(generate_config, "run_scan", scan)
    with pytest.raises(RegistryError, match="requires models_path"):
        generate_config.mode_generate(config, tmp_path / "out")
    scan.assert_not_called()


@pytest.mark.parametrize("entrypoint", [run_scanner, generate_config])
def test_cli_forwards_models_path(entrypoint, monkeypatch):
    class StopAtLoader(Exception):
        pass

    loader = Mock(side_effect=StopAtLoader)
    monkeypatch.setattr(entrypoint, "load_config", loader)
    argv = ["--config", "scan.yaml", "--models", "shared.yaml"]
    if entrypoint is generate_config:
        argv += ["--mode", "generate"]
    with pytest.raises(StopAtLoader):
        entrypoint.main(argv)
    loader.assert_called_once_with(Path("scan.yaml"), models_path=Path("shared.yaml"))


@pytest.mark.parametrize("entrypoint", [run_scanner, generate_config])
@pytest.mark.parametrize("error", ["strict", "missing", "invalid"])
def test_registry_errors_precede_scanning(tmp_path, monkeypatch, entrypoint, error):
    path = write_scan(tmp_path, model="typo")
    registry = tmp_path / "models.yaml"
    if error == "strict":
        write_registry(tmp_path)
    elif error == "invalid":
        registry.write_text("models: []")
    scan = Mock()
    monkeypatch.setattr(entrypoint, "run_scan", scan)
    argv = ["--config", str(path), "--models", str(registry)]
    if entrypoint is generate_config:
        argv += ["--mode", "generate", "--output-dir", str(tmp_path / "out")]
    with pytest.raises(SystemExit) as exc:
        entrypoint.main(argv)
    assert exc.value.code == 2
    scan.assert_not_called()
    assert not (tmp_path / "out").exists()
