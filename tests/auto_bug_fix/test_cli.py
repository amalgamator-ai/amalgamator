import json
from unittest.mock import Mock

import pytest

from auto_bug_fix import run_bug_fix


@pytest.mark.parametrize(
    "argv",
    [
        ["pipeline.yaml", "--models", "custom.yaml"],
        ["--models", "custom.yaml", "pipeline.yaml"],
    ],
)
def test_models_argument_forwarded(argv, monkeypatch):
    class StopAtLoader(Exception):
        pass

    loader = Mock(side_effect=StopAtLoader)
    monkeypatch.setattr(run_bug_fix, "load_multi_target_config", loader)
    with pytest.raises(StopAtLoader):
        run_bug_fix.main(argv)
    loader.assert_called_once_with("pipeline.yaml", models_path="custom.yaml")


@pytest.mark.parametrize(
    "argv,code", [([], 2), (["--help"], 0), (["p.yaml", "--unknown"], 2)]
)
def test_parser_usage_before_loading(argv, code, monkeypatch, capsys):
    loader = Mock()
    monkeypatch.setattr(run_bug_fix, "load_multi_target_config", loader)
    with pytest.raises(SystemExit) as exc:
        run_bug_fix.main(argv)
    assert exc.value.code == code
    loader.assert_not_called()
    out = capsys.readouterr()
    assert "usage:" in out.out + out.err


@pytest.mark.parametrize("invalid", [False, True])
def test_explicit_registry_error_precedes_pipeline_side_effects(
    tmp_path, monkeypatch, invalid
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "pipeline.json").write_text(
        json.dumps(
            {
                "issue_id": "cli",
                "bug_description": "test",
                "repository": {"source_path": str(tmp_path)},
                "branches": {"source": "main", "target": "old"},
                "fix": {"commit": "abc"},
                "build": {},
            }
        )
    )
    (tmp_path / "models.yaml").write_text(
        "models:\n  a:\n    model: gpt-4o\ndefault: a\n"
    )
    if invalid:
        (tmp_path / "explicit.yaml").write_text("models: []")
    git = Mock()
    pipeline = Mock()
    monkeypatch.setattr(run_bug_fix.subprocess, "run", git)
    monkeypatch.setattr(run_bug_fix, "run_targets", pipeline)
    with pytest.raises(SystemExit) as exc:
        run_bug_fix.main(["pipeline.json", "--models", "explicit.yaml"])
    assert exc.value.code == 2
    git.assert_not_called()
    pipeline.assert_not_called()
    assert not (tmp_path / "runs").exists()


def test_strict_profile_typo_fails_before_pipeline_side_effects(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pipeline.json").write_text(
        json.dumps(
            {
                "issue_id": "cli",
                "bug_description": "test",
                "repository": {"source_path": str(tmp_path)},
                "branches": {"source": "main", "target": "old"},
                "fix": {"commit": "abc"},
                "build": {},
                "model": "snnet",
            }
        )
    )
    (tmp_path / "models.yaml").write_text(
        "models:\n  sonnet:\n    model: claude-sonnet-4-6\ndefault: sonnet\nstrict: true\n"
    )
    git = Mock()
    monkeypatch.setattr(run_bug_fix.subprocess, "run", git)
    with pytest.raises(SystemExit) as exc:
        run_bug_fix.main(["pipeline.json"])
    assert exc.value.code == 2
    assert "Unknown model profile 'snnet'" in capsys.readouterr().err
    git.assert_not_called()
    assert not (tmp_path / "runs").exists()
