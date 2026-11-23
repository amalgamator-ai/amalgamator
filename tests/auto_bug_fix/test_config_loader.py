# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for auto_bug_fix.config_loader — YAML/JSON config parsing."""

import os
import tempfile
from typing import Any

import pytest

from auto_bug_fix.bug_fix_config import BugFixConfig
from auto_bug_fix.config_loader import (
    ConfigError,
    _sanitize_ref,
    expand_path,
    load_config_file,
    load_multi_target_config,
    parse_bug_fix_config,
    parse_claude_config,
    parse_targets,
    validate_config,
)
from auto_bug_fix.run_bug_fix import (
    PipelineState,
    _describe_triage_effort,
    _ensure_series_clusters,
    _phase_config,
    _run_fused_pre_red_phase,
    _triage_phase_config,
)
from common.model_registry import ModelProfile


def test_load_yaml_config():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
issue_id: CVE-2024-1234
bug_description: Test CVE
repository:
  source_path: ~/test-repo
  workdir: /tmp/workdir
  build_subdir: src
branches:
  source: main
  target: v1.0
fix:
  commit: abc123
build:
  command: make
  test_command: make test
""")
        f.flush()
        config = load_config_file(f.name)

    assert config["issue_id"] == "CVE-2024-1234"
    assert config["repository"]["source_path"] == "~/test-repo"
    assert config["branches"]["source"] == "main"
    os.unlink(f.name)


def test_load_json_config():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("""{
  "issue_id": "CVE-2024-5678",
  "bug_description": "Test CVE",
  "repository": {
    "source_path": "~/test-repo"
  },
  "branches": {
    "source": "main",
    "target": "v1.0"
  },
  "fix": {
    "commit": "def456"
  },
  "build": {
    "command": "make",
    "test_command": "make test"
  }
}""")
        f.flush()
        config = load_config_file(f.name)

    assert config["issue_id"] == "CVE-2024-5678"
    assert config["fix"]["commit"] == "def456"
    os.unlink(f.name)


def test_load_config_file_not_found():
    with pytest.raises(ConfigError, match="not found"):
        load_config_file("/nonexistent/config.yaml")


def test_load_config_invalid_extension():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("issue_id: test")
        f.flush()
        with pytest.raises(ConfigError, match="Unsupported config format"):
            load_config_file(f.name)
    os.unlink(f.name)


def test_load_config_empty_yaml():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("")
        f.flush()
        with pytest.raises(ConfigError, match="must contain a YAML/JSON mapping"):
            load_config_file(f.name)
    os.unlink(f.name)


def test_load_config_non_mapping_yaml():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("- item1\n- item2\n")
        f.flush()
        with pytest.raises(ConfigError, match="must contain a YAML/JSON mapping"):
            load_config_file(f.name)
    os.unlink(f.name)


def test_validate_config_missing_fields():
    config = {
        "issue_id": "CVE-2024-1234",
    }
    with pytest.raises(ConfigError, match="Missing required fields"):
        validate_config(config)


def test_validate_config_missing_nested():
    config = {
        "issue_id": "CVE-2024-1234",
        "bug_description": "Test",
        "repository": {},
        "branches": {"source": "main", "target": "v1.0"},
        "fix": {"commit": "abc"},
        "build": {"command": "make", "test_command": "make test"},
    }
    with pytest.raises(ConfigError, match="repository.source_path is required"):
        validate_config(config)


def _fix_cfg(fix, branches=None):
    """Minimal valid config with the given fix section (and optional branches override)."""
    return {
        "issue_id": "CVE-2026-1234",
        "bug_description": "Test",
        "repository": {"source_path": "/tmp/test-repo", "workdir": "/tmp/workdir"},
        "branches": (
            {"source": "main", "target": "v1.0"} if branches is None else branches
        ),
        "fix": fix,
        "build": {},
    }


def _write_patch(directory) -> str:
    patch_file = directory / "fix.patch"
    patch_file.write_text("diff content")
    return str(patch_file)


def test_validate_config_fix_commit_xor_patch_both(tmp_path):
    config = _fix_cfg({"commit": "abc123", "patch": _write_patch(tmp_path)})
    with pytest.raises(ConfigError, match="mutually exclusive"):
        validate_config(config)


def test_validate_config_fix_neither():
    config = _fix_cfg({})
    with pytest.raises(ConfigError, match="fix.commit or fix.patch is required"):
        validate_config(config)


def test_validate_config_fix_commits_requires_commit():
    config = _fix_cfg({"commits": ["abc123"]})
    with pytest.raises(ConfigError, match="fix.commit is required with fix.commits"):
        validate_config(config)


def test_validate_config_fix_commits_xor_patch(tmp_path):
    config = _fix_cfg(
        {"commit": "abc123", "commits": ["abc123"], "patch": _write_patch(tmp_path)}
    )
    with pytest.raises(ConfigError, match="mutually exclusive"):
        validate_config(config)


def test_validate_config_fix_commits_must_be_non_empty_list():
    config = _fix_cfg({"commit": "abc123", "commits": []})
    with pytest.raises(ConfigError, match="non-empty list"):
        validate_config(config)


def test_parse_bug_fix_config_fix_commits():
    config = _fix_cfg(
        {"commit": "merge123", "commits": ["abc123", "def456"]},
        branches={"source": "origin/main", "target": "v1.0"},
    )
    parsed, _, _ = parse_bug_fix_config(config)
    assert parsed.source_fix_commit == "merge123"
    assert parsed.source_fix_commits == ["abc123", "def456"]


# ---------- advanced.triage_effort ----------


def _triage_cfg(advanced=None, fix=None):
    config = _fix_cfg(fix or {"commit": "abc123"})
    if advanced is not None:
        config["advanced"] = advanced
    return config


def test_triage_effort_defaults_to_low_for_series():
    parsed, _, _ = parse_bug_fix_config(
        _triage_cfg(fix={"commit": "merge123", "commits": ["abc123", "def456"]})
    )
    assert parsed.triage_effort == "low"


def test_triage_effort_defaults_to_low_for_single_commit():
    parsed, _, _ = parse_bug_fix_config(_triage_cfg())
    assert parsed.triage_effort == "low"


def test_triage_effort_inherit_clears_single_commit_default():
    parsed, _, _ = parse_bug_fix_config(_triage_cfg({"triage_effort": "inherit"}))
    assert parsed.triage_effort == ""


def test_triage_effort_override():
    parsed, _, _ = parse_bug_fix_config(
        _triage_cfg(
            {"triage_effort": "high"},
            fix={"commit": "merge123", "commits": ["abc123"]},
        )
    )
    assert parsed.triage_effort == "high"


def test_triage_effort_inherit_clears_series_default():
    parsed, _, _ = parse_bug_fix_config(
        _triage_cfg(
            {"triage_effort": "inherit"},
            fix={"commit": "merge123", "commits": ["abc123"]},
        )
    )
    assert parsed.triage_effort == ""


def test_triage_effort_rejects_unknown_level():
    with pytest.raises(ConfigError, match="triage_effort"):
        parse_bug_fix_config(_triage_cfg({"triage_effort": "turbo"}))


def test_triage_effort_rejects_non_string_value():
    with pytest.raises(ConfigError, match="triage_effort"):
        parse_bug_fix_config(_triage_cfg({"triage_effort": True}))


def test_triage_effort_rejects_blank_value():
    with pytest.raises(ConfigError, match="triage_effort"):
        parse_bug_fix_config(_triage_cfg({"triage_effort": "   "}))


def test_validate_config_patch_mode_defaults_source(tmp_path):
    config = _fix_cfg({"patch": _write_patch(tmp_path)}, branches={"target": "v1.0"})
    validate_config(config)
    assert config["branches"]["source"] == "origin/main"


def test_validate_config_commit_mode_requires_source():
    config = _fix_cfg({"commit": "abc123"}, branches={"target": "v1.0"})
    with pytest.raises(ConfigError, match="branches.source is required"):
        validate_config(config)


def test_validate_config_patch_not_found():
    config = _fix_cfg({"patch": "/nonexistent/fix.patch"}, branches={"target": "v1.0"})
    with pytest.raises(ConfigError, match="not found"):
        validate_config(config)


def test_expand_path():
    os.environ["TEST_VAR"] = "/test/path"
    assert expand_path("~/foo") == os.path.expanduser("~/foo")
    assert expand_path("$TEST_VAR/bar") == "/test/path/bar"


def test_parse_bug_fix_config():
    config = {
        "issue_id": "CVE-2024-1234",
        "bug_description": "Test CVE: buffer overflow",
        "repository": {
            "source_path": "/tmp/test-repo",
            "workdir": "/tmp/workdir",
            "build_subdir": "src",
        },
        "branches": {
            "source": "main",
            "target": "v1.0",
        },
        "fix": {
            "commit": "abc123def",
        },
        "build": {
            "command": "make -j4",
            "test_command": "make test",
        },
        "advanced": {
            "max_build_test_retries": 5,
        },
    }

    bug_fix_config, workdir, source_path = parse_bug_fix_config(config)

    assert bug_fix_config.issue_id == "CVE-2024-1234"
    assert bug_fix_config.bug_description == "Test CVE: buffer overflow"
    assert bug_fix_config.source_branch == "main"
    assert bug_fix_config.target_branch == "v1.0"
    assert bug_fix_config.source_fix_commit == "abc123def"
    assert bug_fix_config.build_command == "make -j4"
    assert bug_fix_config.test_command == "make test"
    assert bug_fix_config.max_build_test_retries == 5
    assert workdir == "/tmp/workdir"
    assert source_path == "/tmp/test-repo"
    assert bug_fix_config.repo_path == "/tmp/workdir/test-repo"
    assert bug_fix_config.build_dir == "/tmp/workdir/test-repo/src"


def test_parse_claude_config():
    config = {
        "claude": {
            "model": "claude-opus-4-7",
            "allowed_tools": ["Read", "Bash"],
            "perm_mode": "interactive",
        },
    }

    claude_config = parse_claude_config(config, "/tmp/output")

    assert claude_config.model == "claude-opus-4-7"
    assert claude_config.allowed_tools == ["Read", "Bash"]
    assert claude_config.perm_mode == "interactive"
    assert claude_config.cwd == "/tmp/output"


def test_parse_claude_config_defaults():
    config = {}

    claude_config = parse_claude_config(config, "/tmp/output")

    assert claude_config.model == "claude-sonnet-4-6"
    assert claude_config.allowed_tools == ["Read", "Write", "Bash"]
    assert claude_config.perm_mode == "acceptEdits"
    assert claude_config.xml_tool_fallback is False


def test_parse_claude_config_xml_tool_fallback_opt_in():
    config = {"claude": {"xml_tool_fallback": True}}
    claude_config = parse_claude_config(config, "/tmp/output")
    assert claude_config.xml_tool_fallback is True


def test_parse_claude_config_xml_tool_fallback_must_be_bool():
    config = {"claude": {"xml_tool_fallback": "yes"}}
    with pytest.raises(ConfigError):
        parse_claude_config(config, "/tmp/output")


def test_parse_claude_config_effort_and_thinking():
    config = {
        "claude": {
            "effort": "high",
            "thinking": {"type": "adaptive"},
        },
    }
    claude_config = parse_claude_config(config, "/tmp/output")
    assert claude_config.effort == "high"
    assert claude_config.thinking == {"type": "adaptive"}


def test_phase_config_inherits_thinking_and_effort():
    config = {
        "claude": {
            "effort": "high",
            "thinking": {"type": "enabled", "budget_tokens": 32000},
        },
    }
    claude_config = parse_claude_config(config, "/tmp/output")
    bug_fix_config = BugFixConfig(
        repo_path="/tmp/repo",
        build_dir="/tmp/repo",
        source_branch="origin/main",
        target_branch="abc123",
        source_fix_commit="abc123",
        bug_description="test",
        issue_id="TEST-1",
    )

    phase_config = _phase_config(claude_config, bug_fix_config)

    assert phase_config.thinking == {"type": "enabled", "budget_tokens": 32000}
    assert phase_config.effort == "high"


def test_phase_config_thinking_override_wins():
    config = {
        "claude": {
            "effort": "high",
            "thinking": {"type": "enabled", "budget_tokens": 32000},
        },
    }
    claude_config = parse_claude_config(config, "/tmp/output")
    bug_fix_config = BugFixConfig(
        repo_path="/tmp/repo",
        build_dir="/tmp/repo",
        source_branch="origin/main",
        target_branch="abc123",
        source_fix_commit="abc123",
        bug_description="test",
        issue_id="TEST-1",
    )

    phase_config = _phase_config(
        claude_config, bug_fix_config, thinking={"type": "disabled"}
    )

    assert phase_config.thinking == {"type": "disabled"}


# ---------- _triage_phase_config: advanced.triage_effort ----------


def _bug_fix_config(**overrides) -> BugFixConfig:
    fields = {
        "repo_path": "/tmp/repo",
        "build_dir": "/tmp/repo",
        "source_branch": "origin/main",
        "target_branch": "abc123",
        "source_fix_commit": "abc123",
        "bug_description": "test",
        "issue_id": "TEST-1",
    }
    fields.update(overrides)
    return BugFixConfig(**fields)


def test_triage_phase_config_applies_effort():
    claude_config = parse_claude_config({"claude": {"effort": "max"}}, "/tmp/output")
    phase_config = _triage_phase_config(
        claude_config, _bug_fix_config(triage_effort="low")
    )
    assert phase_config.effort == "low"
    assert phase_config.thinking == claude_config.thinking


def test_triage_phase_config_inherits_when_empty():
    claude_config = parse_claude_config({"claude": {"effort": "high"}}, "/tmp/output")
    phase_config = _triage_phase_config(claude_config, _bug_fix_config())
    assert phase_config.effort == "high"


def test_triage_phase_config_overrides_litellm_reasoning_effort():
    profile = ModelProfile(
        id="m1",
        provider="litellm",
        model="openai/deepseek",
        params={"reasoning_effort": "max", "temperature": 0.2},
    )
    claude_config = parse_claude_config({"claude": {"effort": "max"}}, "/tmp/output")
    claude_config.profile = profile

    phase_config = _triage_phase_config(
        claude_config, _bug_fix_config(triage_effort="low")
    )

    assert phase_config.profile is not None
    assert phase_config.profile is not profile
    assert phase_config.profile.params["reasoning_effort"] == "low"
    assert phase_config.profile.params["temperature"] == 0.2
    # The run's profile must stay untouched for every other phase.
    assert profile.params["reasoning_effort"] == "max"


def test_triage_phase_config_leaves_unsupported_litellm_profile_alone():
    profile = ModelProfile(
        id="m1", provider="litellm", model="openai/plain", params={"temperature": 0.2}
    )
    claude_config = parse_claude_config({"claude": {"effort": "max"}}, "/tmp/output")
    claude_config.profile = profile

    phase_config = _triage_phase_config(
        claude_config, _bug_fix_config(triage_effort="low")
    )

    assert phase_config.profile is profile
    assert "reasoning_effort" not in phase_config.profile.params


def test_triage_phase_config_leaves_anthropic_profile_alone():
    """The SDK path takes effort separately; never rewrite anthropic params."""
    profile = ModelProfile(
        id="m1",
        provider="anthropic",
        model="claude-opus-4-6",
        params={"temperature": 0.2},
    )
    claude_config = parse_claude_config({"claude": {"effort": "max"}}, "/tmp/output")
    claude_config.profile = profile

    phase_config = _triage_phase_config(
        claude_config, _bug_fix_config(triage_effort="low")
    )

    assert phase_config.profile is profile
    assert phase_config.effort == "low"
    assert "reasoning_effort" not in phase_config.profile.params


def test_fused_pre_red_phase_runs_at_triage_effort(monkeypatch):
    """fused_pre_red replaces Phase 0.5, so it must honour triage_effort."""
    profile = ModelProfile(
        id="m1",
        provider="litellm",
        model="openai/deepseek",
        params={"reasoning_effort": "high"},
    )
    claude_config = parse_claude_config({"claude": {"effort": "high"}}, "/tmp/output")
    claude_config.profile = profile
    config = _bug_fix_config(
        triage_effort="low", fused_pre_red=True, source_fix_commits=[]
    )

    seen = {}

    def fake_run_fused_pre_red(cfg, phase_config, state, tracker):
        seen["claude_config"] = phase_config
        return {"commit_type": "bug-fix"}

    monkeypatch.setattr(
        "auto_bug_fix.run_bug_fix._run_fused_pre_red", fake_run_fused_pre_red
    )

    _run_fused_pre_red_phase(config, claude_config, PipelineState(), None)

    used = seen["claude_config"]
    assert used.profile is not None
    assert used.profile.params["reasoning_effort"] == "low"
    assert used.effort == "low"
    # The run-wide config must stay at high effort for resolution and later phases.
    assert claude_config.profile.params["reasoning_effort"] == "high"
    assert claude_config.effort == "high"


def test_fused_pre_red_phase_inherits_run_effort(monkeypatch):
    """triage_effort: inherit keeps the run effort in the fused session."""
    claude_config = parse_claude_config({"claude": {"effort": "high"}}, "/tmp/output")
    config = _bug_fix_config(
        triage_effort="", fused_pre_red=True, source_fix_commits=[]
    )

    seen = {}

    def fake_run_fused_pre_red(cfg, phase_config, state, tracker):
        seen["claude_config"] = phase_config
        return {"commit_type": "bug-fix"}

    monkeypatch.setattr(
        "auto_bug_fix.run_bug_fix._run_fused_pre_red", fake_run_fused_pre_red
    )

    _run_fused_pre_red_phase(config, claude_config, PipelineState(), None)

    assert seen["claude_config"].effort == "high"


def test_series_partition_runs_at_triage_effort(monkeypatch, tmp_path):
    """The series cluster partition is the third triage path; same effort."""
    profile = ModelProfile(
        id="m1",
        provider="litellm",
        model="openai/deepseek",
        params={"reasoning_effort": "high"},
    )
    claude_config = parse_claude_config({"claude": {"effort": "high"}}, "/tmp/output")
    claude_config.profile = profile
    config = _bug_fix_config(
        triage_effort="low", source_fix_commits=["abc123", "def456"]
    )

    class _Tracker:
        output_dir = str(tmp_path)

        def record_gate(self, name, value):
            pass

    seen = {}

    async def fake_claude_run(cfg, prompts, tracker=None):
        seen["claude_config"] = cfg
        return []

    monkeypatch.setattr("auto_bug_fix.run_bug_fix.claude_run", fake_claude_run)
    monkeypatch.setattr(
        "auto_bug_fix.run_bug_fix._series_prompt_section", lambda cfg: ""
    )

    _ensure_series_clusters(config, claude_config, PipelineState(), _Tracker())

    used = seen["claude_config"]
    # ``effort`` is the SDK option; on this LiteLLM profile the override that
    # reaches the backend is the rewritten ``reasoning_effort``.
    assert used.effort == "low"
    assert used.profile is not None
    assert used.profile.params["reasoning_effort"] == "low"
    # The run-wide config keeps the run effort for resolution.
    assert claude_config.profile.params["reasoning_effort"] == "high"


def test_describe_triage_effort_reports_applied_sdk_effort():
    claude_config = parse_claude_config({"claude": {"effort": "max"}}, "/tmp/output")
    config = _bug_fix_config(triage_effort="low")

    assert _describe_triage_effort(claude_config, config) == "low (applied)"


def test_describe_triage_effort_reports_applied_anthropic_profile():
    profile = ModelProfile(
        id="m1",
        provider="anthropic",
        model="claude-opus-4-6",
        params={"temperature": 0.2},
    )
    claude_config = parse_claude_config({"claude": {"effort": "max"}}, "/tmp/output")
    claude_config.profile = profile
    config = _bug_fix_config(triage_effort="low")

    assert _describe_triage_effort(claude_config, config) == "low (applied)"


def test_describe_triage_effort_reports_unapplied_litellm_effort():
    profile = ModelProfile(
        id="m1", provider="litellm", model="openai/plain", params={"temperature": 0.2}
    )
    claude_config = parse_claude_config({"claude": {"effort": "max"}}, "/tmp/output")
    claude_config.profile = profile
    config = _bug_fix_config(triage_effort="low")

    assert _describe_triage_effort(claude_config, config) == (
        "low (not applied: profile declares no reasoning_effort)"
    )


def test_describe_triage_effort_reports_inherited_run_effort():
    claude_config = parse_claude_config({"claude": {"effort": "high"}}, "/tmp/output")
    config = _bug_fix_config(triage_effort="")

    assert _describe_triage_effort(claude_config, config) == (
        "inherit (run effort: high)"
    )


def test_parse_claude_config_invalid_effort():
    config = {"claude": {"effort": "turbo"}}
    with pytest.raises(ConfigError):
        parse_claude_config(config, "/tmp/output")


def test_parse_claude_config_thinking_must_be_mapping():
    config = {"claude": {"thinking": "adaptive"}}
    with pytest.raises(ConfigError):
        parse_claude_config(config, "/tmp/output")


def test_parse_claude_config_top_level_model():
    config = {"model": "claude-opus-4-6"}
    claude_config = parse_claude_config(config, "/tmp/output")
    assert claude_config.model == "claude-opus-4-6"


# --- timeout_minutes -> command_timeout (seconds) ---


def _timeout_config(advanced):
    return {
        "issue_id": "CVE-2024-0000",
        "bug_description": "Test",
        "repository": {"source_path": "/tmp/test-repo", "workdir": "/tmp/workdir"},
        "branches": {"source": "main", "target": "v1.0"},
        "fix": {"commit": "abc123"},
        "build": {"command": "make", "test_command": "make test"},
        "advanced": advanced,
    }


def test_timeout_minutes_unset_means_no_limit():
    cfg, _, _ = parse_bug_fix_config(_timeout_config({}))
    assert cfg.command_timeout is None


def test_timeout_minutes_zero_means_no_limit():
    cfg, _, _ = parse_bug_fix_config(_timeout_config({"timeout_minutes": 0}))
    assert cfg.command_timeout is None


def test_timeout_minutes_positive_to_seconds():
    cfg, _, _ = parse_bug_fix_config(_timeout_config({"timeout_minutes": 5}))
    assert cfg.command_timeout == 300


def test_timeout_minutes_fractional():
    cfg, _, _ = parse_bug_fix_config(_timeout_config({"timeout_minutes": 1.5}))
    assert cfg.command_timeout == 90


def test_timeout_minutes_negative_means_no_limit():
    cfg, _, _ = parse_bug_fix_config(_timeout_config({"timeout_minutes": -3}))
    assert cfg.command_timeout is None


# --- _sanitize_ref ---


def test_sanitize_ref_strips_origin():
    assert _sanitize_ref("origin/release-1.34") == "release-1.34"


def test_sanitize_ref_replaces_slashes():
    assert _sanitize_ref("feature/deep/path") == "feature_deep_path"


def test_sanitize_ref_sha_passthrough():
    assert _sanitize_ref("cc8e8609e") == "cc8e8609e"


def test_sanitize_ref_origin_and_slashes():
    assert _sanitize_ref("origin/release/1.34") == "release_1.34"


# --- parse_targets ---


def _base_config(**overrides) -> dict[str, Any]:
    """Minimal valid config dict for parse_targets tests."""
    cfg = {
        "issue_id": "CVE-2024-1234",
        "bug_description": "Test CVE",
        "repository": {"source_path": "/tmp/test-repo"},
        "branches": {"source": "main", "target": "v1.0"},
        "fix": {"commit": "abc123"},
        "build": {"command": "make", "test_command": "make test"},
    }
    cfg.update(overrides)
    return cfg


def test_parse_targets_list_of_strings():
    config = _base_config()
    del config["branches"]["target"]
    config["branches"]["targets"] = ["cc8e8609e", "b9ac8350f"]
    specs = parse_targets(config)
    assert len(specs) == 2
    assert specs[0].branch == "cc8e8609e"
    assert specs[1].branch == "b9ac8350f"
    assert specs[0].issue_id is None
    assert specs[0].build_command is None


def test_parse_targets_list_of_mappings():
    config = _base_config()
    del config["branches"]["target"]
    config["branches"]["targets"] = [
        {"branch": "cc8e8609e", "issue_id": "fix-to-1.34"},
        {"branch": "b9ac8350f", "issue_id": "fix-to-1.33", "build_command": "make -j4"},
    ]
    specs = parse_targets(config)
    assert len(specs) == 2
    assert specs[0].issue_id == "fix-to-1.34"
    assert specs[0].build_command is None
    assert specs[1].issue_id == "fix-to-1.33"
    assert specs[1].build_command == "make -j4"


def test_parse_targets_mixed():
    config = _base_config()
    del config["branches"]["target"]
    config["branches"]["targets"] = [
        "cc8e8609e",
        {"branch": "b9ac8350f", "issue_id": "fix-to-1.33"},
    ]
    specs = parse_targets(config)
    assert len(specs) == 2
    assert specs[0].branch == "cc8e8609e"
    assert specs[0].issue_id is None
    assert specs[1].branch == "b9ac8350f"
    assert specs[1].issue_id == "fix-to-1.33"


def test_parse_targets_singular_fallback():
    config = _base_config()
    specs = parse_targets(config)
    assert len(specs) == 1
    assert specs[0].branch == "v1.0"
    assert specs[0].issue_id is None


def test_parse_targets_rejects_both():
    config = _base_config()
    config["branches"]["targets"] = ["cc8e8609e"]
    with pytest.raises(ConfigError, match="not both"):
        parse_targets(config)


def test_parse_targets_requires_one():
    config = _base_config()
    del config["branches"]["target"]
    with pytest.raises(ConfigError, match="must have 'target' or 'targets'"):
        parse_targets(config)


def test_parse_targets_rejects_duplicate_branches():
    config = _base_config()
    del config["branches"]["target"]
    config["branches"]["targets"] = ["origin/release-1.34", "release-1.34"]
    with pytest.raises(ConfigError, match="duplicate target after sanitization"):
        parse_targets(config)


def test_parse_targets_rejects_empty_list():
    config = _base_config()
    del config["branches"]["target"]
    config["branches"]["targets"] = []
    with pytest.raises(ConfigError, match="non-empty list"):
        parse_targets(config)


def test_parse_targets_rejects_mapping_without_branch():
    config = _base_config()
    del config["branches"]["target"]
    config["branches"]["targets"] = [{"issue_id": "no-branch"}]
    with pytest.raises(ConfigError, match="must have a 'branch' key"):
        parse_targets(config)


# --- load_multi_target_config ---


def test_load_multi_target_config_injects_synthetic_target():
    """Verify parse_bug_fix_config doesn't KeyError on targets-only YAML."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
issue_id: CVE-2024-1234
bug_description: Test CVE
repository:
  source_path: /tmp/test-repo
  workdir: /tmp/workdir
branches:
  source: main
  targets:
    - branch: "v1.0"
      issue_id: "fix-to-v1.0"
    - "v2.0"
fix:
  commit: abc123
build:
  command: make
  test_command: make test
""")
        f.flush()
        targets, base_config, claude_config, workdir, source_path = (
            load_multi_target_config(f.name)
        )

    os.unlink(f.name)
    assert len(targets) == 2
    assert targets[0].branch == "v1.0"
    assert targets[0].issue_id == "fix-to-v1.0"
    assert targets[1].branch == "v2.0"
    assert targets[1].issue_id is None
    assert base_config.target_branch == "v1.0"
    assert base_config.issue_id == "CVE-2024-1234"
    assert workdir == "/tmp/workdir"


def test_load_multi_target_config_singular_target():
    """Singular target YAML still works via load_multi_target_config."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
issue_id: CVE-2024-5678
bug_description: Test CVE
repository:
  source_path: /tmp/test-repo
branches:
  source: main
  target: v1.0
fix:
  commit: def456
build:
  command: make
  test_command: make test
""")
        f.flush()
        targets, base_config, _, _, _ = load_multi_target_config(f.name)

    os.unlink(f.name)
    assert len(targets) == 1
    assert targets[0].branch == "v1.0"
    assert base_config.target_branch == "v1.0"


# ---------- advanced_float: tolerant numeric parsing ----------


def _cfg_with_advanced(advanced: dict) -> dict:
    return {
        "issue_id": "t-1",
        "bug_description": "d",
        "repository": {"source_path": "/tmp/repo"},
        "branches": {"source": "main", "target": "v1.0"},
        "fix": {"commit": "abc123"},
        "build": {},
        "advanced": advanced,
    }


def test_advanced_float_accepts_quoted_string():
    """YAML quoting must not silently disable a configured threshold."""
    cfg, _, _ = parse_bug_fix_config(
        _cfg_with_advanced({"partial_success_skip_ratio": "0.7"})
    )
    assert cfg.partial_success_skip_ratio == 0.7


def test_advanced_float_accepts_number():
    cfg, _, _ = parse_bug_fix_config(
        _cfg_with_advanced({"partial_success_skip_ratio": 0.8})
    )
    assert cfg.partial_success_skip_ratio == 0.8


def test_advanced_float_null_falls_back_without_crashing(caplog):
    """A bare `partial_success_skip_ratio:` key yields None — must not raise."""
    cfg, _, _ = parse_bug_fix_config(
        _cfg_with_advanced({"partial_success_skip_ratio": None})
    )
    assert cfg.partial_success_skip_ratio == 0.5
    assert "not a number" in caplog.text


def test_advanced_float_garbage_warns_and_defaults(caplog):
    cfg, _, _ = parse_bug_fix_config(
        _cfg_with_advanced({"partial_success_skip_ratio": "high"})
    )
    assert cfg.partial_success_skip_ratio == 0.5
    assert "not a number" in caplog.text


def test_advanced_float_absent_is_silent(caplog):
    cfg, _, _ = parse_bug_fix_config(_cfg_with_advanced({}))
    assert cfg.partial_success_skip_ratio == 0.5
    assert "not a number" not in caplog.text


# --- advanced_float bounds ---
# The ratio gates a downgrade to partial_success. A value outside [0, 1] does
# not fail loudly, it silently disables (or inverts) that gate, so it must be
# rejected at load time.


@pytest.mark.parametrize(
    "bad", [-0.5, 2, 1.0001, float("nan"), float("inf"), float("-inf")]
)
def test_skip_ratio_outside_unit_interval_falls_back(bad, caplog):
    cfg, _, _ = parse_bug_fix_config(
        _cfg_with_advanced({"partial_success_skip_ratio": bad})
    )
    assert cfg.partial_success_skip_ratio == 0.5


def test_skip_ratio_nan_does_not_silently_disable_the_gate():
    """float('nan') makes every `ratio >= threshold` comparison false."""
    cfg, _, _ = parse_bug_fix_config(
        _cfg_with_advanced({"partial_success_skip_ratio": float("nan")})
    )
    assert cfg.partial_success_skip_ratio <= 1.0


def test_skip_ratio_above_one_cannot_suppress_a_total_drop():
    """A ratio of 2 would let a 100%-dropped port still report success."""
    cfg, _, _ = parse_bug_fix_config(
        _cfg_with_advanced({"partial_success_skip_ratio": 2})
    )
    assert cfg.partial_success_skip_ratio <= 1.0


@pytest.mark.parametrize("flag", [True, False])
def test_skip_ratio_boolean_is_rejected(flag, caplog):
    """float(True) is 1.0 — a config saying `true` never meant that."""
    cfg, _, _ = parse_bug_fix_config(
        _cfg_with_advanced({"partial_success_skip_ratio": flag})
    )
    assert cfg.partial_success_skip_ratio == 0.5
    assert "boolean" in caplog.text


@pytest.mark.parametrize(
    "good,expected", [(0, 0.0), (1, 1.0), ("0.75", 0.75), (0.25, 0.25)]
)
def test_skip_ratio_accepts_the_whole_unit_interval(good, expected):
    cfg, _, _ = parse_bug_fix_config(
        _cfg_with_advanced({"partial_success_skip_ratio": good})
    )
    assert cfg.partial_success_skip_ratio == expected


# --- phase_4b_model ---
# Phase 4b defaults to the pipeline model. If you want a different model for
# 4b, set it explicitly via advanced.phase_4b_model or a profile id.


def test_phase_4b_model_defaults_to_pipeline_model():
    """When no phase_4b_model is set, 4b uses the pipeline model."""
    cfg, _, _ = parse_bug_fix_config(_cfg_with_advanced({}))
    assert cfg.phase_4b_model == "claude-sonnet-4-6"


def test_phase_4b_model_follows_pipeline_model():
    """Setting model: moves 4b with it (no independent escalation by default)."""
    config = _cfg_with_advanced({})
    config["model"] = "openai/deepseek-r1"
    cfg, _, _ = parse_bug_fix_config(config)
    assert cfg.phase_4b_model == "openai/deepseek-r1"


def test_phase_4b_model_override_is_honoured():
    cfg, _, _ = parse_bug_fix_config(
        _cfg_with_advanced({"phase_4b_model": "claude-opus-5"})
    )
    assert cfg.phase_4b_model == "claude-opus-5"


def test_phase_4b_model_is_stripped():
    cfg, _, _ = parse_bug_fix_config(
        _cfg_with_advanced({"phase_4b_model": "  claude-opus-5  "})
    )
    assert cfg.phase_4b_model == "claude-opus-5"


@pytest.mark.parametrize("bad", ["", "   ", None, 42, ["claude-opus-5"], True])
def test_phase_4b_model_unusable_value_falls_back(bad, caplog):
    """An empty model name would surface as an opaque API error inside 4b."""
    cfg, _, _ = parse_bug_fix_config(_cfg_with_advanced({"phase_4b_model": bad}))
    assert cfg.phase_4b_model == "claude-sonnet-4-6"
    assert "non-empty string" in caplog.text


# --- patch mode in parse_bug_fix_config ---


def test_parse_patch_mode_populates_fix_patch(tmp_path):
    patch = _write_patch(tmp_path)
    cfg, _, _ = parse_bug_fix_config(_fix_cfg({"patch": patch}))
    assert cfg.fix_patch == patch
    assert cfg.source_fix_commit == ""
    assert cfg.fix_upstream_ref == ""


def test_parse_patch_mode_upstream_ref(tmp_path):
    patch = _write_patch(tmp_path)
    cfg, _, _ = parse_bug_fix_config(
        _fix_cfg({"patch": patch, "upstream_ref": "deadbeef1234"})
    )
    assert cfg.fix_upstream_ref == "deadbeef1234"


def test_parse_patch_mode_relative_path_resolves_against_config_dir(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    patch = _write_patch(config_dir)
    cfg, _, _ = parse_bug_fix_config(
        _fix_cfg({"patch": "fix.patch"}), config_dir=str(config_dir)
    )
    assert cfg.fix_patch == patch


def test_parse_commit_mode_leaves_patch_empty():
    cfg, _, _ = parse_bug_fix_config(_fix_cfg({"commit": "abc123"}))
    assert cfg.fix_patch == ""
    assert cfg.source_fix_commit == "abc123"


def test_fused_resolution_flag_defaults_off_and_parses():
    cfg, _, _ = parse_bug_fix_config(_cfg_with_advanced({}))
    assert cfg.fused_resolution is False
    cfg, _, _ = parse_bug_fix_config(_cfg_with_advanced({"fused_resolution": True}))
    assert cfg.fused_resolution is True


def test_fused_pre_red_flag_defaults_off_and_parses():
    cfg, _, _ = parse_bug_fix_config(_cfg_with_advanced({}))
    assert cfg.fused_pre_red is False
    cfg, _, _ = parse_bug_fix_config(_cfg_with_advanced({"fused_pre_red": True}))
    assert cfg.fused_pre_red is True


def test_tool_caps_accept_numeric_strings_and_warn_on_garbage(caplog):
    cfg, _, _ = parse_bug_fix_config(
        _cfg_with_advanced({"tool_read_cap": "12000", "tool_bash_cap": 8000})
    )
    assert cfg.tool_read_cap == 12000
    assert cfg.tool_bash_cap == 8000

    cfg, _, _ = parse_bug_fix_config(_cfg_with_advanced({"tool_read_cap": "big"}))
    assert cfg.tool_read_cap is None
    assert "not a positive integer" in caplog.text


def test_zero_budget_values_disable_instead_of_defaulting(caplog):
    cfg, _, _ = parse_bug_fix_config(
        _cfg_with_advanced({"review_wrapup_at": 0, "review_idle_turns": 0})
    )
    assert cfg.review_wrapup_at is None
    assert cfg.review_idle_turns is None
    assert "not a positive integer" in caplog.text


def test_budget_relationship_warnings(caplog):
    parse_bug_fix_config(
        _cfg_with_advanced(
            {
                "resolution_max_iterations": 50,
                "resolution_wrapup_at": 60,
                "review_max_iterations": 10,
                "review_wrapup_at": 20,
                "context_prune_tokens": 100000,
                "context_keep_tokens": 100000,
            }
        )
    )
    assert "resolution_wrapup_at > resolution_max_iterations" in caplog.text
    assert "review_wrapup_at > review_max_iterations" in caplog.text
    assert "context_keep_tokens >= context_prune_tokens" in caplog.text


def test_fused_optimization_flags_parse():
    cfg, _, _ = parse_bug_fix_config(
        _cfg_with_advanced(
            {
                "fused_compact_prompts": True,
                "tool_read_cap": 12000,
                "tool_bash_cap": 8000,
                "context_prune_tokens": 150000,
                "context_keep_tokens": "60000",
                "resolution_max_iterations": 160,
                "resolution_wrapup_at": "90",
                "resolution_idle_turns": 40,
                "review_max_iterations": 30,
                "review_wrapup_at": 20,
                "review_idle_turns": 15,
            }
        )
    )
    assert cfg.fused_compact_prompts is True
    assert cfg.tool_read_cap == 12000
    assert cfg.tool_bash_cap == 8000
    assert cfg.context_prune_tokens == 150000
    assert cfg.context_keep_tokens == 60000
    assert cfg.resolution_max_iterations == 160
    assert cfg.resolution_wrapup_at == 90
    assert cfg.resolution_idle_turns == 40
    assert cfg.review_max_iterations == 30
    assert cfg.review_wrapup_at == 20
    assert cfg.review_idle_turns == 15
    cfg, _, _ = parse_bug_fix_config(_cfg_with_advanced({}))
    assert cfg.fused_compact_prompts is False
    assert cfg.tool_read_cap is None and cfg.tool_bash_cap is None
    assert cfg.context_prune_tokens is None and cfg.context_keep_tokens is None
    assert cfg.resolution_max_iterations == 600
    assert cfg.resolution_wrapup_at is None
    assert cfg.resolution_idle_turns == 100
    assert cfg.review_max_iterations == 60
    assert cfg.review_wrapup_at == 45
    assert cfg.review_idle_turns is None
