import json
import os
import pytest
from auto_code_gen.code_gen_configs import BugFixConfig, PipelineConfig
from common.claude_utils import ClaudeConfig


class TestBugFixConfigInheritance:
    def test_inherits_from_pipeline_config(self):
        assert issubclass(BugFixConfig, PipelineConfig)

    def test_has_pipeline_config_fields(self, bug_fix_cfg):
        assert hasattr(bug_fix_cfg, "output_dir")
        assert hasattr(bug_fix_cfg, "source_code_dir")
        assert hasattr(bug_fix_cfg, "num_code_port_plan_iterations")
        assert hasattr(bug_fix_cfg, "num_test_plan_iterations")
        assert hasattr(bug_fix_cfg, "num_code_gen_iterations")
        assert hasattr(bug_fix_cfg, "disallowed_modules")
        assert hasattr(bug_fix_cfg, "thinking_mode")


class TestBugFixConfigConstruction:
    def test_branch_fields(self, bug_fix_cfg):
        assert bug_fix_cfg.source_branch == "gcc-14-branch"
        assert bug_fix_cfg.target_branch == "gcc-13-branch"
        assert bug_fix_cfg.source_fix_commit == "abc1234"

    def test_build_fields(self, bug_fix_cfg):
        assert bug_fix_cfg.build_command == "make -j$(nproc)"
        assert bug_fix_cfg.test_command == "make check -j$(nproc)"
        assert bug_fix_cfg.build_dir == "/path/to/gcc/build"
        assert bug_fix_cfg.max_build_test_retries == 3

    def test_porting_fields(self, bug_fix_cfg):
        assert "gcc/config/arm/" in bug_fix_cfg.disallowed_modules


class TestBugFixConfigFromJson:
    def _write_config(self, tmp_path, overrides=None):
        data = {
            "use_case": "bug_fix",
            "repo_path": str(tmp_path),
            "build_dir": str(tmp_path),
            "source_branch": "gcc-14-branch",
            "target_branch": "gcc-13-branch",
            "source_fix_commit": "abc1234",
            "bug_description": "CVE-2024-1234: test bug",
            "issue_id": "CVE-2024-1234",
            "output_dir": str(tmp_path / "output"),
            "build_command": "make -j$(nproc)",
            "test_command": "make check -j$(nproc)",
            "max_build_test_retries": 3,
            "disallowed_modules": ["gcc/config/arm/"],
            "num_code_port_plan_iterations": 2,
            "num_test_plan_iterations": 2,
            "num_code_gen_iterations": 2,
            "thinking-mode": "deep",
        }
        if overrides:
            data.update(overrides)
        config_path = tmp_path / "bug_fix_config.json"
        with open(config_path, "w") as f:
            json.dump(data, f)
        return str(config_path)

    def test_from_json_valid(self, tmp_path):
        path = self._write_config(tmp_path)
        config = BugFixConfig.from_json(path)
        assert config.source_branch == "gcc-14-branch"
        assert config.target_branch == "gcc-13-branch"
        assert config.source_fix_commit == "abc1234"
        assert config.num_code_port_plan_iterations == 2
        assert config.thinking_mode == "deep"
        assert "gcc/config/arm/" in config.disallowed_modules

    def test_from_json_missing_required_field(self, tmp_path):
        path = self._write_config(tmp_path, overrides={"source_branch": ""})
        with pytest.raises(ValueError, match="source_branch"):
            BugFixConfig.from_json(path)

    def test_from_json_missing_output_dir(self, tmp_path):
        path = self._write_config(tmp_path, overrides={"output_dir": ""})
        with pytest.raises(ValueError, match="output_dir"):
            BugFixConfig.from_json(path)

    def test_from_json_missing_build_command(self, tmp_path):
        path = self._write_config(tmp_path, overrides={"build_command": ""})
        with pytest.raises(ValueError, match="build_command"):
            BugFixConfig.from_json(path)

    def test_from_json_invalid_thinking_mode(self, tmp_path):
        path = self._write_config(tmp_path, overrides={"thinking-mode": "invalid"})
        with pytest.raises(ValueError, match="thinking-mode"):
            BugFixConfig.from_json(path)


class TestBugFixConfigMakeClaudeConfig:
    def test_make_claude_config_deep(self):
        config = BugFixConfig(
            output_dir="/tmp/output",
            thinking_mode="deep",
        )
        cc = config.make_claude_config()
        assert isinstance(cc, ClaudeConfig)
        assert cc.model == "claude-opus-4-6[1m]"
        assert cc.cwd == "/tmp/output"
        assert cc.thinking == {"type": "adaptive"}
        assert cc.effort == "max"
        assert cc.max_thinking_tokens == 1048576
        assert cc.perm_mode == "acceptEdits"
        assert "Bash" in cc.allowed_tools

    def test_make_claude_config_normal(self):
        config = BugFixConfig(
            output_dir="/tmp/output",
            thinking_mode="normal",
        )
        cc = config.make_claude_config()
        assert cc.model == "claude-sonnet-4-6"
        assert cc.effort == "medium"
        assert cc.max_thinking_tokens == 65536
