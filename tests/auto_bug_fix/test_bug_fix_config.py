import pytest
from auto_bug_fix.bug_fix_config import BugFixConfig


def test_construction_sets_branch_fields(bug_cfg):
    assert bug_cfg.source_branch == "gcc-14-branch"
    assert bug_cfg.target_branch == "gcc-13-branch"
    assert bug_cfg.source_fix_commit == "abc1234"


def test_construction_sets_build_fields(bug_cfg):
    assert bug_cfg.build_command == "make -j$(nproc)"
    assert bug_cfg.test_command == "make check -j$(nproc)"
    assert bug_cfg.build_dir == "/path/to/gcc/build"
    assert bug_cfg.max_build_test_retries == 3


def test_construction_sets_porting_fields(bug_cfg):
    assert bug_cfg.port_tests is True
    assert "gcc/config/arm/" in bug_cfg.disallowed_modules


def test_port_tests_false():
    config = BugFixConfig(
        repo_path="/path/to/gcc",
        build_dir="/path/to/gcc/build",
        source_branch="gcc-14-branch",
        target_branch="gcc-13-branch",
        source_fix_commit="abc1234",
        bug_description="test",
        issue_id="GH-1",
        disallowed_modules=[],
        port_tests=False,
        build_command="make -j$(nproc)",
        test_command="make check -j$(nproc)",
        max_build_test_retries=3,
    )
    assert config.port_tests is False
