import sys
import types

# Stub external dependencies before any project imports
_cm = types.ModuleType("colorama")
_cm.Fore = type("Fore", (), {k: "" for k in ["RED", "GREEN", "YELLOW", "CYAN", "MAGENTA", "RESET", "WHITE"]})()
_cm.Style = type("Style", (), {"RESET_ALL": "", "BRIGHT": ""})()
sys.modules["colorama"] = _cm

_sdk = types.ModuleType("claude_agent_sdk")
for _n in [
    "ClaudeSDKClient", "ClaudeAgentOptions", "AssistantMessage", "TextBlock",
    "ThinkingBlock", "ToolUseBlock", "ToolResultBlock", "ContentBlock",
    "ResultMessage", "SystemMessage", "UserMessage",
]:
    setattr(_sdk, _n, type(_n, (), {}))
sys.modules["claude_agent_sdk"] = _sdk
_sdk_types = types.ModuleType("claude_agent_sdk.types")
_sdk_types.StreamEvent = type("StreamEvent", (), {})
sys.modules["claude_agent_sdk.types"] = _sdk_types

import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from auto_bug_fix.bug_fix_config import BugFixConfig
from common.claude_utils import ClaudeConfig


@pytest.fixture
def claude_cfg():
    return ClaudeConfig(
        model="claude-opus-4-6",
        allowed_tools=["Read", "Write", "Bash"],
        perm_mode="acceptEdits",
        cwd="/tmp/test_cwd",
    )


@pytest.fixture
def bug_cfg():
    return BugFixConfig(
        repo_path="/path/to/gcc",
        build_dir="/path/to/gcc/build",
        source_branch="gcc-14-branch",
        target_branch="gcc-13-branch",
        source_fix_commit="abc1234",
        bug_description="CVE-2024-1234: buffer overflow in fold_convert()",
        issue_id="CVE-2024-1234",
        disallowed_modules=["gcc/config/arm/"],
    )


@dataclass
class RepoFixture:
    repo_path: str
    fix_sha: str
    source_branch: str = "main"
    target_branch: str = "target"


@pytest.fixture
def tmp_git_repo(tmp_path):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    rp = str(repo_path)

    subprocess.run(["git", "init", "-b", "main"], check=True, cwd=rp)
    subprocess.run(["git", "config", "user.name", "Test User"], check=True, cwd=rp)
    subprocess.run(["git", "config", "user.email", "test@example.com"], check=True, cwd=rp)

    # Create initial files
    lib_dir = repo_path / "lib"
    lib_dir.mkdir()
    (lib_dir / "url.c").write_text("void vulnerable_func() { /* bad code */ }")

    tests_dir = repo_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_url.c").write_text("void test_vuln() { /* test */ }")

    subprocess.run(["git", "add", "."], check=True, cwd=rp)
    subprocess.run(["git", "commit", "-m", "initial commit"], check=True, cwd=rp)

    # Create target branch from here
    subprocess.run(["git", "branch", "target"], check=True, cwd=rp)

    # On main: apply the fix
    (lib_dir / "url.c").write_text("void vulnerable_func() { /* fixed code */ }")
    (tests_dir / "test_fix.c").write_text("void test_fix() { /* fix test */ }")

    subprocess.run(["git", "add", "."], check=True, cwd=rp)
    subprocess.run(
        ["git", "commit", "-m", "fix: CVE-2024-1234 buffer overflow"],
        check=True,
        cwd=rp,
    )

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        cwd=rp,
        capture_output=True,
        text=True,
    )
    fix_sha = result.stdout.strip()

    return RepoFixture(repo_path=rp, fix_sha=fix_sha)
