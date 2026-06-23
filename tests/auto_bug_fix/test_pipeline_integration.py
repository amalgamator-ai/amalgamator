"""Integration test for the cherry-pick pipeline with mocked LLM calls."""
import json
import os
import subprocess
from unittest.mock import patch, AsyncMock

import pytest

from auto_bug_fix.bug_fix_config import BugFixConfig
from auto_bug_fix.run_bug_fix import run_pipeline, PipelineStop
from common.claude_utils import ClaudeConfig


TRIAGE_CLEAN = {
    "commit_type": "bug-fix",
    "new_packages": [],
    "new_types_interfaces": [],
    "dependency_depth": 0,
    "target_has_prerequisites": True,
    "prerequisite_details": "none",
    "prerequisites": [],
    "generated_code_in_diff": False,
    "binary_files_in_diff": False,
    "files_to_skip": [],
    "regeneration_commands": [],
    "lint_commands": [],
    "verification_commands": [],
    "estimated_difficulty": "EASY",
    "recommendation": "PROCEED",
    "priority_files": [],
    "reasoning": "test commit",
}


def _git(repo, *args, check=True):
    return subprocess.run(
        ["git"] + list(args), cwd=repo,
        capture_output=True, text=True, check=check,
    )


@pytest.fixture
def clean_cherry_pick_repo(tmp_path):
    """Create a repo where the fix cherry-picks cleanly onto target."""
    repo = str(tmp_path / "repo")
    os.makedirs(repo)

    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "t@t.com")

    # Initial commit on main
    src = tmp_path / "repo" / "src"
    src.mkdir()
    (src / "app.py").write_text("def hello():\n    return 'hello'\n")
    (src / "utils.py").write_text("def add(a, b):\n    return a + b\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")

    # Create target branch with a diverging commit (so cherry-pick isn't empty)
    _git(repo, "branch", "target")
    _git(repo, "checkout", "target")
    (src / "utils.py").write_text("def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "target: add sub function")
    _git(repo, "checkout", "main")

    # Fix on main — touches app.py which target didn't change
    (src / "app.py").write_text("def hello():\n    return 'hello world'\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fix: improve greeting")
    fix_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()

    # Switch repo to target branch (pipeline operates on repo_path directly)
    _git(repo, "checkout", "target")

    return {"repo": repo, "fix_sha": fix_sha}


@pytest.fixture
def conflict_cherry_pick_repo(tmp_path):
    """Create a repo where the fix conflicts with target."""
    repo = str(tmp_path / "repo")
    os.makedirs(repo)

    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "t@t.com")

    src = tmp_path / "repo" / "src"
    src.mkdir()
    (src / "app.py").write_text("def hello():\n    return 'hello'\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")

    # Create target branch
    _git(repo, "branch", "target")

    # Diverge on target — change the same line
    _git(repo, "checkout", "target")
    (src / "app.py").write_text("def hello():\n    return 'hi there'\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "target: different greeting")
    _git(repo, "checkout", "main")

    # Fix on main — also changes the same line
    (src / "app.py").write_text("def hello():\n    return 'hello world'\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fix: improve greeting")
    fix_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()

    # Switch to target for the pipeline
    _git(repo, "checkout", "target")

    return {"repo": repo, "fix_sha": fix_sha}


def _make_mock_claude_run(repo_path):
    """Create a mock that simulates LLM responses for each pipeline phase."""
    call_count = [0]

    async def mock_claude_run(claude_config, prompts, tracker=None):
        call_count[0] += 1
        cwd = claude_config.cwd

        prompt_text = prompts[0] if isinstance(prompts[0], str) else prompts[0].prompt
        if isinstance(prompt_text, dict):
            return []

        if "Semantic Triage Agent" in prompt_text:
            triage_path = os.path.join(cwd, "semantic_triage.json")
            with open(triage_path, "w") as f:
                json.dump(TRIAGE_CLEAN, f)
            return [{"name": "step_1", "duration": 0.1, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0}]

        if "Test Port Agent" in prompt_text:
            manifest_path = os.path.join(cwd, "test_port_manifest.txt")
            with open(manifest_path, "w") as f:
                f.write("")
            return [{"name": "step_1", "duration": 0.1, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0}]

        if "Narrow Resolution Agent" in prompt_text:
            # Resolve all conflicts with --theirs and commit
            status = subprocess.run(
                ["git", "diff", "--name-only", "--diff-filter=U"],
                cwd=cwd, capture_output=True, text=True,
            ).stdout.strip().splitlines()
            for f in status:
                subprocess.run(["git", "checkout", "--theirs", "--", f], cwd=cwd, capture_output=True)
                subprocess.run(["git", "add", "--", f], cwd=cwd, capture_output=True)
            subprocess.run(
                ["git", "commit", "--allow-empty", "-m", "mock resolution"],
                cwd=cwd, capture_output=True,
            )
            return [{"name": "step_1", "duration": 0.1, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0}]

        if "Build Error Recovery Agent" in prompt_text:
            return [{"name": "step_1", "duration": 0.1, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0}]

        return []

    return mock_claude_run


@patch("auto_bug_fix.run_bug_fix.phase_4b_external_pipeline")
@patch("auto_bug_fix.run_bug_fix.claude_run")
def test_clean_cherry_pick_pipeline(mock_run, mock_4b, clean_cherry_pick_repo, tmp_path):
    repo = clean_cherry_pick_repo["repo"]
    fix_sha = clean_cherry_pick_repo["fix_sha"]
    output_dir = str(tmp_path / "runs")
    os.makedirs(output_dir, exist_ok=True)

    mock_run.side_effect = _make_mock_claude_run(repo)

    config = BugFixConfig(
        repo_path=repo,
        build_dir=repo,
        source_branch="main",
        target_branch="target",
        source_fix_commit=fix_sha,
        bug_description="test fix",
        issue_id="test-clean",
        build_command="true",
        test_command="true",
    )
    claude_config = ClaudeConfig(
        model="claude-sonnet-4-6",
        allowed_tools=["Bash"],
        perm_mode="acceptEdits",
        cwd=repo,
    )

    dossier, tracker = run_pipeline(config, claude_config, output_dir=output_dir)

    # Pipeline succeeded
    assert tracker._run.outcome == "success"

    # Fix was applied — file should have the fix content
    with open(os.path.join(repo, "src", "app.py")) as f:
        content = f.read()
    assert "hello world" in content

    # No pipeline artifacts in the last commit
    committed_files = _git(repo, "show", "--name-only", "--format=", "HEAD").stdout.strip()
    assert "semantic_triage.json" not in committed_files
    assert "test_port_manifest.txt" not in committed_files


@patch("auto_bug_fix.run_bug_fix.phase_4b_external_pipeline")
@patch("auto_bug_fix.run_bug_fix.claude_run")
def test_conflict_cherry_pick_pipeline(mock_run, mock_4b, conflict_cherry_pick_repo, tmp_path):
    repo = conflict_cherry_pick_repo["repo"]
    fix_sha = conflict_cherry_pick_repo["fix_sha"]
    output_dir = str(tmp_path / "runs")
    os.makedirs(output_dir, exist_ok=True)

    mock_run.side_effect = _make_mock_claude_run(repo)

    config = BugFixConfig(
        repo_path=repo,
        build_dir=repo,
        source_branch="main",
        target_branch="target",
        source_fix_commit=fix_sha,
        bug_description="test fix with conflict",
        issue_id="test-conflict",
        build_command="true",
        test_command="true",
    )
    claude_config = ClaudeConfig(
        model="claude-sonnet-4-6",
        allowed_tools=["Bash"],
        perm_mode="acceptEdits",
        cwd=repo,
    )

    dossier, tracker = run_pipeline(config, claude_config, output_dir=output_dir)

    # Pipeline succeeded (mock resolved the conflict)
    assert tracker._run.outcome == "success"

    # A commit was made
    log_output = _git(repo, "log", "--oneline", "-3").stdout
    assert len(log_output.strip().splitlines()) >= 2

    # No pipeline artifacts in commits
    all_files = _git(repo, "diff", "--name-only", "target", "HEAD").stdout.strip()
    assert "semantic_triage.json" not in all_files
    assert "test_port_manifest.txt" not in all_files
