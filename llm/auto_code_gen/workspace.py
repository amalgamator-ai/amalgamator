"""Cleanup for codegen-owned source branches."""

import subprocess
from pathlib import Path


def clear_vllm_source_tree(source_dir):
    """Reset generated changes while retaining local agent configuration."""
    source_dir = Path(source_dir).resolve()
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=source_dir,
        text=True,
        capture_output=True,
        check=True,
    )
    root = Path(result.stdout.strip()).resolve()
    commands = [
        ["git", "reset", "--hard"],
        [
            "git",
            "clean",
            "-fd",
            "-e",
            "/.agents/",
            "-e",
            "/.codex/",
            "-e",
            "/.claude/",
            "-e",
            "/.venv/",
        ],
    ]
    for command in commands:
        result = subprocess.run(command, cwd=root, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(f"{' '.join(command)} failed:\n{result.stderr.strip()}")
