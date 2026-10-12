"""NSYS profiling orchestration — wraps benchmark commands with nsys."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from llm.auto_profile_server_client.config import NSYS_DEFAULT_FLAGS
from llm.auto_profile_server_client.models import ProfileWindow

log = logging.getLogger(__name__)


def build_nsys_command(
    benchmark_cmd: str,
    output_prefix: str,
    window: ProfileWindow | None = None,
) -> str:
    """Wrap a benchmark command with nsys profile.

    For time-based profiling (vLLM/SGLang), nsys uses cudaProfilerApi
    which is triggered by the framework's internal profiling hooks.
    For iteration-based profiling (TRT), the TLLM_PROFILE_START_STOP
    env var controls when profiling starts/stops.
    """
    flags = " ".join(NSYS_DEFAULT_FLAGS)
    return f"nsys profile {flags} -o {output_prefix} {benchmark_cmd}"


def export_nsys_to_sqlite(nsys_rep_path: str) -> str:
    """Convert .nsys-rep file to .sqlite format."""
    sqlite_path = nsys_rep_path.replace(".nsys-rep", ".sqlite")
    log.info("Exporting nsys trace to SQLite: %s -> %s", nsys_rep_path, sqlite_path)

    result = subprocess.run(
        ["nsys", "export", "--type=sqlite", f"--output={sqlite_path}", nsys_rep_path],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        log.error("nsys export failed: %s", result.stderr[-1000:])
        raise RuntimeError(
            f"nsys export failed for {nsys_rep_path}: {result.stderr[-500:]}"
        )

    log.info("Exported: %s", sqlite_path)
    return sqlite_path


def find_nsys_rep_files(directory: str) -> list[str]:
    """Find all .nsys-rep files in a directory."""
    return sorted(str(p) for p in Path(directory).rglob("*.nsys-rep"))
