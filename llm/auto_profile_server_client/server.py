"""Framework-agnostic server lifecycle: start, health-check, wait-ready."""

from __future__ import annotations

import logging
import shlex
import subprocess
import time
from pathlib import Path

from llm.auto_profile_server_client.config import (
    SERVER_HEALTH_POLL_SEC,
    SERVER_HEALTH_TIMEOUT_SEC,
)

log = logging.getLogger(__name__)

ERROR_MARKERS = (
    "RuntimeError:",
    "Error:",
    "FATAL",
    "OOMKilled",
    "CUDA error",
    "OutOfMemoryError",
    "NCCL error",
    "CalledProcessError",
)


class ServerProcess:
    """Manages a serving process running inside a docker container."""

    def __init__(self, process: subprocess.Popen, port: int, health_url: str):
        self.process = process
        self.port = port
        self.health_url = health_url
        self._started = False
        self._log_path: str | None = None

    def _extract_error_context(
        self, context_before: int = 5, context_after: int = 3
    ) -> str:
        """Extract the root-cause error from the server log with context.

        Finds the first error marker in the log (the root cause), includes
        surrounding context lines, and appends the log tail for final status.
        """
        if not self._log_path:
            return ""
        try:
            text = Path(self._log_path).read_text()
        except Exception:
            return ""

        lines = text.splitlines()
        if not lines:
            return ""

        # Find the first line with an error marker (root cause)
        first_error = None
        for i, line in enumerate(lines):
            if any(m in line for m in ERROR_MARKERS):
                first_error = i
                break

        if first_error is None:
            tail = "\n".join(lines[-30:])
            return f"\n--- server log (last 30 lines of {len(lines)}) ---\n{tail}"

        # Grab the error block: from first marker, continue through
        # traceback lines (indented, File "...", ^^^, or more error markers)
        block_start = max(0, first_error - context_before)
        block_end = first_error
        for i in range(first_error, len(lines)):
            line = lines[i].rstrip()
            stripped = line.lstrip()
            if (
                any(m in line for m in ERROR_MARKERS)
                or stripped.startswith("File ")
                or stripped.startswith("Traceback")
                or stripped.startswith("^^^")
                or stripped.startswith("raise ")
                or (stripped and stripped[0] == " ")
                or stripped == ""
            ):
                block_end = i
            else:
                if i > first_error + 2:
                    break
                block_end = i

        block_end = min(block_end + context_after, len(lines) - 1)
        excerpt = lines[block_start : block_end + 1]

        parts = [f"\n--- root cause (line {first_error + 1} of {len(lines)}) ---"]
        parts.extend(excerpt)

        # Add tail if the error block doesn't reach the end
        if block_end < len(lines) - 5:
            parts.append(f"\n--- log tail (last 5 of {len(lines)} lines) ---")
            parts.extend(lines[-5:])

        return "\n".join(parts)

    def wait_until_ready(self, timeout: int = SERVER_HEALTH_TIMEOUT_SEC) -> None:
        """Poll the health endpoint until the server is ready."""
        log.info("Waiting for server at %s (timeout=%ds)", self.health_url, timeout)
        deadline = time.time() + timeout
        last_error = ""

        while time.time() < deadline:
            if self.process.poll() is not None:
                error_ctx = self._extract_error_context()
                raise RuntimeError(
                    f"Server process exited with code {self.process.returncode} "
                    f"before becoming ready{error_ctx}"
                )

            try:
                r = subprocess.run(
                    ["curl", "-sf", "--max-time", "5", self.health_url],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if r.returncode == 0:
                    self._started = True
                    log.info("Server is ready at %s", self.health_url)
                    return
                last_error = r.stderr[:200] if r.stderr else f"curl exit {r.returncode}"
            except Exception as e:
                last_error = str(e)

            time.sleep(SERVER_HEALTH_POLL_SEC)

        raise TimeoutError(
            f"Server not ready after {timeout}s. Last error: {last_error}"
        )

    def stop(self) -> None:
        """Kill the entire process group (server + tee + workers)."""
        import os
        import signal

        if self.process.poll() is not None:
            log.info("Server already exited (code=%d)", self.process.returncode)
            return

        pid = self.process.pid
        log.info("Stopping server process group (PID=%d)", pid)
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except ProcessLookupError, PermissionError:
            self.process.terminate()

        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            log.warning("Server did not exit after 30s, killing process group")
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except ProcessLookupError, PermissionError:
                self.process.kill()
            self.process.wait(timeout=10)

    @property
    def is_running(self) -> bool:
        return self.process.poll() is None


def start_server_in_container(
    cmd: str,
    port: int,
    health_url: str,
    log_path: str | None = None,
) -> ServerProcess:
    """Launch the server command as a background subprocess.

    Output is captured to log_path via tee, with tee's stdout sent to
    /dev/null to avoid filling the subprocess pipe buffer.
    """
    log.info("Starting server: %s", cmd[:200])

    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        header = f"{'=' * 80}\nSERVER COMMAND\n{'=' * 80}\n{cmd}\n{'=' * 80}\n\n"
        Path(log_path).write_text(header)
        cmd = f"{cmd} 2>&1 | tee -a {shlex.quote(log_path)} > /dev/null; exit ${{PIPESTATUS[0]}}"
        log.info("Server logs: %s", log_path)

    proc = subprocess.Popen(["bash", "-c", cmd], start_new_session=True)
    server = ServerProcess(process=proc, port=port, health_url=health_url)
    server._log_path = log_path
    return server
