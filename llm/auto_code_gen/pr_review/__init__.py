import re
import subprocess
import sys


def parse_pr_url(url: str) -> tuple[str, str, int]:
    m = re.match(r"https://github\.com/([^/]+)/([^/]+)/pull/(\d+)", url)
    if not m:
        raise ValueError(f"Invalid GitHub PR URL: {url}")
    return m.group(1), m.group(2), int(m.group(3))


def run_gh(args: list[str], cwd: str | None = None) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["gh"] + args,
        cwd=cwd,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        stderr_lower = result.stderr.lower()
        if "not logged in" in stderr_lower or "auth" in stderr_lower:
            print("ERROR: gh is not authenticated.")
            print("  Run:  gh auth login")
            sys.exit(1)
        raise RuntimeError(f"gh command failed: {result.stderr.strip()}")
    return result


def run_git(
    args: list[str], cwd: str, check: bool = True
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git"] + args,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=check,
    )


def check_gh_auth() -> bool:
    result = subprocess.run(
        ["gh", "auth", "status"],
        text=True,
        capture_output=True,
    )
    return result.returncode == 0
