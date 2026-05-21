from __future__ import annotations

import os
import sys
import shutil
import subprocess
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(_PROJECT_ROOT, "logs")

PROFILE_RESULTS_DIR = "./auto_profile/results"
_OUTPUT_DIR_PREFIX = "./auto_analyze/results/results_analyze_"


def test_name_from_run_config(run_config: str) -> str:
    basename = os.path.basename(run_config)
    if not basename.startswith("run_") or not basename.endswith(".json"):
        raise ValueError(
            f"run_config filename must match run_<test_name>.json, got: {basename}"
        )
    return basename[len("run_"):-len(".json")]


def output_dir_from_run_config(run_config: str) -> str:
    return _OUTPUT_DIR_PREFIX + test_name_from_run_config(run_config)


def setup_logging(name: str) -> None:
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = os.path.join(LOGS_DIR, "run_{}.log".format(name))
    log_file = open(log_path, "w")
    sys.stdout = Tee(sys.stdout, log_file)


class Tee(object):
    """A file-like object that writes to multiple files simultaneously."""

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # Ensure immediate writing

    def flush(self):
        for f in self.files:
            f.flush()


_SENSITIVE_DIRS = frozenset({
    "/", "/bin", "/boot", "/dev", "/etc", "/home", "/lib", "/lib64",
    "/media", "/mnt", "/opt", "/proc", "/root", "/run", "/sbin",
    "/srv", "/sys", "/tmp", "/usr", "/var",
})


def safe_clean_dir(dir_path: str | Path) -> None:
    """Remove all contents of a directory, leaving the directory itself."""
    dir_path = Path(dir_path).expanduser().resolve()

    resolved = str(dir_path)
    if resolved in _SENSITIVE_DIRS:
        raise ValueError(f"Refusing to clean sensitive directory: {resolved}")

    if Path(resolved) == Path.home():
        raise ValueError(f"Refusing to clean home directory: {resolved}")

    # Must be at least 3 levels deep (e.g. /home/user/something)
    if len(dir_path.parts) < 3:
        raise ValueError(
            f"Refusing to clean directory with fewer than 3 path components: {resolved}"
        )

    if not dir_path.exists():
        return

    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {resolved}")

    if dir_path.is_symlink():
        raise ValueError(f"Refusing to clean symlink target: {resolved}")

    cwd_resolved = str(Path.cwd().resolve())
    project_resolved = str(Path(_PROJECT_ROOT).resolve())
    if resolved == cwd_resolved or resolved == project_resolved:
        raise ValueError(
            f"Refusing to clean directory that resolves to the current "
            f"working directory or project root.\n"
            f"  Resolved dir:   {resolved}\n"
            f"  CWD:            {cwd_resolved}\n"
            f"  Project root:   {project_resolved}\n"
            f"  Use a dedicated subdirectory to avoid deleting project files."
        )

    if (dir_path / ".git").is_dir():
        raise ValueError(
            f"Refusing to clean directory that is a git repository root: {resolved}\n"
            f"  Use a dedicated subdirectory to avoid deleting repository files."
        )

    print(f"Cleaning directory: {resolved}")
    for child in dir_path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def clean_output_dir(output_dir: str | Path, keep_files: dict[str, str],
                     extra_keep: list[str] | None = None) -> None:
    """Remove intermediate files from output_dir, keeping only known outputs.

    Walks output_dir and deletes any file not in the keep set. The
    run_originals/ subdirectory and run_params*.txt are always preserved.

    Parameters
    ----------
    output_dir:
        The analysis output directory.
    keep_files:
        The output_files dict returned by gen_*_prompts(). Values are
        absolute paths or filenames relative to output_dir.
    extra_keep:
        Additional filenames (basenames) to preserve, e.g. result_metadata.json.
    """
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_dir():
        return

    keep_abs = set()
    for path in keep_files.values():
        p = Path(path)
        keep_abs.add(str(p if p.is_absolute() else output_dir / p))

    always_keep_basenames = {"run_params.txt", "run_params_cross.txt",
                             "result_metadata.json"}
    if extra_keep:
        always_keep_basenames.update(extra_keep)

    removed = []
    for root, dirs, files in os.walk(output_dir, topdown=False):
        root_path = Path(root)

        if "run_originals" in root_path.parts:
            continue

        for fname in files:
            fpath = root_path / fname
            if str(fpath) in keep_abs:
                continue
            if fname in always_keep_basenames:
                continue
            fpath.unlink()
            removed.append(str(fpath))

        if root_path != output_dir and not any(root_path.iterdir()):
            root_path.rmdir()
            removed.append(str(root_path) + "/")

    if removed:
        print(f"  Cleaned {len(removed)} intermediate file(s) from {output_dir}")


def clear_vllm_source_tree(
    source_dir: str | Path,
    *,
    remove_ignored: bool = False,
    dry_run: bool = False,
) -> None:
    """
    Reset a vLLM source tree to a clean git state.

    This removes:
    - tracked modifications
    - staged changes
    - untracked files
    - untracked directories

    Parameters
    ----------
    source_dir:
        Path to the vLLM git repository or any directory inside it.
    remove_ignored:
        If True, also remove ignored files/directories (equivalent to `git clean -fdx`).
    dry_run:
        If True, only print what would be removed.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist.
    NotADirectoryError
        If the path is not a directory.
    RuntimeError
        If git is missing, the path is not a git repo, or a git command fails.
    """
    source_dir = Path(source_dir).expanduser().resolve()

    if shutil.which("git") is None:
        raise RuntimeError("git is not installed or not in PATH")

    if not source_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {source_dir}")

    if not source_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {source_dir}")

    repo_check = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=source_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    if repo_check.returncode != 0 or repo_check.stdout.strip() != "true":
        raise RuntimeError(f"Not a git repository: {source_dir}")

    root_result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=source_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    if root_result.returncode != 0:
        raise RuntimeError(
            f"Failed to determine repository root:\n{root_result.stderr.strip()}"
        )

    repo_root = Path(root_result.stdout.strip()).resolve()

    if dry_run:
        clean_cmd = ["git", "clean", "-fdn"]
        if remove_ignored:
            clean_cmd = ["git", "clean", "-fdxn"]

        print(f"[dry-run] Repo root: {repo_root}")
        print("[dry-run] Would run: git reset --hard")
        print(f"[dry-run] Would run: {' '.join(clean_cmd)}")

        result = subprocess.run(
            clean_cmd,
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.returncode != 0:
            raise RuntimeError(f"Dry-run clean failed:\n{result.stderr.strip()}")
        return

    reset_result = subprocess.run(
        ["git", "reset", "--hard"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if reset_result.returncode != 0:
        raise RuntimeError(f"`git reset --hard` failed:\n{reset_result.stderr.strip()}")

    clean_cmd = ["git", "clean", "-fdx"] if remove_ignored else ["git", "clean", "-fd"]
    clean_result = subprocess.run(
        clean_cmd,
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if clean_result.returncode != 0:
        raise RuntimeError(
            f"`{' '.join(clean_cmd)}` failed:\n{clean_result.stderr.strip()}"
        )


def clear_repo(path: str | Path) -> None:
    """Reset a git repository to a clean state. Generic alias for clear_vllm_source_tree."""
    clear_vllm_source_tree(path)
