"""
Backport Benchmark Evaluation Runner

Reads benchmark cases from the JavaBackports CSV, runs each configured
evaluator, collects diffs, judges semantic equivalence, and writes a
leaderboard report.

Key insight: the target branch in the dataset ALREADY contains the fix
(Backport Commit). So for each case we set up a git worktree at
`BACKPORT_SHA~1` — one commit before the fix — giving the tool a clean
target to actually backport onto. The ground truth is `git show BACKPORT_SHA`.

Usage:
  uv run python -m benchmark.evaluate.run_evaluation \\
      --config benchmark/evaluate/config.yaml \\
      [--dry-run]             # print cases without running evaluators
      [--evaluator NAME]      # run only one evaluator (e.g. raw_claude_opus)
      [--limit N]             # cap total cases (handy for quick tests)
      [--resume]              # skip cases that already have result.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import yaml

from .metrics import (
    CaseResult,
    AggregatedMetrics,
    aggregate,
    check_exact_match,
    estimate_cost,
    semantic_judge,
)
from .report import print_leaderboard, save_leaderboard

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ---------------------------------------------------------------------------
# Per-project build configuration
#
# We run compile-only commands (no full test suites — those take hours per
# project). Compilation validates that the backport applies cleanly to the
# target branch's API. The pipeline's Phase 4a/4b loop then fixes any errors.
#
# JAVA_HOME is injected into the subprocess environment so ./gradlew and mvn
# can find the JDK even when it is not on the system PATH.
# ---------------------------------------------------------------------------
JAVA_HOME = "/opt/homebrew/opt/openjdk"

# build_command: compile the project to catch API incompatibilities
# test_command:  "" means skip the test suite (compile is the key check)
PROJECT_BUILD_CONFIGS: dict[str, dict[str, str]] = {
    "crate": {
        "build_command": f"JAVA_HOME={JAVA_HOME} PATH={JAVA_HOME}/bin:/opt/homebrew/bin:$PATH ./gradlew compileJava --no-daemon -q",
        "test_command": "",
    },
    "grpc-java": {
        "build_command": f"JAVA_HOME={JAVA_HOME} PATH={JAVA_HOME}/bin:/opt/homebrew/bin:$PATH ./gradlew compileJava --no-daemon -q",
        "test_command": "",
    },
    "elasticsearch": {
        "build_command": f"JAVA_HOME={JAVA_HOME} PATH={JAVA_HOME}/bin:/opt/homebrew/bin:$PATH ./gradlew compileJava --no-daemon -q",
        "test_command": "",
    },
    "hadoop": {
        # Leave build_command empty — Hadoop is a 50+-module Maven monorepo and a
        # full compile takes 30-60 min.  The pipeline's triage phase auto-generates
        # targeted mvn test / checkstyle commands (verification_commands / lint_commands)
        # for the specific module that changed; those run in Phase 4.1 regardless.
        "build_command": "",
        "test_command": "",
    },
    "druid": {
        "build_command": f"JAVA_HOME={JAVA_HOME} PATH={JAVA_HOME}/bin:/opt/homebrew/bin:$PATH mvn compile -DskipTests -q -T 4",
        "test_command": "",
    },
    "graylog": {
        "build_command": f"JAVA_HOME={JAVA_HOME} PATH={JAVA_HOME}/bin:/opt/homebrew/bin:$PATH ./gradlew compileJava --no-daemon -q",
        "test_command": "",
    },
    "hbase": {
        "build_command": f"JAVA_HOME={JAVA_HOME} PATH={JAVA_HOME}/bin:/opt/homebrew/bin:$PATH mvn compile -DskipTests -q -T 4",
        "test_command": "",
    },
}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Dataset loading & sampling
# ---------------------------------------------------------------------------

def load_cases(cfg: dict) -> list[dict]:
    ds        = cfg["dataset"]
    csv_path  = ds["csv_path"]
    projects  = set(ds.get("projects") or [])
    types     = set(ds.get("types") or [])
    sample    = ds.get("sample_per_type", 999)
    seed      = ds.get("seed", 42)

    rows: list[dict] = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if projects and row["Project"] not in projects:
                continue
            if types and row["Type"] not in types:
                continue
            rows.append(row)

    if not rows:
        log.error("No cases found — check projects/types filters in config.")
        sys.exit(1)

    # Sample per TYPE globally.  Hadoop cases are always included first (the
    # user specifically requested Hadoop) since it has few cases and would
    # often be crowd-out by the larger projects.  Remaining slots are filled
    # with a random draw from the rest of the pool.
    rng = random.Random(seed)
    priority_projects = {"hadoop"}

    by_type: dict[str, list[dict]] = {}
    for row in rows:
        by_type.setdefault(row["Type"], []).append(row)

    sampled: list[dict] = []
    for t, t_rows in by_type.items():
        priority = [r for r in t_rows if r["Project"] in priority_projects]
        others   = [r for r in t_rows if r["Project"] not in priority_projects]
        rng.shuffle(others)
        # Always take ALL priority cases (up to `sample`), then fill with others
        bucket = priority[:sample] + others[: max(0, sample - len(priority))]
        sampled.extend(bucket)

    log.info(
        "Loaded %d cases — %s",
        len(sampled),
        ", ".join(f"{t}×{min(sample, len(v))}" for t, v in sorted(by_type.items())),
    )
    return sampled


# ---------------------------------------------------------------------------
# Repo management
# ---------------------------------------------------------------------------

def ensure_repo(project: str, cfg: dict) -> Path:
    """Clone repo if not already present, then fetch all branches."""
    repos_root = Path(cfg["repos"]["root"])
    repo_path  = repos_root / project

    if not repo_path.exists():
        repo_cfg = cfg["repos"].get(project)
        if not repo_cfg:
            raise ValueError(
                f"No repo URL configured for '{project}'. Add it under repos: in config.yaml."
            )
        url     = repo_cfg["url"]
        shallow = repo_cfg.get("shallow", True)
        log.info("Cloning %s (shallow=%s)…", url, shallow)
        repos_root.mkdir(parents=True, exist_ok=True)
        cmd = ["git", "clone", "--filter=blob:none"]
        if shallow:
            cmd += ["--depth", "2000"]
        cmd += [url, str(repo_path)]
        subprocess.run(cmd, check=True)

    log.info("Fetching all refs for %s…", project)
    subprocess.run(
        ["git", "-C", str(repo_path), "fetch", "--all", "--quiet"],
        check=True,
    )
    return repo_path


# ---------------------------------------------------------------------------
# Target state helpers
#
# The dataset's Backport Commit is the ALREADY-FIXED state.
# Both evaluators need to work against BACKPORT_SHA~1 (one before the fix):
#
#  raw_claude  — reads file content via `git show BACKPORT_SHA~1:{file}`,
#                no worktree needed.
#
#  pipeline    — needs a real branch pointing to BACKPORT_SHA~1 so the
#                pipeline's own worktree setup works correctly. We create a
#                temp branch, run the pipeline, then delete it.
# ---------------------------------------------------------------------------

def resolve_pre_fix_sha(repo_path: Path, backport_sha: str) -> str | None:
    """Return the SHA one commit before the backport (BACKPORT_SHA~1)."""
    result = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", f"{backport_sha}~1"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        log.warning("Could not resolve %s~1: %s", backport_sha[:8], result.stderr.strip())
        return None
    return result.stdout.strip()


def create_temp_branch(repo_path: Path, sha: str, branch_name: str) -> bool:
    """Create a local branch pointing to sha. Returns True on success."""
    # Unlock and remove any worktree on this branch before recreating it.
    # Handles "locked initializing" state left by killed runs.
    _unlock_and_remove_worktrees_for_branch(repo_path, branch_name)
    # Force-delete any stale branch so we can recreate it at the target SHA
    subprocess.run(
        ["git", "-C", str(repo_path), "branch", "-D", branch_name],
        capture_output=True,
    )
    r = subprocess.run(
        ["git", "-C", str(repo_path), "branch", branch_name, sha],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        log.warning("Could not create branch %s at %s: %s", branch_name, sha[:8], r.stderr.strip())
        return False
    return True


def _unlock_and_remove_worktrees_for_branch(repo_path: Path, branch_name: str) -> None:
    """Remove any worktrees (including locked/initializing ones) tied to branch_name."""
    # First, remove lock files from any .git/worktrees/* entry that references
    # this branch — git refuses to prune "locked initializing" worktrees otherwise.
    git_worktrees_dir = repo_path / ".git" / "worktrees"
    if git_worktrees_dir.exists():
        for wt_meta in git_worktrees_dir.iterdir():
            head_file = wt_meta / "HEAD"
            lock_file = wt_meta / "locked"
            if head_file.exists() and branch_name in head_file.read_text():
                lock_file.unlink(missing_ok=True)

    # Prune expired / gone worktrees
    subprocess.run(
        ["git", "-C", str(repo_path), "worktree", "prune", "--expire=now"],
        capture_output=True,
    )

    # Remove any still-registered worktree whose branch matches
    worktree_list = subprocess.run(
        ["git", "-C", str(repo_path), "worktree", "list", "--porcelain"],
        capture_output=True, text=True,
    ).stdout
    for block in worktree_list.split("\n\n"):
        if f"branch refs/heads/{branch_name}" in block:
            lines = block.splitlines()
            if lines and lines[0].startswith("worktree "):
                wt_path = lines[0].replace("worktree ", "").strip()
                subprocess.run(
                    ["git", "-C", str(repo_path), "worktree", "remove", wt_path, "--force"],
                    capture_output=True,
                )


def delete_temp_branch(repo_path: Path, branch_name: str) -> None:
    _unlock_and_remove_worktrees_for_branch(repo_path, branch_name)
    subprocess.run(
        ["git", "-C", str(repo_path), "branch", "-D", branch_name],
        capture_output=True,
    )


# ---------------------------------------------------------------------------
# Ground-truth diff extraction
# ---------------------------------------------------------------------------

def get_groundtruth_diff(repo_path: Path, backport_sha: str) -> str:
    """
    The ground truth is the actual human backport commit — what a developer
    committed when porting the fix to the target branch.
    """
    result = subprocess.run(
        ["git", "-C", str(repo_path), "show", "--unified=10", backport_sha],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git show {backport_sha} failed: {result.stderr.strip()}")
    return result.stdout


# ---------------------------------------------------------------------------
# Evaluator: Raw Claude (single-shot)
# ---------------------------------------------------------------------------

def run_raw_claude(
    case: dict,
    repo_path: Path,
    pre_fix_sha: str,
    out_diff_path: Path,
    eval_cfg: dict,
    amalgamator_path: str,
) -> tuple[int, int, float, str]:
    """
    Run the single-shot Claude evaluator.
    Passes pre_fix_sha so the script can read file content at BACKPORT_SHA~1
    via git show without needing a real worktree.
    Returns (input_tokens, output_tokens, latency_s, error_msg).
    """
    orig_sha      = case["Original Commit"]
    target_branch = case["Backport Version"]
    model         = eval_cfg.get("model", "claude-opus-4-6")

    cmd = (
        f"cd {amalgamator_path} && "
        f"uv run python -m benchmark.generic_claude_backport "
        f"--repo {repo_path} "
        f"--orig {orig_sha} "
        f"--target-sha {pre_fix_sha} "
        f"--target-branch {target_branch} "
        f"--model {model} "
        f"--out {out_diff_path}"
    )

    t0 = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
    latency = time.time() - t0

    if result.returncode != 0:
        return 0, 0, latency, result.stderr[-2000:]

    # Parse token counts logged by generic_claude_backport
    input_tok = output_tok = 0
    for line in (result.stdout + result.stderr).splitlines():
        if "input:" in line and "output:" in line:
            try:
                input_tok  = int(line.split("input:")[1].split(",")[0].strip())
                output_tok = int(line.split("output:")[1].strip())
            except Exception:
                pass

    return input_tok, output_tok, latency, ""


# ---------------------------------------------------------------------------
# Evaluator: Amalgamator Pipeline
# ---------------------------------------------------------------------------

def _write_pipeline_config(
    case: dict,
    repo_path: Path,
    target_worktree: Path,
    template_path: str,
    model: str,
    config_out: Path,
) -> None:
    """Fill in benchmark_template.yaml for this specific case."""
    with open(template_path) as f:
        cfg_data = yaml.safe_load(f)

    orig_sha      = case["Original Commit"]
    target_branch = case["Backport Version"]
    project       = case["Project"]

    cfg_data["issue_id"]                   = f"{project}-{orig_sha[:12]}"
    cfg_data["model"]                      = model
    cfg_data["repository"]["source_path"]  = str(repo_path)
    cfg_data["repository"]["workdir"]      = str(target_worktree)
    cfg_data["branches"]["source"]         = f"origin/{case['Original Version']}"
    cfg_data["branches"]["target"]         = f"origin/{target_branch}"
    cfg_data["fix"]["commit"]              = orig_sha
    cfg_data["bug_description"]            = (
        f"Benchmark case: backport {orig_sha[:12]} onto {target_branch}. "
        f"Complexity: {case['Type']}."
    )

    config_out.parent.mkdir(parents=True, exist_ok=True)
    with open(config_out, "w") as f:
        yaml.dump(cfg_data, f)


def run_pipeline(
    case: dict,
    repo_path: Path,
    temp_branch: str,
    out_diff_path: Path,
    eval_cfg: dict,
    amalgamator_path: str,
    scratch_dir: Path,
) -> tuple[int, int, float, str]:
    """
    Run the amalgamator auto_bug_fix pipeline for one case.

    The pipeline creates its own worktree from `source_path` at `branches.target`.
    We point it at the temp branch (sitting at BACKPORT_SHA~1) so it has a clean
    target that doesn't already have the fix.

    After the pipeline runs, the worktree is at workdir/{issue_id}/. We read
    `git show HEAD` from there (the cherry-picked commit) to get the diff.

    Returns (input_tokens, output_tokens, latency_s, error_msg).
    """
    orig_sha  = case["Original Commit"]
    project   = case["Project"]
    issue_id  = f"{project}-{orig_sha[:12]}"
    model     = eval_cfg.get("model", "claude-opus-4-6")
    workdir   = str(scratch_dir / "pipeline_workdir")

    # Per-project build/test commands (compile-only; full test suites take hours)
    build_cfg = PROJECT_BUILD_CONFIGS.get(project, {"build_command": "", "test_command": ""})

    # Write per-case pipeline config
    config_path = scratch_dir / f"{orig_sha[:12]}-pipeline-config.yaml"
    cfg_data = {
        "issue_id": issue_id,
        "bug_description": (
            f"Benchmark: backport {orig_sha[:12]} onto {case['Backport Version']}. "
            f"Complexity: {case['Type']}."
        ),
        "repository": {
            "source_path": str(repo_path),
            "workdir": workdir,
        },
        "branches": {
            "source": f"origin/{case['Original Version']}",
            # Use the temp branch at BACKPORT_SHA~1 — not the real target head
            "target": temp_branch,
        },
        "fix": {"commit": orig_sha},
        "build": {
            "command": build_cfg["build_command"],
            "test_command": build_cfg["test_command"],
        },
        "model": model,
        # Cap LLM resolution loops to prevent runaway token burn / timeouts
        "advanced": {
            "max_resolution_retries": 2,
            "max_build_test_retries": 2,
        },
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(cfg_data, f)

    # Pass JAVA_HOME so Gradle/Maven wrappers inside the worktree find the JDK
    env = os.environ.copy()
    env["JAVA_HOME"] = JAVA_HOME
    env["PATH"] = f"{JAVA_HOME}/bin:/opt/homebrew/bin:{env.get('PATH', '')}"

    cmd = (
        f"cd {amalgamator_path} && "
        f"uv run python -m auto_bug_fix.run_bug_fix {config_path}"
    )

    t0 = time.time()
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True,
        timeout=3600,  # 1 hour — compile + LLM fix loop can take 30-60 min
        env=env,
    )
    latency = time.time() - t0

    if result.returncode != 0:
        return 0, 0, latency, (result.stdout + result.stderr)[-2000:]

    # The pipeline's worktree lives at workdir/{issue_id}/
    pipeline_worktree = Path(workdir) / issue_id
    if not pipeline_worktree.exists():
        return 0, 0, latency, f"Pipeline worktree not found at {pipeline_worktree}"

    # Get the cherry-picked commit diff (top commit in the worktree)
    diff_result = subprocess.run(
        ["git", "-C", str(pipeline_worktree), "show", "HEAD", "--unified=10"],
        capture_output=True, text=True,
    )
    if diff_result.returncode != 0 or not diff_result.stdout.strip():
        # Fall back to diff against the branch base
        diff_result = subprocess.run(
            ["git", "-C", str(pipeline_worktree), "diff", "HEAD~1", "HEAD"],
            capture_output=True, text=True,
        )

    out_diff_path.parent.mkdir(parents=True, exist_ok=True)
    out_diff_path.write_text(diff_result.stdout)

    # Extract total token cost from the pipeline tracker JSON.
    # The pipeline writes: {amalgamator_path}/runs/**/{issue_id}_{timestamp}.json
    in_tok = out_tok = 0
    try:
        import glob as _glob
        runs_root = Path(amalgamator_path) / "runs"
        tracker_files = sorted(_glob.glob(str(runs_root / "**" / f"{issue_id}_*.json"), recursive=True))
        if tracker_files:
            tracker_data = json.loads(Path(tracker_files[-1]).read_text())
            in_tok  = tracker_data.get("total_input_tokens", 0)
            out_tok = tracker_data.get("total_output_tokens", 0)
            log.info("Pipeline cost — input: %d  output: %d  total: $%.4f",
                     in_tok, out_tok, tracker_data.get("total_cost_usd", 0))
        else:
            log.warning("No tracker JSON found for %s in %s", issue_id, runs_root)
    except Exception as e:
        log.warning("Could not parse tracker JSON for %s: %s", orig_sha[:8], e)

    return in_tok, out_tok, latency, ""


# ---------------------------------------------------------------------------
# Per-case evaluation
# ---------------------------------------------------------------------------

def evaluate_case(
    case: dict,
    repo_path: Path,
    eval_name: str,
    eval_cfg: dict,
    output_root: Path,
    judge_model: str,
    scratch_dir: Path,
) -> CaseResult:
    orig_sha      = case["Original Commit"]
    backport_sha  = case["Backport Commit"]
    project       = case["Project"]
    backport_type = case["Type"]
    target_branch = case["Backport Version"]

    result = CaseResult(
        evaluator=eval_name,
        project=project,
        backport_type=backport_type,
        orig_sha=orig_sha,
        target_branch=target_branch,
    )

    case_dir      = output_root / eval_name / project / orig_sha[:12]
    gen_diff_path = case_dir / "generated.diff"
    gt_diff_path  = case_dir / "groundtruth.diff"
    result.generated_diff_path   = str(gen_diff_path)
    result.groundtruth_diff_path = str(gt_diff_path)

    # 1. Get ground-truth diff (what the human actually committed)
    try:
        gt_diff = get_groundtruth_diff(repo_path, backport_sha)
        gt_diff_path.parent.mkdir(parents=True, exist_ok=True)
        gt_diff_path.write_text(gt_diff)
    except Exception as e:
        result.error_msg = f"groundtruth extraction failed: {e}"
        return result

    # 2. Resolve BACKPORT_SHA~1 — the state one commit before the fix
    pre_fix_sha = resolve_pre_fix_sha(repo_path, backport_sha)
    if pre_fix_sha is None:
        result.error_msg = f"Could not resolve {backport_sha[:8]}~1"
        return result

    amalgamator_path = eval_cfg.get("amalgamator_path", ".")
    strategy         = eval_cfg.get("strategy", "raw_claude")

    # For pipeline strategy: create a temp local branch at pre_fix_sha so the
    # pipeline's worktree setup points to the pre-fix state, not the fixed HEAD.
    temp_branch = f"bench-target-{orig_sha[:12]}"
    if strategy == "pipeline":
        if not create_temp_branch(repo_path, pre_fix_sha, temp_branch):
            result.error_msg = f"Could not create temp branch at {pre_fix_sha[:8]}"
            return result

    try:
        t0 = time.time()
        if strategy == "raw_claude":
            in_tok, out_tok, latency, err = run_raw_claude(
                case, repo_path, pre_fix_sha, gen_diff_path, eval_cfg, amalgamator_path
            )
        elif strategy == "pipeline":
            in_tok, out_tok, latency, err = run_pipeline(
                case, repo_path, temp_branch, gen_diff_path, eval_cfg, amalgamator_path, scratch_dir
            )
        else:
            err     = f"Unknown strategy: {strategy}"
            in_tok  = out_tok = 0
            latency = time.time() - t0

    except Exception:
        result.error_msg = traceback.format_exc()
        result.latency_s = time.time() - t0
        return result
    finally:
        if strategy == "pipeline":
            delete_temp_branch(repo_path, temp_branch)

    result.latency_s     = latency
    result.input_tokens  = in_tok
    result.output_tokens = out_tok
    result.cost_usd      = estimate_cost(in_tok, out_tok)

    if err:
        result.error_msg = err
        return result

    # 3. Check if a diff was produced
    if not gen_diff_path.exists() or gen_diff_path.stat().st_size == 0:
        result.error_msg = "Evaluator produced no diff"
        return result

    gen_diff = gen_diff_path.read_text()

    # 4. Exact match
    result.exact_match = check_exact_match(gen_diff, gt_diff)

    # 5. Semantic equivalence judge
    try:
        verdict, reasoning, j_in, j_out = semantic_judge(
            generated_diff=gen_diff,
            groundtruth_diff=gt_diff,
            project=project,
            orig_sha=orig_sha,
            target_branch=target_branch,
            model=judge_model,
        )
        result.semantic_equiv  = verdict
        result.judge_reasoning = reasoning
        result.input_tokens  += j_in
        result.output_tokens += j_out
        result.cost_usd      += estimate_cost(j_in, j_out)
    except Exception as e:
        result.semantic_equiv  = "uncertain"
        result.judge_reasoning = f"Judge failed: {e}"

    return result


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(config_path: str, args: argparse.Namespace) -> None:
    cfg         = load_config(config_path)
    output_root = Path(cfg["output"]["root"])
    judge_model = cfg["judge"]["model"]
    scratch_dir = output_root / "_scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    cases = load_cases(cfg)
    if args.limit:
        cases = cases[:args.limit]

    evaluators = {
        k: v for k, v in cfg["evaluators"].items()
        if v.get("enabled", True)
        and (not args.evaluator or k == args.evaluator)
    }
    if not evaluators:
        log.error("No evaluators enabled. Check config or --evaluator flag.")
        sys.exit(1)

    log.info(
        "Plan: %d cases × %d evaluators = %d total runs",
        len(cases), len(evaluators), len(cases) * len(evaluators),
    )

    all_results: dict[str, list[CaseResult]] = {name: [] for name in evaluators}

    # Ensure all required repos are cloned
    repo_cache: dict[str, Path] = {}
    for project in {c["Project"] for c in cases}:
        try:
            repo_cache[project] = ensure_repo(project, cfg)
        except Exception as e:
            log.error("Failed to set up repo for %s: %s", project, e)

    # Main evaluation loop
    for i, case in enumerate(cases, 1):
        project = case["Project"]
        if project not in repo_cache:
            log.warning("Skipping %s — repo unavailable", project)
            continue

        repo_path = repo_cache[project]
        log.info(
            "[%d/%d]  %s | orig=%s | target=%s | %s",
            i, len(cases),
            project,
            case["Original Commit"][:8],
            case["Backport Version"],
            case["Type"],
        )

        for eval_name, eval_cfg in evaluators.items():
            if args.dry_run:
                log.info("  [DRY RUN] evaluator=%s  strategy=%s  model=%s",
                         eval_name, eval_cfg.get("strategy"), eval_cfg.get("model"))
                continue

            result_path = (
                output_root / eval_name / project /
                case["Original Commit"][:12] / "result.json"
            )
            if args.resume and result_path.exists():
                log.info("  Skipping %s (already done)", eval_name)
                cached = CaseResult(**json.loads(result_path.read_text()))
                all_results[eval_name].append(cached)
                continue

            log.info("  → evaluator=%s  strategy=%s  model=%s",
                     eval_name, eval_cfg.get("strategy"), eval_cfg.get("model"))
            result = evaluate_case(
                case=case,
                repo_path=repo_path,
                eval_name=eval_name,
                eval_cfg=eval_cfg,
                output_root=output_root,
                judge_model=judge_model,
                scratch_dir=scratch_dir,
            )
            all_results[eval_name].append(result)

            # Save per-case result
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(result.to_dict(), indent=2))

            if result.error_msg:
                log.warning("  ✗ error: %s", result.error_msg[:80])
            else:
                log.info("  ✓ semantic=%s  exact=%s  latency=%.1fs  cost=$%.4f",
                         result.semantic_equiv, result.exact_match,
                         result.latency_s, result.cost_usd)

    if args.dry_run:
        log.info("Dry-run complete — %d cases would be evaluated.", len(cases))
        return

    # Aggregate & report
    aggregated: list[AggregatedMetrics] = []
    for eval_name, results in all_results.items():
        if results:
            agg = aggregate(results, eval_name)
            aggregated.append(agg)
            agg_path = output_root / eval_name / "aggregated.json"
            agg_path.write_text(json.dumps(agg.to_dict(), indent=2))

    if aggregated:
        print_leaderboard(aggregated, cfg["evaluators"])
        if cfg["output"].get("leaderboard"):
            save_leaderboard(aggregated, output_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backport benchmark evaluation runner")
    parser.add_argument("--config",    default="benchmark/evaluate/config.yaml")
    parser.add_argument("--dry-run",   action="store_true", help="Preview cases without running")
    parser.add_argument("--evaluator", default="", help="Run only this evaluator key")
    parser.add_argument("--limit",     type=int, default=0, help="Max total cases (0=no limit)")
    parser.add_argument("--resume",    action="store_true", help="Skip completed cases")
    args = parser.parse_args()
    run(args.config, args)


if __name__ == "__main__":
    main()
