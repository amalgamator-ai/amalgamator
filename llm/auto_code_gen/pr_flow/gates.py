# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pass/fail rules for a PR-review verification iteration.

Three gates, all fail-closed:
  perf        -- throughput must clear the original improvement gate AND keep
                 most of the delta achieved when the PR was accepted, so review
                 fixes cannot silently trade the win away;
  correctness -- lm_eval gsm8k must stay within tolerance of the shared OFF
                 baseline and above an absolute floor;
  closure     -- every fetched review comment must be marked fixed or explained.
"""

import os as _os
import sys as _sys
from typing import TypedDict

_project_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import hashlib
import json
import os
import re

from llm.auto_code_gen.run_code_gen import (
    _bench_throughput_pair,
    _check_performance_meets_target,
)

REGRESSION_RETENTION = 0.8
LM_EVAL_MIN_DELTA = -0.01
# Sanity floor, not a precision instrument: the unmodified baseline itself
# samples 0.91-0.97 on the 200-question gsm8k slice, so a 0.94 floor sat inside
# the baseline's own noise band and failed the unmodified model two runs out of
# three. Regression protection is the (stderr-aware) delta check against the
# OFF baseline; this floor only has to catch catastrophic breakage, so it sits
# below the baseline's observed noise band.
LM_EVAL_MIN_ABSOLUTE = 0.90
# Two independent single-sample scores are compared, so the noise on the delta
# is sqrt(2) x a single sample's stderr. With the default 200-question gsm8k
# slice (stderr ~0.017) that is ~0.024: the old fixed -0.01 tolerance sat well
# inside it and false-failed unchanged code about a third of the time. The
# delta check instead requires the degradation to clear this many combined
# standard errors before it counts as real.
LM_EVAL_DELTA_SIGMAS = 1.65

VERDICT_FILE = "verify_verdict.json"

RESOLUTION_STATUSES = ("fixed", "explained", "unresolved")


def recorded_perf_pct(state, output_dir=None):
    """Best-known achieved throughput delta for this PR, in percent.

    Prefers the newest accepted gate record; falls back to success_result.txt,
    which is what the pipeline wrote while this was still a local patch.
    """
    for gate in reversed(list(state.get("gates") or [])):
        if gate.get("overall") == "PASS" or gate.get("accepted"):
            value = (gate.get("perf") or {}).get("actual_pct")
            if isinstance(value, (int, float)):
                return float(value)
    if output_dir:
        from llm.auto_code_gen.code_gen_prompts import RUNTIME_DIR, RUNTIME_SUCCESS_FILE

        try:
            with open(os.path.join(output_dir, RUNTIME_DIR, RUNTIME_SUCCESS_FILE)) as f:
                text = f.read()
        except OSError:
            return None
        m = re.search(r"throughput[^\n]*?\(\s*([-+]\d+(?:\.\d+)?)%", text, re.I)
        if not m:
            m = re.search(r"output_throughput[^\n]*?([-+]\d+(?:\.\d+)?)%", text, re.I)
        if m:
            return float(m.group(1))
    return None


def measure_throughput(iter_dir):
    """Read (measured, actual_pct) straight out of the two bench artifacts.

    Independent of any threshold, so it can tell "nothing to compare" apart from
    "compared, and the win is smaller than asked for".
    """
    try:
        with open(os.path.join(iter_dir, "bench_result.json")) as f:
            patched = json.load(f)
        with open(os.path.join(iter_dir, "baseline_bench_result.json")) as f:
            baseline = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False, 0.0

    patched_tps, baseline_tps, _ = _bench_throughput_pair(patched, baseline)
    if patched_tps is None or baseline_tps is None:
        return False, 0.0
    return True, (patched_tps - baseline_tps) / baseline_tps * 100


def evaluate_perf(
    iter_dir, expected_pct, recorded_pct=None, retention=REGRESSION_RETENTION
):
    """Compare this iteration's bench ON against the shared bench OFF baseline."""
    _, actual_pct, floor_pct = _check_performance_meets_target(
        iter_dir, None, expected_pct
    )

    # The first return of _check_performance_meets_target means "diluted plan
    # gate met", which is False both for an absent measurement and for a real
    # regression. Conflating them tells a repair agent its artifacts are missing
    # when the win simply shrank, so it hunts files instead of kernels.
    measured, measured_pct = measure_throughput(iter_dir)
    if measured and not actual_pct:
        actual_pct = measured_pct
    if expected_pct is None:
        # No expectation exists for this plan, so there is no threshold to fail
        # against -- the one case allowed to pass without a measurement.
        measured = True

    regression_floor = (abs(recorded_pct) * retention) if recorded_pct else floor_pct
    required = max(floor_pct, regression_floor)
    passed = bool(measured) and actual_pct >= required

    reason = ""
    if not measured:
        reason = "bench result missing/unreadable or no shared throughput metric"
    elif not passed:
        reason = f"actual {actual_pct:+.2f}% < required {required:+.2f}% (gate {floor_pct:+.2f}%, retention {regression_floor:+.2f}%)"

    return {
        "pass": passed,
        "measured": bool(measured),
        "actual_pct": round(actual_pct, 4),
        "gate_floor_pct": round(floor_pct, 4),
        "regression_floor_pct": round(regression_floor, 4),
        "required_pct": round(required, 4),
        "recorded_pct": recorded_pct,
        "reason": reason,
    }


_LMResult = TypedDict(
    "_LMResult",
    {
        "pass": bool,
        "verdict": str,
        "reason": str,
        "flexible": float | None,
        "strict": float | None,
        "delta_flexible": float | None,
        "delta_strict": float | None,
        "delta_stderr": float | None,
        "delta_tolerance": float,
    },
)


def _lm_eval_stderr(iter_dir):
    """Combined standard error of (ON score - OFF score), or None.

    lm_eval writes a results JSON (and stdout table) with per-metric stderr.
    The two scores being compared are independent single samples, so the
    delta's noise is sqrt(se_on^2 + se_off^2). Returns None when neither
    run left a parsable stderr, so the caller can fall back to the fixed
    tolerance.
    """
    stderrs = []
    patterns = [
        # results_*.json: "exact_match_stderr,flexible-extract": 0.0168...
        (os.path.join(iter_dir, "lm_eval_output"), "json"),
        # baseline copy: lm_eval_output_disabled/...
        (os.path.join(iter_dir, "lm_eval_output_disabled"), "json"),
    ]
    for base, _kind in patterns:
        best = None
        if os.path.isdir(base):
            for root, _dirs, files in os.walk(base):
                for name in files:
                    if not name.endswith(".json"):
                        continue
                    try:
                        with open(os.path.join(root, name)) as f:
                            data = json.load(f)
                    except (OSError, ValueError):
                        continue
                    task = (data.get("results") or {}).get("gsm8k") or {}
                    se = task.get("exact_match_stderr,flexible-extract")
                    if (
                        isinstance(se, (int, float))
                        and se > 0
                        and (best is None or se > best)
                    ):
                        best = float(se)
        if best is not None:
            stderrs.append(best)
    if not stderrs:
        return None
    if len(stderrs) == 1:
        # Only one arm's stderr is known: assume both arms are equally noisy
        # (same task, same sample count) and scale by sqrt(2).
        return (2 * stderrs[0] ** 2) ** 0.5
    return (stderrs[0] ** 2 + stderrs[1] ** 2) ** 0.5


def evaluate_lm_eval(iter_dir):
    """Read lm_eval.txt as rewritten by _recompute_lm_eval_verdict()."""
    result: _LMResult = {
        "pass": False,
        "verdict": "MISSING",
        "flexible": None,
        "strict": None,
        "delta_flexible": None,
        "delta_strict": None,
        "delta_stderr": None,
        "delta_tolerance": 0.0,
        "reason": "",
    }
    path = os.path.join(iter_dir, "lm_eval.txt")
    if not os.path.isfile(path):
        result["reason"] = "lm_eval.txt not written (lm_eval stage did not run)"
        return result

    with open(path) as f:
        text = f.read()

    m = re.search(r"flexible-extract=([0-9.]+),\s*strict-match=([0-9.]+)", text)
    if m:
        result["flexible"] = float(m.group(1))
        result["strict"] = float(m.group(2))
    flex_delta = re.search(r"Delta flexible-extract:\s*([-+][0-9.]+)", text)
    if flex_delta:
        result["delta_flexible"] = float(flex_delta.group(1))
    strict_delta = re.search(r"Delta strict-match:\s*([-+][0-9.]+)", text)
    if strict_delta:
        result["delta_strict"] = float(strict_delta.group(1))
    verdict = re.search(r"Verdict:\s*(\w+)", text)
    if verdict:
        result["verdict"] = verdict.group(1).upper()

    # The delta is a difference of two noisy single-sample scores: compare it
    # against its own standard error, not a fixed tolerance that sits inside
    # the noise band (with 200 questions a -0.03 delta is ~1.25 sigma, i.e.
    # indistinguishable from zero). Falls back to the fixed tolerance when no
    # stderr was recorded.
    se = _lm_eval_stderr(iter_dir)
    result["delta_stderr"] = round(se, 6) if se is not None else None
    if se is not None:
        result["delta_tolerance"] = -LM_EVAL_DELTA_SIGMAS * se
    else:
        result["delta_tolerance"] = LM_EVAL_MIN_DELTA

    deltas = [
        d for d in (result["delta_flexible"], result["delta_strict"]) if d is not None
    ]
    if result["verdict"] != "PASS":
        result["reason"] = "verdict is {}".format(result["verdict"])
    elif not deltas:
        result["reason"] = "no accuracy delta available to compare"
    elif any(d < result["delta_tolerance"] for d in deltas):
        result["reason"] = "accuracy delta {} below tolerance {:.4f}{}".format(
            deltas,
            result["delta_tolerance"],
            (
                f" ({LM_EVAL_DELTA_SIGMAS} sigma of the measured noise)"
                if se is not None
                else ""
            ),
        )
    elif result["flexible"] is not None and result["flexible"] < LM_EVAL_MIN_ABSOLUTE:
        result["reason"] = "absolute score {} below floor {}".format(
            result["flexible"], LM_EVAL_MIN_ABSOLUTE
        )
    else:
        result["pass"] = True
    return result


def parse_comment_resolution(path):
    """Parse the strict per-comment blocks the fix step is required to write.

    Expected block form (see pr_flow/prompts.py):

        ## Comment 12345678: vllm/foo.py:42
        STATUS: fixed
        REPLY: <one-line answer to the reviewer>
    """
    summary = {
        "total": 0,
        "fixed": 0,
        "explained": 0,
        "unresolved": 0,
        "comment_ids": [],
        "unresolved_ids": [],
        "entries": [],
        "pass": False,
        "reason": "",
    }
    if not os.path.isfile(path):
        summary["reason"] = "comment_resolution.md not written"
        return summary

    with open(path) as f:
        lines = f.read().splitlines()

    current = None
    for line in lines:
        header = re.match(r"^##\s*Comment\s+(\d+)\s*(?::\s*(.*))?$", line.strip())
        if header:
            current = {
                "comment_id": int(header.group(1)),
                "location": (header.group(2) or "").strip(),
                "status": "unresolved",
                "reply": "",
            }
            summary["entries"].append(current)
            continue
        if current is None:
            continue
        status = re.match(r"^STATUS:\s*(\w+)", line.strip())
        if status:
            value = status.group(1).lower()
            current["status"] = value if value in RESOLUTION_STATUSES else "unresolved"
            continue
        reply = re.match(r"^REPLY:\s*(.*)$", line.strip())
        if reply:
            current["reply"] = reply.group(1).strip()

    status_counts: dict[str, int] = {}
    for entry in summary["entries"]:
        status_counts[entry["status"]] = status_counts.get(entry["status"], 0) + 1
        summary["comment_ids"].append(entry["comment_id"])
        if entry["status"] == "unresolved":
            summary["unresolved_ids"].append(entry["comment_id"])

    summary["total"] = len(summary["entries"])
    summary["fixed"] = status_counts.get("fixed", 0)
    summary["explained"] = status_counts.get("explained", 0)
    summary["unresolved"] = status_counts.get("unresolved", 0)
    summary["pass"] = summary["total"] > 0 and summary["unresolved"] == 0
    if summary["total"] == 0:
        summary["reason"] = "no '## Comment <id>' blocks found"
    elif summary["unresolved"]:
        summary["reason"] = "{} unresolved comment(s): {}".format(
            summary["unresolved"], summary["unresolved_ids"]
        )
    return summary


MEASURED_CONTENT_FILE = "measured_content.json"


def _sha256(text):
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def _file_hash(repo_dir, rel):
    path = os.path.join(repo_dir, rel)
    try:
        with open(path, errors="replace") as f:
            return _sha256(f.read())
    except OSError:
        return "MISSING"


def worktree_fingerprint(repo_dir, files):
    """Content hash of the exact bytes the cluster job was given."""
    return {rel: _file_hash(repo_dir, rel) for rel in sorted(set(files))}


def committed_fingerprint(repo_dir, files, commit="HEAD"):
    """Same hash computed from a commit's tree instead of the worktree."""
    import subprocess

    result = {}
    for rel in sorted(set(files)):
        proc = subprocess.run(
            ["git", "-C", repo_dir, "show", f"{commit}:{rel}"],
            capture_output=True,
            text=True,
        )
        result[rel] = _sha256(proc.stdout) if proc.returncode == 0 else "MISSING"
    return result


def write_measured_content(iter_dir, pr_head_sha, base_commit, files, repo_dir):
    """Record what was measured, so a later commit of the same bytes is provable.

    The fix step edits the worktree and the harness measures that worktree; the
    human commits those bytes afterwards, which necessarily produces a different
    commit. Fingerprinting the content is what lets step 4 prove it is pushing
    exactly what was measured -- commit identity cannot prove that.
    """
    fingerprint = worktree_fingerprint(repo_dir, files)
    payload = {
        "pr_head_sha": pr_head_sha,
        "base_commit": base_commit,
        "files": fingerprint,
    }
    write_verdict(os.path.join(iter_dir, MEASURED_CONTENT_FILE), payload)
    return payload


def _read_measured_content(verdict_path):
    if not verdict_path:
        return None
    path = os.path.join(
        os.path.dirname(os.path.abspath(verdict_path)), MEASURED_CONTENT_FILE
    )
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def measured_content_matches_head(state, verdict_path):
    """Does HEAD hold exactly the bytes the verdict's iteration measured?

    Returns (matched, detail). matched is None when there is no fingerprint to
    compare against, so the caller keeps its stricter commit-identity rule.
    """
    recorded = _read_measured_content(verdict_path)
    if not recorded or not (recorded.get("files") or {}):
        return None, "no measured_content.json beside the verdict"
    repo_dir = state["repo_dir"]
    now = committed_fingerprint(repo_dir, recorded["files"].keys())
    differ = sorted(
        rel for rel, digest in recorded["files"].items() if now.get(rel) != digest
    )
    if differ:
        return False, "content differs from what was measured in: {}".format(
            ", ".join(differ[:10])
        )
    return True, "{} file(s) verified byte-for-byte against the measured tree".format(
        len(recorded["files"])
    )


def baseline_is_valid(state, baseline_kind, current_base_commit):
    """A shared OFF baseline is reusable only while the base commit is unchanged.

    A rebased PR invalidates every OFF measurement: the comparison would mix two
    upstream trees. An unstamped baseline counts as invalid so it is re-measured
    and stamped once instead of being trusted blindly.
    """
    if not current_base_commit:
        return False
    recorded = (state.get("baselines") or {}).get(f"{baseline_kind}_base_commit", "")
    return bool(recorded) and recorded == current_base_commit


def toggle_fingerprint(repo_dir, disable_var, pr_files):
    """Hash the code that decides what the feature-OFF arm actually turns off.

    A shared OFF baseline only means "the feature is off" while the kill switch
    still gates the same code. P-3's switch silently failed to gate its primary
    GEMM, so every OFF number on record had that GEMM switched on -- and a base
    commit stamp cannot see that, because the switch's own code is PR-owned and
    moves between iterations. Hashing the PR files that read the switch makes any
    change to the switch's reach invalidate the reference instead of quietly
    reusing one that measured something else.
    """
    if not disable_var:
        return ""
    digest = hashlib.sha256(disable_var.encode())
    for rel in sorted(set(pr_files or ())):
        try:
            with open(os.path.join(repo_dir, rel), "rb") as handle:
                source = handle.read()
        except OSError:
            continue
        if disable_var.encode() in source:
            digest.update(rel.encode())
            digest.update(source)
    return digest.hexdigest()[:16]


def baseline_toggle_is_valid(state, baseline_kind, fingerprint):
    """Whether the shared OFF baseline was measured with this exact toggle code."""
    if fingerprint is None:
        return True
    recorded = (state.get("baselines") or {}).get(
        f"{baseline_kind}_toggle_fingerprint", ""
    )
    return recorded == fingerprint


def build_verdict(
    round_no,
    iteration,
    pr_head_sha,
    perf,
    lm_eval,
    closure,
    nsys_run,
    notes=None,
    mode="review-round",
):
    overall = bool(perf["pass"] and lm_eval["pass"] and closure["pass"])
    return {
        "mode": mode,
        "round": round_no,
        "iteration": iteration,
        "pr_head_sha": pr_head_sha,
        "perf": perf,
        "lm_eval": lm_eval,
        "closure": {
            key: closure[key]
            for key in (
                "pass",
                "total",
                "fixed",
                "explained",
                "unresolved",
                "unresolved_ids",
                "reason",
            )
        },
        "nsys_run": bool(nsys_run),
        "overall": "PASS" if overall else "FAIL",
        "notes": notes or [],
    }


def write_verdict(path, verdict):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(verdict, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def latest_verdict(output_dir):
    """Newest verify_verdict.json under <output_dir>/runtime/V*/."""
    from llm.auto_code_gen.code_gen_prompts import RUNTIME_DIR

    root = os.path.join(output_dir, RUNTIME_DIR)
    if not os.path.isdir(root):
        return None
    found = []
    for name in os.listdir(root):
        match = re.match(r"^V(\d+)$", name)
        candidate = os.path.join(root, name, VERDICT_FILE)
        if match and os.path.isfile(candidate):
            found.append((int(match.group(1)), candidate))
    if not found:
        return None
    return max(found)[1]


def read_verdict(path):
    if not path:
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def latest_runtime_iteration(output_dir):
    """Highest existing <output_dir>/runtime/V<n>, or 0."""
    from llm.auto_code_gen.code_gen_prompts import RUNTIME_DIR

    root = os.path.join(output_dir, RUNTIME_DIR)
    found = [
        int(m.group(1))
        for name in (os.listdir(root) if os.path.isdir(root) else [])
        for m in [re.match(r"^V(\d+)$", name)]
        if m
    ]
    return max(found) if found else 0


def verify_rebase(base_commit, state=None):
    """Verdict-shaped result for a rebase, whose gates have not run yet.

    A rebase always has to be re-measured: the shared OFF baselines were taken
    against the previous base, so they are invalid the moment the base moves.
    """
    invalid = []
    for kind in ("bench_off", "lm_eval_off", "nsys_off"):
        if state is not None and not baseline_is_valid(state, kind, base_commit):
            invalid.append(kind)
    return {
        "mode": "rebase",
        "base_commit": base_commit,
        "overall": "PENDING",
        "stale_baselines": invalid,
        "reason": (
            "shared OFF baseline(s) {} were measured on the previous base and must be "
            "re-measured".format(", ".join(invalid))
            if invalid
            else "baselines already valid for this base"
        ),
    }
