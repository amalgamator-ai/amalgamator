"""The OFF arm must be measured, never assumed.

These cover two ways a rebased tree used to be graded against numbers from the
previous base: a failed OFF run that still stamped the base as measured, and a
successful re-measurement whose files were silently dropped because the shared
baseline refuses to overwrite.
"""

import json
import os

from llm.auto_code_gen.code_gen_prompts import RUNTIME_DIR
from llm.auto_code_gen.pr_flow.run_pr_review_iters import _baseline_run_problem
from llm.auto_code_gen.run_code_gen import (
    SHARED_BASELINES_DIR,
    _has_shared_baseline,
    _retire_shared_baseline,
    _save_shared_baseline,
    _shared_baseline_dir,
)


def _write_bench_result(run_dir, throughput=119.5, exit_code="0"):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "baseline_exit_code.txt").write_text(exit_code + "\n")
    (run_dir / "baseline_bench_result.json").write_text(
        json.dumps({"output_throughput": throughput})
    )
    (run_dir / "baseline_combined.log").write_text("bench ran\n")


def test_failed_off_run_is_rejected(tmp_path):
    run = tmp_path / "baseline_run"
    run.mkdir()
    problem = _baseline_run_problem(str(run), "bench_off", 1, "SERVER_START_FAILED")
    assert problem is not None and "SERVER_START_FAILED" in problem


def test_off_run_that_started_nothing_is_rejected(tmp_path):
    """exit 0 with no artifacts is what a half-failed stage job looks like."""
    run = tmp_path / "baseline_run"
    run.mkdir()
    problem = _baseline_run_problem(str(run), "bench_off", 0, "")
    assert problem is not None and "baseline_exit_code.txt" in problem


def test_off_run_with_unusable_throughput_is_rejected(tmp_path):
    run = tmp_path / "baseline_run"
    _write_bench_result(run, throughput=0)
    problem = _baseline_run_problem(str(run), "bench_off", 0, "COMPLETED")
    assert problem is not None and "output_throughput" in problem


def test_nonzero_exit_code_inside_the_run_dir_is_rejected(tmp_path):
    run = tmp_path / "baseline_run"
    _write_bench_result(run, exit_code="1")
    problem = _baseline_run_problem(str(run), "bench_off", 0, "COMPLETED")
    assert problem is not None


def test_healthy_off_run_is_accepted(tmp_path):
    run = tmp_path / "baseline_run"
    _write_bench_result(run)
    assert _baseline_run_problem(str(run), "bench_off", 0, "COMPLETED") is None


def _output_dir(tmp_path):
    out = tmp_path / "out"
    (out / RUNTIME_DIR / SHARED_BASELINES_DIR / "bench_off").mkdir(parents=True)
    return out


def test_retiring_lets_a_new_base_baseline_replace_the_old_one(tmp_path):
    """Without retiring, the fresh files are dropped and the old numbers stand."""
    out = _output_dir(tmp_path)
    old_run = tmp_path / "old_run"
    _write_bench_result(old_run, throughput=119.5)
    _save_shared_baseline(str(out), str(old_run), "bench_off")
    shared = _shared_baseline_dir(str(out), "bench_off")
    assert (
        "119.5"
        in (
            tmp_path
            / "out"
            / RUNTIME_DIR
            / SHARED_BASELINES_DIR
            / "bench_off"
            / "baseline_bench_result.json"
        ).read_text()
    )
    assert _has_shared_baseline(str(out), "bench_off")

    new_run = tmp_path / "new_run"
    _write_bench_result(new_run, throughput=118.2)
    _retire_shared_baseline(str(out), "bench_off", reason="re-measured")
    _save_shared_baseline(str(out), str(new_run), "bench_off")

    recorded = json.loads(
        (
            tmp_path
            / "out"
            / RUNTIME_DIR
            / SHARED_BASELINES_DIR
            / "bench_off"
            / "baseline_bench_result.json"
        ).read_text()
    )
    assert recorded["output_throughput"] == 118.2
    assert shared.endswith("bench_off")


def test_retiring_keeps_the_previous_numbers_for_audit(tmp_path):
    out = _output_dir(tmp_path)
    old_run = tmp_path / "old_run"
    _write_bench_result(old_run, throughput=119.5)
    _save_shared_baseline(str(out), str(old_run), "bench_off")
    aside = _retire_shared_baseline(str(out), "bench_off", reason="re-measured")
    assert aside and os.path.isdir(aside) and ".retired-" in os.path.basename(aside)
    with open(os.path.join(aside, "baseline_bench_result.json")) as f:
        stored = json.load(f)
    assert stored["output_throughput"] == 119.5
    assert not os.path.isdir(_shared_baseline_dir(str(out), "bench_off"))


def test_retiring_an_absent_baseline_is_a_no_op(tmp_path):
    out = tmp_path / "nothing"
    (out / RUNTIME_DIR).mkdir(parents=True)
    assert _retire_shared_baseline(str(out), "bench_off") is None


def _nsys_trace(output_dir):
    dst = output_dir / RUNTIME_DIR / SHARED_BASELINES_DIR / "nsys_off"
    dst.mkdir(parents=True)
    (dst / ".done").write_text("1\n")
    (dst / "trace_baseline.sqlite").write_text("trace\n")
    return dst


def _state(tmp_path, baselines):
    path = tmp_path / "pr_state.json"
    path.write_text(json.dumps({"baselines": baselines}))
    return {"_path": str(path), "baselines": baselines}


def test_nsys_trace_from_another_base_is_retired(tmp_path):
    from llm.auto_code_gen.pr_flow.run_pr_review_iters import _prepare_nsys_baseline

    _nsys_trace(tmp_path)
    state = _state(tmp_path, {"nsys_off_base_commit": "oldbase"})

    assert _prepare_nsys_baseline(str(tmp_path), state, "newbase") is True
    assert not _has_shared_baseline(str(tmp_path), "nsys_off")
    retired = list(
        (tmp_path / RUNTIME_DIR / SHARED_BASELINES_DIR).glob("nsys_off.retired-*")
    )
    assert len(retired) == 1


def test_nsys_trace_never_measured_is_treated_as_stale(tmp_path):
    """An unstamped trace is not trusted, matching baseline_is_valid."""
    from llm.auto_code_gen.pr_flow.run_pr_review_iters import _prepare_nsys_baseline

    _nsys_trace(tmp_path)
    assert (
        _prepare_nsys_baseline(str(tmp_path), _state(tmp_path, {}), "newbase") is True
    )
    assert not _has_shared_baseline(str(tmp_path), "nsys_off")


def test_nsys_trace_for_the_current_base_is_kept(tmp_path):
    from llm.auto_code_gen.pr_flow.run_pr_review_iters import _prepare_nsys_baseline

    _nsys_trace(tmp_path)
    state = _state(tmp_path, {"nsys_off_base_commit": "newbase"})
    assert _prepare_nsys_baseline(str(tmp_path), state, "newbase") is False
    assert _has_shared_baseline(str(tmp_path), "nsys_off")


def test_measured_nsys_trace_is_stamped_with_its_base(tmp_path):
    from llm.auto_code_gen.pr_flow.run_pr_review_iters import (
        _prepare_nsys_baseline,
        _stamp_nsys_baseline,
    )

    state = _state(tmp_path, {})
    _stamp_nsys_baseline(str(tmp_path), state, "newbase")  # nothing on disk
    assert (
        "nsys_off_base_commit"
        not in json.loads((tmp_path / "pr_state.json").read_text())["baselines"]
    )

    _nsys_trace(tmp_path)
    _stamp_nsys_baseline(str(tmp_path), state, "newbase")
    stamped = json.loads((tmp_path / "pr_state.json").read_text())["baselines"]
    assert stamped["nsys_off_base_commit"] == "newbase"
    assert _prepare_nsys_baseline(str(tmp_path), state, "newbase") is False


TOGGLE = "VLLM_DISABLE_THING"


def _repo(tmp_path, gated_extra="", other_extra=""):
    repo = tmp_path / "repo"
    gated = repo / "vllm" / "gated.py"
    other = repo / "vllm" / "other.py"
    gated.parent.mkdir(parents=True, exist_ok=True)
    gated.write_text("import os\nENABLED = os.getenv('%s')\n" % TOGGLE + gated_extra)
    other.write_text("VALUE = 1\n" + other_extra)
    return repo


def test_toggle_fingerprint_follows_the_switch_code(tmp_path):
    from llm.auto_code_gen.pr_flow import gates

    repo = _repo(tmp_path)
    files = ["vllm/gated.py", "vllm/other.py"]
    base = gates.toggle_fingerprint(str(repo), TOGGLE, files)
    assert base and base == gates.toggle_fingerprint(str(repo), TOGGLE, files)

    (repo / "vllm" / "gated.py").write_text(
        (repo / "vllm" / "gated.py").read_text() + "\nEXTRA = 1\n"
    )
    assert gates.toggle_fingerprint(str(repo), TOGGLE, files) != base


def test_toggle_fingerprint_ignores_files_that_do_not_read_the_switch(tmp_path):
    from llm.auto_code_gen.pr_flow import gates

    repo = _repo(tmp_path)
    files = ["vllm/gated.py", "vllm/other.py"]
    base = gates.toggle_fingerprint(str(repo), TOGGLE, files)
    (repo / "vllm" / "other.py").write_text("VALUE = 2\n")
    assert gates.toggle_fingerprint(str(repo), TOGGLE, files) == base


def test_reference_is_rejected_when_the_toggle_reach_changes(tmp_path):
    from llm.auto_code_gen.pr_flow import gates

    repo = _repo(tmp_path)
    files = ["vllm/gated.py"]
    before = gates.toggle_fingerprint(str(repo), TOGGLE, files)
    state = {
        "baselines": {
            "bench_off_base_commit": "abc",
            "bench_off_toggle_fingerprint": before,
        }
    }
    assert gates.baseline_toggle_is_valid(state, "bench_off", before) is True
    assert gates.baseline_toggle_is_valid(state, "bench_off", "other") is False
    # An unstamped reference cannot be claimed to match the current toggle.
    assert (
        gates.baseline_toggle_is_valid(
            {"baselines": {"bench_off_base_commit": "abc"}}, "bench_off", before
        )
        is False
    )
    # Callers that do not track the toggle keep the previous base-only rule.
    assert gates.baseline_toggle_is_valid(state, "bench_off", None) is True
