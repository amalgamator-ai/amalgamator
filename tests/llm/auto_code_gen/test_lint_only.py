import json
import shutil
import subprocess

import pytest

from llm.auto_code_gen.pr_flow import lint_only

# ruff from PATH, or a venv install, else assume it is on PATH at run time.
RUFF_BIN = shutil.which("ruff") or "ruff"

# Old/new pairs for the classifier. The ruff-format difference is the exact
# shape of the production lint fix: joining an over-wrapped call.
WRAPPED_OLD = "f(    1)\n"
WRAPPED_NEW = "f(1)\n"
# Semantically different code -- the fixers would never produce it.
WRAPPED_BAD = "g(1)\n"


@pytest.fixture
def git_repo(tmp_path):
    if not shutil.which("git"):
        pytest.skip("git not available")
    repo = tmp_path / "repo"
    repo.mkdir()

    def run(*args):
        return subprocess.run(
            ["git", "-C", str(repo)] + list(args),
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    run("init", "-q", "-b", "perf/branch")
    run("config", "user.email", "dev@example.com")
    run("config", "user.name", "Dev")
    # Minimal stand-in for the repo's ruff config: import sorting is a
    # selected rule (I001), not a default, so the classifier only sees it
    # when the repo's own config asks for it -- same as vLLM's setup.
    (repo / "ruff.toml").write_text('[lint]\nselect = ["I"]\n')
    (repo / "mod.py").write_text("import sys\nimport os\n\nprint(1)\n")
    run("add", "ruff.toml", "mod.py")
    run("commit", "-qm", "base commit")
    return repo, run


def _commit_change(repo, run, filename, content, message="change"):
    (repo / filename).write_text(content)
    run("add", filename)
    run("commit", "-qm", message)


def test_classifier_accepts_formatter_output(git_repo):
    repo, run = git_repo
    old = (repo / "mod.py").read_text()
    new = old.replace("import sys\nimport os\n", "import os\nimport sys\n")
    _commit_change(repo, run, "mod.py", new, "fix(lint): import order")
    prev = run("rev-parse", "HEAD~1").strip()
    kind, detail = lint_only.classify_lint_only(
        str(repo), prev, ["mod.py"], ruff_bin=RUFF_BIN
    )
    assert kind == "lint_only", detail


def test_classifier_accepts_ruff_format_join(git_repo):
    repo, run = git_repo
    _commit_change(repo, run, "extra.py", WRAPPED_OLD, "add wrapped call")
    prev1 = run("rev-parse", "HEAD").strip()
    _commit_change(repo, run, "extra.py", WRAPPED_NEW, "fix(lint): ruff format")
    kind, detail = lint_only.classify_lint_only(
        str(repo), prev1, ["extra.py"], ruff_bin=RUFF_BIN
    )
    assert kind == "lint_only", detail


def test_classifier_rejects_semantic_changes(git_repo):
    repo, run = git_repo
    _commit_change(
        repo, run, "mod.py", "import sys\nimport os\n\nprint(42)\n", "tweak constant"
    )
    prev = run("rev-parse", "HEAD~1").strip()
    kind, detail = lint_only.classify_lint_only(
        str(repo), prev, ["mod.py"], ruff_bin=RUFF_BIN
    )
    assert kind == "not_lint_only", detail
    assert "mod.py" in detail


def test_classifier_rejects_function_rename(git_repo):
    repo, run = git_repo
    _commit_change(repo, run, "extra.py", WRAPPED_OLD, "add wrapped call")
    prev1 = run("rev-parse", "HEAD").strip()
    # Same shape, different function: format-identical but semantically new.
    _commit_change(repo, run, "extra.py", WRAPPED_BAD, "rename function")
    kind, _ = lint_only.classify_lint_only(
        str(repo), prev1, ["extra.py"], ruff_bin=RUFF_BIN
    )
    assert kind == "not_lint_only"


def test_classifier_reports_error_on_missing_ruff(git_repo):
    repo, run = git_repo
    _commit_change(repo, run, "mod.py", "import sys\nimport os\n\nprint(2)\n")
    prev = run("rev-parse", "HEAD~1").strip()
    kind, detail = lint_only.classify_lint_only(
        str(repo), prev, ["mod.py"], ruff_bin="/nonexistent/ruff"
    )
    assert kind == "error"
    assert "ruff" in detail.lower()


def _write_source_verdict(output_dir, iteration, head, actual_pct=1.35):
    """A prior cluster-measured PASS verdict to carry from."""
    iter_dir = output_dir / "runtime" / f"V{iteration}"
    iter_dir.mkdir(parents=True)
    verdict = {
        "mode": "review-round",
        "round": 1,
        "iteration": iteration,
        "pr_head_sha": head,
        "overall": "PASS",
        "nsys_run": True,
        "perf": {
            "pass": True,
            "measured": True,
            "actual_pct": actual_pct,
            "required_pct": 1.08,
        },
        "lm_eval": {"pass": True, "measured": True, "score": 0.955},
        "closure": {
            "pass": True,
            "total": 26,
            "fixed": 26,
            "explained": 0,
            "unresolved": 0,
            "unresolved_ids": [],
            "reason": "",
        },
        "notes": [],
    }
    (iter_dir / "verify_verdict.json").write_text(json.dumps(verdict))
    return verdict


class _Config:
    def __init__(self, output_dir):
        self.output_dir = str(output_dir)


def _state(repo, run, output_dir, base):
    return {
        "repo_dir": str(repo),
        "output_dir": str(output_dir),
        "base_commit": base,
        "head_sha": base,
        "round": 1,
        "gates": [],
    }


def test_record_creates_carried_v17(git_repo, tmp_path):
    repo, run = git_repo
    base = run("rev-parse", "HEAD").strip()
    output_dir = tmp_path / "out"
    state_path = output_dir / "pr_p3" / "pr_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps(_state(repo, run, output_dir, base)))
    src = _write_source_verdict(output_dir, 16, base)

    # A formatter-only commit on top of the measured tree.
    old = (repo / "mod.py").read_text()
    new = old.replace("import sys\nimport os\n", "import os\nimport sys\n")
    _commit_change(repo, run, "mod.py", new, "fix(lint): import order")
    head = run("rev-parse", "HEAD").strip()

    iteration = lint_only.record_lint_only_iteration(
        _Config(output_dir), "pre-commit fixes", ruff_bin=RUFF_BIN
    )
    assert iteration == 17

    verdict = json.loads(
        (output_dir / "runtime" / "V17" / "verify_verdict.json").read_text()
    )
    assert verdict["overall"] == "PASS"
    assert verdict["lint_only"] is True
    assert verdict["perf"]["measured"] is False
    assert verdict["perf"]["actual_pct"] == src["perf"]["actual_pct"]
    assert verdict["perf"]["required_pct"] == src["perf"]["required_pct"]
    assert verdict["lint_only_proof"]["carried_from"] == "V16"
    assert verdict["lint_only_proof"]["verified_against"] != head

    # Fresh fingerprint at the new HEAD covering the PR's files.
    measured = json.loads(
        (output_dir / "runtime" / "V17" / "measured_content.json").read_text()
    )
    assert measured["pr_head_sha"] == head
    assert "mod.py" in measured["files"]

    # Gate trail + state updated.
    state = json.loads(state_path.read_text())
    assert state["gates"][-1]["stage"] == "lint-only"
    assert state["head_sha"] == head
    assert state["iteration"] == 17


def test_record_refuses_semantic_change(git_repo, tmp_path):
    repo, run = git_repo
    base = run("rev-parse", "HEAD").strip()
    output_dir = tmp_path / "out"
    state_path = output_dir / "pr_p3" / "pr_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps(_state(repo, run, output_dir, base)))
    _write_source_verdict(output_dir, 16, base)

    _commit_change(
        repo, run, "mod.py", "import sys\nimport os\n\nprint(99)\n", "change behavior"
    )
    with pytest.raises(SystemExit, match="not_lint_only"):
        lint_only.record_lint_only_iteration(
            _Config(output_dir), "test", ruff_bin=RUFF_BIN
        )
    assert not (output_dir / "runtime" / "V17").exists()


def test_record_refuses_when_no_measured_verdict(git_repo, tmp_path):
    repo, run = git_repo
    base = run("rev-parse", "HEAD").strip()
    output_dir = tmp_path / "out"
    state_path = output_dir / "pr_p3" / "pr_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps(_state(repo, run, output_dir, base)))

    old = (repo / "mod.py").read_text()
    new = old.replace("import sys\nimport os\n", "import os\nimport sys\n")
    _commit_change(repo, run, "mod.py", new, "fix(lint): import order")
    with pytest.raises(SystemExit, match="cluster-measured"):
        lint_only.record_lint_only_iteration(
            _Config(output_dir), "test", ruff_bin=RUFF_BIN
        )


def test_record_refuses_non_python_diff(git_repo, tmp_path):
    repo, run = git_repo
    base = run("rev-parse", "HEAD").strip()
    output_dir = tmp_path / "out"
    state_path = output_dir / "pr_p3" / "pr_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps(_state(repo, run, output_dir, base)))
    _write_source_verdict(output_dir, 16, base)

    (repo / "notes.md").write_text("# hello\n")
    run("add", "notes.md")
    run("commit", "-qm", "docs")
    with pytest.raises(SystemExit, match="not lint-only"):
        lint_only.record_lint_only_iteration(
            _Config(output_dir), "test", ruff_bin=RUFF_BIN
        )


def test_record_accepts_ci_wiring_addition(git_repo, tmp_path):
    repo, run = git_repo
    base = run("rev-parse", "HEAD").strip()
    output_dir = tmp_path / "out"
    state_path = output_dir / "pr_p3" / "pr_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps(_state(repo, run, output_dir, base)))
    _write_source_verdict(output_dir, 16, base)

    # A cosmetic Python fix plus a Buildkite yaml that only adds the test path.
    tests_dir = repo / "tests"
    tests_dir.mkdir(exist_ok=True)
    (tests_dir / "test_mod.py").write_text("def test_mod():\n    assert True\n")
    bk = repo / ".buildkite" / "test_areas" / "models_basic.yaml"
    bk.parent.mkdir(parents=True)
    bk.write_text("- label: CPU job\n  commands:\n    - pytest -v -s mod.py\n")
    run("add", "tests", ".buildkite")
    run("commit", "-qm", "add test and CI job")

    old = (repo / "mod.py").read_text()
    new = old.replace("import sys\nimport os\n", "import os\nimport sys\n")
    (repo / "mod.py").write_text(new)
    # Addition-only wiring edit: one new line referencing the existing test.
    bk.write_text(
        "- label: CPU job\n  commands:\n    - pytest -v -s mod.py\n"
        "    - pytest -v -s tests/test_mod.py\n"
    )
    run("add", "mod.py", ".buildkite")
    run("commit", "-qm", "fix(lint): ruff + wire test")
    run("rev-parse", "HEAD").strip()

    lint_only.record_lint_only_iteration(
        _Config(output_dir), "pre-commit fixes", ruff_bin=RUFF_BIN
    )
    verdict = json.loads(
        (output_dir / "runtime" / "V17" / "verify_verdict.json").read_text()
    )
    assert verdict["lint_only_proof"]["files"] == ["mod.py"]
    assert verdict["lint_only_proof"]["ci_wiring_files"] == [
        ".buildkite/test_areas/models_basic.yaml"
    ]


def test_record_refuses_ci_wiring_removal(git_repo, tmp_path):
    repo, run = git_repo
    base = run("rev-parse", "HEAD").strip()
    output_dir = tmp_path / "out"
    state_path = output_dir / "pr_p3" / "pr_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps(_state(repo, run, output_dir, base)))
    _write_source_verdict(output_dir, 16, base)

    bk = repo / ".buildkite" / "test_areas" / "models_basic.yaml"
    bk.parent.mkdir(parents=True)
    bk.write_text(
        "- label: CPU job\n  commands:\n    - pytest -v -s mod.py\n"
        "    - pytest -v -s missing_test.py\n"
    )
    run("add", ".buildkite")
    run("commit", "-qm", "add wiring")
    # Now remove a line -- not an addition-only edit.
    bk.write_text("- label: CPU job\n  commands:\n    - pytest -v -s mod.py\n")
    run("add", ".buildkite")
    run("commit", "-qm", "drop wiring")
    with pytest.raises(SystemExit, match="not lint-only"):
        lint_only.record_lint_only_iteration(
            _Config(output_dir), "test", ruff_bin=RUFF_BIN
        )


def test_record_refuses_ci_wiring_pointing_at_missing_test(git_repo, tmp_path):
    repo, run = git_repo
    base = run("rev-parse", "HEAD").strip()
    output_dir = tmp_path / "out"
    state_path = output_dir / "pr_p3" / "pr_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps(_state(repo, run, output_dir, base)))
    _write_source_verdict(output_dir, 16, base)

    bk = repo / ".buildkite" / "test_areas" / "models_basic.yaml"
    bk.parent.mkdir(parents=True)
    # References a test path that does not exist on disk.
    bk.write_text(
        "- label: CPU job\n  commands:\n"
        "    - pytest -v -s tests/nope/test_missing.py\n"
    )
    run("add", ".buildkite")
    run("commit", "-qm", "wire missing test")
    with pytest.raises(SystemExit, match="not lint-only"):
        lint_only.record_lint_only_iteration(
            _Config(output_dir), "test", ruff_bin=RUFF_BIN
        )


def test_record_refuses_failing_source_verdict(git_repo, tmp_path):
    repo, run = git_repo
    base = run("rev-parse", "HEAD").strip()
    output_dir = tmp_path / "out"
    state_path = output_dir / "pr_p3" / "pr_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps(_state(repo, run, output_dir, base)))
    src = _write_source_verdict(output_dir, 16, base)
    src["overall"] = "FAIL"
    src["perf"]["pass"] = False
    (output_dir / "runtime" / "V16" / "verify_verdict.json").write_text(json.dumps(src))

    old = (repo / "mod.py").read_text()
    new = old.replace("import sys\nimport os\n", "import os\nimport sys\n")
    _commit_change(repo, run, "mod.py", new, "fix(lint): import order")
    with pytest.raises(SystemExit, match="not PASS"):
        lint_only.record_lint_only_iteration(
            _Config(output_dir), "test", ruff_bin=RUFF_BIN
        )
