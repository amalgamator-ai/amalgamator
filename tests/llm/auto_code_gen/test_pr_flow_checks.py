import json
import shutil
import subprocess

import pytest

from llm.auto_code_gen.pr_flow import checks


STATE = {
    "repo_dir": "/tmp/unused",
    "output_dir": "/tmp/unused",
    "upstream_repo": "vllm-project/vllm",
    "upstream_branch": "main",
    "base_branch": "main",
    "local_branch": "local-perf-branch",
    "branch_prefix": "perf/",
    "fork_remote": "fork",
    "fork_repo": "example-dev/vllm",
    "fork_owner": "example-dev",
    "head_sha": "",
}


def _rejected(fn, *args, code=None, match="", **kwargs):
    with pytest.raises(checks.CheckError) as excinfo:
        fn(*args, **kwargs)
    if code is not None:
        assert excinfo.value.code == code, excinfo.value
    if match:
        assert match in str(excinfo.value)
    return excinfo.value


def test_protected_and_self_targeting_branch_names_are_refused():
    for name in ("main", "master", "release/0.9", "develop"):
        _rejected(checks.check_branch_name, STATE, name, code=checks.EXIT_BRANCH)
    base_only = dict(STATE, upstream_branch="integration", base_branch="integration")
    _rejected(checks.check_branch_name, base_only, "integration", code=checks.EXIT_BRANCH,
              match="base branch")
    _rejected(checks.check_branch_name, STATE, "local-perf-branch", code=checks.EXIT_BRANCH)


def test_branch_name_rules():
    _rejected(checks.check_branch_name, STATE, "no-prefix-here", code=checks.EXIT_BRANCH,
              match="branch_prefix")
    _rejected(checks.check_branch_name, STATE, "perf/WithCaps", code=checks.EXIT_BRANCH)
    _rejected(checks.check_branch_name, STATE, "perf/a..b", code=checks.EXIT_BRANCH)
    _rejected(checks.check_branch_name, STATE, "perf/-lead", code=checks.EXIT_BRANCH)
    _rejected(checks.check_branch_name, STATE, "perf//double", code=checks.EXIT_BRANCH)
    _rejected(checks.check_branch_name, STATE, "", code=checks.EXIT_BRANCH)
    assert checks.check_branch_name(STATE, "perf/example-low-latency-gemm")
    assert checks.check_branch_name(STATE, "no-prefix-ok", allow_prefixless=True)


def test_internal_strings_are_detected_in_text():
    for sample in ("/home/alex/workspace/x", "quay.io/internal/image:1",
                   "applied by auto_code_gen", "Co-Authored-By: Claude <x@y>",
                   "Generated with an AI assistant"):
        assert checks._scan_for_internal(sample), sample
    assert checks._scan_for_internal("Enable the toggle for lower decode latency.") == []


def test_inbox_guardrail(tmp_path):
    _rejected(checks.check_inbox, str(tmp_path), code=checks.EXIT_NOTHING)
    inbox = tmp_path / "round_1"
    inbox.mkdir()
    (inbox / "pending_comments.json").write_text(json.dumps({"pending": []}))
    _rejected(checks.check_inbox, str(inbox), code=checks.EXIT_NOTHING)
    (inbox / "pending_comments.json").write_text(json.dumps(
        {"pending": [{"id": 1}, {"id": 2}]}))
    assert "2 pending" in checks.check_inbox(str(inbox))


def test_descs_guardrail(tmp_path):
    state = dict(STATE, output_dir=str(tmp_path))
    _rejected(checks.check_descs, state, code=checks.EXIT_DESCS)

    descs = tmp_path / "pr-generate"
    descs.mkdir(parents=True)
    (descs / "commit_desc.txt").write_text(
        "Add a low-latency GEMM toggle\n\nbody line\nand another\n")
    (descs / "pr_desc.txt").write_text("## Summary\n\nAdds the toggle.\n")
    assert "descriptions OK" in checks.check_descs(state)

    (descs / "pr_desc.txt").write_text("## Summary\n\nSee /home/alex/notes.md\n")
    _rejected(checks.check_descs, state, code=checks.EXIT_DESCS, match="leaks")


@pytest.fixture
def git_repo(tmp_path):
    if not shutil.which("git"):
        pytest.skip("git not available")
    repo = tmp_path / "repo"
    repo.mkdir()

    def run(*args):
        return subprocess.run(["git", "-C", str(repo)] + list(args),
                              check=True, capture_output=True, text=True).stdout

    run("init", "-q", "-b", "perf/branch")
    run("config", "user.email", "dev@example.com")
    run("config", "user.name", "Dev")
    (repo / "a.py").write_text("print(1)\n")
    run("add", "a.py")
    run("commit", "-qm", "base commit")
    return repo, run


def test_clean_tree_tracks_state_head(git_repo):
    repo, run = git_repo
    base = run("rev-parse", "HEAD").strip()
    state = dict(STATE, repo_dir=str(repo), local_branch="perf/branch", head_sha=base)
    assert "tree clean" in checks.check_clean_tree(state)

    _rejected(checks.check_clean_tree, state, expect_ahead=True, code=checks.EXIT_TREE,
              match="nothing new")

    (repo / "b.py").write_text("print(2)\n")
    run("add", "b.py")
    run("commit", "-qm", "review fix")
    head = run("rev-parse", "HEAD").strip()

    assert "1 new commit" in checks.check_clean_tree(state, expect_ahead=True)
    with pytest.raises(checks.CheckError) as excinfo:
        checks.check_clean_tree(state)
    assert excinfo.value.code == checks.EXIT_TREE

    (repo / "b.py").write_text("print(3)\n")
    with pytest.raises(checks.CheckError) as excinfo:
        checks.check_clean_tree(dict(state, head_sha=head), expect_ahead=True)
    assert excinfo.value.code == checks.EXIT_TREE
    assert "not clean" in str(excinfo.value)


def test_clean_tree_detects_a_rebase(git_repo):
    repo, run = git_repo
    (repo / "b.py").write_text("print(2)\n")
    run("add", "b.py")
    run("commit", "-qm", "review fix")
    pushed = run("rev-parse", "HEAD").strip()
    run("commit", "--amend", "-qm", "rewritten")
    state = dict(STATE, repo_dir=str(repo), local_branch="perf/branch", head_sha=pushed)
    with pytest.raises(checks.CheckError) as excinfo:
        checks.check_clean_tree(state, expect_ahead=True)
    assert excinfo.value.code == checks.EXIT_TREE
    assert "descendant" in str(excinfo.value)


def test_clean_tree_detects_the_wrong_branch(git_repo):
    repo, run = git_repo
    run("checkout", "-q", "-b", "something-else")
    state = dict(STATE, repo_dir=str(repo), local_branch="perf/branch",
                 head_sha=run("rev-parse", "HEAD").strip())
    with pytest.raises(checks.CheckError) as excinfo:
        checks.check_clean_tree(state)
    assert "expects" in str(excinfo.value)


def _dco_state(git_repo):
    repo, run = git_repo
    base = run("rev-parse", "HEAD").strip()
    return dict(STATE, repo_dir=str(repo), base_commit=base)


def test_dco_passes_when_every_commit_is_signed(git_repo):
    repo, run = git_repo
    state = _dco_state(git_repo)
    (repo / "b.py").write_text("print(2)\n")
    run("add", "b.py")
    run("commit", "-qm", "signed change\n\nSigned-off-by: Dev <dev@example.com>")
    assert "DCO OK" in checks.check_dco(state)


def test_dco_rejects_unsigned_commit(git_repo):
    repo, run = git_repo
    state = _dco_state(git_repo)
    (repo / "b.py").write_text("print(2)\n")
    run("add", "b.py")
    run("commit", "-qm", "unsigned change")
    err = _rejected(checks.check_dco, state, code=checks.EXIT_DCO,
                    match="rebase --signoff")
    assert "Signed-off-by: Dev <dev@example.com>" in str(err)


def test_dco_rejects_mismatched_signoff_identity(git_repo):
    repo, run = git_repo
    state = _dco_state(git_repo)
    (repo / "b.py").write_text("print(2)\n")
    run("add", "b.py")
    # Signed, but by a different identity than the author
    run("commit", "-qm", "mismatched signoff\n\nSigned-off-by: Other <other@example.com>")
    _rejected(checks.check_dco, state, code=checks.EXIT_DCO,
              match="needs 'Signed-off-by: Dev <dev@example.com>'")


def test_dco_checks_every_commit_not_just_the_tip(git_repo):
    repo, run = git_repo
    state = _dco_state(git_repo)
    (repo / "b.py").write_text("print(2)\n")
    run("add", "b.py")
    run("commit", "-qm", "unsigned change")
    (repo / "c.py").write_text("print(3)\n")
    run("add", "c.py")
    run("commit", "-qm", "signed on top\n\nSigned-off-by: Dev <dev@example.com>")
    _rejected(checks.check_dco, state, code=checks.EXIT_DCO,
              match="1 commit(s) missing")


def test_dco_requires_base_commit_in_state(git_repo):
    state = dict(STATE, repo_dir="/tmp/unused", base_commit="")
    _rejected(checks.check_dco, state, code=checks.EXIT_STATE)
