# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Wiring tests for the external resolver context.

The parser/renderer tests live in ``test_external_context.py``; these tests
cover the pipeline seams the renderers are injected into.  A dropped
``external_context_section=`` kwarg or a typo'd phase name is otherwise
silent: the prompt simply falls back to "".
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock

from auto_bug_fix import run_bug_fix as rbf
from auto_bug_fix.bug_fix_config import BugFixConfig
from auto_bug_fix.claude_utils import BugFixClaudeConfig
from auto_bug_fix.external_context import load_external_context
from auto_bug_fix.run_bug_fix import PipelineState

CONTEXT_MD = """# External Resolver Context

## Metadata
- resolver: synthmerge
- checker: check-backport.py
- findings-count: 1

## Suspect findings
- [advisory][dead_symbol][inferred/low] a.c: symbol 'dead' declared but has zero references

## Resolved files

### a.c
```diff
diff --git a/a.c b/a.c
+resolved-marker
```
"""


def _repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    (repo / "a.c").write_text("int a;\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-q",
            "-m",
            "base",
        ],
        check=True,
    )
    return repo


def _context_file(tmp_path):
    path = tmp_path / "context.md"
    path.write_text(CONTEXT_MD)
    return path


def _config(repo, context_path, *, load=True):
    config = BugFixConfig(
        repo_path=str(repo),
        build_dir=str(repo),
        source_branch="main",
        target_branch="main",
        source_fix_commit="fix",
        bug_description="test",
        issue_id="CTX-1",
        build_command="",
        test_command="",
        external_context_file=str(context_path),
        max_resolution_retries=1,
    )
    if load:
        config.external_context = load_external_context(str(context_path))
    return config


def _loaded(config):
    ctx = config.external_context
    assert ctx is not None
    return ctx


def _claude_config(repo):
    return BugFixClaudeConfig(
        model="test-model",
        allowed_tools=["Read", "Write", "Bash"],
        perm_mode="acceptEdits",
        cwd=str(repo),
    )


def _capture_prompts(monkeypatch):
    prompts = []

    async def fake_claude_run(claude_config, prompt_list, tracker=None):
        prompts.append(prompt_list[0])
        return []

    monkeypatch.setattr(rbf, "claude_run", fake_claude_run)
    return prompts


def test_section_helper_inactive_and_unknown_phase(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    context_path = _context_file(tmp_path)

    inactive = _config(repo, context_path, load=False)
    assert rbf._external_context_section(inactive, "verification") == ""

    active = _config(repo, context_path)
    assert rbf._external_context_section(active, "verification") == (
        _loaded(active).render_verification_section()
    )
    assert rbf._external_context_section(active, "verfication") == ""


def test_triage_site_injects_section(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    config = _config(repo, _context_file(tmp_path))
    monkeypatch.setattr(rbf, "is_merge_commit", lambda *a, **k: False)
    prompts = _capture_prompts(monkeypatch)

    state = PipelineState()
    state.seed = ["a.c"]
    rbf.phase_0_5_semantic_triage(config, _claude_config(repo), state, MagicMock())

    assert _loaded(config).render_triage_section() in prompts[0]


def test_resolution_site_injects_section(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    config = _config(repo, _context_file(tmp_path))

    class _Result:
        success = False
        exit_code = 1
        stderr = ""

    statuses = iter([["UU a.c"], ["UU a.c"], ["UU a.c"], ["UU a.c"], []])
    monkeypatch.setattr(rbf, "git_cherry_pick", lambda *a, **k: _Result())
    monkeypatch.setattr(rbf, "git_status_porcelain", lambda *a, **k: next(statuses))
    monkeypatch.setattr(rbf, "git_cherry_pick_abort", lambda *a, **k: None)
    monkeypatch.setattr(rbf, "git_rev_parse", lambda *a, **k: "base")
    monkeypatch.setattr(rbf, "_compute_fix_diff", lambda *a, **k: "diff")
    monkeypatch.setattr(rbf, "create_context_str", lambda *a, **k: "ctx")
    monkeypatch.setattr(rbf, "resolve_mechanically", lambda *a, **k: {})
    monkeypatch.setattr(rbf, "_finalize_resolution", lambda *a, **k: "resolved")
    prompts = _capture_prompts(monkeypatch)

    state = PipelineState()
    state.allowed_modules = ["a.c"]
    result = rbf.cherry_pick_and_resolve(
        config, _claude_config(repo), state, MagicMock(), "fix", "label"
    )

    assert result == "resolved"
    expected = _loaded(config).render_resolution_section(files=["a.c"])
    assert expected in prompts[0]


def test_verification_and_review_sites_inject_sections(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    config = _config(repo, _context_file(tmp_path))
    config.evidence_checks = False
    config.post_resolution_check = True
    config.blind_review = True

    monkeypatch.setattr(rbf, "git_rev_parse", lambda *a, **k: "head")
    monkeypatch.setattr(rbf, "git_diff_name_only_range", lambda *a, **k: ["a.c"])
    monkeypatch.setattr(rbf, "git_diff_name_only", lambda *a, **k: "")
    monkeypatch.setattr(rbf, "snap_back_files", lambda *a, **k: {})
    monkeypatch.setattr(rbf, "git_status_porcelain", lambda *a, **k: "")
    monkeypatch.setattr(rbf, "git_commit", lambda *a, **k: None)
    monkeypatch.setattr(rbf, "_is_followup_target", lambda *a, **k: False)
    monkeypatch.setattr(rbf, "render_evidence", lambda *a, **k: "evidence")
    monkeypatch.setattr(rbf, "_safe_git_diff", lambda *a, **k: "diff")
    prompts = _capture_prompts(monkeypatch)

    result = rbf._finalize_resolution(
        config,
        _claude_config(repo),
        PipelineState(),
        None,
        "fix",
        "label",
        [],
        "base",
        "fix-diff",
        "ctx",
    )

    assert result == "resolved"
    assert _loaded(config).render_verification_section() in prompts[0]
    assert _loaded(config).render_review_section() in prompts[1]


def test_run_pipeline_loads_the_context_file(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    context_path = _context_file(tmp_path)
    config = _config(repo, context_path, load=False)
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    captured = []

    def fake_phases(config, claude_config, state, tracker):
        captured.append(config)
        return state.dossier, tracker

    monkeypatch.setattr(rbf, "_run_pipeline_phases", fake_phases)

    rbf.run_pipeline(config, _claude_config(repo), output_dir=str(output_dir))

    assert captured[0].external_context is not None
    assert captured[0].external_context.active
    assert captured[0].external_context.source_path == str(context_path)
