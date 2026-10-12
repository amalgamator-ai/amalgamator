"""PR description refresh: success-proof discovery + post-PASS hook wiring.

update_pr_descs / refresh_after_pass keep pr-generate/commit_desc.txt and
pr_desc.txt in sync with the latest passing runtime iteration, so the publish
scripts (step 2 / step 4) never carry stale numbers. These tests cover
find_success_proof and the refresh path with a stubbed agent.
"""

import json
import os

import pytest

from llm.auto_code_gen.pr_flow import pr_descs


def _write_verdict(root, number, overall, mode="rebase"):
    vdir = root / "runtime" / f"V{number}"
    vdir.mkdir(parents=True)
    (vdir / "verify_verdict.json").write_text(
        json.dumps(
            {
                "iteration": number,
                "overall": overall,
                "mode": mode,
            }
        )
    )
    return str(vdir / "verify_verdict.json")


def _write_success(root):
    runtime = root / "runtime"
    runtime.mkdir(parents=True)
    path = runtime / "success_result.txt"
    path.write_text("PASS\n")
    return str(path)


class TestFindSuccessProof:

    def test_success_result_file_wins(self, tmp_path):
        root = tmp_path / "run"
        root.mkdir()
        success = _write_success(root)
        _write_verdict(root, 3, "PASS")
        assert pr_descs.find_success_proof(str(root)) == success

    def test_failing_verdicts_do_not_count(self, tmp_path):
        root = tmp_path / "run"
        root.mkdir()
        _write_verdict(root, 2, "FAIL")
        _write_verdict(root, 3, "FAIL")
        with pytest.raises(pr_descs.PrDescsError) as excinfo:
            pr_descs.find_success_proof(str(root))
        assert "no passing result" in str(excinfo.value)

    def test_latest_passing_verdict_wins(self, tmp_path):
        root = tmp_path / "run"
        root.mkdir()
        v2 = _write_verdict(root, 2, "PASS")
        _write_verdict(root, 3, "FAIL")
        v4 = _write_verdict(root, 4, "PASS")
        assert pr_descs.find_success_proof(str(root)) == v4
        assert v4 != v2

    def test_newest_passing_is_preferred_over_older_passing(self, tmp_path):
        root = tmp_path / "run"
        root.mkdir()
        _write_verdict(root, 2, "PASS")
        newest = _write_verdict(root, 5, "PASS")
        assert pr_descs.find_success_proof(str(root)) == newest

    def test_corrupt_verdict_is_ignored(self, tmp_path):
        root = tmp_path / "run"
        root.mkdir()
        vdir = root / "runtime" / "V1"
        vdir.mkdir(parents=True)
        (vdir / "verify_verdict.json").write_text("{not json")
        with pytest.raises(pr_descs.PrDescsError):
            pr_descs.find_success_proof(str(root))

    def test_no_runtime_dir_at_all(self, tmp_path):
        root = tmp_path / "run"
        root.mkdir()
        with pytest.raises(pr_descs.PrDescsError):
            pr_descs.find_success_proof(str(root))

    def test_non_iteration_dirs_are_skipped(self, tmp_path):
        root = tmp_path / "run"
        root.mkdir()
        (root / "runtime" / "baselines").mkdir(parents=True)
        (root / "runtime" / "baselines" / "verify_verdict.json").write_text(
            json.dumps({"overall": "PASS"})
        )
        with pytest.raises(pr_descs.PrDescsError):
            pr_descs.find_success_proof(str(root))


class TestUpdatePrDescs:

    @pytest.fixture
    def stub_agent(self, monkeypatch):
        """Replace agent_run with a stub that writes the two desc files.

        Returns (steps_run, output_dir_holder) so each test can point the stub
        at its own output_dir before invoking update_pr_descs.
        """
        state = {"steps": [], "output_dir": ""}

        async def fake_agent_run(claude_config, steps, resume=False, client=None):
            state["steps"].extend(steps)
            pr_dir = os.path.join(state["output_dir"], "pr-generate")
            os.makedirs(pr_dir, exist_ok=True)
            with open(os.path.join(pr_dir, "commit_desc.txt"), "w") as f:
                f.write("perf: stub title\n\nstub body\n")
            with open(os.path.join(pr_dir, "pr_desc.txt"), "w") as f:
                f.write("## Summary\n\nstub\n")
            return []

        monkeypatch.setattr(pr_descs, "agent_run", fake_agent_run)
        return state

    def _config(self, tmp_path, name="run"):
        from types import SimpleNamespace

        root = tmp_path / name
        root.mkdir()
        return SimpleNamespace(
            output_dir=str(root), config_json_path=str(tmp_path / "cfg.json")
        )

    def _use_case(self):
        class StubUseCase:
            def create_context_str(self, claude_config, config):
                return "context"

        return StubUseCase()

    def test_generates_when_absent(self, tmp_path, stub_agent):
        import asyncio

        config = self._config(tmp_path)
        stub_agent["output_dir"] = config.output_dir
        _write_verdict(tmp_path / "run", 1, "PASS")

        desc, body = asyncio.run(
            pr_descs.update_pr_descs(
                config, claude_config=None, use_case=self._use_case()
            )
        )
        assert os.path.isfile(desc)
        assert os.path.isfile(body)
        assert open(desc).readline().strip() == "perf: stub title"

    def test_updates_in_place(self, tmp_path, stub_agent):
        import asyncio

        config = self._config(tmp_path)
        stub_agent["output_dir"] = config.output_dir
        pr_dir = tmp_path / "run" / "pr-generate"
        pr_dir.mkdir()
        (pr_dir / "commit_desc.txt").write_text("old title\n")
        (pr_dir / "pr_desc.txt").write_text("old body\n")
        _write_verdict(tmp_path / "run", 2, "PASS")

        asyncio.run(
            pr_descs.update_pr_descs(
                config, claude_config=None, use_case=self._use_case()
            )
        )
        assert "stub" in (pr_dir / "commit_desc.txt").read_text()
        assert "stub" in (pr_dir / "pr_desc.txt").read_text()

    def test_refuses_without_success_proof(self, tmp_path, stub_agent):
        import asyncio

        config = self._config(tmp_path)
        desc, body = asyncio.run(
            pr_descs.update_pr_descs(
                config, claude_config=None, use_case=self._use_case()
            )
        )
        assert desc is None and body is None
        assert not stub_agent["steps"]  # the agent step never ran

    def test_refresh_after_pass_survives_agent_crash(self, tmp_path, monkeypatch):
        import asyncio

        config = self._config(tmp_path)
        _write_verdict(tmp_path / "run", 1, "PASS")

        async def exploding_agent(claude_config, steps, resume=False, client=None):
            raise RuntimeError("agent exploded")

        monkeypatch.setattr(pr_descs, "agent_run", exploding_agent)
        # must not raise -- the PASS verdict survives a description failure
        asyncio.run(
            pr_descs.refresh_after_pass(
                config, claude_config=None, use_case=self._use_case()
            )
        )
