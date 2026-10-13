# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CodeGenConfig loading, including fields derived from the cross-trace data."""

import json
import subprocess

from llm.auto_analyze.configs.cross_trace_config import CROSS_IMPROVEMENT_FILE
from llm.auto_analyze.configs.single_trace_config import (
    HIGH_LEVEL_OPS_FILE,
    MEDIAN_BLOCK_FILE,
)
from llm.auto_code_gen.code_gen_configs import CodeGenConfig


def _git_repo(path):
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "-C", str(path), "init", "-q"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--allow-empty",
            "-m",
            "init",
        ],
        check=True,
        capture_output=True,
    )
    return path


def _trace_dir(root, framework):
    d = root / f"{framework}_results"
    d.mkdir()
    (d / "run_params.txt").write_text(
        f"framework_name: {framework}\n"
        "framework_source_code: /code/does/not/matter\n"
        "commit_id: 0123456789abcdef\n"
        "model: test-model\n"
        "gpu_type: B200\n"
        "batch_size_range: 1\n"
        "prefill_size_range: 2048\n"
        "output_size_range: 128\n"
        f"run_command: bench {framework}\n"
    )
    (d / MEDIAN_BLOCK_FILE).write_text("median block\n")
    (d / HIGH_LEVEL_OPS_FILE).write_text("high level ops\n")
    return d


def test_load_from_cross_trace_populates_faster_slower_views(tmp_path):
    # target_trace_id=1 -> traces[1] is the framework being improved (the
    # slower target) and traces[0] is the source (faster) ported from.
    fast = _trace_dir(tmp_path, "fastfw")
    slow = _trace_dir(tmp_path, "slowfw")
    source_repo = _git_repo(tmp_path / "source_repo")

    cross_out = tmp_path / "cross_out"
    cross_out.mkdir()
    (cross_out / CROSS_IMPROVEMENT_FILE).write_text("1. do the thing\n")

    cross_cfg = tmp_path / "cross_trace.json"
    cross_cfg.write_text(
        json.dumps(
            {
                "traces": [{"result_dir": str(fast)}, {"result_dir": str(slow)}],
                "target_trace_id": 1,
                "output_dir": str(cross_out),
            }
        )
    )

    code_gen_cfg = tmp_path / "code_gen.json"
    code_gen_cfg.write_text(
        json.dumps(
            {
                "cross_trace_config": str(cross_cfg),
                "improvement_id": 1,
                "source_code_dir": str(source_repo),
                "output_dir": str(tmp_path / "run"),
            }
        )
    )

    cfg = CodeGenConfig.from_json(str(code_gen_cfg))

    # Framework identities: source is the faster donor, target the slower
    # framework the improvement plan applies to.
    assert cfg.faster_framework == "fastfw"
    assert cfg.slower_framework == "slowfw"
    assert cfg.source_framework == "fastfw"
    assert cfg.target_framework == "slowfw"

    # Evidence paths must be derived and non-empty.
    assert cfg.faster_framework_test_dir == str(fast / "run_originals")
    assert cfg.slower_framework_test_dir == str(slow / "run_originals")
    assert cfg.faster_transformer_block_high_level_ops_file == str(
        fast / HIGH_LEVEL_OPS_FILE
    )
    assert cfg.slower_transformer_block_high_level_ops_file == str(
        slow / HIGH_LEVEL_OPS_FILE
    )
    assert cfg.faster_median_transformer_block_file == str(fast / MEDIAN_BLOCK_FILE)
    assert cfg.slower_median_transformer_block_file == str(slow / MEDIAN_BLOCK_FILE)

    # Consistency with the trace-index-ordered parallel lists.
    assert cfg.framework_names == ["fastfw", "slowfw"]
    assert cfg.faster_framework_test_dir in cfg.framework_test_dirs
    assert cfg.slower_framework_test_dir in cfg.framework_test_dirs
    assert (
        cfg.faster_transformer_block_high_level_ops_file
        in cfg.transformer_block_high_level_ops_files
    )
    assert (
        cfg.slower_median_transformer_block_file in cfg.median_transformer_block_files
    )
