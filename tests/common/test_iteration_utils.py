# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

from common.iteration_utils import resolve_output_file


def test_resolve_output_file_prefers_review_of_latest_version(tmp_path):
    (tmp_path / "artifact_V1.txt").write_text("draft 1")
    (tmp_path / "artifact_review_V1.txt").write_text("review 1")
    (tmp_path / "artifact_V2.txt").write_text("draft 2")
    (tmp_path / "artifact_review_V2.txt").write_text("review 2")

    resolved = resolve_output_file(str(tmp_path), "artifact.txt")

    assert resolved == str(tmp_path / "artifact_review_V2.txt")


def test_resolve_output_file_uses_latest_draft_without_review(tmp_path):
    (tmp_path / "artifact_V1.txt").write_text("draft 1")
    (tmp_path / "artifact_review_V1.txt").write_text("review 1")
    (tmp_path / "artifact_V2.txt").write_text("draft 2")

    resolved = resolve_output_file(str(tmp_path), "artifact.txt")

    assert resolved == str(tmp_path / "artifact_V2.txt")


def test_resolve_output_file_ignores_empty_review(tmp_path):
    (tmp_path / "artifact_V1.txt").write_text("draft")
    (tmp_path / "artifact_review_V1.txt").write_text("")

    resolved = resolve_output_file(str(tmp_path), "artifact.txt")

    assert resolved == str(tmp_path / "artifact_V1.txt")
