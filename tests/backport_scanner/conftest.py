# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

from common import llm_client


@pytest.fixture(autouse=True)
def isolate_model_configuration(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in ("OPENAI_API_BASE", "OPENAI_API_KEY", "CLAUDE_CODE_USE_VERTEX"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(llm_client, "_api_base", None)
    monkeypatch.setattr(llm_client, "_api_key", None)
    monkeypatch.setattr(llm_client, "_model_params", {})
