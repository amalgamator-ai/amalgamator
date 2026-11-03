# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fused shared experts are requested by default on ROCm/AITER.

The default only decides whether the *request* is made. Whether a given layer
actually fuses is still resolved per layer by
``resolve_layer_fused_shared_expert``: quantization compatibility and expert
parallelism can still turn it off, and the environment variable remains a hard
kill switch in both directions.
"""

from unittest.mock import patch

import pytest

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.utils import (
    resolve_layer_fused_shared_expert,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    current_platform.is_xpu(),
    reason="ROCm-specific aiter ops are not supported on XPU",
)

ENV_KEY = "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS"


def _env_value(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(ENV_KEY, raising=False)
    else:
        monkeypatch.setenv(ENV_KEY, value)
    return envs.environment_variables[ENV_KEY]()


def test_enabled_by_default(monkeypatch):
    assert _env_value(monkeypatch, None) is True


@pytest.mark.parametrize("value", ["0", "false", "False"])
def test_kill_switch(monkeypatch, value):
    assert _env_value(monkeypatch, value) is False


def test_resolver_still_gates_incompatible_quantization(monkeypatch):
    """The default must not bypass the compatibility resolver."""
    with patch(
        "vllm._aiter_ops.rocm_aiter_ops.is_fusion_moe_shared_experts_enabled",
        return_value=True,
    ), patch(
        "vllm.model_executor.layers.fused_moe.utils"
        ".is_shared_expert_quant_fse_compatible",
        return_value=(False, "unsupported quantization for FSE"),
    ):
        enabled = resolve_layer_fused_shared_expert(
            quant_config=None,
            prefix="model.layers.0.mlp",
            shared_expert_name="shared_expert",
        )
    assert enabled is False


def test_resolver_enables_when_compatible(monkeypatch):
    with patch(
        "vllm._aiter_ops.rocm_aiter_ops.is_fusion_moe_shared_experts_enabled",
        return_value=True,
    ), patch(
        "vllm.model_executor.layers.fused_moe.utils"
        ".is_shared_expert_quant_fse_compatible",
        return_value=(True, None),
    ):
        enabled = resolve_layer_fused_shared_expert(
            quant_config=None,
            prefix="model.layers.0.mlp",
            shared_expert_name="shared_expert",
        )
    assert enabled is True


def test_expert_parallel_still_disables_for_models_that_opt_out(monkeypatch):
    with patch(
        "vllm._aiter_ops.rocm_aiter_ops.is_fusion_moe_shared_experts_enabled",
        return_value=True,
    ), patch(
        "vllm.config.get_current_vllm_config_or_none"
    ) as get_cfg:
        get_cfg.return_value.parallel_config.enable_expert_parallel = True
        enabled = resolve_layer_fused_shared_expert(
            quant_config=None,
            prefix="model.layers.0.mlp",
            shared_expert_name="shared_expert",
            disable_under_ep=True,
        )
    assert enabled is False
