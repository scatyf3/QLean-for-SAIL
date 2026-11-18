# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared factory reproducibility, deep copies, and shape constraints."""

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.cpu]


def test_recipe_factory_copies_nested_configuration(recipe_factory):
    regions = [{"name": "fallback", "match": ["*"]}]
    template = recipe_factory(regions=regions)
    first = recipe_factory(template)
    second = recipe_factory(template)
    first_params = first["quant_stage"]["quant_modifiers"]["GeneralQuantEngineModifier"]
    first_params["regions"][0]["match"].append("changed")
    assert template == second
    assert first != second
    template_params = template["quant_stage"]["quant_modifiers"]["GeneralQuantEngineModifier"]
    template_params["regions"][0]["match"].append("template-change")
    assert regions == [{"name": "fallback", "match": ["*"]}]


def test_tensor_factory_is_repeatable_without_global_rng_changes(tensor_factory):
    import torch

    before = torch.random.get_rng_state().clone()
    first = tensor_factory((4, 8), seed=7)
    second = tensor_factory((4, 8), seed=7)
    different = tensor_factory((4, 8), seed=8)
    assert torch.equal(first, second)
    assert not torch.equal(first, different)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert first.dtype == torch.float32
    assert first.device.type == "cpu"


def test_tensor_factory_supports_group_block_and_experts(tensor_factory):
    tensor = tensor_factory((4, 8), group_size=4, block_size=(2, 4), experts=2, pattern="arange")
    assert tuple(tensor.shape) == (2, 4, 8)
    assert tensor[0, 0, 0].item() == 0
    assert tensor[1, 0, 0].item() == 32
    assert tensor[1, 3, 7].item() == 63


@pytest.mark.parametrize("params, message", [
    ({"group_size": 3}, "group_size"),
    ({"group_size": True}, "group_size"),
    ({"group_size": -1.0}, "group_size"),
    ({"block_size": (3, 4)}, "block_size"),
    ({"block_size": False}, "block_size"),
    ({"block_size": 2.0}, "block_size"),
    ({"experts": 0}, "experts"),
])
def test_tensor_factory_rejects_invalid_constraints(tensor_factory, params, message):
    with pytest.raises(ValueError, match=message):
        tensor_factory((4, 8), **params)
