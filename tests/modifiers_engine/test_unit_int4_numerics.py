# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""INT4 absmax and MSE quantization against independent references."""

from fractions import Fraction

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.cpu]


def _scalar_grid_reference(rows, divisors):
    """Search candidates with exact fractions, keeping the first on ties."""
    codes, scales = [], []
    for row in rows:
        values = [Fraction(value) for value in row]
        peak = max(abs(value) for value in values)
        candidates = []
        for divisor in divisors:
            scale = max(peak / Fraction(divisor), Fraction(1, 2 ** 23))
            quant = [max(-8, min(7, round(value / scale))) for value in values]
            error = sum((value - code * scale) ** 2 for value, code in zip(values, quant))
            candidates.append((error, quant, scale))
        _, quant, scale = min(candidates, key=lambda candidate: candidate[0])
        codes.append(quant)
        scales.append([float(scale)])
    return codes, scales


@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("experts", [None, 2])
def test_absmax_codes_rounding_and_zero_floor(tensor_factory, dtype_name, experts):
    import torch
    from qlean.modifiers_engine._runtime.plan import _int4_channel_weight_quant_for_cfg

    weight = tensor_factory((3, 8), pattern="zeros", experts=experts,
                            dtype=getattr(torch, dtype_name))
    row = torch.tensor([-7, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 7], dtype=weight.dtype)
    weight[..., 0, :] = row
    weight[..., 1, :] = row * 2
    expected_codes = torch.tensor([
        [-7, -2, -2, 0, 0, 2, 2, 7], [-7, -2, -2, 0, 0, 2, 2, 7], [0] * 8,
    ], dtype=torch.int8)
    expected_scale = torch.tensor([[1], [2], [2 ** -23]], dtype=torch.float32)
    if experts is not None:
        weight[1].mul_(-2)
        expected_codes = expected_codes.repeat(experts, 1, 1)
        expected_codes[1].neg_()
        expected_scale = expected_scale.repeat(experts, 1, 1)
        expected_scale[1, :2].mul_(2)
    before = weight.clone()

    codes, scale = _int4_channel_weight_quant_for_cfg(weight, {})

    torch.testing.assert_close(codes, expected_codes, rtol=0, atol=0)
    torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(weight, before, rtol=0, atol=0)


@pytest.mark.parametrize("values, divisors", [
    ([[8, 1, 1, 1], [-8, 1, 1, 1], [8, 0, 0, 0], [0, 0, 0, 0]], (4, 8)),
    ([[8, 4, 0, 0]], (2, 4)),
    ([[8, 1, 1, 1]], (4,)),
    ([[7, 1.5, -1.5, 0]], (7,)),
])
def test_mse_matches_independent_candidate_search(tensor_factory, recipe_factory, values, divisors):
    import torch
    from qlean.modifiers_engine._runtime.plan import _int4_channel_weight_quant_for_cfg

    weight = tensor_factory((len(values), len(values[0])), pattern="zeros")
    weight.copy_(torch.tensor(values, dtype=torch.float32))
    recipe = recipe_factory(scale_search="mse", mse_divisor_min=divisors[0],
                            mse_divisor_max=divisors[-1],
                            mse_divisor_step=divisors[1] - divisors[0] if len(divisors) > 1 else 1)
    cfg = recipe["quant_stage"]["quant_modifiers"]["GeneralQuantEngineModifier"]
    expected_codes, expected_scales = _scalar_grid_reference(values, divisors)

    codes, scale = _int4_channel_weight_quant_for_cfg(weight, cfg)

    torch.testing.assert_close(codes, torch.tensor(expected_codes, dtype=torch.int8),
                               rtol=0, atol=0)
    torch.testing.assert_close(scale, torch.tensor(expected_scales, dtype=torch.float32),
                               rtol=0, atol=0)


@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("grid", [(4, 24, 0.25), (6, 10, 0.5)])
def test_mse_wide_grid_dtype_experts_and_absmax_bound(dtype_name, batched, grid):
    import torch
    from qlean.modifiers_engine._runtime.plan import _int4_channel_weight_quant_for_cfg

    values = [[8, 1, -1, 1, -1, 1, 0, 0], [0] * 8, [-8, 2, -2, 1, -1, 0, 0, 0]]
    weight = torch.tensor(values, dtype=getattr(torch, dtype_name))
    if batched:
        weight = torch.stack((weight, -weight * 2))
    before = weight.clone()
    low, high, step = grid
    divisors = [low + i * step for i in range(round((high - low) / step) + 1)]
    expected_codes, expected_scales = _scalar_grid_reference(
        weight.float().reshape(-1, 8).tolist(), divisors)
    cfg = dict(scale_search="mse", mse_divisor_min=low,
               mse_divisor_max=high, mse_divisor_step=step)
    codes, scale = _int4_channel_weight_quant_for_cfg(weight, cfg)
    torch.testing.assert_close(codes, torch.tensor(expected_codes, dtype=torch.int8).reshape(weight.shape),
                               rtol=0, atol=0)
    torch.testing.assert_close(scale, torch.tensor(expected_scales, dtype=torch.float32).reshape(scale.shape),
                               rtol=1e-7, atol=0)
    abs_codes, abs_scale = _int4_channel_weight_quant_for_cfg(weight, {})
    mse = ((weight.float() - codes.float() * scale) ** 2).mean(-1)
    baseline = ((weight.float() - abs_codes.float() * abs_scale) ** 2).mean(-1)
    assert 7 in divisors
    assert (mse <= baseline + 1e-7).all()
    torch.testing.assert_close(weight, before, rtol=0, atol=0)


@pytest.mark.parametrize("cfg, message", [
    ({"scale_search": "unknown"}, "Unknown scale_search"),
    ({"scale_search": "mse"}, "requires transform_cfg fields"),
    ({"scale_search": "mse", "mse_divisor_min": 0, "mse_divisor_max": 8,
      "mse_divisor_step": 1}, "Invalid MSE int4 divisor grid"),
    ({"scale_search": "mse", "mse_divisor_min": 8, "mse_divisor_max": 4,
      "mse_divisor_step": 1}, "Invalid MSE int4 divisor grid"),
    ({"scale_search": "mse", "mse_divisor_min": 4, "mse_divisor_max": 8,
      "mse_divisor_step": 0}, "Invalid MSE int4 divisor grid"),
])
def test_invalid_search_configuration_fails(tensor_factory, cfg, message):
    from qlean.modifiers_engine._runtime.plan import _int4_channel_weight_quant_for_cfg

    with pytest.raises(ValueError, match=message):
        _int4_channel_weight_quant_for_cfg(tensor_factory((2, 4)), cfg)


@pytest.mark.parametrize("shape", [(4,), (1, 1, 2, 4)])
def test_invalid_rank_fails(tensor_factory, shape):
    from qlean.modifiers_engine._runtime.plan import _int4_channel_weight_quant_for_cfg

    with pytest.raises(ValueError, match="requires 2D"):
        _int4_channel_weight_quant_for_cfg(tensor_factory(shape), {})
