# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""FP8 quantization codes, scales, and complete-block requirements."""

import pytest

pytestmark = pytest.mark.unit
_DEVICES = [pytest.param("cpu", marks=pytest.mark.cpu),
            pytest.param("cuda", marks=pytest.mark.gpu)]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("batched", [False, True])
def test_dynamic_codes_scales_and_ties(assert_tensor_equal, device, dtype_name, batched):
    import torch
    from qlean.modifiers.utils import weight_quant

    row = [-448, -1.1875, -1.0625, 0, 1.0625, 1.1875, 448]
    weight = torch.tensor([row, [v * 2 for v in row], [0] * 7],
                          dtype=getattr(torch, dtype_name), device=device)
    values = torch.tensor([[-448, -1.25, -1, 0, 1, 1.25, 448]] * 2 + [[0] * 7],
                          dtype=torch.float32, device=device)
    scales = torch.tensor([[1], [2], [2 ** -23]], dtype=torch.float32, device=device)
    if batched:
        weight = torch.stack((weight, weight * 2))
        values = torch.stack((values, values))
        scales = torch.stack((scales, torch.tensor([[2], [4], [2 ** -23]],
                                                  dtype=torch.float32, device=device)))
    weight = weight.transpose(-1, -2).contiguous().transpose(-1, -2)
    before = weight.clone()
    codes, scale = weight_quant(weight, "FP8_DYNAMIC")
    assert codes.dtype == torch.float8_e4m3fn
    assert codes.device == scale.device == weight.device
    assert_tensor_equal(codes.float(), values, scheme="FP8_DYNAMIC/codes", seed=0)
    assert_tensor_equal(scale, scales, scheme="FP8_DYNAMIC/scale", seed=0)
    assert_tensor_equal(weight, before, scheme="FP8_DYNAMIC/input", seed=0)
    error = (codes.float() * scale - weight.float()).abs()
    assert (error <= scales / 16).all(), "Tie-case error must not exceed scale / 16"


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("batched", [False, True])
def test_block_independent_scales_and_zero(assert_tensor_equal, device, dtype_name, batched):
    import torch
    from qlean.modifiers.utils import weight_quant

    row = torch.tensor([-448, -1.1875, -1.0625, 0, 1.0625, 1.1875, 448, 0],
                       dtype=getattr(torch, dtype_name), device=device).repeat(128, 16)
    weight = torch.cat((torch.cat((row, row * 2), 1),
                        torch.cat((row * 4, torch.zeros_like(row)), 1)), 0)
    quant_row = torch.tensor([-448, -1.25, -1, 0, 1, 1.25, 448, 0],
                             dtype=torch.float32, device=device).repeat(128, 16)
    values = torch.cat((torch.cat((quant_row, quant_row), 1),
                        torch.cat((quant_row, torch.zeros_like(quant_row)), 1)), 0)
    scales = torch.tensor([[1, 2], [4, 1]], dtype=torch.float32, device=device)
    if batched:
        weight = torch.stack((weight, weight * 2))
        values = torch.stack((values, values))
        scales = torch.stack((scales, torch.tensor([[2, 4], [8, 1]],
                                                  dtype=torch.float32, device=device)))
    before = weight.clone()
    codes, scale = weight_quant(weight, "FP8_BLOCK")
    assert codes.dtype == torch.float8_e4m3fn
    assert codes.device == scale.device == weight.device
    assert_tensor_equal(codes.float(), values, scheme="FP8_BLOCK/codes", seed=0)
    assert_tensor_equal(scale, scales, scheme="FP8_BLOCK/scale", seed=0)
    expanded = scales.repeat_interleave(128, -2).repeat_interleave(128, -1)
    assert ((codes.float() * expanded - weight.float()).abs() <= expanded / 16).all()
    assert_tensor_equal(weight, before, scheme="FP8_BLOCK/input", seed=0)


@pytest.mark.cpu
@pytest.mark.parametrize("scheme", ["FP8_DYNAMIC", "FP8_BLOCK"])
@pytest.mark.parametrize("shape", [(8,), (1, 1, 8, 8)])
def test_fp8_rejects_unsupported_rank(scheme, shape):
    import torch
    from qlean.modifiers.utils import weight_quant

    with pytest.raises(ValueError, match="Unsupported tensor dimension"):
        weight_quant(torch.zeros(shape, dtype=torch.float32), scheme)
