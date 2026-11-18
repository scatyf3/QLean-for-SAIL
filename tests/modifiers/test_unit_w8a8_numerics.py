# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Real W8A8 quantization against handwritten codes and scales."""

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("device", [
    pytest.param("cpu", marks=pytest.mark.cpu),
    pytest.param("cuda", marks=pytest.mark.gpu),
])
@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("batched", [False, True], ids=["matrix", "experts"])
def test_w8a8_codes_scales_and_rounding(assert_tensor_equal, device, dtype_name, batched):
    import torch
    from qlean.modifiers.utils import weight_quant

    weight = torch.tensor([
        [-127, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 127],
        [-254, -5, -3, -1, 1, 3, 5, 254],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=getattr(torch, dtype_name), device=device)
    expected_codes = torch.tensor([
        [-127, -2, -2, 0, 0, 2, 2, 127],
        [-127, -2, -2, 0, 0, 2, 2, 127],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=torch.int8, device=device)
    expected_scale = torch.tensor([[1], [2], [2 ** -23]], dtype=torch.float32, device=device)
    if batched:
        weight = torch.stack((weight, -weight * 2))
        expected_codes = torch.stack((expected_codes, -expected_codes))
        second_scale = torch.tensor([[2], [4], [2 ** -23]], dtype=torch.float32, device=device)
        expected_scale = torch.stack((expected_scale, second_scale))
    original = weight.clone()

    codes, scale = weight_quant(weight, "W8A8")

    assert_tensor_equal(codes, expected_codes, scheme="W8A8", seed=0)
    assert_tensor_equal(scale, expected_scale, scheme="W8A8", seed=0)
    assert_tensor_equal(weight, original, scheme="W8A8", seed=0)
    assert codes.device == weight.device == scale.device
    error = (codes.float() * scale - weight.float()).abs()
    assert_tensor_equal(error <= scale / 2, torch.ones_like(error, dtype=torch.bool),
                        scheme="W8A8/bound", seed=0)


@pytest.mark.cpu
@pytest.mark.parametrize("shape", [(8,), (2, 2, 2, 2)])
def test_w8a8_rejects_unsupported_rank(tensor_factory, shape):
    from qlean.modifiers.utils import weight_quant

    with pytest.raises(ValueError, match="Unsupported tensor dimension"):
        weight_quant(tensor_factory(shape), "W8A8")


@pytest.mark.cpu
def test_unknown_scheme_is_not_silently_accepted(tensor_factory):
    from qlean.modifiers.utils import weight_quant

    with pytest.raises(ValueError, match="Unsupported quantization scheme"):
        weight_quant(tensor_factory(), "UNKNOWN_SCHEME")
