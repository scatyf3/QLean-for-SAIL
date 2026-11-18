# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""INT8 dequantization with independent input codes and expected values."""

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.cpu]


@pytest.mark.parametrize("codes_data, scale_data, expected_data", [
    pytest.param([-128, -3, 0, 127], [0.5, 2, 4, 0.25],
                 [-64, -6, 0, 31.75], id="vector"),
    pytest.param([[-128, -3, 127], [0, 2, -4]], [0.5, 2],
                 [[-64, -1.5, 63.5], [0, 4, -8]], id="channel-vector"),
    pytest.param([[-128, -3, 127], [0, 2, -4]], [[0.5], [2]],
                 [[-64, -1.5, 63.5], [0, 4, -8]], id="channel-column"),
    pytest.param([[[-128, 2, 127], [4, -6, 0]], [[-2, 3, 8], [7, 0, -1]]],
                 [0.5, 2],
                 [[[-64, 1, 63.5], [2, -3, 0]], [[-4, 6, 16], [14, 0, -2]]],
                 id="three-dimensional-dim0"),
    pytest.param([[[-128, 2, 127], [4, -6, 0]], [[-2, 3, 8], [7, 0, -1]]],
                 [[[0.5], [2]], [[4], [0.25]]],
                 [[[-64, 1, 63.5], [8, -12, 0]], [[-8, 12, 32], [1.75, 0, -0.25]]],
                 id="three-dimensional-explicit-rows"),
])
@pytest.mark.parametrize("scale_dtype, output_dtype", [
    ("float16", "float32"), ("bfloat16", "float32"),
    ("float32", "float32"), ("float64", "float64"),
])
def test_int8_dequantization_broadcast_and_dtype(
    assert_tensor_equal, codes_data, scale_data, expected_data, scale_dtype, output_dtype,
):
    import torch
    from qlean.utils.bound_check import dequantize_int8_per_channel

    codes = torch.tensor(codes_data, dtype=torch.int8)
    scale = torch.tensor(scale_data, dtype=getattr(torch, scale_dtype))
    codes_before, scale_before = codes.clone(), scale.clone()
    actual = dequantize_int8_per_channel(codes, scale)
    expected = torch.tensor(expected_data, dtype=getattr(torch, output_dtype))
    assert_tensor_equal(actual, expected, scheme="W8A8/dequant", seed=0)
    assert actual.shape == codes.shape
    assert actual.device == codes.device == scale.device
    assert_tensor_equal(codes, codes_before, scheme="W8A8/dequant", seed=0)
    assert_tensor_equal(scale, scale_before, scheme="W8A8/dequant", seed=0)


@pytest.mark.parametrize("scalar_scale", [False, True])
def test_int8_dequantization_noncontiguous_and_scalar(assert_tensor_equal, scalar_scale):
    import torch
    from qlean.utils.bound_check import dequantize_int8_per_channel

    codes = torch.tensor([[-128, 0, 127], [-2, 4, -6]], dtype=torch.int8).t().contiguous().t()
    scale = (torch.tensor(0.5) if scalar_scale else
             torch.tensor([[0.5, 9], [2, 9]], dtype=torch.float32)[:, 0])
    assert not codes.is_contiguous()
    if not scalar_scale:
        assert not scale.is_contiguous()
    before = codes.clone(), scale.clone()
    expected = torch.tensor([[-64, 0, 63.5], [-1, 2, -3] if scalar_scale else [-4, 8, -12]])
    assert_tensor_equal(dequantize_int8_per_channel(codes, scale), expected,
                        scheme="W8A8/dequant", seed=0)
    assert_tensor_equal(codes, before[0], scheme="W8A8/dequant", seed=0)
    assert_tensor_equal(scale, before[1], scheme="W8A8/dequant", seed=0)


@pytest.mark.parametrize("shape, scale_shape", [
    ((2, 3), (4,)), ((2, 3), (2, 2)), ((2, 2, 3), (3,)),
])
def test_int8_dequantization_rejects_incompatible_broadcast(tensor_factory, shape, scale_shape):
    import torch
    from qlean.utils.bound_check import dequantize_int8_per_channel

    codes = tensor_factory(shape, pattern="zeros", dtype=torch.int8)
    scale = tensor_factory(scale_shape, pattern="arange") + 1
    before = codes.clone(), scale.clone()
    with pytest.raises(RuntimeError, match="size of tensor"):
        dequantize_int8_per_channel(codes, scale)
    assert torch.equal(codes, before[0])
    assert torch.equal(scale, before[1])
