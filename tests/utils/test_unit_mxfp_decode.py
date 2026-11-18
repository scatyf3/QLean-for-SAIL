# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""MXFP encoding semantics for the CPU verification decoder, not Triton kernels."""

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.cpu]

_FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
               -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


@pytest.mark.parametrize("experts", [None, 2])
@pytest.mark.parametrize("scale_kind", ["e8m0", "float32"])
def test_fp4_nibbles_group_boundary_and_tail(tensor_factory, experts, scale_kind):
    import torch
    from qlean.utils.bound_check import dequantize_mxfp4, dequantize_mxfp4_scale32

    indices = tensor_factory((2, 17), experts=experts, pattern="arange").to(torch.int64) % 16
    packed = (indices + (15 - indices) * 16).to(torch.uint8)
    exponent = (tensor_factory((*packed.shape[:-1], 2), pattern="arange") + 126).to(torch.uint8)
    expected_rows = []
    for row, scales in zip(packed.reshape(-1, 17).tolist(), exponent.reshape(-1, 2).tolist()):
        decoded = []
        for column in range(34):
            code = row[column // 2] // (16 ** (column % 2)) % 16
            decoded.append(_FP4_VALUES[code] * 2.0 ** (scales[column // 32] - 127))
        expected_rows.append(decoded)
    expected = torch.tensor(expected_rows, dtype=torch.float32).reshape(*packed.shape[:-1], 34)
    if scale_kind == "e8m0":
        actual = dequantize_mxfp4(packed, exponent)
    else:
        scales = torch.tensor([2.0 ** (value - 127) for value in exponent.flatten().tolist()],
                              dtype=torch.float32).reshape(exponent.shape)
        actual = dequantize_mxfp4_scale32(packed, scales)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.device.type == "cpu"


def test_e8m0_powers_are_decoded_exactly(tensor_factory):
    import torch
    from qlean.utils.bound_check import _e8m0_to_float32

    exponents = (tensor_factory((1, 5), pattern="arange") + 125).to(torch.uint8)
    expected = torch.tensor([[0.25, 0.5, 1.0, 2.0, 4.0]], dtype=torch.float32)
    torch.testing.assert_close(_e8m0_to_float32(exponents), expected, rtol=0, atol=0)


@pytest.mark.parametrize("scale_kind, bad_part", [
    ("e8m0", "weight"), ("e8m0", "scale"),
    ("float32", "weight"), ("float32", "scale"),
])
def test_decode_rejects_wrong_storage_dtype(tensor_factory, scale_kind, bad_part):
    import torch
    from qlean.utils.bound_check import dequantize_mxfp4, dequantize_mxfp4_scale32

    weight = tensor_factory((1, 2), pattern="zeros", dtype=torch.uint8)
    scale_dtype = torch.uint8 if scale_kind == "e8m0" else torch.float32
    scale = tensor_factory((1, 1), pattern="zeros", dtype=scale_dtype)
    if bad_part == "weight":
        weight = weight.to(torch.int8)
    else:
        scale = scale.to(torch.int32)
    decode = dequantize_mxfp4 if scale_kind == "e8m0" else dequantize_mxfp4_scale32
    with pytest.raises(TypeError, match="expects"):
        decode(weight, scale)


def test_e8m0_rejects_numeric_instead_of_byte_storage(tensor_factory):
    from qlean.utils.bound_check import _e8m0_to_float32

    with pytest.raises(TypeError, match="uint8"):
        _e8m0_to_float32(tensor_factory((1, 2)))
