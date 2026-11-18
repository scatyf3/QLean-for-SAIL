# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Triton MXFP encoding and decoding against independent scalar references."""

import math

import pytest

pytestmark = pytest.mark.unit
_FP4 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _fp4_reference(row, mode):
    payload, scales = [], []
    for start in range(0, len(row), 32):
        group = row[start:start + 32]
        peak = max(abs(v) for v in group)
        exponent = (math.ceil(math.log2(peak / 6)) if mode == "ROUND_UP"
                    else math.floor(math.log2(peak / 4))) if peak else -127
        scales.append(exponent + 127)
        codes = []
        for value in group:
            magnitude = abs(value) / 2 ** exponent if peak else 0
            code = min(range(8), key=lambda i: (abs(_FP4[i] - magnitude), i % 2))
            codes.append(code | (8 if math.copysign(1, value) < 0 else 0))
        payload.extend(codes[i] | (codes[i + 1] << 4) for i in range(0, len(codes), 2))
    return payload, scales


def _fp8_value(byte, kind):
    bits, bias = (3, 7) if kind == "float8_e4m3fn" else (2, 15)
    exponent, mantissa = (byte & 127) >> bits, byte & ((1 << bits) - 1)
    if (kind == "float8_e4m3fn" and exponent == 15 and mantissa == 7
            or kind == "float8_e5m2" and exponent == 31):
        return None
    value = (mantissa * 2 ** (1 - bias - bits) if exponent == 0
             else (1 + mantissa / 2 ** bits) * 2 ** (exponent - bias))
    return -value if byte & 128 else value


@pytest.mark.gpu
@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("mode", ["ROUND_UP", "ROUND_DOWN"])
def test_fp4_downcast_rounding_scale_and_tail(dtype_name, axis, mode):
    import torch
    from qlean.utils.downcast_to_mxfp import downcast_to_mxfp, DequantScaleRoundingMode

    row = [0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 6, 7,
           -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5, -6, -7] + [0] * 13 + [0.5, -1]
    rows = [row, [v * 2 for v in row], [0.0] * 34]
    source = torch.tensor(rows, dtype=getattr(torch, dtype_name), device="cuda").unsqueeze(0)
    expected = [_fp4_reference(r, mode) for r in rows]
    quant = torch.tensor([r[0] for r in expected], dtype=torch.uint8, device="cuda").unsqueeze(0)
    scales = torch.tensor([r[1] for r in expected], dtype=torch.uint8, device="cuda").unsqueeze(0)
    source = source.transpose(axis, -1)
    before = source.clone()
    actual, scale = downcast_to_mxfp(source, torch.uint8, axis,
                                     getattr(DequantScaleRoundingMode, mode))
    torch.testing.assert_close(actual, quant.transpose(axis, -1), rtol=0, atol=0)
    torch.testing.assert_close(scale, scales.transpose(axis, -1), rtol=0, atol=0)
    torch.testing.assert_close(source, before, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("mode", ["ROUND_UP", "ROUND_DOWN"])
def test_fp4_halfway_adjacent_ulp(dtype_name, mode):
    import torch
    from qlean.utils.downcast_to_mxfp import downcast_to_mxfp, DequantScaleRoundingMode

    dtype = getattr(torch, dtype_name)
    midpoint = torch.tensor(0.25, dtype=dtype)
    above = torch.nextafter(midpoint, torch.tensor(float("inf"), dtype=dtype))
    below = torch.nextafter(midpoint, torch.tensor(0.0, dtype=dtype))
    row = [6.0, float(below), float(midpoint), float(above), -float(above), 0.0] + [0.0] * 26
    source = torch.tensor([row], dtype=dtype, device="cuda")
    expected, scales = _fp4_reference(row, mode)
    payload, actual_scales = downcast_to_mxfp(source, torch.uint8, -1,
                                             getattr(DequantScaleRoundingMode, mode))
    torch.testing.assert_close(payload.cpu(), torch.tensor([expected], dtype=torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(actual_scales.cpu(), torch.tensor([scales], dtype=torch.uint8), rtol=0, atol=0)


@pytest.mark.gpu
def test_fp4_small_multitile_noncontiguous_roundtrip():
    import torch
    from qlean.utils.downcast_to_mxfp import downcast_to_mxfp
    from qlean.utils.upcast_from_mxfp import upcast_from_mxfp

    row = [6, -6, 0.5, -0.5, 1, 0, 2, -2] * 16 + [6, -6]
    source = torch.tensor([row] * 33, dtype=torch.float32, device="cuda").t().contiguous().t()
    assert not source.is_contiguous()
    expected, scales = _fp4_reference(row, "ROUND_UP")
    payload, actual_scales = downcast_to_mxfp(source, torch.uint8, -1)
    torch.testing.assert_close(payload.cpu(), torch.tensor([expected] * 33, dtype=torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(actual_scales.cpu(), torch.tensor([scales] * 33, dtype=torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(upcast_from_mxfp(payload, actual_scales, torch.float32, -1), source, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["float8_e5m2"])
@pytest.mark.parametrize("mode", ["ROUND_UP", "ROUND_DOWN"])
def test_fp8_downcast_exact_grid_and_roundtrip(kind, mode):
    import torch
    from qlean.utils.downcast_to_mxfp import downcast_to_mxfp, DequantScaleRoundingMode
    from qlean.utils.upcast_from_mxfp import upcast_from_mxfp

    maximum = 57344
    row = [maximum, -maximum, 1, -1, 0.5, -0.5, 0, 2] * 4 + [1, -1]
    source = torch.tensor([row, [v * 2 for v in row]], dtype=torch.float32, device="cuda")
    codes, scales = downcast_to_mxfp(source, getattr(torch, kind), -1,
                                    getattr(DequantScaleRoundingMode, mode))
    tail_exp = -15
    expected_scale = torch.tensor([[127, 127 + tail_exp], [128, 128 + tail_exp]],
                                  dtype=torch.uint8, device="cuda")
    expected_values = torch.tensor([[*row[:32], 2 ** -tail_exp, -(2 ** -tail_exp)]] * 2,
                                   dtype=torch.float32, device="cuda")
    assert codes.dtype == getattr(torch, kind)
    torch.testing.assert_close(codes.float(), expected_values, rtol=0, atol=0)
    torch.testing.assert_close(scales, expected_scale, rtol=0, atol=0)
    decoded = upcast_from_mxfp(codes, scales, torch.float32, -1)
    torch.testing.assert_close(decoded, source, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["uint8", "float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_upcast_independent_codes_axes_and_tail(kind, dtype_name, axis):
    import torch
    from qlean.utils.upcast_from_mxfp import upcast_from_mxfp

    if kind == "uint8":
        payload = [i | ((i + 1) << 4) for i in range(0, 16, 2)] * 2 + [0xF1]
        values = [(-1 if i & 8 else 1) * _FP4[i & 7] for i in range(16)] * 2 + [0.5, -6]
    else:
        payload = [b if _fp8_value(b, kind) is not None else 0 for b in range(256)] + [1, 129]
        values = [_fp8_value(b, kind) for b in payload]
    groups = math.ceil(len(values) / 32)
    exponents = [126 + i % 2 for i in range(groups)]
    raw = torch.tensor(payload, dtype=torch.uint8, device="cuda").view(getattr(torch, kind))
    raw = raw.repeat(2, 3, 1).transpose(axis, -1)
    scales = torch.tensor(exponents, dtype=torch.uint8, device="cuda").repeat(2, 3, 1).transpose(axis, -1)
    expected = torch.tensor([v * 2 ** (exponents[i // 32] - 127) for i, v in enumerate(values)],
                            dtype=getattr(torch, dtype_name), device="cuda").repeat(2, 3, 1).transpose(axis, -1)
    raw_before, scale_before = raw.clone(), scales.clone()
    actual = upcast_from_mxfp(raw, scales, getattr(torch, dtype_name), axis)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(raw.view(torch.uint8), raw_before.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(scales, scale_before, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
def test_upcast_scale_extremes_saturate_and_nan(dtype_name):
    import torch
    from qlean.utils.upcast_from_mxfp import upcast_from_mxfp

    dtype = getattr(torch, dtype_name)
    raw = torch.full((4, 16), 0xF7, dtype=torch.uint8, device="cuda")
    scales = torch.tensor([[0], [127], [254], [255]], dtype=torch.uint8, device="cuda")
    actual = upcast_from_mxfp(raw, scales, dtype, -1)
    maximum = torch.finfo(dtype).max
    expected = torch.tensor([[0.0, -0.0] * 16, [6, -6] * 16,
                             [maximum, -maximum] * 16], dtype=dtype, device="cuda")
    torch.testing.assert_close(actual[:3], expected, rtol=0, atol=0)
    assert torch.isnan(actual[3]).all()


@pytest.mark.gpu
@pytest.mark.parametrize("axis", [0, -1])
@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
def test_fp32_scale_upcast(axis, dtype_name):
    import torch
    from qlean.utils.upcast_from_mxfp import upcast_from_mxfp_fp32_scale

    raw = torch.full((2, 17), 0xF1, dtype=torch.uint8, device="cuda")
    scale = torch.tensor([[0.75, 1.25], [1.5, 2.5]], dtype=torch.float32, device="cuda")
    expected = torch.tensor([[0.375, -4.5] * 16 + [0.625, -7.5],
                             [0.75, -9] * 16 + [1.25, -15]],
                            dtype=getattr(torch, dtype_name), device="cuda")
    actual = upcast_from_mxfp_fp32_scale(raw.transpose(axis, -1), scale.transpose(axis, -1),
                                        getattr(torch, dtype_name), axis)
    torch.testing.assert_close(actual, expected.transpose(axis, -1), rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(0, 32), (2, 0)])
def test_empty_mxfp_shape_contract(shape):
    import torch
    from qlean.utils.downcast_to_mxfp import downcast_to_mxfp
    from qlean.utils.upcast_from_mxfp import upcast_from_mxfp

    source = torch.empty(shape, dtype=torch.float32, device="cuda")
    raw, scale = downcast_to_mxfp(source, torch.uint8, -1)
    assert raw.dtype == scale.dtype == torch.uint8
    assert raw.device == scale.device == source.device
    assert raw.shape == (shape[0], shape[1] // 2)
    assert scale.shape == (shape[0], math.ceil(shape[1] / 32))
    assert upcast_from_mxfp(raw, scale, torch.float32, -1).shape == source.shape


@pytest.mark.cpu
@pytest.mark.parametrize("fault", ["axis", "odd", "dtype"])
def test_downcast_rejects_invalid_contract_before_kernel(fault):
    import torch
    from qlean.utils.downcast_to_mxfp import downcast_to_mxfp

    raw = torch.ones((2, 31 if fault == "odd" else 32), dtype=torch.float32)
    with pytest.raises(AssertionError):
        downcast_to_mxfp(raw, torch.int8 if fault == "dtype" else torch.uint8,
                         2 if fault == "axis" else -1)
