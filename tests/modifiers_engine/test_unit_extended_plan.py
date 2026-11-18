# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Groupwise INT4, quantized passthrough, and output contracts for all Plan ops."""

from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.cpu]


def _pack_offset_words(rows):
    import torch

    words = []
    for row in rows:
        output = []
        for start in range(0, len(row), 8):
            word = sum((int(code) + 8) << (4 * i) for i, code in enumerate(row[start:start + 8]))
            output.append(word if word < 2 ** 31 else word - 2 ** 32)
        words.append(output)
    return torch.tensor(words, dtype=torch.int32)


@pytest.mark.parametrize("group_size", [32, -1])
@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
def test_groupwise_codes_scale_and_fixed_triplet(group_size, dtype_name):
    import torch
    from qlean.modifiers_engine._runtime.plan import execute_op, op_output_names

    row = [-7, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 7] * 4
    weight = torch.tensor([row + [v * 2 for v in row], [0] * 64], dtype=getattr(torch, dtype_name))
    first = [-7, -2, -2, 0, 0, 2, 2, 7] if group_size == 32 else [-4, -1, -1, 0, 0, 1, 1, 4]
    codes = [first * 4 + [-7, -2, -2, 0, 0, 2, 2, 7] * 4, [0] * 64]
    scales = [[1, 2], [2 ** -23, 2 ** -23]] if group_size == 32 else [[2], [2 ** -23]]
    op = ("quant_int4_groupwise", "x.weight", group_size)
    before = weight.clone()
    produced = execute_op(op, weight, {})
    assert [key for key, _ in produced] == op_output_names(op) == [
        "x.weight_packed", "x.weight_scale", "x.weight_shape"]
    packed, scale, shape = [v for _, v in produced]
    torch.testing.assert_close(packed, _pack_offset_words(codes), rtol=0, atol=0)
    torch.testing.assert_close(scale, torch.tensor(scales, dtype=torch.bfloat16), rtol=0, atol=0)
    torch.testing.assert_close(shape, torch.tensor([2, 64], dtype=torch.int64), rtol=0, atol=0)
    assert shape.device.type == "cpu"
    torch.testing.assert_close(weight, before, rtol=0, atol=0)


def test_groupwise_quantizes_before_bf16_scale_cast():
    import torch
    from qlean.modifiers_engine._runtime.plan import _int4_groupwise_weight_quant

    row = [1, 0.2141, -0.2141, 0, 0, 0, 0, 0] * 4
    raw = torch.tensor([row], dtype=torch.float32)
    packed, scale, _ = _int4_groupwise_weight_quant(raw, 32)
    expected = _pack_offset_words([[7, 1, -1, 0, 0, 0, 0, 0] * 4])
    torch.testing.assert_close(packed, expected, rtol=0, atol=0)
    torch.testing.assert_close(scale, torch.tensor([[1 / 7]], dtype=torch.bfloat16), rtol=0, atol=0)
    assert round(0.2141 / float(scale[0, 0])) == 2


@pytest.mark.parametrize("target", ["fp32", "bf16"])
def test_groupwise_provider_decodes_independent_words(target):
    import torch
    from qlean.modifiers_engine._runtime.dequant_providers import int4_groupwise

    raw = torch.tensor([[0x76543210] * 4 + [0xFEDCBA98 - 2 ** 32] * 4], dtype=torch.int32)
    scales = torch.tensor([[0.5, 2]], dtype=torch.bfloat16)
    ctx = SimpleNamespace(weight_map={"x.weight_scale": "other.safetensors"},
                          transform_cfg={"dequant_dtype": target, "int4_group_size": 32},
                          get_tensor=lambda key: scales)
    dtype = torch.float32 if target == "fp32" else torch.bfloat16
    expected = torch.tensor([[v * 0.5 for v in range(-8, 0)] * 4
                              + [v * 2 for v in range(8)] * 4], dtype=dtype)
    torch.testing.assert_close(int4_groupwise(ctx, "x.weight_packed", raw), expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape, dtype_name, group_size", [
    ((2, 32), "float32", 0), ((2, 32), "float32", True),
    ((2, 32), "float32", 2.5), ((2, 32), "float32", -2),
    ((2, 33), "float32", 32), ((2, 7), "float32", -1),
    ((0, 32), "float32", 32), ((2, 0), "float32", 32),
    ((1, 2, 32), "float32", 32), ((2, 32), "int8", 32),
    ((2, 32), "float64", 32),
])
def test_groupwise_rejects_invalid_input(shape, dtype_name, group_size):
    import torch
    from qlean.modifiers_engine._runtime.plan import _int4_groupwise_weight_quant

    with pytest.raises(ValueError, match="INT4 groupwise"):
        _int4_groupwise_weight_quant(torch.zeros(shape, dtype=getattr(torch, dtype_name)), group_size)


@pytest.mark.parametrize("kind", ["fp8_block", "mxfp4", "mxfp8"])
@pytest.mark.parametrize("naming", ["dot", "underscore"])
def test_passthrough_keeps_bytes_and_normalizes_keys(monkeypatch, kind, naming):
    import torch
    from qlean.modifiers_engine._runtime import plan
    from qlean.modifiers_engine._runtime.region_router import normalize_regions, spec_plan_fn

    if kind == "mxfp4":
        raw = torch.full((2, 17), -15, dtype=torch.int8)
        scale = torch.tensor([[126, -128], [127, -127]], dtype=torch.int8)
        name, scale_name = "x.weight_packed", "x.weight_scale"
    else:
        raw = torch.tensor([[1, -2] * 17] * 2, dtype=torch.float32).to(torch.float8_e4m3fn)
        scale = (torch.tensor([[0.5]], dtype=torch.float32) if kind == "fp8_block"
                 else torch.tensor([[126, 128], [127, 129]], dtype=torch.uint8))
        name, scale_name = "x.weight", "x.weight_scale_inv" if kind == "fp8_block" else "x.scale"
    reads = []

    def get(key):
        reads.append(key)
        assert key == scale_name
        return scale

    def forbidden(*args, **kwargs):
        pytest.fail("Passthrough must not quantize")

    monkeypatch.setattr(plan, "weight_quant", forbidden)
    ctx = SimpleNamespace(weight_map={name: "a", scale_name: "b"}, get_tensor_cpu=get)
    cfg = {"_regions": normalize_regions([{"name": "pass", "match": "*", "src": kind, "dst": "passthrough"}]),
           "scale_naming": naming}
    op, = spec_plan_fn(name, list(raw.shape), cfg, weight_map=ctx.weight_map).ops
    before, scale_before = raw.view(torch.uint8).clone(), scale.view(torch.uint8).clone()
    actual = plan.execute_op(op, raw, cfg, ctx, src_name=name)
    out_scale = "x.scale" if naming == "dot" else "x.weight_scale"
    assert [k for k, _ in actual] == plan.op_output_names(op) == ["x.weight", out_scale]
    torch.testing.assert_close(actual[0][1].view(torch.uint8), before, rtol=0, atol=0)
    torch.testing.assert_close(actual[1][1].view(torch.uint8), scale_before, rtol=0, atol=0)
    assert actual[0][1].dtype == (torch.uint8 if kind == "mxfp4" else raw.dtype)
    assert actual[1][1].dtype == (torch.uint8 if kind == "mxfp4" else scale.dtype)
    assert actual[0][1].data_ptr() == raw.data_ptr()
    assert reads == [scale_name]


@pytest.mark.parametrize("fault", ["ctx", "src_name", "scale", "dtype", "shape", "unsupported"])
def test_passthrough_rejects_incomplete_contract(fault):
    import torch
    from qlean.modifiers_engine._runtime.plan import execute_op

    raw = torch.zeros((2, 16), dtype=torch.uint8)
    scale = torch.ones((2, 1), dtype=torch.uint8)
    if fault == "dtype":
        raw = raw.float()
    if fault == "shape":
        scale = torch.ones((1, 1), dtype=torch.uint8)
    ctx = SimpleNamespace(weight_map={} if fault == "scale" else {"x.scale": "b"},
                          get_tensor_cpu=lambda key: scale)
    op = ("passthrough_quant", "x.weight", "nvfp4" if fault == "unsupported" else "mxfp4")
    with pytest.raises(ValueError):
        execute_op(op, raw, {}, None if fault == "ctx" else ctx,
                   src_name=None if fault == "src_name" else "x.weight")


@pytest.mark.parametrize("kind", ["copy", "cast", "quant", "expert_quant",
                                 "quant_pack_acext", "quant_int4_groupwise", "passthrough_quant"])
def test_all_op_types_emit_predicted_keys(kind):
    import torch
    from qlean.modifiers_engine._runtime.plan import OP_TYPES, execute_op, op_output_names

    assert OP_TYPES == {"copy", "cast", "quant", "expert_quant", "quant_pack_acext",
                        "quant_int4_groupwise", "passthrough_quant"}
    weight = torch.full((64, 64), 7.0, dtype=torch.float32)
    ctx = None
    if kind == "copy":
        op, expected = (kind, "x.weight"), {"x.weight": weight}
    elif kind == "cast":
        op, expected = (kind, "x.weight", "bf16"), {"x.weight": weight.to(torch.bfloat16)}
    elif kind == "quant_int4_groupwise":
        op = (kind, "x.weight", 32)
        expected = {"x.weight_packed": torch.full((64, 8), -1, dtype=torch.int32),
                    "x.weight_scale": torch.ones((64, 2), dtype=torch.bfloat16),
                    "x.weight_shape": torch.tensor([64, 64], dtype=torch.int64)}
    elif kind == "passthrough_quant":
        weight = torch.full((64, 32), 0x77, dtype=torch.uint8)
        scale = torch.full((64, 2), 127, dtype=torch.uint8)
        ctx = SimpleNamespace(weight_map={"x.scale": "b"}, get_tensor_cpu=lambda key: scale)
        op = (kind, "x.weight", "mxfp4")
        expected = {"x.weight": weight, "x.weight_scale": scale}
    elif kind == "quant":
        op = (kind, "x.weight", "FP8_DYNAMIC")
        expected = {"x.weight": torch.full((64, 64), 448, dtype=torch.float32),
                    "x.weight_scale": torch.full((64, 1), 1 / 64, dtype=torch.float32)}
    else:
        if kind == "expert_quant":
            weight = torch.full((1, 64, 128), 7.0, dtype=torch.float32)
            op = (kind, "x.weight", 0, "gate", True, "acext_int4")
        else:
            op = (kind, "x.weight")
        expected = {"x.weight": torch.full((64, 32), 0x77, dtype=torch.int8),
                    "x.weight_scale": torch.ones((64, 1), dtype=torch.float32)}
    actual = dict(execute_op(op, weight, {}, ctx, src_name="x.weight"))
    if kind == "quant":
        assert actual["x.weight"].dtype == torch.float8_e4m3fn
    assert set(actual) == set(expected) == set(op_output_names(op))
    for name in expected:
        value = actual[name].float() if kind == "quant" and name == "x.weight" else actual[name]
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


def test_unknown_op_rejected_explicitly():
    import torch
    from qlean.modifiers_engine._runtime.plan import execute_op

    with pytest.raises(ValueError, match="^Unknown op type 'unknown'; registered:"):
        execute_op(("unknown", "x.weight"), torch.ones((2, 4), dtype=torch.float32), {})
