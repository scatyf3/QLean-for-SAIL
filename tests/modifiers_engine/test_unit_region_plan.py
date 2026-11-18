# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Region routing, Plan outputs, and quantizer arguments with small inputs."""

from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.cpu]


@pytest.fixture
def quantizer_spy(monkeypatch):
    import torch
    from qlean.modifiers_engine._runtime import plan

    calls = []

    def record(tensor, scheme):
        codes = torch.zeros_like(tensor, dtype=torch.int8)
        scale = torch.ones((*tensor.shape[:-1], 1), dtype=torch.float32, device=tensor.device)
        calls.append(SimpleNamespace(tensor=tensor.clone(), scheme=scheme,
                                     codes=codes, scale=scale))
        return codes, scale

    monkeypatch.setattr(plan, "weight_quant", record)
    return plan, calls


@pytest.mark.parametrize("scale_naming, scale_key", [
    ("underscore", "model.proj.weight_scale"), ("dot", "model.proj.scale"),
])
@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_first_match_drives_op_and_quantizer(
    recipe_factory, tensor_factory, quantizer_spy, assert_tensor_equal,
    scale_naming, scale_key, dtype_name, batched, noncontiguous,
):
    import torch
    from qlean.modifiers_engine._runtime import region_router as router

    executor, calls = quantizer_spy
    recipe = recipe_factory(regions=[
        {"name": "specific", "match": ["*.unused", "model.proj.*"], "dst": "W8A8"},
        {"name": "general", "match": "*.weight", "dst": "FP8_DYNAMIC"},
        {"name": "fallback", "match": "*"},
    ])
    regions = recipe["quant_stage"]["quant_modifiers"]["GeneralQuantEngineModifier"]["regions"]
    router.validate_regions(regions)
    cfg = {"_regions": regions, "scale_naming": scale_naming}
    weight = tensor_factory((2, 4), pattern="arange", dtype=getattr(torch, dtype_name),
                            experts=2 if batched else None)
    if noncontiguous:
        weight = weight.transpose(-1, -2).contiguous().transpose(-1, -2)
        assert not weight.is_contiguous()
    before = weight.clone()
    result = router.spec_plan_fn("model.proj.weight", list(weight.shape), cfg)

    assert result.meta["region"] == "specific"
    expected_spec = None if scale_naming == "underscore" else scale_key
    assert result.ops == [("quant", "model.proj.weight", "W8A8", expected_spec)]
    outputs = executor.execute_op(result.ops[0], weight, cfg)
    assert [name for name, _ in outputs] == ["model.proj.weight", scale_key]
    assert executor.op_output_names(result.ops[0]) == ["model.proj.weight", scale_key]
    assert len(calls) == 1
    assert calls[0].scheme == "W8A8"
    assert_tensor_equal(calls[0].tensor, before, scheme="W8A8/plan-input", seed=0)
    assert_tensor_equal(weight, before, scheme="W8A8/plan-input", seed=0)
    assert outputs[0][1] is calls[0].codes
    assert outputs[1][1] is calls[0].scale


@pytest.mark.parametrize("has_scale, expected_region, expected_scheme", [
    (True, "scaled", "W8A8"), (False, "unscaled", "FP8_DYNAMIC"),
])
def test_when_condition_selects_actual_plan(has_scale, expected_region, expected_scheme):
    from qlean.modifiers_engine._runtime import region_router as router

    name = "model.proj.weight"
    regions = [
        {"name": "scaled", "match": "*.weight", "src": "fp8_block",
         "dst": "W8A8", "when": "has_source_scale"},
        {"name": "unscaled", "match": "*.weight", "src": "fp8_block",
         "dst": "FP8_DYNAMIC", "when": "no_source_scale"},
        {"name": "fallback", "match": "*"},
    ]
    router.validate_regions(regions)
    weight_map = {name: "weights.safetensors"}
    if has_scale:
        weight_map[f"{name}_scale_inv"] = "scales.safetensors"
    result = router.spec_plan_fn(name, [2, 4], {"_regions": regions}, weight_map=weight_map)
    assert result.meta["region"] == expected_region
    assert result.ops == [("quant", name, expected_scheme, None)]


def test_fallback_copies_without_quantization(recipe_factory, tensor_factory, quantizer_spy):
    from qlean.modifiers_engine._runtime import region_router as router

    executor, calls = quantizer_spy
    regions = recipe_factory()["quant_stage"]["quant_modifiers"]["GeneralQuantEngineModifier"]["regions"]
    cfg = {"_regions": regions}
    weight = tensor_factory((4,), pattern="arange")
    result = router.spec_plan_fn("model.proj.bias", [4], cfg)
    assert result.meta["region"] == "fallback"
    assert result.ops == [("copy", "model.proj.bias")]
    outputs = executor.execute_op(result.ops[0], weight, cfg)
    assert [name for name, _ in outputs] == ["model.proj.bias"]
    assert outputs[0][1] is weight
    assert calls == []


@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
def test_experts_transpose_before_gate_up_split(
    tensor_factory, quantizer_spy, assert_tensor_equal, dtype_name,
):
    import torch
    from qlean.modifiers_engine._runtime import region_router as router

    executor, calls = quantizer_spy
    weight = tensor_factory((3, 8), experts=2, pattern="arange", dtype=getattr(torch, dtype_name))
    before = weight.clone()
    cfg = {"_regions": [
        {"name": "experts", "match": "*.experts.gate_up_proj", "dst": "W8A8",
         "expert_mode": "unpack", "transpose": True, "split_gate_up": True},
        {"name": "fallback", "match": "*"},
    ]}
    result = router.spec_plan_fn("model.experts.gate_up_proj", list(weight.shape), cfg)
    expected_ops = [
        ("expert_quant", "model.experts.0.gate_proj.weight", 0, "gate", True, "W8A8", None),
        ("expert_quant", "model.experts.0.up_proj.weight", 0, "up", True, "W8A8", None),
        ("expert_quant", "model.experts.1.gate_proj.weight", 1, "gate", True, "W8A8", None),
        ("expert_quant", "model.experts.1.up_proj.weight", 1, "up", True, "W8A8", None),
    ]
    assert result.ops == expected_ops
    expected_inputs = [
        [[0, 8, 16], [1, 9, 17], [2, 10, 18], [3, 11, 19]],
        [[4, 12, 20], [5, 13, 21], [6, 14, 22], [7, 15, 23]],
        [[24, 32, 40], [25, 33, 41], [26, 34, 42], [27, 35, 43]],
        [[28, 36, 44], [29, 37, 45], [30, 38, 46], [31, 39, 47]],
    ]
    for op, expected_op in zip(result.ops, expected_ops):
        outputs = executor.execute_op(op, weight, cfg)
        expected_keys = [expected_op[1], f"{expected_op[1]}_scale"]
        assert [name for name, _ in outputs] == expected_keys
        assert executor.op_output_names(op) == expected_keys
        assert outputs[0][1] is calls[-1].codes
        assert outputs[1][1] is calls[-1].scale
    assert len(calls) == 4
    for call, expected in zip(calls, expected_inputs):
        assert call.scheme == "W8A8"
        assert_tensor_equal(call.tensor, torch.tensor(expected, dtype=getattr(torch, dtype_name)),
                            scheme="W8A8/expert-input", seed=0)
    assert_tensor_equal(weight, before, scheme="W8A8/expert-input", seed=0)


@pytest.mark.parametrize("dtype_name", ["int8", "uint8", "float64"])
def test_quant_op_rejects_unsupported_storage_before_quantizer(tensor_factory, quantizer_spy, dtype_name):
    import torch

    executor, calls = quantizer_spy
    weight = tensor_factory((2, 4), dtype=getattr(torch, dtype_name))
    with pytest.raises(ValueError, match="quant requires a 2- or 4-byte source"):
        executor.execute_op(("quant", "x.weight", "W8A8", None), weight, {})
    assert calls == []


@pytest.mark.parametrize("region, message", [
    ({"name": "bad", "match": "re:.*", "dst": "W8A8"}, "regex"),
    ({"name": "bad", "match": "*", "src": "fp8_block"}, "missing dst"),
    ({"name": "bad", "match": "*", "src": "bf16", "dst": "bf16"}, "no-op"),
    ({"name": "bad", "match": "*", "chunk_rows": True}, "chunk_rows"),
])
def test_invalid_regions_fail_before_execution(region, message):
    from qlean.modifiers_engine._runtime.region_router import validate_regions

    with pytest.raises(ValueError, match=message):
        validate_regions([region, {"name": "fallback", "match": "*"}])


@pytest.mark.parametrize("weight_map", [None, {}, {"x.weight_scale_inv": "scales.safetensors"}])
def test_explicit_always_ignores_scale_presence(weight_map):
    from qlean.modifiers_engine._runtime import region_router as router

    regions = [
        {"name": "unconditional", "match": "*.weight", "src": "fp8_block",
         "when": "always", "dst": "W8A8"},
        {"name": "fallback", "match": "*"},
    ]
    router.validate_regions(regions)
    result = router.spec_plan_fn("x.weight", [2, 4], {"_regions": regions}, weight_map)
    assert result.meta["region"] == "unconditional"
    assert result.ops == [("quant", "x.weight", "W8A8", None)]


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "2"])
def test_invalid_chunk_rows_reports_region(value):
    from qlean.modifiers_engine._runtime.region_router import validate_regions

    regions = [
        {"name": "stream", "match": "*.weight", "src": "mxfp8", "dst": "W8A8",
         "chunked": True, "chunk_rows": value},
        {"name": "fallback", "match": "*"},
    ]
    with pytest.raises(ValueError, match="region 'stream': chunk_rows must be a positive integer"):
        validate_regions(regions)


@pytest.mark.parametrize("fields, fragments", [
    ({"unknown_option": 1}, ["regions[1]", "unknown fields", "unknown_option"]),
    ({"src": "invalid"}, ["regions[1]", "bad", "invalid src"]),
    ({"dst": 123}, ["regions[1]", "bad", "invalid dst"]),
    ({"when": "sometimes"}, ["region 'bad'", "invalid when"]),
    ({"expert_mode": "guess"}, ["region 'bad'", "invalid expert_mode"]),
    ({"split_gate_up": "yes"}, ["region 'bad'", "split_gate_up must be bool"]),
    ({"transpose": "yes"}, ["region 'bad'", "invalid transpose"]),
    ({"chunked": "yes"}, ["region 'bad'", "chunked must be bool"]),
    ({"chunk_rows": 2}, ["region 'bad'", "requires chunked: true"]),
    ({"chunked": True, "dst": "FP8_BLOCK"}, ["region 'bad'", "row-wise dst"]),
    ({"chunked": True, "src": "fp8_block"}, ["region 'bad'", "requires src=mxfp8"]),
    ({"chunked": True, "src": "mxfp8", "expert_mode": "unpack"}, ["region 'bad'", "expert_mode"]),
    ({"chunked": True, "src": "mxfp8", "split_gate_up": True}, ["region 'bad'", "split_gate_up"]),
    ({"match": []}, ["region 'bad'", "nonempty"]),
    ({"match": ["*.weight", "re:.*"]}, ["region 'bad'", "regex"]),
])
def test_invalid_region_fields_have_actionable_location(fields, fragments):
    from qlean.modifiers_engine._runtime import region_router as router

    regions = [
        {"name": "first", "match": "first.*", "dst": "W8A8"},
        {"name": "bad", "match": "*.weight", "dst": "W8A8", **fields},
        {"name": "fallback", "match": "*"},
    ]
    with pytest.raises(ValueError) as error:
        router.validate_regions(router.normalize_regions(regions))
    assert all(fragment in str(error.value) for fragment in fragments)


def test_duplicate_region_names_are_rejected():
    from qlean.modifiers_engine._runtime.region_router import validate_regions

    with pytest.raises(ValueError, match="Duplicate region name: 'same'"):
        validate_regions([{"name": "same", "match": "x.*"}, {"name": "same", "match": "*"}])


@pytest.mark.parametrize("conflict", [
    {"expert_mode": "unpack"}, {"split_gate_up": True}, {"transpose": True},
])
def test_passthrough_cannot_change_layout(conflict):
    from qlean.modifiers_engine._runtime.region_router import validate_regions

    with pytest.raises(ValueError, match="region 'keep': passthrough cannot unpack, split or transpose"):
        validate_regions([
            {"name": "keep", "match": "*.weight", "src": "fp8_block",
             "dst": "passthrough", **conflict},
            {"name": "fallback", "match": "*"},
        ])


@pytest.mark.parametrize("shape, fragments", [
    ([3, 8], ["region 'experts'", "model.experts.gate_up_proj", "requires 3D", "2D"]),
    ([2, 3, 8], ["transpose: auto", "hidden_size"]),
])
def test_expert_layout_requires_rank_and_hidden_size(shape, fragments):
    from qlean.modifiers_engine._runtime.region_router import spec_plan_fn

    cfg = {"_regions": [
        {"name": "experts", "match": "*.experts.*", "expert_mode": "unpack", "dst": "W8A8"},
        {"name": "fallback", "match": "*"},
    ]}
    with pytest.raises(ValueError) as error:
        spec_plan_fn("model.experts.gate_up_proj", shape, cfg)
    assert all(fragment in str(error.value) for fragment in fragments)


def test_normalization_is_case_insensitive_and_does_not_mutate_recipe():
    from copy import deepcopy
    from qlean.modifiers_engine._runtime import region_router as router

    regions = [{"name": "quant", "match": ["*.weight"], "src": "FP8_BLOCK", "dst": "w8a8"},
               {"name": "fallback", "match": "*"}]
    before = deepcopy(regions)
    normalized = router.normalize_regions(regions)
    router.validate_regions(normalized)
    result = router.spec_plan_fn("x.weight", [2, 4], {"_regions": normalized})
    assert result.meta["src"] == "fp8_block"
    assert result.ops == [("quant", "x.weight", "W8A8", None)]
    normalized[0]["match"].append("changed.*")
    assert regions == before


def test_bf16_cast_plan_uses_numeric_rounding_not_storage_reinterpretation(tensor_factory, quantizer_spy):
    import torch
    from qlean.modifiers_engine._runtime.region_router import spec_plan_fn

    executor, calls = quantizer_spy
    weight = tensor_factory((1, 4), pattern="zeros")
    weight.copy_(torch.tensor([[1 + 1 / 256, 1 + 3 / 256, -1 - 1 / 256, -1 - 3 / 256]]))
    before = weight.clone()
    cfg = {"_regions": [{"name": "decode", "match": "*", "src": "int4", "dst": "bf16"}]}
    result = spec_plan_fn("x.weight_packed", [1, 4], cfg)
    assert result.ops == [("cast", "x.weight", "bf16")]
    outputs = executor.execute_op(result.ops[0], weight, cfg)
    assert executor.op_output_names(result.ops[0]) == ["x.weight"]
    assert [name for name, _ in outputs] == ["x.weight"]
    expected = torch.tensor([[1, 1 + 1 / 64, -1, -1 - 1 / 64]], dtype=torch.bfloat16)
    torch.testing.assert_close(outputs[0][1], expected, rtol=0, atol=0)
    assert torch.equal(weight, before)
    assert calls == []


def test_fp32_cast_and_unknown_dtype(tensor_factory, quantizer_spy):
    import torch

    executor, calls = quantizer_spy
    weight = tensor_factory((2, 3), pattern="arange", dtype=torch.bfloat16)
    outputs = executor.execute_op(("cast", "x.weight", "fp32"), weight, {})
    expected = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.float32)
    torch.testing.assert_close(outputs[0][1], expected, rtol=0, atol=0)
    assert outputs[0][1].device == weight.device
    assert weight.dtype == torch.bfloat16
    with pytest.raises(ValueError, match="Unknown cast dtype 'invalid'"):
        executor.execute_op(("cast", "x.weight", "invalid"), weight, {})
    assert calls == []
