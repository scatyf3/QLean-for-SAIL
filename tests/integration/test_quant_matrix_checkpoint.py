# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Every supported dequant source against every supported quant target, end to end."""

import json
import math

import pytest

pytestmark = pytest.mark.integration

SRC_KINDS = ("bf16", "fp8_block", "nvfp4", "int4", "mxfp4", "mxfp8")
DST_KINDS = ("bf16", "W8A8", "FP8_DYNAMIC", "FP8_BLOCK", "MXFP4", "acext_int4", "passthrough")

# The block/MX dequantizers and the MXFP4 encoder are Triton kernels; they need a device.
_DEVICE_SRCS = frozenset({"fp8_block", "mxfp4", "mxfp8"})
_DEVICE_DSTS = frozenset({"MXFP4"})
# `dst: passthrough` copies weight bytes plus exactly one scale tensor, so sources whose
# auxiliary metadata cannot be expressed that way (nvfp4 needs a local + a global scale,
# int4 needs a groupwise bf16 scale whose group_size lives in the recipe) are rejected.
_PASSTHROUGH_SRCS = frozenset({"bf16", "fp8_block", "mxfp4", "mxfp8"})

_COMPRESSED = {
    "W8A8": ("int-quantized", 8, "int", "channel"),
    "FP8_DYNAMIC": ("float-quantized", 8, "float", "channel"),
    "FP8_BLOCK": ("float-quantized", 8, "float", "block"),
    "MXFP4": ("mxfp4-pack-quantized", 4, "float", "group"),
}


def _matrix():
    cases = []
    for src in SRC_KINDS:
        for dst in DST_KINDS:
            if dst == "passthrough" and src not in _PASSTHROUGH_SRCS:
                continue
            if src == dst == "bf16":
                # The engine rejects bf16->bf16 as a no-op; passthrough already covers it.
                continue
            mark = (pytest.mark.gpu if src in _DEVICE_SRCS or dst in _DEVICE_DSTS
                    else pytest.mark.cpu)
            cases.append(pytest.param(src, dst, marks=mark, id=f"{src}--{dst}"))
    return cases


def _layout(dst, rows, cols):
    """Weight and scale layout a target scheme writes for a logical [rows, cols] tensor."""
    import torch

    if dst == "bf16":
        return torch.bfloat16, (rows, cols), None, None
    if dst == "W8A8":
        return torch.int8, (rows, cols), torch.float32, (rows, 1)
    if dst == "FP8_DYNAMIC":
        return torch.float8_e4m3fn, (rows, cols), torch.float32, (rows, 1)
    if dst == "FP8_BLOCK":
        return (torch.float8_e4m3fn, (rows, cols), torch.float32,
                (math.ceil(rows / 128), math.ceil(cols / 128)))
    if dst == "MXFP4":
        return (torch.uint8, (rows, math.ceil(cols / 2)),
                torch.uint8, (rows, math.ceil(cols / 32)))
    if dst == "acext_int4":
        return torch.int8, (rows, cols // 2), torch.float32, (rows, 1)
    raise ValueError(dst)


def _writer(dst):
    """Export configuration that can describe the requested target."""
    if dst in _COMPRESSED:
        return {"config_writer": "compressed_tensors", "primary_scheme": dst, "verify": "bound"}
    if dst == "acext_int4":
        # No compressed-tensors preset claims 4-bit channelwise; the caller owns the config.
        return {"config_writer": "none", "primary_scheme": None, "verify": "bound"}
    return {"config_writer": "none", "primary_scheme": None, "verify": "off"}


def _read_outputs(case):
    from safetensors import safe_open

    tensors, shards = {}, {}
    for path in sorted(case.output.glob("*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                assert name not in tensors, f"Duplicate output key: {name}"
                tensors[name] = handle.get_tensor(name)
                shards[name] = path.name
    assert json.loads((case.output / "model.safetensors.index.json").read_text())["weight_map"] == shards
    return tensors


def _bytes(tensor):
    import torch

    return tensor.reshape(-1).view(torch.uint8)


def _assert_compressed(config, dst):
    expected_format, num_bits, quant_type, strategy = _COMPRESSED[dst]
    qconfig = config["quantization_config"]
    assert qconfig["quant_method"] == "compressed-tensors"
    assert qconfig["format"] == expected_format
    weights = qconfig["config_groups"][dst]["weights"]
    assert (weights["num_bits"], weights["type"], weights["strategy"]) == (num_bits, quant_type, strategy)
    assert config["text_config"]["quantization_config"] == qconfig


def _assert_bound_check(case, scheme, status):
    report = json.loads((case.output / "qlean_verify" / "verify_run_info.json").read_text())
    assert report["violation_shards"] == 0
    assert report["error_rows"] == 0
    assert report["cross_check"]["rows_match"] is True
    assert report["by_scheme_status"] == {f"{scheme}|{status}": 1}


@pytest.mark.parametrize("src,dst", _matrix())
def test_supported_dequant_quant_matrix(checkpoint_factory, build_conversion, recipe_factory, src, dst):
    """Convert one checkpoint per src x dst pair and check keys, layout, config, and bounds."""
    import torch
    from qlean.args import OtherArguments

    case = checkpoint_factory(src=src, src_shape=(128, 128), text_quant=True)
    source = case.source
    assert source is not None and case.weight_map[source.name] != case.weight_map["lm_head.weight"]
    rows, cols = source.reference.shape
    recipe = recipe_factory(
        regions=[{"name": "weights", "match": source.name, "src": src, "dst": dst},
                 {"name": "fallback", "match": "*"}],
        num_gpus=1, resume=False, sglang_wa=False, scale_naming="underscore",
        dequant_dtype="fp32", dequant_block_size=128, int4_group_size=32, nvfp4_group_size=16,
        **_writer(dst))
    conversion = build_conversion(case, custom_recipe=recipe)
    conversion.compressor.post_train(OtherArguments())
    tensors = _read_outputs(case)
    config = json.loads((case.output / "config.json").read_text())
    copies = {"x.bias", "lm_head.weight"}

    if dst == "passthrough":
        assert config == case.config
        if src == "bf16":
            assert set(tensors) == copies | {"x.weight"}
            assert torch.equal(_bytes(tensors["x.weight"]), _bytes(source.raw))
        else:
            assert set(tensors) == copies | {"x.weight", "x.weight_scale"}
            scale, = source.scales.values()
            weight = source.raw.view(torch.uint8) if src == "mxfp4" else source.raw
            stored = scale.view(torch.uint8) if src == "mxfp4" else scale
            assert torch.equal(_bytes(tensors["x.weight"]), _bytes(weight))
            assert tensors["x.weight_scale"].dtype == stored.dtype
            assert torch.equal(_bytes(tensors["x.weight_scale"]), _bytes(stored))
        return

    if dst == "bf16":
        assert set(tensors) == copies | {"x.weight"}
        for section in (config, config["text_config"]):
            assert "quantization_config" not in section
        assert torch.equal(_bytes(tensors["x.weight"]),
                           _bytes(source.reference.to(torch.bfloat16)))
        return

    dtype, shape, scale_dtype, scale_shape = _layout(dst, rows, cols)
    assert set(tensors) == copies | {"x.weight", "x.weight_scale"}
    assert (tensors["x.weight"].dtype, tuple(tensors["x.weight"].shape)) == (dtype, shape)
    assert (tensors["x.weight_scale"].dtype, tuple(tensors["x.weight_scale"].shape)) == (
        scale_dtype, scale_shape)
    if dst in _COMPRESSED:
        _assert_compressed(config, dst)
    else:
        assert "quantization_config" not in config
        assert "quantization_config" not in config["text_config"]
    # Block schemes and unsupported layouts report a skip instead of a bound violation.
    _assert_bound_check(case, dst, "skipped_scheme" if dst == "FP8_BLOCK" else "ok")


@pytest.mark.cpu
def test_matrix_covers_every_engine_kind():
    """Fail loudly when the engine learns a format the matrix above does not exercise."""
    from qlean.modifiers_engine._runtime.region_router import DST_KINDS as ENGINE_DSTS
    from qlean.modifiers_engine._runtime.region_router import PASSTHROUGH_SRCS as ENGINE_PT
    from qlean.modifiers_engine._runtime.region_router import SRC_KINDS as ENGINE_SRCS

    assert set(SRC_KINDS) == set(ENGINE_SRCS)
    assert set(DST_KINDS) == set(ENGINE_DSTS)
    # Widening passthrough to nvfp4/int4 means the matrix owes those cells.
    assert _PASSTHROUGH_SRCS == set(ENGINE_PT)
