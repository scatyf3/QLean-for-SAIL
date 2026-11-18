# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Auto-recipe generation with temporary templates and small shards."""

from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.cpu]


@pytest.fixture
def auto_case(tmp_path, monkeypatch, tensor_factory):
    from safetensors.torch import save_file
    from qlean.core import auto_recipe

    template_text = auto_recipe.files("qlean.configs").joinpath(
        "day0/_arch/default.yaml",
    ).read_text(encoding="utf-8")
    resources = tmp_path / "configs"
    template = resources / "day0" / "_arch" / "default.yaml"
    template.parent.mkdir(parents=True)
    template.write_text(template_text, encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(auto_recipe, "files", lambda package: resources)

    def build(keys, source_format="BF16", target="INT8_W8A8", architecture=None):
        save_file({key: tensor_factory((2, 4)) for key in keys}, str(source / "a.safetensors"))
        recipe = auto_recipe.build_auto_recipe(source_format, target, str(source), architecture)
        return recipe["quant_stage"]["quant_modifiers"]["GeneralQuantEngineModifier"]

    return SimpleNamespace(build=build, template=template, text=template_text, source=source)


@pytest.mark.parametrize("source_format, scale_keys", [
    ("BF16", ["model.norm.weight_scale_inv", "model.norm.scale"]),
    ("FP8", ["model.norm.weight_scale_inv", "model.norm.scale", "model.norm.weight_scale"]),
    ("NVFP4", ["model.norm.weight_scale", "model.norm.weight_scale_2", "model.norm.input_scale"]),
])
def test_source_scales_do_not_create_regions(auto_case, source_format, scale_keys):
    params = auto_case.build(["model.proj.weight", *scale_keys], source_format)
    assert [region["name"] for region in params["regions"]] == ["weights", "fallback"]
    assert params["regions"][0]["match"] == ["*.weight"]


def test_zero_hit_regions_and_or_patterns_are_pruned(auto_case):
    from qlean.modifiers_engine._runtime.region_router import spec_plan_fn, validate_regions

    params = auto_case.build(["model.experts.0.weight", "model.proj.bias"])
    assert params["regions"] == [
        {"name": "experts", "match": ["*.experts.*.weight"], "src": "bf16", "dst": "W8A8"},
        {"name": "fallback", "match": "*"},
    ]
    validate_regions(params["regions"])
    cfg = {"_regions": params["regions"]}
    assert spec_plan_fn("model.experts.0.weight", [2, 4], cfg).ops == [
        ("quant", "model.experts.0.weight", "W8A8", None),
    ]
    assert spec_plan_fn("model.proj.bias", [2], cfg).ops == [("copy", "model.proj.bias")]


def test_fallback_is_retained_without_weight_hits(auto_case):
    params = auto_case.build(["counter"])
    assert params["regions"] == [{"name": "fallback", "match": "*"}]


@pytest.mark.parametrize("source_format, src", [("FP8", "fp8_block"), ("NVFP4", "nvfp4")])
def test_quantized_ignore_generates_bf16_cast(auto_case, source_format, src):
    from qlean.modifiers_engine._runtime.region_router import spec_plan_fn, validate_regions

    params = auto_case.build(["lm_head.weight", "model.proj.weight"], source_format)
    assert params["regions"][0] == {
        "name": "ignore", "match": ["*lm_head.*"], "src": src, "dst": "bf16",
    }
    validate_regions(params["regions"])
    result = spec_plan_fn("lm_head.weight", [2, 4], {"_regions": params["regions"]})
    assert result.meta["src"] == src
    assert result.ops == [("cast", "lm_head.weight", "bf16")]


@pytest.mark.parametrize("architecture, qwen_vl", [
    (None, False), ("ToyModel", False),
    ("Qwen3VLForConditionalGeneration", True), ("qwen3_vl", True),
])
@pytest.mark.parametrize("target, scheme", [("INT8_W8A8", "W8A8"), ("MXFP4", "MXFP4")])
def test_architecture_visual_mtp_and_gate_rules(auto_case, architecture, qwen_vl, target, scheme):
    from qlean.modifiers_engine._runtime.region_router import spec_plan_fn, validate_regions

    special = ("visual.proj.weight", "model.visual.proj.weight", "mtp.proj.weight",
               "model.mtp.proj.weight")
    gate = "model.layers.0.mlp.gate.weight"
    regular = "model.layers.0.mlp.up_proj.weight"
    params = auto_case.build([*special, gate, regular, "lm_head.weight"], target=target,
                            architecture=architecture)
    validate_regions(params["regions"])
    cfg = {"_regions": params["regions"]}
    ignored = {"lm_head.weight"}
    if qwen_vl or target == "MXFP4":
        ignored.update(special)
    if qwen_vl:
        ignored.add(gate)
    for key in (*special, gate, regular, "lm_head.weight"):
        result = spec_plan_fn(key, [2, 4], cfg)
        expected = [("copy", key)] if key in ignored else [("quant", key, scheme, None)]
        assert result.ops == expected, key
        assert result.meta["region"] == ("ignore" if key in ignored else "weights"), key


@pytest.mark.parametrize("source_format, src", [("BF16", "bf16"), ("FP8", "fp8_block"), ("NVFP4", "nvfp4")])
@pytest.mark.parametrize("target, scheme", [
    ("INT8_W8A8", "W8A8"), ("FP8_DYNAMIC", "FP8_DYNAMIC"),
    ("FP8_BLOCK", "FP8_BLOCK"), ("MXFP4", "MXFP4"),
])
def test_generated_output_configuration(auto_case, source_format, src, target, scheme):
    params = auto_case.build(["model.proj.weight"], source_format, target)
    assert params["primary_scheme"] == scheme
    assert params["config_writer"] == "compressed_tensors"
    assert params["dequant_dtype"] == "fp32"
    assert params["sglang_wa"] == (target == "MXFP4")
    assert params["scale_naming"] == "underscore"
    assert params["regions"][0] == {
        "name": "weights", "match": ["*.weight"], "src": src, "dst": scheme,
    }
    assert params.get("dequant_block_size") == (128 if source_format == "FP8" else None)
    assert params.get("nvfp4_group_size") == (16 if source_format == "NVFP4" else None)


@pytest.mark.parametrize("source_format", ["FP8", "NVFP4"])
def test_bf16_output_disables_quantization_config(auto_case, source_format):
    params = auto_case.build(["model.proj.weight"], source_format, "BF16")
    assert params["primary_scheme"] is None
    assert params["config_writer"] == "none"
    assert params["dequant_dtype"] == "bf16"
    assert params["sglang_wa"] is False
    assert params["regions"][0]["dst"] == "bf16"


def test_generation_does_not_pollute_template_or_later_calls(auto_case):
    keys = ["lm_head.weight", "visual.proj.weight", "model.proj.weight"]
    first = auto_case.build(keys, target="MXFP4", architecture="qwen3_vl")
    first["regions"][0]["match"].append("polluted.*")
    second = auto_case.build(keys)
    assert second["regions"] == [
        {"name": "ignore", "match": ["*lm_head.*"], "src": "bf16"},
        {"name": "weights", "match": ["*.weight"], "src": "bf16", "dst": "W8A8"},
        {"name": "fallback", "match": "*"},
    ]
    assert second["sglang_wa"] is False
    assert auto_case.template.read_text(encoding="utf-8") == auto_case.text
