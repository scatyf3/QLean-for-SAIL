# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""New and legacy W8A8 checkpoint conversions against handwritten references."""

from datetime import datetime
import json

import pytest

pytestmark = pytest.mark.integration


def _scale_key(name, naming):
    return name + "_scale" if naming == "underscore" else name.removesuffix(".weight") + ".scale"


def _expectations(case, naming, quantize_experts):
    import torch

    expected, references, source_of = {}, {}, {}

    def quantized(name, codes, scales, reference, source_name):
        expected[name] = codes
        scale_name = _scale_key(name, naming)
        expected[scale_name] = scales
        references[name] = reference.float()
        source_of[name] = source_of[scale_name] = source_name

    if case.source_format == "BF16":
        codes = torch.tensor([
            [-127, -2, -2, 0, 0, 2, 2, 127],
            [-127, -2, -2, 0, 0, 2, 2, 127],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=torch.int8)
        quantized("model.proj.weight", codes,
                  torch.tensor([[1], [2], [2 ** -23]], dtype=torch.float32),
                  case.tensors["model.proj.weight"], "model.proj.weight")
        quantized("model.other.weight", -codes,
                  torch.tensor([[2], [4], [2 ** -23]], dtype=torch.float32),
                  case.tensors["model.other.weight"], "model.other.weight")
        expert_name = "model.mlp.experts.gate_up_proj"
        if quantize_experts:
            expert_codes = torch.tensor([
                [127, -127, 0], [-127, 0, 127], [0, 0, 0], [0, 127, -127],
                [127, 0, -127], [0, -127, 127], [0, 0, 0], [-127, 127, 0],
            ], dtype=torch.int8)
            for expert, sign, factor in ((0, 1, 1), (1, -1, 2)):
                scales = torch.tensor([[factor], [2 * factor], [2 ** -23], [4 * factor],
                                       [2 * factor], [factor], [2 ** -23], [4 * factor]],
                                      dtype=torch.float32)
                for role, start in (("gate", 0), ("up", 4)):
                    part = slice(start, start + 4)
                    name = f"model.mlp.experts.{expert}.{role}_proj.weight"
                    quantized(name, expert_codes[part] * sign, scales[part],
                              case.tensors[expert_name][expert].t()[part], expert_name)
        else:
            expected[expert_name] = case.tensors[expert_name]
            source_of[expert_name] = expert_name
    else:
        codes = torch.tensor([-127, -64, -32, -16, 0, 16, 32, 64], dtype=torch.int8).repeat(128, 16)
        reference = torch.tensor([-127, -63.5, -31.75, -15.875, 0, 15.875, 31.75, 63.5],
                                 dtype=torch.float32).repeat(128, 16)
        for name, sign, scale in (("model.proj.weight", 1, 1), ("model.other.weight", -1, 2)):
            quantized(name, codes * sign, torch.full((128, 1), scale, dtype=torch.float32),
                      reference * sign * scale, name)

    for name in ("model.proj.bias", "model.norm.weight", "lm_head.weight", "step"):
        expected[name] = case.tensors[name]
        source_of[name] = name
    return expected, references, source_of


def _assert_artifacts(case, conversion, assert_tensor_equal, *, backend="new",
                      naming="underscore", quantize_experts=True):
    import torch
    from safetensors import safe_open

    expected, references, source_of = _expectations(case, naming, quantize_experts)
    index = json.loads((case.output / "model.safetensors.index.json").read_text())
    actual, observed_map, shard_stats = {}, {}, []
    shard_paths = sorted(case.output.glob("*.safetensors"))
    assert shard_paths
    for path in shard_paths:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                assert name not in actual, f"Duplicate key across shards: {name}"
                actual[name] = handle.get_tensor(name)
                observed_map[name] = path.name
            if backend == "new":
                metadata = handle.metadata()
                assert metadata["format"] == "pt"
                shard_stats.append(json.loads(metadata["qlean_shard_stats"]))
    assert index["weight_map"] == observed_map
    assert set(actual) == set(expected)
    assert not (set(actual) & case.source_scales)
    assert not list(case.output.glob("*.qlean_tmp"))
    for name, tensor in expected.items():
        assert_tensor_equal(actual[name], tensor, scheme=f"W8A8/checkpoint/{name}", seed=0)
    for name, reference in references.items():
        scales = actual[_scale_key(name, naming)]
        error = (actual[name].float() * scales - reference).abs()
        assert_tensor_equal(error <= scales / 2, torch.ones_like(error, dtype=torch.bool),
                            scheme=f"W8A8/checkpoint-bound/{name}", seed=0)

    if backend == "new" or case.source_format == "BF16":
        assert {p.name for p in shard_paths} == set(case.shard_names)
        assert observed_map == {name: case.weight_map[source_of[name]] for name in expected}
    if backend == "new":
        from qlean.modifiers_engine._runtime.plan import make_hooks
        from qlean.modifiers_engine._runtime.region_router import spec_plan_fn

        modifier, = conversion.compressor.modifiers
        cfg = {"_regions": modifier.regions, "hidden_size": 3, "scale_naming": naming}
        _, predict = make_hooks(spec_plan_fn)
        for filename in case.shard_names:
            with safe_open(str(case.path / filename), framework="pt") as handle:
                src_meta = {name: {"shape": list(handle.get_slice(name).get_shape())}
                            for name in handle.keys() if name not in case.source_scales}
            predicted = set(predict(src_meta, cfg, weight_map=case.weight_map))
            assert predicted == {name for name, shard in observed_map.items() if shard == filename}
        assert sum(s.get("processed_tensors", 0) for s in shard_stats) == (
            7 if case.source_format == "BF16" else 6)
        assert sum(s.get("quant_count", 0) for s in shard_stats) == (
            3 if case.source_format == "BF16" and quantize_experts else 2)

    assert (case.output / "tokenizer_config.json").read_bytes() == (
        case.path / "tokenizer_config.json").read_bytes()
    assert all(m.initialized_ and m.started_ and m.ended_ for m in conversion.compressor.modifiers)


def _assert_config(case, *, expected_ignore=("lm_head", "model.norm")):
    config = json.loads((case.output / "config.json").read_text())
    assert config["model_type"] == case.config["model_type"]
    assert config["hidden_size"] == 3
    assert config["test_sentinel"] == "preserve-me"
    qconfig = config["quantization_config"]
    assert qconfig["quant_method"] == "compressed-tensors"
    assert qconfig["format"] == "int-quantized"
    assert qconfig["quantization_status"] == "compressed"
    assert set(qconfig["config_groups"]) == {"W8A8"}
    scheme = qconfig["config_groups"]["W8A8"]
    assert scheme["targets"] == ["Linear"]
    for key, strategy, dynamic in (("weights", "channel", False),
                                   ("input_activations", "token", True)):
        assert scheme[key]["num_bits"] == 8
        assert scheme[key]["type"] == "int"
        assert scheme[key]["symmetric"] is True
        assert scheme[key]["strategy"] == strategy
        assert scheme[key]["dynamic"] is dynamic
    assert set(qconfig["ignore"]) == set(expected_ignore)

    def check_removed(value):
        if isinstance(value, dict):
            assert not ({"scale_dtype", "zp_dtype"} & value.keys())
            for child in value.values():
                check_removed(child)
        elif isinstance(value, list):
            for child in value:
                check_removed(child)

    check_removed(config)
    return config


def _assert_metadata(case, conversion, expected_chain):
    import yaml

    metadata = json.loads((case.output / "qlean_config.json").read_text())
    expected = {
        "model_name": "QLeanTestModel", "model_path": str(case.path),
        "input_format": "user_specified", "output_format": "user_specified",
        "recipe_path": str(conversion.recipe_path), "recipe_content": conversion.recipe,
        "modifier_chain": expected_chain, "scheme": None, "cli_args": conversion.argv[1:],
    }
    assert {key: metadata[key] for key in expected} == expected
    assert set(metadata) == set(expected) | {"timestamp", "qlean_version"}
    assert isinstance(metadata["qlean_version"], str) and metadata["qlean_version"]
    datetime.fromisoformat(metadata["timestamp"])
    assert yaml.safe_load(conversion.recipe_path.read_text()) == conversion.recipe
    assert [type(m).__name__ for m in conversion.compressor.modifiers] == expected_chain


@pytest.mark.cpu
@pytest.mark.parametrize("naming", ["underscore", "dot"])
def test_new_bf16_checkpoint_roundtrip(checkpoint_factory, build_conversion,
                                       assert_tensor_equal, naming):
    from qlean.args import OtherArguments

    case = checkpoint_factory()
    conversion = build_conversion(case, scale_naming=naming)
    conversion.compressor.post_train(OtherArguments())
    _assert_artifacts(case, conversion, assert_tensor_equal, naming=naming)
    config = _assert_config(case)
    assert "expert_dtype" not in config
    assert "expert_dtype" not in config["quantization_config"]
    _assert_metadata(case, conversion, ["GeneralQuantEngineModifier"])


@pytest.mark.cpu
@pytest.mark.parametrize("text_quant", [None, False, True], ids=["no-text", "text-only", "text-quant"])
@pytest.mark.parametrize("quantize_experts", [True, False], ids=["expert-int8", "expert-copy"])
def test_new_config_syncs_text_and_expert_markers(checkpoint_factory, build_conversion,
                                                  assert_tensor_equal, text_quant,
                                                  quantize_experts):
    from qlean.args import OtherArguments

    case = checkpoint_factory(text_quant=text_quant, expert_markers=True)
    conversion = build_conversion(case, quantize_experts=quantize_experts)
    conversion.compressor.post_train(OtherArguments())
    _assert_artifacts(case, conversion, assert_tensor_equal, quantize_experts=quantize_experts)
    expected_ignore = ("lm_head", "model.norm")
    if not quantize_experts:
        expected_ignore += ("model.mlp.experts",)
    config = _assert_config(case, expected_ignore=expected_ignore)
    sections = [config, config["quantization_config"]]
    if text_quant is None:
        assert "text_config" not in config
    else:
        text = config["text_config"]
        assert text["hidden_size"] == 3 and text["test_sentinel"] == "text-preserve"
        sections.append(text)
        if text_quant:
            assert text["quantization_config"] == config["quantization_config"]
            sections.append(text["quantization_config"])
        else:
            assert "quantization_config" not in text
    for section in sections:
        if quantize_experts:
            assert section["expert_dtype"] == "int8"
        else:
            assert "expert_dtype" not in section


@pytest.mark.gpu
@pytest.mark.parametrize("text_quant", [None, False, True], ids=["no-text", "text-only", "text-quant"])
def test_legacy_bf16_checkpoint_roundtrip(checkpoint_factory, build_conversion,
                                          assert_tensor_equal, text_quant):
    from qlean.args import OtherArguments

    case = checkpoint_factory(text_quant=text_quant)
    conversion = build_conversion(case, backend="legacy")
    conversion.compressor.post_train(OtherArguments())
    _assert_artifacts(case, conversion, assert_tensor_equal, backend="legacy")
    config = _assert_config(case)
    if text_quant is None:
        assert "text_config" not in config
    else:
        assert config["text_config"] == {"hidden_size": 3, "test_sentinel": "text-preserve"}
    _assert_metadata(case, conversion, ["generalDay0Modifier"])


@pytest.mark.gpu
@pytest.mark.parametrize("backend,naming", [("new", "underscore"), ("new", "dot"),
                                            ("legacy", "underscore")])
def test_fp8_cross_shard_scales_to_w8a8(checkpoint_factory, build_conversion,
                                       assert_tensor_equal, backend, naming):
    from qlean.args import OtherArguments

    case = checkpoint_factory(source_format="FP8")
    for weight in ("model.proj.weight", "model.other.weight"):
        assert case.weight_map[weight] != case.weight_map[weight + "_scale_inv"]
    conversion = build_conversion(case, backend=backend, scale_naming=naming)
    conversion.compressor.post_train(OtherArguments())
    _assert_artifacts(case, conversion, assert_tensor_equal, backend=backend, naming=naming)
    expected_ignore = ("lm_head", "model.norm") if backend == "new" else ("lm_head",)
    _assert_config(case, expected_ignore=expected_ignore)
    chain = (["GeneralQuantEngineModifier"] if backend == "new"
             else ["dequantModifier", "fp8BasedDay0Modifier"])
    _assert_metadata(case, conversion, chain)
