# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Legacy Modifier routing with stubbed tensor I/O and quantizers."""

import json
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.cpu, pytest.mark.usefixtures("isolated_state")]


@pytest.fixture(scope="module")
def legacy_modules():
    """Import before registry isolation to keep cached modules and registrations aligned."""
    from importlib import import_module

    return {name: import_module(f"qlean.modifiers.{name}.base")
            for name in ("generalDay0", "dpskV4mixGeneralDay0", "fp8BasedDay0")}


@pytest.fixture
def run_legacy(tmp_path, monkeypatch, legacy_modules):
    import torch

    def run(weights, *, backend="generalDay0", scheme="W8A8", hidden_size=3, source_map=None, **params):
        legacy = legacy_modules[backend]
        config_path = tmp_path / "config.json"
        index_path = tmp_path / "model.safetensors.index.json"
        config_path.write_text(json.dumps({"hidden_size": hidden_size}))
        index_path.write_text(json.dumps({"weight_map": source_map or {}}))
        calls, saved = [], {}

        def record(tensor, selected_scheme):
            calls.append((tensor.clone(), selected_scheme))
            return (torch.full_like(tensor, len(calls), dtype=torch.int8),
                    torch.full((tensor.shape[0], 1), len(calls) / 2, dtype=torch.float32))

        def save(tensors, path):
            saved.update(tensors)

        monkeypatch.setattr(legacy, "copy_necessary_files", lambda *args: (config_path, index_path))
        monkeypatch.setattr(legacy, "load_file_to_cuda", lambda path: weights)
        monkeypatch.setattr(legacy, "safe_save_file", save)
        monkeypatch.setattr(legacy, "weight_quant", record)
        monkeypatch.setattr(legacy, "update_config", lambda *args, **kwargs: None)
        monkeypatch.setattr(legacy, "WA_for_sglang_config", lambda *args: None)
        monkeypatch.setattr(legacy.torch.cuda, "empty_cache", lambda: None)
        modifier = getattr(legacy, f"{backend}Modifier")(scheme=scheme, valid_scheme_name=scheme, **params)
        state = SimpleNamespace(
            model=SimpleNamespace(path=str(tmp_path), model_=[("toy.safetensors", "unused")]),
            save_path=str(tmp_path),
        )
        modifier.on_start(state)
        return calls, saved

    return run


def test_ignore_norm_and_bias_do_not_reach_quantizer(run_legacy, tensor_factory):
    import torch

    weights = {
        "model.proj.weight": tensor_factory((2, 4), dtype=torch.bfloat16),
        "model.proj.bias": tensor_factory((2,), dtype=torch.bfloat16),
        "model.norm.weight": tensor_factory((4,), dtype=torch.bfloat16),
        "model.skip.weight": tensor_factory((2, 4), seed=1, dtype=torch.bfloat16),
    }
    calls, saved = run_legacy(weights, ignore=["model.skip"])
    assert len(calls) == 1
    assert calls[0][1] == "W8A8"
    assert torch.equal(calls[0][0], weights["model.proj.weight"])
    assert set(saved) == set(weights) | {"model.proj.weight_scale"}
    for name in ("model.proj.bias", "model.norm.weight", "model.skip.weight"):
        assert saved[name] is weights[name]


@pytest.mark.parametrize("params, expected_scheme", [
    ({"fp8_blockwise_layers": ["model.proj"], "fp8_channelwise_layers": ["model.proj"],
      "mxfp4_layers": ["model.proj"]}, "FP8_BLOCK"),
    ({"fp8_channelwise_layers": ["model.proj"], "mxfp4_layers": ["model.proj"]}, "FP8_DYNAMIC"),
    ({"mxfp4_layers": ["model.proj"]}, "MXFP4"),
    ({}, "MXFP4"),
    ({"mxfp4_layers": ["model.other"]}, None),
])
def test_legacy_layer_lists_select_scheme(run_legacy, tensor_factory, params, expected_scheme):
    import torch

    weight = tensor_factory((2, 4), dtype=torch.bfloat16)
    calls, saved = run_legacy({"model.proj.weight": weight}, scheme="MXFP4", **params)
    if expected_scheme is None:
        assert calls == []
        assert set(saved) == {"model.proj.weight"}
        assert saved["model.proj.weight"] is weight
    else:
        assert len(calls) == 1
        assert calls[0][1] == expected_scheme
        assert torch.equal(calls[0][0], weight)
        assert set(saved) == {"model.proj.weight", "model.proj.weight_scale"}


@pytest.mark.parametrize("backend, scale_key", [
    ("generalDay0", "model.proj.weight_scale"), ("dpskV4mixGeneralDay0", "model.proj.scale"),
])
def test_legacy_w8a8_callers_preserve_input_and_output_contract(
    run_legacy, tensor_factory, backend, scale_key,
):
    import torch

    weight = tensor_factory((2, 4), dtype=torch.bfloat16, pattern="arange")
    before = weight.clone()
    calls, saved = run_legacy({"model.proj.weight": weight}, backend=backend)
    assert len(calls) == 1
    assert calls[0][1] == "W8A8"
    torch.testing.assert_close(calls[0][0], before, rtol=0, atol=0)
    assert set(saved) == {"model.proj.weight", scale_key}
    torch.testing.assert_close(saved["model.proj.weight"], torch.ones((2, 4), dtype=torch.int8))
    torch.testing.assert_close(saved[scale_key], torch.full((2, 1), 0.5, dtype=torch.float32))
    assert torch.equal(weight, before)


@pytest.mark.parametrize("backend", ["generalDay0", "dpskV4mixGeneralDay0"])
@pytest.mark.parametrize("transpose", [False, True])
def test_legacy_gate_up_expert_slices_and_names(run_legacy, tensor_factory, backend, transpose):
    import torch

    shape = (3, 8) if transpose else (8, 3)
    weight = tensor_factory(shape, experts=2, pattern="arange", dtype=torch.bfloat16)
    before = weight.clone()
    calls, saved = run_legacy({"model.experts.gate_up_proj": weight}, backend=backend)
    expected_inputs, expected_keys = [], []
    for expert in range(2):
        rows = [[expert * 24 + (col * 8 + row if transpose else row * 3 + col)
                 for col in range(3)] for row in range(8)]
        for role, part in (("gate", rows[:4]), ("up", rows[4:])):
            expected_inputs.append(torch.tensor(part, dtype=torch.bfloat16))
            name = f"model.experts.{expert}.{role}_proj.weight"
            expected_keys.extend((name, name + "_scale"))
    assert len(calls) == 4
    assert set(saved) == set(expected_keys)
    for index, ((actual, scheme), expected) in enumerate(zip(calls, expected_inputs), start=1):
        assert scheme == "W8A8"
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(saved[expected_keys[(index - 1) * 2]],
                                   torch.full((4, 3), index, dtype=torch.int8))
        torch.testing.assert_close(saved[expected_keys[(index - 1) * 2 + 1]],
                                   torch.full((4, 1), index / 2, dtype=torch.float32))
    assert torch.equal(weight, before)


@pytest.mark.parametrize("backend", ["generalDay0", "dpskV4mixGeneralDay0"])
@pytest.mark.parametrize("transpose", [False, True])
def test_legacy_full_expert_transpose_and_names(run_legacy, tensor_factory, backend, transpose):
    import torch

    weight = tensor_factory((4, 3), experts=2, pattern="arange", dtype=torch.bfloat16)
    before = weight.clone()
    calls, saved = run_legacy({"model.experts.down_proj": weight}, backend=backend,
                              hidden_size=3 if transpose else 9)
    assert len(calls) == 2
    expected_keys = []
    for expert, (actual, scheme) in enumerate(calls):
        if transpose:
            values = [[expert * 12 + row + col * 3 for col in range(4)] for row in range(3)]
        else:
            values = [[expert * 12 + row * 3 + col for col in range(3)] for row in range(4)]
        torch.testing.assert_close(actual, torch.tensor(values, dtype=torch.bfloat16), rtol=0, atol=0)
        assert scheme == "W8A8"
        name = f"model.experts.{expert}.down_proj.weight"
        expected_keys.extend((name, name + "_scale"))
        assert saved[name].shape == actual.shape
        assert torch.all(saved[name] == expert + 1)
        assert saved[name + "_scale"].shape == (len(values), 1)
    assert set(saved) == set(expected_keys)
    assert torch.equal(weight, before)


def test_dpsk_numbered_experts_override_dense_fp8_list(run_legacy, tensor_factory):
    import torch

    keys = ["model.experts.0.down_proj.weight", "model.dense.weight", "model.keep.weight",
            "model.norm.weight", "model.dense.bias", "counter"]
    weights = {key: tensor_factory((2, 4), seed=index, dtype=torch.bfloat16)
               for index, key in enumerate(keys)}
    calls, saved = run_legacy(weights, backend="dpskV4mixGeneralDay0",
                              fp8_channelwise_layers=["model.dense", "model.experts.0.down_proj"])
    assert [scheme for _, scheme in calls] == ["W8A8", "FP8_DYNAMIC"]
    for (actual, _), key in zip(calls, keys[:2]):
        torch.testing.assert_close(actual, weights[key], rtol=0, atol=0)
    assert set(saved) == set(keys) | {keys[0] + "_scale", keys[1] + "_scale"}
    for key in keys[2:]:
        assert saved[key] is weights[key]


@pytest.mark.parametrize("source_scale, output_scale", [
    ("model.proj.weight_scale_inv", "model.proj.weight_scale"),
    ("model.proj.weight.scale", "model.proj.weight.scale"),
    ("model.proj.scale", "model.proj.scale"),
])
def test_fp8_based_caller_uses_source_map_and_respects_ignore(
    run_legacy, tensor_factory, source_scale, output_scale,
):
    import torch

    keys = ["model.proj.weight", "model.skip.weight", "model.plain.weight", "model.norm.weight",
            "model.proj.bias"]
    weights = {key: tensor_factory((2, 4), seed=index, dtype=torch.bfloat16)
               for index, key in enumerate(keys)}
    source_map = dict.fromkeys([*keys, source_scale, "model.skip.weight_scale_inv"], "toy.safetensors")
    calls, saved = run_legacy(weights, backend="fp8BasedDay0", source_map=source_map,
                              ignore=["model.skip"])
    assert len(calls) == 1
    assert calls[0][1] == "W8A8"
    torch.testing.assert_close(calls[0][0], weights[keys[0]], rtol=0, atol=0)
    assert set(saved) == set(keys) | {output_scale}
    torch.testing.assert_close(saved[output_scale], torch.full((2, 1), 0.5, dtype=torch.float32))
    for key in keys[1:]:
        assert saved[key] is weights[key]
