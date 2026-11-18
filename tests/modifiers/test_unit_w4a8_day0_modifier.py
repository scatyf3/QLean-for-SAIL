# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test W4A8 Day0 numerics, expert packing, and synthetic checkpoint conversion."""

import json

import pytest

pytestmark = pytest.mark.unit


def _module():
    from qlean.modifiers.W4a8Day0 import base

    return base


def _build_checkpoint(tmp_path, quant_config="present"):
    """Build BF16 intermediates, expert and passthrough weights, and index-only legacy scale markers."""
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    save_dir = tmp_path / "save"
    save_dir.mkdir()

    expert_codes = torch.tensor([
        [1, 2, 3, 4] * 4 + [-7, -6, -5, -4] * 4,
        [-1, -2, -3, -4] * 4 + [7, 6, 5, 4] * 4,
    ], dtype=torch.float32).repeat(32, 2)
    expert_scales = torch.tensor([[0.5], [2.0]], dtype=torch.float32).repeat(32, 1)

    tensors = {
        # The index's _scale_inv marker routes BF16 intermediates to INT8.
        "model.layers.0.self_attn.q_proj.weight": torch.eye(4, dtype=torch.bfloat16),
        # plain bf16 linear weight without scale -> kept as-is
        "model.layers.0.self_attn.k_proj.weight": torch.full((2, 2), 2.0, dtype=torch.bfloat16),
        # Aligned 64x64 expert weights use different scales on odd and even rows.
        "model.layers.0.mlp.experts.0.w1.weight": expert_codes * expert_scales,
        # Pass norm weights through unchanged.
        "model.norm.weight": torch.ones(2, dtype=torch.bfloat16),
    }
    save_file(tensors, str(model_dir / "model-00001-of-00001.safetensors"))

    config = {"model_type": "test"}
    if quant_config == "present":
        config["quantization_config"] = {
            "fmt": "fp8", "scale_fmt": "e4m3", "weight_block_size": [16, 16],
        }
    (model_dir / "config.json").write_text(json.dumps(config))

    # Add a dangling _scale_inv index entry; no legacy scale tensor exists.
    weight_map = {
        name: "model-00001-of-00001.safetensors" for name in tensors
    }
    weight_map["model.layers.0.self_attn.q_proj.weight_scale_inv"] = "model-00001-of-00001.safetensors"
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    return model_dir, save_dir


def _make_state(model_dir, save_dir):
    from qlean.models.day0.day0 import day0Model
    from qlean.core.state import State
    from qlean.args import ModelArguments

    model_args = ModelArguments(
        model_name="test", model_path=str(model_dir), save_path=str(save_dir)
    )
    model = day0Model(model_args)
    model.load_model(model_args)
    return State(model=model, save_path=str(save_dir))


@pytest.mark.cpu
def test_quantize_bf16_to_int_per_row_symmetric_range():
    torch = pytest.importorskip("torch")
    quantize = _module().quantize_bf16_to_int

    weight = torch.tensor([
        [1.0, -1.0, 0.25, -0.25],
        [0.0, 0.1, -0.1, 0.0],
    ], dtype=torch.bfloat16)

    codes, scales = quantize(weight, num_bits=4)

    expected_codes = torch.tensor([
        [7, -7, 2, -2],
        [0, 7, -7, 0],
    ], dtype=torch.int8)
    expected_scales = torch.tensor([[1.0], [weight[1, 1].item()]], dtype=torch.float32) / 7
    torch.testing.assert_close(codes, expected_codes, rtol=0, atol=0)
    torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)

    dequant = codes.float() * scales
    err = (dequant - weight.float()).abs()
    bound = (expected_scales / 2).expand_as(weight)
    assert bool((err <= bound + torch.finfo(torch.float32).eps).all())


@pytest.mark.gpu
def test_w4a8_day0_quantizes_checkpoint_end_to_end(tmp_path, isolated_state):
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file

    model_dir, save_dir = _build_checkpoint(tmp_path, quant_config="present")
    state = _make_state(model_dir, save_dir)

    modifier = _module().W4a8Day0Modifier()
    modifier.on_initialize(state)
    modifier.on_start(state)
    assert modifier.started_ is True

    out = load_file(str(save_dir / "model-00001-of-00001.safetensors"))

    # Check each INT8 code and per-row scale for the BF16 identity matrix.
    q = out["model.layers.0.self_attn.q_proj.weight"]
    torch.testing.assert_close(q, torch.eye(4, dtype=torch.int8) * 127, rtol=0, atol=0)
    torch.testing.assert_close(
        out["model.layers.0.self_attn.q_proj.weight_scale"],
        torch.full((4, 1), 1.0 / 127, dtype=torch.float32), rtol=1e-6, atol=0,
    )

    # plain bf16 linear (no scale in map) kept as-is
    k = out["model.layers.0.self_attn.k_proj.weight"]
    assert k.dtype == torch.bfloat16 and torch.all(k == 2.0)

    # Fixed A8W4 bytes check column layout, nibble order, and signed codes.
    e = out["model.layers.0.mlp.experts.0.w1.weight"]
    expected_bytes = torch.tensor([
        [0x21, 0x43, 0xA9, 0xCB] * 8,
        [0xEF, 0xCD, 0x67, 0x45] * 8,
    ], dtype=torch.uint8).repeat(32, 1)
    assert e.dtype == torch.int8
    torch.testing.assert_close(e.view(torch.uint8), expected_bytes, rtol=0, atol=0)
    torch.testing.assert_close(
        out["model.layers.0.mlp.experts.0.w1.weight_scale"],
        torch.tensor([[0.5], [2.0]], dtype=torch.float32).repeat(32, 1), rtol=0, atol=0,
    )

    # norm-like module: not linear weight -> passthrough
    assert torch.equal(out["model.norm.weight"], torch.ones(2, dtype=torch.bfloat16))

    # The rebuilt index matches all output tensors and drops the dangling _scale_inv entry.
    index = json.loads((save_dir / "model.safetensors.index.json").read_text())
    wm = index["weight_map"]
    assert wm == {name: "model-00001-of-00001.safetensors" for name in out}
    assert "model.layers.0.self_attn.q_proj.weight_scale_inv" not in wm

    # config: existing quantization_config updated in place; scale_fmt kept
    config = json.loads((save_dir / "config.json").read_text())
    qc = config["quantization_config"]
    assert qc["quant_method"] == "mixed_precision_w4"
    assert qc["activation_scheme"] == "dynamic"
    assert qc["scale_fmt"] == "e4m3"
    assert "fmt" not in qc and "weight_block_size" not in qc


@pytest.mark.gpu
def test_w4a8_day0_fresh_config_and_ignore_list(tmp_path, isolated_state):
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file

    model_dir, save_dir = _build_checkpoint(tmp_path, quant_config="absent")
    state = _make_state(model_dir, save_dir)

    modifier = _module().W4a8Day0Modifier()
    modifier.on_initialize(state)
    # Ignored linear layers keep BF16 weights and appear in the config's ignore list.
    modifier.ignore = ["model.layers.0.self_attn.q_proj"]
    modifier.on_start(state)

    out = load_file(str(save_dir / "model-00001-of-00001.safetensors"))
    name = "model.layers.0.self_attn.q_proj.weight"
    # ignored module stays bf16, untouched
    assert out[name].dtype == torch.bfloat16
    assert torch.equal(out[name], torch.eye(4, dtype=torch.bfloat16))

    # config built from scratch with the ignore module reported
    config = json.loads((save_dir / "config.json").read_text())
    qc = config["quantization_config"]
    assert qc["quant_method"] == "mixed_precision_w4"
    assert "model.layers.0.self_attn.q_proj" in qc["ignore"]
    assert "scale_fmt" not in qc


@pytest.mark.gpu
def test_w4a8_day0_epoch_events_trigger_lifecycle(tmp_path, isolated_state):
    from qlean.core.event import Event, EventType

    model_dir, save_dir = _build_checkpoint(tmp_path)
    state = _make_state(model_dir, save_dir)

    modifier = _module().W4a8Day0Modifier()
    modifier.on_initialize(state)

    modifier.on_event(state, Event(type_=EventType.CALIBRATION_EPOCH_START))
    assert modifier.started_ is True
    assert (save_dir / "model-00001-of-00001.safetensors").is_file()

    modifier.on_event(state, Event(type_=EventType.CALIBRATION_EPOCH_END))
    assert modifier.ended_ is True

    modifier.on_finalize(state)
    assert modifier.finalized_ is True
