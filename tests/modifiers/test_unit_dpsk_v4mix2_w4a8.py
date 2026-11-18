# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""W4A8 mixed-quantization helpers and the dpskV4mix2W4a8 modifier happy path."""

import json

import pytest

pytestmark = pytest.mark.unit


def _module():
    from qlean.modifiers.dpskV4mix2W4a8 import base

    return base


def _build_checkpoint(tmp_path):
    """Build BF16 intermediates, legacy scale markers, expert weights, and a single-shard index."""
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    save_dir = tmp_path / "save"
    save_dir.mkdir()

    tensors = {
        # BF16 intermediates with legacy scale markers convert to INT8.
        "model.layers.0.self_attn.q_proj.weight": torch.eye(4, dtype=torch.bfloat16),
        "model.layers.0.self_attn.q_proj.weight_scale_inv": torch.full((4, 1), 0.5, dtype=torch.float32),
        # plain bf16 linear weight without scale -> kept as-is
        "model.layers.0.self_attn.k_proj.weight": torch.full((2, 2), 2.0, dtype=torch.bfloat16),
        # expert weight -> packed int4; packer needs the byte grid (k/2, n/2) to be 32-byte aligned,
        # i.e. both dims multiples of 64
        "model.layers.0.mlp.ffn.experts.0.w1.weight": torch.arange(4096, dtype=torch.float32).view(64, 64),
        # top-level key without a dot -> kept as-is
        "step": torch.tensor(0),
    }
    save_file(tensors, str(model_dir / "model-00001-of-00001.safetensors"))

    (model_dir / "config.json").write_text(json.dumps({
        "model_type": "test",
        "quantization_config": {"fmt": "fp8", "scale_fmt": "e4m3", "weight_block_size": [16, 16]},
    }))
    weight_map = {
        name: "model-00001-of-00001.safetensors" for name in tensors
    }
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    return model_dir, save_dir


def _build_dot_scale_checkpoint(tmp_path):
    """Checkpoint whose linear scale uses the dot-form ``<module>.scale`` name."""
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    save_dir = tmp_path / "save"
    save_dir.mkdir()

    tensors = {
        "model.layers.0.self_attn.q_proj.weight": torch.eye(4, dtype=torch.bfloat16),
        "model.layers.0.self_attn.q_proj.scale": torch.full((4, 1), 0.5, dtype=torch.float32),
        "model.embed_tokens.weight": torch.eye(4, dtype=torch.bfloat16),
        "model.layers.0.self_attn.q_proj.scale_inv_unused": torch.full((4, 1), 0.5, dtype=torch.float32),
    }
    save_file(tensors, str(model_dir / "model-00001-of-00001.safetensors"))

    # config without quantization_config exercises the fresh-config branch;
    # embedding in the ignore list exercises the embed_tokens pop-out logic.
    (model_dir / "config.json").write_text(json.dumps({"model_type": "test"}))
    weight_map = {
        name: "model-00001-of-00001.safetensors" for name in tensors
    }
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
@pytest.mark.parametrize("weight_name, weight_map, expected", [
    ("a.b.weight", {"a.b.weight_scale_inv": 1}, "a.b.weight_scale_inv"),
    ("a.b.weight", {"a.b.weight.scale": 1}, "a.b.weight.scale"),        # second suffix wins
    ("a.b.weight", {"a.b.scale": 1}, "a.b.scale"),                       # dot-form fallback
    ("a.b.weight", {"a.b.weight_scale_inv": 1, "a.b.weight.scale": 1}, "a.b.weight_scale_inv"),  # suffix order
    ("a.b.weight", {"other.weight_scale_inv": 1}, None),                 # no candidate present
    ("a.b.bias", {"a.b.bias": 1}, None),                                 # non-weight suffixes not tried
])
def test_find_scale_name_matches_suffixes_in_order(weight_name, weight_map, expected):
    assert _module()._find_scale_name(weight_name, weight_map) == expected


@pytest.mark.cpu
def test_quantize_bf16_to_int_rounds_within_bounds():
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
def test_modifier_quantizes_checkpoint_end_to_end(tmp_path, isolated_state):
    # isolated_state: on_start calls torch.set_default_dtype(bfloat16);
    # the fixture restores the global default dtype afterwards so later
    # tests constructing expected tensors with the default dtype stay consistent.
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file

    model_dir, save_dir = _build_checkpoint(tmp_path)
    state = _make_state(model_dir, save_dir)

    modifier = _module().dpskV4mix2W4a8Day0Modifier()
    modifier.on_initialize(state)
    modifier.on_start(state)
    assert modifier.started_ is True

    out = load_file(str(save_dir / "model-00001-of-00001.safetensors"))

    # Convert BF16 intermediates to INT8 and write a new .weight_scale.
    # NOTE: the legacy _scale_inv entry is NOT removed from the output (observed
    # behavior); if this is unintended, fix on_start before changing the test.
    q = out["model.layers.0.self_attn.q_proj.weight"]
    assert q.dtype == torch.int8
    assert "model.layers.0.self_attn.q_proj.weight_scale" in out

    # plain bf16 linear weight kept as-is
    k = out["model.layers.0.self_attn.k_proj.weight"]
    assert k.dtype == torch.bfloat16 and torch.all(k == 2.0)

    # expert weight packed to int4: base.py keeps (rows, cols/2) of the original layout
    e = out["model.layers.0.mlp.ffn.experts.0.w1.weight"]
    assert e.shape == (64, 32)  # 64 int4 codes packed into 32 byte columns
    assert "model.layers.0.mlp.ffn.experts.0.w1.scale" in out

    # top-level key untouched
    assert int(out["step"]) == 0

    # index rewritten and config quantization_config updated
    index = json.loads((save_dir / "model.safetensors.index.json").read_text())
    assert index["weight_map"]["model.layers.0.mlp.ffn.experts.0.w1.scale"] == "model-00001-of-00001.safetensors"
    config = json.loads((save_dir / "config.json").read_text())
    qc = config["quantization_config"]
    assert qc["quant_method"] == "mixed_precision_w4"
    assert qc["activation_scheme"] == "dynamic"
    assert "fmt" not in qc and "scale_fmt" not in qc and "weight_block_size" not in qc


@pytest.mark.gpu
def test_modifier_ignore_list_keeps_module_weights(tmp_path, isolated_state):
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file

    model_dir, save_dir = _build_checkpoint(tmp_path)
    original = load_file(str(model_dir / "model-00001-of-00001.safetensors"))
    state = _make_state(model_dir, save_dir)

    modifier = _module().dpskV4mix2W4a8Day0Modifier()
    modifier.on_initialize(state)
    modifier.ignore = ["model.layers.0.self_attn.q_proj"]
    modifier.on_start(state)

    out = load_file(str(save_dir / "model-00001-of-00001.safetensors"))
    name = "model.layers.0.self_attn.q_proj.weight"
    assert out[name].dtype == torch.bfloat16
    assert torch.equal(out[name].float(), original[name].float())
    # the ignored module is reported in the config's ignore list
    config = json.loads((save_dir / "config.json").read_text())
    assert "model.layers.0.self_attn.q_proj" in config["quantization_config"]["ignore"]


@pytest.mark.gpu
def test_modifier_dot_scale_name_is_kept_and_fresh_config_written(tmp_path, isolated_state):
    """Dot-form ``<module>.scale`` scales quantize to int8 and keep their name.

    The source config has no quantization_config, so the modifier must create
    one from scratch; model.embed_tokens in the ignore list is popped out of
    the final ignore entry (it is never reported there).
    """
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file

    model_dir, save_dir = _build_dot_scale_checkpoint(tmp_path)
    state = _make_state(model_dir, save_dir)

    modifier = _module().dpskV4mix2W4a8Day0Modifier()
    modifier.on_initialize(state)
    modifier.ignore = ["model.embed_tokens"]
    modifier.on_start(state)

    out = load_file(str(save_dir / "model-00001-of-00001.safetensors"))

    # dot-form scale: int8 codes written, scale name kept unchanged (no rename)
    q = out["model.layers.0.self_attn.q_proj.weight"]
    assert q.dtype == torch.int8
    assert "model.layers.0.self_attn.q_proj.scale" in out

    # embed_tokens is ignored (bf16 kept) but popped out of the ignore list
    emb = out["model.embed_tokens.weight"]
    assert emb.dtype == torch.bfloat16
    config = json.loads((save_dir / "config.json").read_text())
    assert config["quantization_config"]["quant_method"] == "mixed_precision_w4"
    assert "model.embed_tokens" not in config["quantization_config"]["ignore"]


@pytest.mark.gpu
def test_modifier_lifecycle_events_delegate_to_hooks(tmp_path, isolated_state):
    """on_end/on_finalize set their flags and epoch events trigger start/end lazily."""
    from qlean.core.event import Event, EventType

    model_dir, save_dir = _build_checkpoint(tmp_path)
    state = _make_state(model_dir, save_dir)

    modifier = _module().dpskV4mix2W4a8Day0Modifier()
    modifier.on_initialize(state)
    assert modifier.initialized_ is True

    # epoch start event runs the full on_start pipeline lazily
    modifier.on_event(state, Event(type_=EventType.CALIBRATION_EPOCH_START))
    assert modifier.started_ is True
    assert (save_dir / "model-00001-of-00001.safetensors").is_file()

    modifier.on_event(state, Event(type_=EventType.CALIBRATION_EPOCH_END))
    assert modifier.ended_ is True

    modifier.on_finalize(state)
    assert modifier.finalized_ is True
