# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""INT4 dequantization numerics and the int4DequantModifier checkpoint conversion."""

import json

import pytest

pytestmark = pytest.mark.unit


def _module():
    from qlean.modifiers.int4Dequant import base

    return base


def _pack_int4(codes):
    """Pack eight offset-binary INT4 values into one int32, low nibble first."""
    torch = pytest.importorskip("torch")
    assert codes.dtype == torch.int8
    assert codes.ndim == 2 and codes.shape[1] % 8 == 0
    nibbles = (codes.to(torch.int64) + 8).reshape(codes.shape[0], -1, 8)
    shifts = torch.arange(0, 32, 4, dtype=torch.int64, device=codes.device)
    return (nibbles << shifts).sum(dim=-1).to(torch.int32)


@pytest.mark.gpu
def test_weight_dequant_unpacks_codes_and_applies_group_scales():
    """Check int32 unpacking with separate scales for two 32-element groups per row."""
    torch = pytest.importorskip("torch")
    dequant = _module().weight_dequant

    codes = torch.tensor([
        [7, -8, 1, -1, 2, -2, 3, -3],
        [0, 1, 2, 3, 4, 5, 6, 7],
    ], dtype=torch.int8).repeat(1, 8)
    x = _pack_int4(codes)
    s = torch.tensor([[0.5, 2.0], [4.0, 0.25]], dtype=torch.float32)

    y = dequant(x.cuda(), s.cuda(), group_size=32)

    expected = codes.float() * s.repeat_interleave(32, dim=1)
    assert y.dtype == torch.get_default_dtype()
    torch.testing.assert_close(y.cpu(), expected.to(y.dtype), rtol=0, atol=0)


@pytest.mark.gpu
def test_weight_dequant_decodes_all_nibbles_in_int32_words():
    """Use fixed int32 words to check packing order and all sixteen INT4 codes."""
    torch = pytest.importorskip("torch")
    dequant = _module().weight_dequant

    packed = torch.tensor([[0x76543210, 0xFEDCBA98 - 2 ** 32] * 2], dtype=torch.int32)
    codes = torch.tensor([list(range(-8, 8)) * 2], dtype=torch.int8)
    torch.testing.assert_close(_pack_int4(codes), packed, rtol=0, atol=0)
    s = torch.tensor([[1.0]], dtype=torch.float32).cuda()

    y = dequant(packed.cuda(), s, group_size=32)

    assert y.dtype == torch.get_default_dtype()
    torch.testing.assert_close(y.cpu(), codes.to(y.dtype), rtol=0, atol=0)


@pytest.mark.gpu
def test_weight_dequant_rejects_violated_shape_contract():
    torch = pytest.importorskip("torch")
    dequant = _module().weight_dequant

    x = _pack_int4(torch.zeros(2, 32, dtype=torch.int8))
    non_contiguous = x.t().contiguous().t()
    assert not non_contiguous.is_contiguous()
    wrong_rows = torch.ones(3, 1, dtype=torch.float32)   # row mismatch
    wrong_cols = torch.ones(2, 3, dtype=torch.float32)   # violates x*8 == s*32
    with pytest.raises(AssertionError):
        dequant(x, wrong_rows)
    with pytest.raises(AssertionError):
        dequant(x, wrong_cols)
    with pytest.raises(AssertionError):
        dequant(non_contiguous, torch.ones(2, 1, dtype=torch.float32))


def _build_checkpoint(tmp_path):
    """Minimal int4 checkpoint: one packed weight + scale + shape, one passthrough."""
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    save_dir = tmp_path / "save"
    save_dir.mkdir()

    codes = torch.tensor([
        [7, -8, 1, -1, 4, 4, 4, 4],
        [2, 3, -2, -3, 0, 0, 0, 0],
    ], dtype=torch.int8).repeat(1, 4)
    packed = _pack_int4(codes)
    scale = torch.tensor([[0.5], [2.0]], dtype=torch.float32)

    tensors = {
        "model.layers.0.self_attn.q_proj.weight_packed": packed,
        "model.layers.0.self_attn.q_proj.weight_scale": scale,
        "model.layers.0.self_attn.q_proj.weight_shape": torch.tensor(list(codes.shape), dtype=torch.int64),
        "model.norm.weight": torch.ones(2, dtype=torch.bfloat16),
    }
    save_file(tensors, str(model_dir / "model-00001-of-00001.safetensors"))

    (model_dir / "config.json").write_text(json.dumps({
        "model_type": "test",
        "quantization_config": {"quant_method": "int4"},
        "text_config": {"quantization_config": {"quant_method": "int4"}},
    }))
    weight_map = {name: "model-00001-of-00001.safetensors" for name in tensors}
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    return model_dir, save_dir, codes, scale


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


@pytest.mark.gpu
def test_modifier_dequantizes_packed_weights_end_to_end(tmp_path, isolated_state):
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file

    model_dir, save_dir, codes, scale = _build_checkpoint(tmp_path)
    state = _make_state(model_dir, save_dir)

    modifier = _module().int4DequantModifier()
    modifier.on_initialize(state)
    assert modifier.initialized_ is True

    out = load_file(str(save_dir / "model-00001-of-00001.safetensors"))

    dequantized = out["model.layers.0.self_attn.q_proj.weight"]
    expected = (codes.float() * scale).to(torch.bfloat16)
    torch.testing.assert_close(dequantized, expected, rtol=0, atol=0)

    # scale and shape tensors dropped from the output shard
    assert "model.layers.0.self_attn.q_proj.weight_packed" not in out
    assert "model.layers.0.self_attn.q_proj.weight_scale" not in out
    assert "model.layers.0.self_attn.q_proj.weight_shape" not in out

    # passthrough tensor untouched
    assert torch.equal(out["model.norm.weight"], torch.ones(2, dtype=torch.bfloat16))

    # index rewritten: packed/scale/shape entries replaced by the plain weight name
    index = json.loads((save_dir / "model.safetensors.index.json").read_text())
    wm = index["weight_map"]
    assert wm["model.layers.0.self_attn.q_proj.weight"] == "model-00001-of-00001.safetensors"
    assert "model.layers.0.self_attn.q_proj.weight_packed" not in wm
    assert "model.layers.0.self_attn.q_proj.weight_scale" not in wm
    assert "model.layers.0.self_attn.q_proj.weight_shape" not in wm

    # quantization_config removed from config.json (also nested under text_config)
    config = json.loads((save_dir / "config.json").read_text())
    assert "quantization_config" not in config
    assert "quantization_config" not in config["text_config"]

    # state.model.model_ repointed to the output shard list
    assert [name for name, _ in state.model.model_] == ["model-00001-of-00001.safetensors"]

    # lifecycle hooks still work after the heavy on_initialize
    modifier.on_start(state)
    assert modifier.started_ is True
    modifier.on_end(state)
    assert modifier.ended_ is True


@pytest.mark.gpu
def test_modifier_missing_scale_falls_back_to_passthrough(tmp_path, isolated_state):
    """A packed weight without a scale tensor is kept packed with a warning."""
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file

    model_dir, save_dir, _, _ = _build_checkpoint(tmp_path)
    # remove the scale tensor from the shard and the index
    path = model_dir / "model-00001-of-00001.safetensors"
    tensors = load_file(str(path))
    tensors.pop("model.layers.0.self_attn.q_proj.weight_scale")
    from safetensors.torch import save_file
    save_file(tensors, str(path))
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    index["weight_map"].pop("model.layers.0.self_attn.q_proj.weight_scale")
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(index))

    state = _make_state(model_dir, save_dir)
    modifier = _module().int4DequantModifier()
    modifier.on_initialize(state)

    out = load_file(str(save_dir / "model-00001-of-00001.safetensors"))
    # no scale -> the packed tensor is passed through unchanged
    assert "model.layers.0.self_attn.q_proj.weight_packed" in out
    torch.testing.assert_close(
        out["model.layers.0.self_attn.q_proj.weight_packed"],
        tensors["model.layers.0.self_attn.q_proj.weight_packed"],
        rtol=0, atol=0,
    )
