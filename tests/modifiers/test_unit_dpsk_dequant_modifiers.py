# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared weight_dequant numerics plus the two dpsk mix dequant modifiers.

Both modules ship an identical triton block-dequant kernel; their modifiers
share the shard-rebuild pipeline and differ only in the FP8 branch output
(dpskV4mixDequant writes bf16, dpskV4mix2Mxfp4Fp8Channel re-quantizes to
FP8_DYNAMIC and records the layer in the config).
"""

import json

import pytest

pytestmark = pytest.mark.unit

_MODULES = {
    "dpskV4mixDequant": "qlean.modifiers.dpskV4mixDequant.base",
    "dpskV4mix2Mxfp4Fp8Channel": "qlean.modifiers.dpskV4mix2Mxfp4Fp8Channel.base",
}


def _module(name):
    import importlib

    return importlib.import_module(_MODULES[name])


def _make_fp8_checkpoint(tmp_path, with_text_config=False):
    """Minimal checkpoint: one FP8 e4m3 weight with .scale sibling, one passthrough."""
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    save_dir = tmp_path / "save"
    save_dir.mkdir()

    # 128x128 so the default block_size=128 grid covers it exactly
    fp8_codes = torch.tensor(24.0, device="cpu").to(torch.float8_e4m3fn).repeat(128, 128)
    scale = torch.full((1, 1), 2.0, dtype=torch.float32)  # per-block scale grid
    passthrough = torch.eye(8, dtype=torch.bfloat16)

    tensors = {
        "model.layers.0.self_attn.q_proj.weight": fp8_codes,
        "model.layers.0.self_attn.q_proj.scale": scale,
        "model.norm.weight": passthrough,
    }
    save_file(tensors, str(model_dir / "model-00001-of-00001.safetensors"))

    config = {
        "model_type": "test",
        "quantization_config": {"quant_method": "mixed"},
    }
    if with_text_config:
        config["text_config"] = {"quantization_config": {"quant_method": "mixed"}}
    (model_dir / "config.json").write_text(json.dumps(config))
    weight_map = {name: "model-00001-of-00001.safetensors" for name in tensors}
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


def _run_modifier(name, state):
    modifier = _module(name).__dict__[f"{name}Modifier"]()
    modifier.on_initialize(state)
    return modifier


@pytest.mark.gpu
@pytest.mark.parametrize("module_name", list(_MODULES))
def test_weight_dequant_applies_block_scale(module_name):
    """The shared kernel dequantizes an (M, N) grid of blocks: y = x * s per block."""
    torch = pytest.importorskip("torch")
    dequant = _module(module_name).weight_dequant

    # 2x2 block grid with a distinct scale per block; x is a 2D block matrix
    x = torch.full((256, 256), 16.0, dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn)
    # block layout: s[m, n] indexes the (m, n) block, one value per block
    s = torch.tensor([[1.0, 2.0], [4.0, 8.0]], dtype=torch.float32).cuda()

    y = dequant(x.float(), s.float())

    expected = x.float() * s.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    assert y.dtype == torch.get_default_dtype()
    torch.testing.assert_close(y, expected.to(y.dtype), rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("module_name", list(_MODULES))
def test_weight_dequant_handles_3d_batches(module_name):
    torch = pytest.importorskip("torch")
    dequant = _module(module_name).weight_dequant

    x = torch.stack([torch.full((2, 2), 16.0), torch.full((2, 2), 32.0)]).cuda()
    s = torch.stack([torch.ones(2, 2), torch.full((2, 2), 2.0)]).cuda()

    y = dequant(x, s, block_size=1)

    assert y.shape == x.shape
    assert torch.equal(y[0], x[0] * 1.0)
    assert torch.equal(y[1], x[1] * 2.0)


@pytest.mark.gpu
def test_weight_dequant_rejects_unsupported_rank():
    torch = pytest.importorskip("torch")
    dequant = _module("dpskV4mixDequant").weight_dequant

    x = torch.zeros(1, 2, 2, 2).cuda()
    s = torch.zeros(1, 2, 2, 2).cuda()
    with pytest.raises(ValueError, match="Unsupported tensor dimension"):
        dequant(x, s)


@pytest.mark.gpu
def test_dpsk_dequant_converts_fp8_to_bf16(tmp_path, isolated_state):
    """dpskV4mixDequant: FP8 weight dequantized to bf16, quantization_config stripped."""
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file

    model_dir, save_dir = _make_fp8_checkpoint(tmp_path, with_text_config=True)
    state = _make_state(model_dir, save_dir)

    modifier = _run_modifier("dpskV4mixDequant", state)
    assert modifier.initialized_ is True

    # Shards were renamed to the model-XXXXX-of-YYYYY convention.
    import glob as _glob
    shards = _glob.glob(str(save_dir / "model-*-of-*.safetensors"))
    assert len(shards) == 1
    out = load_file(shards[0])

    # FP8 codes 24 * scale 2.0 -> bf16 value 48; the .scale entry is gone
    w = out["model.layers.0.self_attn.q_proj.weight"]
    assert w.dtype == torch.bfloat16
    assert w.shape == (128, 128)
    assert torch.all(w == 48.0)
    assert "model.layers.0.self_attn.q_proj.scale" not in out

    # passthrough tensor survives with its original dtype
    assert torch.equal(out["model.norm.weight"], torch.eye(8, dtype=torch.bfloat16))

    # quantization_config removed at the top level only; the nested copy under
    # text_config survives (observed behavior, unlike int4Dequant which strips
    # both) - fix on_initialize if the nested config should also be removed.
    config = json.loads((save_dir / "config.json").read_text())
    assert "quantization_config" not in config
    assert "quantization_config" in config["text_config"]
    index = json.loads((save_dir / "model.safetensors.index.json").read_text())
    wm = index["weight_map"]
    assert "model.layers.0.self_attn.q_proj.weight" in wm
    assert "model.layers.0.self_attn.q_proj.scale" not in wm

    # state.model.model_ repointed to the renamed shard list
    names = [name for name, _ in state.model.model_]
    assert names[0].startswith("model-") and "of" in names[0]


@pytest.mark.gpu
def test_channel_dequant_requantizes_fp8_and_updates_config(tmp_path, isolated_state):
    """dpskV4mix2Mxfp4Fp8Channel: FP8 weight re-quantized to FP8_DYNAMIC, layer listed."""
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file

    model_dir, save_dir = _make_fp8_checkpoint(tmp_path)
    state = _make_state(model_dir, save_dir)

    modifier = _run_modifier("dpskV4mix2Mxfp4Fp8Channel", state)
    assert modifier.initialized_ is True

    import glob as _glob
    shards = _glob.glob(str(save_dir / "model-*-of-*.safetensors"))
    assert len(shards) == 1
    out = load_file(shards[0])

    # the FP8 branch: weight stays float8_e4m3fn (re-quantized) and a new
    # .scale entry is written alongside it
    w = out["model.layers.0.self_attn.q_proj.weight"]
    assert w.dtype == torch.float8_e4m3fn
    assert w.shape == (128, 128)
    assert "model.layers.0.self_attn.q_proj.scale" in out

    # passthrough tensor survives untouched
    assert torch.equal(out["model.norm.weight"], torch.eye(8, dtype=torch.bfloat16))

    # config gains the mxfp4 quantization_config with the layer listed
    config = json.loads((save_dir / "config.json").read_text())
    qc = config["quantization_config"]
    assert qc["quant_method"] == "mxfp4"
    assert "model.layers.0.self_attn.q_proj" in qc["fp8_channelwise_layers"]

    # lifecycle hooks work after the heavy on_initialize
    modifier.on_start(state)
    assert modifier.started_ is True
    modifier.on_end(state)
    assert modifier.ended_ is True
