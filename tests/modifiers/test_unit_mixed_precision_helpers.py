# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pure-function helpers of the mixedPrecision modifier against handwritten expectations."""

import json

import pytest

pytestmark = pytest.mark.unit


def _module():
    from qlean.modifiers.mixedPrecision import base

    return base


@pytest.mark.cpu
@pytest.mark.parametrize("raw, expected", [
    (["3"], [3]),
    (["0", "7"], [0, 7]),
    (["5-7"], [5, 6, 7]),
    (["0", "92-93", "89-90"], [0, 89, 90, 92, 93]),
    (["5-5"], [5]),                       # degenerate single-point range
    (["3", "3", "5-6", "5"], [3, 5, 6]),  # duplicates collapse, result sorted
])
def test_parse_layers_accepts_numbers_ranges_and_combinations(raw, expected):
    assert _module().parse_layers(raw) == expected


@pytest.mark.cpu
@pytest.mark.parametrize("raw", [
    ["5-3"],    # start > end
    ["a"],      # non-numeric single entry
    ["a-b"],    # non-numeric range
])
def test_parse_layers_rejects_invalid_input(raw):
    with pytest.raises(ValueError):
        _module().parse_layers(raw)


@pytest.mark.cpu
def test_get_layer_num_extracts_layer_index(tmp_path):
    module = _module()

    assert module.get_layer_num("model.layers.12.self_attn.q_proj.weight") == 12
    assert module.get_layer_num("model.layers.0.mlp.gate_proj.weight") == 0
    # Keys without a layer segment return the documented sentinel -1.
    assert module.get_layer_num("model.embed_tokens.weight") == -1
    assert module.get_layer_num("lm_head.weight") == -1


@pytest.mark.cpu
def test_files_are_identical_fast_compares_by_size(tmp_path):
    module = _module()

    same = tmp_path / "a.bin"
    other = tmp_path / "b.bin"
    third = tmp_path / "c.bin"

    same.write_bytes(b"12345")
    other.write_bytes(b"67890")   # same size, different bytes
    third.write_bytes(b"123")     # different size

    assert module.files_are_identical_fast(str(same), str(other)) is True   # size-only semantics
    assert module.files_are_identical_fast(str(same), str(third)) is False
    assert module.files_are_identical_fast(str(same), str(tmp_path / "missing")) is False


@pytest.mark.cpu
def test_count_safetensor_files_counts_only_safetensors(tmp_path):
    module = _module()

    for name in ("model-00000.safetensors", "model-00001.safetensors", "config.json", "notes.txt"):
        (tmp_path / name).write_text("x")

    assert module.count_safetensor_files(str(tmp_path)) == 2
    assert module.count_safetensor_files(str(tmp_path / "missing_dir")) == 0


@pytest.mark.cpu
def test_shard_weights_routes_tensors_by_layer(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    module = _module()

    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()

    tensors = {
        "model.layers.0.self_attn.q_proj.weight": torch.ones(4, 4),
        "model.layers.1.mlp.gate_proj.weight": torch.full((2, 2), 2.0),
        "model.embed_tokens.weight": torch.full((3, 3), 3.0),  # no layer -> lands in shard 0
    }
    save_file(tensors, str(input_dir / "model-00001-of-00001.safetensors"))

    module.shard_weights(
        input_path=str(input_dir / "model-{:05d}-of-00001.safetensors"),
        output_dir=str(output_dir),
        prefix="model",
        total_files=2,       # second file missing -> skipped with a warning
        num_layers=2,
    )

    from safetensors.torch import safe_open

    expected_shards = [
        {"model.layers.0.self_attn.q_proj.weight", "model.embed_tokens.weight"},
        {"model.layers.1.mlp.gate_proj.weight"},
    ]
    for i, expected_keys in enumerate(expected_shards):
        out_path = output_dir / f"model-{i:05d}-of-00001.safetensors"
        assert out_path.is_file()
        with safe_open(str(out_path), framework="pt") as f:
            assert set(f.keys()) == expected_keys


@pytest.mark.cpu
def test_copy_other_files_excludes_safetensors_and_index(tmp_path):
    module = _module()

    src = tmp_path / "w8a8"
    dst = tmp_path / "target"
    (src / "nested").mkdir(parents=True)

    keep = {
        "config.json": '{"model_type": "test"}',
        "tokenizer.json": "{}",
        "nested/extra.txt": "x",
    }
    skip = {
        "model-00000-of-00000.safetensors": "",
        "model.safetensors.index.json": "{}",
    }
    for rel, content in {**keep, **skip}.items():
        (src / rel).write_text(content)

    module.copy_other_files(str(src), str(dst))

    for rel in keep:
        assert (dst / rel).is_file(), f"{rel} should be copied"
    for rel in skip:
        assert not (dst / rel).exists(), f"{rel} should be excluded"
    # Calling with a missing source is a logged no-op, not an error.
    module.copy_other_files(str(tmp_path / "missing"), str(dst))


@pytest.mark.cpu
def test_update_config_writes_mix_layer_into_quantization_config(tmp_path):
    module = _module()

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "test"}))
    module.update_config([3, 7], str(tmp_path))

    config = json.loads((tmp_path / "config.json").read_text())
    assert config["quantization_config"]["mix_layer"] == [3, 7]

    # A second call overwrites rather than duplicates the entry.
    module.update_config([9], str(tmp_path))
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["quantization_config"]["mix_layer"] == [9]

    # Missing config file is a logged no-op, not an error.
    empty = tmp_path / "empty"
    empty.mkdir()
    module.update_config([1], str(empty))


@pytest.mark.cpu
def test_find_layer_count_counts_only_requested_layers(tmp_path):
    module = _module()

    index = {
        "weight_map": {
            "model.layers.0.a": "shard-0.safetensors",
            "model.layers.0.b": "shard-0.safetensors",
            "model.layers.1.a": "shard-1.safetensors",
            "model.layers.2.a": "shard-1.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    counts = module.find_layer_count(str(tmp_path), {0, 2})
    assert counts == {"shard-0.safetensors": 2, "shard-1.safetensors": 1}

    # No matching layers -> empty dict.
    assert module.find_layer_count(str(tmp_path), {9}) == {}

    # index without weight_map raises.
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}}))
    with pytest.raises(ValueError, match="weight_map"):
        module.find_layer_count(str(tmp_path), {0})


@pytest.mark.cpu
def test_update_model_index_removes_and_adds_entries(tmp_path):
    module = _module()

    index = {
        "weight_map": {
            "model.layers.0.a": "shard-0.safetensors",
            "model.layers.1.a": "shard-1.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    module.update_model_index(
        str(tmp_path),
        layer_ids={0},
        add_layers={"model.layers.0.a": "replacement.safetensors"},
    )

    data = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    assert data["weight_map"] == {
        "model.layers.0.a": "replacement.safetensors",  # removed then re-added
        "model.layers.1.a": "shard-1.safetensors",      # untouched
    }


@pytest.mark.cpu
def test_update_model_index_requires_existing_file_and_weight_map(tmp_path):
    module = _module()

    with pytest.raises(Exception, match="does not exist"):
        module.update_model_index(str(tmp_path), {0}, {})

    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}}))
    with pytest.raises(ValueError, match="weight_map"):
        module.update_model_index(str(tmp_path), {0}, {})


@pytest.mark.cpu
def test_copy_w8a8_files_copies_only_non_replacement_files(tmp_path):
    module = _module()

    src = tmp_path / "w8a8"
    dst = tmp_path / "target"
    src.mkdir()

    files = {
        "model-00000-of-00000.safetensors": "quantized",
        "model-00001-of-00001.safetensors": "quantized",
        "config.json": "{}",
    }
    for rel, content in files.items():
        (src / rel).write_text(content)

    # The first shard is being replaced (mapping shard -> replacement source);
    # its keys drive which files are excluded from the bulk copy.
    module.copy_w8a8_files(str(dst), str(src), {"model-00000-of-00000.safetensors": "replacement.safetensors"})

    assert (dst / "model-00001-of-00001.safetensors").is_file()
    assert (dst / "config.json").is_file()
    assert not (dst / "model-00000-of-00000.safetensors").exists()

    with pytest.raises(Exception, match="does not exist"):
        module.copy_w8a8_files(str(dst), str(tmp_path / "missing"), {})
