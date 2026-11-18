# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Small checkpoints and real conversion fixtures with deferred production imports."""

from collections import namedtuple
from contextlib import nullcontext
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
import json
import shutil
import sys

import pytest


def _snapshot(path):
    return {p.relative_to(path): p.read_bytes() for p in path.rglob("*") if p.is_file()}


# Every `src` format a recipe region may declare, in the order the engine lists them.
SRC_KINDS = ("bf16", "fp8_block", "nvfp4", "int4", "mxfp4", "mxfp8")
Source = namedtuple("Source", "name raw scales reference")


def _aligned_source(kind, rows, cols):
    """Analytic source payload for a [rows, cols] weight with both sides 128-aligned.

    Block and MX encoders reject ragged shapes: fp8 block scales only cover whole 128x128
    tiles, and the MX kernels tile 32-element groups, so a matrix test needs sources that
    divide evenly. Every value below is a power of two (or an integer), which keeps the
    reference exactly representable after dequantization.
    """
    import torch

    if rows % 128 or cols % 128:
        raise ValueError(f"aligned sources need 128-multiples, got {(rows, cols)}")
    if kind == "bf16":
        reference = torch.tensor([0.5, -6.0, 1.5, 4.0]).repeat(rows, cols // 4)
        return Source(name="x.weight", raw=reference.to(torch.bfloat16),
                      scales={}, reference=reference)
    if kind == "fp8_block":
        codes = torch.tensor([1.0, -2.0, 0.5, -0.25]).repeat(rows, cols // 4)
        scales = {"x.weight_scale_inv": torch.full((rows // 128, cols // 128), 0.5)}
        return Source(name="x.weight", raw=codes.to(torch.float8_e4m3fn),
                      scales=scales, reference=codes * 0.5)
    # MX scales carry one e8m0 byte per 32-element group; 126..129 decode to 0.5, 1, 2, 4.
    shifts = torch.arange(cols // 32) % 4
    factors = torch.tensor([0.5, 1.0, 2.0, 4.0])[shifts]
    if kind == "int4":
        # 0x76543210 stores nibbles 0..7 LSB-first, and int4 here is offset by 8 (-8..-1).
        values = (torch.arange(cols) % 8) - 8
        scale = factors.repeat(rows, 1).to(torch.bfloat16)
        reference = (values.view(-1, 32) * factors[:, None]).reshape(cols).repeat(rows, 1)
        return Source(name="x.weight_packed",
                      raw=torch.full((rows, cols // 8), 0x76543210, dtype=torch.int32),
                      scales={"x.weight_scale": scale}, reference=reference)
    if kind == "nvfp4":
        # 0xF1 packs 0.5 (low nibble) and -6.0 (high nibble); the global scale halves both.
        # nvfp4 groups 16 elements per local scale, so it needs its own factor cycle.
        values = torch.tensor([0.5, -6.0]).repeat(cols // 2)
        local = torch.tensor([0.5, 1.0, 2.0, 4.0])[torch.arange(cols // 16) % 4]
        reference = (values.view(-1, 16) * (local * 0.5)[:, None]).reshape(cols).repeat(rows, 1)
        # nvfp4 resolves its scale as name + "_scale", so the weight keeps the plain name.
        return Source(name="x.weight",
                      raw=torch.full((rows, cols // 2), 0xF1, dtype=torch.uint8),
                      scales={"x.weight_scale": local.repeat(rows, 1),
                              "x.weight_scale_2": torch.tensor(0.5)},
                      reference=reference)
    scales = torch.tensor([126, 127, 128, 129], dtype=torch.uint8)[shifts].repeat(rows, 1)
    if kind == "mxfp8":
        values = torch.tensor([1.0, -4.0]).repeat(cols // 2)
        reference = (values.view(-1, 32) * factors[:, None]).reshape(cols).repeat(rows, 1)
        return Source(name="x.weight", raw=values.repeat(rows, 1).to(torch.float8_e4m3fn),
                      scales={"x.scale": scales}, reference=reference)
    if kind == "mxfp4":
        # A 0xF1 byte holds the fp4 codes 0.5 (low nibble) and -6.0 (high nibble).
        values = torch.tensor([0.5, -6.0]).repeat(cols // 2)
        reference = (values.view(-1, 32) * factors[:, None]).reshape(cols).repeat(rows, 1)
        return Source(name="x.weight_packed",
                      raw=torch.full((rows, cols // 2), 0xF1, dtype=torch.uint8).view(torch.int8),
                      scales={"x.weight_scale": scales.view(torch.int8)}, reference=reference)
    raise ValueError(kind)


def source_tensors(kind, shape=None):
    """Build one encoded source weight, its scale keys, and the fp32 reference it decodes to.

    The payloads are hand-encoded so no production quantizer is exercised to build the
    source side: values are chosen to survive dequantization exactly, which keeps the
    reference usable for bit-exact assertions after a dequant-only conversion.

    Without `shape` the payloads are two rows wide, which is all a byte-exact golden needs.
    Pass `shape=(rows, cols)` to get the 128-aligned analytic payloads instead.
    """
    if shape is not None:
        return _aligned_source(kind, *shape)
    import torch

    if kind == "bf16":
        reference = torch.tensor([[0.5, -6.0] * 17, [1.5, 4.0] * 17], dtype=torch.float32)
        return Source(name="x.weight", raw=reference.to(torch.bfloat16),
                      scales={}, reference=reference)
    name = "x.weight_packed" if kind in ("mxfp4", "int4") else "x.weight"
    if kind == "int4":
        raw = torch.tensor([[0x76543210] * 4 + [0xFEDCBA98 - 2 ** 32] * 4] * 2, dtype=torch.int32)
        scales = {"x.weight_scale": torch.tensor([[0.5, 2], [1, 4]], dtype=torch.bfloat16)}
        row = [v * 0.5 for v in range(-8, 0)] * 4 + [v * 2 for v in range(8)] * 4
        reference = torch.tensor([row, [v * 2 for v in row]], dtype=torch.float32)
    elif kind == "fp8_block":
        row = [1, -2] * 17
        raw = torch.tensor([row] * 2, dtype=torch.float32).to(torch.float8_e4m3fn)
        scales = {"x.weight_scale_inv": torch.tensor([[0.5]], dtype=torch.float32)}
        reference = torch.tensor([[v * 0.5 for v in row]] * 2, dtype=torch.float32)
    elif kind in ("mxfp4", "mxfp8", "nvfp4"):
        row = [0.5, -4 if kind == "mxfp8" else -6] * 17
        raw = (torch.tensor([row] * 2, dtype=torch.float32).to(torch.float8_e4m3fn)
               if kind == "mxfp8" else torch.full((2, 17), 0xF1, dtype=torch.uint8))
        if kind == "nvfp4":
            scales = {"x.weight_scale": torch.tensor([[0.5, 1, 2], [1, 2, 4]], dtype=torch.float32),
                      "x.weight_scale_2": torch.tensor(0.5, dtype=torch.float32)}
            factors = [0.25] * 16 + [0.5] * 16 + [1] * 2
        else:
            scales = {"x.scale": torch.tensor([[126, 128], [127, 129]], dtype=torch.uint8)}
            factors = [0.5] * 32 + [2] * 2
            if kind == "mxfp4":
                raw = raw.view(torch.int8)
                scales["x.weight_scale"] = scales.pop("x.scale").view(torch.int8)
        reference = torch.tensor([[v * f for v, f in zip(row, factors)],
                                  [v * f * 2 for v, f in zip(row, factors)]], dtype=torch.float32)
    else:
        raise ValueError(kind)
    return Source(name=name, raw=raw, scales=scales, reference=reference)


@pytest.fixture
def quantized_source():
    """Expose the shared source builder to tests that assemble their own shards."""
    return source_tensors


@pytest.fixture
def checkpoint_factory(tmp_path):
    """Create two shards; corrupt copies only and verify source bytes stay unchanged.

    `source_format` selects the legacy full-model layouts (BF16 / FP8). `src` instead
    builds the canonical single-weight source for any dequantizable format, carrying the
    encoded payload, scale keys, and reference on `case.source`; `src_shape` switches to the
    128-aligned payloads that block and MX targets require.
    """
    import torch
    from safetensors.torch import save_file

    snapshots = []

    def make(*, source_format="BF16", text_quant=None, expert_markers=False,
             defect=None, custom_shards=None, config_patch=None, src=None, src_shape=None):
        if source_format not in ("BF16", "FP8"):
            raise ValueError(source_format)
        if src is not None:
            if src not in SRC_KINDS:
                raise ValueError(src)
            if source_format != "BF16" or custom_shards is not None:
                raise ValueError("src replaces the legacy layouts; pass neither source_format nor custom_shards")
        elif src_shape is not None:
            raise ValueError("src_shape only applies to src")
        root = tmp_path / f"checkpoint-{len(snapshots)}"
        source = root / "source"
        source.mkdir(parents=True)
        names = [f"model-{i:05d}-of-00002.safetensors" for i in (1, 2)]
        config = {"model_type": "qlean_test", "architectures": ["QLeanTestModel"],
                  "hidden_size": 3, "test_sentinel": "preserve-me",
                  "quantization_config": {"quant_method": "source-stale"}}
        if text_quant is not None:
            config["text_config"] = {"hidden_size": 3, "test_sentinel": "text-preserve"}
            if text_quant:
                config["text_config"]["quantization_config"] = {"quant_method": "source-stale"}
        if expert_markers:
            for section in (config, config.get("text_config", {})):
                section["expert_dtype"] = "fp8"
                if "quantization_config" in section:
                    section["quantization_config"]["expert_dtype"] = "fp8"

        shards = [{
            "model.proj.bias": torch.tensor([1.25, -2.5, 0], dtype=torch.float32),
            "model.norm.weight": torch.tensor([1, 2, 3], dtype=torch.bfloat16),
        }, {
            "lm_head.weight": torch.tensor([[0.5, -1.5, 2.5]], dtype=torch.bfloat16),
            "step": torch.tensor(7, dtype=torch.int64),
        }]
        scale_names = set()
        checkpoint_source = None
        if src is not None:
            checkpoint_source = source_tensors(src, shape=src_shape)
            # Keep the weight and its scales in different shards to catch swapped lookups.
            shards = [
                {checkpoint_source.name: checkpoint_source.raw,
                 "x.bias": torch.tensor([1.25, -2.5], dtype=torch.float32)},
                {**checkpoint_source.scales,
                 "lm_head.weight": torch.tensor([[0.5, -1.5]], dtype=torch.bfloat16)},
            ]
            scale_names = set(checkpoint_source.scales)
        elif source_format == "BF16":
            weight = torch.tensor([
                [-127, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 127],
                [-254, -5, -3, -1, 1, 3, 5, 254],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ], dtype=torch.bfloat16)
            shards[0]["model.proj.weight"] = weight
            shards[1]["model.other.weight"] = -weight * 2
            # Store [expert, hidden, gate+up]; transpose, then split into four-row halves.
            expert_rows = torch.tensor([
                [127, -127, 0], [-254, 0, 254], [0, 0, 0], [0, 508, -508],
                [254, 0, -254], [0, -127, 127], [0, 0, 0], [-508, 508, 0],
            ], dtype=torch.bfloat16)
            shards[1]["model.mlp.experts.gate_up_proj"] = torch.stack(
                (expert_rows.t(), -expert_rows.t() * 2)).contiguous()
        else:
            row = torch.tensor([-8, -4, -2, -1, 0, 1, 2, 4], dtype=torch.float32)
            raw = row.repeat(128, 16)
            shards[0]["model.proj.weight"] = raw.to(torch.float8_e4m3fn)
            shards[1]["model.other.weight"] = (-raw).to(torch.float8_e4m3fn)
            # Use distinct cross-shard scales to catch swapped lookups.
            shards[1]["model.proj.weight_scale_inv"] = torch.tensor([[15.875]], dtype=torch.float32)
            shards[0]["model.other.weight_scale_inv"] = torch.tensor([[31.75]], dtype=torch.float32)
            scale_names = {"model.proj.weight_scale_inv", "model.other.weight_scale_inv"}

        if custom_shards is not None:
            assert len(custom_shards) == 2
            shards = [{key: tensor.clone().contiguous() for key, tensor in shard.items()}
                      for shard in custom_shards]
            keys = [key for shard in shards for key in shard]
            assert len(keys) == len(set(keys))
            scale_names = {key for key in keys if key.endswith((".scale", "_scale", "_scale_inv", "_scale_2"))}
        if config_patch is not None:
            config.update(deepcopy(config_patch))
        weight_map = {key: filename for filename, shard in zip(names, shards) for key in shard}
        (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (source / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8")
        (source / "tokenizer_config.json").write_text(
            '{"model_max_length": 128, "eos_token": "<eos>"}', encoding="utf-8")
        for filename, shard in zip(names, shards):
            save_file(shard, str(source / filename), metadata={"format": "pt"})
        pristine = source
        snapshots.append((pristine, _snapshot(pristine)))

        if defect is not None:
            source = root / "broken-source"
            shutil.copytree(pristine, source)
            if defect == "truncated":
                shard_path = source / names[1]
                shard_path.write_bytes(shard_path.read_bytes()[:-1])
            elif defect == "missing-scale" and source_format == "FP8":
                missing = "model.proj.weight_scale_inv"
                broken = {key: value for key, value in shards[1].items() if key != missing}
                save_file(broken, str(source / names[1]))
                weight_map = {key: value for key, value in weight_map.items() if key != missing}
                (source / "model.safetensors.index.json").write_text(
                    json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8")
            else:
                raise ValueError(f"Unsupported defect: {defect}")
            snapshots.append((source, _snapshot(source)))

        return SimpleNamespace(
            path=source, pristine=pristine, output=root / "output", config=deepcopy(config),
            source_format=source_format, src=src, source=checkpoint_source,
            weight_map=weight_map, shard_names=names,
            tensors={key: tensor.clone() for shard in shards for key, tensor in shard.items()},
            source_scales=scale_names,
        )

    yield make
    for path, before in snapshots:
        assert _snapshot(path) == before, f"Source checkpoint was modified: {path}"


@pytest.fixture(scope="session")
def checkpoint_backends():
    """Import before registry isolation to keep cached modules and registrations aligned."""
    modules = {
        "model": import_module("qlean.models.day0.day0"),
        "new": import_module("qlean.modifiers_engine.generalQuantEngine.base"),
        "legacy": import_module("qlean.modifiers.generalDay0.base"),
        "dequant": import_module("qlean.modifiers.dequant.base"),
        "fp8_legacy": import_module("qlean.modifiers.fp8BasedDay0.base"),
        "compressor": import_module("qlean.core.qleancompressor"),
    }
    return SimpleNamespace(**modules)


@pytest.fixture
def checkpoint_runtime(checkpoint_backends, isolated_state, monkeypatch, request):
    """Stub installation, plugin scans, and device selection; keep I/O and conversion real."""
    import torch
    from qlean.models import ModelFactory
    from qlean.modifiers import ModifierFactory
    from qlean.utils import weight_file

    modules = checkpoint_backends
    classes = [modules.new.GeneralQuantEngineModifier, modules.legacy.generalDay0Modifier,
               modules.dequant.dequantModifier, modules.fp8_legacy.fp8BasedDay0Modifier]
    monkeypatch.setattr(ModelFactory, "_loaded", True)
    monkeypatch.setattr(ModifierFactory, "_loaded", True)
    monkeypatch.setitem(ModelFactory._main_registry, "day0", modules.model.day0Model)
    monkeypatch.delitem(ModelFactory._errors, "day0", raising=False)
    for cls in classes:
        monkeypatch.setitem(ModifierFactory._main_registry, cls.__name__, cls)
        monkeypatch.delitem(ModifierFactory._errors, cls.__name__, raising=False)
    monkeypatch.setattr(modules.compressor, "install_dependencies", lambda name: "day0")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    def reject_remote(*args, **kwargs):
        pytest.fail("Small integration tests must not access remote models")

    monkeypatch.setattr(weight_file, "list_repo_files", reject_remote)
    if request.node.get_closest_marker("cpu"):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
        device_scope = nullcontext()
    else:
        device_scope = torch.cuda.device(0)
    with device_scope:
        yield modules


@pytest.fixture
def build_conversion(checkpoint_runtime, recipe_factory, monkeypatch):
    import yaml
    from qlean.args import ModelArguments
    from qlean.modifiers import ModifierFactory

    def build(case, *, backend="new", scale_naming="underscore", quantize_experts=True,
              custom_recipe=None, auto_output_format=None):
        if auto_output_format is not None:
            assert backend == "new" and custom_recipe is None
            recipe = None
        elif custom_recipe is not None:
            recipe = deepcopy(custom_recipe)
            for module in ("nvfp4Dequant", "nvfp4BasedMxfp4Day0"):
                name = module + "Modifier"
                if name in recipe["quant_stage"]["quant_modifiers"]:
                    cls = getattr(import_module(f"qlean.modifiers.{module}.base"), name)
                    monkeypatch.setitem(ModifierFactory._main_registry, name, cls)
                    monkeypatch.delitem(ModifierFactory._errors, name, raising=False)
        elif backend == "new":
            regions = [{"name": "kept", "match": ["lm_head.weight", "model.norm.weight"]}]
            if case.source_format == "BF16":
                experts = {"name": "experts", "match": "model.mlp.experts.*"}
                if quantize_experts:
                    experts.update(dst="W8A8", expert_mode="unpack", split_gate_up=True,
                                   transpose="auto")
                regions.append(experts)
            regions.extend([
                {"name": "weights", "match": "*.weight", "dst": "W8A8",
                 "src": "fp8_block" if case.source_format == "FP8" else "bf16"},
                {"name": "fallback", "match": "*"},
            ])
            recipe = recipe_factory(regions=regions, num_gpus=1, resume=False,
                                    sglang_wa=False, scale_naming=scale_naming)
        elif backend == "legacy" and case.source_format == "BF16":
            recipe = recipe_factory(modifier="generalDay0Modifier",
                                    ignore=["lm_head", "model.norm"])
        elif backend == "legacy" and case.source_format == "FP8":
            recipe = {"quant_stage": {"quant_modifiers": {
                "dequantModifier": {},
                "fp8BasedDay0Modifier": {"scheme": "W8A8", "ignore": ["lm_head"]},
            }}}
        else:
            raise ValueError(backend)
        args = ModelArguments(model_name="QLeanTestModel", model_path=str(case.path),
                              save_path=str(case.output))
        argv = ["qlean", "--model_name", args.model_name, "--model_path", args.model_path,
                "--save_path", args.save_path]
        if auto_output_format is None:
            recipe_path = case.output.parent / "recipe.yaml"
            recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8")
            args.recipe = str(recipe_path)
            argv.extend(["--recipe", args.recipe])
        else:
            recipe_path = "generated:GeneralQuantEngineModifier"
            args.output_format = auto_output_format
            argv.extend(["--output-format", auto_output_format])
        monkeypatch.setattr(sys, "argv", argv)
        compressor = checkpoint_runtime.compressor.QLeanCompressor(args)
        if auto_output_format is not None:
            assert compressor._quant_meta["recipe_path"] == recipe_path
            recipe = deepcopy(compressor._quant_meta["recipe_content"])
        assert type(compressor.state.model) is checkpoint_runtime.model.day0Model
        assert {Path(path).name for _, path in compressor.state.model.model_} == set(case.shard_names)
        return SimpleNamespace(compressor=compressor, recipe=recipe, recipe_path=recipe_path,
                               args=args, argv=argv)

    return build
