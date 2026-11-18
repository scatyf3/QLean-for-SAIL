# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Small-checkpoint conversion, passthrough, and config output for extended formats."""

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
import csv
import json
import logging

import pytest

pytestmark = pytest.mark.integration


def _scale_key(name, naming):
    return name + "_scale" if naming == "underscore" else name.removesuffix(".weight") + ".scale"


def _copies():
    import torch

    return {"x.bias": torch.tensor([1.25, -2.5], dtype=torch.float32),
            "lm_head.weight": torch.tensor([[0.5, -1.5]], dtype=torch.bfloat16)}


def _engine_recipe(recipe_factory, regions, **params):
    options = dict(num_gpus=1, resume=False, sglang_wa=False)
    options.update(params)
    return recipe_factory(regions=[*regions, {"name": "fallback", "match": "*"}], **options)


@pytest.fixture
def resume_conversion(build_conversion):
    import torch
    from qlean.core.event_lifecycle import EventLifecycle
    from qlean.utils.singleton import SingletonMixin
    from qlean.modifiers_engine.generalQuantEngine import base
    from qlean.modifiers_engine._runtime import engine, verify
    from types import SimpleNamespace

    @contextmanager
    def build(case, *, custom_recipe):
        trace = SimpleNamespace(processed=[], stats=None, summaries=[], conversion=None)
        process, run, summary = engine._process_one_shard, base.run_engine, base.write_verify_summary
        default_dtype = torch.get_default_dtype()

        def capture_process(ctx, spec, file_name, resolved_path):
            trace.processed.append(file_name)
            assert resolved_path == str(case.path / file_name)
            return process(ctx, spec, file_name, resolved_path)

        def capture_run(*args, **kwargs):
            assert trace.stats is None
            stats = run(*args, **kwargs)
            trace.stats = deepcopy(stats)
            return stats

        def capture_summary(directory, source_files, stats):
            trace.summaries.append(deepcopy(stats))
            return summary(directory, source_files, stats)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(SingletonMixin, "_instances", {})
            patch.setattr(EventLifecycle, "initialized", False)
            patch.setattr(EventLifecycle, "finalized", False)
            patch.setattr(engine, "_process_one_shard", capture_process)
            patch.setattr(base, "run_engine", capture_run)
            patch.setattr(base, "write_verify_summary", capture_summary)
            for name in ("_cur_file", "_cur_handle", "_cur_writer"):
                patch.setattr(verify, name, None)
            patch.setattr(verify, "_logged", set())
            try:
                trace.conversion = build_conversion(case, custom_recipe=custom_recipe)
                yield trace
            finally:
                verify.close_writer()
                torch.set_default_dtype(default_dtype)

    return build


def _sidecar(case, filename):
    return case.output / "qlean_verify" / (filename.removesuffix(".safetensors") + ".csv")


def _sidecar_bytes(case):
    return {fn: _sidecar(case, fn).read_bytes() for fn in case.shard_names}


def _stale_summary(case):
    for filename in ("verify_run_info.json", "verify_summary.csv"):
        (case.output / "qlean_verify" / filename).write_text("stale-summary", encoding="utf-8")


def _assert_resume_state(case, run, *, processed, expected_stats, statuses=None):
    from safetensors import safe_open

    assert sorted(run.processed) == sorted(processed)
    reused = sorted(set(case.shard_names) - set(processed))
    assert run.stats.get("resumed_shards", 0) == len(reused)
    assert sorted(run.stats.get("resumed_files", [])) == reused
    assert {key: value for key, value in run.stats.items()
            if key not in ("resumed_shards", "resumed_files")} == expected_stats
    persisted = {}
    for filename in case.shard_names:
        with safe_open(str(case.output / filename), framework="pt") as handle:
            shard_stats = json.loads(handle.metadata()["qlean_shard_stats"])
        for key, value in shard_stats.items():
            if isinstance(value, list):
                persisted.setdefault(key, []).extend(value)
            else:
                persisted[key] = persisted.get(key, 0) + value
    assert persisted == expected_stats
    if statuses is None:
        assert run.summaries == []
        assert not (case.output / "qlean_verify").exists()
        return

    assert run.summaries == [run.stats]
    errors = expected_stats.get("verify_errors", 0)
    report = json.loads((case.output / "qlean_verify" / "verify_run_info.json").read_text())
    assert report == {
        "coverage": {"covered_shards": 2, "total_shards": 2, "missing": []},
        "data_rows": sum(statuses.values()), "error_rows": errors,
        "violation_shards": 0, "error_shards": 1 if errors else 0,
        "cross_check": {"stats_verify_rows": expected_stats["verify_rows"],
                        "stats_verify_errors": errors, "rows_match": True, "errors_match": True},
        "by_scheme_status": statuses,
    }
    with (case.output / "qlean_verify" / "verify_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    schemes = {key.split("|")[0] for key in statuses}
    assert len(rows) == len(schemes) and {row["scheme"] for row in rows} == schemes
    for row in rows:
        scheme = row["scheme"]
        ok, error = statuses.get(f"{scheme}|ok", 0), statuses.get(f"{scheme}|error_RuntimeError", 0)
        assert {key: int(row[key]) for key in (
            "tensors", "ok", "violation", "skipped", "error", "total_violations", "total_boundary"
        )} == {"tensors": ok + error, "ok": ok, "violation": 0, "skipped": 0,
               "error": error, "total_violations": 0, "total_boundary": 0}
        assert 0 <= float(row["worst_max_over_ratio"]) <= 1
        assert case.weight_map[row["worst_tensor"]] == row["worst_shard"]


def _run_and_check(case, conversion, expected, source_of, *, backend="new",
                   formats=("user_specified", "user_specified"), output_shard=None):
    import torch
    from safetensors import safe_open
    from qlean.args import OtherArguments

    assert output_shard is None or backend == "legacy"
    expected_map = {name: output_shard or case.weight_map[source_of[name]] for name in expected}
    conversion.compressor.post_train(OtherArguments())
    actual, observed_map = {}, {}
    paths = sorted(case.output.glob("*.safetensors"))
    assert {path.name for path in paths} == ({output_shard} if output_shard else set(case.shard_names))
    for path in paths:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if backend == "new":
                assert handle.metadata()["format"] == "pt"
                assert isinstance(json.loads(handle.metadata()["qlean_shard_stats"]), dict)
            for name in handle.keys():
                assert name not in actual, f"Duplicate key across shards: {name}"
                actual[name] = handle.get_tensor(name)
                observed_map[name] = path.name
    index = json.loads((case.output / "model.safetensors.index.json").read_text())
    assert index["weight_map"] == observed_map
    assert observed_map == expected_map
    assert set(actual) == set(expected)
    assert not ((case.source_scales - set(expected)) & set(actual))
    for name, tensor in expected.items():
        assert actual[name].dtype == tensor.dtype, name
        assert actual[name].shape == tensor.shape, name
        # Compare bytes after dtype/shape checks to avoid unsupported CPU FP8 operations.
        torch.testing.assert_close(actual[name].contiguous().reshape(-1).view(torch.uint8),
                                   tensor.contiguous().reshape(-1).view(torch.uint8),
                                   rtol=0, atol=0, msg=name)
    if backend == "new":
        from qlean.modifiers_engine._runtime.plan import make_hooks
        from qlean.modifiers_engine._runtime.region_router import spec_plan_fn

        modifier, = conversion.compressor.modifiers
        cfg = {"_regions": modifier.regions, "hidden_size": 3,
               "scale_naming": modifier.scale_naming}
        _, predict = make_hooks(spec_plan_fn)
        for filename in case.shard_names:
            source_meta = {name: {"shape": list(tensor.shape)} for name, tensor in case.tensors.items()
                           if case.weight_map[name] == filename and name not in case.source_scales}
            assert set(predict(source_meta, cfg, weight_map=case.weight_map)) == {
                name for name, shard in observed_map.items() if shard == filename}
    assert not list(case.output.rglob("*.qlean_tmp"))
    assert (case.output / "tokenizer_config.json").read_bytes() == (
        case.path / "tokenizer_config.json").read_bytes()
    assert all(m.initialized_ and m.started_ and m.ended_ for m in conversion.compressor.modifiers)
    metadata = json.loads((case.output / "qlean_config.json").read_text())
    assert metadata["recipe_content"] == conversion.recipe
    assert metadata["recipe_path"] == str(conversion.recipe_path)
    assert metadata["modifier_chain"] == [type(m).__name__ for m in conversion.compressor.modifiers]
    assert (metadata["input_format"], metadata["output_format"]) == formats
    assert metadata["cli_args"] == conversion.argv[1:]
    datetime.fromisoformat(metadata["timestamp"])
    config = json.loads((case.output / "config.json").read_text())
    for key in ("model_type", "architectures", "hidden_size", "test_sentinel"):
        assert config[key] == case.config[key]
    return config


def _assert_compressed_config(config, scheme):
    qconfig = config["quantization_config"]
    assert qconfig["quant_method"] == "compressed-tensors"
    assert qconfig["quantization_status"] == "compressed"
    assert qconfig["format"] == {
        "W8A8": "int-quantized", "FP8_DYNAMIC": "float-quantized",
        "FP8_BLOCK": "float-quantized", "MXFP4": "mxfp4-pack-quantized",
    }[scheme]
    assert set(qconfig["config_groups"]) == {scheme}
    group = qconfig["config_groups"][scheme]
    assert group["targets"] == ["Linear"]
    assert group["weights"]["num_bits"] == (4 if scheme == "MXFP4" else 8)
    quant_type = "int" if scheme == "W8A8" else "float"
    assert group["weights"]["type"] == quant_type
    assert group["weights"]["symmetric"] is True
    assert group["weights"]["dynamic"] is False
    assert group["weights"]["strategy"] == {
        "W8A8": "channel", "FP8_DYNAMIC": "channel", "FP8_BLOCK": "block", "MXFP4": "group",
    }[scheme]
    activation = group["input_activations"]
    assert activation["dynamic"] is True
    assert activation["num_bits"] == group["weights"]["num_bits"]
    assert activation["type"] == quant_type and activation["symmetric"] is True
    assert activation["strategy"] == ("token" if scheme in ("W8A8", "FP8_DYNAMIC") else "group")
    if scheme == "FP8_BLOCK":
        assert group["weights"]["block_structure"] == [128, 128]
        assert activation["group_size"] == 128
    elif scheme == "MXFP4":
        assert group["weights"]["group_size"] == activation["group_size"] == 32
    assert set(qconfig["ignore"]) == {"lm_head"}


@pytest.mark.parametrize("backend,scheme,naming", [
    pytest.param("new", "FP8_DYNAMIC", "underscore", marks=pytest.mark.cpu),
    pytest.param("new", "FP8_DYNAMIC", "dot", marks=pytest.mark.cpu),
    pytest.param("new", "FP8_BLOCK", "underscore", marks=pytest.mark.cpu),
    pytest.param("new", "FP8_BLOCK", "dot", marks=pytest.mark.cpu),
    pytest.param("new", "MXFP4", "underscore", marks=pytest.mark.gpu),
    pytest.param("new", "MXFP4", "dot", marks=pytest.mark.gpu),
    pytest.param("legacy", "FP8_DYNAMIC", "underscore", marks=pytest.mark.gpu),
    pytest.param("legacy", "FP8_BLOCK", "underscore", marks=pytest.mark.gpu),
    pytest.param("legacy", "MXFP4", "underscore", marks=pytest.mark.gpu),
])
def test_bf16_to_extended_format_checkpoint(checkpoint_factory, build_conversion,
                                            recipe_factory, backend, scheme, naming):
    import torch

    maximum = 6 if scheme == "MXFP4" else 448
    row = [maximum, -maximum, 1, -1, 0, 0.5, -0.5, 4]
    weight = torch.tensor(row, dtype=torch.bfloat16).repeat(128, 16)
    copies = _copies()
    case = checkpoint_factory(text_quant=True, custom_shards=[
        {"x.weight": weight, "x.bias": copies["x.bias"]},
        {"y.weight": weight * 2, "lm_head.weight": copies["lm_head.weight"]},
    ])
    if backend == "new":
        recipe = _engine_recipe(recipe_factory, [{"name": "weights", "match": ["x.weight", "y.weight"],
                                                 "dst": scheme}],
                                primary_scheme=scheme, scale_naming=naming)
    else:
        recipe = recipe_factory(modifier="generalDay0Modifier", scheme=scheme, ignore=["lm_head"])
    conversion = build_conversion(case, custom_recipe=recipe)
    expected, source_of = dict(copies), {name: name for name in copies}
    for name, factor in (("x.weight", 1), ("y.weight", 2)):
        if scheme == "MXFP4":
            codes = torch.tensor([0xF7, 0xA2, 0x10, 0x69], dtype=torch.uint8).repeat(128, 16)
            scale = torch.full((128, 4), 127 if factor == 1 else 128, dtype=torch.uint8)
        else:
            codes = torch.tensor(row, dtype=torch.float32).repeat(128, 16).to(torch.float8_e4m3fn)
            scale = torch.full((128, 1) if scheme == "FP8_DYNAMIC" else (1, 1),
                               factor, dtype=torch.float32)
        scale_name = _scale_key(name, naming)
        expected.update({name: codes, scale_name: scale})
        source_of[name] = source_of[scale_name] = name
    config = _run_and_check(case, conversion, expected, source_of, backend=backend)
    if backend == "legacy" and scheme == "MXFP4":
        qconfig = config["quantization_config"]
        assert qconfig["quant_method"] == "mxfp4"
        assert qconfig["ignore"] == ["lm_head"]
        assert qconfig["fp8_channelwise_layers"] == qconfig["fp8_blockwise_layers"] == []
        assert not ({"config_groups", "format", "quantization_status"} & qconfig.keys())
    else:
        _assert_compressed_config(config, scheme)
    if backend == "new":
        assert config["text_config"]["quantization_config"] == config["quantization_config"]
    else:
        assert "quantization_config" not in config["text_config"]
    assert config["text_config"]["test_sentinel"] == "text-preserve"


@pytest.mark.gpu
@pytest.mark.parametrize("scheme", ["FP8_DYNAMIC", "MXFP4"])
def test_fp8_block_to_extended_format_checkpoint(checkpoint_factory, build_conversion,
                                                recipe_factory, scheme):
    import torch

    maximum = 6 if scheme == "MXFP4" else 224
    row = [maximum, -maximum, 1, -1, 0, 0.5, -0.5, 4]
    raw = torch.tensor(row, dtype=torch.float32).repeat(128, 16).to(torch.float8_e4m3fn)
    copies = _copies()
    case = checkpoint_factory(source_format="FP8", text_quant=True, custom_shards=[
        {"x.weight": raw, "y.weight_scale_inv": torch.tensor([[2]], dtype=torch.float32),
         "x.bias": copies["x.bias"]},
        {"y.weight": raw, "x.weight_scale_inv": torch.tensor([[0.5]], dtype=torch.float32),
         "lm_head.weight": copies["lm_head.weight"]},
    ])
    for name in ("x.weight", "y.weight"):
        assert case.weight_map[name] != case.weight_map[name + "_scale_inv"]
    recipe = _engine_recipe(recipe_factory, [
        {"name": "weights", "match": ["x.weight", "y.weight"], "src": "fp8_block", "dst": scheme},
    ], primary_scheme=scheme, scale_naming="underscore", dequant_dtype="fp32", dequant_block_size=128)
    conversion = build_conversion(case, custom_recipe=recipe)
    expected, source_of = dict(copies), {name: name for name in copies}
    for name, factor in (("x.weight", 0.5), ("y.weight", 2)):
        if scheme == "MXFP4":
            codes = torch.tensor([0xF7, 0xA2, 0x10, 0x69], dtype=torch.uint8).repeat(128, 16)
            scale = torch.full((128, 4), 126 if factor == 0.5 else 128, dtype=torch.uint8)
        else:
            codes = torch.tensor([448, -448, 2, -2, 0, 1, -1, 8], dtype=torch.float32)
            codes = codes.repeat(128, 16).to(torch.float8_e4m3fn)
            scale = torch.full((128, 1), factor / 2, dtype=torch.float32)
        scale_name = _scale_key(name, "underscore")
        expected.update({name: codes, scale_name: scale})
        source_of[name] = source_of[scale_name] = name
    config = _run_and_check(case, conversion, expected, source_of)
    _assert_compressed_config(config, scheme)
    assert config["text_config"]["quantization_config"] == config["quantization_config"]
    assert config["text_config"]["test_sentinel"] == "text-preserve"


@pytest.mark.gpu
@pytest.mark.parametrize("kind,scheme", [
    ("fp8_block", "FP8_DYNAMIC"), ("fp8_block", "BF16"),
    ("nvfp4", "MXFP4"), ("nvfp4", "BF16"),
])
def test_legacy_quantized_source_checkpoint(checkpoint_factory, build_conversion, quantized_source,
                                            kind, scheme):
    import torch

    if kind == "fp8_block":
        row = [224, -224, 1, -1, 0, 0.5, -0.5, 4]
        raw = torch.tensor(row, dtype=torch.float32).repeat(128, 16).to(torch.float8_e4m3fn)
        scales = {"x.weight_scale_inv": torch.tensor([[0.5]], dtype=torch.float32)}
        reference = torch.tensor(row, dtype=torch.float32).repeat(128, 16) * 0.5
        dequant, quant = "dequantModifier", "fp8BasedDay0Modifier"
    else:
        _, raw, scales, reference = quantized_source("nvfp4")
        raw, reference = raw[:, :16], reference[:, :32]
        scales = {key: value[:, :2] if value.ndim == 2 else value for key, value in scales.items()}
        dequant, quant = "nvfp4DequantModifier", "nvfp4BasedMxfp4Day0Modifier"
    other_scales = {key.replace("x.", "y.", 1): value * 4
                    if key.endswith(("_scale_inv", "_scale_2")) else value
                    for key, value in scales.items()}
    copies = _copies()
    case = checkpoint_factory(text_quant=False, custom_shards=[
        {"x.weight": raw, **other_scales, "x.bias": copies["x.bias"]},
        {"y.weight": raw, **scales, "lm_head.weight": copies["lm_head.weight"]},
    ], config_patch={"quantization_config": {"quant_method": "fp8" if kind == "fp8_block" else "nvfp4"}})
    for key in scales:
        assert case.weight_map["x.weight"] != case.weight_map[key]
        assert case.weight_map["y.weight"] != case.weight_map[key.replace("x.", "y.", 1)]
    modifiers = {dequant: {}}
    if scheme != "BF16":
        modifiers[quant] = {"scheme": scheme, "ignore": ["lm_head"]}
    recipe = {"quant_stage": {"quant_modifiers": modifiers}}
    conversion = build_conversion(case, backend="legacy", custom_recipe=recipe)
    assert [type(m).__name__ for m in conversion.compressor.modifiers] == list(modifiers)

    expected = dict(copies)
    for name, factor in (("x.weight", 1), ("y.weight", 4)):
        if scheme == "BF16":
            expected[name] = (reference * factor).to(torch.bfloat16)
        else:
            if scheme == "FP8_DYNAMIC":
                codes = torch.tensor([448, -448, 2, -2, 0, 1, -1, 8], dtype=torch.float32)
                codes = codes.repeat(128, 16).to(torch.float8_e4m3fn)
                scale = torch.full((128, 1), factor / 4, dtype=torch.float32)
            else:
                codes = torch.tensor([[0xD0] * 8 + [0xF1] * 8] * 2, dtype=torch.uint8)
                scale = torch.tensor([[126], [127]], dtype=torch.uint8) + (0 if factor == 1 else 2)
            expected.update({name: codes, name + "_scale": scale})
    source_of = {key: key.removesuffix("_scale") for key in expected}
    config = _run_and_check(case, conversion, expected, source_of, backend="legacy",
                            output_shard="model-00001-of-00001.safetensors")
    assert config["text_config"] == case.config["text_config"]
    if scheme == "BF16":
        assert "quantization_config" not in config
    else:
        assert conversion.compressor.modifiers[-1].valid_scheme_name == scheme
        if scheme == "FP8_DYNAMIC":
            _assert_compressed_config(config, scheme)
        else:
            qconfig = config["quantization_config"]
            assert qconfig["quant_method"] == "mxfp4"
            assert qconfig["ignore"] == ["lm_head"]
            assert not ({"config_groups", "format", "quantization_status"} & qconfig.keys())


@pytest.mark.parametrize("kind,scheme", [
    pytest.param("int4", "W8A8", marks=pytest.mark.cpu),
    pytest.param("int4", "FP8_BLOCK", marks=pytest.mark.cpu),
    pytest.param("int4", "FP8_DYNAMIC", marks=pytest.mark.cpu),
    pytest.param("mxfp4", "W8A8", marks=pytest.mark.gpu),
    pytest.param("mxfp4", "FP8_BLOCK", marks=pytest.mark.gpu),
    pytest.param("mxfp4", "FP8_DYNAMIC", marks=pytest.mark.gpu),
    pytest.param("nvfp4", "W8A8", marks=pytest.mark.cpu),
    pytest.param("nvfp4", "FP8_BLOCK", marks=pytest.mark.cpu),
    pytest.param("nvfp4", "FP8_DYNAMIC", marks=pytest.mark.cpu),
    pytest.param("nvfp4", "MXFP4", marks=pytest.mark.gpu),
])
def test_packed_source_requantization_checkpoint(
    checkpoint_factory, build_conversion, recipe_factory, quantized_source, kind, scheme,
):
    import torch

    name, raw, scales, _ = quantized_source(kind)
    maximum = {"int4": 14, "mxfp4": 12, "nvfp4": 6}[kind]
    if scheme == "FP8_BLOCK":
        width = 64 if kind == "int4" else 32
        values_per_element = 8 if kind == "int4" else 2
        group_size = 16 if kind == "nvfp4" else 32
        repeats = 128 // width
        raw = raw[:, :width // values_per_element].repeat(64, repeats)
        scales = {key: value[:, :width // group_size].repeat(64, repeats)
                  if value.ndim == 2 else value for key, value in scales.items()}

    copies = _copies()
    automatic = kind == "nvfp4" and scheme == "W8A8"
    case = checkpoint_factory(text_quant=True, custom_shards=[
        {name: raw, "x.bias": copies["x.bias"]},
        {**scales, "lm_head.weight": copies["lm_head.weight"]},
    ], config_patch={"quantization_config": {"quant_method": "nvfp4"}} if automatic else None)
    assert all(case.weight_map[name] != case.weight_map[key] for key in scales)
    if automatic:
        conversion = build_conversion(case, auto_output_format="INT8_W8A8")
        assert conversion.args.recipe is None and conversion.args.input_format is None
        params = conversion.recipe["quant_stage"]["quant_modifiers"]["GeneralQuantEngineModifier"]
        assert params["regions"] == [
            {"name": "ignore", "match": ["*lm_head.*"], "src": "nvfp4", "dst": "bf16"},
            {"name": "weights", "match": ["*.weight"], "src": "nvfp4", "dst": "W8A8"},
            {"name": "fallback", "match": "*"},
        ]
        assert params["primary_scheme"] == "W8A8"
        assert params["dequant_dtype"] == "fp32" and params["nvfp4_group_size"] == 16
    else:
        recipe = _engine_recipe(recipe_factory, [
            {"name": "weights", "match": name, "src": kind, "dst": scheme},
        ], primary_scheme=scheme, scale_naming="underscore", dequant_dtype="fp32",
            int4_group_size=32, nvfp4_group_size=16)
        conversion = build_conversion(case, custom_recipe=recipe)

    if scheme == "MXFP4":
        codes = torch.tensor([[0xD0] * 8 + [0xF1] * 9] * 2, dtype=torch.uint8)
        output_scale = torch.tensor([[126, 127], [127, 128]], dtype=torch.uint8)
    else:
        if scheme == "W8A8":
            row = {
                "int4": [-36, -32, -27, -23, -18, -14, -9, -5] * 4
                        + [0, 18, 36, 54, 73, 91, 109, 127] * 4,
                "mxfp4": [3, -32] * 16 + [11, -127],
                "nvfp4": [3, -32] * 8 + [5, -64] * 8 + [11, -127],
            }[kind]
            dtype, divisor = torch.int8, 127
        else:
            row = {
                "int4": [-128, -112, -96, -80, -64, -48, -32, -16] * 4
                        + [0, 64, 128, 192, 256, 320, 384, 448] * 4,
                "mxfp4": [9, -112] * 16 + [36, -448],
                "nvfp4": [9, -112] * 8 + [18, -224] * 8 + [36, -448],
            }[kind]
            dtype, divisor = torch.float8_e4m3fn, 448
        device = "cuda" if kind == "mxfp4" else "cpu"
        if scheme == "FP8_BLOCK":
            factor = {"int4": 0.5, "mxfp4": 2, "nvfp4": 1}[kind]
            row = [value * factor for value in row[:width]]
            codes = torch.tensor([row, [value * 2 for value in row]], dtype=torch.float32)
            codes = codes.repeat(64, repeats).to(dtype)
            block_maximum = 28 if kind == "int4" else 6
            output_scale = torch.tensor([[block_maximum]], dtype=torch.float32, device=device)
        else:
            codes = torch.tensor([row] * 2, dtype=torch.float32).to(dtype)
            output_scale = torch.tensor([[maximum], [maximum * 2]], dtype=torch.float32, device=device)
        output_scale = (output_scale / divisor).cpu()
    expected = {**copies, "x.weight": codes, "x.weight_scale": output_scale}
    source_of = {**{key: key for key in copies}, "x.weight": name, "x.weight_scale": name}
    formats = ("NVFP4", "INT8_W8A8") if automatic else ("user_specified", "user_specified")
    config = _run_and_check(case, conversion, expected, source_of, formats=formats)
    _assert_compressed_config(config, scheme)
    assert config["text_config"]["quantization_config"] == config["quantization_config"]
    assert config["text_config"]["test_sentinel"] == "text-preserve"
    if automatic:
        metadata = json.loads((case.output / "qlean_config.json").read_text())
        assert metadata["scheme"] == "W8A8"


@pytest.mark.gpu
@pytest.mark.parametrize("scheme", ["W8A8", "FP8_DYNAMIC"])
def test_mxfp8_streaming_checkpoint(
    checkpoint_factory, build_conversion, recipe_factory, monkeypatch, scheme,
):
    import torch
    from safetensors import safe_open
    from qlean.modifiers_engine._runtime.context import ShardContext

    row = [-448, -112, 0, 112, 448, 0, 0, 0] * 4 + [1, -1]
    raw = torch.tensor([row] * 5, dtype=torch.float32).to(torch.float8_e4m3fn)
    scales = torch.tensor([[127 + r, 128 + r] for r in range(5)], dtype=torch.uint8)
    copies = _copies()
    case = checkpoint_factory(text_quant=True, custom_shards=[
        {"x.weight": raw, "x.bias": copies["x.bias"]},
        {"x.scale": scales, "lm_head.weight": copies["lm_head.weight"]},
    ])
    assert case.weight_map["x.weight"] != case.weight_map["x.scale"]
    recipe = _engine_recipe(recipe_factory, [
        {"name": "stream", "match": "x.weight", "src": "mxfp8", "dst": scheme,
         "chunked": True, "chunk_rows": 2},
    ], primary_scheme=scheme, dequant_dtype="fp32")
    conversion = build_conversion(case, custom_recipe=recipe)
    reads = []
    real_slice = ShardContext.get_slice

    def read(ctx, name, start, end):
        reads.append((name, start, end))
        return real_slice(ctx, name, start, end)

    monkeypatch.setattr(ShardContext, "get_slice", read)
    if scheme == "W8A8":
        codes = torch.tensor([[-127, -32, 0, 32, 127, 0, 0, 0] * 4 + [1, -1]] * 5,
                             dtype=torch.int8)
        output_scale = torch.tensor([[448 * 2 ** r] for r in range(5)],
                                    dtype=torch.float32, device="cuda")
        output_scale = (output_scale / 127).cpu()
    else:
        codes = torch.tensor([row[:32] + [2, -2]] * 5, dtype=torch.float32).to(torch.float8_e4m3fn)
        output_scale = torch.tensor([[2 ** r] for r in range(5)], dtype=torch.float32)
    expected = {**copies, "x.weight": codes, "x.weight_scale": output_scale}
    source_of = {key: key.removesuffix("_scale") for key in expected}
    config = _run_and_check(case, conversion, expected, source_of)
    assert reads == [(name, start, min(start + 2, 5)) for start in (0, 2, 4)
                     for name in ("x.weight", "x.scale")]
    for filename in case.shard_names:
        with safe_open(str(case.output / filename), framework="pt") as handle:
            stats = json.loads(handle.metadata()["qlean_shard_stats"])
        assert stats.get("streamed_tensors", 0) == int(filename == case.weight_map["x.weight"])
    _assert_compressed_config(config, scheme)
    assert config["text_config"]["quantization_config"] == config["quantization_config"]
    assert config["text_config"]["test_sentinel"] == "text-preserve"


@pytest.mark.gpu
def test_int4_to_mxfp4_checkpoint(checkpoint_factory, build_conversion, recipe_factory, quantized_source):
    import torch

    name, raw, scales, _ = quantized_source("int4")
    copies = _copies()
    case = checkpoint_factory(text_quant=True, custom_shards=[
        {name: raw, "x.bias": copies["x.bias"]},
        {**scales, "lm_head.weight": copies["lm_head.weight"]},
    ])
    assert all(case.weight_map[name] != case.weight_map[key] for key in scales)
    recipe = _engine_recipe(recipe_factory, [
        {"name": "weights", "match": name, "src": "int4", "dst": "MXFP4"},
    ], primary_scheme="MXFP4", scale_naming="underscore", dequant_dtype="fp32", int4_group_size=32)
    conversion = build_conversion(case, custom_recipe=recipe)
    packed_row = [0xEE, 0xCD, 0xBC, 0x9A] * 4 + [0x10, 0x32, 0x44, 0x65] * 4
    expected = {
        **copies,
        "x.weight": torch.tensor([packed_row, packed_row], dtype=torch.uint8),
        "x.weight_scale": torch.tensor([[127, 129], [128, 130]], dtype=torch.uint8),
    }
    source_of = {**{key: key for key in copies}, "x.weight": name, "x.weight_scale": name}
    config = _run_and_check(case, conversion, expected, source_of)
    _assert_compressed_config(config, "MXFP4")
    assert config["text_config"]["quantization_config"] == config["quantization_config"]
    assert config["text_config"]["test_sentinel"] == "text-preserve"


@pytest.mark.parametrize("kind", [
    pytest.param("fp8_block", marks=pytest.mark.gpu),
    pytest.param("mxfp4", marks=pytest.mark.gpu),
    pytest.param("mxfp8", marks=pytest.mark.gpu),
    pytest.param("nvfp4", marks=pytest.mark.cpu),
    pytest.param("int4", marks=pytest.mark.cpu),
])
def test_quantized_source_to_bf16_clears_config(checkpoint_factory, build_conversion, recipe_factory,
                                               quantized_source, kind):
    import torch

    name, raw, scales, reference = quantized_source(kind)
    copies = _copies()
    case = checkpoint_factory(text_quant=True, expert_markers=True, custom_shards=[
        {name: raw, "x.bias": copies["x.bias"]}, {**scales, "lm_head.weight": copies["lm_head.weight"]}])
    assert all(case.weight_map[name] != case.weight_map[key] for key in scales)
    recipe = _engine_recipe(recipe_factory, [{"name": "experts", "match": name, "src": kind, "dst": "bf16"}],
                            primary_scheme=None, config_writer="none")
    conversion = build_conversion(case, custom_recipe=recipe)
    expected = {**copies, "x.weight": reference.to(torch.bfloat16)}
    config = _run_and_check(case, conversion, expected, {**{k: k for k in copies}, "x.weight": name})
    for section in (config, config["text_config"]):
        assert "quantization_config" not in section
        assert "expert_dtype" not in section
    assert config["text_config"]["test_sentinel"] == "text-preserve"


@pytest.mark.cpu
@pytest.mark.parametrize("kind", ["fp8_block", "mxfp4", "mxfp8"])
@pytest.mark.parametrize("naming", ["underscore", "dot"])
def test_passthrough_checkpoint_preserves_source_config(checkpoint_factory, build_conversion, recipe_factory,
                                                         monkeypatch, quantized_source, kind, naming):
    import torch
    from qlean.modifiers_engine._runtime import plan

    name, raw, scales, _ = quantized_source(kind)
    copies = _copies()
    case = checkpoint_factory(text_quant=True, expert_markers=True, custom_shards=[
        {name: raw, "x.bias": copies["x.bias"]}, {**scales, "lm_head.weight": copies["lm_head.weight"]}])
    recipe = _engine_recipe(recipe_factory, [{"name": "experts", "match": name, "src": kind, "dst": "passthrough"}],
                            primary_scheme=None, config_writer="none", scale_naming=naming)
    conversion = build_conversion(case, custom_recipe=recipe)

    def forbidden(*args, **kwargs):
        pytest.fail("Passthrough must not call the quantizer")

    monkeypatch.setattr(plan, "weight_quant", forbidden)
    scale, = scales.values()
    scale_name = _scale_key("x.weight", naming)
    expected = {**copies, "x.weight": raw.view(torch.uint8) if kind == "mxfp4" else raw,
                scale_name: scale.view(torch.uint8) if kind == "mxfp4" else scale}
    config = _run_and_check(case, conversion, expected,
                            {**{k: k for k in copies}, "x.weight": name, scale_name: name})
    assert config == case.config


@pytest.mark.cpu
def test_mixed_precision_writer_and_packed_checkpoint(checkpoint_factory, build_conversion, recipe_factory):
    import torch

    copies = _copies()
    case = checkpoint_factory(text_quant=True, expert_markers=True, custom_shards=[
        {"x.weight": torch.full((64, 64), 7, dtype=torch.bfloat16), "x.bias": copies["x.bias"]},
        {"y.weight": torch.full((64, 64), 448, dtype=torch.bfloat16), "lm_head.weight": copies["lm_head.weight"]}])
    recipe = _engine_recipe(recipe_factory, [
        {"name": "experts", "match": "x.weight", "dst": "acext_int4"},
        {"name": "dense", "match": "y.weight", "dst": "FP8_DYNAMIC"}],
        primary_scheme=None, config_writer="mixed_precision_w4")
    conversion = build_conversion(case, custom_recipe=recipe)
    expected = {**copies, "x.weight": torch.full((64, 32), 0x77, dtype=torch.int8),
                "x.weight_scale": torch.ones((64, 1), dtype=torch.float32),
                "y.weight": torch.full((64, 64), 448, dtype=torch.float32).to(torch.float8_e4m3fn),
                "y.weight_scale": torch.ones((64, 1), dtype=torch.float32)}
    source_of = {key: key.removesuffix("_scale") for key in expected}
    config = _run_and_check(case, conversion, expected, source_of)
    qconfig = {"activation_scheme": "dynamic", "quant_method": "mixed_precision_w4",
               "ignore": ["lm_head"], "expert_dtype": "int4"}
    assert config["quantization_config"] == config["text_config"]["quantization_config"] == qconfig
    assert config["expert_dtype"] == config["text_config"]["expert_dtype"] == "int4"


@pytest.mark.cpu
def test_sglang_mixed_passthrough_writer(checkpoint_factory, build_conversion, recipe_factory):
    import torch

    copies = _copies()
    raw4 = torch.full((128, 64), -9, dtype=torch.int8)
    raw8 = torch.full((128, 128), 2, dtype=torch.float32).to(torch.float8_e4m3fn)
    scale4 = torch.full((128, 4), 127, dtype=torch.uint8)
    scale8 = torch.tensor([[0.5]], dtype=torch.float32)
    case = checkpoint_factory(text_quant=True, expert_markers=True, custom_shards=[
        {"x.weight_packed": raw4, "y.weight_scale_inv": scale8, "x.bias": copies["x.bias"]},
        {"y.weight": raw8, "x.weight_scale": scale4, "lm_head.weight": copies["lm_head.weight"],
         "z.weight": torch.full((2, 32), 448, dtype=torch.bfloat16)}])
    recipe = _engine_recipe(recipe_factory, [
        {"name": "experts", "match": "x.weight_packed", "src": "mxfp4", "dst": "passthrough"},
        {"name": "block", "match": "y.weight", "src": "fp8_block", "dst": "passthrough"},
        {"name": "channel", "match": "z.weight", "dst": "FP8_DYNAMIC"}],
        primary_scheme="MXFP4", sglang_wa=True)
    conversion = build_conversion(case, custom_recipe=recipe)
    expected = {**copies, "x.weight": raw4.view(torch.uint8), "x.weight_scale": scale4,
                "y.weight": raw8, "y.weight_scale": scale8,
                "z.weight": torch.full((2, 32), 448, dtype=torch.float32).to(torch.float8_e4m3fn),
                "z.weight_scale": torch.ones((2, 1), dtype=torch.float32)}
    source_of = {key: key.removesuffix("_scale") for key in expected}
    source_of.update({"x.weight": "x.weight_packed", "x.weight_scale": "x.weight_packed"})
    config = _run_and_check(case, conversion, expected, source_of)
    qconfig = {"quant_method": "mxfp4", "ignore": ["lm_head"], "expert_dtype": "fp4",
               "fp8_blockwise_layers": ["y"], "fp8_channelwise_layers": ["z"]}
    assert {key: config["quantization_config"][key] for key in qconfig} == qconfig
    assert config["text_config"]["quantization_config"] == config["quantization_config"]
    assert not ({"config_groups", "format", "quantization_status"} & config["quantization_config"].keys())
    assert config["expert_dtype"] == config["text_config"]["expert_dtype"] == "fp4"


@pytest.mark.cpu
@pytest.mark.parametrize("kind,overrides", [
    ("mxfp8", {}), ("fp8_block", {"sglang_wa": False}),
    ("fp8_block", {"scale_naming": "dot"}), ("fp8_block", {"dequant_block_size": 32}),
    ("fp8_block", {"primary_scheme": "FP8_DYNAMIC"}),
])
def test_passthrough_accepts_any_export_configuration(
    checkpoint_factory, build_conversion, recipe_factory, quantized_source, kind, overrides,
):
    """Explicit passthrough regions convert with any export-side configuration."""
    from qlean.args import OtherArguments

    name, raw, scales, _ = quantized_source(kind)
    case = checkpoint_factory(custom_shards=[{name: raw}, scales])
    options = dict(primary_scheme="MXFP4", sglang_wa=True)
    options.update(overrides)
    recipe = _engine_recipe(recipe_factory, [{"name": "weights", "match": name, "src": kind, "dst": "passthrough"}],
                            **options)
    conversion = build_conversion(case, custom_recipe=recipe)
    conversion.compressor.post_train(OtherArguments())
    assert list(case.output.rglob("*.safetensors"))
    for filename in ("config.json", "model.safetensors.index.json"):
        assert (case.output / filename).exists()


@pytest.mark.cpu
@pytest.mark.parametrize("scenario", [
    "truncated-weight", "missing-scale", "missing-sidecar", "short-sidecar", "verify-error", "off-to-bound",
])
def test_resume_checkpoint_verification_recovery(
    checkpoint_factory, resume_conversion, recipe_factory, monkeypatch, caplog, scenario,
):
    import torch
    from qlean.modifiers_engine._runtime import verify
    from qlean.modifiers_engine._runtime.fingerprint import FINGERPRINT_FILENAME

    weight = torch.tensor([[127, 0, -2, 1], [-254, 4, 0, -2]], dtype=torch.bfloat16)
    copies = _copies()
    case = checkpoint_factory(custom_shards=[
        {"x.weight": weight, "x.bias": copies["x.bias"]},
        {"y.weight": weight * 2, "lm_head.weight": copies["lm_head.weight"]},
    ])
    codes = torch.tensor([[127, 0, -2, 1], [-127, 2, 0, -1]], dtype=torch.int8)
    expected = {**copies, "x.weight": codes, "y.weight": codes.clone(),
                "x.weight_scale": torch.tensor([[1], [2]], dtype=torch.float32),
                "y.weight_scale": torch.tensor([[2], [4]], dtype=torch.float32)}
    source_of = {name: name.removesuffix("_scale") for name in expected}
    core_stats = {"hit_weights": 2, "hit_fallback": 2, "quant_count": 2,
                  "kept_as_is": 2, "processed_tensors": 4, "bf16_unquantized": ["lm_head"]}
    caplog.set_level(logging.INFO, logger=verify.__name__)

    def perform(mode, processed, *, errors=0):
        recipe = _engine_recipe(recipe_factory, [
            {"name": "weights", "match": ["x.weight", "y.weight"], "dst": "W8A8"},
        ], verify=mode, resume=True)
        expected_stats = dict(core_stats)
        statuses = None
        if mode == "bound":
            expected_stats["verify_rows"] = 2
            statuses = {"W8A8|ok": 2 - errors}
            if errors:
                expected_stats["verify_errors"] = errors
                statuses["W8A8|error_RuntimeError"] = errors
        caplog.clear()
        with resume_conversion(case, custom_recipe=recipe) as run:
            config = _run_and_check(case, run.conversion, expected, source_of)
        _assert_resume_state(case, run, processed=processed,
                             expected_stats=expected_stats, statuses=statuses)
        qconfig = config["quantization_config"]
        assert qconfig["config_groups"]["W8A8"]["weights"]["strategy"] == "channel"
        assert set(qconfig["ignore"]) == {"lm_head"}
        if mode == "bound":
            assert f"Result: {'FAILED' if errors else 'PASSED'}" in caplog.text
            if not errors:
                assert "Result: FAILED" not in caplog.text

    initial_mode = "off" if scenario == "off-to-bound" else "bound"
    injected = []
    with monkeypatch.context() as patch:
        if scenario == "verify-error":
            check = verify._bound_check_chunked

            def fail_x(ref, *args, **kwargs):
                if ref[0, 0].item() == 127:
                    injected.append("x.weight")
                    raise RuntimeError("injected verification failure")
                return check(ref, *args, **kwargs)

            patch.setattr(verify, "_bound_check_chunked", fail_x)
        perform(initial_mode, case.shard_names, errors=int(scenario == "verify-error"))
    assert injected == (["x.weight"] if scenario == "verify-error" else [])
    before = {fn: (case.output / fn).read_bytes() for fn in case.shard_names}
    sidecars = _sidecar_bytes(case) if initial_mode == "bound" else {}
    fingerprint = (case.output / FINGERPRINT_FILENAME).read_bytes()
    if scenario != "verify-error":
        if initial_mode == "bound":
            _stale_summary(case)
        perform(initial_mode, [])
        assert before == {fn: (case.output / fn).read_bytes() for fn in case.shard_names}
        if sidecars:
            assert _sidecar_bytes(case) == sidecars

    target = case.shard_names[0]
    if scenario == "truncated-weight":
        (case.output / target).write_bytes(before[target][:-1])
    elif scenario == "missing-scale":
        from safetensors import safe_open
        from safetensors.torch import save_file

        path = case.output / target
        with safe_open(str(path), framework="pt") as handle:
            tensors = {key: handle.get_tensor(key).clone() for key in handle.keys()}
            metadata = handle.metadata()
        assert "x.weight_scale" in tensors
        tensors.pop("x.weight_scale")
        save_file(tensors, str(path), metadata=metadata)
    elif scenario == "missing-sidecar":
        path = _sidecar(case, target)
        path.rename(path.with_suffix(".saved"))
    elif scenario == "short-sidecar":
        lines = sidecars[target].splitlines(keepends=True)
        assert len(lines) == 2
        _sidecar(case, target).write_bytes(b"".join(lines[:-1]))
    elif scenario == "verify-error":
        assert b"error_RuntimeError" in sidecars[target]
        assert b"injected verification failure" in sidecars[target]

    if initial_mode == "bound":
        _stale_summary(case)
    processed = case.shard_names if scenario == "off-to-bound" else [target]
    perform("bound", processed)
    assert (case.output / FINGERPRINT_FILENAME).read_bytes() == fingerprint
    for filename in set(case.shard_names) - set(processed):
        assert (case.output / filename).read_bytes() == before[filename]
        assert _sidecar(case, filename).read_bytes() == sidecars[filename]
    if initial_mode == "bound" and scenario != "verify-error":
        assert _sidecar_bytes(case) == sidecars
    assert b"error_RuntimeError" not in _sidecar(case, target).read_bytes()
    after = {fn: (case.output / fn).read_bytes() for fn in case.shard_names}
    repaired_sidecars = _sidecar_bytes(case)
    _stale_summary(case)
    perform("bound", [])
    assert after == {fn: (case.output / fn).read_bytes() for fn in case.shard_names}
    assert _sidecar_bytes(case) == repaired_sidecars
