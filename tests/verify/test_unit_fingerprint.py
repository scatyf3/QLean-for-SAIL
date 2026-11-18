# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
import json
import os
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.cpu]


def _unexpected(*args, **kwargs):
    pytest.fail("No cleanup, writes, or conversion before rerun approval")


def _snapshot(path):
    return {p.relative_to(path): p.read_bytes() for p in path.rglob("*") if p.is_file()}


@pytest.fixture
def fingerprint_case(tmp_path):
    from qlean.modifiers_engine._runtime import fingerprint

    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (source / "config.json").write_text('{"hidden_size": 4}', encoding="utf-8")
    weight_map = {"x.weight": "a.safetensors", "y.weight": "b.safetensors"}
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8")
    files = []
    for name in weight_map.values():
        path = source / name
        path.write_bytes(b"metadata-only-source")
        files.append((name, str(path)))
    cfg = {
        "_regions": [
            {"name": "first", "match": "*.weight", "dst": "W8A8"},
            {"name": "second", "match": "*.weight", "dst": "MXFP4"},
        ],
        "layout": [0, 2, 1],
    }
    return SimpleNamespace(fp=fingerprint, source=source, output=output,
                           files=files, weight_map=weight_map, cfg=cfg)


def _build(case, *, files=None, cfg=None, num_gpus=1):
    return case.fp.build_fingerprint(
        str(case.source), case.files if files is None else files,
        case.cfg if cfg is None else cfg, None, num_gpus)


def test_fingerprint_ignores_mtime_and_enumeration_order(fingerprint_case):
    case = fingerprint_case
    before = deepcopy(case.cfg)
    expected = _build(case)
    for path in case.source.iterdir():
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    reordered = {key: case.cfg[key] for key in reversed(case.cfg)}

    assert _build(case, files=list(reversed(case.files)), cfg=reordered) == expected
    assert case.cfg == before
    case.fp.save_fingerprint(str(case.output), expected)
    assert case.fp.load_fingerprint(str(case.output)) == expected


def test_fingerprint_ignores_gpu_count(fingerprint_case):
    expected = _build(fingerprint_case)
    actual = _build(fingerprint_case, num_gpus=8)

    assert actual == {**expected, "num_gpus": 8}


def test_fingerprint_detects_same_size_index_edit(fingerprint_case):
    case = fingerprint_case
    path = case.source / "model.safetensors.index.json"
    before = path.read_bytes()
    expected = _build(case)
    index = json.loads(before)
    index["weight_map"] = {"x.weight": "b.safetensors", "y.weight": "a.safetensors"}
    path.write_text(json.dumps(index), encoding="utf-8")

    assert path.read_bytes() != before
    assert path.stat().st_size == len(before)
    assert _build(case)["hash"] != expected["hash"]


@pytest.mark.parametrize("field", ["layout", "_regions"], ids=["numeric-order", "region-order"])
def test_fingerprint_preserves_semantic_list_order(fingerprint_case, field):
    case = fingerprint_case
    before = deepcopy(case.cfg)
    changed = deepcopy(case.cfg)
    changed[field].reverse()

    assert _build(case, cfg=changed)["hash"] != _build(case)["hash"]
    assert case.cfg == before


@pytest.mark.parametrize("record", ["missing", "corrupt", "mismatch"])
def test_untrusted_fingerprint_preserves_existing_outputs(fingerprint_case, monkeypatch, record):
    import torch
    from qlean.modifiers_engine._runtime import engine

    case = fingerprint_case
    spec = engine.ShardRunSpec(
        model_path=str(case.source), save_path=str(case.output),
        source_files=case.files, weight_map=case.weight_map, transform_cfg=case.cfg,
        process_tensor=_unexpected, expected_output_keys=_unexpected,
        finalize=_unexpected, num_gpus=1, resume=True, force_rerun=False,
    )
    if record == "corrupt":
        (case.output / case.fp.FINGERPRINT_FILENAME).write_text("{broken", encoding="utf-8")
        assert case.fp.load_fingerprint(str(case.output))["corrupt"] is True
    elif record == "mismatch":
        previous = _build(case, cfg=spec.transform_cfg)
        previous["hash"] = "different-configuration"
        case.fp.save_fingerprint(str(case.output), previous)
    else:
        assert case.fp.load_fingerprint(str(case.output)) is None
    for name in ("a.safetensors", "config.json", "model.safetensors.index.json"):
        (case.output / name).write_bytes(b"existing-output-must-stay")
    sidecars = case.output / "qlean_verify"
    sidecars.mkdir()
    (sidecars / "a.csv").write_text("previous-report\n", encoding="utf-8")
    before = (_snapshot(case.source), _snapshot(case.output))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(engine.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    monkeypatch.setattr("builtins.input", _unexpected)
    for name in ("_clean_outputs", "save_fingerprint", "_dispatch"):
        monkeypatch.setattr(engine, name, _unexpected)

    with pytest.raises(RuntimeError, match="Cancelled:"):
        engine.ShardEngine().run(spec)

    assert (_snapshot(case.source), _snapshot(case.output)) == before
