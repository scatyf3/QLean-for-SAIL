# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Single-shard resume checks with small temporary outputs."""

import csv
import json
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.cpu]

_HEADER = ["tensor_name", "out_name", "op_kind", "scheme", "quant_mode", "numel",
           "violations", "boundary", "max_over_ratio", "status", "detail"]


def _unused_process(ctx, file_name, weight_name, weight):
    raise AssertionError("Resume gate tests must not start conversion")


def _copy_keys(src_meta, transform_cfg):
    return list(src_meta)


def _unused_finalize(save_path, stats, transform_cfg):
    raise AssertionError("Resume gate tests must not finalize a full conversion")


def _write_sidecar(path, count):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_HEADER)
        for index in range(count):
            writer.writerow([f"x{index}.weight", f"x{index}.weight", "quant", "W8A8",
                             "int8", 8, 0, 0, 0, "ok", ""])


@pytest.fixture
def resume_case(tmp_path, tensor_factory):
    from safetensors.torch import save_file
    from qlean.modifiers_engine._runtime import engine

    source_dir, output_dir, verify_dir = (tmp_path / name for name in ("source", "output", "verify"))
    for directory in (source_dir, output_dir, verify_dir):
        directory.mkdir()
    source, output = source_dir / "a.safetensors", output_dir / "a.safetensors"
    tensors = {"x.weight": tensor_factory((2, 4), pattern="arange")}
    save_file(tensors, str(source))
    spec = engine.ShardRunSpec(
        model_path=str(source_dir), save_path=str(output_dir),
        source_files=[("a.safetensors", str(source))], weight_map={"x.weight": "a.safetensors"},
        process_tensor=_unused_process, expected_output_keys=_copy_keys,
        finalize=_unused_finalize, verify_dir=str(verify_dir),
    )
    return SimpleNamespace(engine=engine, source=source, output=output,
                           sidecar=verify_dir / "a.csv", tensors=tensors, spec=spec)


def _save_output(case, stats):
    from safetensors.torch import save_file

    save_file(case.tensors, str(case.output), metadata={"qlean_shard_stats": json.dumps(stats)})


def _resume(case):
    return case.engine._try_resume_shard(str(case.source), str(case.output), "a.safetensors", case.spec)


@pytest.mark.parametrize("stats, rows, accepted", [
    ({"numel": 8, "verify_rows": 2, "verify_errors": 0}, 2, True),
    ({"numel": 8, "verify_rows": 0}, 0, True),
    ({"numel": 8}, 0, False),
    ({"verify_rows": 1, "verify_errors": 1}, 1, False),
    ({"verify_rows": 2}, 1, False),
    ({"verify_rows": 1}, 2, False),
    ({"verify_rows": 1}, None, False),
    ({"verify_rows": 0}, None, False),
])
def test_verification_gate_and_restored_statistics(resume_case, stats, rows, accepted):
    case = resume_case
    _save_output(case, stats)
    if rows is not None:
        _write_sidecar(case.sidecar, rows)
    before = (case.source.read_bytes(), case.output.read_bytes())

    result = _resume(case)

    assert result == (({"x.weight": "a.safetensors"}, stats) if accepted else None)
    assert (case.source.read_bytes(), case.output.read_bytes()) == before


def test_verify_disabled_does_not_require_sidecar(resume_case):
    case = resume_case
    stats = {"numel": 8}
    _save_output(case, stats)
    case.spec.verify_dir = None

    assert _resume(case) == ({"x.weight": "a.safetensors"}, stats)
    assert not case.sidecar.exists()


@pytest.mark.parametrize("raw_stats", [None, "{broken-json"])
@pytest.mark.parametrize("verify_enabled", [False, True])
def test_missing_or_corrupt_stats_require_redo_only_with_verify(resume_case, raw_stats, verify_enabled):
    from safetensors.torch import save_file

    case = resume_case
    metadata = {} if raw_stats is None else {"qlean_shard_stats": raw_stats}
    save_file(case.tensors, str(case.output), metadata=metadata)
    _write_sidecar(case.sidecar, 0)
    if not verify_enabled:
        case.spec.verify_dir = None

    assert _resume(case) == (None if verify_enabled else ({"x.weight": "a.safetensors"}, {}))


@pytest.mark.parametrize("damage", [
    "missing", "short_header", "truncated_data", "extra_bytes", "missing_key", "extra_key",
])
def test_incomplete_output_is_not_reused(resume_case, tensor_factory, damage):
    from safetensors.torch import save_file

    case = resume_case
    if damage != "missing":
        _save_output(case, {"verify_rows": 0})
        if damage == "short_header":
            case.output.write_bytes(b"\x00\x00")
        elif damage == "truncated_data":
            case.output.write_bytes(case.output.read_bytes()[:-1])
        elif damage == "extra_bytes":
            case.output.write_bytes(case.output.read_bytes() + b"\x00")
        else:
            tensors = {"unexpected": tensor_factory((1, 1), pattern="zeros")}
            if damage == "extra_key":
                tensors.update(case.tensors)
            save_file(tensors, str(case.output), metadata={"qlean_shard_stats": '{"verify_rows": 0}'})
    _write_sidecar(case.sidecar, 0)

    assert _resume(case) is None


@pytest.mark.parametrize("rows, expected", [(None, -1), (0, 0), (1, 1), (3, 3)])
def test_sidecar_row_count_excludes_header(tmp_path, rows, expected):
    from qlean.modifiers_engine._runtime.engine import _count_sidecar_rows

    path = tmp_path / "a.csv"
    if rows is not None:
        _write_sidecar(path, rows)

    assert _count_sidecar_rows(str(path)) == expected
