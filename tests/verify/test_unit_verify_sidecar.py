# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verification status, quantized carriers, and sidecar output."""

import csv
import json
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.cpu]

_HEADER = ["tensor_name", "out_name", "op_kind", "scheme", "quant_mode", "numel",
           "violations", "boundary", "max_over_ratio", "status", "detail"]


@pytest.fixture
def verify_case(tmp_path, tensor_factory, monkeypatch):
    import torch
    from qlean.modifiers_engine._runtime import verify
    from qlean.modifiers_engine._runtime.context import ShardContext

    for name in ("_cur_file", "_cur_handle", "_cur_writer"):
        monkeypatch.setattr(verify, name, None)
    monkeypatch.setattr(verify, "_logged", set())
    monkeypatch.setattr(verify, "_device_label", lambda: "device=cpu")
    weight = tensor_factory((2, 4), pattern="zeros")
    weight[:, 0] = 127
    produced = [("x.weight", weight.to(torch.int8)),
                ("x.weight_scale", tensor_factory((2, 1), pattern="zeros").fill_(1))]
    case = SimpleNamespace(v=verify, ctx=ShardContext("cpu", str(tmp_path), {}, {}),
                           directory=tmp_path, weight=weight, produced=produced)
    try:
        yield case
    finally:
        verify.close_writer()
        case.ctx.close()


def _emit(case, op=("quant", "x.weight", "W8A8"), *, shard="a.safetensors"):
    case.v.verify_op(op, case.weight, case.produced, case.ctx, shard, "x.weight",
                     out_dir=str(case.directory))


def _rows(case, shard="a.safetensors"):
    with open(case.v.sidecar_path(str(case.directory), shard), newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == _HEADER
        return list(reader)


@pytest.mark.parametrize("case_name, status, violations, boundary, ratio", [
    ("valid", "ok", 0, 0, "0"),
    ("boundary", "ok", 0, 1, "1.00003"),
    ("damaged_scale", "violation", 2, 0, "254"),
    ("bad_shape", "error_RuntimeError", -1, -1, "-1"),
])
def test_status_and_statistics_have_independent_expectations(
    verify_case, tensor_factory, case_name, status, violations, boundary, ratio,
):
    import torch

    case = verify_case
    if case_name == "boundary":
        case.weight[0, 1] = 0.5 - 2 ** -16
        case.produced[0][1][0, 1] = 1
    elif case_name == "damaged_scale":
        case.produced[1][1].mul_(2)
    elif case_name == "bad_shape":
        case.produced[1] = ("x.weight_scale", tensor_factory((2, 2), pattern="zeros").fill_(1))
    original_weight = case.weight.clone()
    original_outputs = [(name, tensor.clone()) for name, tensor in case.produced]

    _emit(case)

    rows = _rows(case)
    assert len(rows) == 1
    row = rows[0]
    detail = row.pop("detail")
    assert bool(detail) == (case_name == "bad_shape")
    assert row == {
        "tensor_name": "x.weight", "out_name": "x.weight", "op_kind": "quant",
        "scheme": "W8A8", "quant_mode": "int8", "numel": "8",
        "violations": str(violations), "boundary": str(boundary),
        "max_over_ratio": ratio, "status": status,
    }
    assert case.ctx.stats == ({"verify_rows": 1, "verify_errors": 1}
                              if case_name == "bad_shape" else {"verify_rows": 1})
    torch.testing.assert_close(case.weight, original_weight, rtol=0, atol=0)
    for (name, tensor), (old_name, old_tensor) in zip(case.produced, original_outputs):
        assert name == old_name
        torch.testing.assert_close(tensor, old_tensor, rtol=0, atol=0)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")],
                         ids=["nan", "positive-inf", "negative-inf"])
def test_nonfinite_scale_records_violation_and_failed_summary(verify_case, caplog, invalid):
    case = verify_case
    case.produced[1][1][0, 0] = invalid

    _emit(case)

    row, = _rows(case)
    assert row == {
        "tensor_name": "x.weight", "out_name": "x.weight", "op_kind": "quant",
        "scheme": "W8A8", "quant_mode": "int8", "numel": "8",
        "violations": "4", "boundary": "0", "max_over_ratio": "0",
        "status": "violation", "detail": "nonfinite=4",
    }
    assert case.ctx.stats == {"verify_rows": 1}
    case.v.close_writer()
    case.v.write_verify_summary(str(case.directory), [("a.safetensors", "unused")],
                                case.ctx.stats)
    info = json.loads((case.directory / "verify_run_info.json").read_text())
    assert info == {
        "coverage": {"covered_shards": 1, "total_shards": 1, "missing": []},
        "data_rows": 1, "error_rows": 0, "violation_shards": 1, "error_shards": 0,
        "cross_check": {"stats_verify_rows": 1, "stats_verify_errors": 0,
                        "rows_match": True, "errors_match": True},
        "by_scheme_status": {"W8A8|violation": 1},
    }
    with (case.directory / "verify_summary.csv").open(newline="", encoding="utf-8") as handle:
        summary, = csv.DictReader(handle)
    assert summary == {
        "scheme": "W8A8", "tensors": "1", "ok": "0", "violation": "1", "skipped": "0",
        "error": "0", "total_violations": "4", "total_boundary": "0",
        "worst_max_over_ratio": "0", "worst_tensor": "x.weight", "worst_shard": "a.safetensors",
    }
    assert "Result: FAILED" in caplog.text
    assert "Result: PASSED" not in caplog.text


@pytest.mark.parametrize("scheme", ["FP8_DYNAMIC", "MXFP4"])
@pytest.mark.parametrize("damaged", [False, True], ids=["valid", "damaged-scale"])
def test_extended_formats_detect_scale_damage(verify_case, scheme, damaged):
    import torch

    case = verify_case
    maximum = 448 if scheme == "FP8_DYNAMIC" else 6
    case.weight = torch.zeros((2, 32), dtype=torch.float32)
    case.weight[:, 0] = torch.tensor([maximum, -2 * maximum], dtype=torch.float32)
    if scheme == "FP8_DYNAMIC":
        raw = torch.zeros((2, 32), dtype=torch.uint8)
        raw[:, 0] = torch.tensor([0x7E, 0xFE], dtype=torch.uint8)
        raw = raw.view(torch.float8_e4m3fn)
        scale = torch.tensor([[1], [2]], dtype=torch.float32)
        if damaged:
            scale.mul_(2)
    else:
        raw = torch.zeros((2, 16), dtype=torch.uint8)
        raw[:, 0] = torch.tensor([0x07, 0x0F], dtype=torch.uint8)
        scale = torch.tensor([[127], [128]], dtype=torch.uint8)
        if damaged:
            scale.add_(1)
    case.produced = [("x.weight", raw), ("x.weight_scale", scale)]
    before = [tensor.clone() for tensor in (case.weight, raw, scale)]

    _emit(case, ("quant", "x.weight", scheme))

    row, = _rows(case)
    ratio = (28 if scheme == "FP8_DYNAMIC" else 6) if damaged else 0
    assert row == {
        "tensor_name": "x.weight", "out_name": "x.weight", "op_kind": "quant",
        "scheme": scheme, "quant_mode": "fp8" if scheme == "FP8_DYNAMIC" else "mxfp4",
        "numel": "64", "violations": "2" if damaged else "0", "boundary": "0",
        "max_over_ratio": str(ratio), "status": "violation" if damaged else "ok", "detail": "",
    }
    assert case.ctx.stats == {"verify_rows": 1}
    for actual, expected in zip((case.weight, raw, scale), before):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)


@pytest.mark.parametrize("scheme, batched, status, numel", [
    ("FP8_BLOCK", False, "skipped_scheme", -1),
    ("W8A8", True, "skipped_3d", 16),
])
def test_unsupported_checks_are_skipped_not_passed(verify_case, scheme, batched, status, numel):
    case = verify_case
    if batched:
        case.weight = case.weight.unsqueeze(0).expand(2, -1, -1)
    case.produced = []
    _emit(case, ("quant", "x.weight", scheme))
    row, = _rows(case)
    assert row["status"] == status
    assert int(row["numel"]) == numel
    assert int(row["violations"]) == int(row["boundary"]) == -1
    assert case.ctx.stats == {"verify_rows": 1}


@pytest.mark.parametrize("carrier", ["quant_pack_acext", "expert_quant"])
@pytest.mark.parametrize("saved_scale, status", [(1.0, "ok"), (2.0, "skipped_int4_nonabsmax")])
def test_int4_carriers_distinguish_absmax_from_other_scales(
    verify_case, tensor_factory, carrier, saved_scale, status,
):
    import torch

    case = verify_case
    case.weight = tensor_factory((64, 64), pattern="zeros").fill_(7)
    if carrier == "expert_quant":
        case.weight = case.weight.unsqueeze(0).repeat(2, 1, 1)
        case.weight[0].zero_()
        op = ("expert_quant", "x.weight", 1, "full", False, "acext_int4")
    else:
        op = ("quant_pack_acext", "x.weight")
    case.produced = [
        ("x.weight", tensor_factory((64, 32), pattern="zeros", dtype=torch.uint8).fill_(0x77)),
        ("x.weight_scale", tensor_factory((64, 1), pattern="zeros").fill_(saved_scale)),
    ]
    _emit(case, op)
    row, = _rows(case)
    assert row["op_kind"] == carrier
    assert row["scheme"] == "acext_int4"
    assert row["quant_mode"] == "int4"
    assert row["status"] == status
    assert int(row["numel"]) == 4096
    assert int(row["violations"]) == int(row["boundary"]) == (0 if status == "ok" else -1)
    assert float(row["max_over_ratio"]) == (0.0 if status == "ok" else -1.0)
    assert case.ctx.stats == {"verify_rows": 1}


def test_copy_only_creates_header_and_close_is_idempotent(verify_case):
    case = verify_case
    _emit(case, ("copy", "x.weight"))
    assert _rows(case) == []
    assert case.ctx.stats == {"verify_rows": 0}
    case.v.close_writer()
    case.v.close_writer()
    assert case.v._cur_handle is case.v._cur_writer is case.v._cur_file is None


def test_shard_switch_and_retry_overwrite_instead_of_append(verify_case):
    from qlean.modifiers_engine._runtime.context import ShardContext

    case = verify_case
    _emit(case)
    _emit(case)
    _emit(case, shard="b.safetensors")
    assert len(_rows(case)) == 2
    assert len(_rows(case, "b.safetensors")) == 1
    assert case.ctx.stats == {"verify_rows": 3}
    case.ctx.close()
    case.ctx = ShardContext("cpu", str(case.directory), {}, {})
    _emit(case)
    assert len(_rows(case)) == 1
    assert len(_rows(case, "b.safetensors")) == 1
    assert case.ctx.stats == {"verify_rows": 1}
