# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Summary aggregation and shard coverage using independent CSV data."""

import csv
import json
import logging

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.cpu]

_SIDECAR_HEADER = ["tensor_name", "out_name", "op_kind", "scheme", "quant_mode", "numel",
                   "violations", "boundary", "max_over_ratio", "status", "detail"]
_SUMMARY_HEADER = ["scheme", "tensors", "ok", "violation", "skipped", "error",
                   "total_violations", "total_boundary", "worst_max_over_ratio",
                   "worst_tensor", "worst_shard"]


def _write_sidecar(directory, stem, rows):
    with (directory / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_SIDECAR_HEADER)
        for tensor, scheme, status, violations, boundary, ratio in rows:
            writer.writerow([tensor, tensor, "quant", scheme, "", 8,
                             violations, boundary, ratio, status, ""])


def _reports(directory):
    info = json.loads((directory / "verify_run_info.json").read_text(encoding="utf-8"))
    with (directory / "verify_summary.csv").open(newline="", encoding="utf-8") as handle:
        header, *rows = list(csv.reader(handle))
    assert header == _SUMMARY_HEADER
    return info, rows


def _sources(directory, *stems):
    return [(f"{stem}.safetensors", str(directory / f"{stem}.safetensors")) for stem in stems]


def test_aggregate_status_counts_totals_worst_tensor_and_missing_shard(tmp_path, caplog):
    from qlean.modifiers_engine._runtime.verify import write_verify_summary

    _write_sidecar(tmp_path, "a", [
        ("w0", "W8A8", "ok", 0, 2, 0.5),
        ("w1", "W8A8", "ok", 0, 3, 1),
        ("m0", "MXFP4", "violation", 2, 1, 2),
        ("m1", "MXFP4", "violation", 3, 0, 4),
        ("block", "FP8_BLOCK", "skipped_scheme", -1, -1, -1),
    ])
    _write_sidecar(tmp_path, "b", [
        ("f0", "FP8_DYNAMIC", "ok", 0, 0, 0.75),
        ("bad0", "W8A8", "error_ValueError", -1, -1, -1),
        ("bad1", "W8A8", "error_ValueError", -1, -1, -1),
    ])

    write_verify_summary(str(tmp_path), _sources(tmp_path, "a", "b", "c"),
                         {"verify_rows": 8, "verify_errors": 2})

    info, rows = _reports(tmp_path)
    assert info == {
        "coverage": {"covered_shards": 2, "total_shards": 3, "missing": ["c.safetensors"]},
        "data_rows": 8, "error_rows": 2, "violation_shards": 1, "error_shards": 1,
        "cross_check": {"stats_verify_rows": 8, "stats_verify_errors": 2,
                        "rows_match": True, "errors_match": True},
        "by_scheme_status": {"FP8_BLOCK|skipped_scheme": 1, "FP8_DYNAMIC|ok": 1,
                             "MXFP4|violation": 2, "W8A8|error_ValueError": 2, "W8A8|ok": 2},
    }
    assert rows == [
        ["FP8_BLOCK", "1", "0", "0", "1", "0", "0", "0", "-1", "", ""],
        ["FP8_DYNAMIC", "1", "1", "0", "0", "0", "0", "0", "0.75", "f0", "b.safetensors"],
        ["MXFP4", "2", "0", "2", "0", "0", "5", "1", "4", "m1", "a.safetensors"],
        ["W8A8", "4", "2", "0", "0", "2", "0", "5", "1", "w1", "a.safetensors"],
    ]
    assert "Result: FAILED" in caplog.text


@pytest.mark.parametrize("stats, rows_match, errors_match, passed", [
    ({"verify_rows": 1, "verify_errors": 0}, True, True, True),
    ({"verify_rows": 2, "verify_errors": 0}, False, True, False),
    ({"verify_rows": 1, "verify_errors": 1}, True, False, False),
    ({}, None, True, True),
])
def test_cross_check_uses_independent_sidecar_counts(
    tmp_path, caplog, stats, rows_match, errors_match, passed,
):
    from qlean.modifiers_engine._runtime.verify import write_verify_summary

    caplog.set_level(logging.INFO)
    _write_sidecar(tmp_path, "a", [("w", "W8A8", "ok", 0, 0, 0)])
    write_verify_summary(str(tmp_path), _sources(tmp_path, "a"), stats)

    info, rows = _reports(tmp_path)
    assert info["cross_check"] == {
        "stats_verify_rows": stats.get("verify_rows"),
        "stats_verify_errors": stats.get("verify_errors", 0),
        "rows_match": rows_match, "errors_match": errors_match,
    }
    assert info["data_rows"] == 1
    assert info["error_rows"] == 0
    assert rows == [["W8A8", "1", "1", "0", "0", "0", "0", "0", "0", "w", "a.safetensors"]]
    assert f"Result: {'PASSED' if passed else 'FAILED'}" in caplog.text


def test_header_only_shard_is_covered_without_data_rows(tmp_path, caplog):
    from qlean.modifiers_engine._runtime.verify import write_verify_summary

    caplog.set_level(logging.INFO)
    _write_sidecar(tmp_path, "a", [])
    write_verify_summary(str(tmp_path), _sources(tmp_path, "a"), {"verify_rows": 0})

    info, rows = _reports(tmp_path)
    assert rows == []
    assert info == {
        "coverage": {"covered_shards": 1, "total_shards": 1, "missing": []},
        "data_rows": 0, "error_rows": 0, "violation_shards": 0, "error_shards": 0,
        "cross_check": {"stats_verify_rows": 0, "stats_verify_errors": 0,
                        "rows_match": True, "errors_match": True},
        "by_scheme_status": {},
    }
    assert "Result: PASSED" in caplog.text


def test_infinite_ratio_remains_worst_in_summary(tmp_path, caplog):
    from qlean.modifiers_engine._runtime.verify import write_verify_summary

    _write_sidecar(tmp_path, "a", [
        ("finite", "W8A8", "violation", 1, 0, 4),
        ("overflow", "W8A8", "violation", 1, 0, float("inf")),
    ])
    _write_sidecar(tmp_path, "b", [("unevaluable", "W8A8", "violation", 8, 0, -1)])

    write_verify_summary(str(tmp_path), _sources(tmp_path, "a", "b"), {"verify_rows": 3})

    info, rows = _reports(tmp_path)
    assert info == {
        "coverage": {"covered_shards": 2, "total_shards": 2, "missing": []},
        "data_rows": 3, "error_rows": 0, "violation_shards": 2, "error_shards": 0,
        "cross_check": {"stats_verify_rows": 3, "stats_verify_errors": 0,
                        "rows_match": True, "errors_match": True},
        "by_scheme_status": {"W8A8|violation": 3},
    }
    assert rows == [["W8A8", "3", "0", "3", "0", "0", "10", "0",
                     "inf", "overflow", "a.safetensors"]]
    assert "Result: FAILED" in caplog.text
    assert "Result: PASSED" not in caplog.text


def test_repeat_summary_replaces_previous_totals(tmp_path):
    from qlean.modifiers_engine._runtime.verify import write_verify_summary

    sources = _sources(tmp_path, "a")
    _write_sidecar(tmp_path, "a", [("bad", "W8A8", "violation", 2, 0, 4)])
    write_verify_summary(str(tmp_path), sources, {"verify_rows": 1})
    assert _reports(tmp_path)[0]["violation_shards"] == 1
    _write_sidecar(tmp_path, "a", [("fixed", "W8A8", "ok", 0, 0, 0)])

    write_verify_summary(str(tmp_path), sources, {"verify_rows": 1})

    info, rows = _reports(tmp_path)
    assert info["data_rows"] == 1
    assert info["violation_shards"] == 0
    assert info["by_scheme_status"] == {"W8A8|ok": 1}
    assert rows == [["W8A8", "1", "1", "0", "0", "0", "0", "0", "0", "fixed", "a.safetensors"]]
