# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import csv
import json
import logging

import torch

from qlean.utils.bound_check import (
    HALF_MAX_GAP,
    compute_bound_check,
    dequantize_int8_per_channel,
    dequantize_fp8_per_channel,
    dequantize_mxfp4,
    unpack_int4_acext,
)
from qlean.modifiers_engine._runtime.plan import (
    op_output_names, op_scheme, op_input_view,
)

logger = logging.getLogger(__name__)

# Block schemes lack aligned bounds; acext_int4 is checked in its own branch below.
SCHEME_TABLE = {
    "W8A8": ("int8", dequantize_int8_per_channel),
    "FP8_DYNAMIC": ("fp8", dequantize_fp8_per_channel),
    "MXFP4": ("mxfp4", dequantize_mxfp4),
}

MXFP4_GROUP_SIZE = 32

# At roughly 22 bytes per element, 32M elements keep temporary memory below ~1 GB.
VERIFY_CHUNK_ELEMS = 32 << 20

_VERIFIABLE_KINDS = ("quant", "expert_quant", "quant_pack_acext")

# Resolve optional torch attributes before exception handling can fail on them.
_OOM_ERRORS = tuple(
    e for e in (getattr(torch.cuda, "OutOfMemoryError", None), MemoryError)
    if e is not None
)

# Boundary counts track bound < error <= bound * FP_NOISE_THRESHOLD, not violations.
CSV_HEADER = [
    "tensor_name", "out_name", "op_kind", "scheme", "quant_mode",
    "numel", "violations", "boundary", "max_over_ratio", "status", "detail",
]

# One sidecar slot suffices because each worker processes shards serially.
_cur_file = None
_cur_handle = None
_cur_writer = None

_logged = set()


def sidecar_path(verify_dir: str, file_name: str) -> str:
    return os.path.join(verify_dir, os.path.splitext(file_name)[0] + ".csv")


def _device_label() -> str:
    if not torch.cuda.is_available():
        return "device=cpu"
    index = torch.cuda.current_device()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        ids = [item.strip() for item in visible.split(",")]
        if index < len(ids):
            return f"device=cuda:{index} (CUDA_VISIBLE_DEVICES={ids[index]})"
    return f"device=cuda:{index}"


def _log_once(key, msg: str, level=logging.WARNING) -> None:
    if key in _logged:
        return
    _logged.add(key)
    logger.log(level, msg)


def _get_writer(out_dir: str, file_name: str):
    global _cur_file, _cur_handle, _cur_writer
    if _cur_file == file_name:
        return _cur_writer, _cur_handle
    if _cur_handle is not None:
        _cur_handle.close()
        _cur_handle = None
        _cur_writer = None
    os.makedirs(out_dir, exist_ok=True)
    path = sidecar_path(out_dir, file_name)
    # Overwrite on retry so each shard's sidecar is idempotent.
    handle = open(path, "w", newline="", encoding="utf-8")
    try:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        handle.flush()
    except BaseException:
        # close_writer cannot reach this handle until initialization succeeds.
        handle.close()
        raise
    _cur_file = file_name
    _cur_handle = handle
    _cur_writer = writer
    return writer, handle


def close_writer() -> None:
    # Call only at worker shutdown; later writes would overwrite the same sidecar.
    global _cur_file, _cur_handle, _cur_writer
    if _cur_handle is None:
        return
    try:
        _cur_handle.close()
    except Exception as exc:
        # Close failures must not mask the original processing error.
        logger.warning(f"[verify] Failed to close sidecar {_cur_file}: {exc}")
    finally:
        _cur_file = None
        _cur_handle = None
        _cur_writer = None


def _produced_pair(op: tuple, produced: list):
    base, scale_name = op_output_names(op)[:2]
    out = dict(produced)
    return out[base], out[scale_name]


def _bound_check_chunked(w_ref, qw, qs, quant_mode: str, dequant_fn) -> dict:
    # Independent row chunks limit temporary memory without changing bounds.
    half_max_gap = HALF_MAX_GAP[quant_mode]
    rows, cols = w_ref.shape
    step = max(1, VERIFY_CHUNK_ELEMS // max(1, cols))

    violations = 0
    nonfinite = 0
    boundary = 0
    max_over_ratio = -1.0
    for start in range(0, rows, step):
        end = min(start + step, rows)
        # Compute in fp32 to match the offline CLI's bounds.
        ref = w_ref[start:end].float()
        recon = dequant_fn(qw[start:end], qs[start:end]).float()
        if recon.shape[-1] > cols:
            # MXFP4 dequantization includes group-alignment padding.
            recon = recon[..., :cols]
        part = compute_bound_check(
            ref, recon, half_max_gap, quant_mode=quant_mode,
            group_size=MXFP4_GROUP_SIZE, skip_tail_stats=True,
        )
        violations += part["violations"]
        nonfinite += part["nonfinite"]
        boundary += part["boundary"]
        max_over_ratio = max(max_over_ratio, part["max_over_ratio"])
        del ref, recon

    return {
        "violations": violations,
        "nonfinite": nonfinite,
        "boundary": boundary,
        "max_over_ratio": max_over_ratio,
    }


def verify_op(op, weight, produced, ctx, file_name, name, *, out_dir) -> None:
    # Emit one row per verifiable op, including skips and errors, for resume accounting.
    kind = op[0]

    # Copy-only shards still need a sidecar and verify_rows=0 to prove verification ran.
    writer, handle = _get_writer(out_dir, file_name)
    ctx.bump("verify_rows", 0)
    if kind not in _VERIFIABLE_KINDS:
        return

    out_name = op_output_names(op)[0]
    scheme = op_scheme(op)

    def emit(status, quant_mode="", numel=-1, violations=-1, boundary=-1,
             max_over_ratio=-1.0, detail=""):
        writer.writerow([
            name, out_name, kind, scheme or "", quant_mode, numel, violations,
            boundary, f"{max_over_ratio:.6g}", status, detail,
        ])
        handle.flush()    # Limit redo work after a crash.
        ctx.bump("verify_rows")

    if scheme == "acext_int4":
        # Carriers: quant_pack_acext (2D/3D) and expert_quant (per-expert 2D slice).
        # Unpack + path identification + bound check (absmax only).
        result = None
        status = ""
        detail = ""
        numel = -1
        try:
            w_ref = op_input_view(op, weight)
            numel = w_ref.numel()
            qw_packed, qs = _produced_pair(op, produced)
            # 3D flatten → 2D
            if qw_packed.dim() == 3:
                orig_shape_3d = (qw_packed.shape[0], qw_packed.shape[1], qw_packed.shape[2] * 2)
                qw_packed = qw_packed.reshape(-1, qw_packed.shape[-1])
                out_shape = (orig_shape_3d[0] * orig_shape_3d[1], orig_shape_3d[2])
            else:
                out_shape = (qw_packed.shape[0], qw_packed.shape[1] * 2)
            # Unpack on w_ref's device: chain is pure torch, GPU is ~50x faster
            # than CPU (and avoids 8-worker CPU contention).
            dev = w_ref.device
            codes = unpack_int4_acext(qw_packed.to(dev), out_shape)
            qs = qs.to(dev)
            # Path identification: absmax iff s_saved == clamp(absmax/7, eps)
            eps = torch.finfo(torch.float32).eps
            w_ref_2d = w_ref.reshape(out_shape[0], -1) if w_ref.dim() != 2 else w_ref
            # Reduce in the native dtype to avoid a full fp32 copy on GPU; bf16 -> fp32
            # is exact and order preserving, so the max matches the quantizer bit-for-bit.
            row_absmax = w_ref_2d.abs().max(dim=1, keepdim=True)[0].float()
            expected_scale = torch.clamp(row_absmax / 7.0, min=eps)
            qs_flat = qs.reshape(-1, 1).float() if qs.dim() != 2 else qs.float()
            is_absmax = torch.equal(qs_flat, expected_scale)
            if is_absmax:
                _log_once(
                    ("int4_path", file_name, "absmax"),
                    f"[verify] acext_int4 absmax; shard={file_name}, "
                    f"first: {name} -> {out_name}",
                    logging.INFO,
                )
                result = _bound_check_chunked(
                    w_ref_2d, codes, qs_flat, "int4", dequantize_int8_per_channel,
                )
            else:
                _log_once(
                    ("int4_path", file_name, "non_absmax"),
                    f"[verify] acext_int4 non-absmax skipped; shard={file_name}, "
                    f"first: {name} -> {out_name}",
                    logging.WARNING,
                )
                status = "skipped_int4_nonabsmax"
        except Exception as exc:
            if isinstance(exc, _OOM_ERRORS) and torch.cuda.is_available():
                torch.cuda.empty_cache()
            ctx.bump("verify_errors")
            status = f"error_{type(exc).__name__}"
            detail = str(exc).replace("\n", " ").replace("\r", " ")[:200]
            _log_once(
                ("error", file_name, "acext_int4", type(exc).__name__),
                f"[verify] acext_int4 check failed ({type(exc).__name__}); "
                f"shard={file_name}, first: {name} -> {out_name}: {detail}",
                logging.ERROR,
            )
        if result is None:
            emit(status, "int4", numel, detail=detail)
            return
        violations = result["violations"]
        nonfinite = result["nonfinite"]
        if nonfinite > 0:
            detail = f"nonfinite={nonfinite}"
        if violations > 0:
            _log_once(
                ("violation", file_name, "acext_int4"),
                f"[verify] acext_int4 bound violations; "
                f"shard={file_name}, first: {name} -> {out_name}: violations={violations}, "
                f"nonfinite={nonfinite}, max_over_ratio={result['max_over_ratio']:.4f}",
                logging.ERROR,
            )
        emit("violation" if violations > 0 else "ok",
             "int4", numel, violations, result["boundary"],
             result["max_over_ratio"], detail=detail)
        return
    entry = SCHEME_TABLE.get(scheme)
    if entry is None:
        _log_once(("scheme", scheme),
                  f"[verify] scheme {scheme!r} does not support bound checks; skipped and recorded in sidecar")
        emit("skipped_scheme")
        return
    quant_mode, dequant_fn = entry

    # Keep emit outside the try block so disk failures are not swallowed as check errors.
    result = None
    status = ""
    detail = ""
    numel = -1
    try:
        w_ref = op_input_view(op, weight)
        numel = w_ref.numel()
        if w_ref.dim() != 2:
            # 3D quantization scales [E, rows, 1] differ from flattened reference scales [E].
            status = f"skipped_{w_ref.dim()}d"
        else:
            qw, qs = _produced_pair(op, produced)
            result = _bound_check_chunked(w_ref, qw, qs, quant_mode, dequant_fn)
    except Exception as exc:
        if isinstance(exc, _OOM_ERRORS) and torch.cuda.is_available():
            # Return cached segments so verification OOM does not burden quantization.
            torch.cuda.empty_cache()
        ctx.bump("verify_errors")
        status = f"error_{type(exc).__name__}"
        detail = str(exc).replace("\n", " ").replace("\r", " ")[:200]
        _log_once(
            ("error", file_name, scheme, type(exc).__name__),
            f"[verify] {scheme} check failed ({type(exc).__name__}); recording an error row and continuing; "
            f"logged once per error type in {file_name}; {_device_label()}. "
            f"First case {name} -> {out_name}: {detail}; "
            f"details: {sidecar_path(out_dir, file_name)}",
            logging.ERROR,
        )

    if result is None:
        emit(status, quant_mode, numel, detail=detail)
        return

    violations = result["violations"]
    nonfinite = result["nonfinite"]
    if nonfinite > 0:
        detail = f"nonfinite={nonfinite}"
    if violations > 0:
        _log_once(
            ("violation", file_name, scheme),
            f"[verify] {scheme} bound violations (logged once per scheme in {file_name}); "
            f"{_device_label()}. First case {name} -> {out_name}: violations={violations}, "
            f"nonfinite={nonfinite}, max_over_ratio={result['max_over_ratio']:.4f}; "
            f"details: {sidecar_path(out_dir, file_name)}",
            logging.ERROR,
        )
    emit("violation" if violations > 0 else "ok",
         quant_mode, numel, violations, result["boundary"],
         result["max_over_ratio"], detail=detail)


def write_verify_summary(verify_dir, source_files, stats) -> None:
    # Reporting failures must not invalidate completed model artifacts.
    try:
        by_key: dict = {}          # (scheme, status) -> [count, violations_sum]
        scheme_agg: dict = {}
        data_rows = err_rows = 0
        viol_shards = err_shards = 0
        covered = 0
        missing: list = []
        for file_name, _ in source_files:
            path = sidecar_path(verify_dir, file_name)
            if not os.path.exists(path):
                missing.append(file_name)
                continue
            covered += 1
            shard_v = shard_e = False
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    data_rows += 1
                    sc, st = row["scheme"], row["status"]
                    cell = by_key.setdefault((sc, st), [0, 0])
                    cell[0] += 1
                    v = int(row["violations"])
                    if v > 0:
                        cell[1] += v
                    if st.startswith("error_"):
                        err_rows += 1
                    a = scheme_agg.setdefault(sc, {
                        "tensors": 0, "ok": 0, "violation": 0, "skipped": 0,
                        "error": 0, "tv": 0, "tb": 0,
                        "worst_ratio": -1.0, "worst_tensor": "", "worst_shard": "",
                    })
                    a["tensors"] += 1
                    if st == "ok":
                        a["ok"] += 1
                    elif st == "violation":
                        a["violation"] += 1
                        shard_v = True
                    elif st.startswith("skipped_"):
                        a["skipped"] += 1
                    elif st.startswith("error_"):
                        a["error"] += 1
                        shard_e = True
                    if v > 0:
                        a["tv"] += v
                    b = int(row["boundary"])
                    if b > 0:
                        a["tb"] += b
                    r = float(row["max_over_ratio"])
                    if r > a["worst_ratio"]:
                        a["worst_ratio"] = r
                        a["worst_tensor"] = row["tensor_name"]
                        a["worst_shard"] = file_name
            if shard_v:
                viol_shards += 1
            if shard_e:
                err_shards += 1
        total = len(source_files)

        logger.info(f"[verify] Coverage: {covered}/{total} shards")
        if missing:
            logger.error(f"[verify] {len(missing)} shards lack sidecars (expected 100% coverage; "
                         f"missing sidecars indicate an implementation bug): {missing}")

        exp_rows = stats.get("verify_rows")
        exp_err = stats.get("verify_errors", 0)
        if exp_rows is not None and exp_rows != data_rows:
            logger.error(f"[verify] Row count mismatch with stats: sidecar={data_rows} vs "
                         f"verify_rows={exp_rows}")
        if exp_err != err_rows:
            logger.error(f"[verify] Error row count mismatch with stats: sidecar={err_rows} vs "
                         f"verify_errors={exp_err}")

        summary_csv = os.path.join(verify_dir, "verify_summary.csv")
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["scheme", "tensors", "ok", "violation", "skipped",
                        "error", "total_violations", "total_boundary",
                        "worst_max_over_ratio", "worst_tensor", "worst_shard"])
            for sc, a in sorted(scheme_agg.items()):
                w.writerow([sc, a["tensors"], a["ok"], a["violation"],
                            a["skipped"], a["error"], a["tv"], a["tb"],
                            f"{a['worst_ratio']:.6g}", a["worst_tensor"],
                            a["worst_shard"]])

        info = {
            "coverage": {"covered_shards": covered, "total_shards": total,
                         "missing": missing},
            "data_rows": data_rows,
            "error_rows": err_rows,
            "violation_shards": viol_shards,
            "error_shards": err_shards,
            "cross_check": {
                "stats_verify_rows": exp_rows,
                "stats_verify_errors": exp_err,
                "rows_match": (exp_rows == data_rows) if exp_rows is not None else None,
                "errors_match": exp_err == err_rows,
            },
            "by_scheme_status": {f"{s}|{st}": c
                                 for (s, st), (c, _v) in sorted(by_key.items())},
        }
        with open(os.path.join(verify_dir, "verify_run_info.json"), "w",
                  encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        logger.info(f"[verify] Reports written to {summary_csv} and verify_run_info.json")

        ok_rows = sum(c for (_s, st), (c, _v) in by_key.items() if st == "ok")
        violation_rows = sum(c for (_s, st), (c, _v) in by_key.items()
                             if st == "violation")
        skipped_rows = sum(c for (_s, st), (c, _v) in by_key.items()
                           if st.startswith("skipped_"))
        total_violations = sum(v for _k, (_c, v) in by_key.items())
        rows_match = exp_rows is None or exp_rows == data_rows
        passed = (not missing and err_rows == 0 and violation_rows == 0
                  and rows_match and exp_err == err_rows)
        detail = (f"Checked {covered}/{total} shards, {data_rows} rows = ok {ok_rows} / "
                  f"violation {violation_rows}(total violations {total_violations}) / "
                  f"skipped {skipped_rows} / error {err_rows}"
                  f"; affected shards: {viol_shards} violation / {err_shards} error")
        if passed:
            logger.info(f"[verify] Result: PASSED | {detail}")
        else:
            logger.error(f"[verify] Result: FAILED | {detail}")
    except Exception as exc:
        logger.error(f"[verify] Summary generation failed (model artifacts unaffected): {exc}")
