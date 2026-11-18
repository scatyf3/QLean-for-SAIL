# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import json
import logging
import multiprocessing as mp
import pickle
import shutil
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from glob import glob
from threading import Thread
from typing import Callable, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


def _is_interactive():
    return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()


class _TqdmLoggingHandler(logging.Handler):
    # Keep logs from overwriting progress bars.
    def emit(self, record):
        try:
            tqdm.write(self.format(record), file=sys.stderr)
        except Exception:
            self.handleError(record)


def _setup_worker_logging(use_tqdm_handler: bool = False):
    root = logging.getLogger()
    if root.handlers:
        return
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    if use_tqdm_handler:
        handler = _TqdmLoggingHandler()
    else:
        handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(fmt)
    root.addHandler(handler)
    root.setLevel(logging.INFO)


from qlean.modifiers.utils import FP8_SCALE_SUFFIXES
from qlean.modifiers_engine._runtime.context import ShardContext, LazyWeight
from qlean.modifiers_engine._runtime.dequant_providers import PROVIDERS
from qlean.modifiers_engine._runtime.plan import (
    _expected_keys_accepts_weight_map,
    PLAN_IR_VERSION,
)
from qlean.modifiers_engine._runtime.preflight import preflight
from qlean.modifiers_engine._runtime.region_router import _match_pattern, _validate_match_pattern
from qlean.modifiers_engine._runtime.verify import close_writer, sidecar_path
from qlean.modifiers_engine._runtime.fingerprint import (
    build_fingerprint,
    load_fingerprint,
    save_fingerprint,
)

logger = logging.getLogger(__name__)

STATS_METADATA_KEY = "qlean_shard_stats"


def _dequant_id(dequant_map: Optional[dict]) -> Optional[str]:
    def _id(x):
        if x is None:
            return None
        if isinstance(x, str):
            return x
        return f"{x.__module__}.{x.__qualname__}"

    if dequant_map is None:
        return None
    return json.dumps(
        [[p, _id(v)] for p, v in dequant_map.items()],
        ensure_ascii=False,
    )


def _resolve_provider(provider):
    if provider is None:
        return None
    if isinstance(provider, str):
        if provider not in PROVIDERS:
            raise ValueError(
                f"Unregistered dequant provider '{provider}'; "
                f"available: {sorted(PROVIDERS.keys())}"
            )
        return PROVIDERS[provider]
    if callable(provider):
        return provider
    raise TypeError(
        f"dequant provider must be callable or str, got {type(provider)}"
    )


def _resolve_dequant(name: str, dequant_map: Optional[dict]):
    # A matching None blocks later providers.
    if dequant_map:
        for pattern, entry in dequant_map.items():
            if _match_pattern(name, pattern):
                return _resolve_provider(entry)
    return None


def _resolve_stream(name: str, stream_map: Optional[dict]) -> Optional[int]:
    if stream_map:
        for pattern, chunk_rows in stream_map.items():
            if _match_pattern(name, pattern):
                return chunk_rows
    return None


@dataclass
class ShardRunSpec:
    """Maps use first-match, case-sensitive full-name globs; '*' spans dots, 're:' is unsupported."""

    model_path: str
    save_path: str
    source_files: list                 # list[(file_name, resolved_path)]
    weight_map: dict
    process_tensor: Callable           # (ctx, file_name, weight_name, weight) -> list[(out_name, tensor)]
    expected_output_keys: Callable     # (src_meta, transform_cfg) -> list[str]
    finalize: Callable                 # (save_path, stats, transform_cfg) -> None
    transform_cfg: dict = field(default_factory=dict)
    num_gpus: int = 0                  # 0 uses all visible GPUs.
    dequant_map: Optional[dict] = field(default=None)
    resume: bool = True
    force_rerun: bool = False
    max_shard_bytes_warn: int = 20 * 1024 ** 3
    filter_source_key: Optional[Callable] = None
    verify_dir: Optional[str] = None
    # Row-independent streaming is bit-exact, so it does not affect fingerprints.
    stream_map: Optional[dict] = field(default=None)
    lazy_input: Optional[Callable] = None

    def __post_init__(self):
        for fname in ("dequant_map", "stream_map"):
            for pattern in getattr(self, fname) or {}:
                try:
                    _validate_match_pattern(pattern)
                except ValueError as exc:
                    raise ValueError(f"ShardRunSpec.{fname}[{pattern!r}]: {exc}") from exc

        # Fail early rather than during worker dispatch.
        for fname in ("process_tensor", "expected_output_keys", "finalize",
                      "filter_source_key", "lazy_input"):
            fn = getattr(self, fname)
            if fn is None:
                continue
            try:
                pickle.dumps(fn)
            except Exception as exc:
                raise TypeError(
                    f"ShardRunSpec.{fname} must be a pickleable module-level function, "
                    f"not a lambda, closure or instance method. Original error: {exc}"
                ) from exc
        if self.dequant_map:
            for pattern, prov in self.dequant_map.items():
                if prov is None or isinstance(prov, str):
                    continue
                try:
                    pickle.dumps(prov)
                except Exception as exc:
                    raise TypeError(
                        f"ShardRunSpec.dequant_map[{pattern!r}] callable provider "
                        f"must be a pickleable module-level function. Original error: {exc}"
                    ) from exc

        # Include IR semantics in fingerprints even when callers bypass run_engine.
        if "_plan_ir_version" not in self.transform_cfg:
            self.transform_cfg = {
                **self.transform_cfg, "_plan_ir_version": PLAN_IR_VERSION
            }


class ShardEngine:
    def run(self, spec: ShardRunSpec) -> dict:
        src_real = os.path.realpath(spec.model_path)
        dst_real = os.path.realpath(spec.save_path)
        if src_real == dst_real:
            raise ValueError(
                f"save_path equals model_path ({src_real}). "
                f"Refusing to overwrite source weights with same-name outputs."
            )
        common = os.path.commonpath([src_real, dst_real])
        if common == src_real:
            raise ValueError(
                f"save_path ({dst_real}) is inside model_path ({src_real}). "
                f"Refusing to write same-name outputs into the source directory."
            )
        if common == dst_real:
            raise ValueError(
                f"model_path ({src_real}) is inside save_path ({dst_real}). "
                f"Refusing to run because _clean_outputs could delete source weights."
            )

        preflight(spec.source_files, spec.max_shard_bytes_warn)

        visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
        n_gpus = spec.num_gpus if spec.num_gpus > 0 else max(1, visible)
        n_gpus = max(1, min(n_gpus, max(1, visible)))
        fp = build_fingerprint(spec.model_path, spec.source_files, spec.transform_cfg,
                               _dequant_id(spec.dequant_map), n_gpus)
        prev = load_fingerprint(spec.save_path)

        resume_mode = (spec.resume and prev is not None
                       and not prev.get("corrupt")
                       and prev.get("hash") == fp.get("hash"))
        if resume_mode:
            logger.info("Fingerprint matches; resuming")
        else:
            existing = _existing_outputs(spec.save_path)
            if prev is None:
                if existing:
                    logger.warning(
                        f"save_path contains {len(existing)} .safetensors files without a fingerprint"
                        f"; these may be unrelated outputs, so deletion requires confirmation"
                    )
            elif not spec.resume:
                logger.info("resume=False: resuming disabled; rerunning all shards")
            elif prev.get("corrupt"):
                logger.warning(
                    f"Corrupt fingerprint ({prev.get('error', 'unknown reason')}); "
                    f"cannot determine the previous run configuration"
                )
            else:
                logger.warning("Fingerprint mismatch (configuration or source changed)")
                _show_fp_diff(prev, fp)
            if existing and not spec.force_rerun:
                if not _confirm_rerun(spec.save_path, len(existing)):
                    raise RuntimeError(
                        f"Cancelled: {spec.save_path} was not cleared; run aborted. "
                        f"Set force_rerun=True to skip confirmation."
                    )

        os.makedirs(spec.save_path, exist_ok=True)
        if not resume_mode:
            _clean_outputs(spec.save_path)
            save_fingerprint(spec.save_path, fp)

        groups = partition_by_bytes(spec.source_files, n_gpus)
        logger.info(f"Using {n_gpus} GPUs; shards per GPU={[len(g) for g in groups]}")

        # Source sizes underestimate peak memory when dequantization expands outputs.
        max_shard = max(
            (os.path.getsize(rp) for _, rp in spec.source_files), default=0
        )
        peak_gb = n_gpus * 2 * max_shard / 1024 ** 3
        if peak_gb > 64:
            logger.warning(
                f"Estimated peak RAM: ~{peak_gb:.0f} GB"
                f" ({n_gpus} GPUs x 2 x {max_shard / 1024 ** 3:.1f} GB/shard). "
                f"Insufficient host memory may trigger the OOM killer without a clear error."
            )
        else:
            logger.info(f"Estimated peak RAM: ~{peak_gb:.0f} GB")

        results = _dispatch(spec, groups, n_gpus)

        weight_map: dict = {}
        stats: dict = {}
        for res in results:
            weight_map.update(res["weight_map"])
            merge_stats(stats, res["stats"])

        index_file = os.path.join(spec.save_path, "model.safetensors.index.json")
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)
        logger.info(f"Index written to {index_file} ({len(weight_map)} entries)")

        spec.finalize(spec.save_path, stats, spec.transform_cfg)

        logger.info(f"Completed: {_stats_summary(stats)}")
        return stats


def merge_stats(total: dict, part: dict) -> None:
    # Stats must be JSON-serializable for shard metadata.
    for k, v in part.items():
        if isinstance(v, list):
            total.setdefault(k, []).extend(v)
        else:
            total[k] = total.get(k, 0) + v


def _stats_summary(stats: dict) -> dict:
    return {k: (f"list[{len(v)}]" if isinstance(v, list) else v)
            for k, v in stats.items()}


class _Heartbeat:
    def __init__(self, total: int, desc: str = "",
                 interval: float = 15.0, step_frac: float = 0.05):
        self.total = max(int(total), 1)
        self.desc = desc
        self.interval = interval
        self.step = max(1, int(self.total * step_frac))
        self._t0 = time.monotonic()
        self._last_t = self._t0
        self._last_val = 0
        self._finished = False

    def update(self, cur: int, skipped: int = 0) -> None:
        if self._finished:
            return
        now = time.monotonic()
        if cur >= self.total:
            logger.info(
                f"{self.desc} completed: {self.total}/{self.total}"
                f" (skipped={skipped}, elapsed {now - self._t0:.0f}s)"
            )
            self._finished = True
            return
        if (now - self._last_t >= self.interval) or (cur - self._last_val >= self.step):
            pct = cur * 100 // self.total
            logger.info(f"{self.desc} progress: {cur}/{self.total} ({pct}%) skipped={skipped}")
            self._last_t = now
            self._last_val = cur


class _ProgressView:
    def __init__(self, total: int, desc: str = ""):
        self.total = total
        self._cur = 0
        self._skipped = 0
        if _is_interactive():
            self._bar = tqdm(total=total, desc=desc, position=0)
            self._hb = None
        else:
            self._bar = None
            self._hb = _Heartbeat(total, desc=desc)

    def advance_to(self, cur: int) -> None:
        delta = cur - self._cur
        if delta <= 0:
            return
        self._cur = cur
        if self._bar is not None:
            self._bar.update(delta)
        else:
            self._hb.update(cur, self._skipped)

    def set_skipped(self, n: int) -> None:
        if n == self._skipped:
            return
        self._skipped = n
        if self._bar is not None:
            self._bar.set_postfix_str(f"skipped={n}")

    def close(self) -> None:
        if self._bar is not None:
            if self._skipped:
                self._bar.set_postfix_str(f"skipped={self._skipped}")
            self._bar.close()
        elif self._hb is not None:
            self._hb.update(self.total, self._skipped)


def _dispatch(spec: ShardRunSpec, groups: list, n_gpus: int) -> list:
    total_shards = sum(len(g) for g in groups)
    if n_gpus == 1:
        res = _process_shards(0, 0, groups[0], spec)
        _log_resumed(res["stats"])
        return [res]
    # Fork would inherit an invalid CUDA context.
    spawn_ctx = mp.get_context("spawn")
    shared_counter = spawn_ctx.Value("i", 0)
    resumed_counter = spawn_ctx.Value("i", 0)
    logger.info(
        f"all GPUs: {total_shards} shards -> {n_gpus} GPUs"
        f" (per GPU: {[len(g) for g in groups]})"
        + ("; resume=on" if spec.resume else "")
    )
    results = []
    view = _ProgressView(total_shards, desc="all GPUs")
    # Synchronized counters and locks must be inherited via initializer, not submit.
    with ProcessPoolExecutor(
        max_workers=n_gpus,
        mp_context=spawn_ctx,
        initializer=_pool_initializer,
        initargs=(shared_counter, resumed_counter),
    ) as ex:
        futures = []
        for rank, grp in enumerate(groups):
            if grp:
                futures.append(
                    ex.submit(_process_shards, rank, rank, grp, spec)
                )
        while True:
            done_count = sum(1 for f in futures if f.done())
            with shared_counter.get_lock():
                cur = shared_counter.value
            with resumed_counter.get_lock():
                sk = resumed_counter.value
            view.advance_to(cur)
            view.set_skipped(sk)
            if done_count == len(futures):
                break
            time.sleep(0.5)
        view.close()
        for fut in futures:
            results.append(fut.result())
    merged = {}
    for r in results:
        merge_stats(merged, r["stats"])
    _log_resumed(merged)
    return results


def _log_resumed(stats: dict) -> None:
    files = stats.get("resumed_files", [])
    if files:
        logger.info(f"Resume skipped {len(files)} completed shards: {files}")


_shared_progress_counter = None
_shared_resumed_counter = None


def _pool_initializer(counter, resumed_counter):
    global _shared_progress_counter, _shared_resumed_counter
    _shared_progress_counter = counter
    _shared_resumed_counter = resumed_counter

def _tensors_to_cpu(tensors: dict) -> dict:
    return {k: (v.cpu()).contiguous() for k, v in tensors.items()}

def safe_save_file(tensors: dict, path: str, metadata=None) -> None:
    save_file(_tensors_to_cpu(tensors), path, metadata=metadata)

def _process_shards(rank: int, device_index: int, files: list,
                    spec: ShardRunSpec) -> dict:
    shared_counter = _shared_progress_counter
    resumed_counter = _shared_resumed_counter
    is_inline = shared_counter is None
    _setup_worker_logging(use_tqdm_handler=is_inline and _is_interactive())
    torch.set_default_dtype(torch.bfloat16)
    if torch.cuda.is_available():
        torch.cuda.set_device(device_index)
    device = f"cuda:{device_index}" if torch.cuda.is_available() else "cpu"

    ctx = ShardContext(device, spec.model_path, spec.weight_map, spec.transform_cfg)
    out_weight_map: dict = {}
    totals: dict = {}

    write_thread: Optional[Thread] = None
    write_exc: list = []

    def _join_write(reraise: bool = True):
        nonlocal write_thread
        if write_thread is not None:
            write_thread.join()
            write_thread = None
        if reraise and write_exc:
            raise write_exc.pop()

    # Only the main process renders progress to avoid competing stderr output.
    view = _ProgressView(len(files), desc=f"rank{rank}") if is_inline else None
    try:
        n = len(files)
        if is_inline:
            logger.info(f"[rank{rank}] Processing {n} shards (device={device})")
        for i, (file_name, resolved_path) in enumerate(files):
            out_path = os.path.join(spec.save_path, file_name)

            if spec.resume:
                resumed = _try_resume_shard(resolved_path, out_path, file_name, spec)
                if resumed is not None:
                    fragment, shard_stats = resumed
                    out_weight_map.update(fragment)
                    merge_stats(totals, shard_stats)
                    totals["resumed_shards"] = totals.get("resumed_shards", 0) + 1
                    totals.setdefault("resumed_files", []).append(file_name)
                    if shared_counter is not None:
                        with shared_counter.get_lock():
                            shared_counter.value += 1
                    if resumed_counter is not None:
                        with resumed_counter.get_lock():
                            resumed_counter.value += 1
                    if view is not None:
                        view.advance_to(i + 1)
                        view.set_skipped(totals["resumed_shards"])
                    continue

            ctx.stats = {}
            new_state_dict = _process_one_shard(ctx, spec, file_name, resolved_path)

            # Reject plan/execution mismatches before writing, not on the next resume.
            with safe_open(resolved_path, framework="pt") as _s:
                _src_meta = _source_meta(_s, spec)
            _ek_kwargs = {}
            if _expected_keys_accepts_weight_map(spec.expected_output_keys):
                _ek_kwargs["weight_map"] = spec.weight_map
            _expected = set(spec.expected_output_keys(
                _src_meta, spec.transform_cfg, **_ek_kwargs))
            _actual = set(new_state_dict.keys())
            if _actual != _expected:
                _missing = sorted(_expected - _actual)
                _extra = sorted(_actual - _expected)
                raise RuntimeError(
                    f"[{file_name}] Output keys do not match expected_output_keys!\n"
                    f"  Missing ({len(_missing)}): {_missing[:10]}\n"
                    f"  Extra ({len(_extra)}): {_extra[:10]}"
                )

            shard_stats = ctx.take_stats()
            merge_stats(totals, shard_stats)

            # Write final paths directly to avoid NFS rename deadlocks; resume checks truncation.
            _join_write()
            meta_json = json.dumps(shard_stats, ensure_ascii=False)

            def _write(sd=new_state_dict, path=out_path, meta=meta_json):
                try:
                    safe_save_file(sd, path, metadata={
                        "format": "pt",
                        STATS_METADATA_KEY: meta,
                    })
                except BaseException as exc:  # noqa: BLE001 -- Re-raise on the main thread.
                    write_exc.append(exc)

            write_thread = Thread(target=_write, daemon=True)
            write_thread.start()

            for k in new_state_dict:
                out_weight_map[k] = file_name
            del new_state_dict  # The writer retains tensors through its default arguments.
            torch.cuda.empty_cache()
            if shared_counter is not None:
                with shared_counter.get_lock():
                    shared_counter.value += 1
            if view is not None:
                view.advance_to(i + 1)

        _join_write()
    finally:
        if view is not None:
            view.close()
        _join_write(reraise=False)  # Finish pending writes even when processing fails.
        ctx.close()
        close_writer()

    return {"weight_map": out_weight_map, "stats": totals}


def _process_one_shard(ctx: ShardContext, spec: ShardRunSpec,
                       file_name: str, resolved_path: str) -> dict:
    new_state_dict: dict = {}

    def _emit(out_name: str, tensor: torch.Tensor) -> None:
        # Free GPU memory while satisfying safetensors' contiguous layout requirement.
        new_state_dict[out_name] = tensor.detach().contiguous().cpu()

    with safe_open(resolved_path, framework="pt", device="cpu") as f:
        ctx.begin_shard(file_name, f)
        for weight_name in _weight_keys(f, spec):
            if spec.lazy_input is not None:
                sl = f.get_slice(weight_name)
                shape = list(sl.get_shape())
                if spec.lazy_input(weight_name, shape, spec.transform_cfg,
                                   weight_map=spec.weight_map):
                    lazy = LazyWeight(weight_name, shape, sl.get_dtype(), chunk_rows=0)
                    for out_name, tensor in spec.process_tensor(
                            ctx, file_name, weight_name, lazy):
                        _emit(out_name, tensor)
                    ctx.bump("processed_tensors")
                    continue

            # LazyWeight avoids OOM from materializing and dequantizing entire tables.
            chunk_rows = _resolve_stream(weight_name, spec.stream_map)
            if chunk_rows is not None:
                sl = f.get_slice(weight_name)
                lazy = LazyWeight(
                    name=weight_name,
                    shape=list(sl.get_shape()),
                    dtype=sl.get_dtype(),
                    chunk_rows=chunk_rows,
                )
                for out_name, tensor in spec.process_tensor(
                        ctx, file_name, weight_name, lazy):
                    _emit(out_name, tensor)
                ctx.bump("processed_tensors")
                ctx.bump("streamed_tensors")
                continue

            raw = f.get_tensor(weight_name)
            if ctx.device != "cpu":
                raw = raw.to(ctx.device)
            provider = _resolve_dequant(weight_name, spec.dequant_map)
            weight = provider(ctx, weight_name, raw) if provider is not None else raw
            if weight is not raw:
                del raw

            for out_name, tensor in spec.process_tensor(ctx, file_name, weight_name, weight):
                _emit(out_name, tensor)
            ctx.bump("processed_tensors")
            del weight

    return new_state_dict


def _weight_keys(handle, spec: ShardRunSpec) -> list:
    # Default filtering drops source scales because transforms regenerate them.
    if spec.filter_source_key is not None:
        return [k for k in handle.keys() if spec.filter_source_key(k)]
    return [k for k in handle.keys()
            if not any(k.endswith(s) for s in FP8_SCALE_SUFFIXES)]


def _source_meta(handle, spec: ShardRunSpec) -> dict:
    meta = {}
    for k in _weight_keys(handle, spec):
        sl = handle.get_slice(k)
        meta[k] = {"dtype": sl.get_dtype(), "shape": list(sl.get_shape())}
    return meta


def _count_sidecar_rows(csv_path: str) -> int:
    # With verify_errors == 0, details contain no embedded newlines.
    if not os.path.exists(csv_path):
        return -1
    with open(csv_path, "rb") as f:
        return f.read().count(b"\n") - 1


def _try_resume_shard(src_path: str, out_path: str, file_name: str,
                      spec: ShardRunSpec) -> Optional[tuple]:
    if not os.path.exists(out_path):
        return None
    try:
        # A valid header and key set can still hide truncated tensor data.
        expected_size = _safetensors_expected_size(out_path)
        actual_size = os.path.getsize(out_path)
        if expected_size < 0 or actual_size != expected_size:
            logger.warning(
                f"[resume] Invalid size for {os.path.basename(out_path)}"
                f" (actual {actual_size}, expected {expected_size}); redoing shard"
            )
            return None

        with safe_open(src_path, framework="pt") as s:
            src_meta = _source_meta(s, spec)
        ek_kwargs = {}
        if _expected_keys_accepts_weight_map(spec.expected_output_keys):
            ek_kwargs["weight_map"] = spec.weight_map
        expected = set(spec.expected_output_keys(
            src_meta, spec.transform_cfg, **ek_kwargs))

        with safe_open(out_path, framework="pt") as o:
            if set(o.keys()) != expected:
                logger.warning(
                    f"[resume] Output keys in {os.path.basename(out_path)} do not match expectations; redoing shard"
                )
                return None
            metadata = o.metadata() or {}

        # Restore persisted stats so resumed shards contribute to final totals.
        shard_stats: dict = {}
        raw_stats = metadata.get(STATS_METADATA_KEY)
        if raw_stats:
            try:
                shard_stats = json.loads(raw_stats)
            except Exception as exc:
                logger.warning(
                    f"[resume] {os.path.basename(out_path)} has corrupt stats metadata ({exc}); "
                    f"restoring only weight_map; final totals will be incomplete"
                )
        else:
            logger.warning(
                f"[resume] {os.path.basename(out_path)} has no stats metadata (legacy output?); "
                f"restoring only weight_map; final totals will be incomplete"
            )
        # Valid weights alone do not prove verification completed.
        if spec.verify_dir is not None:
            if "verify_rows" not in shard_stats:
                # Missing differs from zero: copy-only shards still record verify_rows=0.
                logger.warning(
                    f"[resume] {os.path.basename(out_path)} has no verify_rows (verification was disabled); "
                    f"redoing shard to verify"
                )
                return None
            if shard_stats.get("verify_errors", 0):
                logger.warning(
                    f"[resume] {os.path.basename(out_path)} has {shard_stats['verify_errors']} "
                    f"verification error rows; verification was incomplete, redoing shard"
                )
                return None
            # Error-free sidecar row counts must match persisted metadata.
            csv_rows = _count_sidecar_rows(sidecar_path(spec.verify_dir, file_name))
            if csv_rows != shard_stats["verify_rows"]:
                logger.warning(
                    f"[resume] {os.path.basename(out_path)} sidecar rows {csv_rows} "
                    f"!= verify_rows {shard_stats['verify_rows']}; redoing shard"
                )
                return None
        return {k: file_name for k in expected}, shard_stats
    except Exception as exc:
        logger.warning(f"[resume] Invalid output {os.path.basename(out_path)} ({exc}); redoing shard")
        return None


def _safetensors_expected_size(path: str) -> int:
    try:
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                return -1
            header_len = struct.unpack("<Q", raw)[0]
            header_bytes = f.read(header_len)
            if len(header_bytes) < header_len:
                return -1
        header = json.loads(header_bytes)
        max_end = 0
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            offsets = meta.get("data_offsets", [0, 0])
            if offsets[1] > max_end:
                max_end = offsets[1]
        return 8 + header_len + max_end
    except Exception:
        return -1


def partition_by_bytes(source_files: list, n_gpus: int) -> list:
    sized = [(item, os.path.getsize(item[1])) for item in source_files]
    sized.sort(key=lambda x: -x[1])
    groups: list = [[] for _ in range(n_gpus)]
    load = [0] * n_gpus
    for item, size in sized:
        r = min(range(n_gpus), key=lambda i: load[i])
        groups[r].append(item)
        load[r] += size
    return groups


def _existing_outputs(save_path: str) -> list:
    return sorted(glob(os.path.join(save_path, "*.safetensors")))


def _clean_outputs(save_path: str) -> None:
    # Full reruns remove stale shards and sidecars; resumes must retain both.
    for path in _existing_outputs(save_path):
        os.remove(path)
    verify_dir = os.path.join(save_path, "qlean_verify")
    if os.path.isdir(verify_dir):
        shutil.rmtree(verify_dir)


def _show_fp_diff(prev: dict, cur: dict) -> None:
    old_c, new_c = prev.get("file_count"), cur.get("file_count")
    if old_c != new_c:
        logger.info(f"  file_count: {old_c} → {new_c}")
    logger.info("  hash differs (transform_cfg / dequant / source files / index.json content changed)")


def _confirm_rerun(save_path: str, n_files: int) -> bool:
    if not sys.stdin.isatty():
        logger.error(
            f"Non-interactive stdin: refusing to clear {save_path} containing "
            f"{n_files} .safetensors files. Set force_rerun=True to confirm a full rerun."
        )
        return False
    try:
        answer = input(
            f"Clear {save_path} containing {n_files} .safetensors files and rerun all shards? [y/N] "
        )
        return answer.strip().lower() == "y"
    except (EOFError, KeyboardInterrupt):
        return False
