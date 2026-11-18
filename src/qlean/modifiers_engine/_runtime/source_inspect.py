# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import struct
from collections import Counter
from pathlib import Path

from qlean.modifiers_engine._runtime.scale_utils import find_scale_name
from qlean.modifiers_engine._runtime.region_router import (
    CONSUMED_SOURCE_SUFFIXES,
    SRC_KIND,
    effective_src,
    is_passthrough,
    match_patterns,
    match_region,
    region_patterns,
)

logger = logging.getLogger(__name__)

_QUANT_DTYPES = frozenset({"I8", "U8", "I32", "F8_E4M3", "F8_E5M2"})

_SRC_EXPECT_DTYPE = {
    "fp8_block": frozenset({"F8_E4M3"}),
    "mxfp4": frozenset({"I8", "U8"}),
    "mxfp8": frozenset({"F8_E4M3", "F8_E5M2"}),
    "nvfp4": frozenset({"U8"}),
    "int4": frozenset({"I32"}),
}

# Source auxiliary keys are filtered before engine routing.
_SCALE_KEY_SUFFIXES = frozenset(
    suffix for suffixes in CONSUMED_SOURCE_SUFFIXES.values() for suffix in suffixes
)


def read_source_index(index_path: str) -> dict:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate key {key!r} in source index {index_path}")
            result[key] = value
        return result

    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f, object_pairs_hook=unique_object)["weight_map"]


def _read_safetensors_header(path: str) -> dict:
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            raise ValueError(f"Incomplete safetensors header: {path}")
        (n,) = struct.unpack("<Q", raw)
        header = json.loads(f.read(n).decode("utf-8"))
    header.pop("__metadata__", None)
    return header


def _is_scale_key(name: str) -> bool:
    return any(name.endswith(s) for s in _SCALE_KEY_SUFFIXES)


def inspect_source_regions(model_path: str, weight_map: dict,
                           regions: list) -> dict:
    stats = {}
    problems = []
    # Raw hits ignore routing order and when; attributed hits count the winning pattern.
    raw = Counter()
    attributed = Counter()

    shard_names = set(weight_map.values())
    for file_name in sorted(shard_names):
        path = os.path.join(model_path, file_name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[source inspect] Shard referenced by weight_map does not exist: {path}"
            )
    shard_names.update(path.name for path in Path(model_path).glob("*.safetensors"))
    headers = {}
    tensor_files = {}
    for file_name in sorted(shard_names):
        headers[file_name] = _read_safetensors_header(os.path.join(model_path, file_name))
        for key in headers[file_name]:
            if key in tensor_files:
                raise ValueError(
                    f"[source inspect] Duplicate tensor {key!r} in shards "
                    f"{tensor_files[key]!r} and {file_name!r}"
                )
            tensor_files[key] = file_name

    for key, file_name in weight_map.items():
        if "." not in key or _is_scale_key(key):
            continue
        info = headers[file_name].get(key)
        if info is None:
            problems.append(f"{key}: not found in shard {file_name}'s header")
            continue
        shape = info.get("shape", [])

        hits = {}
        for r in regions:
            patterns = match_patterns(key, r)
            if patterns:
                hits[r["name"]] = patterns
                for p in patterns:
                    raw[(r["name"], p)] += 1

        region = match_region(key, shape, weight_map, regions)
        src = effective_src(region) if region is not None else "bf16"
        dst = region.get("dst") if region is not None else None
        rname = region["name"] if region is not None else "(no region matched)"
        if region is not None:
            attributed[(rname, hits[rname][0])] += 1

        st = stats.setdefault(
            rname, {"src": src, "dst": dst, "count": 0, "dtypes": Counter(),
                    "quant": 0, "scale_ok": 0})
        st["count"] += 1
        dtype = info.get("dtype", "?")
        st["dtypes"][dtype] += 1

        if dtype not in _QUANT_DTYPES:
            continue
        st["quant"] += 1
        if src == "bf16":
            problems.append(
                f"{key}: quantized dtype {dtype} falls into src=bf16 region {rname!r}"
                f" (for mixed checkpoints, declare region src for this region)"
            )
            continue
        expect = _SRC_EXPECT_DTYPE.get(src)
        if expect is not None and dtype not in expect:
            problems.append(
                f"{key}: declares src={src} but received quantized dtype {dtype}"
                f" (missing or incorrect region src declaration?)"
            )
            continue
        scale_name = find_scale_name(key, weight_map, kind=SRC_KIND[src])
        if scale_name is None:
            problems.append(
                f"{key}: dtype {dtype} + src={src} but the scale name cannot be resolved"
                f" (source index is incomplete or scale naming is not covered by {src} rules)"
            )
        else:
            st["scale_ok"] += 1
            if src in ("mxfp4", "mxfp8"):
                scale_info = headers[weight_map[scale_name]].get(scale_name, {})
                scale_dtype = scale_info.get("dtype", "?")
                if scale_dtype not in ("I8", "U8", "F8_E8M0"):
                    problems.append(
                        f"{key}: src={src} requires e8m0 scale bytes, "
                        f"but {scale_name} has dtype {scale_dtype}"
                    )

    passthrough_names = {r["name"] for r in regions if is_passthrough(r)}

    for rname, st in stats.items():
        dt = ", ".join(f"{d}×{c}" for d, c in sorted(st["dtypes"].items()))
        dst_desc = st["dst"] or "passthrough"
        if rname in passthrough_names:
            dst_desc += " (CPU copy: normalized keys, no dequantization or requantization)"
        line = (f"[source inspect] region={rname} src={st['src']} "
                f"dst={dst_desc} "
                f"count={st['count']} dtype: {dt}")
        if st["src"] != "bf16" and st["quant"]:
            line += f" | scales resolved {st['scale_ok']}/{st['quant']}"
        logger.info(line)

    # Use raw hits to detect dead patterns without rejecting valid overlapping patterns.
    for region in regions:
        rname = region["name"]
        for p in region_patterns(region):
            if raw[(rname, p)] == 0:
                problems.append(
                    f"region {rname!r} pattern {p!r} matches no tensor names"
                    f" (region.match matches full tensor names, not module names; "
                    f"replace patterns like `*conv1d` with `*conv1d.*`; "
                    f"if this checkpoint has no such structure, remove the pattern from the recipe)"
                )
            elif attributed[(rname, p)] == 0:
                logger.info(
                    f"[source inspect] region={rname} pattern={p!r} "
                    f"has {raw[(rname, p)]} raw hits but 0 attributed hits"
                    f" (shadowed by earlier patterns / regions)"
                )

    if problems:
        shown = "\n  ".join(problems[:20])
        more = f"\n  …{len(problems)} problems total" if len(problems) > 20 else ""
        raise ValueError(
            f"[source inspect] Source format inspection failed (stopped before execution; see "
            f"source_inspect.py):\n  {shown}{more}"
        )
    return stats
