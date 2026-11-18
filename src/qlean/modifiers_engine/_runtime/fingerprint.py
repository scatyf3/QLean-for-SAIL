# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import json
import hashlib
import logging

logger = logging.getLogger(__name__)

FINGERPRINT_FILENAME = ".qlean_engine_fingerprint.json"


def build_fingerprint(model_path: str, source_files, transform_cfg: dict,
                      dequant_id, num_gpus: int) -> dict:
    source_meta = []
    extra = ["config.json", "model.safetensors.index.json"]
    for name in extra:
        path = os.path.join(model_path, name)
        if os.path.exists(path):
            source_meta.append(_file_meta(name, path))
    for file_name, resolved_path in source_files:
        source_meta.append(_file_meta(file_name, resolved_path))
    source_meta.sort(key=lambda m: m["name"])

    # Detect weight_map edits even when the index file size is unchanged.
    index_hash = None
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "rb") as f:
            index_hash = hashlib.sha256(f.read()).hexdigest()

    # GPU count does not affect same-name shard contents or resume compatibility.
    payload = {
        "source": source_meta,
        "transform_cfg": _normalize_cfg(transform_cfg),
        "dequant_map": dequant_id,
        "index_hash": index_hash,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return {
        "algorithm": "sha256:shard-engine-v1",
        "hash": hashlib.sha256(blob.encode("utf-8")).hexdigest(),
        "num_gpus": num_gpus,
        "file_count": len(source_meta),
    }


def _normalize_cfg(cfg):
    """Only string lists have set semantics; numeric layouts must retain order."""
    if isinstance(cfg, dict):
        return {k: _normalize_cfg(v) for k, v in cfg.items()}
    if isinstance(cfg, list):
        if all(isinstance(item, str) for item in cfg):
            return sorted(cfg)
        return [_normalize_cfg(item) for item in cfg]
    if isinstance(cfg, set):
        return sorted(_normalize_cfg(item) for item in cfg)
    return cfg


def _file_meta(name: str, path: str) -> dict:
    # NFS mtime jitter must not invalidate resume for unchanged sources.
    real = os.path.realpath(path)
    return {"name": name, "size": os.stat(real).st_size}


def load_fingerprint(save_path: str):
    """Distinguish missing and corrupt records so old output is not silently cleared."""
    path = os.path.join(save_path, FINGERPRINT_FILENAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning(f"Fingerprint file is corrupt ({exc}); treating existing output as having unknown configuration")
        return {"corrupt": True, "error": str(exc)}


def save_fingerprint(save_path: str, fp: dict) -> None:
    path = os.path.join(save_path, FINGERPRINT_FILENAME)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(fp, f, indent=2, ensure_ascii=False, sort_keys=True)
    os.replace(tmp, path)
