# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Callable, Optional

from pydantic import Field

from qlean.core.event import Event, EventType
from qlean.modifiers_engine._runtime.engine import ShardEngine, ShardRunSpec
from qlean.modifiers_engine._runtime.source_inspect import _read_safetensors_header, read_source_index

logger = logging.getLogger(__name__)


def load_source_metadata(model_path: str, *, strip_quant_config: bool = True) -> tuple:
    source = Path(model_path)
    with (source / "config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)
    if strip_quant_config:
        config.pop("quantization_config", None)
        if "text_config" in config:
            config["text_config"].pop("quantization_config", None)

    index = source / "model.safetensors.index.json"
    if index.is_file():
        weight_map = read_source_index(index)
    else:
        weight_map = {}
        shards = sorted(source.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"No .safetensors files found in {model_path}")
        for shard in shards:
            for name in _read_safetensors_header(str(shard)):
                if name in weight_map:
                    raise ValueError(
                        f"Duplicate tensor {name!r} in shards {weight_map[name]!r} and {shard.name!r}"
                    )
                weight_map[name] = shard.name

    return config, weight_map


def copy_model_assets(model_path: str, save_path: str) -> None:
    excluded = {
        "README.md", "LICENSE", "safetensors-md5sum.txt", "hf_quant_config.json",
    }
    root_excluded = {
        "config.json", "model.safetensors.index.json",
        ".qlean_engine_fingerprint.json", "qlean_config.json", "qlean_verify",
    }
    source = Path(model_path)
    target = Path(save_path)
    for root, dirs, filenames in os.walk(source):
        root_path = Path(root)
        excluded_names = excluded | root_excluded if root_path == source else excluded
        dirs[:] = [name for name in dirs if name not in excluded_names]
        target_dir = target / root_path.relative_to(source)
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            if filename.endswith(".safetensors") or filename in excluded_names:
                continue
            shutil.copy2(root_path / filename, target_dir / filename)
    logger.info(f"non-weight files copied to {save_path}")


def _default_finalize(save_path, stats, transform_cfg):
    int_stats = {k: v for k, v in stats.items() if isinstance(v, int)}
    logger.info(f"Engine completed: {int_stats}")


def run_engine(
    model_path: str,
    save_path: str,
    source_files: list,
    weight_map: dict,
    *,
    process_tensor: Callable,
    expected_output_keys: Callable,
    transform_cfg: dict,
    finalize: Optional[Callable] = None,
    filter_source_key: Optional[Callable] = None,
    dequant_map: Optional[dict] = None,
    stream_map: Optional[dict] = None,
    lazy_input: Optional[Callable] = None,
    num_gpus: int = 0,
    resume: bool = True,
    force_rerun: bool = False,
    verify_dir: Optional[str] = None,
) -> dict:
    """Maps use first-match, case-sensitive full-name globs; '*' spans dots, 're:' is unsupported."""
    spec = ShardRunSpec(
        model_path=model_path,
        save_path=save_path,
        source_files=source_files,
        weight_map=weight_map,
        process_tensor=process_tensor,
        expected_output_keys=expected_output_keys,
        finalize=finalize if finalize is not None else _default_finalize,
        transform_cfg=transform_cfg,
        num_gpus=num_gpus,
        dequant_map=dequant_map,
        stream_map=stream_map,
        lazy_input=lazy_input,
        resume=resume,
        force_rerun=force_rerun,
        filter_source_key=filter_source_key,
        verify_dir=verify_dir,
    )
    return ShardEngine().run(spec)


class EngineModifierMixin:
    """Place before Modifier in the MRO to forward lifecycle events."""

    num_gpus: int = Field(default=0, description="GPUs to use; 0 = all visible")
    resume: bool = Field(default=True, description="Resume from completed shards")
    force_rerun: bool = Field(default=False, description="Rerun without prompting")

    def on_event(self, state, event: Event, **kwargs):
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state)
        if event.type_ == EventType.CALIBRATION_EPOCH_END:
            if not self.ended_:
                self.on_end(state)
