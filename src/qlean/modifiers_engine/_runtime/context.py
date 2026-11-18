# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import logging
import torch
from safetensors import safe_open

logger = logging.getLogger(__name__)


class LazyWeight:
    """Metadata-only streaming input; kept here to avoid a plan/engine import cycle."""

    __slots__ = ("name", "shape", "dtype", "chunk_rows")

    def __init__(self, name: str, shape: list, dtype, chunk_rows: int):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.chunk_rows = chunk_rows

    def dim(self) -> int:
        return len(self.shape)

    def __repr__(self) -> str:
        return (f"LazyWeight(name={self.name!r}, shape={self.shape}, "
                f"dtype={self.dtype}, chunk_rows={self.chunk_rows})")


class ShardContext:
    def __init__(self, device: str, model_path: str, weight_map: dict,
                 transform_cfg: dict):
        self.device = device
        self.model_path = model_path
        self.weight_map = weight_map
        self.transform_cfg = transform_cfg
        self._cross_file_handles: dict = {}
        self._cur_file: str = None
        self._cur_handle = None
        self.stats: dict = {}

    def begin_shard(self, file_name: str, handle) -> None:
        self._cur_file = file_name
        self._cur_handle = handle

    def bump(self, key: str, n: int = 1) -> None:
        self.stats[key] = self.stats.get(key, 0) + n

    def append(self, key: str, value) -> None:
        """Values must be JSON-serializable for shard metadata and resume."""
        self.stats.setdefault(key, []).append(value)

    def take_stats(self) -> dict:
        stats = self.stats
        self.stats = {}
        return stats

    def _handle_for(self, name: str):
        target_file = self.weight_map.get(name)
        if target_file is None:
            raise KeyError(
                f"[ShardContext] '{name}' is not in weight_map and cannot be read. "
                f"Check the source index.json or whether ensure_scales_in_weight_map has run."
            )
        if target_file == self._cur_file:
            return self._cur_handle
        if target_file not in self._cross_file_handles:
            self._cross_file_handles[target_file] = safe_open(
                os.path.join(self.model_path, target_file),
                framework="pt", device="cpu",
            )
        return self._cross_file_handles[target_file]

    def get_tensor_cpu(self, name: str) -> torch.Tensor:
        return self._handle_for(name).get_tensor(name)

    def get_tensor(self, name: str) -> torch.Tensor:
        tensor = self.get_tensor_cpu(name)
        if self.device != "cpu" and tensor.device.type == "cpu":
            tensor = tensor.to(self.device)
        return tensor

    def get_slice(self, name: str, a: int, b: int) -> torch.Tensor:
        """Read rows [a:b) along M without materializing the full tensor."""
        tensor = self._handle_for(name).get_slice(name)[a:b]
        if self.device != "cpu" and tensor.device.type == "cpu":
            tensor = tensor.to(self.device)
        return tensor

    def get_shape(self, name: str) -> list:
        return list(self._handle_for(name).get_slice(name).get_shape())

    def close(self) -> None:
        for handle in self._cross_file_handles.values():
            try:
                handle.__exit__(None, None, None)
            except Exception:
                pass
        self._cross_file_handles.clear()
        self._cur_file = None
        self._cur_handle = None
