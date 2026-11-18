# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import json
import torch
from safetensors.torch import save_file


def _tensor_bytes(t: torch.Tensor) -> int:
    """Return the size of the tensor in bytes."""
    return t.numel() * t.element_size()


def flush_shard(
    current_shard: dict,
    output_dir: str,
    shard_idx: int,
    shard_weight_map: dict,
    tmp_shard_paths: list,
) -> None:
    """Save current_shard to a temp shard file, recording its path and weight_map entries."""
    tmp_name = f"tmp_shard_{shard_idx:05d}.safetensors"
    tmp_path = os.path.join(output_dir, tmp_name)
    tmp_current_shard = {k: (v.cpu()).contiguous() for k, v in current_shard.items()}
    save_file(tmp_current_shard, tmp_path)
    tmp_shard_paths.append(tmp_path)
    for wname in current_shard:
        shard_weight_map[wname] = shard_idx


def rename_shards(all_tmp_paths: list, bf16_path: str) -> list[tuple[str, str]]:
    """Rename temporary shards to model-XXXXX-of-YYYYY.safetensors.

    Return a list of (file name, absolute or relative output path) pairs.
    """
    total_shards = len(all_tmp_paths)
    weight_files: list[tuple[str, str]] = []
    for i, tmp_path in enumerate(all_tmp_paths):
        final_name = f"model-{i+1:05d}-of-{total_shards:05d}.safetensors"
        final_path = os.path.join(bf16_path, final_name)
        os.rename(tmp_path, final_path)
        weight_files.append((final_name, final_path))
    return weight_files


def build_weight_map(
    all_shard_weight_map: dict,
    all_tmp_paths: list,
    weight_files: list[tuple[str, str]],
) -> dict:
    """Build the final weight_map from shard_weight_map and weight_files."""
    final_names = [name for name, path in weight_files]
    shard_idx_to_final: dict = {}
    for i, tmp_path in enumerate(all_tmp_paths):
        tmp_basename = os.path.basename(tmp_path)
        shard_num = int(tmp_basename.replace("tmp_shard_", "").replace(".safetensors", ""))
        shard_idx_to_final[shard_num] = final_names[i]

    new_weight_map: dict = {}
    for wname, shard_num in all_shard_weight_map.items():
        new_weight_map[wname] = shard_idx_to_final[shard_num]
    return new_weight_map