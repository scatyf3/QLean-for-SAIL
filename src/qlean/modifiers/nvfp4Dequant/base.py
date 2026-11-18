# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import json
from glob import glob
from tqdm import tqdm

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from qlean.modifiers import Modifier, ModifierFactory
from qlean.modifiers import QuantizationMixin
from qlean.core.event import Event, EventType
from qlean.modifiers.utils import copy_necessary_files, ensure_scales_in_weight_map
from qlean.modifiers.dequant_utils import _tensor_bytes, flush_shard, rename_shards, build_weight_map

logger = logging.getLogger(__name__)


FP4_LUT = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=torch.float32, device="cuda")

def weight_dequant_kernel(x, s, y, M, N, group_size):

    low_nibbles = x & 0xF
    high_nibbles = (x >> 4) & 0xF

    low_nibbles = low_nibbles.to(torch.int64)
    high_nibbles = high_nibbles.to(torch.int64)
    low_vals = FP4_LUT[low_nibbles]
    high_vals = FP4_LUT[high_nibbles]

    j_indices = torch.arange(N // 2, device=x.device)
    scale_n_idx = (2 * j_indices) // group_size

    scales = s[:, scale_n_idx]

    low_dequant = low_vals * scales
    high_dequant = high_vals * scales

    y[:, 0::2] = low_dequant
    y[:, 1::2] = high_dequant

def weight_dequant(x: torch.Tensor, s: torch.Tensor, group_size: int = 16) -> torch.Tensor:
    assert x.is_contiguous() and s.is_contiguous()
    assert x.dim() == 2 and s.dim() == 2
    assert x.size(0) == s.size(0)
    assert x.size(1)*2 == s.size(1) * group_size

    M = x.size(0)
    N = x.size(1)*2

    y = torch.empty(M, N, dtype=torch.get_default_dtype())
    weight_dequant_kernel(x, s, y, M, N, group_size)
    return y

@ModifierFactory.register("nvfp4DequantModifier")
class nvfp4DequantModifier(QuantizationMixin, Modifier):

    def on_initialize(self, state, *args, **kwargs):
        logger.info("nvfp4Dequant: on_initialize")
        self.initialized_ = True

        config_file, model_index_file = copy_necessary_files(
            state.model.path, state.save_path
        )

        # delete quantization_config
        config = json.load(open(config_file))
        config.pop("quantization_config", None)
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=True)
        logger.info(f"config.json modified and saved to {config_file}")
        with open(model_index_file, "r") as f:
            model_index = json.load(f)
        weight_map = model_index["weight_map"]

        safetensor_files = list(glob(os.path.join(state.model.path, "*.safetensors")))
        safetensor_files.sort()

        ensure_scales_in_weight_map(weight_map, state.model.path)

        # shard parameters
        max_shard_bytes = int(5.0 * 1024 ** 3)
        global_shard_idx = 0
        all_shard_weight_map: dict = {}
        all_tmp_paths: list = []

        # Cache cross-file safe_open handles so each file is opened only once
        cross_file_handles: dict = {}

        def get_scale_tensor(scale_name, current_file_name, current_handle):
            scale_file = weight_map[scale_name]
            if scale_file == current_file_name:
                return current_handle.get_tensor(scale_name).cuda()
            else:
                if scale_file not in cross_file_handles:
                    cross_file_handles[scale_file] = safe_open(
                        os.path.join(state.model.path, scale_file),
                        framework="pt", device="cpu"
                    )
                return cross_file_handles[scale_file].get_tensor(scale_name).cuda()

        # inline shard state
        current_shard: dict = {}
        current_shard_bytes: int = 0

        def do_flush_shard():
            nonlocal global_shard_idx, current_shard, current_shard_bytes
            if not current_shard:
                return
            flush_shard(current_shard, state.save_path, global_shard_idx, all_shard_weight_map, all_tmp_paths)
            global_shard_idx += 1
            del current_shard
            torch.cuda.empty_cache()
            current_shard = {}
            current_shard_bytes = 0

        _SKIP_SUFFIXES = (
            "input_scale", "weight_scale", "weight_scale_2",
            "q_scale", "k_scale", "v_scale"
        )

        weight_files = []
        for safetensor_file in tqdm(safetensor_files):
            file_name = os.path.basename(safetensor_file)
            with safe_open(safetensor_file, framework="pt", device="cpu") as f:
                all_keys = list(f.keys())
                weight_keys = [k for k in all_keys if not any(k.endswith(s) for s in _SKIP_SUFFIXES)]

                for weight_name in weight_keys:
                    weight = f.get_tensor(weight_name).cuda()
                    scale_name = f"{weight_name}_scale"

                    if scale_name in weight_map:
                        assert weight.element_size() == 1
                        # unpack and apply scale1
                        scale_local = get_scale_tensor(scale_name, file_name, f)
                        f32_weight = weight_dequant(weight, scale_local.float())
                        del weight, scale_local
                        # apply scale2
                        scale2_name = f"{scale_name}_2"
                        scale_global = get_scale_tensor(scale2_name, file_name, f).item()
                        bf16_weight = (f32_weight * scale_global).to(torch.bfloat16)
                        del f32_weight
                    else:
                        bf16_weight = weight
                        del weight

                    tensor_bytes = _tensor_bytes(bf16_weight)
                    if not current_shard and tensor_bytes > max_shard_bytes:
                        logger.warning(
                            f"Tensor {weight_name} ({tensor_bytes / 1024**3:.2f} GB) "
                            f"exceeds max_shard_size ({max_shard_bytes / 1024**3:.2f} GB), "
                            f"will be stored in an oversized shard."
                        )
                    if current_shard and (current_shard_bytes + tensor_bytes > max_shard_bytes):
                        do_flush_shard()

                    current_shard[weight_name] = bf16_weight
                    current_shard_bytes += tensor_bytes

        # flush last shard
        do_flush_shard()

        # close cross-file handles
        for handle in cross_file_handles.values():
            handle.__exit__(None, None, None)
        cross_file_handles.clear()

        weight_files = rename_shards(all_tmp_paths, state.save_path)
        new_weight_map = build_weight_map(
            all_shard_weight_map, all_tmp_paths, weight_files
        )

        # Update model index
        new_model_index_file = os.path.join(
            state.save_path, "model.safetensors.index.json"
        )
        with open(new_model_index_file, "w") as f:
            json.dump({"metadata": {}, "weight_map": new_weight_map}, f, indent=2)

        state.model.model_ = weight_files

    def on_start(self, state, *args, **kwargs):
        logger.info("nvfp4Dequant: on_start")
        super().on_start(state)

    def on_end(self, state):
        logger.info("nvfp4Dequant: on_end")
        super().on_end(state)

    def on_finalize(self, state):
        logger.info("nvfp4Dequant: on_finalize")
        super().on_finalize(state)

    def on_event(self, state, event: Event, **kwargs):
        logger.info("nvfp4Dequant: on_event")
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state)

        if event.type_ == EventType.CALIBRATION_EPOCH_END:
            if not self.ended_:
                self.on_end(state)
