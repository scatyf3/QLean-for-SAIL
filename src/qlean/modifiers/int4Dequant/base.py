# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import json
import logging
from tqdm import tqdm
from glob import glob
from safetensors.torch import load_file, save_file

import torch
import triton
import triton.language as tl
from triton import Config

from qlean.modifiers import Modifier, ModifierFactory
from qlean.modifiers import QuantizationMixin
from qlean.core.event import Event, EventType
from qlean.modifiers.utils import copy_necessary_files, load_file_to_cuda, safe_save_file

logger = logging.getLogger(__name__)

INT4_LUT = torch.tensor([
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
    -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0
], dtype=torch.float32, device="cuda")

shifts = torch.tensor([
    0, 4, 8, 12, 16, 20, 24, 28
], dtype=torch.int32, device="cuda")

def weight_dequant_kernel(x, s, y, M, N, group_size):

    if s.dtype != torch.float32:
        s = s.to(torch.float32)

    # unpack
    indices = (x.unsqueeze(-1) >> shifts) & 0xF
    offset = 1 << ( 4 - 1)
    values = (indices - offset).to(torch.int8) #[M, N/8, 8]
    x_unpacked = values.view(M, N)

    # dequant
    x_unpacked = x_unpacked.view(M, -1, group_size)
    s = s.view(M, -1, 1)
    result = (x_unpacked * s).view(M, -1).to(torch.bfloat16)
    y.copy_(result)

def weight_dequant(
    x: torch.Tensor, s: torch.Tensor, group_size: int = 32
) -> torch.Tensor:
    assert x.is_contiguous() and s.is_contiguous()
    assert x.dim() == 2 and s.dim() == 2
    assert x.size(0) == s.size(0)
    assert x.size(1)*8 == s.size(1)*32

    M = x.size(0)
    N = x.size(1)*8

    y = torch.empty(M, N, dtype=torch.get_default_dtype())
    weight_dequant_kernel(x, s, y, M, N, group_size)
    return y


@ModifierFactory.register("int4DequantModifier")
class int4DequantModifier(QuantizationMixin, Modifier):

    def on_initialize(self, state, *args, **kwargs):
        logger.info("dequant: on_initialize")
        self.initialized_ = True

        torch.set_default_dtype(torch.bfloat16)
        config_file, model_index_file = copy_necessary_files(
            state.model.path, state.save_path
        )

        # delete quantization_config
        config = json.load(open(config_file))
        config.pop("quantization_config", None)
        if "text_config" in config:
            config["text_config"].pop("quantization_config", None)
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=True)
        logger.info(f"config.json modified and saved to {config_file}")
        with open(model_index_file, "r") as f:
            model_index = json.load(f)
        weight_map = model_index["weight_map"]

        # Cache for loaded safetensor files
        loaded_files = {}
        int4_weight_names = []

        # Helper function to get tensor from the correct file
        def get_tensor(tensor_name):
            file_name = weight_map[tensor_name]
            if file_name not in loaded_files:
                file_path = os.path.join(state.model.path, file_name)
                loaded_files[file_name] = load_file_to_cuda(file_path)
            return loaded_files[file_name][tensor_name]

        safetensor_files = list(glob(os.path.join(state.model.path, "*.safetensors")))
        safetensor_files.sort()
        weight_files = []
        for safetensor_file in tqdm(safetensor_files):
            file_name = os.path.basename(safetensor_file)
            save_file_path = os.path.join(state.save_path, file_name)
            weight_files.append((file_name, save_file_path))
            current_state_dict = load_file_to_cuda(safetensor_file)
            loaded_files[file_name] = current_state_dict

            new_state_dict = {}
            for weight_name, weight in current_state_dict.items():
                if weight_name.endswith("weight_shape") or weight_name.endswith("weight_scale"):
                    continue
                elif weight_name.endswith("weight_packed"):
                    scale_name = weight_name.replace("weight_packed", "weight_scale")
                    try:
                        # Get scale from the correct file
                        scale = get_tensor(scale_name)
                        int4_weight_names.append(weight_name)
                        new_weight_name = weight_name.replace("weight_packed", "weight")
                        new_state_dict[new_weight_name] = weight_dequant(
                            weight, scale
                        )
                    except KeyError:
                        print(
                            f"Warning: Missing scale tensor for {weight_name}, skipping conversion"
                        )
                        new_state_dict[weight_name] = weight
                else:
                    new_state_dict[weight_name] = weight

            new_safetensor_file = os.path.join(state.save_path, file_name)
            safe_save_file(new_state_dict, new_safetensor_file)

            # Memory management: keep only the 2 most recently used files
            if len(loaded_files) > 2:
                oldest_file = next(iter(loaded_files))
                del loaded_files[oldest_file]
                torch.cuda.empty_cache()

        # Update model index
        new_model_index_file = os.path.join(
            state.save_path, "model.safetensors.index.json"
        )
        for weight_name in int4_weight_names:
            scale_name = weight_name.replace("weight_packed", "weight_scale")
            shape_name = weight_name.replace("weight_packed", "weight_shape")
            new_weight_name = weight_name.replace("weight_packed", "weight")
            if scale_name in weight_map:
                weight_map.pop(scale_name)
            if shape_name in weight_map:
                weight_map.pop(shape_name)
            if weight_name in weight_map:
                value = weight_map.pop(weight_name)
                weight_map[new_weight_name] = value
        with open(new_model_index_file, "w") as f:
            json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)

        #state.model.path = state.save_path
        state.model.model_ = weight_files

    def on_start(self, state, *args, **kwargs):
        logger.info("dequant: on_start")
        super().on_start(state)

    def on_end(self, state):
        logger.info("dequant: on_end")
        super().on_end(state)

    def on_finalize(self, state):
        logger.info("dequant: on_finalize")
        super().on_finalize(state)

    def on_event(self, state, event: Event, **kwargs):
        logger.info("dequant: on_event")
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state)

        if event.type_ == EventType.CALIBRATION_EPOCH_END:
            if not self.ended_:
                self.on_end(state)
