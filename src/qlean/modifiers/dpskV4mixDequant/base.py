# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import json
import logging
from tqdm import tqdm
from glob import glob
from safetensors import safe_open
from safetensors.torch import load_file, save_file

import torch
import triton
import triton.language as tl
from triton import Config

from qlean.modifiers import Modifier, ModifierFactory
from qlean.modifiers import QuantizationMixin
from qlean.core.event import Event, EventType
from qlean.modifiers.utils import copy_necessary_files, ensure_scales_in_weight_map, weight_quant
from qlean.modifiers.dequant_utils import _tensor_bytes, flush_shard, rename_shards, build_weight_map
from qlean.utils.upcast_from_mxfp import upcast_from_mxfp

logger = logging.getLogger(__name__)
@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)

def weight_dequant(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    assert x.is_contiguous() and s.is_contiguous()
    assert x.dim() == s.dim()
    assert x.dim() >= 2
    M, N = x.shape[-2:]
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE"]),
        triton.cdiv(N, meta["BLOCK_SIZE"]),
    )
    if x.dim() == 2:
        y = torch.empty_like(x, dtype=torch.get_default_dtype())
        weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    elif x.dim() == 3:
        stack_y = []
        for idx in range(x.shape[0]):
            weight = x[idx]
            scale = s[idx]
            slice_y = torch.empty(*x.shape[-2:], dtype=torch.get_default_dtype(), device=x.device)
            weight_dequant_kernel[grid](weight, scale, slice_y, M, N, BLOCK_SIZE=block_size)
            stack_y.append(slice_y)
        y = torch.stack(stack_y, dim=0)
    else:
        raise ValueError(f"Unsupported tensor dimension: {x.dim()}")
    return y

@ModifierFactory.register("dpskV4mixDequantModifier")
class dpskV4mixDequantModifier(QuantizationMixin, Modifier):

    def on_initialize(self, state, *args, **kwargs):
        logger.info("dpskV4mixDequant: on_initialize")
        self.initialized_ = True

        torch.set_default_dtype(torch.bfloat16)
        config_file, model_index_file = copy_necessary_files(
            state.model.path, state.save_path
        )
        config_file = os.path.join(state.save_path, "config.json")
        model_index_file = os.path.join(state.save_path, "model.safetensors.index.json")

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

        def _write_to_shard(name: str, tensor: torch.Tensor):
            nonlocal current_shard_bytes
            t_bytes = _tensor_bytes(tensor)
            if not current_shard and t_bytes > max_shard_bytes:
                print(
                    f"Warning: tensor {name} ({t_bytes / 1024**3:.2f} GB) "
                    f"exceeds max_shard_size, will be stored in an oversized shard."
                )
            if current_shard and (current_shard_bytes + t_bytes > max_shard_bytes):
                do_flush_shard()
            current_shard[name] = tensor
            current_shard_bytes += t_bytes

        weight_files = []
        fp8_count =0
        for safetensor_file in tqdm(safetensor_files):
            file_name = os.path.basename(safetensor_file)
            with safe_open(safetensor_file, framework="pt", device="cpu") as f:
                all_keys = list(f.keys())
                weight_keys = [k for k in all_keys if not k.endswith(".scale")]

                for weight_name in weight_keys:
                    weight = f.get_tensor(weight_name).cuda()

                    if weight.element_size() == 1 and weight.dtype == torch.float8_e4m3fn:  # FP8 weight
                        scale_inv_name = weight_name.removesuffix(".weight") + ".scale"

                        if scale_inv_name not in weight_map:
                            print(f"Warning: Missing scale for {weight_name}, skipping conversion")
                            _write_to_shard(weight_name, weight)
                            del weight
                        else:
                            fp8_count += 1
                            module_name, param_name = weight_name.rsplit(".", 1)
                            logger.debug(f"Fp8 dequant {weight_name} with {scale_inv_name}")
                            scale_inv = get_scale_tensor(scale_inv_name, file_name, f)
                            bf16_weight = weight_dequant(weight.float(), scale_inv.float())
                            del weight, scale_inv
                            _write_to_shard(weight_name, bf16_weight)
                            del bf16_weight
                    else:

                        if "ffn.experts" in weight_name and weight.dtype == torch.int8:
                            scale_inv_name = weight_name.removesuffix(".weight") + ".scale"
                            if scale_inv_name in weight_map:
                                if "ffn.experts.0.w1.weight" in weight_name:
                                    logger.debug(f"Fp4 dequant {weight_name} with {scale_inv_name}")
                                scale_inv = get_scale_tensor(scale_inv_name, file_name, f).view(torch.uint8)
                                weight = weight.view(torch.uint8)
                                bf16_weight = upcast_from_mxfp(weight, scale_inv, torch.bfloat16, axis=1)
                                del weight, scale_inv
                                _write_to_shard(weight_name, bf16_weight)
                                del bf16_weight
                            else:
                                logger.warning(f"{weight_name} has no scale_inv_name {scale_inv_name}")

                                weight = weight.view(torch.uint8)
                                _write_to_shard(weight_name, weight)
                                del weight
                        else:
                            _write_to_shard(weight_name, weight)
                            del weight

        #  flush last shard
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

        logger.info(f"Dequant and quant {fp8_count} layers")

        state.model.model_ = weight_files

    def on_start(self, state, *args, **kwargs):
        logger.info("dpskV4mixDequant: on_start")
        super().on_start(state)

    def on_end(self, state):
        logger.info("dpskV4mixDequant: on_end")
        super().on_end(state)

    def on_finalize(self, state):
        logger.info("dpskV4mixDequant: on_finalize")
        super().on_finalize(state)

    def on_event(self, state, event: Event, **kwargs):
        logger.info("dpskV4mixDequant: on_event")
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state)

        if event.type_ == EventType.CALIBRATION_EPOCH_END:
            if not self.ended_:
                self.on_end(state)
