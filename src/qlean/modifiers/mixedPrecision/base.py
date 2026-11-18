# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import List
from types import SimpleNamespace
import os
import re
import json
import shutil
import logging
from tqdm import tqdm
from collections import defaultdict
from safetensors import safe_open

from qlean.modifiers.utils import safe_save_file

from qlean.modifiers import Modifier, ModifierFactory
from qlean.modifiers import QuantizationMixin
from qlean.core.event import Event, EventType

logger = logging.getLogger(__name__)


def parse_layers(layers_input):
    """
    Parse layer input, supports single numbers, range format (e.g., 92-93), and multiple range combinations (e.g., ['89-90', '91-92'])
    """
    layers = []
    for item in layers_input:
        if '-' in item:
            try:
                start, end = map(int, item.split('-'))
                if start > end:
                    raise ValueError(f"Invalid range: {item} (start > end)")
                layers.extend(range(start, end + 1))
            except ValueError as e:
                if "invalid literal" in str(e):
                    raise ValueError(f"Invalid range format: {item}")
                raise
        else:
            try:
                layers.append(int(item))
            except ValueError:
                raise ValueError(f"Invalid layer number: {item}")
    return sorted(list(set(layers)))

def files_are_identical_fast(file1, file2):
    """Compare file sizes only; equal sizes do not guarantee equal contents."""
    try:
        stat1 = os.stat(file1)
        stat2 = os.stat(file2)

        # Compare file size
        if stat1.st_size != stat2.st_size:
            return False

        return True
    except (OSError, FileNotFoundError):
        return False

def get_layer_num(key_name):
    """
    Extract X from model.layers.X. in the key name using regex.
    Return -1 if no match found or out of range.
    """
    m = re.search(r"model.layers.(\d+).", key_name)
    if m:
        layer_idx = int(m.group(1))
        return layer_idx
    return -1

def count_safetensor_files(path):
    """
    Count the number of safetensor files in the specified path

    Args:
        path: Folder path

    Returns:
        int: Number of safetensor files
    """
    if not os.path.exists(path):
        logger.error(f"Warning: Path {path} does not exist")
        return 0

    # Count files ending with .safetensors
    safetensor_files = [f for f in os.listdir(path) if f.endswith('.safetensors')]
    logger.info(f"Found {len(safetensor_files)} safetensor files in {path}")
    return len(safetensor_files)

def shard_weights(input_path, output_dir, prefix, total_files, num_layers, desc="Processing"):
    """
    Shard weight files

    Args:
        input_path: Input file path template
        output_dir: Output directory
        prefix: Output file prefix
        total_files: Total number of files
        num_layers: Number of layers
        desc: Progress bar description
    """
    shards = {i: {} for i in range(num_layers)} # 0~(num_layers-1)
    all_weights = {}

    for i in tqdm(range(0, total_files), desc=f"{desc} - Loading files"):
        safetensor_path = input_path.format(i+1, total_files)
        # If file doesn't exist, skip it
        if not os.path.isfile(safetensor_path):
            logger.info(f"Warning: {safetensor_path} not found, skipping...")
            continue

        with safe_open(safetensor_path, framework="pt") as f:
            weights = {k: f.get_tensor(k) for k in f.keys()}

        # Accumulate to all_weights
        all_weights = {**all_weights, **weights}

    for k in tqdm(all_weights.keys(), desc=f"{desc} - Sharding"):
        tensor = all_weights[k]
        layer_idx = get_layer_num(k)
        if 0 <= layer_idx < num_layers:
            # Belongs to layers 0~(num_layers-1)
            shards[layer_idx][k] = tensor
        else:
            # Not in [0, num_layers-1] range, put in first layer shards[0]
            shards[0][k] = tensor

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for i in tqdm(range(num_layers), desc=f"{desc} - Saving"):
        out_path = f"{output_dir}/{prefix}-{i:05d}-of-{num_layers-1:05d}.safetensors"
        safe_save_file(shards[i], out_path)

    # Output statistics
    for i in range(num_layers):
        out_path = f"{output_dir}/{prefix}-{i:05d}-of-{num_layers-1:05d}.safetensors"
        logger.info(f"Layer-{i} => {len(shards[i])} weights => {out_path}")

def copy_other_files(w8a8_path, target_dir):
    """
    Copy all files from w8a8 path to target directory except safetensors files and model.safetensors.index.json

    Args:
        w8a8_path: INT8 model path
        target_dir: Target directory path
    """
    if not os.path.exists(w8a8_path):
        logger.error(f"Warning: INT8 path {w8a8_path} does not exist")
        return

    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Get all files in int8 path
    all_files = []
    for root, dirs, files in os.walk(w8a8_path):
        for file in files:
            # Calculate relative path
            rel_path = os.path.relpath(os.path.join(root, file), w8a8_path)
            all_files.append(rel_path)

    # Filter out files that don't need to be copied
    files_to_copy = []
    for file_path in all_files:
        filename = os.path.basename(file_path)
        # Exclude safetensors files and model.safetensors.index.json
        if not filename.endswith('.safetensors') and filename != 'model.safetensors.index.json':
            files_to_copy.append(file_path)

    logger.info(f"Found {len(files_to_copy)} other files to copy...")

    # Copy files
    for rel_path in files_to_copy:
        src_file = os.path.join(w8a8_path, rel_path)
        dst_file = os.path.join(target_dir, rel_path)

        # Ensure target file directory exists
        dst_dir = os.path.dirname(dst_file)
        os.makedirs(dst_dir, exist_ok=True)

        # Check if file already exists and is identical
        if os.path.exists(dst_file):
            if files_are_identical_fast(src_file, dst_file):
                logger.info(f"File already exists and is identical, skipping: {rel_path}")
                continue
            else:
                logger.info(f"File already exists but is different, will overwrite: {rel_path}")

        # Copy file
        try:
            shutil.copy2(src_file, dst_file)
            logger.info(f"Copied: {rel_path}")
        except Exception as e:
            logger.info(f"Failed to copy file {rel_path}: {e}")

    logger.info(f"Other files copying completed! Copied {len(files_to_copy)} files in total")

def update_config(mix_layers, target_dir):
    """Update configuration file"""
    config_file = os.path.join(target_dir, "config.json")

    # Check if config file exists
    if not os.path.exists(config_file):
        logger.error(f"Warning: Config file {config_file} does not exist, please ensure other files have been copied")
        return

    # Read and update config.json
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Add or update mix_layer field in quantization_config
    if 'quantization_config' not in config:
        config['quantization_config'] = {}
    config['quantization_config']['mix_layer'] = mix_layers

    # Write back to config.json
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"Config file updated: {config_file}")

def move_safetensors(mix_layers, num_layers, target_dir):
    """Move safetensor files"""
    bf16_all = set()

    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)

    for i in range(num_layers):  # Use 0~(num_layers-1) range
        # Original bf16 directory
        bf16_path = f"./bf16weights/model-{i:05d}-of-{num_layers-1:05d}.safetensors"
        w8a8_path = f"./int8weights/model-{i:05d}-of-{num_layers-1:05d}.safetensors"
        target_path = os.path.join(target_dir, f"model-{i:05d}-of-{num_layers-1:05d}.safetensors")

        # Determine which source file to copy
        source_path = bf16_path if i in mix_layers else w8a8_path

        # Quick check if target file already exists and has same content, skip if identical
        if os.path.exists(target_path):
            if files_are_identical_fast(source_path, target_path):
                logger.info(f"Target file already exists with same content, skipping layer {i}")
                continue
            else:
                logger.info(f"Target file exists but content differs, will overwrite layer {i}")

        # If it's a layer to skip (use bf16), copy bf16 weights
        if i in mix_layers:
            logger.info(f"Copying bf16 weights for layer {i}")
            shutil.copy2(bf16_path, target_path)
            with safe_open(bf16_path, framework="pt") as f:
                layer_keys = set(f.keys())
            bf16_all |= layer_keys
        else:
            # Otherwise copy w8a8 weights
            logger.info(f"Copying w8a8 weights for layer {i}")
            shutil.copy2(w8a8_path, target_path)

    bf16_all = sorted(bf16_all)
    return bf16_all

def find_layer_count(model_path: str, layer_ids: set[int]):
    model_path_json = os.path.join(model_path, "model.safetensors.index.json")

    shard_counts = defaultdict(int)
    with open(model_path_json, "r", encoding="utf-8") as f:
        data = json.load(f)
        weight_map = data.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError("index file have no `weight_map` ")
        for layer_name, shard in weight_map.items():
            layerid = get_layer_num(layer_name)
            if layerid in layer_ids:
                shard_counts[shard] += 1

    return shard_counts

def copy_w8a8_files(target_path: str, w8a8_path: str, w8a8_layer_dict):
    if not os.path.exists(w8a8_path):
        raise Exception(f"Warning: INT8 path {w8a8_path} does not exist")

    os.makedirs(target_path, exist_ok=True)

    #get all replace file names
    replace_files = set(w8a8_layer_dict.keys())

    #copy file to target path
    for root, dirs, files in os.walk(w8a8_path):
        for file in files:
            # Calculate relative path
            rel_path = os.path.relpath(os.path.join(root, file), w8a8_path)

            filename = os.path.basename(rel_path)
            if filename not in replace_files:
                #copy no need change files
                src_file = os.path.join(w8a8_path, rel_path)
                dst_file = os.path.join(target_path, rel_path)

                # Check if file already exists and is identical
                if os.path.exists(dst_file):
                    if files_are_identical_fast(src_file, dst_file):
                        logger.info(f"File already exists and is identical, skipping: {rel_path}")
                        continue
                    else:
                        logger.info(f"File already exists but is different, will overwrite: {rel_path}")

                # Copy file
                try:
                    shutil.copy2(src_file, dst_file)
                    logger.info(f"Copied: {rel_path}")
                except Exception as e:
                    raise Exception(f"Failed to copy file {rel_path}: {e}")

def replace_bf16_layers(target_path: str, w8a8_path: str, bf16_path: str, w8a8_shard_dict, bf16_shard_dict, mix_layers):
    # calculate w8a8 shard allocate rate
    w8a8_total_cnt = sum(w8a8_shard_dict.values())
    w8a8_shard_rate = {}
    if w8a8_total_cnt == 0:
        raise ValueError("no layers to replace")
    for shard, cnt in w8a8_shard_dict.items():
        w8a8_shard_rate[shard] = cnt / w8a8_total_cnt

    # get all replace src BF16 layers
    bf16_tensors = []
    for shard in bf16_shard_dict.keys():
        shard_path = os.path.join(bf16_path, shard)
        if not os.path.isfile(shard_path):
            raise Exception(f"safetensors file not exist: {shard_path}")

        with safe_open(shard_path, framework="torch") as f:
            for layer_name in f.keys():
                layer_id = get_layer_num(layer_name)
                if layer_id in mix_layers:
                    weight = f.get_tensor(layer_name)
                    bf16_tensors.append((layer_name, weight))

    bf16_layer_cnt = len(bf16_tensors)
    logger.info(f"{bf16_layer_cnt} bf16 layers will be added, {w8a8_total_cnt} w8a8 layers will be removed.")

    replace_index = 0
    save_shard_cnt = 0
    bf16_layer_index = {}
    for shard, rate in w8a8_shard_rate.items():
        shard_path = os.path.join(w8a8_path, shard)
        if not os.path.isfile(shard_path):
            raise Exception(f"safetensors file not exist: {shard_path}")

        save_shard_cnt += 1
        new_state_dict = {}
        # add no need change layers
        with safe_open(shard_path, framework="torch") as f:
            for layer_name in f.keys():
                layer_id = get_layer_num(layer_name)
                if layer_id in mix_layers:
                    continue
                new_state_dict[layer_name] = f.get_tensor(layer_name)

        # add BF16 layers
        to_be_replace = int(rate * bf16_layer_cnt)
        for i in tqdm(range(to_be_replace)):
            if replace_index >= bf16_layer_cnt:
                logger.info(f"all BF16 layers({replace_index}) already been replaced!")
                break
            layer_name, weight = bf16_tensors[replace_index]
            new_state_dict[layer_name] = weight
            bf16_layer_index[layer_name] = shard
            replace_index += 1

        if save_shard_cnt == len(w8a8_shard_rate) and replace_index < bf16_layer_cnt:
            logger.info(f"still have {bf16_layer_cnt - replace_index} layers need to save in the last safetensors file.")
            for i in tqdm(range(bf16_layer_cnt - replace_index)):
                layer_name, weight = bf16_tensors[replace_index]
                new_state_dict[layer_name] = weight
                bf16_layer_index[layer_name] = shard
                replace_index += 1

        #save to target
        os.makedirs(target_path, exist_ok=True)
        target_shard_path = os.path.join(target_path, shard)
        safe_save_file(new_state_dict, target_shard_path)

    logger.info(f"all {bf16_layer_cnt} BF16 layers has been saved in {save_shard_cnt} safetensors files")
    return bf16_layer_index

def update_model_index(model_path: str, layer_ids, add_layers):
    model_index_json = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(model_index_json):
        raise Exception(f"Error: Config file {model_index_json} does not exist!")

    with open(model_index_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "weight_map" not in data:
        raise ValueError("index.json has no weight_map content")
    weight_map = data["weight_map"]

    for key in list(weight_map.keys()):
        layerid = get_layer_num(key)
        if layerid in layer_ids:
            weight_map.pop(key)

    weight_map.update(add_layers)
    # save json
    with open(model_index_json, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"model index file updated: {model_index_json}")

def replace(args, mix_layers):
    mix_layers_set = set(mix_layers)

    bf16_layer_dict = find_layer_count(args.bf16_path, mix_layers_set)
    w8a8_layer_dict = find_layer_count(args.w8a8_path, mix_layers_set)
    logger.info(f"{len(w8a8_layer_dict)} int8 safetensors files will be updated from {len(bf16_layer_dict)} bf16 files")

    add_layers = replace_bf16_layers(args.target_dir, args.w8a8_path, args.bf16_path, w8a8_layer_dict, bf16_layer_dict, mix_layers)
    copy_w8a8_files(args.target_dir, args.w8a8_path, w8a8_layer_dict)
    update_config(mix_layers, args.target_dir)
    update_model_index(args.target_dir, mix_layers_set, add_layers)
    logger.info('All operations completed!')

def replace_debug(args, mix_layers):
    # If sharding is needed
    if args.shard:
        logger.info("Starting weight sharding...")

        # Auto-detect number of files in BF16 path
        bf16_total_files = count_safetensor_files(args.bf16_path)
        if bf16_total_files > 0:
            logger.info(f"Starting BF16 weight sharding... (detected {bf16_total_files} files)")
            shard_weights(
                input_path=f"{args.bf16_path}/model-{{:05d}}-of-{{:05d}}.safetensors",
                output_dir="./bf16weights",
                prefix="model",
                total_files=bf16_total_files,
                num_layers=args.num_layers,
                desc="BF16"
            )

        # Auto-detect number of files in W8A8 path
        w8a8_total_files = count_safetensor_files(args.w8a8_path)
        if w8a8_total_files > 0:
            logger.info(f"Starting W8A8 weight sharding... (detected {w8a8_total_files} files)")
            shard_weights(
                input_path=f"{args.w8a8_path}/model-{{:05d}}-of-{{:05d}}.safetensors",
                output_dir="./int8weights",
                prefix="model",
                total_files=w8a8_total_files,
                num_layers=args.num_layers,
                desc="W8A8"
            )

        logger.info("Sharding completed!")

            # Copy other files (config files, tokenizer, etc.) - must be executed before updating config file
        logger.info("Copying other files...")
        copy_other_files(args.w8a8_path, args.target_dir)


    # Update config file
    logger.info("Updating config file...")
    update_config(mix_layers, args.target_dir)

    # If moving safetensor files is needed
    if args.movesafetensors:
        logger.info("Starting to move safetensor files...")
        move_safetensors(mix_layers, args.num_layers, args.target_dir)
        logger.info("File moving completed!")

    logger.info("All operations completed!")


@ModifierFactory.register("mixedPrecisionModifier")
class mixedPrecisionModifier(QuantizationMixin, Modifier):

    layers: List[str]
    num_layers: int
    ifDebug: bool = False
    ifShard: bool = False
    ifMoveSafetensors: bool = False

    def on_initialize(self, state, *args, **kwargs):
        logger.info("mix: on_initialize")
        # Fail fast before quantization: backfill needs --mix_path distinct from --save_path
        if state.mix_path is None:
            raise ValueError(
                "mixedPrecisionModifier requires --mix_path: --save_path holds the fully-quantized "
                "model, while --mix_path is a separate directory for the mixed model (selected "
                "layers copied from the source model). Remove this modifier if no mixed model is needed."
            )
        if os.path.abspath(state.mix_path) == os.path.abspath(state.save_path):
            raise ValueError(
                "--mix_path must differ from --save_path: --save_path holds the fully-quantized model "
                "and --mix_path the mixed model."
            )
        self.initialized_ = True

    def on_start(self, state, *args, **kwargs):
        logger.info("mix: on_start")
        super().on_start(state)

    def on_end(self, state):
        logger.info("mix: on_end")

        try:
            mix_layers = parse_layers(self.layers)
            # Validate layer indices are within valid range
            if any(layer < 0 or layer >= self.num_layers for layer in mix_layers):
                raise ValueError(f"Layer indices must be between 0 and {self.num_layers-1}")
        except ValueError as e:
            logger.error(f"Error parsing layers: {e}")
            exit(1)

        logger.info(f"Number of layers used: {self.num_layers}")
        logger.info(f"Mixed layers: {mix_layers}")
        logger.info(f"Target directory: {state.mix_path}")

        fake_args = SimpleNamespace(
            bf16_path=state.model.path,
            w8a8_path=state.save_path,
            target_dir=state.mix_path,
            layers=self.layers,
            num_layers=self.num_layers,
            debug=self.ifDebug,
            shard=self.ifShard,
            movesafetensors=self.ifMoveSafetensors
        )
        if self.ifDebug:
            replace_debug(fake_args, mix_layers)
        else:
            replace(fake_args, mix_layers)

        super().on_end(state)

    def on_finalize(self, state):
        logger.info("mix: on_finalize")
        super().on_finalize(state)

    def on_event(self, state, event: Event, **kwargs):
        logger.info("mix: on_event")
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state)

        if event.type_ == EventType.CALIBRATION_EPOCH_END:
            if not self.ended_:
                self.on_end(state)
