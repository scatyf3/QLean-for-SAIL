# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
import os

from compressed_tensors import __version__ as ct_version
from compressed_tensors.base import (
    COMPRESSION_VERSION_NAME,
    QUANTIZATION_CONFIG_NAME,
    QUANTIZATION_METHOD_NAME,
    SPARSITY_CONFIG_NAME,
    TRANSFORM_CONFIG_NAME,
)
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
)
from transformers.file_utils import CONFIG_NAME
import logging

logger = logging.getLogger(__name__)

__all__ = ["update_config", "WA_for_sglang_config"]


def WA_for_sglang_config(config_file, fp8_channelwise_layers=None, fp8_blockwise_layers=None):

    config = json.load(open(config_file))
    if "quantization_config" in config:
        qc = config["quantization_config"]

        if "config_groups" in qc:
            del qc["config_groups"]
        if "format" in qc:
            del qc["format"]
        if "global_compression_ratio" in qc:
            del qc["global_compression_ratio"]

        if "kv_cache_scheme" in qc:
            del qc["kv_cache_scheme"]
        if "quant_method" in qc:
            qc["quant_method"] = "mxfp4"
        if "quantization_status" in qc:
            del qc["quantization_status"]

        if "sparsity_config" in qc:
            del qc["sparsity_config"]
        if "transform_config" in qc:
            del qc["transform_config"]
        if "version" in qc:
            del qc["version"]
        if fp8_channelwise_layers is not None:
            qc["fp8_channelwise_layers"] = fp8_channelwise_layers
        if fp8_blockwise_layers is not None:
            qc["fp8_blockwise_layers"] = fp8_blockwise_layers

    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=True)

def update_config(
    config_path: str | os.PathLike,
    save_directory: str | os.PathLike,
    scheme_name: str,
    scheme: QuantizationScheme,
    ignore: list[str],
):
    from compressed_tensors.utils.match import _match_name

    for i in range(len(ignore) - 1, -1, -1):
        if _match_name("model.embed_tokens", ignore[i]):
            logger.debug(f"ignore list pop {ignore[i]}")
            ignore.pop(i)

    # W8A8_BLOCK: use simple custom quantization_config format
    if scheme_name == "W8A8_BLOCK":
        qconfig_data = {
            "activation_scheme": "dynamic",
            "quant_method": "blockwise_int8",
            "weight_block_size": [128, 128],
        }
    else:
        # construct quantization config (compressed-tensors format)
        qconfig = QuantizationConfig.model_validate(
            {
                "config_groups": {scheme_name: scheme},
                "ignore": ignore,
                "quantization_status": QuantizationStatus.COMPRESSED,
            }
        )

        # construct compression (quantization) config
        qconfig_data = qconfig.model_dump(exclude=[QUANTIZATION_METHOD_NAME, "format"])
        qconfig_data = {
            COMPRESSION_VERSION_NAME: ct_version,
            QUANTIZATION_METHOD_NAME: "compressed-tensors",
            SPARSITY_CONFIG_NAME: {},
            TRANSFORM_CONFIG_NAME: {},
            "format": scheme.format,
            **qconfig_data,
        }

    # write results to config.json file
    logger.info(f"Try find config in {config_path}")
    config_file_path = _find_config_path(config_path)
    if config_file_path is not None:
        config_file_name = os.path.basename(config_file_path)
        with open(config_file_path, "r") as file:
            config_data = json.load(file)

        config_data[QUANTIZATION_CONFIG_NAME] = qconfig_data

        with open(os.path.join(save_directory, config_file_name), "w") as file:
            json.dump(config_data, file, indent=2, sort_keys=True)
            logger.info(
                f"New config file saved to {save_directory}, ori in {config_file_path}"
            )

    else:
        logger.warning(
            f"Could not find config file in {config_path}. "
            f"Please {json.dumps(qconfig_data, indent=2, sort_keys=True)}"
        )


def _find_config_path(save_directory: str | os.PathLike) -> str | None:
    for file_name in os.listdir(save_directory):
        if file_name in (CONFIG_NAME, "params.json"):
            return os.path.join(save_directory, file_name)

    return None
