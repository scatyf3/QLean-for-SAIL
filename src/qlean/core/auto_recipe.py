# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import logging
from importlib.resources import files
from pathlib import Path

import yaml
from safetensors import safe_open

logger = logging.getLogger(__name__)

AUTO_SOURCE_FORMATS = {"BF16": "bf16", "FP8": "fp8_block", "NVFP4": "nvfp4"}
AUTO_SCHEMES = {
    "INT8_W8A8": "W8A8",
    "FP8_BLOCK": "FP8_BLOCK",
    "FP8_DYNAMIC": "FP8_DYNAMIC",
    "MXFP4": "MXFP4",
    "BF16": None,
}
AUTO_CONVERSIONS = frozenset(
    (source, target)
    for source in AUTO_SOURCE_FORMATS
    for target in AUTO_SCHEMES
    if (source, target) != ("BF16", "BF16")
)

_MXFP4_IGNORE = ("visual.*", "model.visual.*", "mtp.*", "*.mtp.*")
ARCHITECTURE_PRESETS = {
    "Qwen3VLForConditionalGeneration": "qwen3_vl",
    "qwen3_vl": "qwen3_vl",
}


def build_auto_recipe(input_format: str, output_format: str, model_path: str,
                      architecture: str | None = None) -> dict:
    from qlean.modifiers_engine._runtime.region_router import (
        CONSUMED_SOURCE_SUFFIXES, _ENGINE_DEFAULT_SCALE_SUFFIXES,
        match_patterns, region_patterns,
    )

    if (input_format, output_format) not in AUTO_CONVERSIONS:
        raise ValueError(
            f"Unsupported automatic conversion: {input_format} → {output_format}; "
            "use --list-formats or an explicit --recipe"
        )
    src = AUTO_SOURCE_FORMATS[input_format]
    scheme = AUTO_SCHEMES[output_format]
    preset = ARCHITECTURE_PRESETS.get(architecture, "default")
    template = files("qlean.configs").joinpath("day0/_arch/default.yaml")
    with template.open("r", encoding="utf-8") as handle:
        recipe = yaml.safe_load(handle)
    modifier = recipe["quant_stage"]["quant_modifiers"]["GeneralQuantEngineModifier"]
    ignore_region = next(r for r in modifier["regions"] if r["name"] == "ignore")
    ignore = region_patterns(ignore_region)
    if output_format == "MXFP4" or preset == "qwen3_vl":
        ignore.extend(_MXFP4_IGNORE)
    if preset == "qwen3_vl":
        ignore.append("*.mlp.gate.*")
    ignore_region["match"] = ignore

    names = set()
    for shard in sorted(Path(model_path).glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            names.update(handle.keys())
    suffixes = CONSUMED_SOURCE_SUFFIXES[src] or _ENGINE_DEFAULT_SCALE_SUFFIXES
    remaining = {name for name in names if not name.endswith(suffixes)}
    regions = []
    for region in modifier["regions"][:-1]:
        hits = {name for name in remaining if match_patterns(name, region)}
        if not hits:
            continue
        if isinstance(region.get("match"), list):
            region["match"] = [pattern for pattern in region["match"]
                               if any(match_patterns(name, {"match": pattern}) for name in hits)]
        region["src"] = src
        if region["name"] == "ignore":
            if src != "bf16":
                region["dst"] = "bf16"
        else:
            region["dst"] = scheme or "bf16"
        regions.append(region)
        remaining -= hits
    regions.append(modifier["regions"][-1])
    modifier["regions"] = regions
    modifier.update(
        primary_scheme=scheme,
        config_writer="compressed_tensors" if scheme else "none",
        sglang_wa=output_format == "MXFP4",
        dequant_dtype="fp32" if scheme else "bf16",
    )
    if src == "fp8_block":
        modifier["dequant_block_size"] = 128
    elif src == "nvfp4":
        modifier["nvfp4_group_size"] = 16
    logger.info("Generated GeneralQuantEngineModifier recipe: preset=%s, %s → %s",
                preset, input_format, output_format)
    return recipe


def retarget_engine_recipe_to_bf16(recipe_dict: dict) -> int:
    """Rewrite every GeneralQuantEngineModifier in `recipe_dict` to emit bf16.

    Returns the number of modifiers rewritten; 0 means nothing to retarget.
    """
    from qlean.modifiers_engine._runtime.region_router import effective_src

    rewritten = 0
    for stage_val in recipe_dict.values():
        if not isinstance(stage_val, dict):
            continue
        for group_val in stage_val.values():
            if not isinstance(group_val, dict):
                continue
            modifier = group_val.get("GeneralQuantEngineModifier")
            if not isinstance(modifier, dict):
                continue
            for region in modifier.get("regions") or []:
                if not isinstance(region, dict):
                    continue
                # Streaming only bounds memory for row-wise quantization; moot without quant.
                region.pop("chunked", None)
                region.pop("chunk_rows", None)
                if "dst" not in region:
                    continue
                # effective_src returns the raw recipe spelling, canonicalized later.
                if effective_src(region).lower() == "bf16":
                    # bf16 source is copied through; `dst: bf16` is a rejected no-op.
                    region.pop("dst")
                    # Layout transforms feed a quant op; pop them when no quant runs.
                    region.pop("expert_mode", None)
                    region.pop("split_gate_up", None)
                    region.pop("transpose", None)
                else:
                    region["dst"] = "bf16"
            # scale_search only tunes acext_int4 scales; moot without quant.
            modifier.pop("scale_search", None)
            modifier.update(
                primary_scheme=None,
                config_writer="none",
                sglang_wa=False,
                dequant_dtype="bf16",
            )
            rewritten += 1
    return rewritten
