# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import yaml
import os
from importlib.resources import files

import logging

logger = logging.getLogger(__name__)

__all__ = [
    "load_env_rules",
]


def load_env_rules() -> dict:
    return _load_yaml("env_rules.yaml")


def _load_yaml(relative_path: str) -> dict:
    try:
        config_file = files("qlean.configs").joinpath(relative_path)
        logger.debug(f"config_file: {config_file}")
        with open(config_file, "r", encoding="utf-8") as f:
            content = f.read()
            logger.debug(f"content in yaml: {repr(content)}")
            try:
                return yaml.safe_load(content)
            except yaml.YAMLError as e:
                raise RuntimeError(f"Error loading YAML file: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load config '{relative_path}': {e}")
