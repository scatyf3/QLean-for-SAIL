# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import List
from huggingface_hub import list_repo_files
from transformers.utils.hub import cached_file

import logging

logger = logging.getLogger(__name__)

__all__ = ["get_checkpoint_files", "is_weights_file"]

weights_files = [
    ".bin",
    ".safetensors",
    ".pth",
    ".msgpack",
    ".pt",
]


def is_weights_file(file_name: str) -> bool:
    """Return whether the name ends with a supported weight-file suffix."""
    return any(file_name.endswith(suffix) for suffix in weights_files)


def get_checkpoint_files(model_stub: str | os.PathLike) -> List[str]:
    """Return (relative name, resolved path) pairs for local or Hub files.

    Existing paths are treated as local directories; others are treated as Hub IDs.
    The returned list is not filtered to weight files.
    """
    if os.path.exists(model_stub):
        file_paths = walk_file_paths(model_stub, ignore=".cache")
    else:
        file_paths = list_repo_files(model_stub)

    return [(file_path, cached_file(model_stub, file_path)) for file_path in file_paths]


def walk_file_paths(root_dir: str, ignore: str | None = None) -> List[str]:
    """List immediate file names, excluding names with the optional ignore prefix."""
    all_files = []
    for filename in os.listdir(root_dir):
        filepath = os.path.join(root_dir, filename)
        if os.path.isfile(filepath):
            rel_path = filename
            if not (ignore and rel_path.startswith(ignore)):
                all_files.append(rel_path)
    return all_files