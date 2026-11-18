# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import logging

logger = logging.getLogger(__name__)


def preflight(source_files, max_shard_bytes_warn: int) -> None:
    basenames = {}
    for file_name, resolved_path in source_files:
        base = os.path.basename(file_name)
        if base in basenames:
            raise ValueError(
                f"Duplicate source shard basename: {base} ({basenames[base]} and {file_name}). "
                f"Same-name output uses basenames; duplicates would overwrite each other."
            )
        basenames[base] = file_name

        size = os.path.getsize(resolved_path)
        if size > max_shard_bytes_warn:
            logger.warning(
                f"Source shard {file_name} is large ({size / 1024 ** 3:.1f} GB > "
                f"warning threshold {max_shard_bytes_warn / 1024 ** 3:.0f} GB); "
                f"the output shard will be comparable in size; check memory and disk capacity."
            )
