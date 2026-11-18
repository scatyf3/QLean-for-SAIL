# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional


@dataclass
class DatasetArguments:
    """Dataset options reserved for the unimplemented calibration loader."""

    dataset_name: Optional[str] = None
    dataset_path: Optional[str] = None
    dataset: Optional[str] = None
