# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Optional

from qlean.models import Model


@dataclass
class State:
    """Model and output paths shared by all modifiers in a conversion."""

    model: Model
    save_path: Optional[str] = None
    mix_path: Optional[str] = None
