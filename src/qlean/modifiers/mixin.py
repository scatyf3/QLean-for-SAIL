# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional, Set

from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationScheme,
)
from qlean.utils.hooks import HooksMixin


class QuantizationMixin(HooksMixin):
    config_group: Optional[Dict[str, QuantizationScheme]] = None
    _resolved_config: Optional[QuantizationConfig] = None
    pass
