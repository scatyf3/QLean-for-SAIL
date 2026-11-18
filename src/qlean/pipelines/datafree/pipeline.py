# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from qlean.core import get_compressor

import logging

logger = logging.getLogger(__name__)


def run_pipeline(model: Any, dataloader: Any, args: "OtherArguments"):
    """Run the quantization pipeline without calibration data."""
    logger.info("Running pipeline: datafree")
    compressor = get_compressor()
    compressor.initialize()
    compressor.calibration_epoch_start()
    compressor.calibration_epoch_end()
