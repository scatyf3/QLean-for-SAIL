# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from qlean.args import DatasetArguments, ModelArguments, OtherArguments
from qlean.core.qleancompressor import QLeanCompressor

import logging

logger = logging.getLogger(__name__)


def post_train(
    model_args: ModelArguments,
    dataset_args: DatasetArguments,
    other_args: OtherArguments,
):
    """Run the configured conversion and write its checkpoint and metadata."""
    logger.debug(f"model_args: {repr(model_args)}")
    logger.debug(f"dataset_args: {repr(dataset_args)}")
    logger.debug(f"other_args: {repr(other_args)}")

    compressor = QLeanCompressor(model_args)
    if dataset_args.dataset_name is not None:
        compressor.set_calibration_dataset(dataset_args)
    compressor.post_train(other_args)
