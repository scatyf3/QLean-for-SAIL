# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
from glob import glob

from qlean.models import ModelFactory, Model
from qlean.args import ModelArguments
from qlean.utils import get_checkpoint_files, is_weights_file

import logging

logger = logging.getLogger(__name__)


@ModelFactory.register("day0")
class day0Model(Model):

    def load_model(self, model_args: ModelArguments):
        logger.debug("day0: load model")
        weight_files = []
        model_files = get_checkpoint_files(self.path)
        for file_path, resolved_path in model_files:
            if is_weights_file(file_path) and file_path.endswith(".safetensors"):
                weight_files.append((file_path, resolved_path))
        weight_files.sort(key=lambda x: x[0])
        self.model_ = weight_files

    def forward():
        logger.debug("day0: forward")
