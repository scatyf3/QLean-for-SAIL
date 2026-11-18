# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Any, Optional
from qlean.args import ModelArguments

__all__ = ["Model"]


class Model(ABC):
    """Base interface for model loading and optional forward execution."""

    name: str
    path: str
    model_: Optional[Any]  # Loaded model or checkpoint file list.
    processor: Optional[Any]

    def __init__(self, model_args: ModelArguments):
        """Store the model name and source path without loading weights."""
        self.name = model_args.model_name
        self.path = model_args.model_path
        self.model_ = None

    @abstractmethod
    def load_model():
        """Load the model representation in a subclass."""
        pass

    @abstractmethod
    def forward():
        """Define forward execution in subclasses that support inference."""
        pass
