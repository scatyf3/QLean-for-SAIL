# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Any, Optional

from qlean.utils.singleton import SingletonMixin
from qlean.args import ModelArguments, DatasetArguments, OtherArguments
from qlean.models import ModelFactory
from qlean.pipelines import get_pipeline_from_modifiers

from qlean.core.event_mixin import EventMixin
from qlean.core.state import State
from qlean.core.utils import (
    get_modifiers_from_recipe,
    preprocess_model,
    install_dependencies,
    remove_unsupported_fields,
    save_qlean_metadata,
)

import logging

logger = logging.getLogger(__name__)


class QLeanCompressor(EventMixin, SingletonMixin):
    """Load a model and run recipe modifiers to write a converted checkpoint."""

    from qlean.modifiers import Modifier

    state: State
    modifiers: List[Modifier]
    _model_args: ModelArguments
    _quant_meta: dict
    dataloader: Optional[Any] = None

    def __init__(self, model_args: ModelArguments):
        """Install declared dependencies, load the model, and create modifiers."""
        mode = install_dependencies(model_args.model_name)
        logger.info(f"Using mode: {mode}")

        if not ModelFactory._loaded:
            ModelFactory.load_from_package()

        model = ModelFactory.create(mode, model_args)
        model.load_model(model_args)
        model = preprocess_model(model)

        self.state = State(model=model, save_path=model_args.save_path, mix_path=model_args.mix_path)
        self._model_args = model_args
        self.modifiers, self._quant_meta = get_modifiers_from_recipe(mode, model_args)


    def set_calibration_dataset(dataset_args: DatasetArguments):
        """Placeholder for calibration loading; not implemented or bound as an instance method."""
        pass

    def post_train(self, other_args: OtherArguments):
        """Run the selected pipeline and write QLean conversion metadata."""
        pipeline = get_pipeline_from_modifiers(self.modifiers)
        pipeline(self.state.model, self.dataloader, other_args)

        # Remove unsupported scale_dtype and zp_dtype fields from compressed-tensors config.
        remove_unsupported_fields(self.state.save_path)

        save_qlean_metadata(
            save_path=self.state.save_path,
            model_args=self._model_args,
            **self._quant_meta,
        )