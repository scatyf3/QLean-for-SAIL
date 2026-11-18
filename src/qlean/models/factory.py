# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import importlib
import pkgutil
from typing import Dict, Type

from qlean.models import Model
from qlean.args import ModelArguments

import logging

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    A factory class for loading and registering models.
    """

    _MAIN_PACKAGE_PATH = "qlean.models"

    _loaded: bool = False
    _main_registry: Dict[str, Type[Model]] = {}
    _errors: Dict[str, Exception] = {}

    @staticmethod
    def load_from_package(package_path: str = _MAIN_PACKAGE_PATH):
        """Import model modules for registration, logging import failures."""
        if ModelFactory._loaded:
            logger.info("Models already loaded.")
            return

        all_succeed = True
        try:
            main_package = importlib.import_module(package_path)
        except Exception as package_err:
            logger.error(f"Failed to import package {package_path}: {package_err}")
            return

        for importer, modname, ispkg in pkgutil.walk_packages(
            main_package.__path__, main_package.__name__ + "."
        ):
            try:
                module = importlib.import_module(modname)
            except Exception as module_err:
                logger.warning(f"Failed to load module {modname}: {module_err}")
                all_succeed = False

        if all_succeed:
            ModelFactory._loaded = True
            logger.info("ModelFactory initialized successfully.")
        else:
            logger.info("Some modules failed to load. Factory remains unloaded.")

    @staticmethod
    def register(modeltype: str):
        """Return a decorator that registers a Model subclass under a unique name."""
        def decorator(cls) -> Type:
            if modeltype in ModelFactory._main_registry:
                raise ValueError(f"Model {modeltype} already registered.")
            if not isinstance(cls, type):
                raise TypeError(f"Object {cls} is not a class.")
            if not issubclass(cls, Model):
                raise TypeError(f"Class {cls} is not a subclass of Model.")

            ModelFactory._main_registry[modeltype] = cls
            return cls

        return decorator

    @staticmethod
    def create(modeltype: str, model_args: ModelArguments) -> Model:
        """Instantiate a registered model, or raise if the type is unavailable."""
        logger.debug(f"ModelFactory.create called with modeltype {modeltype}")
        logger.debug(f"Modelfactory registry: {ModelFactory._main_registry}")

        if modeltype in ModelFactory._errors:
            raise ModelFactory._errors[modeltype]

        if modeltype in ModelFactory._main_registry:
            return ModelFactory._main_registry[modeltype](model_args)

        raise ValueError(f"No model of type {modeltype} found")
