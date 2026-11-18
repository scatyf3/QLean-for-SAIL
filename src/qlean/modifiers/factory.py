# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import importlib
import pkgutil
from typing import Dict, Type

from qlean.modifiers import Modifier
from qlean.args import ModelArguments

import logging

logger = logging.getLogger(__name__)


class ModifierFactory:
    """
    A factory class for loading and registering modifiers.
    """

    _MAIN_PACKAGE_PATH = "qlean.modifiers"
    _ENGINE_PACKAGE_PATH = "qlean.modifiers_engine"

    _loaded: bool = False
    _main_registry: Dict[str, Type[Modifier]] = {}
    _errors: Dict[str, Exception] = {}

    @staticmethod
    def load_from_package(package_path: str = _MAIN_PACKAGE_PATH):
        """Import modifiers for registration; the default scans legacy and engine packages."""
        if ModifierFactory._loaded:
            logger.info("Modifiers already loaded.")
            return

        # Engine modifiers and legacy modifiers live in two sibling packages; scan both.
        # An explicit non-default package_path keeps the old behavior (scan that package only).
        package_paths = [package_path]
        if package_path == ModifierFactory._MAIN_PACKAGE_PATH:
            package_paths.append(ModifierFactory._ENGINE_PACKAGE_PATH)

        all_succeed = True
        for path in package_paths:
            try:
                main_package = importlib.import_module(path)
            except Exception as package_err:
                logger.error(f"Failed to import package {path}: {package_err}")
                all_succeed = False
                continue

            for importer, modname, ispkg in pkgutil.walk_packages(
                main_package.__path__, main_package.__name__ + "."
            ):
                try:
                    module = importlib.import_module(modname)
                except Exception as module_err:
                    logger.warning(f"Failed to load module {modname}: {module_err}")
                    all_succeed = False

        if all_succeed:
            ModifierFactory._loaded = True
            logger.info("ModifierFactory initialized successfully.")
            logger.info(ModifierFactory._main_registry)
        else:
            logger.warning("Some modules failed to load. Factory remains unloaded.")

    @staticmethod
    def register(modeltype: str):
        """Return a decorator that registers a Modifier subclass under a unique name."""
        def decorator(cls) -> Type:
            if modeltype in ModifierFactory._main_registry:
                raise ValueError(f"Modifier {modeltype} already registered.")
            if not isinstance(cls, type):
                raise TypeError(f"Object {cls} is not a class.")
            if not issubclass(cls, Modifier):
                raise TypeError(f"Class {cls} is not a subclass of Modifier.")

            ModifierFactory._main_registry[modeltype] = cls
            return cls

        return decorator

    @staticmethod
    def create(modifiertype: str, **kwargs) -> Modifier:
        """Instantiate a registered modifier, or raise if the type is unavailable."""
        logger.debug(f"ModifierFactory.create called with modeltype {modifiertype}")

        if modifiertype in ModifierFactory._errors:
            raise ModifierFactory._errors[modifiertype]

        if modifiertype in ModifierFactory._main_registry:
            return ModifierFactory._main_registry[modifiertype](**kwargs)

        raise ValueError(f"No modifier of type {modifiertype} found")
