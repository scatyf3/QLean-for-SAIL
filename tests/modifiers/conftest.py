# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Minimal state fixtures for modifier-level unit tests.

Production modifiers are imported lazily (deferred production imports), and
re-registered idempotently per test: ``isolated_state`` snapshots and restores
the factory registries around every test, so a one-time module import during
an earlier test is erased from the live registry. Registering with
``setdefault`` inside the fixture keeps every test self-sufficient.
"""

import pytest


class _MinimalState:
    """Stand-in for PipelineState with only the fields basic modifiers need."""

    def __init__(self, model_path, save_path):
        self.model = type("Model", (), {"path": model_path})()
        self.save_path = save_path


@pytest.fixture
def minimal_state(tmp_path, isolated_state):
    """A state with empty model/save directories and reset factory state."""
    _register_modifiers()
    model_dir = tmp_path / "model"
    save_dir = tmp_path / "save"
    model_dir.mkdir()
    save_dir.mkdir()
    return _MinimalState(str(model_dir), str(save_dir))


def _register_modifiers():
    """Idempotently register the production modifiers under test.

    Importing the module registers its class once per process; after
    isolated_state restores a stale snapshot, setdefault re-adds it to the
    live registry without triggering the "already registered" error.
    """
    from qlean.modifiers.factory import ModifierFactory

    for module_name, modifier_name in (
        ("qlean.modifiers.basic.base", "BasicModifier"),
        # Extend per module as more modifier tests are added:
        # ("qlean.modifiers.W4a8Day0.base", "W4a8Day0Modifier"),
        # ("qlean.modifiers.int4Dequant.base", "int4DequantModifier"),
    ):
        module = __import__(module_name, fromlist=["base"])
        ModifierFactory._main_registry.setdefault(
            modifier_name, getattr(module, modifier_name)
        )
