# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Factory creation and event order with stubbed model loading and execution."""

from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.cpu, pytest.mark.usefixtures("isolated_state")]


@pytest.fixture
def factories(monkeypatch):
    from qlean.models import Model, ModelFactory
    from qlean.modifiers import Modifier, ModifierFactory

    for factory in (ModelFactory, ModifierFactory):
        monkeypatch.setattr(factory, "_main_registry", {})
        monkeypatch.setattr(factory, "_errors", {})
        monkeypatch.setattr(factory, "_loaded", True)
    return Model, ModelFactory, Modifier, ModifierFactory


def test_model_factory_preserves_type_and_arguments(tmp_path, factories):
    Model, factory, _, _ = factories

    @factory.register("toy")
    class Toy(Model):
        def __init__(self, args):
            super().__init__(args)
            self.args = args

        def load_model(self):
            pytest.fail("Factory creation must not load the model")

        def forward(self):
            pytest.fail("Factory creation must not run inference")

    args = SimpleNamespace(model_name="org/Toy", model_path=str(tmp_path))
    model = factory.create("toy", args)
    assert type(model) is Toy
    assert model.args is args
    assert model.name == "org/Toy"
    assert model.path == str(tmp_path)
    assert model.model_ is None


@pytest.mark.parametrize("kind", ["model", "modifier"])
def test_unknown_factory_type_fails_without_loading(factories, monkeypatch, kind):
    _, model_factory, _, modifier_factory = factories
    factory = model_factory if kind == "model" else modifier_factory
    monkeypatch.setattr(factory, "load_from_package", lambda *a: pytest.fail("Plugins must not be scanned"))
    with pytest.raises(ValueError, match=f"No {kind} of type absent found"):
        if kind == "model":
            factory.create("absent", SimpleNamespace())
        else:
            factory.create("absent")
    assert factory._main_registry == {}


@pytest.mark.parametrize("kind", ["model", "modifier"])
def test_factory_preserves_cached_error(factories, kind):
    _, model_factory, _, modifier_factory = factories
    factory = model_factory if kind == "model" else modifier_factory
    expected = RuntimeError("Known constructor error")
    factory._errors["broken"] = expected
    with pytest.raises(RuntimeError) as error:
        if kind == "model":
            factory.create("broken", SimpleNamespace())
        else:
            factory.create("broken")
    assert error.value is expected


def test_pipeline_dispatches_modifiers_in_recipe_order(tmp_path, factories, monkeypatch):
    from qlean.core.event_mixin import EventMixin
    from qlean.core.utils import get_modifiers_from_dict
    from qlean.pipelines.datafree import pipeline

    _, _, Modifier, factory = factories
    calls = []

    class Probe(Modifier):
        def __init__(self, **params):
            self.params = params

        def record(self, action, state):
            calls.append((self.params["label"], action, state))

        def on_initialize(self, state):
            self.initialized_ = True
            self.record("initialize", state)

        def on_event(self, state, event):
            self.record(event.type_.value, state)

        def on_end(self, state):
            super().on_end(state)
            self.record("end", state)

        def on_finalize(self, state):
            super().on_finalize(state)
            self.record("finalize", state)

    First = factory.register("First")(type("First", (Probe,), {}))
    Second = factory.register("Second")(type("Second", (Probe,), {}))
    recipe = {
        "prepare_stage": {"load_modifiers": {"First": {"label": "a"}}},
        "quant_stage": {"quant_modifiers": {"Second": {"label": "b"}}},
    }
    modifiers = get_modifiers_from_dict(recipe)
    assert [type(mod) for mod in modifiers] == [First, Second]
    assert [mod.params for mod in modifiers] == [
        {"label": "a", "stage": "prepare", "group": "load"},
        {"label": "b", "stage": "quant", "group": "quant"},
    ]
    session = EventMixin()
    session.state = SimpleNamespace(model=SimpleNamespace(path=str(tmp_path)))
    session.modifiers = modifiers
    monkeypatch.setattr(pipeline, "get_compressor", lambda: session)
    pipeline.run_pipeline(None, None, None)
    session.finalize()
    assert [(label, action) for label, action, _ in calls] == [
        ("a", "initialize"), ("b", "initialize"),
        ("a", "calibration_epoch_start"), ("b", "calibration_epoch_start"),
        ("a", "calibration_epoch_end"), ("b", "calibration_epoch_end"),
        ("a", "end"), ("a", "finalize"), ("b", "end"), ("b", "finalize"),
    ]
    assert all(state is session.state for _, _, state in calls)
    assert all(mod.ended_ and mod.finalized_ for mod in modifiers)
