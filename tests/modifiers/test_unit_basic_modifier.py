# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle behavior of the BasicModifier, including lazy start via events."""

import pytest

pytestmark = pytest.mark.unit


def _make_modifier():
    from qlean.modifiers.basic.base import BasicModifier

    return BasicModifier()


@pytest.mark.cpu
def test_on_initialize_marks_initialized(minimal_state):
    modifier = _make_modifier()
    assert modifier.initialized_ is False

    modifier.on_initialize(minimal_state)

    assert modifier.initialized_ is True


@pytest.mark.cpu
def test_lifecycle_methods_set_state_flags(minimal_state):
    modifier = _make_modifier()
    modifier.on_initialize(minimal_state)

    modifier.on_start(minimal_state)
    assert modifier.started_ is True

    modifier.on_end(minimal_state)
    assert modifier.ended_ is True

    modifier.on_finalize(minimal_state)
    assert modifier.finalized_ is True


@pytest.mark.cpu
def test_epoch_start_event_lazily_starts_modifier(minimal_state):
    from qlean.core.event import Event, EventType

    modifier = _make_modifier()
    modifier.on_initialize(minimal_state)
    assert modifier.started_ is False

    modifier.on_event(minimal_state, Event(type_=EventType.CALIBRATION_EPOCH_START))

    assert modifier.started_ is True


@pytest.mark.cpu
def test_epoch_end_event_marks_ended_idempotently(minimal_state, monkeypatch):
    from qlean.core.event import Event, EventType

    modifier = _make_modifier()
    modifier.on_initialize(minimal_state)

    # Spy on the real on_end hook to observe actual call counts (the
    # BasicModifier guard should keep repeated events from re-triggering it).
    # BasicModifier is a pydantic model, so the spy patches the class method.
    from qlean.modifiers.basic.base import BasicModifier

    calls = []
    real_on_end = BasicModifier.on_end

    def counting_on_end(self, state):
        calls.append(1)
        real_on_end(self, state)

    monkeypatch.setattr(BasicModifier, "on_end", counting_on_end)

    # The epoch-end event ends the modifier even without an explicit start.
    modifier.on_event(minimal_state, Event(type_=EventType.CALIBRATION_EPOCH_END))
    assert modifier.ended_ is True
    assert len(calls) == 1

    # A repeated epoch-end event must not call on_end a second time.
    modifier.on_event(minimal_state, Event(type_=EventType.CALIBRATION_EPOCH_END))
    assert modifier.ended_ is True
    assert len(calls) == 1  # guard prevented a second on_end call


@pytest.mark.cpu
def test_factory_can_create_explicitly_registered_basic_modifier(minimal_state):
    from qlean.modifiers.basic.base import BasicModifier  # noqa: F401  (registers "BasicModifier")
    from qlean.modifiers.factory import ModifierFactory

    modifier = ModifierFactory.create("BasicModifier")

    assert isinstance(modifier, BasicModifier)
    modifier.on_initialize(minimal_state)
    assert modifier.initialized_ is True
