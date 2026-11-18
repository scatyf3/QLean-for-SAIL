# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Isolated singleton and lifecycle tests without model initialization."""

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.cpu, pytest.mark.usefixtures("isolated_state")]


@pytest.mark.parametrize("case", ["first", "second"])
def test_each_case_starts_with_clean_state(case):
    from qlean.core.event_lifecycle import EventLifecycle
    from qlean.utils.singleton import SingletonMixin

    assert SingletonMixin._instances == {}
    assert EventLifecycle.initialized is False
    assert EventLifecycle.finalized is False
    SingletonMixin._instances[SingletonMixin] = case
    EventLifecycle.initialized = True
    EventLifecycle.finalized = True


def test_singleton_rejects_second_construction():
    from qlean.utils.singleton import SingletonMixin

    class Session(SingletonMixin):
        pass

    with pytest.raises(ValueError, match="has not been created"):
        Session.instance()
    instance = Session()
    assert Session.instance() is instance
    with pytest.raises(ValueError, match="more than one instance"):
        Session()


def test_lifecycle_validates_order_before_handler():
    from qlean.core.event_lifecycle import EventLifecycle

    calls = []

    class Session:
        @EventLifecycle.initialize
        def initialize(self):
            calls.append("initialize")

        @EventLifecycle.finalize
        def finalize(self):
            calls.append("finalize")

    session = Session()
    with pytest.raises(ValueError, match="Cannot finalize"):
        session.finalize()
    assert calls == []
    session.initialize()
    assert EventLifecycle.initialized is True
    session.finalize()
    assert calls == ["initialize", "finalize"]
    assert EventLifecycle.initialized is False
    assert EventLifecycle.finalized is True
    with pytest.raises(ValueError, match="Cannot finalize"):
        session.finalize()
    assert calls == ["initialize", "finalize"]
