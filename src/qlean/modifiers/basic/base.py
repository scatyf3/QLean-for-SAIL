# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import logging

from qlean.modifiers import Modifier, ModifierFactory
from qlean.modifiers import QuantizationMixin
from qlean.core.event import Event, EventType

logger = logging.getLogger(__name__)


@ModifierFactory.register("BasicModifier")
class BasicModifier(QuantizationMixin, Modifier):
    """Track lifecycle events without modifying model weights."""

    def on_initialize(self, state, *args, **kwargs):
        logger.info("basic: on_initialize")
        self.initialized_ = True

    def on_start(self, state, *args, **kwargs):
        logger.info("basic: on_start")
        super().on_start(state)

    def on_end(self, state):
        logger.info("basic: on_end")
        super().on_end(state)

    def on_finalize(self, state):
        logger.info("basic: on_finalize")
        super().on_finalize(state)

    def on_event(self, state, event: Event, **kwargs):
        logger.info("basic: on_event")
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state)

        if event.type_ == EventType.CALIBRATION_EPOCH_END:
            if not self.ended_:
                self.on_end(state)
