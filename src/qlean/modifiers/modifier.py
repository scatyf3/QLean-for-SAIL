# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math
from abc import ABC, abstractmethod

from qlean.core.state import State
from qlean.core.event import Event

__all__ = ["Modifier"]


class Modifier(ABC):
    """Base hooks for initialization, start, events, end, and finalization."""

    initialized_: bool = False
    finalized_: bool = False
    started_: bool = False
    ended_: bool = False

    @abstractmethod
    def on_initialize(self, state: State):
        """Validate settings and initialize resources in a subclass."""
        raise NotImplementedError()

    def on_start(self, state: State):
        """Mark the modifier as started."""
        self.started_ = True

    def on_event(self, state: State, event: Event):
        """Handle a pipeline event; the base implementation does nothing."""
        pass

    def on_end(self, state: State):
        """Mark the modifier as ended."""
        self.ended_ = True

    def on_finalize(self, state: State):
        """Mark the modifier as finalized."""
        self.finalized_ = True
