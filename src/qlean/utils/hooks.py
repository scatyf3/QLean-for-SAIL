# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
from functools import partial, wraps
from typing import Any, Callable, ClassVar, Optional, Set, Union

import torch
from pydantic import BaseModel
from torch.utils.hooks import RemovableHandle

__all__ = ["HooksMixin"]


class HooksMixin(BaseModel):
    """Register removable PyTorch hooks with an optional shared disable switch."""

    # attached to global HooksMixin class
    _HOOKS_DISABLED: ClassVar[bool] = False
    _HOOKS_KEEP_ENABLED: ClassVar[Set[RemovableHandle]] = set()

    # attached to local subclasses
    _hooks: Set[RemovableHandle] = set()

    @classmethod
    @contextlib.contextmanager
    def disable_hooks(cls, keep: Set[RemovableHandle] = frozenset()):
        """Set the class disable flag except for kept handles, then clear it on exit."""

        try:
            cls._HOOKS_DISABLED = True
            cls._HOOKS_KEEP_ENABLED |= keep
            yield
        finally:
            cls._HOOKS_DISABLED = False
            cls._HOOKS_KEEP_ENABLED -= keep

    def register_hook(
        self,
        target: Union[torch.nn.Module, torch.nn.Parameter],
        hook: Callable[[Any], Any],
        hook_type: str,
        **kwargs,
    ) -> RemovableHandle:
        """Register a guarded target hook and return its tracked removable handle."""

        handle = None

        @wraps(hook)
        def wrapped_hook(*args, **kwargs):
            nonlocal handle

            if (
                HooksMixin._HOOKS_DISABLED
                and handle not in HooksMixin._HOOKS_KEEP_ENABLED
            ):
                return

            return hook(*args, **kwargs)

        register_function = self._get_register_function(target, hook_type)
        handle = register_function(wrapped_hook, **kwargs)
        self._hooks.add(handle)
        return handle

    def remove_hooks(self, handles: Optional[Set[RemovableHandle]] = None):
        """Remove selected hooks, or all tracked hooks when handles is None."""

        if handles is None:
            handles = self._hooks

        for hook in handles:
            hook.remove()

        self._hooks -= handles

    def _get_register_function(
        self, target: torch.nn.Module, hook_type: str
    ) -> Callable:

        return getattr(target, f"register_{hook_type}_hook")
