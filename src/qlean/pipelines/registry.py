# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

from qlean.utils.custom_typing import PipelineFn


def get_pipeline_from_modifiers(modifiers: List[str]) -> PipelineFn:
    """Return the data-free pipeline; modifier-based selection is not implemented."""

    all_pipelines = get_pipelines(modifiers)

    # Only the data-free pipeline is currently selected.
    inferred = "datafree"

    return all_pipelines[inferred]


def get_pipelines(modifiers: List[str]) -> str:
    """Return the available pipeline mapping; modifiers is currently unused."""

    from qlean.pipelines import (
        datafree,
    )

    return {
        "datafree": datafree.run_pipeline,
    }
