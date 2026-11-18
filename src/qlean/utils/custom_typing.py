# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

from qlean.models import Model

# Pipeline callable
PipelineFn = Callable[[Model, str, "OtherArguments"], None]
