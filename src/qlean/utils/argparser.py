# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import fields
from typing import Tuple

from qlean.args import DatasetArguments, ModelArguments, OtherArguments


def parse_to_dataclass(
    args,
) -> Tuple[ModelArguments, DatasetArguments, OtherArguments]:
    """Split an argparse namespace into model, dataset, and other options."""
    instances = []
    all_values = vars(args)

    mydataclasses = [ModelArguments, DatasetArguments, OtherArguments]
    for cls in mydataclasses:
        field_names = {f.name for f in fields(cls) if f.init}
        filtered = {k: v for k, v in all_values.items() if k in field_names}
        instance = cls(**filtered)
        instances.append(instance)
    return tuple(instances)
