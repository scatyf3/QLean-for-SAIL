# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Hadamard Transform implementation for rotation-based quantization.
"""

import logging
logger = logging.getLogger(__name__)
import math
import torch
try:
    from fast_hadamard_transform import hadamard_transform
    fwht = True
except ImportError:
    logger.warning(f"Failed to load fast_hadamard_transform")
    fwht = False


# const
GROUP_SIZE = 32
HAD_SCALE = 1 / math.sqrt(GROUP_SIZE)
# Lazily built; scipy is imported only when hadamard rotation is actually used
_HADAMARD_CPU: torch.Tensor | None = None


def _hadamard_cpu() -> torch.Tensor:
    global _HADAMARD_CPU
    if _HADAMARD_CPU is None:
        from scipy.linalg import hadamard
        _HADAMARD_CPU = (
            torch.tensor(hadamard(GROUP_SIZE), dtype=torch.bfloat16) * HAD_SCALE
        )
    return _HADAMARD_CPU


def get_hadamard_matrix(device: torch.device | str = "cuda") -> torch.Tensor:
    """Return the normalized 32x32 BF16 Hadamard matrix on the requested device."""
    return _hadamard_cpu().to(device)


# hadamard transform
def apply_had(x: torch.Tensor) -> torch.Tensor:
    """Rotate 2D weights in 32-column groups using FWHT or a GEMM fallback."""
    assert x.dim() == 2
    m, k = x.shape
    num_groups = k // GROUP_SIZE
    assert k % GROUP_SIZE == 0, (
        f"Last dim ({k}) must be divisible by GROUP_SIZE ({GROUP_SIZE}). "
        f"Got remainder {k % GROUP_SIZE}."
    )
    x_reshaped = x.view(m * num_groups, GROUP_SIZE)
    if fwht:
        out = hadamard_transform(x_reshaped, scale=HAD_SCALE)
    else:
        h = get_hadamard_matrix(x.device)
        if h.dtype != x.dtype:
            h = h.to(dtype=x.dtype)
        out = x_reshaped @ h

    return out.view(m, k)


def apply_fwht_had(x: torch.Tensor) -> torch.Tensor:
    """Rotate 2D weights in 32-column groups; requires fast_hadamard_transform."""
    assert x.dim() == 2
    m, k = x.shape
    num_groups = k // GROUP_SIZE
    assert k % GROUP_SIZE == 0, (
        f"Last dim ({k}) must be divisible by GROUP_SIZE ({GROUP_SIZE}). "
        f"Got remainder {k % GROUP_SIZE}."
    )
    x_reshaped = x.view(m * num_groups, GROUP_SIZE)
    out = hadamard_transform(x_reshaped, scale=HAD_SCALE)
    return out.view(m, k)


def apply_gemm_had(x: torch.Tensor) -> torch.Tensor:
    """Rotate 2D weights in 32-column groups using a normalized Hadamard GEMM."""
    assert x.dim() == 2
    m, k = x.shape
    num_groups = k // GROUP_SIZE
    assert k % GROUP_SIZE == 0, (
        f"Last dim ({k}) must be divisible by GROUP_SIZE ({GROUP_SIZE}). "
        f"Got remainder {k % GROUP_SIZE}."
    )
    x_reshaped = x.view(m * num_groups, GROUP_SIZE)
    h = get_hadamard_matrix(x.device)
    if h.dtype != x.dtype:
        h = h.to(dtype=x.dtype)
    out = x_reshaped @ h
    return out.view(m, k)