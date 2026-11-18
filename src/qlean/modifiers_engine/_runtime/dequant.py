# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

from qlean.utils.upcast_from_mxfp import upcast_from_mxfp

__all__ = [
    "dequantize_fp8_block", "MXFP8_GROUP_SIZE",
    "dequantize_mxfp4", "dequantize_mxfp8", "dequantize_nvfp4", "dequantize_int4",
]

# OCP MX fixes the group size at 32, independent of dequant_block_size.
MXFP8_GROUP_SIZE = 32


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128,
    target_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    assert x.is_contiguous() and s.is_contiguous()
    assert x.dim() == s.dim()
    assert x.dim() >= 2
    M, N = x.shape[-2:]
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE"]),
        triton.cdiv(N, meta["BLOCK_SIZE"]),
    )
    if x.dim() == 2:
        y = torch.empty_like(x, dtype=target_dtype)
        weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    elif x.dim() == 3:
        stack_y = []
        for idx in range(x.shape[0]):
            weight = x[idx]
            scale = s[idx]
            slice_y = torch.empty(*x.shape[-2:], dtype=target_dtype, device=x.device)
            weight_dequant_kernel[grid](weight, scale, slice_y, M, N, BLOCK_SIZE=block_size)
            stack_y.append(slice_y)
        y = torch.stack(stack_y, dim=0)
    else:
        raise ValueError(f"Unsupported tensor dimension: {x.dim()}")
    return y


def dequantize_fp8_block(weight: torch.Tensor, scale: torch.Tensor,
                         block_size: int = 128,
                         target_dtype=torch.bfloat16) -> torch.Tensor:
    return weight_dequant(weight.float(), scale, block_size=block_size,
                          target_dtype=target_dtype)


# The MX kernel needs raw e8m0 bytes; viewing fp32/bf16 scales as uint8 corrupts them.
_E8M0_DTYPES = tuple(
    d for d in (getattr(torch, "float8_e8m0fnu", None), torch.uint8, torch.int8)
    if d is not None
)


def dequantize_mxfp4(weight: torch.Tensor, scale: torch.Tensor,
                     target_dtype=torch.bfloat16) -> torch.Tensor:
    """Upcast packed fp4 pairs with raw e8m0 scale bytes along the last axis."""
    if scale.dtype not in _E8M0_DTYPES:
        raise ValueError(
            f"[mxfp4] scale dtype must be e8m0 or raw bytes, got {scale.dtype}; "
            f"floating-point scales cannot be viewed as uint8 for the MX kernel."
        )
    weight_u8 = weight.view(torch.uint8)
    scale_u8 = scale.view(torch.uint8)
    return upcast_from_mxfp(weight_u8, scale_u8, target_dtype, axis=weight_u8.dim() - 1)


def dequantize_mxfp8(weight: torch.Tensor, scale: torch.Tensor,
                     target_dtype=torch.bfloat16) -> torch.Tensor:
    """Scale shape is [..., M, ceil(K/32)]; independent rows allow M-axis streaming."""
    if scale.dtype not in _E8M0_DTYPES:
        raise ValueError(
            f"[mxfp8] scale dtype must be e8m0 (float8_e8m0fnu or raw int8/uint8 bytes), "
            f"got {scale.dtype}. Floating-point scales cannot be viewed as uint8 for the MX kernel."
        )
    return upcast_from_mxfp(weight, scale.view(torch.uint8), target_dtype,
                            axis=weight.dim() - 1)


_NVFP4_LUT = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


def _dequantize_nvfp4_kernel(x: torch.Tensor, s: torch.Tensor,
                             group_size: int) -> torch.Tensor:
    """Keep local scaling in fp32 until the global scale has been applied."""
    low_nibbles = (x & 0xF).to(torch.int64)
    high_nibbles = ((x >> 4) & 0xF).to(torch.int64)

    lut = _NVFP4_LUT.to(device=x.device, dtype=torch.float32)
    low_vals = lut[low_nibbles]
    high_vals = lut[high_nibbles]

    m, k_packed = x.shape
    k = k_packed * 2
    j_indices = torch.arange(k // 2, device=x.device)
    scale_n_idx = (2 * j_indices) // group_size
    scales = s[:, scale_n_idx]

    y = torch.empty(m, k, dtype=torch.float32, device=x.device)
    y[:, 0::2] = low_vals * scales
    y[:, 1::2] = high_vals * scales
    return y


def dequantize_nvfp4(weight: torch.Tensor, scale_local: torch.Tensor,
                     scale_global: torch.Tensor,
                     group_size: int = 16,
                     target_dtype=torch.bfloat16) -> torch.Tensor:
    """Packed uint8 [M, K/2], local scale [M, K/group_size], scalar global scale."""
    f32 = _dequantize_nvfp4_kernel(weight, scale_local.float(), group_size)
    return (f32 * scale_global.item()).to(target_dtype)


def _unpack_int4(x: torch.Tensor) -> torch.Tensor:
    """int32 [M, K/8], LSB-first uint4b8 (stored value - 8) -> int8 [M, K]."""
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=x.device)
    indices = (x.unsqueeze(-1) >> shifts) & 0xF  # [M, K/8, 8]
    offset = 1 << (4 - 1)
    values = (indices - offset).to(torch.int8)
    m, k_packed, _ = values.shape
    return values.view(m, k_packed * 8)


def dequantize_int4(weight: torch.Tensor, scale: torch.Tensor,
                    group_size: int = 32,
                    target_dtype=torch.bfloat16) -> torch.Tensor:
    """Packed int32 [M, K/8] + scale [M, K/group_size] -> target_dtype [M, K]."""
    x_unpacked = _unpack_int4(weight).to(torch.float32)
    m, k = x_unpacked.shape
    x_grouped = x_unpacked.view(m, -1, group_size)
    s = scale.to(torch.float32).view(m, -1, 1)
    return (x_grouped * s).view(m, k).to(target_dtype)
