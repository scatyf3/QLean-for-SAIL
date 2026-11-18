# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
# Licensed under The MIT License [see LICENSES/MIT.txt for details]

# isort: off
# fmt: off
"""
Self-contained implementation of downcast_to_mxfp.

Extracted from:
  triton/python/triton_kernels/triton_kernels/numerics_details/mxfp.py
  triton/python/triton_kernels/triton_kernels/numerics_details/mxfp_details/_downcast_to_mxfp.py
  triton/python/triton_kernels/triton_kernels/target_info.py

Dependencies: torch, triton
"""

from enum import Enum

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# =============================================================================
# Constants
# =============================================================================

MXFP_BLOCK_SIZE = tl.constexpr(32)

# =============================================================================
# Triton kernel helpers
# =============================================================================

@triton.jit
def _get_max_quant_val(dtype: tl.constexpr):
    if dtype == tl.uint8:
        return 6.0
    elif dtype == tl.float8e5:
        return 57344.0
    elif dtype == tl.float8e4nv:
        return 448.0
    else:
        tl.static_assert(False, f"Invalid {dtype=}")


@triton.jit
def _get_max_power_of_2_quant_val(dtype: tl.constexpr):
    if dtype == tl.uint8:
        return 4.0
    elif dtype == tl.float8e5:
        return 32768.0
    elif dtype == tl.float8e4nv:
        return 256.0


@triton.jit
def _compute_quant_and_scale(src_tensor, valid_src_mask, mx_tensor_dtype: tl.constexpr,
                             DEQUANT_SCALE_ROUNDING_MODE: tl.constexpr = 0):
    """
    Quantize src_tensor to MX format and compute per-group scale.

    Args:
        src_tensor:               2D tile [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM]
        valid_src_mask:           boolean mask of same shape (False for padding)
        mx_tensor_dtype:          target dtype (tl.uint8 for fp4, tl.float8e4nv/float8e5 for fp8)
        DEQUANT_SCALE_ROUNDING_MODE: 0=ROUND_UP, 1=ROUND_DOWN

    Returns:
        out_tensor:   quantized tile
        dequant_scale_exponent: uint8 scale per group (shape [OUT, QUANT/32])
    """
    is_fp8: tl.constexpr = mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5
    BLOCK_SIZE_OUT_DIM: tl.constexpr = src_tensor.shape[0]
    BLOCK_SIZE_QUANT_DIM: tl.constexpr = src_tensor.shape[1]
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = src_tensor.shape[1] // MXFP_BLOCK_SIZE

    # Cast to fp32: most ops unsupported on bfloat16
    f32_tensor = src_tensor.to(tl.float32)
    abs_tensor = tl.abs(f32_tensor)
    # Mask out padding so it doesn't affect scale
    abs_tensor = tl.where(valid_src_mask, abs_tensor, -1.0)
    abs_tensor = tl.reshape(abs_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, MXFP_BLOCK_SIZE])
    max_val = tl.max(abs_tensor, axis=2, keep_dims=True)

    if DEQUANT_SCALE_ROUNDING_MODE == 0:
        # ROUND_UP: 2^ceil(log2(dequant_scale))
        # Adding 0x007FFFFF increments the exponent unless mantissa is all zeros.
        dequant_scale = max_val / _get_max_quant_val(mx_tensor_dtype)
        dequant_scale_exponent = (dequant_scale.to(tl.uint32, bitcast=True) + 0x007FFFFF) & 0x7F800000
    else:
        # ROUND_DOWN: 2^floor(log2(dequant_scale))
        assert DEQUANT_SCALE_ROUNDING_MODE == 1
        dequant_scale = max_val / _get_max_power_of_2_quant_val(mx_tensor_dtype)
        dequant_scale_exponent = dequant_scale.to(tl.uint32, bitcast=True) & 0x7F800000

    dequant_scale_rounded = dequant_scale_exponent.to(tl.float32, bitcast=True)
    quant_scale = tl.where(dequant_scale_rounded == 0, 0, 1.0 / dequant_scale_rounded)

    f32_tensor = tl.reshape(f32_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, MXFP_BLOCK_SIZE])
    quant_tensor = f32_tensor * quant_scale

    # Reshape back and zero out padding
    quant_tensor = quant_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])
    quant_tensor = tl.where(valid_src_mask, quant_tensor, 0)
    dequant_scale_exponent = dequant_scale_exponent.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE])

    # Store only the exponent byte of the fp32 scale
    dequant_scale_exponent = (dequant_scale_exponent >> 23).to(tl.uint8)

    # Convert to target format
    if is_fp8:
        out_tensor = quant_tensor.to(mx_tensor_dtype)
    else:
        # Generic path: manual fp32 -> e2m1 bit manipulation + RTNE rounding
        quant_tensor = quant_tensor.to(tl.uint32, bitcast=True)
        signs = quant_tensor & 0x80000000
        exponents = (quant_tensor >> 23) & 0xFF
        mantissas_orig = (quant_tensor & 0x7FFFFF)

        E8_BIAS = 127
        E2_BIAS = 1
        is_subnormal = exponents < E8_BIAS
        adjusted_exponents = tl.core.sub(E8_BIAS, exponents + 1, sanitize_overflow=False)
        mantissas_pre = (0x400000 | (mantissas_orig >> 1))
        mantissas = tl.where(is_subnormal, mantissas_pre >> adjusted_exponents, mantissas_orig)

        exponents = tl.maximum(exponents, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)

        m2bits = mantissas >> 21
        lsb_keep = (m2bits >> 1) & 0x1
        guard = m2bits & 0x1
        IS_SRC_FP32: tl.constexpr = src_tensor.dtype == tl.float32
        if IS_SRC_FP32:
            bit0_dropped = (mantissas_orig & 0x1) != 0
            mask = (1 << tl.minimum(adjusted_exponents, 31)) - 1
            dropped_post = (mantissas_pre & mask) != 0
            sticky = is_subnormal & (bit0_dropped | dropped_post)
            sticky |= ((mantissas & 0x1FFFFF) != 0).to(tl.uint32)
        else:
            sticky = ((mantissas & 0x1FFFFF) != 0).to(tl.uint32)
        round_inc = guard & (sticky | lsb_keep)
        e2m1_tmp = tl.minimum((((exponents << 2) | m2bits) + round_inc) >> 1, 0x7)
        e2m1_value = ((signs >> 28) | e2m1_tmp).to(tl.uint8)

        # Pack two 4-bit e2m1 values into one uint8: evens in low nibble, odds in high nibble
        e2m1_value = tl.reshape(e2m1_value, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM // 2, 2])
        evens, odds = tl.split(e2m1_value)
        out_tensor = evens | (odds << 4)

    return out_tensor, dequant_scale_exponent


@triton.jit
def _downcast_to_mxfp(
    mx_tensor_ptr, stride_mxt_outer, stride_mxt_quant: tl.constexpr,
    mx_scale_ptr, stride_mx_scale_outer, stride_mx_scale_quant,
    src_ptr, stride_src_outer, stride_src_quant, outer_dim, quant_dim,
    BLOCK_SIZE_OUT_DIM: tl.constexpr,
    BLOCK_SIZE_QUANT_DIM: tl.constexpr,
    DEQUANT_SCALE_ROUNDING_MODE: tl.constexpr,
):
    """
    Triton kernel: quantize a 2D float tensor to MX format.

    Grid: (ceil(outer_dim / BLOCK_SIZE_OUT_DIM), ceil(quant_dim / BLOCK_SIZE_QUANT_DIM))

    Args:
        mx_tensor_ptr:    output quantized tensor pointer (uint8 for fp4, float8 for fp8)
        mx_scale_ptr:     output scale tensor pointer (uint8, one value per 32 elements)
        src_ptr:          input float tensor pointer (bfloat16 / float16 / float32)
        outer_dim:        size of the outer (non-quantized) dimension
        quant_dim:        size of the quantization dimension (must be padded to multiple of 32)
        BLOCK_SIZE_OUT_DIM:   tile size along outer dim
        BLOCK_SIZE_QUANT_DIM: tile size along quant dim (must be multiple of 32)
        DEQUANT_SCALE_ROUNDING_MODE: 0=ROUND_UP, 1=ROUND_DOWN
    """
    tl.static_assert(stride_mxt_quant == 1, f"Output stride, {stride_mxt_quant=} must be 1.")
    tl.static_assert(BLOCK_SIZE_QUANT_DIM % MXFP_BLOCK_SIZE == 0,
                     f"{BLOCK_SIZE_QUANT_DIM=} must be a multiple of 32")

    # uint8 means two fp4 e2m1 values are packed into one byte
    mx_tensor_dtype: tl.constexpr = mx_tensor_ptr.dtype.element_ty
    tl.static_assert(
        mx_tensor_dtype == tl.uint8 or (mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5),
        f"Invalid {mx_tensor_dtype=}. Must be uint8 or float8.")

    src_dtype: tl.constexpr = src_ptr.dtype.element_ty
    tl.static_assert(mx_scale_ptr.dtype.element_ty == tl.uint8,
                     f"{mx_scale_ptr.dtype.element_ty=} must be uint8")
    tl.static_assert(
        (src_dtype == tl.bfloat16) or (src_dtype == tl.float16) or (src_dtype == tl.float32),
        f"{src_dtype=} must be bfloat16 or float16 or float32")

    is_fp4: tl.constexpr = mx_tensor_dtype == tl.uint8

    outer_block = tl.program_id(0).to(tl.int64)
    quant_block = tl.program_id(1).to(tl.int64)

    K_DIVISOR: tl.constexpr = 2 if is_fp4 else 1
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = BLOCK_SIZE_QUANT_DIM // MXFP_BLOCK_SIZE
    BLOCK_SIZE_QUANT_MX_TENSOR: tl.constexpr = BLOCK_SIZE_QUANT_DIM // K_DIVISOR

    start_src_quant = quant_block * BLOCK_SIZE_QUANT_DIM
    start_mx_scale_quant = quant_block * BLOCK_SIZE_QUANT_MX_SCALE
    start_mx_quant = quant_block * BLOCK_SIZE_QUANT_MX_TENSOR
    start_out = outer_block * BLOCK_SIZE_OUT_DIM

    src_ptr += start_src_quant * stride_src_quant + start_out * stride_src_outer
    mx_scale_ptr += start_mx_scale_quant * stride_mx_scale_quant + start_out * stride_mx_scale_outer
    mx_tensor_ptr += start_mx_quant * stride_mxt_quant + start_out * stride_mxt_outer

    offs_src_quant = tl.arange(0, BLOCK_SIZE_QUANT_DIM)[None, :].to(tl.int64)
    offs_mxt_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_TENSOR)[None, :].to(tl.int64)
    offs_scale_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_SCALE)[None, :].to(tl.int64)
    offs_outer = tl.arange(0, BLOCK_SIZE_OUT_DIM)[:, None].to(tl.int64)

    mask_src_quant = start_src_quant + offs_src_quant < quant_dim
    mask_n = start_out + offs_outer < outer_dim
    full_mask_src = mask_src_quant & mask_n

    mask_mxt_quant = start_mx_quant + offs_mxt_quant < quant_dim // K_DIVISOR
    full_mask_mxt = mask_mxt_quant & mask_n

    scale_mask_k = start_mx_scale_quant + offs_scale_quant < quant_dim // MXFP_BLOCK_SIZE
    full_scale_mask = scale_mask_k & mask_n

    src_tensor_offsets = offs_src_quant * stride_src_quant + offs_outer * stride_src_outer
    mx_scale_offsets = offs_scale_quant * stride_mx_scale_quant + offs_outer * stride_mx_scale_outer
    mx_tensor_offsets = offs_mxt_quant * stride_mxt_quant + offs_outer * stride_mxt_outer

    src_tensor = tl.load(src_ptr + src_tensor_offsets, mask=full_mask_src)

    out_tensor, scale_tensor = _compute_quant_and_scale(
        src_tensor, full_mask_src, mx_tensor_dtype, DEQUANT_SCALE_ROUNDING_MODE)

    tl.store(mx_scale_ptr + mx_scale_offsets, scale_tensor, mask=full_scale_mask)
    tl.store(mx_tensor_ptr + mx_tensor_offsets, out_tensor, mask=full_mask_mxt)


# =============================================================================
# Public Python API
# =============================================================================

class DequantScaleRoundingMode(Enum):
    # 2^round_up(log2(max/max_q)) — avoids clipping the max value
    ROUND_UP = 0
    # 2^round_down(log2(max/max_power_of_2_q)) — follows OCP standard,
    # ~50% chance of clipping the max value
    ROUND_DOWN = 1


def downcast_to_mxfp(
    src_tensor: torch.Tensor,
    out_quant_type: torch.dtype,
    axis: int,
    DEQUANT_SCALE_ROUNDING_MODE: DequantScaleRoundingMode = DequantScaleRoundingMode.ROUND_UP,
):
    """
    Convert a float tensor to MX format (mxfp4 or mxfp8) using a Triton kernel.

    Quantization is applied along `axis`. Each group of 32 consecutive elements
    along that axis shares one uint8 scale (storing the fp32 exponent byte).

    Args:
        src_tensor:   Input tensor, dtype must be float32 / float16 / bfloat16, must be on CUDA.
        out_quant_type:
            torch.uint8         -> mxfp4 (e2m1), two values packed per byte;
                                   axis dimension size will be halved.
            torch.float8_e4m3fn -> mxfp8 (e4m3)
            torch.float8_e5m2   -> mxfp8 (e5m2)
        axis:         Quantization axis (supports negative indexing).
        DEQUANT_SCALE_ROUNDING_MODE:
            ROUND_UP   (default) — scale exponent rounded up, never clips max value.
            ROUND_DOWN           — follows OCP MX spec, may clip ~50% of the time.

    Returns:
        out_quant_tensor (torch.Tensor): Quantized tensor.
            - mxfp4: dtype=torch.uint8, shape along `axis` is halved.
            - mxfp8: dtype=out_quant_type, same shape as src_tensor.
        out_scale (torch.Tensor): uint8 scale tensor, shape same as src_tensor
            except the axis dimension is ceil(L / 32).
    """
    ndim = src_tensor.ndim
    assert -ndim <= axis < ndim, f"Invalid axis {axis=}"
    axis = axis if axis >= 0 else axis + ndim

    src_tensor = src_tensor.transpose(axis, src_tensor.ndim - 1)

    is_fp4 = out_quant_type == torch.uint8
    is_fp8 = out_quant_type in (torch.float8_e4m3fn, torch.float8_e5m2)
    assert is_fp4 or is_fp8, f"out_quant_type must be uint8, float8_e4m3fn, or float8_e5m2"

    divisor = 2 if is_fp4 else 1
    L = src_tensor.shape[-1]
    if is_fp4:
        assert L % 2 == 0, f"axis dim must be divisible by 2 for e2m1. Got {L}"

    # Pad last dim to a multiple of MXFP_BLOCK_SIZE (32)
    padded_L = triton.cdiv(L, MXFP_BLOCK_SIZE.value) * MXFP_BLOCK_SIZE.value
    needs_padding = padded_L != L
    out_shape_padded = src_tensor.shape[:-1] + (padded_L // divisor,)
    out_scale_shape = src_tensor.shape[:-1] + (triton.cdiv(L, MXFP_BLOCK_SIZE),)

    out_quant_tensor = src_tensor.new_empty(out_shape_padded, dtype=out_quant_type)
    out_scale = src_tensor.new_empty(out_scale_shape, dtype=torch.uint8)

    if src_tensor.numel() > 0:
        src_tensor_padded = F.pad(src_tensor, (0, padded_L - L)) if needs_padding else src_tensor
        kernel_src_tensor = src_tensor_padded.reshape(-1, src_tensor_padded.shape[-1])
        kernel_quant_tensor = out_quant_tensor.view(-1, out_quant_tensor.shape[-1])
        kernel_scale = out_scale.view(-1, out_scale.shape[-1])

        # Tile sizes (performance hyper-parameters, generic path)
        BLOCK_OUT_DIM = 32
        BLOCK_QUANT_DIM = MXFP_BLOCK_SIZE.value * 4  # = 128
        NUM_WARPS = 4 if src_tensor.dtype == torch.float32 else 8

        blocks_out_dim = triton.cdiv(kernel_src_tensor.shape[0], BLOCK_OUT_DIM)
        blocks_quant_dim = triton.cdiv(kernel_src_tensor.shape[1], BLOCK_QUANT_DIM)

        _downcast_to_mxfp[(blocks_out_dim, blocks_quant_dim)](
            kernel_quant_tensor, *kernel_quant_tensor.stride(),
            kernel_scale, *kernel_scale.stride(),
            kernel_src_tensor, *kernel_src_tensor.stride(), *kernel_src_tensor.shape,
            BLOCK_OUT_DIM,
            BLOCK_QUANT_DIM,
            DEQUANT_SCALE_ROUNDING_MODE.value,
            num_warps=NUM_WARPS,
        )

        if needs_padding:
            out_quant_tensor = out_quant_tensor[..., :(L // divisor)]

    out_quant_tensor = out_quant_tensor.transpose(axis, src_tensor.ndim - 1)
    out_scale = out_scale.transpose(axis, src_tensor.ndim - 1)
    return out_quant_tensor, out_scale