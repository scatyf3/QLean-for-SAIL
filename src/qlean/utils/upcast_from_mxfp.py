# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
# Licensed under The MIT License [see LICENSES/MIT.txt for details]

# isort: off
# fmt: off
"""
Self-contained implementation of upcast_from_mxfp.

Extracted from:
  triton/triton/python/triton_kernels/triton_kernels/numerics_details/mxfp.py
  triton/triton/python/triton_kernels/triton_kernels/numerics_details/mxfp_details/_upcast_from_mxfp.py
  triton/triton/python/triton_kernels/triton_kernels/numerics_details/mxfp_details/_downcast_to_mxfp.py
    (MXFP_BLOCK_SIZE only)

NOTE: This version uses plain pointer-based memory access (no TensorDescriptor / TMA),
      so it is compatible with all CUDA GPUs (SM >= 8.0).

Dependencies: torch, triton
"""

import torch
import triton
import triton.language as tl

# =============================================================================
# Constants (shared with downcast)
# =============================================================================

MXFP_BLOCK_SIZE = tl.constexpr(32)

# Element format of the payload, passed to the kernel as SRC_FMT. Bytes always reach the
# kernel as uint8: a float8-typed pointer parameter is an fp8 dtype, which is not
# lowerable on targets without hardware fp8. SRC_FMT says how to bit-decode the bytes.
SRC_FMT_FP4_E2M1 = tl.constexpr(0)
SRC_FMT_FP8_E4M3 = tl.constexpr(1)
SRC_FMT_FP8_E5M2 = tl.constexpr(2)

_DTYPE_TO_SRC_FMT = {
    torch.uint8:         SRC_FMT_FP4_E2M1.value,
    torch.float8_e4m3fn: SRC_FMT_FP8_E4M3.value,
    torch.float8_e5m2:   SRC_FMT_FP8_E5M2.value,
}

# =============================================================================
# Triton kernel  (plain-pointer version, no TMA)
# =============================================================================

# fmt: off
@triton.jit
def _upcast_from_mxfp(
    out_ptr, stride_o_outer, stride_o_quant: tl.constexpr,
    mx_scale_ptr, stride_scale_outer, stride_scale_quant,
    mx_tensor_ptr, stride_tensor_outer, stride_tensor_quant: tl.constexpr,
    outer_dim, quant_dim,
    BLOCK_SIZE_OUT_DIM: tl.constexpr,
    BLOCK_SIZE_QUANT_DIM: tl.constexpr,
    SRC_FMT: tl.constexpr,
):
    """
    Triton kernel: dequantize an MX-format tensor back to float16 / bfloat16 / float32.

    Uses plain pointer loads/stores — no TensorDescriptor/TMA required.
    Compatible with all CUDA GPUs (SM >= 8.0).

    Grid: (ceil(outer_dim / BLOCK_SIZE_OUT_DIM), ceil(quant_dim / BLOCK_SIZE_QUANT_DIM))

    Args:
        out_ptr:            output float tensor pointer
        mx_scale_ptr:       uint8 scale pointer (one byte per 32 elements)
        mx_tensor_ptr:      quantized tensor pointer, always uint8 raw bytes
                              SRC_FMT_FP4_E2M1 -> two e2m1 values packed per byte
                              SRC_FMT_FP8_E4M3 -> one e4m3 value per byte
                              SRC_FMT_FP8_E5M2 -> one e5m2 value per byte
        outer_dim:          size of the outer (non-quantized) dimension
        quant_dim:          logical size of the quantization dimension
        BLOCK_SIZE_OUT_DIM:   tile size along outer dim
        BLOCK_SIZE_QUANT_DIM: tile size along quant dim (must be a multiple of 32)
        SRC_FMT:              one of the SRC_FMT_* codes describing the byte layout
    """
    tl.static_assert(stride_o_quant == 1,
                     "the output must be contiguous in the quant dimension")
    tl.static_assert(BLOCK_SIZE_QUANT_DIM % MXFP_BLOCK_SIZE.value == 0,
                     "BLOCK_SIZE_QUANT_DIM must be a multiple of 32")

    mx_tensor_dtype: tl.constexpr = mx_tensor_ptr.dtype.element_ty
    dst_dtype:       tl.constexpr = out_ptr.dtype.element_ty
    tl.static_assert(dst_dtype == tl.float16 or dst_dtype == tl.bfloat16 or dst_dtype == tl.float32)
    tl.static_assert(mx_tensor_dtype == tl.uint8, "mx_tensor_ptr must be uint8 raw bytes")
    tl.static_assert(mx_scale_ptr.dtype.element_ty == tl.uint8, "mx_scale_ptr must be uint8")
    tl.static_assert(
        SRC_FMT == SRC_FMT_FP4_E2M1 or SRC_FMT == SRC_FMT_FP8_E4M3 or SRC_FMT == SRC_FMT_FP8_E5M2,
        "SRC_FMT must be one of the SRC_FMT_* codes")

    is_fp4: tl.constexpr = SRC_FMT == SRC_FMT_FP4_E2M1
    is_fp8: tl.constexpr = not is_fp4
    K_DIVISOR:                  tl.constexpr = 2 if is_fp4 else 1
    BLOCK_SIZE_QUANT_MX_SCALE:  tl.constexpr = BLOCK_SIZE_QUANT_DIM // MXFP_BLOCK_SIZE
    BLOCK_SIZE_QUANT_MX_TENSOR: tl.constexpr = BLOCK_SIZE_QUANT_DIM // K_DIVISOR

    outer_block = tl.program_id(0).to(tl.int64)
    quant_block = tl.program_id(1).to(tl.int64)

    start_mxt_quant      = quant_block * BLOCK_SIZE_QUANT_MX_TENSOR
    start_out_quant      = quant_block * BLOCK_SIZE_QUANT_DIM
    start_mx_scale_quant = quant_block * BLOCK_SIZE_QUANT_MX_SCALE
    start_out            = outer_block * BLOCK_SIZE_OUT_DIM

    # Advance base pointers to the tile start
    mx_tensor_ptr += start_mxt_quant      * stride_tensor_quant + start_out * stride_tensor_outer
    mx_scale_ptr  += start_mx_scale_quant * stride_scale_quant  + start_out * stride_scale_outer
    out_ptr       += start_out            * stride_o_outer       + start_out_quant * stride_o_quant

    # Offsets and masks
    offs_src_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_TENSOR)[None, :].to(tl.int64)
    offs_out_quant = tl.arange(0, BLOCK_SIZE_QUANT_DIM)[None, :].to(tl.int64)
    offs_outer     = tl.arange(0, BLOCK_SIZE_OUT_DIM)[:, None].to(tl.int64)
    offs_scale     = tl.arange(0, BLOCK_SIZE_QUANT_MX_SCALE)[None, :].to(tl.int64)

    mask_outer     = start_out            + offs_outer     < outer_dim
    mask_out_quant = start_out_quant      + offs_out_quant < quant_dim
    full_mask_out  = mask_out_quant & mask_outer

    mask_src_quant = start_mxt_quant      + offs_src_quant < tl.cdiv(quant_dim, K_DIVISOR)
    full_mask_src  = mask_src_quant & mask_outer

    mask_scale     = start_mx_scale_quant + offs_scale     < tl.cdiv(quant_dim, MXFP_BLOCK_SIZE)
    full_scale_mask = mask_scale & mask_outer

    tensor_offsets = offs_src_quant * stride_tensor_quant + offs_outer * stride_tensor_outer
    scale_offsets  = offs_scale     * stride_scale_quant  + offs_outer * stride_scale_outer
    out_offsets    = offs_out_quant * stride_o_quant       + offs_outer * stride_o_outer

    # Load quantized values and scale
    tensor = tl.load(mx_tensor_ptr + tensor_offsets, mask=full_mask_src)
    scale  = tl.load(mx_scale_ptr  + scale_offsets,  mask=full_scale_mask)

    # Restore scale: uint8 exponent byte -> floating-point scale value
    # bfloat16: exponent at bits[14:7], shift left by 7 into uint16
    # float32/float16: shift left by 23, bitcast to fp32
    if dst_dtype == tl.bfloat16:
        dst_scale = (scale.to(tl.uint16) << 7).to(dst_dtype, bitcast=True)
    else:
        dst_scale = (scale.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
        if dst_dtype == tl.float16:
            dst_scale = dst_scale.to(tl.float16)

    # Upcast quantized values to intermediate dtype
    # For float32 output, use bfloat16 as intermediate to avoid precision issues
    intermediate_dtype: tl.constexpr = tl.bfloat16 if dst_dtype == tl.float32 else dst_dtype

    if is_fp8:
        # Manual e4m3/e5m2 -> float decode. Every finite code is bit-exact vs the fp8 cast.
        src_m_bits: tl.constexpr = 3      if SRC_FMT == SRC_FMT_FP8_E4M3 else 2
        src_bias:   tl.constexpr = 7      if SRC_FMT == SRC_FMT_FP8_E4M3 else 15
        src_exp_all: tl.constexpr = 0xF   if SRC_FMT == SRC_FMT_FP8_E4M3 else 0x1F
        dst_m_bits: tl.constexpr = 7      if intermediate_dtype == tl.bfloat16 else 10
        dst_bias:   tl.constexpr = 127    if intermediate_dtype == tl.bfloat16 else 15
        dst_nan:    tl.constexpr = 0x7FC0 if intermediate_dtype == tl.bfloat16 else 0x7E00
        dst_inf:    tl.constexpr = 0x7F80 if intermediate_dtype == tl.bfloat16 else 0x7C00

        t    = tensor.to(tl.int32)
        sign = (t >> 7) & 0x1
        exp  = (t >> src_m_bits) & src_exp_all
        mant = t & ((1 << src_m_bits) - 1)

        norm = ((exp + (dst_bias - src_bias)) << dst_m_bits) | (mant << (dst_m_bits - src_m_bits))
        mag = norm.to(tl.uint16).to(intermediate_dtype, bitcast=True)

        # exp==0 is a subnormal: mant * 2**(1-bias-m_bits), widened in fp32 where it is exact.
        sub_scale: tl.constexpr = 2.0 ** (1 - src_bias - src_m_bits)
        mag = tl.where(exp == 0,
                       (mant.to(tl.float32) * sub_scale).to(intermediate_dtype), mag)

        # E4M3FN has no inf and one NaN (exp=0xF, mant=0x7); other exp=0xF are finite.
        if SRC_FMT == SRC_FMT_FP8_E4M3:
            mag = tl.where((exp == src_exp_all) & (mant == (1 << src_m_bits) - 1),
                           float("nan"), mag)
        else:
            spec = tl.where(mant == 0, dst_inf, dst_nan).to(tl.uint16).to(intermediate_dtype, bitcast=True)
            mag = tl.where(exp == src_exp_all, spec, mag)

        dst_tensor = tl.where(sign == 1, -mag, mag)
    else:
        # Generic path: manual e2m1 -> float bit manipulation (compatible with all GPUs)
        assert is_fp4
        dst_bias:   tl.constexpr = 127    if intermediate_dtype == tl.bfloat16 else 15
        dst_0p5:    tl.constexpr = 16128  if intermediate_dtype == tl.bfloat16 else 0x3800
        dst_m_bits: tl.constexpr = 7      if intermediate_dtype == tl.bfloat16 else 10
        # Unpack two e2m1 nibbles from one uint8:
        #   low  nibble: bits[2:0] = e2m1 value, bit[3]  = sign
        #   high nibble: bits[6:4] = e2m1 value, bit[7]  = sign
        em0 = tensor & 0x07
        em1 = tensor & 0x70
        x0 = (em0.to(tl.uint16) << (dst_m_bits - 1)) | ((tensor & 0x08).to(tl.uint16) << 12)
        x1 = (em1.to(tl.uint16) << (dst_m_bits - 5)) | ((tensor & 0x80).to(tl.uint16) << 8)
        # Case 1: normal non-zero -> correct exponent bias (e2m1 bias=1, dst bias=15 or 127)
        x0 = tl.where((em0 & 0x06) != 0, x0 + ((dst_bias - 1) << dst_m_bits), x0)
        x1 = tl.where((em1 & 0x60) != 0, x1 + ((dst_bias - 1) << dst_m_bits), x1)
        # Case 2: subnormal (±0.5 in e2m1) -> map directly to ±0.5 in dst type
        x0 = tl.where(em0 == 0x01, dst_0p5 | (x0 & 0x8000), x0)
        x1 = tl.where(em1 == 0x10, dst_0p5 | (x1 & 0x8000), x1)
        # Case 3: zero -> already zero, nothing to do
        dst_tensor = tl.interleave(x0, x1).to(intermediate_dtype, bitcast=True)

    dst_tensor = dst_tensor.to(dst_dtype)

    # Multiply by scale: reshape for group broadcasting (each scale covers 32 values)
    dst_tensor = dst_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, MXFP_BLOCK_SIZE])
    dst_scale  = dst_scale.reshape( [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 1])
    scale      = scale.reshape(dst_scale.shape)

    out_tensor = dst_tensor * dst_scale
    # Clamp to the output dtype's finite range to prevent overflow to inf.
    # Values are aligned with the TMA version of the kernel.
    if dst_dtype == tl.float32:
        _MAX: tl.constexpr = 3.4028234663852886e+38
        out_tensor = tl.minimum(tl.maximum(out_tensor, -_MAX), _MAX)
    elif dst_dtype == tl.bfloat16:
        _MAX: tl.constexpr = 3.3895313892515355e+38
        out_tensor = tl.minimum(tl.maximum(out_tensor, -_MAX), _MAX)
    else:
        tl.static_assert(dst_dtype == tl.float16)
        _MAX: tl.constexpr = 65504.0
        out_tensor = tl.minimum(tl.maximum(out_tensor, -_MAX), _MAX)
    # scale == 0xFF encodes NaN
    out_tensor = tl.where(scale == 0xFF, float("nan"), out_tensor)
    out_tensor = out_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])
    tl.store(out_ptr + out_offsets, out_tensor, mask=full_mask_out)

# =============================================================================
# Triton kernel variant: float32 scale (no E8M0 encoding, zero precision loss)
# =============================================================================

@triton.jit
def _upcast_from_mxfp_fp32_scale(
    out_ptr, stride_o_outer, stride_o_quant: tl.constexpr,
    mx_scale_ptr, stride_scale_outer, stride_scale_quant,
    mx_tensor_ptr, stride_tensor_outer, stride_tensor_quant: tl.constexpr,
    outer_dim, quant_dim,
    BLOCK_SIZE_OUT_DIM: tl.constexpr,
    BLOCK_SIZE_QUANT_DIM: tl.constexpr,
):
    """
    Same as _upcast_from_mxfp but accepts float32 scale directly.
    No E8M0 encoding assumed — scale values are used as-is (zero precision loss).
    """
    tl.static_assert(stride_o_quant == 1,
                     "the output must be contiguous in the quant dimension")
    tl.static_assert(BLOCK_SIZE_QUANT_DIM % MXFP_BLOCK_SIZE.value == 0,
                     "BLOCK_SIZE_QUANT_DIM must be a multiple of 32")

    mx_tensor_dtype: tl.constexpr = mx_tensor_ptr.dtype.element_ty
    dst_dtype:       tl.constexpr = out_ptr.dtype.element_ty
    tl.static_assert(dst_dtype == tl.float16 or dst_dtype == tl.bfloat16 or dst_dtype == tl.float32)
    tl.static_assert(mx_tensor_dtype == tl.uint8, "mx_tensor_ptr must be uint8 (packed e2m1)")
    tl.static_assert(mx_scale_ptr.dtype.element_ty == tl.float32, "mx_scale_ptr must be float32")

    K_DIVISOR:                  tl.constexpr = 2  # e2m1: two values per uint8
    BLOCK_SIZE_QUANT_MX_SCALE:  tl.constexpr = BLOCK_SIZE_QUANT_DIM // MXFP_BLOCK_SIZE
    BLOCK_SIZE_QUANT_MX_TENSOR: tl.constexpr = BLOCK_SIZE_QUANT_DIM // K_DIVISOR

    outer_block = tl.program_id(0).to(tl.int64)
    quant_block = tl.program_id(1).to(tl.int64)

    start_mxt_quant      = quant_block * BLOCK_SIZE_QUANT_MX_TENSOR
    start_out_quant      = quant_block * BLOCK_SIZE_QUANT_DIM
    start_mx_scale_quant = quant_block * BLOCK_SIZE_QUANT_MX_SCALE
    start_out            = outer_block * BLOCK_SIZE_OUT_DIM

    mx_tensor_ptr += start_mxt_quant      * stride_tensor_quant + start_out * stride_tensor_outer
    mx_scale_ptr  += start_mx_scale_quant * stride_scale_quant  + start_out * stride_scale_outer
    out_ptr       += start_out            * stride_o_outer       + start_out_quant * stride_o_quant

    offs_src_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_TENSOR)[None, :].to(tl.int64)
    offs_out_quant = tl.arange(0, BLOCK_SIZE_QUANT_DIM)[None, :].to(tl.int64)
    offs_outer     = tl.arange(0, BLOCK_SIZE_OUT_DIM)[:, None].to(tl.int64)
    offs_scale     = tl.arange(0, BLOCK_SIZE_QUANT_MX_SCALE)[None, :].to(tl.int64)

    mask_outer     = start_out            + offs_outer     < outer_dim
    mask_out_quant = start_out_quant      + offs_out_quant < quant_dim
    full_mask_out  = mask_out_quant & mask_outer

    mask_src_quant = start_mxt_quant      + offs_src_quant < tl.cdiv(quant_dim, K_DIVISOR)
    full_mask_src  = mask_src_quant & mask_outer

    mask_scale     = start_mx_scale_quant + offs_scale     < tl.cdiv(quant_dim, MXFP_BLOCK_SIZE)
    full_scale_mask = mask_scale & mask_outer

    tensor_offsets = offs_src_quant * stride_tensor_quant + offs_outer * stride_tensor_outer
    scale_offsets  = offs_scale     * stride_scale_quant  + offs_outer * stride_scale_outer
    out_offsets    = offs_out_quant * stride_o_quant       + offs_outer * stride_o_outer

    tensor = tl.load(mx_tensor_ptr + tensor_offsets, mask=full_mask_src)
    # Load float32 scale directly — no bitcast needed
    scale  = tl.load(mx_scale_ptr  + scale_offsets,  mask=full_scale_mask)
    dst_scale = scale.to(dst_dtype) if dst_dtype != tl.float32 else scale

    # e2m1 -> float unpacking (same as original kernel)
    intermediate_dtype: tl.constexpr = tl.bfloat16 if dst_dtype == tl.float32 else dst_dtype
    dst_bias:   tl.constexpr = 127    if intermediate_dtype == tl.bfloat16 else 15
    dst_0p5:    tl.constexpr = 16128  if intermediate_dtype == tl.bfloat16 else 0x3800
    dst_m_bits: tl.constexpr = 7      if intermediate_dtype == tl.bfloat16 else 10

    em0 = tensor & 0x07
    em1 = tensor & 0x70
    x0 = (em0.to(tl.uint16) << (dst_m_bits - 1)) | ((tensor & 0x08).to(tl.uint16) << 12)
    x1 = (em1.to(tl.uint16) << (dst_m_bits - 5)) | ((tensor & 0x80).to(tl.uint16) << 8)
    x0 = tl.where((em0 & 0x06) != 0, x0 + ((dst_bias - 1) << dst_m_bits), x0)
    x1 = tl.where((em1 & 0x60) != 0, x1 + ((dst_bias - 1) << dst_m_bits), x1)
    x0 = tl.where(em0 == 0x01, dst_0p5 | (x0 & 0x8000), x0)
    x1 = tl.where(em1 == 0x10, dst_0p5 | (x1 & 0x8000), x1)
    dst_tensor = tl.interleave(x0, x1).to(intermediate_dtype, bitcast=True)
    dst_tensor = dst_tensor.to(dst_dtype)

    # Multiply by float32 scale directly
    dst_tensor = dst_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, MXFP_BLOCK_SIZE])
    dst_scale  = dst_scale.reshape( [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 1])

    out_tensor = dst_tensor * dst_scale
    if dst_dtype == tl.float32:
        _MAX: tl.constexpr = 3.4028234663852886e+38
        out_tensor = tl.minimum(tl.maximum(out_tensor, -_MAX), _MAX)
    elif dst_dtype == tl.bfloat16:
        _MAX: tl.constexpr = 3.3895313892515355e+38
        out_tensor = tl.minimum(tl.maximum(out_tensor, -_MAX), _MAX)
    else:
        tl.static_assert(dst_dtype == tl.float16)
        _MAX: tl.constexpr = 65504.0
        out_tensor = tl.minimum(tl.maximum(out_tensor, -_MAX), _MAX)
    out_tensor = out_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])
    tl.store(out_ptr + out_offsets, out_tensor, mask=full_mask_out)

# =============================================================================
# Public Python API
# =============================================================================

def upcast_from_mxfp(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    target_dtype: torch.dtype,
    axis: int,
):
    """
    Dequantize an mxfp4 or mxfp8 tensor back to float16 / bfloat16 / float32.

    Uses a plain-pointer Triton kernel — no TensorDescriptor/TMA required.
    Compatible with all CUDA GPUs (SM >= 8.0).

    Args:
        tensor:       Quantized tensor.
                        torch.uint8         -> mxfp4 (e2m1, two values per byte)
                        torch.float8_e4m3fn -> mxfp8 (e4m3)
                        torch.float8_e5m2   -> mxfp8 (e5m2)
        scale:        uint8 scale tensor returned by downcast_to_mxfp.
                      Shape is same as original tensor except axis dim = ceil(L/32).
        target_dtype: Output dtype: torch.float16, torch.bfloat16, or torch.float32.
        axis:         The axis along which quantization was applied (supports negative).

    Returns:
        out (torch.Tensor): Dequantized tensor in target_dtype, same shape as the
                            original (pre-quantization) tensor.
    """
    ndim = tensor.ndim
    assert -ndim <= axis < ndim, f"Invalid axis {axis=}"
    axis = axis if axis >= 0 else axis + ndim
    assert tensor.ndim == scale.ndim, (
        f"tensor and scale must have the same number of dimensions. "
        f"Got {tensor.ndim=} and {scale.ndim=}"
    )
    assert tensor.dtype in _DTYPE_TO_SRC_FMT, f"Invalid tensor dtype {tensor.dtype=}"
    assert scale.dtype == torch.uint8,  f"Invalid scale dtype {scale.dtype=}"
    assert target_dtype in (torch.float16, torch.bfloat16, torch.float32), \
        f"Invalid output dtype {target_dtype=}"

    src_fmt = _DTYPE_TO_SRC_FMT[tensor.dtype]
    logical_quant_dim = tensor.shape[axis] * (2 if src_fmt == SRC_FMT_FP4_E2M1.value else 1)
    tensor = tensor.transpose(axis, tensor.ndim - 1).contiguous()
    scale  = scale.transpose(axis, scale.ndim - 1).contiguous()
    # view() requires the last-dim stride of 1 guaranteed by contiguous() above.
    if tensor.dtype != torch.uint8:
        tensor = tensor.view(torch.uint8)

    out = torch.empty((*tensor.shape[:-1], logical_quant_dim),
                      dtype=target_dtype, device=tensor.device)

    if tensor.numel() > 0:
        reshaped_out    = out.view(-1, out.shape[-1])
        reshaped_tensor = tensor.view(-1, tensor.shape[-1])
        reshaped_scale  = scale.view(-1, scale.shape[-1])

        # Performance hyper-parameters (generic path, all GPUs)
        BLOCK_OUT_DIM   = 128
        BLOCK_QUANT_DIM = MXFP_BLOCK_SIZE.value   # = 32

        blocks_out_dim   = triton.cdiv(reshaped_out.shape[0], BLOCK_OUT_DIM)
        blocks_quant_dim = triton.cdiv(reshaped_out.shape[1], BLOCK_QUANT_DIM)

        _upcast_from_mxfp[(blocks_out_dim, blocks_quant_dim)](
            reshaped_out,    *reshaped_out.stride(),
            reshaped_scale,  *reshaped_scale.stride(),
            reshaped_tensor, *reshaped_tensor.stride(),
            *reshaped_out.shape,
            BLOCK_OUT_DIM,
            BLOCK_QUANT_DIM,
            src_fmt,
            num_warps=8,
        )

    out = out.transpose(axis, scale.ndim - 1).contiguous()
    return out

def upcast_from_mxfp_fp32_scale(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    target_dtype: torch.dtype,
    axis: int,
):
    """
    Dequantize an mxfp4 tensor back to float16 / bfloat16 / float32,
    using float32 scale (not E8M0 uint8) — zero precision loss on scale.

    Args:
        tensor:       torch.uint8 tensor — mxfp4 (e2m1, two values per byte).
        scale:        torch.float32 scale tensor (independent float32 values).
                      Shape: same as tensor except axis dim = ceil(L/32).
        target_dtype: Output dtype: torch.float16, torch.bfloat16, or torch.float32.
        axis:         The axis along which quantization was applied (supports negative).

    Returns:
        out (torch.Tensor): Dequantized tensor in target_dtype.
    """
    ndim = tensor.ndim
    assert -ndim <= axis < ndim, f"Invalid axis {axis=}"
    axis = axis if axis >= 0 else axis + ndim
    assert tensor.ndim == scale.ndim, (
        f"tensor and scale must have the same number of dimensions. "
        f"Got {tensor.ndim=} and {scale.ndim=}"
    )
    assert tensor.dtype == torch.uint8, f"Invalid tensor dtype {tensor.dtype=}"
    assert scale.dtype == torch.float32, f"Invalid scale dtype {scale.dtype=}"
    assert target_dtype in (torch.float16, torch.bfloat16, torch.float32), \
        f"Invalid output dtype {target_dtype=}"

    logical_quant_dim = tensor.shape[axis] * 2  # e2m1: two values per uint8
    tensor = tensor.transpose(axis, tensor.ndim - 1).contiguous()
    scale  = scale.transpose(axis, scale.ndim - 1).contiguous()

    out = torch.empty((*tensor.shape[:-1], logical_quant_dim),
                      dtype=target_dtype, device=tensor.device)

    if tensor.numel() > 0:
        reshaped_out    = out.view(-1, out.shape[-1])
        reshaped_tensor = tensor.view(-1, tensor.shape[-1])
        reshaped_scale  = scale.view(-1, scale.shape[-1])

        BLOCK_OUT_DIM   = 128
        BLOCK_QUANT_DIM = MXFP_BLOCK_SIZE.value  # = 32

        blocks_out_dim   = triton.cdiv(reshaped_out.shape[0], BLOCK_OUT_DIM)
        blocks_quant_dim = triton.cdiv(reshaped_out.shape[1], BLOCK_QUANT_DIM)

        _upcast_from_mxfp_fp32_scale[(blocks_out_dim, blocks_quant_dim)](
            reshaped_out,    *reshaped_out.stride(),
            reshaped_scale,  *reshaped_scale.stride(),
            reshaped_tensor, *reshaped_tensor.stride(),
            *reshaped_out.shape,
            BLOCK_OUT_DIM,
            BLOCK_QUANT_DIM,
            num_warps=8,
        )

    out = out.transpose(axis, scale.ndim - 1).contiguous()
    return out
