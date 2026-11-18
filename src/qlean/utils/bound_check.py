# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Core of quantization-error bound checking: recompute the scale from w_ref to
derive an error upper bound, then flag out-of-bound elements.

Shared by the offline CLI (verify_offline.py)(to be merged) and the online verify.py.
INT4 is checked on the absmax path only (quant_mode='int4'); callers must
recognize and skip the mse path. The bound= channel is reserved.

Dependencies: math / typing / torch only, no CLI deps or CUDA side effects.
"""
import math

import torch
import torch.nn.functional as F

# float32 arithmetic noise threshold (worst case ≈ 1.00003)
FP_NOISE_THRESHOLD = 1.0 + 1e-4  # float32 noise tolerance (0.01%)

# Max half-gap per format: bound = scale * half_max_gap
# int4 covers the absmax path only; the mse path is out of scope
HALF_MAX_GAP = {
    'int8': 0.5,
    'fp8': 16.0,
    'mxfp4': 1.0,
    'int4': 0.5,
}


# =============================================================================
# quant-side dequant: restore stored quantized bytes to float32 for comparison with
# w_ref. (unpack_int4_acext below is quant-side; origin-side groupwise INT4 is not.)
# =============================================================================

def _e8m0_to_float32(scale_uint8: torch.Tensor) -> torch.Tensor:
    """Decode an E8M0 uint8 scale to a float32 power-of-2 scale (bit-level cast)."""
    if scale_uint8.dtype != torch.uint8:
        raise TypeError(
            f"_e8m0_to_float32 expects scale_uint8 dtype=torch.uint8, "
            f"got {scale_uint8.dtype}"
        )
    # E8M0 → float32: reinterpret (uint8 << 23) as an IEEE754 float
    int32_bits = scale_uint8.to(torch.int32) << 23
    return int32_bits.view(torch.float32)


def _create_fp8_e4m3_lookup(device='cpu'):
    """Build the 256-entry FP8 E4M3 lookup table."""
    # FP8 E4M3: 1 sign + 4 exponent + 3 mantissa
    # exponent bias = 7
    lookup = []
    for i in range(256):
        sign = (i >> 7) & 0x1
        exp = (i >> 3) & 0xF
        mantissa = i & 0x7

        if exp == 0:
            # subnormal or zero
            if mantissa == 0:
                val = 0.0
            else:
                # subnormal: (-1)^sign * 2^(-6) * (mantissa/8)
                val = (2.0 ** -6) * (mantissa / 8.0)
        elif exp == 0xF and mantissa == 0x7:
            # In E4M3FN only mantissa=7 is NaN; mantissa=0..6 are finite normals.
            val = float('nan')
        else:
            # normal: (-1)^sign * 2^(exp-7) * (1 + mantissa/8)
            val = (2.0 ** (exp - 7)) * (1.0 + mantissa / 8.0)

        if sign:
            val = -val
        lookup.append(val)

    return torch.tensor(lookup, dtype=torch.float32, device=device)


# Global FP8 E4M3 lookup table (lazily initialized)
_FP8_E4M3_LOOKUP = None


def dequantize_int8_per_channel(int8_storage: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Per-channel INT8 dequantization (channel is dim 0)."""
    if scale.dim() == 1 and int8_storage.dim() > 1:
        scale = scale.view(-1, *([1] * (int8_storage.dim() - 1)))
    return int8_storage.float() * scale


def dequantize_fp8_per_channel(fp8_storage: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Per-channel FP8 E4M3 dequantization (channel is dim 0).

    fp8_storage: float8_e4m3fn or uint8 tensor
    scale: float32 per-channel scale, shape [C]
    """
    global _FP8_E4M3_LOOKUP

    device = fp8_storage.device

    if fp8_storage.dtype == torch.float8_e4m3fn:
        # Native PyTorch FP8: cast straight to float32
        fp32 = fp8_storage.to(torch.float32)
    elif fp8_storage.dtype == torch.uint8:
        # uint8 needs the lookup table to decode FP8 E4M3
        if _FP8_E4M3_LOOKUP is None or _FP8_E4M3_LOOKUP.device != device:
            _FP8_E4M3_LOOKUP = _create_fp8_e4m3_lookup(device)
        fp32 = _FP8_E4M3_LOOKUP[fp8_storage.long()]
    else:
        # Other dtypes: try a direct cast
        fp32 = fp8_storage.to(torch.float32)

    # broadcast scale
    if scale.dim() == 1 and fp32.dim() > 1:
        scale = scale.view(-1, *([1] * (fp32.dim() - 1)))
    return fp32 * scale


def dequantize_mxfp4(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if weight.dtype != torch.uint8 or scale.dtype != torch.uint8:
        raise TypeError(
            f"dequantize_mxfp4 expects weight dtype=torch.uint8, scale dtype=torch.uint8, "
            f"got weight={weight.dtype}, scale={scale.dtype}"
        )

    device = weight.device
    *batch_dims, N, packed_K = weight.shape
    K = packed_K * 2

    quant_int = weight.to(torch.int32)
    evens = quant_int & 0xF
    odds = (quant_int >> 4) & 0xF

    vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    lookup_table = torch.tensor(vals + [-v for v in vals], dtype=torch.float32, device=device)
    fp32_tensor = torch.stack([lookup_table[evens], lookup_table[odds]], dim=-1).view(*batch_dims, N, K)

    dq_scale = _e8m0_to_float32(scale)
    padded_K = dq_scale.shape[-1] * 32
    padded_tensor = F.pad(fp32_tensor, (0, padded_K - K))
    padded_tensor = padded_tensor.view(*batch_dims, N, dq_scale.shape[-1], 32)
    out_padded = (padded_tensor * dq_scale.unsqueeze(-1)).view(*batch_dims, N, padded_K)

    return out_padded[..., :K].to(torch.float32)


def dequantize_mxfp4_scale32(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if weight.dtype != torch.uint8 or scale.dtype != torch.float32:
        raise TypeError(
            f"dequantize_mxfp4_scale32 expects weight dtype=torch.uint8, scale dtype=torch.float32, "
            f"got weight={weight.dtype}, scale={scale.dtype}"
        )

    device = weight.device
    *batch_dims, N, packed_K = weight.shape
    K = packed_K * 2

    quant_int = weight.to(torch.int32)
    evens = quant_int & 0xF
    odds = (quant_int >> 4) & 0xF

    vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    lookup_table = torch.tensor(vals + [-v for v in vals], dtype=torch.float32, device=device)
    fp32_tensor = torch.stack([lookup_table[evens], lookup_table[odds]], dim=-1).view(*batch_dims, N, K)

    padded_K = scale.shape[-1] * 32
    padded_tensor = F.pad(fp32_tensor, (0, padded_K - K))
    padded_tensor = padded_tensor.view(*batch_dims, N, scale.shape[-1], 32)
    out_padded = (padded_tensor * scale.unsqueeze(-1)).view(*batch_dims, N, padded_K)

    return out_padded[..., :K].to(torch.float32)


# =============================================================================
# acext INT4 (W4A8, quint4x2 / int8-mma) unpack: inverse of the forward pack swizzle chain.
# Forward pack: utils/pack_acext_int4.py (bit-exact with the old acext ops); keep both in sync.
# =============================================================================

def _acext_tile_read_row(device=None) -> torch.Tensor:
    """Read-order indices of the int8-mma 32-row permutation."""
    i = torch.arange(32, device=device)
    return (i % 8) // 4 * 16 + i // 8 * 4 + i % 4


def _acext_subbyte_transpose(tensor: torch.Tensor) -> torch.Tensor:
    """Nibble-level transpose: (..., R, C/2) -> (..., C, R/2). Self-inverse."""
    rows = tensor.shape[-2]
    if rows % 2 != 0:
        raise ValueError(f"subbyte transpose requires an even -2 dim, got {rows}")
    cols = tensor.shape[-1] * 2
    u = tensor.contiguous().view(torch.uint8)
    even_rows = u[..., 0::2, :]
    odd_rows = u[..., 1::2, :]
    even_out = ((even_rows & 0x0F) | ((odd_rows & 0x0F) << 4)).transpose(-1, -2)
    odd_out = ((even_rows >> 4) | ((odd_rows >> 4) << 4)).transpose(-1, -2)
    transposed = torch.stack((even_out, odd_out), dim=-2)
    return transposed.reshape(*u.shape[:-2], cols, rows // 2).view(torch.int8)


def _acext_undo_permute_b_rows_int8mma(tensor: torch.Tensor) -> torch.Tensor:
    """Undo the int8-mma row permutation applied every 32 rows."""
    rows = tensor.shape[-2]
    if rows % 32 != 0:
        raise ValueError(f"row permutation requires the -2 dim to be a multiple of 32, got {rows}")
    fwd = _acext_tile_read_row(tensor.device)
    inv = torch.empty_like(fwd)
    inv[fwd] = torch.arange(32, device=tensor.device)
    shape = tensor.shape
    tiled = tensor.contiguous().reshape(*shape[:-2], rows // 32, 32, shape[-1])
    return tiled.index_select(-2, inv).reshape(shape)


def _acext_unpack_nibbles_to_int8(packed: torch.Tensor) -> torch.Tensor:
    """Low nibble maps to even columns; sign-extend two's complement to int8 [-8, 7]."""
    p = packed.contiguous().view(torch.uint8)
    low = (p & 0x0F).to(torch.int16)
    high = (p >> 4).to(torch.int16)
    low = torch.where(low > 7, low - 16, low).to(torch.int8)
    high = torch.where(high > 7, high - 16, high).to(torch.int8)
    return torch.stack((low, high), dim=-1).reshape(*p.shape[:-1], 2 * p.shape[-1])


def unpack_int4_acext(packed: torch.Tensor, out_shape) -> torch.Tensor:
    """acext packed INT4 (W4A8): (N, K/2) int8/uint8 -> (N, K) int8 codes in [-8, 7].

    Inverse of the _pack_int4_acext swizzle chain: subbyte transpose → 32-row
    permutation → nibble expansion. Feed the codes to dequantize_int8_per_channel
    to rebuild FP32 weights.

    Must be called on the **whole tensor at once**: the inverse chain mixes rows and
    columns, so chunking rows breaks it. Shape constraints: N % 32 == 0, K % 32 == 0,
    K even.
    """
    if not isinstance(out_shape, (tuple, list)) or len(out_shape) != 2:
        raise ValueError(f"unpack_int4_acext requires a 2D out_shape, got {out_shape!r}")
    n, k = out_shape
    if k % 2 != 0 or n % 32 != 0 or k % 32 != 0:
        raise ValueError(
            f"acext INT4 layout requires N and K to be multiples of 32 and K even, got N={n}, K={k}"
        )
    if packed.dtype not in (torch.int8, torch.uint8):
        raise ValueError(f"acext packed storage must be int8/uint8, got {packed.dtype}")
    if tuple(packed.shape) != (n, k // 2):
        raise ValueError(
            f"acext packed shape must be [N, K/2], expected {(n, k // 2)}, got {tuple(packed.shape)}"
        )
    x = packed.contiguous().view(torch.int8).view(n, k // 2)
    x = _acext_subbyte_transpose(x)
    x = _acext_undo_permute_b_rows_int8mma(x)
    x = _acext_unpack_nibbles_to_int8(x)
    return x.transpose(0, 1).contiguous()


# =============================================================================
# Bound check: derive the theoretical error upper bound from w_ref alone
# =============================================================================

def compute_channel_scale_from_ref(w_ref: torch.Tensor, max_rep: float) -> torch.Tensor:
    """
    Recompute the per-channel scale from w_ref alone (independent of quantizer output).

    Matches the quantizer: the scale floor is float32 eps, so a tiny scale cannot
    underflow FP8/INT8 to zero.

    Args:
        w_ref: original weights
        max_rep: largest representable value of the format (INT8=127, FP8=448)

    Returns:
        scale tensor, shape [C]

    Note: a 3D input (e.g. experts [E, rows, cols]) is flatten(1)-ed and reduced over
    dim 0, yielding a **per-expert** scale [E] instead of the quantizer's
    per-expert-row [E, rows, 1]. Never bound-check a 3D tensor with this function,
    or every element gets falsely flagged.
    """
    eps = torch.finfo(torch.float32).eps
    if w_ref.dim() == 1:
        scale = w_ref.abs().max() / max_rep
    else:
        # per-channel: the channel is dim 0
        scale = w_ref.flatten(1).abs().max(dim=1).values / max_rep
    return torch.clamp(scale, min=eps)


def compute_group_scale_from_ref(w_ref: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """
    Recompute the MXFP4 per-group E8M0 scale from w_ref alone.

    MXFP4 scale = 2^ceil(log2(group_max / 6))

    Args:
        w_ref: original weights, shape [C, K] or higher
        group_size: elements per group (default 32)

    Returns:
        scale tensor, shape [C, ceil(K/group_size)]
    """
    if w_ref.dim() > 2:
        w_ref = w_ref.flatten(1)  # [C, K]

    C, K = w_ref.shape
    # Group the columns; zero-pad a short tail to match the padded_K logic of MXFP4 dequant.
    n_groups = math.ceil(K / group_size)
    padded_K = n_groups * group_size
    if padded_K != K:
        w_ref = F.pad(w_ref, (0, padded_K - K))
    w_grouped = w_ref.view(C, n_groups, group_size)
    group_max = w_grouped.abs().max(dim=2).values  # [C, n_groups]

    # E8M0 scale = 2^ceil(log2(max / 6))
    # E2M1 max value = 6.0
    scale = torch.pow(2.0, torch.ceil(torch.log2(group_max / 6.0 + 1e-15)))
    return scale


def compute_bound_check(w_ref: torch.Tensor, w_recon: torch.Tensor,
                        half_max_gap: float, quant_mode: str = 'int8',
                        group_size: int = 32, return_bound: bool = False,
                        skip_tail_stats: bool = False) -> dict:
    """
    Run the quantization-error bound check.

    Args:
        w_ref: original weights (float32)
        w_recon: dequantized weights (float32)
        half_max_gap: max half-gap of the format
        quant_mode: 'int8', 'fp8', 'mxfp4', 'int4' (int4 covers the absmax path only)
        group_size: MXFP4 group size (default 32)
        return_bound: whether to also return the bound tensor
        skip_tail_stats: keep violations / boundary / max_over_ratio and skip the three
            stats that need a global quantile or max and cannot be merged from chunks.
            Callers chunking along dim 0 (inline verify) must enable it.

    Returns:
        dict with violations, nonfinite, boundary, max_over_ratio. Non-finite
        elements are reported in `nonfinite` and also included in `violations`.
    """
    error = (w_ref - w_recon).abs()

    # Recompute the scale from w_ref, then broadcast it into a same-shape bound
    if quant_mode == 'int8':
        scale = compute_channel_scale_from_ref(w_ref, max_rep=127.0)
    elif quant_mode == 'fp8':
        scale = compute_channel_scale_from_ref(w_ref, max_rep=448.0)
    elif quant_mode == 'mxfp4':
        scale = compute_group_scale_from_ref(w_ref, group_size)
    elif quant_mode == 'int4':
        # absmax path: scale = clamp(absmax/7, eps); the mse path is out of scope
        scale = compute_channel_scale_from_ref(w_ref, max_rep=7.0)
    else:
        raise ValueError(f"Unknown quant_mode: {quant_mode}")

    bound = scale.abs() * half_max_gap

    # broadcast bound to w_ref shape
    if quant_mode == 'mxfp4':
        original_shape = w_ref.shape
        flat_cols = math.prod(original_shape[1:]) if w_ref.dim() > 1 else original_shape[0]
        bound = bound.unsqueeze(-1).expand(-1, -1, group_size).reshape(w_ref.shape[0], -1)
        bound = bound[:, :flat_cols].reshape(original_shape)
    elif bound.dim() < w_ref.dim():
        bound = bound.view(-1, *([1] * (w_ref.dim() - 1)))
        bound = bound.expand(w_ref.shape)

    # NaN makes `>` False, which would silently pass a broken reconstruction.
    evaluable = torch.isfinite(error) & torch.isfinite(bound)
    nonfinite = int((~evaluable).sum().item())

    # Apply the float32 noise threshold to tolerate boundary jitter
    violations = int(
        (evaluable & (error > bound * FP_NOISE_THRESHOLD)).sum().item()
    ) + nonfinite

    boundary = ((error > bound) & (error <= bound * FP_NOISE_THRESHOLD)).sum().item()

    # error/bound ratio stats
    # NaN/inf in ratio would poison max/quantile and the cross-chunk
    # `max(-1.0, NaN) == -1.0` fold. Use in-place `masked_fill_` (no bool-
    # indexed copy) for the hot skip_tail_stats path; take the compressed
    # view only for the quantile branch.
    inv_mask = ~evaluable
    ratio_full = error / bound.clamp(min=1e-15)
    ratio_full.masked_fill_(inv_mask, float('-inf'))
    max_over_ratio = ratio_full.max().item()
    # All-masked chunk → -inf; normalize to the chunked-verifier sentinel.
    if max_over_ratio == float('-inf'):
        max_over_ratio = -1.0

    if skip_tail_stats:
        result = {
            'violations': violations,
            'nonfinite': nonfinite,
            'boundary': boundary,
            'max_over_ratio': max_over_ratio,
            'max_over_count': -1,
            'max_over_pct': -1.0,
            'ratio_p99': -1.0,
        }
        if return_bound:
            return result, bound
        return result

    # Count elements near max_over_ratio (tolerance 0.01); -inf slots don't match.
    max_over_count = (ratio_full >= max_over_ratio - 0.01).sum().item()
    total = w_ref.numel()
    max_over_pct = max_over_count / total * 100 if total > 0 else 0.0

    # kthvalue can't skip -inf; compact the evaluable slice for p99.
    if evaluable.any():
        ratio_compact = ratio_full[evaluable]
        n = ratio_compact.numel()
        if n > 100:
            p99 = ratio_compact.kthvalue(int(n * 0.99)).values.item()
        else:
            p99 = max_over_ratio
        del ratio_compact
    else:
        p99 = max_over_ratio

    result = {
        'violations': violations,
        'nonfinite': nonfinite,
        'boundary': boundary,
        'max_over_ratio': max_over_ratio,
        'max_over_count': max_over_count,
        'max_over_pct': max_over_pct,
        'ratio_p99': p99,
    }
    if return_bound:
        return result, bound
    return result
