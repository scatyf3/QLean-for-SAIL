# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import logging
import math
import pickle
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional

import torch

from qlean.modifiers.utils import weight_quant
from qlean.modifiers_engine._runtime.context import LazyWeight
from qlean.modifiers_engine._runtime.scale_utils import find_scale_name
from qlean.utils.pack_acext_int4 import preprocess_weight_for_a8w4_torch

logger = logging.getLogger(__name__)

# Bump on IR semantic changes to invalidate output fingerprints.
PLAN_IR_VERSION = "1"

# Only row-independent schemes remain bit-exact across arbitrary row chunks.
ROW_WISE_SCHEMES = frozenset({"W8A8", "FP8_DYNAMIC"})


@dataclass(slots=True)
class Plan:
    """The executor consumes ops; on_plan consumes routing tag and opaque meta."""
    ops: list
    tag: str = ""
    meta: dict = field(default_factory=dict)


def _normalize_plan(result) -> Plan:
    if isinstance(result, Plan):
        return result
    if isinstance(result, list):
        return Plan(ops=result)
    raise TypeError(
        f"plan_fn must return Plan or list[op], got {type(result).__name__}"
    )


# ("copy", out_name)
# ("cast", out_name, dtype_str): bf16 or fp32.
# ("passthrough_quant", out_base, src, scale_spec=None)
# ("quant", out_base, scheme, scale_spec=None)
# ("expert_quant", out_base, expert_idx, role, transpose, scheme, scale_spec=None)
# ("quant_pack_acext", out_base, scale_spec=None)
# expert_quant transposes before splitting; role is full, gate (first half), or up (second half).
# scale_spec: None -> "_scale" suffix; leading "."/"_" -> suffix; otherwise a full key.
# Internal-only op: (quant_int4_groupwise, out_base, group_size).
# Fixed CT triplet names; no scale_spec, reshape, or 3D expansion.
OP_TYPES = frozenset({"copy", "cast", "passthrough_quant",
                      "quant", "expert_quant", "quant_pack_acext",
                      "quant_int4_groupwise"})

# Reject sources without implemented passthrough semantics, even if routing allows them.
_PASSTHROUGH_SRC_KIND = {"mxfp4": "mxfp4", "mxfp8": "mxfp8", "fp8_block": "fp8"}


def _quant_scale_name(op: tuple) -> str:
    kind = op[0]
    out_base = op[1]
    if kind == "quant_int4_groupwise":
        return _int4_groupwise_output_names(op)[1]
    if kind == "quant":
        if len(op) not in (3, 4):
            raise ValueError("quant expects (kind, out_base, scheme[, scale_spec])")
        spec = op[3] if len(op) > 3 else None
    elif kind == "quant_pack_acext":
        if len(op) not in (2, 3):
            raise ValueError("quant_pack_acext expects (kind, out_base[, scale_spec])")
        spec = op[2] if len(op) > 2 else None
    elif kind == "expert_quant":
        spec = op[6] if len(op) > 6 else None
    elif kind == "passthrough_quant":
        spec = op[3] if len(op) > 3 else None
    else:
        raise ValueError(
            f"_quant_scale_name expects quant/quant_pack_acext/expert_quant/"
            f"passthrough_quant, got {kind!r}"
        )

    if spec is None:
        return f"{out_base}_scale"
    if spec.startswith(".") or spec.startswith("_"):
        return f"{out_base}{spec}"
    return spec


def _int4_groupwise_weight_quant(
    tensor: torch.Tensor, group_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return INT4 groupwise packed I32 weights, BF16 scales, and a CPU I64 shape."""
    if tensor.dim() != 2:
        raise ValueError(f"INT4 groupwise requires a 2D tensor, got {tuple(tensor.shape)}")
    if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"INT4 groupwise requires bf16/fp16/fp32 input, got {tensor.dtype}")
    if (not isinstance(group_size, int) or isinstance(group_size, bool)
            or (group_size != -1 and group_size <= 0)):
        raise ValueError("INT4 groupwise group_size must be -1 (per-channel) or a positive integer")
    rows, cols = tensor.shape
    if rows == 0 or cols == 0:
        raise ValueError("INT4 groupwise requires nonempty rows and columns")
    group = cols if group_size == -1 else group_size
    if cols % group:
        raise ValueError(f"INT4 groupwise group_size={group_size} must divide input dimension {cols}")
    if cols % 8:
        raise ValueError(f"INT4 groupwise input dimension must be divisible by 8, got {cols}")

    grouped = tensor.float().reshape(rows, cols // group, group)
    scale = (grouped.abs().amax(dim=-1, keepdim=True) / 7.0).clamp_min(
        torch.finfo(torch.float32).eps)
    # Quantize with FP32 scales; cast only stored scales to BF16 to match Kimi outputs.
    codes = (grouped / scale).round().clamp(-8, 7).reshape(rows, cols)
    unsigned = codes.to(torch.int32) + 8
    packed = torch.zeros(rows, cols // 8, dtype=torch.int32, device=tensor.device)
    for i in range(8):
        packed |= unsigned[:, i::8] << (4 * i)
    shape = torch.tensor([rows, cols], dtype=torch.int64, device="cpu")
    return packed.contiguous(), scale.squeeze(-1).to(torch.bfloat16), shape


def _int4_groupwise_output_names(op: tuple) -> list:
    if len(op) != 3:
        raise ValueError("quant_int4_groupwise expects (kind, out_base, group_size)")
    base = op[1]
    if not isinstance(base, str) or not base.endswith(".weight"):
        raise ValueError("quant_int4_groupwise out_base must end with '.weight'")
    return [f"{base}_packed", f"{base}_scale", f"{base}_shape"]


def _int4_channel_weight_quant(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if tensor.dim() != 2:
        raise ValueError(f"int4 channelwise quantization requires 2D, got {tensor.dim()}D")
    if tensor.dtype != torch.float32:
        tensor = tensor.float()

    q_max = 7.0
    q_min = -8.0
    eps = torch.finfo(torch.float32).eps
    abs_max = torch.abs(tensor).max(dim=1, keepdim=True)[0]
    scale = torch.clamp(abs_max / q_max, min=eps)
    quantized = torch.clamp(torch.round(tensor / scale), q_min, q_max)
    return quantized.to(torch.int8), scale.to(torch.float32)


def _int4_channel_weight_quant_mse(
    tensor: torch.Tensor,
    divisor_min: float,
    divisor_max: float,
    divisor_step: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tensor.dim() != 2:
        raise ValueError(f"int4 channelwise quantization requires 2D, got {tensor.dim()}D")
    if divisor_min <= 0 or divisor_max < divisor_min or divisor_step <= 0:
        raise ValueError(
            "Invalid MSE int4 divisor grid: require min > 0, max >= min, step > 0; "
            f"got min={divisor_min}, max={divisor_max}, step={divisor_step}"
        )
    if tensor.dtype != torch.float32:
        tensor = tensor.float()

    q_max = 7.0
    q_min = -8.0
    eps = torch.finfo(torch.float32).eps
    abs_max = torch.abs(tensor).max(dim=1)[0]
    divisors = torch.arange(
        divisor_min, divisor_max + divisor_step, divisor_step,
        device=tensor.device, dtype=torch.float32,
    )
    best_mse = torch.full_like(abs_max, float("inf"))
    best_scale = torch.empty_like(abs_max)

    for divisor in divisors:
        scale = torch.clamp(abs_max / divisor, min=eps)
        scale_expanded = scale.unsqueeze(1)
        quantized = torch.clamp(torch.round(tensor / scale_expanded), q_min, q_max)
        dequantized = quantized * scale_expanded
        mse = ((tensor - dequantized) ** 2).mean(dim=1)
        improved = mse < best_mse
        best_mse = torch.where(improved, mse, best_mse)
        best_scale = torch.where(improved, scale, best_scale)

    scale = best_scale.unsqueeze(1)
    quantized = torch.clamp(torch.round(tensor / scale), q_min, q_max)
    return quantized.to(torch.int8), scale.to(torch.float32)


def _int4_channel_weight_quant_for_cfg(
    tensor: torch.Tensor, cfg: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    if tensor.dim() == 3:
        quant_l, scale_l = [], []
        for idx in range(tensor.shape[0]):
            q, s = _int4_channel_weight_quant_for_cfg(tensor[idx], cfg)
            quant_l.append(q)
            scale_l.append(s)
        return torch.stack(quant_l, dim=0), torch.stack(scale_l, dim=0)
    scale_search = cfg.get("scale_search", "absmax")
    if scale_search == "absmax":
        return _int4_channel_weight_quant(tensor)
    if scale_search != "mse":
        raise ValueError(
            f"Unknown scale_search: {scale_search!r}; allowed: 'absmax' | 'mse'"
        )

    try:
        divisor_min = cfg["mse_divisor_min"]
        divisor_max = cfg["mse_divisor_max"]
        divisor_step = cfg["mse_divisor_step"]
    except KeyError as exc:
        raise ValueError(
            "scale_search='mse' requires transform_cfg fields "
            "mse_divisor_min, mse_divisor_max, mse_divisor_step"
        ) from exc
    return _int4_channel_weight_quant_mse(
        tensor, divisor_min, divisor_max, divisor_step
    )


def _pack_int4_acext(int8_weight: torch.Tensor) -> torch.Tensor:
    """Pack to [..., rows, cols//2] via the pure torch chain in utils/pack_acext_int4.py."""
    weight_shape = int8_weight.shape
    if weight_shape[-1] % 2 != 0:
        raise ValueError(
            f"int4 channelwise quantization requires even number of columns, "
            f"but got shape {weight_shape} with {weight_shape[-1]} columns"
        )
    # Device-agnostic integer chain; pack on the input's device, the save path moves results to CPU.
    packed = preprocess_weight_for_a8w4_torch(int8_weight)
    return packed.view(*weight_shape[:-1], weight_shape[-1] // 2)


def op_output_names(op: tuple) -> list:
    kind = op[0]
    if kind == "quant_int4_groupwise":
        return _int4_groupwise_output_names(op)
    if kind == "copy":
        return [op[1]]
    if kind == "cast":
        return [op[1]]
    return [op[1], _quant_scale_name(op)]


def op_scheme(op: tuple) -> Optional[str]:
    kind = op[0]
    if kind == "quant":
        return op[2]
    if kind == "expert_quant":
        return op[5]
    if kind == "quant_pack_acext":
        return "acext_int4"
    if kind == "quant_int4_groupwise":
        return "int4_groupwise"
    return None


def op_input_view(op: tuple, weight: torch.Tensor) -> torch.Tensor:
    """Share the quantizer input view between execution and verification."""
    kind = op[0]
    if kind == "quant":
        return weight
    if kind in ("quant_pack_acext", "quant_int4_groupwise"):
        return weight
    if kind == "expert_quant":
        _, _, idx, role, transpose = op[:5]
        cur = weight[idx]
        if transpose:
            cur = cur.t()
        if role == "gate":
            cur = cur[: cur.shape[0] // 2, :]
        elif role == "up":
            cur = cur[cur.shape[0] // 2 :, :]
        return cur.contiguous()
    raise ValueError(f"op_input_view expects quant/expert_quant, got {kind!r}")


_CAST_DTYPE_MAP = {"bf16": torch.bfloat16, "fp32": torch.float32}


def execute_op(op: tuple, weight: torch.Tensor, cfg: dict, ctx=None, *,
               src_name: Optional[str] = None) -> list:
    kind = op[0]

    if kind == "copy":
        return [(op[1], weight)]

    if kind == "cast":
        _, out_name, dtype_str = op
        dtype = _CAST_DTYPE_MAP.get(dtype_str)
        if dtype is None:
            raise ValueError(
                f"Unknown cast dtype {dtype_str!r}; "
                f"allowed: {sorted(_CAST_DTYPE_MAP)}"
            )
        return [(out_name, weight.to(dtype))]

    if kind == "passthrough_quant":
        _, out_base, src = op[:3]
        scale_kind = _PASSTHROUGH_SRC_KIND.get(src)
        if scale_kind is None:
            raise ValueError(
                f"passthrough_quant is not implemented for src={src!r}; "
                f"supported: {sorted(_PASSTHROUGH_SRC_KIND)}"
            )
        if ctx is None:
            raise ValueError(
                "passthrough_quant requires ctx to load source scales across shards via ctx.weight_map"
            )
        if src_name is None:
            raise ValueError(
                f"passthrough_quant requires src_name: normalized output key "
                f"{out_base!r} cannot identify the source scale"
            )
        expected_dtypes = ((torch.int8, torch.uint8) if src == "mxfp4" else
                           (torch.float8_e4m3fn, torch.float8_e5m2))
        if weight.dtype not in expected_dtypes:
            raise ValueError(
                f"[passthrough] Source {src_name} declares src={src}, got {weight.dtype}"
            )
        # Resolve scales from source keys, not normalized output keys.
        scale_name = find_scale_name(src_name, ctx.weight_map, kind=scale_kind)
        if scale_name is None:
            raise ValueError(
                f"[passthrough] Source {src_name} has no {scale_kind} scale in weight_map"
            )
        scale = ctx.get_tensor_cpu(scale_name)
        if src in ("mxfp4", "mxfp8"):
            group = 16 if src == "mxfp4" else 32
            expected_shape = list(weight.shape[:-1]) + [math.ceil(weight.shape[-1] / group)]
            e8m0_dtypes = (torch.uint8, torch.int8, getattr(torch, "float8_e8m0fnu", None))
            if scale.dtype not in e8m0_dtypes or list(scale.shape) != expected_shape:
                raise ValueError(
                    f"[passthrough] {scale_name}: expected e8m0 scale with shape "
                    f"{expected_shape}, got {scale.dtype} {list(scale.shape)}"
                )
        elif (scale.dim() != weight.dim() or scale.shape[:-2] != weight.shape[:-2]
              or any(s <= 0 or s > w for s, w in zip(scale.shape[-2:], weight.shape[-2:]))):
            raise ValueError(
                f"[passthrough] {scale_name}: incompatible fp8 block scale shape "
                f"{list(scale.shape)} for {list(weight.shape)}"
            )
        if src == "mxfp4":
            # Reinterpret storage bytes to match MXFP4 output; do not convert values.
            weight = weight.view(torch.uint8)
            scale = scale.view(torch.uint8)
        return [(out_base, weight), (_quant_scale_name(op), scale)]

    if kind == "quant":
        _, out_base, scheme = op[:3]
        scale_name = _quant_scale_name(op)
        if weight.element_size() not in (2, 4):
            raise ValueError(
                f"quant requires a 2- or 4-byte source (bf16/fp16/fp32), got {weight.dtype}"
            )
        if weight.dim() != 2:
            logger.warning(
                f"quant received {weight.dim()}D (shape={list(weight.shape)}); "
                f"passing unchanged to weight_quant"
            )
        qw, qs = weight_quant(op_input_view(op, weight), scheme)
        return [(out_base, qw), (scale_name, qs)]

    if kind == "quant_int4_groupwise":
        names = _int4_groupwise_output_names(op)
        tensors = _int4_groupwise_weight_quant(weight, group_size=op[2])
        return list(zip(names, tensors))

    if kind == "quant_pack_acext":
        _, out_base = op[:2]
        scale_name = _quant_scale_name(op)
        if weight.element_size() not in (2, 4):
            raise ValueError(
                f"quant_pack_acext requires a 2- or 4-byte source (bf16/fp16/fp32), got {weight.dtype}"
            )
        if weight.dim() not in (2, 3):
            raise ValueError(
                f"quant_pack_acext received {weight.dim()}D (shape={list(weight.shape)}); "
                f"only 2D or 3D is supported (3D packs each expert, preserving keys)"
            )
        int8_w, scale = _int4_channel_weight_quant_for_cfg(weight, cfg)
        packed = _pack_int4_acext(int8_w)
        return [(out_base, packed), (scale_name, scale)]

    if kind == "expert_quant":
        _, out_base, _, _, _, scheme = op[:6]
        cur = op_input_view(op, weight)
        if scheme == "acext_int4":
            if cur.element_size() not in (2, 4):
                raise ValueError(
                    f"expert_quant acext_int4 requires a 2- or 4-byte source (bf16/fp16/fp32), "
                    f"got {cur.dtype}"
                )
            int8_w, scale = _int4_channel_weight_quant_for_cfg(cur, cfg)
            qw, qs = _pack_int4_acext(int8_w), scale
        else:
            qw, qs = weight_quant(cur, scheme)
        return [(out_base, qw), (_quant_scale_name(op), qs)]

    raise ValueError(f"Unknown op type '{kind}'; registered: {sorted(OP_TYPES)}")


def _plan_accepts_weight_map(plan_fn: Callable) -> bool:
    try:
        sig = inspect.signature(plan_fn)
    except ValueError:
        return False
    params = sig.parameters
    return (
        "weight_map" in params
        or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    )


def _expected_keys_accepts_weight_map(expected_output_keys: Callable) -> bool:
    try:
        sig = inspect.signature(expected_output_keys)
    except (TypeError, ValueError):
        return False
    params = sig.parameters
    return (
        "weight_map" in params
        or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    )


def make_hooks(plan_fn: Callable, *, on_plan: Optional[Callable] = None,
               verify: Optional[Callable] = None):
    """Callbacks must be pickleable module-level functions or partials for spawn."""
    for fn, label in [(plan_fn, "plan_fn"), (on_plan, "on_plan"),
                      (verify, "verify")]:
        if fn is None:
            continue
        try:
            pickle.dumps(fn)
        except Exception as exc:
            raise TypeError(
                f"{label} must be a pickleable module-level function, "
                f"not a lambda, closure, or instance method. Original error: {exc}"
            ) from exc

    accepts_weight_map = _plan_accepts_weight_map(plan_fn)

    return (
        partial(
            _generic_process_tensor,
            plan_fn=plan_fn,
            on_plan=on_plan,
            accepts_weight_map=accepts_weight_map,
            verify_fn=verify,
        ),
        partial(
            _generic_expected_output_keys,
            plan_fn=plan_fn,
            accepts_weight_map=accepts_weight_map,
        ),
    )


def _generic_process_tensor(
    ctx, file_name, name, weight, *, plan_fn, on_plan, accepts_weight_map: bool,
    verify_fn=None,
):
    kwargs = {"weight_map": ctx.weight_map} if accepts_weight_map else {}
    plan = _normalize_plan(plan_fn(name, list(weight.shape), ctx.transform_cfg, **kwargs))
    if on_plan is not None:
        on_plan(name, plan, ctx)
    if isinstance(weight, LazyWeight):
        if len(plan.ops) == 1 and plan.ops[0][0] in ("copy", "passthrough_quant"):
            weight = ctx.get_tensor_cpu(name)
        else:
            return _stream_row_wise_quant(ctx, name, weight, plan)
    out = []
    for op in plan.ops:
        produced = execute_op(op, weight, ctx.transform_cfg, ctx,
                              src_name=name)
        if verify_fn is not None:
            # Verification must not mutate outputs or abort quantization, even on sidecar I/O failure.
            try:
                verify_fn(op, weight, produced, ctx, file_name, name)
            except Exception:
                logger.exception("[verify] verify_fn failed; skipping verification")
        out.extend(produced)
    return out


def _stream_row_wise_quant(ctx, name, weight: LazyWeight, plan: Plan) -> list:
    if len(plan.ops) != 1:
        raise ValueError(
            f"[chunked] {name} requires exactly 1 op, got {len(plan.ops)}: "
            f"only single-op row-wise quantization is supported."
        )
    op = plan.ops[0]
    if op[0] != "quant":
        raise ValueError(
            f"[chunked] {name} expects quant, got {op[0]!r}: streaming does not support "
            f"copy/cast/passthrough/expert ops."
        )
    _, out_base, scheme = op[:3]
    if scheme not in ROW_WISE_SCHEMES:
        raise ValueError(
            f"[chunked] {name} scheme={scheme!r} is not row-independent "
            f"(allowed: {sorted(ROW_WISE_SCHEMES)}); chunk boundaries can split "
            f"128x128 quantization blocks."
        )
    scale_out = _quant_scale_name(op)

    shape = weight.shape
    if len(shape) != 2:
        raise ValueError(f"[chunked] {name} expects 2D, got shape={shape}")
    M = shape[0]

    # Defer Triton imports until this path is used.
    from qlean.modifiers_engine._runtime.dequant_providers import _target_dtype
    from qlean.modifiers_engine._runtime.dequant import (
        MXFP8_GROUP_SIZE, dequantize_mxfp8,
    )

    src_scale = find_scale_name(name, ctx.weight_map, kind="mxfp8")
    if src_scale is None:
        raise ValueError(
            f"[chunked] {name} has no source mxfp8 scale; expected mxfp8 "
            f"(fp8 weights + e8m0 scales). Check region src/when or source index.json."
        )
    # Check the actual layout before allocation: 2D-block scales cannot be sliced by row.
    expect_scale = [M, math.ceil(shape[1] / MXFP8_GROUP_SIZE)]
    src_scale_shape = list(ctx.get_shape(src_scale))
    if src_scale_shape != expect_scale:
        raise ValueError(
            f"[chunked] {name} source scale {src_scale} shape={src_scale_shape} mismatches mxfp8 "
            f"(expected {expect_scale}, group={MXFP8_GROUP_SIZE}). "
            f"FP8 2D-block scales also span M; chunk boundaries can split scale "
            f"blocks, which streaming does not support."
        )
    # Match the full-table provider's intermediate precision for identical fingerprints.
    target_dtype = _target_dtype(ctx)

    out_buf = None
    scale_buf = None
    chunk_rows = weight.chunk_rows
    for a in range(0, M, chunk_rows):
        b = min(a + chunk_rows, M)
        w = ctx.get_slice(name, a, b)
        # Do not cast source scales to float: the MX kernel consumes raw e8m0 bytes.
        s = ctx.get_slice(src_scale, a, b)
        wf = dequantize_mxfp8(w, s, target_dtype=target_dtype)
        qw, qs = weight_quant(wf, scheme)
        if out_buf is None:
            # Keep only the full CPU output and one chunk's intermediates in memory.
            out_buf = torch.empty(M, qw.shape[1], dtype=qw.dtype)
            scale_buf = torch.empty(M, qs.shape[1], dtype=qs.dtype)
        out_buf[a:b] = qw.to("cpu")
        scale_buf[a:b] = qs.to("cpu")
        del w, s, wf, qw, qs
    return [(out_base, out_buf), (scale_out, scale_buf)]


def _generic_expected_output_keys(
    src_meta, transform_cfg, *, plan_fn, accepts_weight_map: bool,
    weight_map=None,
):
    keys = []
    kwargs = {}
    if accepts_weight_map:
        if weight_map is None:
            weight_map = transform_cfg.get("weight_map")
        if weight_map is not None:
            kwargs["weight_map"] = weight_map
    for name, meta in src_meta.items():
        plan = _normalize_plan(plan_fn(name, meta["shape"], transform_cfg, **kwargs))
        for op in plan.ops:
            keys.extend(op_output_names(op))
    return keys
