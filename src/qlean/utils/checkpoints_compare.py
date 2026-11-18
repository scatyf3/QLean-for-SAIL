#!/usr/bin/env python3
# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Quantization-error analysis tool: compare origin weights against a quantized dir.

Origin: *.weight (uint8 as fp8_e4m3) + *.weight_scale_inv (blockwise FP8-E4M3, block=128), or BF16.
Quant:  *.weight (int8/uint8) + *.weight_scale (per-channel INT8 / FP8 / MXFP4).
Supported modes are listed under --mode; results are printed and written to a CSV.
"""
import os
import glob
import csv
import math
import argparse
import torch
import torch.nn.functional as F
from typing import Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from safetensors.torch import load_file
from tqdm import tqdm
torch.set_printoptions(edgeitems=4, linewidth=250)

if not hasattr(torch, 'float8_e4m3fn'):
    raise RuntimeError("PyTorch >= 2.1 is required for torch.float8_e4m3fn")

def dequantize_fp8_blockwise(fp8_storage: torch.Tensor, scale_inv: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """Blockwise FP8 dequantization; treats the tensor as a flat run of block_size^2-element blocks.

    fp8_storage: uint8/int8 raw storage (any dim).
    scale_inv: float reciprocal scale, length total_elements // block_size^2.
    """
    assert fp8_storage.numel() % block_size == 0, (
        f"Total elements {fp8_storage.numel()} not divisible by block_size={block_size}; check block size or padding"
    )

    # 1. reinterpret -> float32
    fp32_vals = fp8_storage.view(torch.float8_e4m3fn).to(torch.float32)

    # 2. flatten and regroup into blocks
    original_shape = fp32_vals.shape
    n_blocks = fp32_vals.numel() // block_size // block_size

    w_blocked = fp32_vals.view(n_blocks, block_size, block_size)
    s_blocked = scale_inv.float().view(n_blocks, 1, 1)

    # 3. per-block dequant, then restore the original shape
    return (w_blocked * s_blocked).view(original_shape)

def dequantize_fp8_blockwise_2d(fp8_weight: torch.Tensor, scale_inv: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """2D blockwise FP8-E4M3 dequantization (128x128 blocks).

    fp8_weight: float8_e4m3fn tensor of shape [H, W].
    scale_inv: float32 scale tensor of shape [H//block_size, W//block_size],
               one value per 128x128 block.

    Returns the dequantized FP32 tensor, same shape as fp8_weight.
    """
    assert fp8_weight.dtype == torch.float8_e4m3fn, f"Expected float8_e4m3fn, got {fp8_weight.dtype}"
    assert len(fp8_weight.shape) == 2, f"Currently only support 2D weight tensors, got shape {fp8_weight.shape}"

    original_shape = fp8_weight.shape
    H, W = original_shape

    # 1. convert to float32
    w_fp32 = fp8_weight.to(torch.float32)

    # 2. block count along each dimension
    num_blocks_h = H // block_size
    num_blocks_w = W // block_size

    # 3. validate scale_inv shape: expected [num_blocks_h, num_blocks_w]
    assert scale_inv.shape >= (num_blocks_h, num_blocks_w), (
        f"Expected scale_inv shape [{num_blocks_h}, {num_blocks_w}], got {scale_inv.shape}, weight {H}, {W}"
    )

    # 4. reshape weight to [num_blocks_h, block_size, num_blocks_w, block_size],
    #    then transpose to [num_blocks_h, num_blocks_w, block_size, block_size]
    w_blocked = fp8_weight.view(torch.float8_e4m3fn)
    w_blocked = w_blocked.view(num_blocks_h, block_size, num_blocks_w, block_size)
    w_blocked = w_blocked.permute(0, 2, 1, 3).contiguous()  # [num_blocks_h, num_blocks_w, block_size, block_size]

    # 5. expand scale_inv to [num_blocks_h, num_blocks_w, 1, 1]
    s_blocked = scale_inv.float().unsqueeze(-1).unsqueeze(-1)  # [num_blocks_h, num_blocks_w, 1, 1]
    s_blocked = s_blocked[:num_blocks_h]

    # 6. per-block dequantization
    dequant_blocked = w_blocked.to(torch.float32) * s_blocked

    # 7. restore the original shape
    dequant = dequant_blocked.permute(0, 2, 1, 3).contiguous().view(original_shape)
    return dequant

def dequantize_int8_per_channel(int8_storage: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Per-channel INT8 dequantization (channel assumed to be dim 0)."""
    if scale.dim() == 1 and int8_storage.dim() > 1:
        scale = scale.view(-1, *([1] * (int8_storage.dim() - 1)))
    return int8_storage.float() * scale

def dequantize_mxfp4(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """MXFP4 dequantization: weight packs two E2M1 nibbles per uint8; scale is E8M0 (uint8, one per 32 elems)."""
    assert weight.dtype == torch.uint8 and scale.dtype == torch.uint8

    device = weight.device
    *batch_dims, N, packed_K = weight.shape
    K = packed_K * 2

    quant_int = weight.to(torch.int32)
    evens = quant_int & 0xF
    odds = (quant_int >> 4) & 0xF

    vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    lookup_table = torch.tensor(vals + [-v for v in vals], dtype=torch.float32, device=device)
    fp32_tensor = torch.stack([lookup_table[evens], lookup_table[odds]], dim=-1).view(*batch_dims, N, K)

    dq_scale = (scale.to(torch.int32) << 23).view(torch.float32)
    padded_K = dq_scale.shape[-1] * 32
    padded_tensor = F.pad(fp32_tensor, (0, padded_K - K))
    padded_tensor = padded_tensor.view(*batch_dims, N, dq_scale.shape[-1], 32)
    # print(padded_tensor.shape, "e8m0_padded_tensor")
    # print(padded_tensor[0,:6,:])
    out_padded = (padded_tensor * dq_scale.unsqueeze(-1)).view(*batch_dims, N, padded_K)

    return out_padded[..., :K].to(torch.float32)

def dequantize_mxfp4_scale32(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """MXFP4 dequantization with a float32 scale (one per 32 elems) instead of E8M0."""
    assert weight.dtype == torch.uint8 and scale.dtype == torch.float

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
    # print(padded_tensor.shape, "e8m0_padded_tensor")
    # print(padded_tensor[0,:6,:])
    out_padded = (padded_tensor * scale.unsqueeze(-1)).view(*batch_dims, N, padded_K)

    return out_padded[..., :K].to(torch.float32)

def dequantize_int8_channelwise(int8_weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Channelwise INT8 dequantization (channel assumed to be dim 0).

    int8_weight: int8/uint8 tensor of any dim.
    scale: float32 per-channel scale, shape [C] or [C, 1, ...]; for a 2D weight
           [out_features, in_features] it is [out_features].

    Returns the dequantized FP32 tensor.
    """
    assert int8_weight.dtype in [torch.int8, torch.uint8], f"Expected int8/uint8, got {int8_weight.dtype}"

    # scale broadcast: channel assumed to be dim 0
    if scale.dim() == 1 and int8_weight.dim() > 1:
        scale = scale.view(-1, *([1] * (int8_weight.dim() - 1)))
    elif scale.dim() < int8_weight.dim():
        # scale already has the right dims [C, 1, 1, ...]
        pass

    return int8_weight.float() * scale.float()

def compute_metrics(w_ref: torch.Tensor, w_recon: torch.Tensor):
    """Compute quantization-error metrics between w_ref and w_recon.

    Returns a dict of mse/mae/max_err/cosine_sim/rel_l2/sqnr_db/p99_err, plus
    psnr_db and the signal/noise/peak powers and numel used to derive them.
    """
    diff = w_ref - w_recon
    n = w_ref.numel()

    # Basic error metrics
    mse = (diff ** 2).mean().item()
    mae = diff.abs().mean().item()
    max_err = diff.abs().max().item()
    p99_err = diff.abs().flatten().kthvalue(int(n * 0.99)).values.item() if n > 100 else max_err

    # Relative metrics
    ref_norm = w_ref.norm(p=2).item()
    diff_norm = diff.norm(p=2).item()
    rel_l2 = diff_norm / (ref_norm + 1e-15)

    # Cosine similarity (direction preservation)
    w_ref_flat = w_ref.flatten()
    w_recon_flat = w_recon.flatten()
    dot_product = torch.dot(w_ref_flat, w_recon_flat).item()
    ref_mag = w_ref_flat.norm(p=2).item()
    recon_mag = w_recon_flat.norm(p=2).item()
    cosine_sim = dot_product / (ref_mag * recon_mag + 1e-15)

    # SQNR
    signal_power = (w_ref ** 2).sum().item()
    noise_power = (diff ** 2).sum().item()
    sqnr = 10.0 * math.log10(signal_power / (noise_power + 1e-15)) if noise_power > 1e-15 else float('inf')

    # PSNR
    peak_power = (w_ref ** 2).max().item()
    psnr = 10.0 * math.log10(peak_power / (mse + 1e-15)) if peak_power > 1e-15 else float('inf')

    return {
        'mse': mse,
        'mae': mae,
        'max_err': max_err,
        'cosine_sim': cosine_sim,
        'rel_l2': rel_l2,
        'sqnr_db': sqnr,
        'p99_err': p99_err,
        'signal_power': signal_power,
        'noise_power': noise_power,
        'peak_power': peak_power,
        'psnr_db': psnr,
        'numel': n
    }


def process_single_tensor(key, origin_tensors, quant_tensors, mode, block_size, device_id=0):
    """Process one tensor and compute its metrics; thread-safe worker for parallel execution.

    Returns (key, result_dict, None) on success, or (key, None, reason) on skip/error.
    """
    try:
        scale_inv_key = f"{key}_scale_inv"

        if mode == 'fp8_dequant':
            if scale_inv_key not in origin_tensors or key not in quant_tensors:
                return (key, None, "missing quant")
            w_ref = dequantize_fp8_blockwise_2d(origin_tensors[key][1], origin_tensors[scale_inv_key][1], block_size)
            w_recon = quant_tensors[key][1].float()
        elif mode == 'fp8_channel_int8':
            channel_scale_key = f"{key}_scale"
            if scale_inv_key not in origin_tensors or key not in quant_tensors or channel_scale_key not in quant_tensors:
                return (key, None, "missing scale")
            w_ref = dequantize_fp8_blockwise_2d(origin_tensors[key][1], origin_tensors[scale_inv_key][1], block_size)
            w_recon = dequantize_int8_channelwise(quant_tensors[key][1], quant_tensors[channel_scale_key][1])
        elif mode == "bf16_channel_int8":
            channel_scale_key = f"{key}_scale"
            if key not in quant_tensors or channel_scale_key not in quant_tensors:
                return (key, None, "missing bf16 scale")
            w_ref = origin_tensors[key][1].float()
            w_recon = dequantize_int8_per_channel(quant_tensors[key][1], quant_tensors[channel_scale_key][1])
        elif mode == 'fp8':
            quant_scale_key = f"{key}_scale"
            if scale_inv_key not in origin_tensors or key not in quant_tensors or quant_scale_key not in quant_tensors:
                return (key, None, "missing fp8 scale")
            w_ref = dequantize_fp8_blockwise(origin_tensors[key][1], origin_tensors[scale_inv_key][1], block_size)
            w_recon = dequantize_int8_per_channel(quant_tensors[key][1], quant_tensors[quant_scale_key][1])
        elif mode == 'bf16_mxfp4':
            quant_scale_key = f"{key}_scale"
            if key not in quant_tensors or quant_scale_key not in quant_tensors:
                return (key, None, "missing bf16 mxfp4 scale")
            w_ref = origin_tensors[key][1].float()
            w_data = quant_tensors[key][1]
            scale_data = quant_tensors[quant_scale_key][1]
            if device_id >= 0:
                device = f'cuda:{device_id}'
                w_ref = w_ref.to(device)
                w_recon = dequantize_mxfp4(w_data.to(device), scale_data.to(device))
                # print("key:", key)
                # print('w_ref:', w_ref[0, :32])
                # print('w_recon:', w_recon[0, :32])
            else:
                w_recon = dequantize_mxfp4(w_data, scale_data)
        elif mode == 'bf16_mxfp4_scale32':
            quant_scale_key = f"{key}_scale"
            if key not in quant_tensors or quant_scale_key not in quant_tensors:
                return (key, None, "missing bf16 mxfp4_scale32 scale")
            w_ref = origin_tensors[key][1].float()
            w_data = quant_tensors[key][1]
            scale_data = quant_tensors[quant_scale_key][1]
            if device_id >= 0:
                device = f'cuda:{device_id}'
                w_ref = w_ref.to(device)
                w_recon = dequantize_mxfp4_scale32(w_data.to(device), scale_data.to(device))
            else:
                w_recon = dequantize_mxfp4_scale32(w_data, scale_data)
        elif mode == 'fp8_mxfp4':
            quant_scale_key = f"{key}_scale"
            if scale_inv_key not in origin_tensors or key not in quant_tensors or quant_scale_key not in quant_tensors:
                return (key, None, "missing fp8 mxfp4 data")
            # ref: FP8 blockwise dequant -> bf16
            w_ref = dequantize_fp8_blockwise_2d(origin_tensors[key][1], origin_tensors[scale_inv_key][1], block_size)
            # recon: MXFP4 dequant
            w_data = quant_tensors[key][1]
            scale_data = quant_tensors[quant_scale_key][1]
            if device_id >= 0:
                device = f'cuda:{device_id}'
                w_ref = w_ref.to(device)
                w_recon = dequantize_mxfp4(w_data.to(device), scale_data.to(device))
            else:
                w_recon = dequantize_mxfp4(w_data, scale_data)
        else:
            return (key, None, f"unknown mode {mode}")

        if w_ref.shape != w_recon.shape:
            print(f"shape mismatch: {w_ref.shape} vs {w_recon.shape}")
            return (key, None, f"shape mismatch: {w_ref.shape} vs {w_recon.shape}")

        metrics = compute_metrics(w_ref, w_recon)
        result = {
            "file": origin_tensors[key][0], "tensor_key": key,
            "numel": metrics['numel'],
            "mse": metrics['mse'],
            "mae": metrics['mae'],
            "max_err": metrics['max_err'],
            "cosine_sim": metrics['cosine_sim'],
            "rel_l2": metrics['rel_l2'],
            "sqnr_db": metrics['sqnr_db'],
            "p99_err": metrics['p99_err'],
            "signal_power": metrics['signal_power'],
            "noise_power": metrics['noise_power'],
            "peak_power": metrics['peak_power'],
            "psnr_db": metrics['psnr_db']
        }
        return (key, result, None)
    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ [{key}] CUDA OOM", flush=True)
        return (key, None, str(e))
    except RuntimeError as e:
        if "CUDA" in str(e) or "cu" in str(e):
            print(f"🚨 [{key}] CUDA Runtime Error: {e}", flush=True)
        else:
            print(f"⚙️ [{key}] General RuntimeError: {e}", flush=True)
        return (key, None, str(e))
    except Exception as e:
        print(f"process_single_tensor_Exception: {key} {str(e)}", flush=True)
        return (key, None, str(e))

def scan_all_tensors(directory: str) -> dict:
    """Scan all safetensors files in a directory and build a global tensor index.

    Returns:
        dict: {tensor_key: (filename, tensor_data)} mapping each key to its source file and data.
    """
    tensor_index = {}
    files = sorted(glob.glob(os.path.join(directory, "*.safetensors")))

    for fpath in files:
        fname = os.path.basename(fpath)
        try:
            data = load_file(fpath, device="cpu")
            for key in data.keys():
                if key in tensor_index:
                    continue
                    print(f"⚠️  Warning: tensor '{key}' found in multiple files (last one wins): {tensor_index[key][0]} vs {fname}")
                tensor_index[key] = (fname, data[key])
        except Exception as e:
            print(f"⚠️  Failed to load {fname}: {e}")

    return tensor_index


def analyze_quantization(origin_dir: str, quant_dir: str, mode: str, block_size: int = 128, output_csv: str = None, num_workers: int = 4, num_gpus: int = 1):
    """Compare origin vs quantized weights per tensor, then report and write aggregate error metrics.

    Scans both dirs into global tensor indexes, dequantizes each matched weight pair per mode on a
    round-robin GPU (CPU when num_gpus=0), then prints a summary table and writes output_csv.
    """
    # use gpu for speedup
    if num_gpus > 0:
        # init all cuda
        if torch.cuda.is_available():
            num_gpus = min(num_gpus, torch.cuda.device_count())
            print(f"🔧 Initializing {num_gpus} CUDA in main thread...")
            for i in range(num_gpus):
                try:
                    torch.zeros(1, device=f'cuda:{i}')
                    torch.cuda.synchronize(i)
                    print(f"✅ CUDA initialized on {torch.cuda.get_device_name(i)}")
                except Exception as e:
                    print(f"❌ CUDA init failed: {e}")
                    raise
        else:
            num_gpus = 0
            print("⚠️ CUDA not available. Use Cpu.")

    # Scan all tensors in both directories (global index across all files)
    print("📂 Scanning origin directory...")
    origin_tensors = scan_all_tensors(origin_dir)
    print(f"   Found {len(origin_tensors)} tensors in origin")

    print("📂 Scanning quant directory...")
    quant_tensors = scan_all_tensors(quant_dir)
    print(f"   Found {len(quant_tensors)} tensors in quant")

    # default output CSV filename
    if output_csv is None:
        if mode == 'fp8_dequant':
            output_csv = 'origin_fp8_vs_dequant_bf16.csv'
        elif mode == 'fp8_channel_int8':
            output_csv = 'origin_fp8_vs_channel_int8.csv'
        elif mode == 'bf16_channel_int8':
            output_csv = 'bf16_vs_channel_int8.csv'
        elif mode == 'fp8_mxfp4':
            output_csv = 'fp8_blockwise_vs_mxfp4.csv'
        else:
            output_csv = 'blockwise_fp8_to_int8.csv'

    results = []
    skipped = []

    if mode == 'fp8':
        print(f"🔍 Compare: Blockwise-FP8(block={block_size}) → PerChannel-INT8\n")
    elif mode == 'bf16_channel_int8':
        print(f"🔍 Compare: BF16 → PerChannel-INT8\n")
    elif mode == 'fp8_dequant':
        print(f"🔍 Compare: Origin(Blockwise-FP8 dequant) → Dequant-BF16 (both as FP32)\n")
    elif mode == 'fp8_channel_int8':
        print(f"🔍 Compare: Origin(Blockwise-FP8 dequant) → Channelwise-INT8 (both as FP32)\n")
    elif mode == 'bf16_mxfp4':
        print(f"🔍 Compare: Origin(BF16) → Groupwise-MXFP4 (both as FP32)\n")
    elif mode == 'bf16_mxfp4_scale32':
        print(f"🔍 Compare: Origin(BF16) → Groupwise-MXFP4-Scale32 (both as FP32)\n")
    elif mode == 'fp8_mxfp4':
        print(f"🔍 Compare: Origin(Blockwise-FP8 dequant→BF16) → Groupwise-MXFP4 (both as FP32)\n")
    else:
        print(f"Unknown mode of weights\n")

    # Find all weight tensors and their corresponding scales across files
    weight_keys_org = list({key for key in origin_tensors.keys() if key.endswith(".weight")})
    # mxfp4 modes: drop shared_experts and self_attn keys
    if mode == 'bf16_mxfp4' or mode == 'fp8_mxfp4':
        weight_keys_org = [
            key for key in weight_keys_org
            if "shared_experts" not in key
            and "self_attn" not in key
        ]
    if num_gpus > 0:
        weight_keys = [(key, i % num_gpus) for i, key in enumerate(weight_keys_org)]  # round-robin GPU assignment
    else:
        weight_keys = [(key, -1) for key in weight_keys_org]  # -1 means CPU

    # Process tensors in parallel using ThreadPoolExecutor
    print(f"⚡ Processing {len(weight_keys)} tensors with {num_workers} workers...")

    def process_and_collect(args):
        """Wrapper that returns (result, skip_reason) tuple"""
        key, device_id = args
        ret_key, result, error = process_single_tensor(key, origin_tensors, quant_tensors, mode, block_size, device_id)
        if result is not None:
            return (result, None)
        else:
            return (None, f"{key} ({error})")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_and_collect, key): key for key in weight_keys}
        for future in tqdm(as_completed(futures), total=len(weight_keys), desc="🔍 Computing error"):
            result, skip_reason = future.result()
            if result is not None:
                results.append(result)
            elif skip_reason:
                skipped.append(skip_reason)

    # Aggregate global statistics from all results
    total_elements = sum(r['numel'] for r in results)
    total_signal = sum(r['signal_power'] for r in results)
    total_noise = sum(r['noise_power'] for r in results)

    # global statistics
    overall_mse = total_noise / total_elements if total_elements > 0 else 0.0
    overall_sqnr = 10.0 * math.log10(total_signal / (total_noise + 1e-15)) if total_noise > 1e-15 else float('inf')
    overall_rel_l2 = math.sqrt(total_noise / (total_signal + 1e-15)) if total_signal > 1e-15 else 0.0

    # output
    print("\n" + "="*85)
    if mode == 'fp8_dequant':
        print(f"📊 Origin(Blockwise-FP8) vs Dequant-BF16 comparison stats")
    elif mode == 'fp8_channel_int8':
        print(f"📊 Origin(Blockwise-FP8) vs Channelwise-INT8 comparison stats")
    elif mode == 'bf16_mxfp4':
        print(f"📊 Origin(BF16) vs MXFP4 comparison stats")
    elif mode == 'bf16_mxfp4_scale32':
        print(f"📊 Origin(BF16) vs MXFP4-Scale32 comparison stats")
    elif mode == 'fp8_mxfp4':
        print(f"📊 Origin(FP8-blockwise dequant) vs MXFP4 comparison stats")
    else:
        print(f"📊 Blockwise-FP8 → PerChannel-INT8 requantization error stats")
    print(f"  Compared params  : {total_elements:,}")
    print(f"  Skipped tensors  : {len(skipped)}")
    print(f"  Global MSE       : {overall_mse:.6e}")
    print(f"  Global MAE       : {math.sqrt(overall_mse):.6f}")
    print(f"  Global Rel-L2    : {overall_rel_l2:.6f}")
    print(f"  Global SQNR (dB) : {overall_sqnr:.2f}")
    print("="*85)

    if skipped:
        print(f"⚠️  Skipped {len(skipped)} tensors (no scale / non-weight / shape mismatch)")

    if mode == 'fp8_dequant' or mode == 'fp8_channel_int8' or mode == 'bf16_mxfp4' or mode == 'bf16_mxfp4_scale32' or mode == 'fp8_mxfp4':
        print("\n📉 Top 15 layers with the largest difference (lowest SQNR):")
    else:
        print("\n📉 Top 15 layers with the largest requantization loss (lowest SQNR):")
    print(f"{'Tensor Key':<60} {'Numel':<10} {'MSE':<12} {'MAE':<10} {'MaxErr':<10} {'CosSim':<8} {'SQNR(dB)':<10}")
    print("-" * 105)

    for r in sorted(results, key=lambda x: x['sqnr_db'])[:15]:
        sqnr_str = f"{r['sqnr_db']:.2f}" if r['sqnr_db'] != float('inf') else "∞"
        print(f"{r['tensor_key']:<45} {r['numel']:<10,} {r['mse']:<12.6e} {r['mae']:<10.6f} {r['max_err']:<10.6f} {r['cosine_sim']:<8.6f} {sqnr_str:<10}")

    if output_csv:
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["file", "tensor_key", "numel", "mse", "mae", "max_err", "cosine_sim", "rel_l2", "sqnr_db", "psnr_db", "p99_err", "signal_power", "noise_power", "peak_power"])
            writer.writeheader()
            writer.writerows(results)
        print(f"\n💾 Details saved to: {output_csv}")

# def main():
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant_dir", type=str, required=True)
    parser.add_argument("--origin_dir", type=str, required=True)
    parser.add_argument("--block_size", type=int, default=128, help="FP8 blockwise quantization block size")
    parser.add_argument("--mode", type=str, choices=["bf16_channel_int8", "fp8", "fp8_dequant", "fp8_channel_int8", "bf16_mxfp4", "bf16_mxfp4_scale32", "fp8_mxfp4"], required=True, help="Origin weight type: bf16=BF16, fp8=Blockwise-FP8, fp8_dequant=Origin(FP8) vs Dequant(BF16), fp8_channel_int8=Origin(FP8) vs Channelwise-INT8, bf16_mxfp4=Origin(bf16) vs mxfp4, bf16_mxfp4_scale32=Origin(bf16) vs mxfp4 with float scale")
    parser.add_argument("--output_csv", type=str, default=None, help="Output CSV filename; defaults to a name derived from mode")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of parallel workers (default: 16)")
    parser.add_argument("--num_gpus", type=int, default=0, help="Number of CUDA GPUs (default: 0)")
    args = parser.parse_args()

    analyze_quantization(args.origin_dir, args.quant_dir, args.mode, args.block_size, args.output_csv, args.num_workers, args.num_gpus)
