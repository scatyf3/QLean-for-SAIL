# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import csv
import math
import re
import sys
from collections import defaultdict

# STR_SORT, SORT_reverse = 'psnr_db', False
# STR_SORT, SORT_reverse = 'sqnr_db', False
STR_SORT, SORT_reverse = 'max_err', True
# STR_SORT, SORT_reverse = 'peak_power', True
#STR_SORT, SORT_reverse = 'mse', True
AVG_METRICS = ['max_err', 'sqnr_db', 'peak_power']
_LAYER_PATTERN = re.compile(r'model\.layers\.(\d+)\.')
_MOE_EXPERTS_PATTERN = re.compile(r'77\.mlp\.experts\.(\d+)\.down_proj\.weight')


def extract_layer_number(tensor_key: str) -> int:
    """Extract the layer number from a tensor_key, e.g. model.layers.3.block_sparse_moe... → 3."""
    m = _LAYER_PATTERN.search(tensor_key)
    return int(m.group(1)) if m else -1


def aggregate_by_layer(results: list) -> list[dict]:
    """
    Aggregate all tensors by layer number and return the per-layer metrics.

    Returns:
        list of dicts, each with: layer_no, numel, mse, mae, max_err, rel_l2, sqnr_db, weight_power, count
    """
    layers = defaultdict(list)
    for r in results:
        layers[extract_layer_number(r['tensor_key'])].append(r)

    aggregated = []
    for layer_no, layer_results in layers.items():
        total_numel = sum(r['numel'] for r in layer_results)
        total_signal = sum(r['signal_power'] for r in layer_results)
        total_noise = sum(r['noise_power'] for r in layer_results)

        layer_mse = total_noise / total_numel if total_numel > 0 else 0.0
        layer_sqnr = 10.0 * math.log10(total_signal / (total_noise + 1e-15)) if total_noise > 1e-15 else float('inf')
        layer_rel_l2 = math.sqrt(total_noise / (total_signal + 1e-15)) if total_signal > 1e-15 else 0.0
        layer_mae = math.sqrt(layer_mse)

        # per-layer max_err: max of all tensors in this layer
        layer_max_err = max(r['max_err'] for r in layer_results)
        # per-layer cosine_sim: weighted average
        layer_cosim = sum(r['cosine_sim'] * r['numel'] for r in layer_results) / total_numel if total_numel > 0 else 1.0
        layer_peak_power = max(r['peak_power'] for r in layer_results if r['peak_power'] is not None) if any(r['peak_power'] is not None for r in layer_results) else None
        # layer_psnr = 10.0 * math.log10(layer_peak_power / (layer_mse + 1e-15)) if layer_peak_power is not None and layer_peak_power > 1e-15 and layer_mse > 0 else (float('inf') if layer_peak_power and layer_peak_power > 1e-15 else None)
        layer_psnr = sum(r['psnr_db'] for r in layer_results) / len(layer_results)
        aggregated.append({
            'layer_no': layer_no,
            'numel': total_numel,
            'mse': layer_mse,
            'mae': layer_mae,
            'max_err': layer_max_err,
            'rel_l2': layer_rel_l2,
            'weight_power': total_signal,
            'sqnr_db': layer_sqnr,
            'cosine_sim': layer_cosim,
            'psnr_db': layer_psnr,
            'peak_power': layer_peak_power,
            'count': len(layer_results),
        })
    return aggregated


def classify_tensor_key(tensor_key: str) -> str:
    """
    Classify a tensor_key into attn / experts / other by its naming pattern.

    Args:
        tensor_key: tensor name, e.g. model.layers.0.self_attn.q_proj.weight
                   or model.layers.60.mlp.experts.*.down_proj.weight
    Returns:
        'attn' | 'experts' | 'other'
    """
    if 'attn' in tensor_key or 'attention' in tensor_key:
        return 'attn'
    elif 'block_sparse_moe.experts' in tensor_key or 'mlp.experts' in tensor_key:
        return 'experts'
    else:
        return 'other'


def compute_category_stats(results: list, category: str) -> dict:
    """
    Aggregate the error metrics of one category subset.

    Args:
        results: list of per-tensor error metric dicts
        category: 'attn' | 'experts' | 'other'

    Returns:
        dict of aggregated metrics
    """
    total_elements = sum(r['numel'] for r in results)
    total_signal = sum(r['signal_power'] for r in results)
    total_noise = sum(r['noise_power'] for r in results)
    peak_signal = max(r['peak_power'] for r in results)
    overall_maxerr = max(r['max_err'] for r in results)

    overall_mse = total_noise / total_elements if total_elements > 0 else 0.0
    overall_sqnr = 10.0 * math.log10(total_signal / (total_noise + 1e-15)) if total_noise > 1e-15 else float('inf')
    # overall_psnr = 10.0 * math.log10(peak_signal / (overall_mse + 1e-15)) if peak_signal  > 1e-15 else float('inf')
    overall_psnr = sum(r['psnr_db'] for r in results) / len(results)
    overall_rel_l2 = math.sqrt(total_noise / (total_signal + 1e-15)) if total_signal > 1e-15 else 0.0

    return {
        'category': category,
        'numel': total_elements,
        'signal_power': total_signal,
        'noise_power': total_noise,
        'mse': overall_mse,
        'mae': math.sqrt(overall_mse),
        'rel_l2': overall_rel_l2,
        'MaxErr': overall_maxerr,
        'weight_power': total_signal,
        'sqnr_db': overall_sqnr,
        'psnr_db': overall_psnr,
        'peak_power': peak_signal,
        'count': len(results),
    }


def print_layer_rank_table(layer_agg: list, metrics: list[str] | None = None, top_k: int = 15):
    """
    Print a per-layer error ranking table, ordered by the average rank over several metrics.

    rank 1 always means "worst":
      - error metrics (mse, mae, max_err, p99_err, rel_l2, noise_power): larger is worse
      - quality metrics (sqnr_db, psnr_db, cosine_sim, signal_power, peak_power): smaller is worse

    Args:
        layer_agg: per-layer aggregation returned by aggregate_by_layer
        metrics: subset of metrics used for avg_rank; None means all of them. The table
                 always shows every metric's rank; only avg_rank depends on this argument.
        top_k: print the top k layers
    """
    # Sort direction per metric: True = larger is worse (ranks first), False = smaller is worse
    _METRIC_HIGHER_IS_WORSE = {
        'mse': True,
        'mae': True,
        'max_err': True,
        # 'p99_err': True,
        'rel_l2': True,
        'cosine_sim': False,
        'sqnr_db': False,
        'psnr_db': False,
        # 'noise_power': True,
        # 'signal_power': True,
        'peak_power': True,
    }
    _METRIC_DISPLAY = {
        'mse': 'r_MSE',
        'mae': 'r_MAE',
        'max_err': 'r_MaxErr',
        'p99_err': 'r_P99',
        'rel_l2': 'r_RelL2',
        'cosine_sim': 'r_CosSim',
        'sqnr_db': 'r_SQNR',
        'psnr_db': 'r_PSNR',
        'noise_power': 'r_Noise',
        'signal_power': 'r_Signal',
        'peak_power': 'r_Peak',
    }

    all_metrics = list(_METRIC_HIGHER_IS_WORSE.keys())
    avg_metrics = metrics if metrics is not None else all_metrics

    # Drop layers that cannot be ranked (layer_no == -1 means the layer number was unparsable)
    valid_layers = [r for r in layer_agg if r['layer_no'] != -1]

    # Rank every metric independently: rank 1 = worst
    ranks = {m: {} for m in all_metrics}
    for m in all_metrics:
        higher_is_worse = _METRIC_HIGHER_IS_WORSE.get(m, True)
        # Skip layers whose value is inf / None
        sortable = [r for r in valid_layers
                    if r.get(m) is not None and r[m] != float('inf') and r[m] != float('-inf')]
        # reverse=True puts the largest value first; when higher_is_worse, largest = worst = rank 1
        sorted_layers = sorted(sortable, key=lambda x: x[m], reverse=higher_is_worse)
        for rank_idx, r in enumerate(sorted_layers, start=1):
            ranks[m][r['layer_no']] = rank_idx

    # Average rank per layer, over avg_metrics only
    layer_rank_rows = []
    for r in valid_layers:
        layer_no = r['layer_no']
        metric_ranks = {}
        # Collect every metric's rank for display
        for m in all_metrics:
            metric_ranks[m] = ranks[m].get(layer_no)
        # avg_rank uses avg_metrics only
        valid_rank_values = [ranks[m][layer_no] for m in avg_metrics if layer_no in ranks[m]]
        avg_rank = sum(valid_rank_values) / len(valid_rank_values) if valid_rank_values else float('inf')
        layer_rank_rows.append({
            'layer_no': layer_no,
            'avg_rank': avg_rank,
            'metric_ranks': metric_ranks,
        })

    # Sort by avg_rank ascending (worst layer first)
    layer_rank_rows.sort(key=lambda x: x['avg_rank'])

    # Column order: metrics counted in avg_rank first, the rest after, separated by |
    avg_set = set(avg_metrics)
    part_metrics = [m for m in avg_metrics]
    other_metrics = [m for m in all_metrics if m not in avg_set]

    def _format_headers(metric_list):
        return " ".join(f"{_METRIC_DISPLAY.get(m, f'r_{m}'):<10}" for m in metric_list)

    def _format_row(row, metric_list):
        parts = []
        for m in metric_list:
            mr = row['metric_ranks'][m]
            parts.append(f"{mr:<10}" if mr is not None else f"{'N/A':<10}")
        return " ".join(parts)

    prefix = f"{'#':<4} {'layer.No':<12} {'avg_rank':<10} "
    part_header = _format_headers(part_metrics)
    other_header = _format_headers(other_metrics)
    sep = " | "
    header = prefix + part_header + sep + other_header

    print(f"\n📊 Average-rank table (metrics counted in avg_rank: {', '.join(avg_metrics)}):")
    print(header)
    print("-" * len(header))

    for idx, row in enumerate(layer_rank_rows[:top_k], start=1):
        prefix = f"{idx:<4} {'layer.' + str(row['layer_no']):<12} {row['avg_rank']:<10.1f} "
        part_vals = _format_row(row, part_metrics)
        other_vals = _format_row(row, other_metrics)
        print(prefix + part_vals + sep + other_vals)

    # Print the layer id list so it can be pasted into other scripts
    layer_ids = [row['layer_no'] for row in layer_rank_rows[:top_k]]
    print(f"\nLayer ID order: {', '.join(str(lid) for lid in layer_ids)}")

def print_category_summary(results: list, mode: str):
    """
    Print the quantization error stats grouped by attn / experts / other.

    Args:
        results: list of per-tensor error metric dicts
        mode: comparison mode
    """
    attn_results = [r for r in results if classify_tensor_key(r['tensor_key']) == 'attn']
    experts_results = [r for r in results if classify_tensor_key(r['tensor_key']) == 'experts']
    other_results = [r for r in results if classify_tensor_key(r['tensor_key']) == 'other']
    moe_tensors_results = [r for r in results if _MOE_EXPERTS_PATTERN.search(r['tensor_key'])]

    for category, group_results in [
        ('🔵 Attention', attn_results),
        ('🟡 Experts (MoE)', experts_results),
        ('🟢 Other', other_results),
        (_MOE_EXPERTS_PATTERN.pattern, moe_tensors_results),
    ]:
        if not group_results:
            continue
        stats = compute_category_stats(group_results, category)

        print(f"\n{'='*85}")
        print(f"{category} quantization error stats ({stats['count']} tensors)")
        print(f"  Elements           : {stats['numel']:,}")
        print(f"  Global MSE         : {stats['mse']:.6e}")
        print(f"  Global MAE         : {stats['mae']:.6f}")
        print(f"  Global Rel-L2      : {stats['rel_l2']:.6f}")
        print(f"  Global MaxErr      : {stats['MaxErr']:.6f}")
        print(f"  Global weight_power: {stats['weight_power']:.6e}")
        print(f"  Global SQNR (dB)   : {stats['sqnr_db']:.2f}")
        print(f"  Global peak_power  : {stats['peak_power']:.6e}")
        print(f"  Mean tensor PSNR (dB): {stats['psnr_db']:.2f}")
        print("="*85)

        layer_agg = aggregate_by_layer(group_results)
        print(f"\n📉 Top 15 layers by aggregated {STR_SORT} ({'highest' if SORT_reverse else 'lowest'}):")
        print(f"{'layer.No':<12} {'Elements':<14} {'MSE':<12} {'MAE':<10} {'MaxErr':<10} {'CosSim':<10} {'Weight_power':<13} {'SQNR(dB)':<10} {'Peak_power':<13} {'PSNR(dB)':<10}")
        print("-" * 123)
        count = 0
        for r in sorted(layer_agg, key=lambda x: x[STR_SORT], reverse=SORT_reverse):
            if r[STR_SORT] == float('inf'):
                continue
            sqnr_str = f"{r['sqnr_db']:.2f}"
            psnr_str = f"{r['psnr_db']:.2f}" if r['psnr_db'] is not None else "N/A"
            peak_str = f"{r['peak_power']:.6e}" if r['peak_power'] is not None else "N/A"
            print(f"{'layer.' + str(r['layer_no']):<12} {r['numel']:<14,} {r['mse']:<12.6e} {r['mae']:<10.6f} {r['max_err']:<10.6f} {r['cosine_sim']:<10.6f} {r['weight_power']:<13.6e} {sqnr_str:<10} {peak_str:<13} {psnr_str:<10}")
            count += 1
            if count >= 15:
                break

        print(f"\n📉 Top 15 tensor keys by {STR_SORT} ({'highest' if SORT_reverse else 'lowest'}):")
        print(f"{'tensor_key':<50} {'Elements':<10} {'MSE':<12} {'MAE':<10} {'MaxErr':<10} {'CosSim':<8} {'Weight_power':<13} {'SQNR(dB)':<10} {'Peak_power':<13} {'PSNR(dB)':<10}")
        print("-" * 120)
        count = 0
        for r in sorted(group_results, key=lambda x: x[STR_SORT], reverse=SORT_reverse):
            if r[STR_SORT] == float('inf') or r[STR_SORT] <= 0:
                continue
            sqnr_str = f"{r['sqnr_db']:.2f}"
            psnr_str = f"{r['psnr_db']:.2f}" if r.get('psnr_db') is not None else "N/A"
            peak_str = f"{r['peak_power']:.6e}" if r.get('peak_power') is not None else "N/A"
            print(f"{r['tensor_key']:<45} {r['numel']:<10,} {r['mse']:<12.6e} {r['mae']:<10.6f} {r['max_err']:<10.6f} {r['cosine_sim']:<10.6f} {r['signal_power']:<13.6e} {sqnr_str:<10} {peak_str:<13} {psnr_str:<10}")
            count += 1
            if count >= 15:
                break
        print_layer_rank_table(layer_agg, metrics=AVG_METRICS)


def print_quantization_summary(results: list, mode: str):
    """
    Print the overall aggregated quantization error summary.

    Args:
        results: list of per-tensor error metric dicts
    """
    total_elements = sum(r['numel'] for r in results)
    total_signal = sum(r['signal_power'] for r in results)
    total_noise = sum(r['noise_power'] for r in results)
    peak_signal = max(r['peak_power'] for r in results)
    overall_maxerr = max(r['max_err'] for r in results)

    overall_mse = total_noise / total_elements if total_elements > 0 else 0.0
    overall_sqnr = 10.0 * math.log10(total_signal / (total_noise + 1e-15)) if total_noise > 1e-15 else float('inf')
    # overall_psnr = 10.0 * math.log10(peak_signal / (overall_mse + 1e-15)) if peak_signal  > 1e-15 else float('inf')
    overall_psnr = sum(r['psnr_db'] for r in results) / len(results)
    overall_rel_l2 = math.sqrt(total_noise / (total_signal + 1e-15)) if total_signal > 1e-15 else 0.0

    print("\n" + "="*85)
    print(f"Overall quantization error stats ({len(results)} tensors)")
    print(f"  Elements compared  : {total_elements:,}")
    print(f"  Global MSE         : {overall_mse:.6e}")
    print(f"  Global MAE         : {math.sqrt(overall_mse):.6f}")
    print(f"  Global Rel-L2      : {overall_rel_l2:.6f}")
    print(f"  Global MaxErr      : {overall_maxerr:.6f}")
    print(f"  Global weight_power: {total_signal:.6e}")
    print(f"  Global SQNR (dB)   : {overall_sqnr:.2f}")
    print(f"  Global peak_power  : {peak_signal:.6e}")
    print(f"  Mean tensor PSNR (dB): {overall_psnr:.2f}")
    print("="*85)

    print(f"\n📉 Top 15 layers by {STR_SORT} ({'highest' if SORT_reverse else 'lowest'}, per-layer aggregate):")
    print(f"{'layer.No':<12} {'Elements':<14} {'MSE':<12} {'MAE':<10} {'MaxErr':<10} {'CosSim':<8} {'Weight_power':<8} {'SQNR(dB)':<10} {'Peak_power':<13} {'PSNR(dB)':<10}")
    print("-" * 108)

    layer_agg = aggregate_by_layer(results)
    for r in sorted(layer_agg, key=lambda x: x[STR_SORT], reverse=SORT_reverse)[:15]:
        sqnr_str = f"{r['sqnr_db']:.2f}" if r['sqnr_db'] != float('inf') else "∞"
        psnr_str = f"{r['psnr_db']:.2f}" if r['psnr_db'] is not None else "N/A"
        peak_str = f"{r['peak_power']:.6e}" if r['peak_power'] is not None else "N/A"
        print(f"{'layer.' + str(r['layer_no']):<12} {r['numel']:<14,} {r['mse']:<12.6e} {r['mae']:<10.6f} {r['max_err']:<10.6f} {r['cosine_sim']:<10.6f} {r['weight_power']:<13.6e} {sqnr_str:<10} {peak_str:<13} {psnr_str:<10}")
    print_layer_rank_table(layer_agg, metrics=AVG_METRICS)

def analyze_from_csv(csv_path: str, mode: str):
    """
    Load quantization error results from a CSV file and print the stats.

    Args:
        csv_path: path to the CSV file
        mode: comparison mode
    """
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        results = []
        for row in reader:
            results.append({
                'file': row['file'],
                'tensor_key': row['tensor_key'],
                'numel': int(row['numel']),
                'mse': float(row['mse']),
                'mae': float(row['mae']),
                'max_err': float(row['max_err']),
                'cosine_sim': float(row['cosine_sim']),
                'rel_l2': float(row['rel_l2']),
                'sqnr_db': float(row['sqnr_db']),
                'p99_err': float(row['p99_err']),
                'signal_power': float(row['signal_power']),
                'noise_power': float(row['noise_power']),
                'peak_power': float(row['peak_power']) if row.get('peak_power') else None,
                'psnr_db': float(row['psnr_db']) if row.get('psnr_db') else None,
            })
    print_quantization_summary(results, mode)
    print_category_summary(results, mode)


# analyze_from_csv('fp8_channel_int8-minimax-m2.7', 'fp8_channel_int8')
# analyze_from_csv('fp8_channel_int8_minimax-m2.5', 'fp8_channel_int8')
# analyze_from_csv('fp8_channel_int8_mimo', 'fp8_channel_int8')
# analyze_from_csv('fp8_channel_int8_DeepSeek_V3.2', 'fp8_channel_int8')
# analyze_from_csv('fp8_channel_int8_qwen3.5_397b', 'fp8_channel_int8')
# analyze_from_csv('bf16_channel_int8_glm5', 'fp8_channel_int8')
# analyze_from_csv('bf16_channel_int8_kimi2.5', 'fp8_channel_int8')
# analyze_from_csv('bf16_mxfp4_Kimi-K2.5', 'fp8_channel_int8')
# analyze_from_csv('fp8_w8a8_glm51', 'fp8_channel_int8')
# analyze_from_csv('bf16_mxfp4_scale32_Kimi-K2.5', 'fp8_channel_int8')
# analyze_from_csv('bf16_channel_int4_Kimi-K2.5', 'fp8_channel_int8')
# analyze_from_csv('bf16_mxfp4_mse_Kimi-K2.5', 'fp8_channel_int8')
# analyze_from_csv('bf16_channel_int8_Qwen3-235B-A22B-Instruct-2507', 'fp8_channel_int8')
# analyze_from_csv('bf16_channel_int8_Qwen3-235B-A22B', 'fp8_channel_int8')