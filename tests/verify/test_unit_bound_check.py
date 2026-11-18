# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Quantization bounds with handwritten errors and expected statistics."""

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.cpu]


@pytest.mark.parametrize("mode, peak, half_gap", [
    ("int8", 127, 0.5), ("int4", 7, 0.5), ("fp8", 448, 16.0),
])
@pytest.mark.parametrize("skip_tail", [False, True])
def test_inside_boundary_and_violation_counts(tensor_factory, mode, peak, half_gap, skip_tail):
    import torch
    from qlean.utils.bound_check import compute_bound_check

    reference = tensor_factory((1, 5), pattern="zeros")
    reference[0, 0] = peak
    reconstructed = reference.clone()
    reconstructed[0, 2] = half_gap
    reconstructed[0, 3] = half_gap * (1 + 2 ** -15)
    reconstructed[0, 4] = half_gap * 2

    result, bound = compute_bound_check(reference, reconstructed, half_gap, quant_mode=mode,
                                        return_bound=True, skip_tail_stats=skip_tail)

    assert result == {
        "violations": 1, "nonfinite": 0, "boundary": 1, "max_over_ratio": 2.0,
        "max_over_count": -1 if skip_tail else 1,
        "max_over_pct": -1.0 if skip_tail else 20.0,
        "ratio_p99": -1.0 if skip_tail else 2.0,
    }
    torch.testing.assert_close(bound, torch.full((1, 5), half_gap, dtype=torch.float32),
                               rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["int8", "int4"])
def test_zero_row_has_positive_floor_and_zero_violations(tensor_factory, mode):
    import torch
    from qlean.utils.bound_check import compute_bound_check

    weight = tensor_factory((1, 3), pattern="zeros")
    result, bound = compute_bound_check(weight, weight.clone(), 0.5, quant_mode=mode,
                                        return_bound=True, skip_tail_stats=True)
    assert result["violations"] == result["boundary"] == 0
    assert result["max_over_ratio"] == 0.0
    torch.testing.assert_close(bound, torch.full((1, 3), 2 ** -24, dtype=torch.float32),
                               rtol=0, atol=0)


def test_mxfp_bound_broadcasts_and_crops_tail(tensor_factory):
    import torch
    from qlean.utils.bound_check import compute_bound_check

    weight = tensor_factory((1, 34), pattern="zeros")
    weight[0, 0], weight[0, 32] = 6, 12
    result, bound = compute_bound_check(weight, weight.clone(), 1.0, quant_mode="mxfp4",
                                        return_bound=True, skip_tail_stats=True)
    expected = torch.tensor([[1.0] * 32 + [2.0] * 2], dtype=torch.float32)
    torch.testing.assert_close(bound, expected, rtol=0, atol=0)
    assert result["violations"] == result["boundary"] == 0
    assert result["max_over_ratio"] == 0.0


@pytest.mark.parametrize("chunk_elems", [1, 4, 8, 20])
def test_chunked_counts_match_handwritten_result(tensor_factory, monkeypatch, chunk_elems):
    import torch
    from qlean.modifiers_engine._runtime import verify
    from qlean.utils.bound_check import dequantize_int8_per_channel

    reference = tensor_factory((5, 4), pattern="zeros")
    reference[:, 0] = 127
    codes = reference.to(torch.int8)
    codes[1, 1], codes[4, 2] = 1, -2
    scale = tensor_factory((5, 1), pattern="zeros").fill_(1)
    monkeypatch.setattr(verify, "VERIFY_CHUNK_ELEMS", chunk_elems)

    result = verify._bound_check_chunked(reference, codes, scale, "int8",
                                         dequantize_int8_per_channel)

    assert result == {"violations": 2, "nonfinite": 0, "boundary": 0, "max_over_ratio": 4.0}


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")],
                         ids=["nan", "positive-inf", "negative-inf"])
@pytest.mark.parametrize("in_reference", [False, True], ids=["reconstruction", "reference"])
@pytest.mark.parametrize("skip_tail", [False, True])
def test_nonfinite_values_are_violations_not_success(
    tensor_factory, invalid, in_reference, skip_tail,
):
    from qlean.utils.bound_check import compute_bound_check

    reference = tensor_factory((1, 4), pattern="zeros")
    reference[0, 0] = 127
    reconstructed = reference.clone()
    reconstructed[0, 2] = 1
    (reference if in_reference else reconstructed)[0, 1] = invalid

    result = compute_bound_check(reference, reconstructed, 0.5, quant_mode="int8",
                                 skip_tail_stats=skip_tail)

    assert result == {
        "violations": 4 if in_reference else 2,
        "nonfinite": 4 if in_reference else 1,
        "boundary": 0,
        "max_over_ratio": -1.0 if in_reference else 2.0,
        "max_over_count": -1 if skip_tail else (0 if in_reference else 1),
        "max_over_pct": -1.0 if skip_tail else (0.0 if in_reference else 25.0),
        "ratio_p99": -1.0 if skip_tail or in_reference else 2.0,
    }


@pytest.mark.parametrize("overflow", ["error", "ratio"])
def test_finite_inputs_distinguish_error_overflow_from_ratio_overflow(overflow):
    import torch
    from qlean.utils.bound_check import compute_bound_check

    if overflow == "error":
        reference = torch.tensor([[torch.finfo(torch.float32).max]], dtype=torch.float32)
        reconstructed = -reference
    else:
        reference = torch.zeros((1, 1), dtype=torch.float32)
        reconstructed = torch.tensor([[1e32]], dtype=torch.float32)
    assert torch.isfinite(reference).all() and torch.isfinite(reconstructed).all()

    result = compute_bound_check(reference, reconstructed, 0.5, quant_mode="int8",
                                 skip_tail_stats=True)

    assert result == {
        "violations": 1, "nonfinite": 1 if overflow == "error" else 0,
        "boundary": 0, "max_over_ratio": -1.0 if overflow == "error" else float("inf"),
        "max_over_count": -1, "max_over_pct": -1.0, "ratio_p99": -1.0,
    }


@pytest.mark.parametrize("invalid_first", [False, True])
def test_chunked_nonfinite_counts_preserve_finite_worst_ratio(monkeypatch, invalid_first):
    import torch
    from qlean.modifiers_engine._runtime import verify
    from qlean.utils.bound_check import dequantize_int8_per_channel

    reference = torch.tensor([[127, 0], [127, 0]], dtype=torch.float32)
    codes = torch.tensor([[127, 0], [127, 1]], dtype=torch.int8)
    scales = torch.tensor([[float("nan")], [1]], dtype=torch.float32)
    if not invalid_first:
        reference, codes, scales = (tensor.flip(0) for tensor in (reference, codes, scales))
    monkeypatch.setattr(verify, "VERIFY_CHUNK_ELEMS", 2)

    result = verify._bound_check_chunked(reference, codes, scales, "int8",
                                         dequantize_int8_per_channel)

    assert result == {"violations": 3, "nonfinite": 2, "boundary": 0, "max_over_ratio": 2.0}


def test_unknown_bound_mode_fails(tensor_factory):
    from qlean.utils.bound_check import compute_bound_check

    weight = tensor_factory((1, 2))
    with pytest.raises(ValueError, match="Unknown quant_mode"):
        compute_bound_check(weight, weight, 0.5, quant_mode="unknown")
