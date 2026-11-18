# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""W8A8 edge cases: single elements, negative values, and irregular shapes."""

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.cpu]


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("scale_value", [0.5, 1.0, 2.0])
def test_single_element_has_exact_code_and_scale(tensor_factory, assert_tensor_equal, sign, scale_value):
    import torch
    from qlean.modifiers.utils import weight_quant

    weight = tensor_factory((1, 1), pattern="zeros")
    weight.fill_(sign * 127 * scale_value)
    codes, scale = weight_quant(weight, "W8A8")
    assert_tensor_equal(codes, torch.tensor([[sign * 127]], dtype=torch.int8),
                        scheme="W8A8", seed=0)
    assert_tensor_equal(scale, torch.tensor([[scale_value]], dtype=torch.float32),
                        scheme="W8A8", seed=0)


@pytest.mark.parametrize("noncontiguous", [False, True])
def test_negative_odd_width_rows_keep_independent_scales(
    tensor_factory, assert_tensor_equal, noncontiguous,
):
    import torch
    from qlean.modifiers.utils import weight_quant

    expected_codes = -(tensor_factory((5, 7), pattern="arange") % 7 + 1).to(torch.int8)
    expected_codes[:, -1] = -127
    expected_scale = tensor_factory((5, 1), pattern="arange") + 1
    weight = expected_codes.float() * expected_scale
    if noncontiguous:
        weight = weight.t().contiguous().t()
        assert not weight.is_contiguous()
    before = weight.clone()

    codes, scale = weight_quant(weight, "W8A8")

    assert_tensor_equal(codes, expected_codes, scheme="W8A8", seed=0)
    assert_tensor_equal(scale, expected_scale, scheme="W8A8", seed=0)
    assert_tensor_equal(weight, before, scheme="W8A8", seed=0)


@pytest.mark.parametrize("multiple, expected_code", [(0.25, 0), (0.5, 0), (1.5, 2)])
@pytest.mark.parametrize("sign", [-1, 1])
def test_tiny_nonzero_values_round_using_scale_floor(
    tensor_factory, assert_tensor_equal, multiple, expected_code, sign,
):
    import torch
    from qlean.modifiers.utils import weight_quant

    weight = tensor_factory((1, 1), pattern="zeros")
    weight.fill_(sign * multiple * 2 ** -23)
    codes, scale = weight_quant(weight, "W8A8")

    assert_tensor_equal(codes, torch.tensor([[sign * expected_code]], dtype=torch.int8),
                        scheme="W8A8", seed=0)
    assert_tensor_equal(scale, torch.tensor([[2 ** -23]], dtype=torch.float32),
                        scheme="W8A8", seed=0)
