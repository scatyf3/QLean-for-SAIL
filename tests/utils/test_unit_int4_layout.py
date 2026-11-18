# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""INT4 packing and unpacking layouts, roundtrips, and input validation."""

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.cpu]

_ROW_ORDER = (
    0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23,
    8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31,
)


def _reference_bytes(codes, row_order=_ROW_ORDER):
    """Encode bytes with a fixed 32-column mapping, independent of production helpers."""
    import torch

    if codes.dim() == 3:
        return torch.stack([_reference_bytes(expert, row_order) for expert in codes])
    rows = codes.tolist()
    n, k = codes.shape
    packed = []
    for row in rows:
        for column in range(0, k, 2):
            block = column // 32 * 32
            lo = int(row[block + row_order[column % 32]]) % 16
            hi = int(row[block + row_order[column % 32 + 1]]) % 16
            packed.append(lo + 16 * hi)
    return torch.tensor(packed, dtype=torch.uint8).reshape(n, k // 2)


@pytest.fixture
def layout_case(tensor_factory):
    import torch

    def make(shape):
        codes = tensor_factory(shape, seed=17).mul(31).round().remainder(16).sub(8)
        codes = codes.to(torch.int8)
        codes.reshape(-1)[:3] = torch.tensor([-8, 7, 0], dtype=torch.int8)
        return codes, _reference_bytes(codes)

    return make


def test_nibble_order_and_signed_endpoints(tensor_factory):
    import torch
    from qlean.utils.pack_acext_int4 import pack_int8_tensor_to_packed_int4_torch

    codes = (tensor_factory((1, 16), pattern="arange") - 8).to(torch.int8)
    packed = pack_int8_tensor_to_packed_int4_torch(codes)
    expected = torch.tensor([[0x98, 0xBA, 0xDC, 0xFE, 0x10, 0x32, 0x54, 0x76]],
                            dtype=torch.uint8)
    assert packed.dtype == torch.int8
    torch.testing.assert_close(packed.view(torch.uint8), expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(64, 64), (64, 128), (2, 64, 64)])
@pytest.mark.parametrize("validate", [False, True])
def test_a8w4_packed_bytes_match_scalar_reference(layout_case, shape, validate):
    import torch
    from qlean.utils.pack_acext_int4 import preprocess_weight_for_a8w4_torch

    codes, expected = layout_case(shape)
    before = codes.clone()
    packed = preprocess_weight_for_a8w4_torch(codes, validate=validate)

    assert packed.shape == (*shape[:-2], shape[-1], shape[-2] // 2)
    assert packed.dtype == torch.int8
    assert packed.device.type == "cpu"
    torch.testing.assert_close(packed.view(torch.uint8).reshape(expected.shape), expected,
                               rtol=0, atol=0)
    torch.testing.assert_close(codes, before, rtol=0, atol=0)


@pytest.mark.parametrize("value, byte", [(-8, 0x88), (0, 0), (7, 0x77)])
def test_constant_codes_have_handwritten_bytes(tensor_factory, value, byte):
    import torch
    from qlean.utils.pack_acext_int4 import preprocess_weight_for_a8w4_torch

    codes = tensor_factory((64, 64), pattern="zeros", dtype=torch.int8).fill_(value)
    packed = preprocess_weight_for_a8w4_torch(codes)
    assert packed.view(torch.uint8).unique().tolist() == [byte]


@pytest.mark.parametrize("experts", [False, True])
def test_checkerboard_has_fixed_alternating_bytes(experts):
    import torch
    from qlean.utils.pack_acext_int4 import preprocess_weight_for_a8w4_torch
    from qlean.utils.bound_check import unpack_int4_acext

    parity = (torch.arange(64)[:, None] + torch.arange(64)[None, :]) % 2
    codes = torch.where(parity == 0, -8, 7).to(torch.int8)
    expected = torch.tensor([[0x78] * 32, [0x87] * 32], dtype=torch.uint8).repeat(32, 1)
    if experts:
        codes = torch.stack((codes, codes.flip(0)))
        expected = torch.stack((expected, expected.flip(0)))
    packed = preprocess_weight_for_a8w4_torch(codes)
    torch.testing.assert_close(packed.view(torch.uint8), expected, rtol=0, atol=0)
    if experts:
        decoded = torch.stack([unpack_int4_acext(part, (64, 64)) for part in expected])
    else:
        decoded = unpack_int4_acext(expected, (64, 64))
    torch.testing.assert_close(decoded, codes, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(32, 32), (32, 64), (64, 128)])
@pytest.mark.parametrize("storage_dtype", ["uint8", "int8"])
def test_unpack_uses_independently_encoded_bytes(layout_case, shape, storage_dtype):
    import torch
    from qlean.utils.bound_check import unpack_int4_acext

    expected, packed = layout_case(shape)
    codes = unpack_int4_acext(packed.view(getattr(torch, storage_dtype)), shape)
    torch.testing.assert_close(codes, expected, rtol=0, atol=0)


@pytest.mark.parametrize("fault", ["skip_permutation", "swap_nibbles"])
def test_layout_corruption_is_detectable(layout_case, fault):
    import torch
    from qlean.utils.bound_check import unpack_int4_acext

    expected, packed = layout_case((64, 64))
    if fault == "skip_permutation":
        packed = _reference_bytes(expected, tuple(range(32)))
    else:
        packed = ((packed & 15) << 4) | (packed >> 4)
    assert not torch.equal(unpack_int4_acext(packed, expected.shape), expected)


@pytest.mark.parametrize("case, message", [
    ("dtype", "int8 tensor"), ("empty", "empty"), ("odd", "even"),
    ("high", "int4 range"), ("low", "int4 range"),
])
def test_nibble_pack_rejects_invalid_inputs(tensor_factory, case, message):
    import torch
    from qlean.utils.pack_acext_int4 import pack_int8_tensor_to_packed_int4_torch

    codes = tensor_factory((1, 4), pattern="zeros", dtype=torch.int8)
    if case == "dtype":
        codes = codes.float()
    elif case == "empty":
        codes = codes[:, :0]
    elif case == "odd":
        codes = codes[:, :3]
    else:
        codes[0, 0] = 8 if case == "high" else -9
    with pytest.raises(ValueError, match=message):
        pack_int8_tensor_to_packed_int4_torch(codes)


@pytest.mark.parametrize("shape", [(32, 64), (64, 32), (64, 48), (33, 64)])
def test_a8w4_rejects_unsupported_tiles(tensor_factory, shape):
    import torch
    from qlean.utils.pack_acext_int4 import preprocess_weight_for_a8w4_torch

    codes = tensor_factory(shape, pattern="zeros", dtype=torch.int8)
    with pytest.raises(ValueError):
        preprocess_weight_for_a8w4_torch(codes)


@pytest.mark.parametrize("shape, out_shape, dtype_name, message", [
    ((32, 16), (32, 32, 1), "uint8", "2D out_shape"),
    ((16, 16), (16, 32), "uint8", "multiples of 32"),
    ((32, 50), (32, 100), "uint8", "multiples of 32"),
    ((32, 16), (32, 64), "uint8", "packed shape"),
    ((32, 16), (32, 32), "int32", "int8/uint8"),
])
def test_unpack_rejects_invalid_storage(tensor_factory, shape, out_shape, dtype_name, message):
    import torch
    from qlean.utils.bound_check import unpack_int4_acext

    packed = tensor_factory(shape, pattern="zeros", dtype=getattr(torch, dtype_name))
    with pytest.raises(ValueError, match=message):
        unpack_int4_acext(packed, out_shape)
