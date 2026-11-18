# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Cross-shard MXFP8 streaming and full-tensor quantization against independent references."""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.mark.parametrize("scheme", ["W8A8", "FP8_DYNAMIC"])
@pytest.mark.parametrize("target", ["fp32", "bf16"])
@pytest.mark.parametrize("chunk_rows", [1, 2, 5, 8])
def test_mxfp8_streamed_codes_match_reference_and_full_table(
    checkpoint_factory, monkeypatch, scheme, target, chunk_rows,
):
    import torch
    from qlean.modifiers.utils import weight_quant
    from qlean.modifiers_engine._runtime.context import LazyWeight, ShardContext
    from qlean.modifiers_engine._runtime.dequant_providers import mxfp8
    from qlean.modifiers_engine._runtime.plan import make_hooks
    from qlean.modifiers_engine._runtime.region_router import normalize_regions, spec_plan_fn

    row = [-448, -112, 0, 112, 448, 0, 0, 0] * 4 + [1, -1]
    raw = torch.tensor([row] * 5, dtype=torch.float32).to(torch.float8_e4m3fn)
    scales = torch.tensor([[127 + r, 128 + r] for r in range(5)], dtype=torch.uint8)
    case = checkpoint_factory(custom_shards=[{"x.weight": raw}, {"x.scale": scales}])
    cfg = {"_regions": normalize_regions([
        {"name": "stream", "match": "x.weight", "src": "mxfp8", "dst": scheme,
         "chunked": True, "chunk_rows": chunk_rows}, {"name": "fallback", "match": "*"}]),
        "dequant_dtype": target}
    ctx = ShardContext("cuda:0", str(case.path), case.weight_map, cfg)
    reads = []
    real_slice = ctx.get_slice

    def read(name, a, b):
        reads.append((name, a, b))
        return real_slice(name, a, b)

    monkeypatch.setattr(ctx, "get_slice", read)
    try:
        process, predict = make_hooks(spec_plan_fn)
        lazy = LazyWeight("x.weight", [5, 34], raw.dtype, chunk_rows)
        streamed = dict(process(ctx, case.shard_names[0], "x.weight", lazy))
        expected_keys = {"x.weight", "x.weight_scale"}
        assert set(streamed) == set(predict({"x.weight": {"shape": [5, 34]}}, cfg,
                                            weight_map=case.weight_map)) == expected_keys
        expected_reads = [(name, a, min(a + chunk_rows, 5)) for a in range(0, 5, chunk_rows)
                          for name in ("x.weight", "x.scale")]
        assert reads == expected_reads
        reference = torch.tensor([[v * 2 ** r for v in row[:32]] + [2 ** (r + 1), -2 ** (r + 1)]
                                  for r in range(5)], dtype=torch.float32)
        if scheme == "W8A8":
            expected_codes = torch.tensor([[-127, -32, 0, 32, 127, 0, 0, 0] * 4 + [1, -1]] * 5,
                                          dtype=torch.int8)
            expected_scales = torch.tensor([[448 * 2 ** r / 127] for r in range(5)], dtype=torch.float32)
        else:
            expected_codes = torch.tensor([row[:32] + [2, -2]] * 5, dtype=torch.float32).to(torch.float8_e4m3fn)
            expected_scales = torch.tensor([[2 ** r] for r in range(5)], dtype=torch.float32)
        assert streamed["x.weight"].dtype == expected_codes.dtype
        torch.testing.assert_close(streamed["x.weight"].view(torch.uint8), expected_codes.view(torch.uint8), rtol=0, atol=0)
        torch.testing.assert_close(streamed["x.weight_scale"], expected_scales, rtol=0, atol=0)
        dequantized = mxfp8(ctx, "x.weight", raw.to("cuda:0"))
        dtype = torch.float32 if target == "fp32" else torch.bfloat16
        torch.testing.assert_close(dequantized.cpu(), reference.to(dtype), rtol=0, atol=0)
        full_codes, full_scales = weight_quant(dequantized, scheme)
        assert full_codes.dtype == expected_codes.dtype
        torch.testing.assert_close(full_codes.cpu().view(torch.uint8), expected_codes.view(torch.uint8), rtol=0, atol=0)
        torch.testing.assert_close(full_scales.cpu(), expected_scales, rtol=0, atol=0)
    finally:
        ctx.close()
    assert ctx._cross_file_handles == {}
