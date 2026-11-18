# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""File and writer-thread failures must not report successful conversion."""

from functools import partial
from pathlib import Path
from threading import current_thread, main_thread
import errno
import json
import logging

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.cpu]


def _mismatched_keys(src_meta, cfg, *, weight_map=None, expected_fn, mode):
    keys = sorted(expected_fn(src_meta, cfg, weight_map=weight_map))
    if mode == "missing":
        return keys + ["unexpected.predicted.weight"]
    return keys[1:]


def _assert_not_completed(case, conversion, caplog):
    for name in ("config.json", "model.safetensors.index.json", "qlean_config.json"):
        assert not (case.output / name).exists(), f"Must not publish {name} after failure"
    assert not any(m.ended_ for m in conversion.compressor.modifiers)
    assert not any(record.name == "qlean.modifiers_engine._runtime.engine"
                   and record.getMessage().startswith(("Completed:", "Index written to"))
                   for record in caplog.records)


@pytest.mark.parametrize("source_format", ["BF16", "FP8"])
def test_checkpoint_factory_writes_two_complete_shards(checkpoint_factory,
                                                       assert_tensor_equal, source_format):
    import torch
    from safetensors import safe_open

    case = checkpoint_factory(source_format=source_format, text_quant=True)
    assert {p.name for p in case.path.iterdir()} == {
        "config.json", "model.safetensors.index.json", "tokenizer_config.json", *case.shard_names,
    }
    assert json.loads((case.path / "config.json").read_text()) == case.config
    observed = {}
    for filename in case.shard_names:
        with safe_open(str(case.path / filename), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                assert name not in observed
                observed[name] = filename
                actual, expected = handle.get_tensor(name), case.tensors[name]
                assert actual.dtype == expected.dtype
                assert actual.shape == expected.shape
                if actual.dtype == torch.float8_e4m3fn:
                    actual, expected = actual.view(torch.uint8), expected.view(torch.uint8)
                assert_tensor_equal(actual, expected, scheme="checkpoint/source", seed=0)
    assert observed == case.weight_map
    assert json.loads((case.path / "model.safetensors.index.json").read_text())["weight_map"] == observed
    assert not case.output.exists()
    if source_format == "FP8":
        for name in ("model.proj.weight", "model.other.weight"):
            assert observed[name] != observed[name + "_scale_inv"]


@pytest.mark.parametrize("mode", ["missing", "extra"])
def test_predicted_key_mismatch_aborts_before_write(checkpoint_factory, build_conversion,
                                                    checkpoint_runtime, monkeypatch,
                                                    caplog, mode):
    from qlean.args import OtherArguments
    from qlean.modifiers_engine._runtime import engine

    caplog.set_level(logging.INFO, logger=engine.__name__)
    case = checkpoint_factory()
    conversion = build_conversion(case)
    original_hooks = checkpoint_runtime.new.make_hooks

    def mismatched_hooks(*args, **kwargs):
        process, predict = original_hooks(*args, **kwargs)
        return process, partial(_mismatched_keys, expected_fn=predict, mode=mode)

    writes = []

    def reject_write(tensors, path, metadata=None):
        writes.append(path)
        raise AssertionError("Mismatched keys must prevent saving")

    monkeypatch.setattr(checkpoint_runtime.new, "make_hooks", mismatched_hooks)
    monkeypatch.setattr(engine, "safe_save_file", reject_write)
    with pytest.raises(RuntimeError, match="Output keys do not match expected_output_keys") as caught:
        conversion.compressor.post_train(OtherArguments())
    assert any(filename in str(caught.value) for filename in case.shard_names)
    assert ("Missing (1)" if mode == "missing" else "Extra (1)") in str(caught.value)
    if mode == "missing":
        assert "unexpected.predicted.weight" in str(caught.value)
    assert writes == []
    assert not list(case.output.glob("*.safetensors"))
    _assert_not_completed(case, conversion, caplog)


@pytest.mark.parametrize("fail_on", [1, 2], ids=["join-before-next-write", "join-last-write"])
def test_writer_thread_error_reaches_pipeline_caller(checkpoint_factory, build_conversion,
                                                      monkeypatch, caplog, fail_on):
    from qlean.args import OtherArguments
    from qlean.modifiers_engine._runtime import engine

    caplog.set_level(logging.INFO, logger=engine.__name__)
    case = checkpoint_factory()
    conversion = build_conversion(case)
    original_save = engine.safe_save_file
    failure = OSError(errno.ENOSPC, "Injected checkpoint writer failure")
    calls = []

    def failing_save(tensors, path, metadata=None):
        calls.append((Path(path), current_thread()))
        if len(calls) == fail_on:
            raise failure
        return original_save(tensors, path, metadata=metadata)

    monkeypatch.setattr(engine, "safe_save_file", failing_save)
    with pytest.raises(OSError, match="Injected checkpoint writer failure") as caught:
        conversion.compressor.post_train(OtherArguments())
    assert caught.value is failure
    assert len(calls) == fail_on
    assert all(path.parent == case.output for path, _ in calls)
    assert all(thread is not main_thread() and not thread.is_alive() for _, thread in calls)
    assert len(list(case.output.glob("*.safetensors"))) == fail_on - 1
    _assert_not_completed(case, conversion, caplog)


@pytest.mark.parametrize("source_format,defect", [("BF16", "truncated"), ("FP8", "missing-scale")])
def test_incomplete_source_copy_cannot_report_success(checkpoint_factory, build_conversion,
                                                       caplog, source_format, defect):
    from safetensors import SafetensorError
    from qlean.args import OtherArguments

    caplog.set_level(logging.INFO, logger="qlean.modifiers_engine._runtime.engine")
    case = checkpoint_factory(source_format=source_format, defect=defect)
    assert case.path != case.pristine
    if defect == "truncated":
        assert (case.path / case.shard_names[1]).stat().st_size == (
            case.pristine / case.shard_names[1]).stat().st_size - 1
        exception, message = SafetensorError, r"(?i)(incomplete|buffer|offset)"
    else:
        assert "model.proj.weight_scale_inv" not in case.weight_map
        exception, message = ValueError, r"model\.proj\.weight:.*scale name cannot be resolved"
    conversion = build_conversion(case)
    with pytest.raises(exception, match=message):
        conversion.compressor.post_train(OtherArguments())
    _assert_not_completed(case, conversion, caplog)
