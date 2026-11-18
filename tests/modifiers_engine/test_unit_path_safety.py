# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reject unsafe paths before preflight, worker startup, or writes."""

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.cpu]


def _unused_callback(*args, **kwargs):
    raise AssertionError("Path safety tests must not run conversion callbacks")


@pytest.mark.parametrize("relation, message", [
    ("same", "save_path equals model_path"),
    ("child", "is inside model_path"),
    ("parent", "is inside save_path"),
    ("symlink", "save_path equals model_path"),
])
def test_dangerous_paths_leave_source_untouched(tmp_path, monkeypatch, relation, message):
    from qlean.modifiers_engine._runtime import engine

    source = tmp_path / "source"
    source.mkdir()
    sentinel = source / "weights.safetensors"
    sentinel.write_bytes(b"source-must-not-change")
    if relation == "same":
        target = source
    elif relation == "child":
        target = source / "output"
    elif relation == "parent":
        target = tmp_path
    else:
        target = tmp_path / "alias"
        target.symlink_to(source, target_is_directory=True)

    def unexpected_preflight(*args, **kwargs):
        pytest.fail("Unsafe paths must be rejected before preflight")

    monkeypatch.setattr(engine, "preflight", unexpected_preflight)
    spec = engine.ShardRunSpec(
        model_path=str(source), save_path=str(target), source_files=[], weight_map={},
        process_tensor=_unused_callback, expected_output_keys=_unused_callback,
        finalize=_unused_callback,
    )
    with pytest.raises(ValueError, match=message):
        engine.ShardEngine().run(spec)
    assert sentinel.read_bytes() == b"source-must-not-change"
    assert set(source.iterdir()) == {sentinel}


@pytest.mark.parametrize("field", [
    "process_tensor", "expected_output_keys", "finalize", "filter_source_key", "lazy_input",
])
def test_unpickleable_callback_fails_before_worker(tmp_path, field):
    from qlean.modifiers_engine._runtime.engine import ShardRunSpec

    params = dict(model_path=str(tmp_path / "source"), save_path=str(tmp_path / "output"),
                  source_files=[], weight_map={}, process_tensor=_unused_callback,
                  expected_output_keys=_unused_callback, finalize=_unused_callback)
    params[field] = lambda *args: []
    with pytest.raises(TypeError, match=f"{field} must be a pickleable"):
        ShardRunSpec(**params)
    assert list(tmp_path.iterdir()) == []


def test_unpickleable_dequant_provider_fails_before_worker(tmp_path):
    from qlean.modifiers_engine._runtime.engine import ShardRunSpec

    with pytest.raises(TypeError, match="dequant_map.*callable provider.*pickleable"):
        ShardRunSpec(
            model_path=str(tmp_path / "source"), save_path=str(tmp_path / "output"),
            source_files=[], weight_map={}, process_tensor=_unused_callback,
            expected_output_keys=_unused_callback, finalize=_unused_callback,
            dequant_map={"*.weight": lambda *args: None},
        )
    assert list(tmp_path.iterdir()) == []


def test_module_callbacks_and_plain_configuration_are_pickleable(tmp_path):
    import pickle
    from qlean.modifiers_engine._runtime.engine import ShardRunSpec

    spec = ShardRunSpec(
        model_path=str(tmp_path / "source"), save_path=str(tmp_path / "output"),
        source_files=[], weight_map={}, process_tensor=_unused_callback,
        expected_output_keys=_unused_callback, finalize=_unused_callback,
        filter_source_key=_unused_callback, lazy_input=_unused_callback,
        dequant_map={"*.weight": _unused_callback}, transform_cfg={"nested": {"values": [1, 2]}},
    )
    restored = pickle.loads(pickle.dumps(spec))
    for field in ("process_tensor", "expected_output_keys", "finalize", "filter_source_key", "lazy_input"):
        assert getattr(restored, field) is _unused_callback
    assert restored.dequant_map["*.weight"] is _unused_callback
    assert restored.transform_cfg == spec.transform_cfg
