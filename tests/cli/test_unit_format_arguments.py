# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CLI argument checks and information commands without quantization."""

import argparse
import sys

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.cpu]


@pytest.mark.parametrize("lowercase", [False, True])
@pytest.mark.parametrize("value, expected", [
    ("INT8", "INT8_W8A8"), ("W8A8", "INT8_W8A8"), ("INT8_W8A8", "INT8_W8A8"),
    ("FP8-BLOCK", "FP8_BLOCK"), ("FP8_BLOCK", "FP8_BLOCK"),
    ("FP8-DYNAMIC", "FP8_DYNAMIC"), ("FP8_DYNAMIC", "FP8_DYNAMIC"),
    ("FP8-CHANNEL", "FP8_DYNAMIC"), ("FP8_CHANNEL", "FP8_DYNAMIC"),
    ("MXFP4", "MXFP4"), ("BF16", "BF16"),
])
def test_output_format_aliases(value, expected, lowercase):
    from qlean.cli.__main__ import parse_output_format

    assert parse_output_format(value.lower() if lowercase else value) == expected


@pytest.mark.parametrize("lowercase", [False, True])
@pytest.mark.parametrize("value, expected", [("BF16", "BF16"), ("FP8", "FP8"), ("NVFP4", "NVFP4")])
def test_input_format_aliases(value, expected, lowercase):
    from qlean.cli.__main__ import parse_input_format

    assert parse_input_format(value.lower() if lowercase else value) == expected


@pytest.mark.parametrize("value", ["not-a-format", "Fp8", "", " BF16 "])
@pytest.mark.parametrize("parser_name", ["parse_input_format", "parse_output_format"])
def test_unknown_format_is_rejected(parser_name, value):
    from qlean.cli import __main__ as cli

    with pytest.raises(argparse.ArgumentTypeError, match="Unknown .* format"):
        getattr(cli, parser_name)(value)


@pytest.mark.parametrize("present_mask", range(7))
def test_required_arguments_fail_before_quantization(monkeypatch, capsys, tmp_path, present_mask):
    from qlean.cli import __main__ as cli

    names = ("model_name", "model_path", "save_path")
    argv = ["qlean"]
    for index, name in enumerate(names):
        if present_mask & (1 << index):
            argv.extend((f"--{name}", str(tmp_path / name)))
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(cli, "setup_logging", lambda **kwargs: pytest.fail("Quantization must not start"))

    with pytest.raises(SystemExit) as error:
        cli.main()

    assert error.value.code == 2
    message = capsys.readouterr().err.split("error:")[-1]
    assert "required" in message
    for index, name in enumerate(names):
        assert (f"--{name}" in message) == (not bool(present_mask & (1 << index)))


@pytest.mark.parametrize("arguments, text", [
    (["--list-formats"], "Supported conversions"),
    (["--list-models"], "Toy-8B"),
    (["--list-models", "org/Toy-8B"], "has a dedicated recipe"),
    (["--examples"], "Examples:"),
    (["--help"], "Required Arguments"),
])
def test_info_commands_need_no_model_and_do_not_quantize(
    tmp_path, monkeypatch, capsys, arguments, text,
):
    import builtins
    import importlib.resources
    from qlean.cli import __main__ as cli

    configs = tmp_path / "configs"
    (configs / "day0").mkdir(parents=True)
    (configs / "mxfp4").mkdir()
    (configs / "day0" / "Toy-8B.yaml").write_text("{}", encoding="utf-8")
    original_files = importlib.resources.files
    if arguments[0] == "--list-models":
        monkeypatch.setattr(importlib.resources, "files", lambda package: configs)
    monkeypatch.setattr(sys, "argv", ["qlean", *arguments])
    monkeypatch.setattr(cli, "setup_logging", lambda **kwargs: pytest.fail("Quantization must not start"))
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in ("qlean.entrypoints", "qlean.utils"):
            pytest.fail(f"Info commands must not import quantization entrypoints: {name}")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", guarded_import)
        if arguments == ["--help"]:
            with pytest.raises(SystemExit) as error:
                cli.main()
            assert error.value.code == 0
        else:
            assert cli.main() is None
    output = capsys.readouterr()
    assert text in output.out
    assert output.err == ""
    assert set(tmp_path.iterdir()) == {configs}
    auto_recipe = sys.modules.get("qlean.core.auto_recipe")
    if auto_recipe is not None:
        assert auto_recipe.files is original_files
