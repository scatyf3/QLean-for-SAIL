# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Recipe selection through real routing and factory creation."""

import json
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.cpu, pytest.mark.usefixtures("isolated_state")]


def test_package_yaml_resources_parse():
    from importlib.resources import files
    import yaml

    pending = [files("qlean")]
    parsed = []
    while pending:
        for resource in pending.pop().iterdir():
            if resource.is_dir():
                pending.append(resource)
            elif resource.name.endswith((".yaml", ".yml")):
                yaml.safe_load(resource.read_text(encoding="utf-8"))
                parsed.append(str(resource))
    assert parsed, "No package YAML resources found"


@pytest.fixture
def routing(monkeypatch):
    from qlean.core import utils
    from qlean.modifiers import Modifier, ModifierFactory

    class ProbeModifier(Modifier):
        def __init__(self, **kwargs):
            self.params = kwargs

        def on_initialize(self, state):
            self.initialized_ = True

    names = ("generalDay0Modifier", "dequantModifier", "fp8BasedDay0Modifier",
             "nvfp4DequantModifier", "nvfp4BasedMxfp4Day0Modifier", "GeneralQuantEngineModifier")
    classes = {name: type(name, (ProbeModifier,), {}) for name in names}
    monkeypatch.setattr(ModifierFactory, "_main_registry", classes)
    monkeypatch.setattr(ModifierFactory, "_errors", {})
    monkeypatch.setattr(ModifierFactory, "_loaded", True)
    return utils, classes


@pytest.mark.parametrize("recipe_name", ["Toy.yaml", "_arch/ToyModel.yaml"])
def test_recipe_selection_creates_ordered_modifiers(
    tmp_path, monkeypatch, recipe_factory, routing, recipe_name,
):
    utils, classes = routing
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"architectures": ["ToyModel"]}))
    resources = tmp_path / "configs"
    recipe_path = resources / "day0" / recipe_name
    recipe_path.parent.mkdir(parents=True)
    recipe_path.write_text(json.dumps(recipe_factory(modifier="generalDay0Modifier")))
    monkeypatch.setattr(utils, "files", lambda package: resources)

    def unexpected_detection(*args):
        pytest.fail("Explicit input formats must bypass detection")

    monkeypatch.setattr(utils, "detect_input_format", unexpected_detection)
    args = SimpleNamespace(
        recipe=None, model_path=str(source), model_name="org/Toy",
        input_format="FP8", output_format="FP8_DYNAMIC", verify=None, mix_path=None,
    )
    modifiers, meta = utils.get_modifiers_from_recipe("day0", args)

    expected_types = [classes["dequantModifier"], classes["fp8BasedDay0Modifier"]]
    assert [type(modifier) for modifier in modifiers] == expected_types
    assert [modifier.params for modifier in modifiers] == [
        {"scheme": "FP8_DYNAMIC", "ignore": [], "stage": "quant", "group": "quant"},
        {"scheme": "FP8_DYNAMIC", "ignore": [], "stage": "quant", "group": "quant"},
    ]
    assert meta["recipe_path"] == str(recipe_path)
    assert meta["scheme"] == "FP8_DYNAMIC"
    assert meta["modifier_names"] == ["dequantModifier", "fp8BasedDay0Modifier"]


def test_manual_recipe_keeps_modifier_and_scheme(tmp_path, monkeypatch, recipe_factory, routing):
    utils, classes = routing
    recipe_path = tmp_path / "manual.yaml"
    recipe_path.write_text(json.dumps(recipe_factory(modifier="generalDay0Modifier")))

    def unexpected_route(*args):
        pytest.fail("Custom recipes must bypass automatic routing")

    monkeypatch.setattr(utils, "resolve_recipe", unexpected_route)
    args = SimpleNamespace(recipe=str(recipe_path), input_format="FP8",
                           output_format="FP8_DYNAMIC", verify=None)
    modifiers, meta = utils.get_modifiers_from_recipe("day0", args)

    assert [type(modifier) for modifier in modifiers] == [classes["generalDay0Modifier"]]
    assert modifiers[0].params["scheme"] == "W8A8"
    assert meta["input_format"] == "user_specified"


def test_no_recipe_match_uses_generated_engine(tmp_path, monkeypatch, recipe_factory, routing):
    from qlean.core import auto_recipe

    utils, classes = routing
    monkeypatch.setattr(utils, "files", lambda package: tmp_path / "empty_configs")
    calls = []

    def generate(input_format, output_format, model_path, architecture=None):
        calls.append((input_format, output_format, model_path, architecture))
        return recipe_factory()

    monkeypatch.setattr(auto_recipe, "build_auto_recipe", generate)
    args = SimpleNamespace(
        recipe=None, model_path=str(tmp_path), model_name="Unknown",
        input_format="BF16", output_format="INT8_W8A8", verify="bound", mix_path=None,
    )
    modifiers, meta = utils.get_modifiers_from_recipe("day0", args)

    assert calls == [("BF16", "INT8_W8A8", str(tmp_path), None)]
    assert [type(modifier) for modifier in modifiers] == [classes["GeneralQuantEngineModifier"]]
    assert modifiers[0].params["primary_scheme"] == "W8A8"
    assert modifiers[0].params["verify"] == "bound"
    assert meta["recipe_path"] == "generated:GeneralQuantEngineModifier"


@pytest.mark.parametrize("scheme, expected_dst", [
    ("FP8_DYNAMIC", ["bf16", "passthrough", "FP8_DYNAMIC"]),
    ("MXFP4", ["bf16", "passthrough", "W8A8"]),
])
def test_scheme_override_preserves_fixed_regions(recipe_factory, routing, scheme, expected_dst):
    utils, _ = routing
    recipe = recipe_factory(regions=[
        {"name": "decode", "match": "a.*", "src": "fp8_block", "dst": "bf16"},
        {"name": "keep", "match": "b.*", "dst": "passthrough"},
        {"name": "linear", "match": "*.weight", "dst": "W8A8"},
        {"name": "fallback", "match": "*"},
    ])
    utils.update_scheme(recipe, scheme)
    params = recipe["quant_stage"]["quant_modifiers"]["GeneralQuantEngineModifier"]
    assert params["primary_scheme"] == scheme
    assert [region["dst"] for region in params["regions"][:-1]] == expected_dst
    assert "dst" not in params["regions"][-1]


@pytest.mark.parametrize("input_format, output_format, expected_names, scheme", [
    ("BF16", "INT8_W8A8", ["generalDay0Modifier"], "W8A8"),
    ("BF16", "FP8_BLOCK", ["generalDay0Modifier"], "FP8_BLOCK"),
    ("BF16", "FP8_DYNAMIC", ["generalDay0Modifier"], "FP8_DYNAMIC"),
    ("BF16", "MXFP4", ["generalDay0Modifier"], "MXFP4"),
    ("FP8", "BF16", ["dequantModifier"], "W8A8"),
    ("FP8", "INT8_W8A8", ["dequantModifier", "fp8BasedDay0Modifier"], "W8A8"),
    ("FP8", "FP8_BLOCK", ["dequantModifier", "fp8BasedDay0Modifier"], "FP8_BLOCK"),
    ("FP8", "FP8_DYNAMIC", ["dequantModifier", "fp8BasedDay0Modifier"], "FP8_DYNAMIC"),
    ("NVFP4", "BF16", ["nvfp4DequantModifier"], "W8A8"),
    ("NVFP4", "MXFP4", ["nvfp4DequantModifier", "nvfp4BasedMxfp4Day0Modifier"], "MXFP4"),
])
def test_public_route_matrix_creates_expected_chain(
    tmp_path, monkeypatch, recipe_factory, routing, input_format, output_format, expected_names, scheme,
):
    utils, classes = routing
    for directory in ("day0", "mxfp4"):
        path = tmp_path / directory / "Toy.yaml"
        path.parent.mkdir()
        path.write_text(json.dumps(recipe_factory(modifier="generalDay0Modifier")), encoding="utf-8")
    monkeypatch.setattr(utils, "files", lambda package: tmp_path)
    args = SimpleNamespace(recipe=None, model_path=str(tmp_path), model_name="org/Toy",
                           input_format=input_format, output_format=output_format,
                           verify=None, mix_path=None)
    modifiers, meta = utils.get_modifiers_from_recipe("day0", args)
    assert [type(mod) for mod in modifiers] == [classes[name] for name in expected_names]
    assert [mod.params for mod in modifiers] == [
        {"scheme": scheme, "ignore": [], "stage": "quant", "group": "quant"}
        for _ in expected_names
    ]
    assert meta["modifier_names"] == expected_names


@pytest.mark.parametrize("fallback, input_format, output_format", [
    (False, "BF16", "BF16"), (False, "GPTQ", "INT8_W8A8"),
    (True, "NVFP4", "INT8_W8A8"),
])
def test_unsupported_auto_conversion_fails_before_factory(
    tmp_path, monkeypatch, recipe_factory, routing, fallback, input_format, output_format,
):
    utils, _ = routing
    from qlean.modifiers import ModifierFactory

    (tmp_path / "config.json").write_text(json.dumps({
        "architectures": ["ToyModel"], "quantization_config": {"quant_method": input_format.lower()},
    }), encoding="utf-8")
    if fallback:
        recipe_path = tmp_path / "day0" / "_arch" / "ToyModel.yaml"
        recipe_path.parent.mkdir(parents=True)
        recipe_path.write_text(json.dumps(recipe_factory(modifier="generalDay0Modifier")), encoding="utf-8")
    monkeypatch.setattr(utils, "files", lambda package: tmp_path)
    monkeypatch.setattr(ModifierFactory, "create", lambda *a, **kw: pytest.fail("Invalid conversions must not create a Modifier"))
    args = SimpleNamespace(recipe=None, model_path=str(tmp_path), model_name="org/Unknown",
                           input_format=None, output_format=output_format, verify=None, mix_path=None)
    with pytest.raises(ValueError, match="Unsupported .*conversion") as error:
        utils.get_modifiers_from_recipe("day0", args)
    assert input_format in str(error.value)
    assert output_format in str(error.value)


@pytest.mark.parametrize("entry", ["day0", "custom"])
@pytest.mark.parametrize("with_engine", [False, True])
def test_verify_override_leaves_non_engine_blocks_unchanged(
    tmp_path, recipe_factory, routing, caplog, entry, with_engine,
):
    utils, classes = routing
    recipe = recipe_factory(modifier="generalDay0Modifier")
    if with_engine:
        recipe["quant_stage"]["quant_modifiers"]["GeneralQuantEngineModifier"] = {"verify": "off"}
    path = tmp_path / "recipe.yaml"
    original = json.dumps(recipe)
    path.write_text(original, encoding="utf-8")
    args = SimpleNamespace(recipe=str(path), verify="bound")
    modifiers, meta = utils.get_modifiers_from_recipe(entry, args)
    assert type(modifiers[0]) is classes["generalDay0Modifier"]
    assert modifiers[0].params == {"scheme": "W8A8", "ignore": [], "stage": "quant", "group": "quant"}
    assert meta["recipe_content"]["quant_stage"]["quant_modifiers"]["generalDay0Modifier"] == {
        "scheme": "W8A8", "ignore": [],
    }
    if with_engine:
        assert len(modifiers) == 2
        assert modifiers[1].params == {"verify": "bound", "stage": "quant", "group": "quant"}
    else:
        assert len(modifiers) == 1
        assert "ignoring" in caplog.text
    assert path.read_text(encoding="utf-8") == original
