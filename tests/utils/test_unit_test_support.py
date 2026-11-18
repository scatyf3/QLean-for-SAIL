# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Self-checks for shared fixtures, resource markers, and failure diagnostics."""

import ast
from contextlib import contextmanager, nullcontext
from inspect import unwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.cpu]


@pytest.fixture
def support(request):
    target = Path(__file__).resolve().parents[1] / "conftest.py"
    plugins = [plugin for plugin in request.config.pluginmanager.get_plugins()
               if getattr(plugin, "__file__", None)
               and Path(plugin.__file__).resolve() == target]
    assert len(plugins) == 1
    return plugins[0]


def _item(markers, require_gpu=False):
    return SimpleNamespace(
        get_closest_marker=lambda name: object() if name in markers else None,
        config=SimpleNamespace(getoption=lambda name: require_gpu),
    )


@pytest.mark.parametrize("markers, message", [
    (["unit"], "one layer marker and one cpu/gpu marker"),
    (["unit", "cpu", "gpu"], "one layer marker and one cpu/gpu marker"),
    (["unit", "integration", "cpu"], "one layer marker and one cpu/gpu marker"),
    (["unit", "cpu", "multi_gpu"], "multi_gpu requires the gpu marker"),
])
def test_marker_conflicts_fail_before_device_probe(support, monkeypatch, markers, message):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("Conflicting markers must not trigger GPU detection"))
    with pytest.raises(pytest.fail.Exception, match=message):
        support.pytest_runtest_setup(_item(markers))


@pytest.mark.parametrize("layer", ["unit", "integration", "e2e"])
def test_cpu_marker_does_not_probe_cuda(support, monkeypatch, layer):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CPU tests must not trigger GPU detection"))
    support.pytest_runtest_setup(_item([layer, "cpu"]))


@pytest.mark.parametrize("multi_gpu, cuda_available, count, missing", [
    (False, False, 8, True), (False, True, 0, True), (False, True, 1, False),
    (True, False, 8, True), (True, True, 1, True), (True, True, 2, False),
])
@pytest.mark.parametrize("require_gpu", [False, True])
def test_gpu_resource_policy(support, monkeypatch, multi_gpu, cuda_available, count, missing, require_gpu):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)

    def device_count():
        assert cuda_available, "Do not query device count when CUDA is unavailable"
        return count

    monkeypatch.setattr(torch.cuda, "device_count", device_count)
    markers = ["unit", "gpu"] + (["multi_gpu"] if multi_gpu else [])
    item = _item(markers, require_gpu)
    if missing:
        error = pytest.fail.Exception if require_gpu else pytest.skip.Exception
        required, available = (2 if multi_gpu else 1), (count if cuda_available else 0)
        with pytest.raises(error, match=f"Requires {required} CUDA GPUs; found {available}"):
            support.pytest_runtest_setup(item)
    else:
        support.pytest_runtest_setup(item)


def test_pytest_configuration_and_gpu_option(support):
    from configparser import ConfigParser
    import shlex

    config = ConfigParser()
    config.read(Path(__file__).resolve().parents[2] / "pytest.ini", encoding="utf-8")
    settings = config["pytest"]
    assert "--strict-markers" in shlex.split(settings["addopts"])
    assert shlex.split(settings["testpaths"]) == ["tests"]
    markers = {line.partition(":")[0].strip() for line in settings["markers"].splitlines() if line.strip()}
    assert {"unit", "integration", "e2e", "cpu", "gpu", "multi_gpu"} <= markers
    options = []
    support.pytest_addoption(SimpleNamespace(addoption=lambda *args, **kwargs: options.append((args, kwargs))))
    assert len(options) == 1
    assert options[0][0] == ("--require-gpu",)
    assert options[0][1]["action"] == "store_true"
    assert options[0][1]["default"] is False


@pytest.mark.parametrize("raise_inside", [False, True])
def test_isolated_state_restores_on_normal_and_exception_exit(support, raise_inside):
    import torch
    from qlean.core.event_lifecycle import EventLifecycle
    from qlean.utils.singleton import SingletonMixin
    from qlean.models import ModelFactory
    from qlean.modifiers import ModifierFactory

    original_dtype = torch.get_default_dtype()
    instances, sentinel = {"outside": object()}, object()
    snapshots = []
    try:
        torch.set_default_dtype(torch.float64)
        with pytest.MonkeyPatch.context() as outer:
            outer.setattr(SingletonMixin, "_instances", instances)
            outer.setattr(EventLifecycle, "initialized", True)
            outer.setattr(EventLifecycle, "finalized", True)
            for factory in (ModelFactory, ModifierFactory):
                registry, errors = {"outside": sentinel}, {"outside": RuntimeError("outside")}
                snapshots.append((factory, registry, errors))
                outer.setattr(factory, "_main_registry", registry)
                outer.setattr(factory, "_errors", errors)
                outer.setattr(factory, "_loaded", True)
            outcome = pytest.raises(RuntimeError, match="Controlled exit") if raise_inside else nullcontext()
            with outcome:
                with contextmanager(unwrap(support.isolated_state))():
                    assert SingletonMixin._instances == {}
                    assert not EventLifecycle.initialized and not EventLifecycle.finalized
                    for factory, registry, errors in snapshots:
                        assert factory._main_registry == registry and factory._main_registry is not registry
                        assert factory._errors == errors and factory._errors is not errors
                        factory._main_registry.clear()
                        factory._errors.clear()
                        factory._loaded = False
                    SingletonMixin._instances["inside"] = object()
                    EventLifecycle.initialized = True
                    torch.set_default_dtype(torch.float32)
                    if raise_inside:
                        raise RuntimeError("Controlled exit")
            assert SingletonMixin._instances is instances
            assert set(instances) == {"outside"}
            assert EventLifecycle.initialized and EventLifecycle.finalized
            assert torch.get_default_dtype() == torch.float64
            for factory, registry, errors in snapshots:
                assert factory._main_registry is registry and registry == {"outside": sentinel}
                assert factory._errors is errors and set(errors) == {"outside"}
                assert factory._loaded is True
    finally:
        torch.set_default_dtype(original_dtype)


@pytest.mark.parametrize("raise_inside", [False, True])
def test_local_environment_and_file_handle_are_restored(tmp_path, monkeypatch, raise_inside):
    import os

    key = "QLEAN_UT_SCOPED_ENV"
    monkeypatch.setenv(key, "outside")
    outcome = pytest.raises(RuntimeError, match="Controlled exit") if raise_inside else nullcontext()
    with outcome:
        with pytest.MonkeyPatch.context() as patch, (tmp_path / "probe.txt").open("w") as handle:
            patch.setenv(key, "inside")
            handle.write("local")
            assert os.environ[key] == "inside" and not handle.closed
            if raise_inside:
                raise RuntimeError("Controlled exit")
    assert os.environ[key] == "outside"
    assert handle.closed
    assert (tmp_path / "probe.txt").read_text() == "local"


@pytest.mark.parametrize("defect", ["value", "shape", "dtype"])
def test_tensor_diagnostics_include_context_and_difference(assert_tensor_equal, defect):
    import torch

    expected = torch.ones((2, 3), dtype=torch.float32)
    actual = expected.clone()
    if defect == "value":
        actual[1, 2] = 3
    elif defect == "shape":
        actual = actual[:1]
    else:
        actual = actual.to(torch.bfloat16)
    with pytest.raises(AssertionError) as error:
        assert_tensor_equal(actual, expected, scheme="W8A8", seed=37)
    text = str(error.value)
    for fragment in ("scheme=W8A8", f"shape={tuple(actual.shape)}", f"dtype={actual.dtype}", "seed=37"):
        assert fragment in text
    if defect == "value":
        assert "(1, 2)" in text and "Greatest absolute difference" in text
    else:
        assert f"'{defect}'" in text
    assert_tensor_equal(expected, expected.clone(), scheme="W8A8", seed=37)


def test_unit_modules_defer_known_heavy_imports():
    root = Path(__file__).resolve().parents[1]
    heavy = {"qlean", "torch", "triton", "safetensors", "numpy", "transformers", "acext"}
    violations = []

    class Imports(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            for expression in [*node.decorator_list, *node.args.defaults, *node.args.kw_defaults]:
                if expression is not None:
                    self.visit(expression)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Import(self, node):
            for alias in node.names:
                if alias.name.split(".")[0] in heavy:
                    violations.append((str(path.relative_to(root)), node.lineno, alias.name))

        def visit_ImportFrom(self, node):
            if (node.module or "").split(".")[0] in heavy:
                violations.append((str(path.relative_to(root)), node.lineno, node.module))

    paths = sorted(root.rglob("test_unit_*.py")) + [root / "conftest.py"]
    assert paths
    for path in paths:
        Imports().visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
    assert violations == [], f"Known heavy imports during collection: {violations}"
