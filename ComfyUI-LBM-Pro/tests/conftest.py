"""Shared pytest fixtures and module-level stubs.

The ``lbm_core.model_factory`` module pulls in ``comfy.utils`` at import
time (see C24 — moving ``load_torch_file`` to module top).  Test files
that import anything from ``lbm_core.model_factory`` therefore need a
``comfy.utils`` stub on ``sys.modules`` BEFORE the import.

The autouse fixture below installs a minimal ``comfy`` + ``comfy.utils``
stub once per test, so individual tests do not have to repeat the
boilerplate.  Tests that need richer behaviour (e.g. capturing
downloads) can still override by re-patching ``sys.modules``.
"""
from __future__ import annotations

import sys
import types

import pytest


def _install_stub_comfy() -> None:
    """Insert a fake ``comfy`` package with ``comfy.utils.load_torch_file``."""
    fake_comfy = types.ModuleType("comfy")
    fake_comfy.__path__ = []  # mark as a package
    fake_utils = types.ModuleType("comfy.utils")

    def _stub_load_torch_file(ckpt_path, device=None, safe_load=None):
        # Default: empty state dict.  Tests that exercise loading
        # paths monkeypatch this stub to return controlled values.
        return {}

    fake_utils.load_torch_file = _stub_load_torch_file
    # Link the child back to the parent so the import system recognises
    # ``comfy.utils`` as a submodule of ``comfy`` rather than a
    # top-level orphan (the latter raises "unknown location" when
    # ``from comfy.utils import ...`` is executed).
    fake_utils.__package__ = "comfy"
    setattr(fake_comfy, "utils", fake_utils)
    sys.modules["comfy"] = fake_comfy
    sys.modules["comfy.utils"] = fake_utils


@pytest.fixture(autouse=True)
def _stub_comfy_modules():
    """Install ``comfy.utils.load_torch_file`` for every test.

    If ``comfy`` is already on ``sys.modules`` (e.g. the user's
    real ComfyUI install is importable), do nothing.  Otherwise
    inject a stub that returns ``{}`` — tests that exercise
    loading paths override ``load_torch_file`` directly.
    """
    if "comfy" in sys.modules and hasattr(sys.modules["comfy"], "utils"):
        # Already present (real install or earlier test set it up).
        yield
        return

    saved_comfy = sys.modules.get("comfy")
    saved_utils = sys.modules.get("comfy.utils")
    _install_stub_comfy()
    # Force a re-import of ``lbm_core.model_factory`` so it picks up
    # the stub at module top.
    sys.modules.pop("lbm_core.model_factory", None)

    try:
        yield
    finally:
        # Restore the original sys.modules state.
        sys.modules.pop("comfy.utils", None)
        sys.modules.pop("comfy", None)
        if saved_comfy is not None:
            sys.modules["comfy"] = saved_comfy
        if saved_utils is not None:
            sys.modules["comfy.utils"] = saved_utils
        # Force model_factory to be re-imported on next test so the
        # next autouse fixture call can re-stub it.
        sys.modules.pop("lbm_core.model_factory", None)
