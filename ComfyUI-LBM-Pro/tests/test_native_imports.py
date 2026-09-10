"""Tests that the rewritten ``lbm_native`` package imports cleanly.

The package must be importable in a vanilla environment (no
ComfyUI, no GPU).  The diffusers dependency is required and lives
in ``requirements.txt``.
"""
from __future__ import annotations


def test_package_imports():
    import lbm_native

    assert lbm_native.__all__ is not None
    assert len(lbm_native.__all__) >= 10


def test_all_public_symbols_resolve():
    import lbm_native as pkg

    for name in pkg.__all__:
        obj = getattr(pkg, name)
        assert obj is not None, f"{name} should resolve"


def test_independent_module_path():
    """No symbol should resolve back to the legacy ``lbm.*`` namespace."""
    import inspect
    import lbm_native

    for name in lbm_native.__all__:
        obj = getattr(lbm_native, name)
        try:
            source_file = inspect.getfile(obj)
        except (TypeError, OSError):
            # builtin / C-extension — skip; cannot infer source location
            continue
        assert "lbm_native" in source_file, (
            f"{name} resolves to file '{source_file}'; expected it to live under 'lbm_native/'"
        )


def test_legacy_namespace_removed():
    """The original ``lbm`` package must be gone from the project root."""
    import importlib
    import sys

    # Clean any cached `lbm` module so we can probe fresh
    for cached in list(sys.modules.keys()):
        if cached == "lbm" or cached.startswith("lbm."):
            del sys.modules[cached]
    spec = importlib.util.find_spec("lbm")
    assert spec is None, "legacy 'lbm' package is still importable; it should be gone"
