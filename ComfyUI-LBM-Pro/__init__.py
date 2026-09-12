"""ComfyUI-LBM-Pro — professional LBM image processing pipeline for ComfyUI.

Node modules are imported lazily on first attribute access so that
importing this package in a test/CI environment without ComfyUI does
not crash. ComfyUI's own node scanner calls `NODE_CLASS_MAPPINGS` /
`NODE_DISPLAY_NAME_MAPPINGS`, which triggers the lazy import.
"""
from __future__ import annotations

import os
import sys

# ComfyUI loads custom nodes via importlib with only the custom_nodes/<pkg>
# directory on sys.path, so sibling subpackages (`lbm_core`, `lbm_native`,
# `lbm_nodes`) cannot be imported by their top-level name. Add this package's
# directory to sys.path before any internal imports.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from lbm_core.types import LIGHT_PRESET_TYPE, LBM_MODEL_TYPE

__version__ = "0.1.7"

_NODE_MODULES = {
    "LBM_Model_Loader": "lbm_nodes.lbm_model_loader",
    "LBM_Light_Preset": "lbm_nodes.lbm_light_preset",
    "LBM_Relighting_Pro": "lbm_nodes.lbm_relighting_pro",
    "LBM_DepthNormal_Pro": "lbm_nodes.lbm_depth_normal_pro",
    "LBM_Depth_Visualizer": "lbm_nodes.lbm_depth_visualizer",
    "LBM_Normal_Visualizer": "lbm_nodes.lbm_normal_visualizer",
    "LBM_Compare_Grid": "lbm_nodes.lbm_compare_grid",
    "LBM_Batch_Processor": "lbm_nodes.lbm_batch_processor",
}

_LOADED: dict = {}


def _load_node_module(module_path: str):
    """Import a node module and cache it. Returns the module or None.

    Failure paths are logged to stderr so a typo does not silently
    disable a node (PA6 / N3 audit fix).  When
    ``LBM_PRO_DEBUG_RELOAD=1`` is set in the environment, a fresh
    import is forced so a developer iterating on a node file sees
    the change after one prompt (PA10).
    """
    if os.environ.get("LBM_PRO_DEBUG_RELOAD") == "1":
        sys.modules.pop(module_path, None)
        _LOADED.pop(module_path, None)
    mod = _LOADED.get(module_path)
    if mod is not None:
        return mod
    try:
        mod = __import__(module_path, fromlist=["*"])
    except ImportError as e:
        print(
            f"[ComfyUI-LBM-Pro] failed to import {module_path}: {e}",
            file=sys.stderr,
        )
        return None
    except Exception as e:  # noqa: BLE001
        # Non-ImportError exceptions during import (e.g. a typo in a
        # node file that surfaces as NameError) used to be silently
        # swallowed by the bare `except` clause. Log them so the user
        # can see what actually broke.
        print(
            f"[ComfyUI-LBM-Pro] unexpected error importing "
            f"{module_path}: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return None
    _LOADED[module_path] = mod
    return mod


def __getattr__(name: str):
    if name == "NODE_CLASS_MAPPINGS":
        merged: dict = {}
        for module_path in _NODE_MODULES.values():
            mod = _load_node_module(module_path)
            if mod is None:
                continue
            merged.update(getattr(mod, "NODE_CLASS_MAPPINGS", {}))
        globals()[name] = merged
        return merged
    if name == "NODE_DISPLAY_NAME_MAPPINGS":
        merged: dict = {}
        for module_path in _NODE_MODULES.values():
            mod = _load_node_module(module_path)
            if mod is None:
                continue
            merged.update(getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))
        globals()[name] = merged
        return merged
    raise AttributeError(f"module 'ComfyUI_LBM_Pro' has no attribute '{name}'")


# Optional ComfyUI ≥ 2024 type registration (silently skipped on older versions)
try:
    from comfy_execution.graph_utils import register_type  # type: ignore

    register_type(LBM_MODEL_TYPE, dict)
    register_type(LIGHT_PRESET_TYPE, dict)
except Exception:
    pass


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
