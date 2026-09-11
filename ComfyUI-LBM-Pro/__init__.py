"""ComfyUI-LBM-Pro — professional LBM image processing pipeline for ComfyUI.

Node modules are imported lazily on first attribute access so that
importing this package in a test/CI environment without ComfyUI does
not crash. ComfyUI's own node scanner calls `NODE_CLASS_MAPPINGS` /
`NODE_DISPLAY_NAME_MAPPINGS`, which triggers the lazy import.
"""
from __future__ import annotations

import os
import sys

from lbm_core.types import LIGHT_PRESET_TYPE, LBM_MODEL_TYPE

__version__ = "0.1.3"

_NODE_MODULES = {
    "LBM_Model_Loader": "nodes.lbm_model_loader",
    "LBM_Light_Preset": "nodes.lbm_light_preset",
    "LBM_Relighting_Pro": "nodes.lbm_relighting_pro",
    "LBM_DepthNormal_Pro": "nodes.lbm_depth_normal_pro",
    "LBM_Depth_Visualizer": "nodes.lbm_depth_visualizer",
    "LBM_Normal_Visualizer": "nodes.lbm_normal_visualizer",
    "LBM_Compare_Grid": "nodes.lbm_compare_grid",
    "LBM_Batch_Processor": "nodes.lbm_batch_processor",
}

_LOADED: dict[str, dict] = {}


def __getattr__(name: str):
    # PA10: when the developer sets LBM_PRO_DEBUG_RELOAD=1 in the
    # environment, force a fresh import of every node module on the
    # next attribute access.  Useful when iterating on a node file
    # without restarting ComfyUI — a single ComfyUI restart picks up
    # the new code on the next prompt.
    if os.environ.get("LBM_PRO_DEBUG_RELOAD") == "1":
        for mod_name in list(_NODE_MODULES.values()):
            sys.modules.pop(mod_name, None)
        _LOADED.clear()
    if name in ("NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"):
        kind = "NODE_CLASS_MAPPINGS" if name == "NODE_CLASS_MAPPINGS" else "NODE_DISPLAY_NAME_MAPPINGS"
        merged: dict = {}
        for cls_name, module_path in _NODE_MODULES.items():
            mod = _LOADED.get(module_path)
            if mod is None:
                try:
                    mod = __import__(module_path, fromlist=["*"])
                except ImportError as e:
                    # ComfyUI may not be available; skip this module
                    # but log so a typo doesn't disappear silently.
                    print(
                        f"[ComfyUI-LBM-Pro] failed to import {module_path}: {e}",
                        file=sys.stderr,
                    )
                    continue
                _LOADED[module_path] = mod
            attr_name = "NODE_CLASS_MAPPINGS" if kind == "NODE_CLASS_MAPPINGS" else "NODE_DISPLAY_NAME_MAPPINGS"
            mapping = getattr(mod, attr_name, {})
            merged.update(mapping)
        # Cache on the module so subsequent lookups are fast
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
