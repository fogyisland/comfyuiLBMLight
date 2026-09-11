"""Smoke test — verify all node modules import without errors.

Run: python tests/smoke_import.py
Expected (with ComfyUI installed): all 8 modules import OK.
Without ComfyUI: only `lbm_light_preset` and `lbm_compare_grid` and the
visualizer nodes import OK; the rest require `comfy.*`.
"""
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXPECTED_CLASSES = {
    "LBM_Model_Loader",
    "LBM_Light_Preset",
    "LBM_Relighting_Pro",
    "LBM_DepthNormal_Pro",
    "LBM_Depth_Visualizer",
    "LBM_Normal_Visualizer",
    "LBM_Compare_Grid",
    "LBM_Batch_Processor",
}

# Modules that do NOT need ComfyUI at class-definition time
COMFYUI_FREE_MODULES = [
    "nodes.lbm_light_preset",
    "nodes.lbm_compare_grid",
    "nodes.lbm_depth_visualizer",
    "nodes.lbm_normal_visualizer",
]

# Modules that DO need ComfyUI (model loading)
COMFYUI_MODULES = [
    "nodes.lbm_model_loader",
    "nodes.lbm_relighting_pro",
    "nodes.lbm_depth_normal_pro",
    "nodes.lbm_batch_processor",
]

# Names that indicate a "ComfyUI is missing" import failure rather than a
# genuine bug in the node module. Only these get the SKIP treatment.
_COMFYUI_MODULE_NAMES = {"comfy", "folder_paths", "comfy_execution", "comfy_extras"}


def main() -> int:
    failed = []
    for mod in COMFYUI_FREE_MODULES:
        try:
            __import__(mod)
            print(f"OK  {mod}")
        except Exception as e:
            print(f"FAIL {mod}: {type(e).__name__}: {e}")
            failed.append(mod)

    for mod in COMFYUI_MODULES:
        try:
            __import__(mod)
            print(f"OK  {mod}")
        except ImportError as e:
            missing = e.name if hasattr(e, "name") else None
            if missing in _COMFYUI_MODULE_NAMES or "comfy" in str(e):
                print(f"SKIP {mod} (ComfyUI not installed): {e}")
            else:
                print(f"FAIL {mod}: ImportError not related to ComfyUI: {e}")
                traceback.print_exc()
                failed.append(mod)
        except Exception as e:
            print(f"FAIL {mod}: {type(e).__name__}: {e}")
            failed.append(mod)

    if failed:
        print(f"\n{len(failed)} modules failed.")
        return 1
    print("\nAll node modules importable (or skipped due to missing ComfyUI).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
