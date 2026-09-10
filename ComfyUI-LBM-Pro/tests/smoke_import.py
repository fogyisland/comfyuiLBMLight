"""Smoke test — verify all node modules import without errors.

Run: python tests/smoke_import.py
Expected (with ComfyUI installed): all 8 modules import OK.
Without ComfyUI: only `lbm_light_preset` and `lbm_compare_grid` and the
visualizer nodes import OK; the rest require `comfy.*`.
"""
import sys
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
            # ComfyUI is optional for import-time check; only fail on real errors
            print(f"SKIP {mod} (ComfyUI not installed): {e}")
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
