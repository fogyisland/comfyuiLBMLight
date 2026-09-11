"""Core modules for ComfyUI-LBM-Pro.

Importing this package does NOT eagerly load `model_factory` because that
module is heavy (it builds the LBM solver graph on first import). Nodes
that need `build_lbm_model`/`load_lbm_checkpoint` should import them
directly from `lbm_core.model_factory`.
"""
from .cache import LBMModelCache
from .presets import PRESETS, LightPreset, build_custom_preset
from .types import LIGHT_PRESET_TYPE, LBM_MODEL_TYPE
from .visualizers import COLORMAPS, depth_to_colormap, normalize_normal_map

__all__ = [
    "LBM_MODEL_TYPE",
    "LIGHT_PRESET_TYPE",
    "LightPreset",
    "PRESETS",
    "build_custom_preset",
    "LBMModelCache",
    "COLORMAPS",
    "depth_to_colormap",
    "normalize_normal_map",
]
