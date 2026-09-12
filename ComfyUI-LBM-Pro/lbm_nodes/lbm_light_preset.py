"""LBM Light Preset — choose a lighting style or build a custom one."""
from __future__ import annotations

from lbm_core import LIGHT_PRESET_TYPE, PRESETS, build_custom_preset


class LBM_Light_Preset:
    """Emit a LIGHT_PRESET dict for use by Relighting Pro."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["preset", "custom"], {"default": "preset"}),
            },
            "optional": {
                "preset_name": (
                    list(PRESETS.keys()),
                    {"default": "warm_neutral"},
                ),
                "azimuth_deg": (
                    "FLOAT",
                    {"default": 45.0, "min": 0.0, "max": 360.0, "step": 1.0},
                ),
                "elevation_deg": (
                    "FLOAT",
                    {"default": 45.0, "min": 0.0, "max": 90.0, "step": 1.0},
                ),
                "intensity": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "temperature_k": (
                    "INT",
                    {"default": 5500, "min": 1000, "max": 40000, "step": 100},
                ),
            },
        }

    RETURN_TYPES = (LIGHT_PRESET_TYPE,)
    RETURN_NAMES = ("light_preset",)
    FUNCTION = "build"
    CATEGORY = "🧪BMLab/🔆LBM-Pro"

    def build(
        self,
        mode: str,
        preset_name: str = "warm_neutral",
        azimuth_deg: float = 45.0,
        elevation_deg: float = 45.0,
        intensity: float = 1.0,
        temperature_k: int = 5500,
    ) -> tuple[dict]:
        if mode == "preset":
            preset = PRESETS.get(preset_name)
            if preset is None:
                raise ValueError(f"Unknown preset '{preset_name}'")
        else:
            preset = build_custom_preset(
                azimuth_deg, elevation_deg, intensity, float(temperature_k)
            )
        payload = {
            "name": preset.name,
            "rgb_tint": preset.rgb_tint,
            "intensity": preset.intensity,
            "bridge_noise_sigma": preset.bridge_noise_sigma,
            "description": preset.description,
        }
        return (payload,)


NODE_CLASS_MAPPINGS = {"LBM_Light_Preset": LBM_Light_Preset}
NODE_DISPLAY_NAME_MAPPINGS = {"LBM_Light_Preset": "LBM Light Preset"}
