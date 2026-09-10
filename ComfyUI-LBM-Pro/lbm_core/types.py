"""Custom ComfyUI type identifiers used by LBM-Pro nodes.

These strings are matched against node connection types in the
ComfyUI graph. Keeping them in one module avoids accidental typos
that would silently break node wiring.
"""
from typing import Final

LBM_MODEL_TYPE: Final[str] = "LBM_MODEL"
LIGHT_PRESET_TYPE: Final[str] = "LIGHT_PRESET"
