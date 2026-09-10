"""LBM Normal Visualizer — validate and re-normalize a normal map."""
from __future__ import annotations

import numpy as np
import torch

from lbm_core.visualizers import normalize_normal_map


class LBM_Normal_Visualizer:
    """Ensure a normal map has unit-length vectors per pixel."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "normal_image": ("IMAGE",),
            },
            "optional": {
                "normalize_range": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "visualize"
    CATEGORY = "🧪AILab/🔆LBM-Pro"

    def visualize(
        self,
        normal_image: torch.Tensor,
        normalize_range: bool = True,
    ) -> tuple[torch.Tensor]:
        if normal_image.shape[-1] != 3:
            raise ValueError(f"Normal image must have 3 channels; got {normal_image.shape}")
        arr = normal_image.detach().cpu().numpy()
        if normalize_range:
            arr = arr * 2.0 - 1.0
        out = normalize_normal_map(arr.astype(np.float32))
        out = (out + 1.0) / 2.0
        return (torch.from_numpy(out),)


NODE_CLASS_MAPPINGS = {"LBM_Normal_Visualizer": LBM_Normal_Visualizer}
NODE_DISPLAY_NAME_MAPPINGS = {"LBM_Normal_Visualizer": "LBM Normal Visualizer"}
