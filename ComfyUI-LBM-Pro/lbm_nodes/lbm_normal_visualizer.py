"""LBM Normal Visualizer — validate and re-normalize a normal map."""
from __future__ import annotations

import numpy as np
import torch

from lbm_core.visualizers import normalize_normal_map


class LBM_Normal_Visualizer:
    """Ensure a normal map has unit-length vectors per pixel.

    The ``input_range`` widget replaces the older ``normalize_range``
    toggle; it explicitly accepts ``"auto"``, ``"[0,1]"`` (input is
    in ``[0, 1]``, mapped to ``[-1, 1]`` before normalisation), or
    ``"[-1,1]"`` (input is already in ``[-1, 1]`` and is used as-is).
    ``"auto"`` infers the range by sampling the input tensor: if any
    value is below ``-0.5``, the input is treated as already in
    ``[-1, 1]``.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "normal_image": ("IMAGE",),
                "input_range": (
                    ["auto", "[0,1]", "[-1,1]"],
                    {
                        "default": "auto",
                        "tooltip": "Range of the input tensor. 'auto' "
                                   "infers from data; '[0,1]' is mapped to "
                                    "'[-1,1]' before normalising; '[-1,1]' "
                                    "is used as-is.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "visualize"
    CATEGORY = "🧪BMLab/🔆LBM-Pro"

    @staticmethod
    def _resolve_range(input_range: str, sample: torch.Tensor) -> bool:
        """Return True if the input should be mapped from ``[0, 1]`` to ``[-1, 1]``."""
        if input_range == "[0,1]":
            return True
        if input_range == "[-1,1]":
            return False
        # auto — if any sample value is below -0.5, assume already
        # centred; otherwise treat as [0, 1] and remap.
        try:
            finite_min = float(sample.min().item())
        except Exception:
            finite_min = 0.0
        return finite_min >= -0.5

    def visualize(
        self,
        normal_image: torch.Tensor,
        input_range: str = "auto",
    ) -> tuple[torch.Tensor]:
        if normal_image.shape[-1] != 3:
            raise ValueError(f"Normal image must have 3 channels; got {normal_image.shape}")
        arr = normal_image.detach().cpu().numpy()
        if self._resolve_range(input_range, normal_image):
            arr = arr * 2.0 - 1.0
        out = normalize_normal_map(arr.astype(np.float32))
        out = (out + 1.0) / 2.0
        # N15: always return fp32 so downstream nodes don't have to
        # cast a different dtype depending on the input range.
        return (torch.from_numpy(out).float(),)


NODE_CLASS_MAPPINGS = {"LBM_Normal_Visualizer": LBM_Normal_Visualizer}
NODE_DISPLAY_NAME_MAPPINGS = {"LBM_Normal_Visualizer": "LBM Normal Visualizer"}
