"""LBM Depth Visualizer — colorize a depth map using viridis/inferno/turbo/gray."""
from __future__ import annotations

import numpy as np
import torch

from lbm_core import COLORMAPS as _COLORMAPS
from lbm_core.visualizers import depth_to_colormap


class LBM_Depth_Visualizer:
    """Apply a colormap to a single-channel depth image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_image": ("IMAGE",),
                "colormap": (_COLORMAPS, {"default": "turbo"}),
            },
            "optional": {
                "invert": ("BOOLEAN", {"default": False}),
                "auto_normalize": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "visualize"
    CATEGORY = "🧪AILab/🔆LBM-Pro"

    def visualize(
        self,
        depth_image: torch.Tensor,
        colormap: str,
        invert: bool = False,
        auto_normalize: bool = True,
    ) -> tuple[torch.Tensor]:
        if depth_image.ndim != 4:
            raise ValueError(f"Expected (B, H, W, C); got {depth_image.shape}")
        c = depth_image.shape[-1]
        if c not in (1, 3):
            raise ValueError(
                f"Depth image must have 1 or 3 channels; got {c} "
                f"(shape {tuple(depth_image.shape)})"
            )
        batch = depth_image.detach().cpu()
        if c == 3:
            # N18: when the input carries 3 channels (a colourised
            # depth map or RGB triplets of a depth), use channel 0
            # rather than averaging across RGB — averaging a colourised
            # map destroys the colour information and produces muddy
            # output.  Callers wanting the RGB-channel average should
            # pre-process via a separate node.
            batch = batch[..., 0]
        else:
            batch = batch.squeeze(-1)
        out_frames: list[np.ndarray] = []
        for i in range(batch.shape[0]):
            depth_np = batch[i].numpy()
            rgb = depth_to_colormap(depth_np, colormap, invert=invert, normalize=auto_normalize)
            out_frames.append(rgb)
        arr = np.stack(out_frames, axis=0)
        return (torch.from_numpy(arr).float() / 255.0,)


NODE_CLASS_MAPPINGS = {"LBM_Depth_Visualizer": LBM_Depth_Visualizer}
NODE_DISPLAY_NAME_MAPPINGS = {"LBM_Depth_Visualizer": "LBM Depth Visualizer"}
