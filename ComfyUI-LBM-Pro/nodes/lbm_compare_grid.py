"""LBM Compare Grid — stitch 2–9 images into a comparison grid."""
from __future__ import annotations

import math

import torch


_LAYOUTS = ["auto", "horizontal", "vertical", "grid_2x2", "grid_3x3"]


def _resolve_grid(n: int, layout: str) -> tuple[int, int]:
    if layout == "horizontal":
        return 1, n
    if layout == "vertical":
        return n, 1
    if layout == "grid_2x2":
        if n > 4:
            raise ValueError("grid_2x2 supports up to 4 images")
        return 2, 2
    if layout == "grid_3x3":
        if n > 9:
            raise ValueError("grid_3x3 supports up to 9 images")
        return 3, 3
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    return rows, cols


class LBM_Compare_Grid:
    """Compose multiple images into a single comparison grid."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "layout": (_LAYOUTS, {"default": "auto"}),
            },
            "optional": {
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
                "image_9": ("IMAGE",),
                "padding": (
                    "INT",
                    {"default": 8, "min": 0, "max": 64},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("grid",)
    FUNCTION = "compose"
    CATEGORY = "🧪AILab/🔆LBM-Pro"

    def compose(
        self,
        image_1: torch.Tensor,
        image_2: torch.Tensor,
        layout: str = "auto",
        image_3: torch.Tensor | None = None,
        image_4: torch.Tensor | None = None,
        image_5: torch.Tensor | None = None,
        image_6: torch.Tensor | None = None,
        image_7: torch.Tensor | None = None,
        image_8: torch.Tensor | None = None,
        image_9: torch.Tensor | None = None,
        padding: int = 8,
    ) -> tuple[torch.Tensor]:
        imgs = [image_1, image_2, image_3, image_4, image_5,
                image_6, image_7, image_8, image_9]
        imgs = [im for im in imgs if im is not None]
        n = len(imgs)
        if n < 2:
            raise ValueError("Compare Grid requires at least 2 images")

        firsts = [im[0] for im in imgs]
        H = max(f.shape[0] for f in firsts)
        W = max(f.shape[1] for f in firsts)
        firsts = [
            torch.nn.functional.interpolate(
                im.permute(2, 0, 1).unsqueeze(0), size=(H, W),
                mode="bilinear", align_corners=False,
            ).squeeze(0).permute(1, 2, 0)
            for im in firsts
        ]

        rows, cols = _resolve_grid(n, layout)
        canvas = torch.zeros(
            (rows * H + (rows + 1) * padding,
             cols * W + (cols + 1) * padding, 3),
            dtype=firsts[0].dtype,
        )
        for i, im in enumerate(firsts):
            r = i // cols
            c = i % cols
            y0 = padding + r * (H + padding)
            x0 = padding + c * (W + padding)
            canvas[y0:y0 + H, x0:x0 + W] = im
        return (canvas.unsqueeze(0),)


NODE_CLASS_MAPPINGS = {"LBM_Compare_Grid": LBM_Compare_Grid}
NODE_DISPLAY_NAME_MAPPINGS = {"LBM_Compare_Grid": "LBM Compare Grid"}
