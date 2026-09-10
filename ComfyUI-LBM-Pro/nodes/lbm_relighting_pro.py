"""LBM Relighting Pro — enhanced relighting with light preset + tinting."""
from __future__ import annotations

import torch

import comfy.model_management as mm

from lbm_core import LIGHT_PRESET_TYPE, LBM_MODEL_TYPE
from lbm_core.presets import PRESETS


def _apply_tint(image: torch.Tensor, preset: dict) -> torch.Tensor:
    """Apply (rgb_tint × intensity) per-pixel to an image batch (B, H, W, C)."""
    tint = torch.tensor(preset["rgb_tint"], dtype=image.dtype, device=image.device)
    intensity = float(preset["intensity"])
    return (image * tint * intensity).clamp(0.0, 1.0)


class LBM_Relighting_Pro:
    """Run the cached LBM relighting model and apply a light preset tint."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lbm_model": (LBM_MODEL_TYPE,),
                "image": ("IMAGE",),
                "steps": (
                    "INT",
                    {"default": 28, "min": 1, "max": 100},
                ),
            },
            "optional": {
                "light_preset": (LIGHT_PRESET_TYPE,),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "relight"
    CATEGORY = "🧪AILab/🔆LBM-Pro"

    def relight(
        self,
        lbm_model: dict,
        image: torch.Tensor,
        steps: int,
        light_preset: dict | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        if light_preset is None:
            wp = PRESETS["warm_neutral"]
            light_preset = {
                "name": wp.name,
                "rgb_tint": wp.rgb_tint,
                "intensity": wp.intensity,
                "bridge_noise_sigma": wp.bridge_noise_sigma,
                "description": wp.description,
            }

        model = lbm_model["model"]
        dtype = lbm_model["dtype"]
        device = lbm_model["device"]

        x = image.clone().permute(0, 3, 1, 2).to(device, dtype) * 2 - 1
        batch = {"source_image": x}
        if mask is not None:
            m = mask
            if m.ndim == 2:
                m = m.unsqueeze(0).unsqueeze(0)
            elif m.ndim == 3:
                m = m.unsqueeze(0)
            batch["mask"] = m.to(device, dtype)

        model.vae.to(device)
        z = model.vae.encode(batch[model.source_key])
        model.vae.cpu()
        model.to(device)

        sigma = float(light_preset.get("bridge_noise_sigma", 0.005))
        prev_sigma = model.bridge_noise_sigma
        model.bridge_noise_sigma = sigma
        try:
            out = model.sample(z=z, num_steps=steps, conditioner_inputs=batch).clamp(-1, 1)
        finally:
            model.bridge_noise_sigma = prev_sigma

        out = out.permute(0, 2, 3, 1).cpu().float()
        out = (out + 1) / 2
        out = _apply_tint(out, light_preset)
        model.cpu()
        mm.soft_empty_cache()
        return (out,)


NODE_CLASS_MAPPINGS = {"LBM_Relighting_Pro": LBM_Relighting_Pro}
NODE_DISPLAY_NAME_MAPPINGS = {"LBM_Relighting_Pro": "LBM Relighting Pro"}
