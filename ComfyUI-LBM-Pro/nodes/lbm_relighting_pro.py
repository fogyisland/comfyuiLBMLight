"""LBM Relighting Pro — enhanced relighting with light preset + tinting.

This node uses the rewritten LBM runtime (``lbm_native``) under the
hood.  Public UI strings and node wiring are unchanged.
"""
from __future__ import annotations

import torch

import comfy.model_management as mm
from comfy.utils import ProgressBar

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

        solver = lbm_model["model"]
        dtype = lbm_model["dtype"]
        device = lbm_model["device"]

        x = image.clone().permute(0, 3, 1, 2).to(device, dtype) * 2 - 1
        batch = {solver.schedule.anchor_field: x}
        if mask is not None:
            m = mask
            if m.ndim == 2:
                m = m.unsqueeze(0).unsqueeze(0)
            elif m.ndim == 3:
                m = m.unsqueeze(0)
            batch[solver.schedule.mask_field or "mask"] = m.to(device, dtype)

        solver.codec.to(device)
        anchor_key = solver.schedule.anchor_field
        z = solver.codec.encode(batch[anchor_key])
        solver.codec.cpu()
        solver.to(device)

        sigma = float(light_preset.get("bridge_noise_sigma", 0.005))
        # Pass the override as a per-call argument instead of mutating
        # solver.schedule.noise_jitter.  Mutating the shared schedule
        # is racy when two nodes reuse the same model cache entry.
        pbar = ProgressBar(steps)
        out = solver.decode_latents_to_pixels(
            z=z,
            num_steps=steps,
            conditioner_inputs=batch,
            noise_jitter=sigma,
            progress_cb=lambda completed, _total: pbar.update_absolute(completed, steps),
        ).clamp(-1, 1)

        out = out.permute(0, 2, 3, 1).cpu().float()
        out = (out + 1) / 2
        out = _apply_tint(out, light_preset)
        solver.cpu()
        mm.soft_empty_cache()
        return (out,)


NODE_CLASS_MAPPINGS = {"LBM_Relighting_Pro": LBM_Relighting_Pro}
NODE_DISPLAY_NAME_MAPPINGS = {"LBM_Relighting_Pro": "LBM Relighting Pro"}
