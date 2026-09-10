"""LBM Batch Processor — apply the same relighting params to a batch of images."""
from __future__ import annotations

import torch

import comfy.model_management as mm

from lbm_core import LIGHT_PRESET_TYPE, LBM_MODEL_TYPE
from lbm_core.presets import PRESETS


def _apply_tint(image: torch.Tensor, preset: dict) -> torch.Tensor:
    tint = torch.tensor(preset["rgb_tint"], dtype=image.dtype, device=image.device)
    intensity = float(preset["intensity"])
    return (image * tint * intensity).clamp(0.0, 1.0)


class LBM_Batch_Processor:
    """Process a batch of images with identical parameters (consistency)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lbm_model": (LBM_MODEL_TYPE,),
                "images": ("IMAGE",),
                "steps": ("INT", {"default": 28, "min": 1, "max": 100}),
            },
            "optional": {
                "light_preset": (LIGHT_PRESET_TYPE,),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process_batch"
    CATEGORY = "🧪AILab/🔆LBM-Pro"

    def process_batch(
        self,
        lbm_model: dict,
        images: torch.Tensor,
        steps: int,
        light_preset: dict | None = None,
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

        x = images.clone().permute(0, 3, 1, 2).to(device, dtype) * 2 - 1
        batch = {solver.schedule.anchor_field: x}

        solver.codec.to(device)
        anchor_key = solver.schedule.anchor_field
        z = solver.codec.encode(batch[anchor_key])
        solver.codec.cpu()
        solver.to(device)

        prev_sigma = solver.schedule.noise_jitter
        solver.schedule.noise_jitter = float(light_preset.get("bridge_noise_sigma", 0.005))
        try:
            out = solver.decode_latents_to_pixels(
                z=z, num_steps=steps, conditioner_inputs=batch
            ).clamp(-1, 1)
        finally:
            solver.schedule.noise_jitter = prev_sigma

        out = out.permute(0, 2, 3, 1).cpu().float()
        out = (out + 1) / 2
        out = _apply_tint(out, light_preset)

        solver.cpu()
        mm.soft_empty_cache()
        return (out,)


NODE_CLASS_MAPPINGS = {"LBM_Batch_Processor": LBM_Batch_Processor}
NODE_DISPLAY_NAME_MAPPINGS = {"LBM_Batch_Processor": "LBM Batch Processor"}
