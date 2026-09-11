"""LBM Batch Processor — apply the same relighting params to a batch of images."""
from __future__ import annotations

import torch

import comfy.model_management as mm
from comfy.utils import ProgressBar

from lbm_core import LIGHT_PRESET_TYPE, LBM_MODEL_TYPE, apply_tint
from lbm_core.presets import PRESETS


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

        sigma = float(light_preset.get("bridge_noise_sigma", 0.005))
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
        out = apply_tint(out, light_preset)

        solver.cpu()
        mm.soft_empty_cache()
        return (out,)


NODE_CLASS_MAPPINGS = {"LBM_Batch_Processor": LBM_Batch_Processor}
NODE_DISPLAY_NAME_MAPPINGS = {"LBM_Batch_Processor": "LBM Batch Processor"}
