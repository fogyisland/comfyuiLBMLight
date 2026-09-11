"""LBM Depth/Normal Pro — emit raw + post-processed depth/normal maps."""
from __future__ import annotations

import torch

import comfy.model_management as mm
from comfy.utils import ProgressBar

from lbm_core import LBM_MODEL_TYPE
from nodes.lbm_model_loader import resolve_lbm_device


class LBM_DepthNormal_Pro:
    """Run the LBM depth/normal model and emit both raw and post-processed images."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lbm_model": (LBM_MODEL_TYPE,),
                "image": ("IMAGE",),
                "task": (["depth", "normal"], {"default": "depth"}),
                "steps": (
                    "INT",
                    {"default": 28, "min": 1, "max": 100},
                ),
            },
            "optional": {
                "bridge_noise_sigma": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 0.1, "step": 0.001},
                ),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("raw", "post_processed")
    FUNCTION = "process"
    CATEGORY = "🧪AILab/🔆LBM-Pro"

    def process(
        self,
        lbm_model: dict,
        image: torch.Tensor,
        task: str,
        steps: int,
        bridge_noise_sigma: float = 0.1,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # N12: empty batch is a hard error.
        if image.shape[0] == 0:
            raise ValueError("Empty batch (B == 0); need at least one image")

        # N2: refuse to run a model cached for a different task.
        # A relighting model loaded into this node would produce
        # plausible-looking but wrong outputs.
        if lbm_model.get("task") != task:
            raise ValueError(
                f"LBM_DepthNormal_Pro was given a model cached for "
                f"task={lbm_model.get('task')!r} but the node is "
                f"configured for task={task!r}. Load the correct model "
                "or change the task parameter to match."
            )
        solver = lbm_model["model"]
        dtype = lbm_model["dtype"]
        device = resolve_lbm_device(lbm_model)

        x = image.clone().permute(0, 3, 1, 2).to(device, dtype) * 2 - 1
        batch = {solver.schedule.anchor_field: x}
        if mask is not None:
            m = mask
            if m.ndim == 2:
                m = m.unsqueeze(0).unsqueeze(0)
            elif m.ndim == 3:
                m = m.unsqueeze(0)
            # N11: keep the mask in fp32 — the bridge solver expects a
            # fp32 mask channel; casting to the model dtype would lose
            # precision in the masked regions.
            batch[solver.schedule.mask_field or "mask"] = m.to(device, dtype=torch.float32)

        solver.codec.to(device)
        anchor_key = solver.schedule.anchor_field
        z = solver.codec.encode(batch[anchor_key])
        solver.codec.cpu()
        solver.to(device)

        pbar = ProgressBar(steps)
        out = solver.decode_latents_to_pixels(
            z=z,
            num_steps=steps,
            conditioner_inputs=batch,
            noise_jitter=float(bridge_noise_sigma),
            progress_cb=lambda completed, _total: pbar.update_absolute(completed, steps),
        ).clamp(-1, 1)

        out = out.permute(0, 2, 3, 1).cpu().float()
        out = (out + 1) / 2

        if task == "depth":
            post = 1 - out
        else:
            post = out

        solver.cpu()
        mm.soft_empty_cache()
        return (out, post)


NODE_CLASS_MAPPINGS = {"LBM_DepthNormal_Pro": LBM_DepthNormal_Pro}
NODE_DISPLAY_NAME_MAPPINGS = {"LBM_DepthNormal_Pro": "LBM Depth/Normal Pro"}
