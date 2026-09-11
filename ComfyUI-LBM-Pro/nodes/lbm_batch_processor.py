"""LBM Batch Processor — apply the same relighting params to a batch of images."""
from __future__ import annotations

import torch

import comfy.model_management as mm
from comfy.utils import ProgressBar

from lbm_core import LIGHT_PRESET_TYPE, LBM_MODEL_TYPE, apply_tint
from lbm_core.presets import PRESETS
from nodes.lbm_model_loader import resolve_lbm_device


def _normalize_mask(mask: torch.Tensor) -> torch.Tensor:
    """Coerce a ComfyUI ``MASK`` tensor to ``(B, 1, H, W)`` float32.

    Accepts 2-D ``(H, W)`` (one frame), 3-D ``(B, H, W)`` (multi-frame
    mask), or 4-D ``(B, 1, H, W)`` (already correctly shaped).  The
    output is left on CPU and in fp32 so the caller can move it to the
    target device with the right dtype (N11 — keep the mask in fp32).
    """
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    return mask.to(torch.float32)


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
                "mask": ("MASK",),
                "max_batch": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 64,
                        "tooltip": "Maximum frames per solver invocation. "
                                   "Batches larger than this are split into "
                                   "chunks and processed sequentially.",
                    },
                ),
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
        mask: torch.Tensor | None = None,
        max_batch: int = 4,
    ) -> tuple[torch.Tensor]:
        # N12: empty batch is a hard error — there is no reasonable
        # default to fall back to and silently returning an empty
        # tensor would surprise downstream nodes.
        if images.shape[0] == 0:
            raise ValueError("Empty batch (B == 0); need at least one image")

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
        device_capture = resolve_lbm_device(lbm_model)

        # N6: chunk the batch so we never feed more than ``max_batch``
        # frames at once to the solver.  This caps VRAM for large
        # batches and keeps the ProgressBar honest.
        bsz = images.shape[0]
        max_batch = max(1, int(max_batch))
        chunks: list[torch.Tensor] = []
        pbar = ProgressBar(steps * bsz)
        for start in range(0, bsz, max_batch):
            end = min(start + max_batch, bsz)
            chunk_imgs = images[start:end]
            chunk_mask = mask[start:end] if mask is not None else None

            out = self._run_chunk(
                solver=solver,
                dtype=dtype,
                device=device_capture,
                images=chunk_imgs,
                mask=chunk_mask,
                steps=steps,
                light_preset=light_preset,
                pbar=pbar,
                step_offset=start * steps,
            )
            chunks.append(out)

        out = torch.cat(chunks, dim=0)
        solver.cpu()
        mm.soft_empty_cache()
        return (out,)

    def _run_chunk(
        self,
        solver,
        dtype: torch.dtype,
        device: torch.device,
        images: torch.Tensor,
        mask: torch.Tensor | None,
        steps: int,
        light_preset: dict,
        pbar: ProgressBar,
        step_offset: int = 0,
    ) -> torch.Tensor:
        """Run one chunk through the solver and return the (B, H, W, C) output."""
        x = images.clone().permute(0, 3, 1, 2).to(device, dtype) * 2 - 1
        batch = {solver.schedule.anchor_field: x}
        if mask is not None:
            m = _normalize_mask(mask).to(device)
            batch[solver.schedule.mask_field or "mask"] = m

        solver.codec.to(device)
        anchor_key = solver.schedule.anchor_field
        z = solver.codec.encode(batch[anchor_key])
        solver.codec.cpu()
        solver.to(device)

        def _cb(completed: int, _total: int) -> None:
            # Translate chunk-local progress into global progress
            # across all chunks in the batch.
            pbar.update_absolute(step_offset + completed, step_offset + steps)

        out = solver.decode_latents_to_pixels(
            z=z,
            num_steps=steps,
            conditioner_inputs=batch,
            progress_cb=_cb,
        ).clamp(-1, 1)

        out = out.permute(0, 2, 3, 1).cpu().float()
        out = (out + 1) / 2
        out = apply_tint(out, light_preset)
        return out


NODE_CLASS_MAPPINGS = {"LBM_Batch_Processor": LBM_Batch_Processor}
NODE_DISPLAY_NAME_MAPPINGS = {"LBM_Batch_Processor": "LBM Batch Processor"}
