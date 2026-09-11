"""UNet wrappers adapted to the guide-dict calling convention.

Each wrapper accepts a ``guide`` argument structured as::

    {
        "guide_pack": {
            "class_vec": Optional[Tensor[B, ...]],
            "attn_ctx":   Optional[Tensor[B, T, D]],
            "tile_stack": Optional[Tensor[B, extra_channels, H, W]],
        }
    }

When ``tile_stack`` is supplied it is concatenated channel-wise with
the denoiser input before the first conv — this is what enables
latent-bridge conditioning without cross-attention.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
from diffusers.models import UNet2DConditionModel, UNet2DModel

from .inference_core import InferenceCore


class _BaseDiffusersUNet(InferenceCore):
    """Common scaffolding for both wrapper flavours.

    Centralises the guide-dict parsing and the optional
    ``tile_stack`` concatenation so the two subclasses can stay
    focused on their diffusers-specific call signatures.
    """

    def _parse_guide(self, guide):
        """Return ``(class_vec, attn_ctx, tile_stack)`` or three ``None``."""
        if guide is None:
            return None, None, None
        pack = guide.get("guide_pack") if isinstance(guide, dict) else None
        if not isinstance(pack, dict):
            return None, None, None
        return (
            pack.get("class_vec"),
            pack.get("attn_ctx"),
            pack.get("tile_stack"),
        )

    def _maybe_stack_tiles(self, sample, tile_stack):
        if tile_stack is None:
            return sample
        if tile_stack.shape[0] != sample.shape[0]:
            raise ValueError(
                "tile_stack batch dim must equal sample batch dim "
                f"(got {tile_stack.shape[0]} vs {sample.shape[0]})"
            )
        return torch.cat([sample, tile_stack.to(sample.dtype)], dim=1)

    def hard_freeze(self) -> None:
        """Disable gradients on every parameter and switch to eval mode.

        Iterates ``self.parameters()`` explicitly so the behaviour is
        independent of the MRO chain — the diffusers parent classes
        do not override ``hard_freeze`` but relying on
        ``super().hard_freeze()`` would silently break if any future
        mix-in added its own freeze hook.
        """
        self.eval()
        for param in self.parameters():
            param.requires_grad_(False)


class PlainUNet2D(_BaseDiffusersUNet, UNet2DModel):
    """Wraps ``diffusers.UNet2DModel`` so it speaks the guide convention.

    The original UNet2D does not consume cross-attention context —
    only an optional class embedding.  We still expose the full guide
    interface so a caller can swap wrappers without re-plumbing
    graph nodes.
    """

    def __init__(self, *args, **kwargs) -> None:
        UNet2DModel.__init__(self, *args, **kwargs)
        InferenceCore.__init__(self)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        guide=None,
        *args,
        **kwargs,
    ):
        class_vec, _attn, tile_stack = self._parse_guide(guide)
        merged = self._maybe_stack_tiles(sample, tile_stack)
        out = UNet2DModel.forward(
            self,
            merged,
            timestep,
            class_labels=class_vec,
        )
        return out.sample


class CondUNet2D(_BaseDiffusersUNet, UNet2DConditionModel):
    """Wraps ``diffusers.UNet2DConditionModel`` with guide-dict plumbing.

    Supports ``tile_stack`` (channel concat) and ``attn_ctx`` (cross-
    attention context) but ignores ``class_vec`` because the SD
    UNet does not consume it.  The class is laid out so a checkpoint
    trained with the legacy ``conditioning`` dict still loads.
    """

    def __init__(self, *args, **kwargs) -> None:
        UNet2DConditionModel.__init__(self, *args, **kwargs)
        InferenceCore.__init__(self)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        guide=None,
        ip_adapter_embeds: Optional[Sequence[torch.Tensor]] = None,
        down_residuals=None,
        mid_residual=None,
        intra_residuals=None,
        *args,
        **kwargs,
    ):
        _cls, attn_ctx, tile_stack = self._parse_guide(guide)
        merged = self._maybe_stack_tiles(sample, tile_stack)

        added_cond_kwargs = None
        if ip_adapter_embeds is not None:
            added_cond_kwargs = {
                "image_embeds": [t.unsqueeze(1) for t in ip_adapter_embeds]
            }

        intra_clone = None
        if intra_residuals is not None:
            intra_clone = [t.clone() for t in intra_residuals]

        out = UNet2DConditionModel.forward(
            self,
            sample=merged,
            timestep=timestep,
            encoder_hidden_states=attn_ctx,
            class_labels=None,
            added_cond_kwargs=added_cond_kwargs,
            down_block_additional_residuals=down_residuals,
            mid_block_additional_residual=mid_residual,
            down_intrablock_additional_residuals=intra_clone,
        )
        return out.sample
