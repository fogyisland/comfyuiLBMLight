"""Image-and-mask conditioning branch.

Combines one or more images and one or more masks into a single
``tile_stack`` tensor at the latent resolution, which downstream
UNet wrappers concatenate channel-wise with the denoiser input.

The branch is intentionally stateless apart from its config; the
codec is taken from the call signature so the branch does not own
a reference to the heavyweight VAE.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from .condition_aggregator import BaseCondition
from .latent_codec import LatentCodec


@dataclass(slots=True)
class ImageConcatConfig:
    image_keys: List[str] = field(default_factory=list)
    mask_keys: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.image_keys and not self.mask_keys:
            raise ValueError("ImageConcatConfig requires at least one image_key or mask_key")


# Maps the number of spatial dimensions of the produced tensor to the
# modality name under which the aggregator should stash it.
_NDIM_TO_MODALITY = {
    4: "tile_stack",
}


class ImageConcatCondition(BaseCondition):
    """Build a latent-resolution ``tile_stack`` conditioning tensor.

    Args:
        config:      identifies which batch entries to consume.
        input_key:   the batch key that triggers this branch (usually
                     the primary source-image key).
        codec:       optional default codec.  Required when the branch
                     runs inside a :class:`ConditionAggregator`, which
                     does not forward extras to its branches.  Direct
                     callers may instead pass ``codec=`` per call.
        ucg_rate:    per-instance unconditional-dropout rate.
    """

    input_key: str = "source_image"

    def __init__(
        self,
        config: ImageConcatConfig,
        input_key: str = "source_image",
        codec: Optional[LatentCodec] = None,
        ucg_rate: float = 0.0,
    ) -> None:
        super().__init__(ucg_rate=ucg_rate)
        self.config = config
        self.input_key = input_key
        # Bypass ``nn.Module.__setattr__`` so the (frozen, heavyweight)
        # VAE is not registered as a submodule — it must not appear in
        # this branch's ``state_dict`` nor be moved by ``.to()``.
        object.__setattr__(self, "_codec", codec)

    @property
    def codec(self) -> Optional[LatentCodec]:
        return self._codec

    def forward(
        self,
        batch: Dict[str, Any],
        codec: Optional[LatentCodec] = None,
        force_zero_embedding: bool = False,
    ) -> Dict[str, torch.Tensor]:
        codec = codec if codec is not None else self._codec
        if codec is None:
            raise ValueError(
                "ImageConcatCondition requires a codec: pass one to __init__ "
                "(when used inside a ConditionAggregator) or to forward()"
            )
        if force_zero_embedding:
            # Match the latent spatial shape by reading a sentinel key
            # so the caller still receives a properly-shaped tile_stack.
            any_key = self.config.image_keys[0] if self.config.image_keys else self.config.mask_keys[0]
            probe = batch[any_key]
            latent_h = probe.shape[-2] // codec.downsampling_factor
            latent_w = probe.shape[-1] // codec.downsampling_factor
            zero = torch.zeros(
                probe.shape[0],
                self._out_channels(),
                latent_h,
                latent_w,
                device=probe.device,
                dtype=probe.dtype,
            )
            return {self._modality(): zero}

        dims = [batch[k].shape[-2:] for k in self.config.image_keys + self.config.mask_keys]
        if not dims:
            raise ValueError("ImageConcatConfig received a batch with no usable keys")
        first = dims[0]
        if any(d != first for d in dims[1:]):
            raise ValueError(
                f"image/mask dims must agree; got {dims}"
            )
        latent_h = first[0] // codec.downsampling_factor
        latent_w = first[1] // codec.downsampling_factor

        pieces: List[torch.Tensor] = []
        for mask_key in self.config.mask_keys:
            resized = F.interpolate(
                batch[mask_key],
                size=(latent_h, latent_w),
                mode="bilinear",
                align_corners=False,
            )
            pieces.append(resized)
        for image_key in self.config.image_keys:
            pieces.append(codec.encode(batch[image_key]))

        stacked = torch.cat(pieces, dim=1)
        return {self._modality(): stacked}

    def _modality(self) -> str:
        return _NDIM_TO_MODALITY.get(4, "tile_stack")

    def _out_channels(self) -> int:
        return len(self.config.mask_keys) + len(self.config.image_keys) * 4
