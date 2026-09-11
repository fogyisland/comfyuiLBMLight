"""Latent codec wrapping the diffusers SD1 autoencoder.

The codec owns:
  * the underlying VAE (frozen),
  * latent scaling and optional shift factor,
  * tile-by-tile decoding so large latents do not blow the VRAM.

The implementation deliberately does NOT inherit from any shared
base class — ``InferenceCore`` is used only as a reference for the
``device``/``dtype`` pattern.  Inheritance would force a coupling
that is undesirable for a frozen pretrained component.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
from diffusers.models import AutoencoderKL


@dataclass(slots=True)
class TilePlan:
    """Static description of how a single 2-D feature map is chopped."""

    tile_h: int = 128
    tile_w: int = 128
    overlap_h: int = 16
    overlap_w: int = 16
    downscale: int = 8


def _slice_grid(total: int, tile: int, overlap: int) -> list[int]:
    """Return start offsets covering ``[0, total)`` with overlaps.

    The last tile always reaches ``total``; intermediate tiles are
    spaced ``tile - overlap`` apart.
    """
    if total <= tile:
        return [0]
    stride = max(1, tile - overlap)
    starts = list(range(0, total - tile + 1, stride))
    if starts[-1] + tile < total:
        starts.append(total - tile)
    return starts


def _linear_blend(a: torch.Tensor, b: torch.Tensor, weight: float) -> torch.Tensor:
    """Return ``a * (1 - w) + b * w`` broadcast on the last dims."""
    return a.mul(1.0 - weight).add(b, alpha=weight)


class LatentCodec(nn.Module):
    """Thin wrapper around a frozen ``AutoencoderKL``.

    Public surface:
      * ``encode(x)``        — pixels → latents (B, C, H/8, W/8)
      * ``decode(z)``        — latents → pixels (B, 3, H, W)
      * ``downsampling_factor`` / ``latent_channels`` — introspected
    """

    def __init__(
        self,
        vae_model: AutoencoderKL,
        tiling: Optional[TilePlan] = None,
        normalize: Literal["sd1", "sdxl"] = "sd1",
    ) -> None:
        super().__init__()
        self.vae_model = vae_model
        self.tile_plan = tiling or TilePlan()
        config = vae_model.config
        self.latent_channels = int(getattr(config, "latent_channels", 4))
        self.scaling_factor = float(getattr(config, "scaling_factor", 0.13025))
        self.shift_factor = float(getattr(config, "shift_factor", 0.0) or 0.0)
        self._has_latents_mean = getattr(config, "latents_mean", None) is not None
        self._has_latents_std = getattr(config, "latents_std", None) is not None
        # C3: gate the SDXL ``latents_mean``/``latents_std`` branch on an
        # explicit caller-supplied mode so an SD1 VAE config that happens
        # to carry these fields (e.g. malformed checkpoint) cannot be
        # silently mis-normalised.
        self.normalize_mode: Literal["sd1", "sdxl"] = normalize
        if normalize == "sdxl":
            if not (self._has_latents_mean and self._has_latents_std):
                raise ValueError(
                    "LatentCodec(normalize='sdxl') requires the VAE config "
                    "to carry both latents_mean and latents_std."
                )
        else:  # sd1
            if (
                self._has_latents_mean
                and self.shift_factor != 0.0
            ):
                raise ValueError(
                    "LatentCodec(normalize='sd1') cannot apply shift_factor "
                    f"({self.shift_factor}) together with config.latents_mean; "
                    "the VAE config carries conflicting normalisation fields."
                )
        self.downsampling_factor = 8
        for piece in self.vae_model.parameters():
            piece.requires_grad_(False)
        self.vae_model.eval()

    @property
    def device(self) -> torch.device:
        return next(self.vae_model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.vae_model.parameters()).dtype

    def encode(self, x: torch.Tensor, chunk: int = 8) -> torch.Tensor:
        """Encode ``x`` in chunks and rescale with the codec's scaling.

        The rescale formula assumes the diffusers default
        ``scaling_factor`` and an optional ``shift_factor`` — exactly
        what SD1 VAE exposes.
        """
        outs: list[torch.Tensor] = []
        for i in range(0, x.shape[0], chunk):
            tile = x[i : i + chunk]
            dist = self.vae_model.encode(tile).latent_dist
            sample = dist.sample()
            outs.append(sample)
        stacked = torch.cat(outs, dim=0)
        return (stacked - self.shift_factor) * self.scaling_factor

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`encode`; tiles when latent is large.

        C4: the output dtype matches ``z.dtype`` so callers can rely on
        encode/decode symmetry regardless of the underlying VAE weights'
        dtype.
        """
        if self.normalize_mode == "sdxl":
            mean = self.vae_model.config.latents_mean
            std = self.vae_model.config.latents_std
            mean_t = torch.tensor(mean, device=z.device, dtype=z.dtype).view(
                1, self.latent_channels, 1, 1
            )
            std_t = torch.tensor(std, device=z.device, dtype=z.dtype).view(
                1, self.latent_channels, 1, 1
            )
            rescaled = z * std_t / self.scaling_factor + mean_t
        else:
            rescaled = z / self.scaling_factor + self.shift_factor

        plan = self.tile_plan
        too_big = (
            rescaled.shape[2] > plan.tile_h or rescaled.shape[3] > plan.tile_w
        )
        if not too_big:
            pixels = self.vae_model.decode(rescaled).sample
            return pixels.to(z.dtype)

        pixels = self._decode_tiled(rescaled)
        return pixels.to(z.dtype)

    def _decode_tiled(self, z: torch.Tensor) -> torch.Tensor:
        """Decode each spatial tile and blend overlapping regions."""
        plan = self.tile_plan
        B = z.shape[0]
        ds = plan.downscale
        out_pixels: list[torch.Tensor] = []
        for b in range(B):
            sample = self._decode_single_image(z[b])
            out_pixels.append(sample.unsqueeze(0))
        return torch.cat(out_pixels, dim=0)

    def _decode_single_image(self, z_b: torch.Tensor) -> torch.Tensor:
        """Decode a single (C, H, W) latent with overlapping tiles."""
        plan = self.tile_plan
        ds = plan.downscale
        H_lat = z_b.shape[1]
        W_lat = z_b.shape[2]
        # output buffer in pixel space
        H_pix = H_lat * ds
        W_pix = W_lat * ds
        tile_h_pix = plan.tile_h * ds
        tile_w_pix = plan.tile_w * ds
        overlap_h_pix = plan.overlap_h * ds
        overlap_w_pix = plan.overlap_w * ds

        accum = torch.zeros(3, H_pix, W_pix, device=z_b.device, dtype=z_b.dtype)
        weight = torch.zeros(1, H_pix, W_pix, device=z_b.device, dtype=z_b.dtype)

        starts_h = _slice_grid(H_pix, tile_h_pix, overlap_h_pix)
        starts_w = _slice_grid(W_pix, tile_w_pix, overlap_w_pix)

        for y0 in starts_h:
            for x0 in starts_w:
                # slice latent region, pad to full tile size if needed
                ly0 = y0 // ds
                ly1 = min(H_lat, (y0 + tile_h_pix) // ds)
                lx0 = x0 // ds
                lx1 = min(W_lat, (x0 + tile_w_pix) // ds)
                tile = z_b[:, ly0:ly1, lx0:lx1].unsqueeze(0)
                tile_padded = self._pad_latent(tile, plan.tile_h, plan.tile_w)
                decoded = self.vae_model.decode(tile_padded).sample
                # crop back to actual region
                actual_h = (ly1 - ly0) * ds
                actual_w = (lx1 - lx0) * ds
                decoded = decoded[0, :, :actual_h, :actual_w]
                # Hann-shaped linear window to soften the seam
                win = self._make_window(actual_h, actual_w, overlap_h_pix, overlap_w_pix)
                windowed = decoded * win
                accum[:, y0 : y0 + actual_h, x0 : x0 + actual_w] += windowed
                weight[:, y0 : y0 + actual_h, x0 : x0 + actual_w] += win

        return accum / weight.clamp(min=1e-6)

    @staticmethod
    def _pad_latent(tile: torch.Tensor, tile_h: int, tile_w: int) -> torch.Tensor:
        """Pad latent to ``(tile_h, tile_w)`` if short.

        ``reflect`` padding produces a smoother extension than
        ``replicate`` at edges that contain high-frequency content
        (text, hair), which reduces the visible colour band in the
        VAE's first decoder layer.

        C10: ``reflect`` requires the padding to be strictly smaller
        than the input dim.  When the input is so small that
        ``pad_h >= H`` (or ``pad_w >= W``) we fall back to
        ``replicate`` instead of crashing.
        """
        _, _, H, W = tile.shape
        if H >= tile_h and W >= tile_w:
            return tile
        pad_h = tile_h - H
        pad_w = tile_w - W
        # reflect requires pad < input dim on the corresponding axis;
        # otherwise the input alone can't supply a neighbour to mirror.
        if pad_h >= H or pad_w >= W:
            mode = "replicate"
        else:
            mode = "reflect"
        return torch.nn.functional.pad(tile, (0, pad_w, 0, pad_h), mode=mode)

    @staticmethod
    def _make_window(H: int, W: int, ovh: int, oVw: int) -> torch.Tensor:
        """A 2-D linear ramp that ramps up from edges and plateaus in the middle.

        C11: when a tile is shorter than ``2 * overlap`` on either axis,
        build a full ramp that covers the whole dimension rather than
        returning ones — otherwise the last (rightmost / bottom) tile
        leaves a visible seam where it meets the penultimate tile.
        """
        # 1-D ramps on Y.  ``H`` is always >= ds >= 1 in this codec, so
        # the degenerate ``H <= 0`` case is unreachable — only the two
        # real shapes (full ramp vs. ramp-with-plateau) remain.
        if H <= 2 * ovh:
            ramp_y = torch.linspace(0.0, 1.0, H)
        else:
            ramp_y = torch.ones(H)
            ramp_y[:ovh] = torch.linspace(0.0, 1.0, ovh)
            ramp_y[-ovh:] = torch.linspace(1.0, 0.0, ovh)
        # 1-D ramps on X — same two-way split as Y.
        if W <= 2 * oVw:
            ramp_x = torch.linspace(0.0, 1.0, W)
        else:
            ramp_x = torch.ones(W)
            ramp_x[:oVw] = torch.linspace(0.0, 1.0, oVw)
            ramp_x[-oVw:] = torch.linspace(1.0, 0.0, oVw)
        return (ramp_y[:, None] * ramp_x[None, :]).unsqueeze(0)
