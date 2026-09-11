"""Tests for the hardened LatentCodec (C3 + C4 + C10 + C11).

The tests target individual branches on a randomly-initialised
``AutoencoderKL`` — no checkpoint download is required.  Each test
isolates one of the four bugs from the task brief:

  * C3 — explicit ``normalize`` mode rejects conflicting config
    fields (``shift_factor`` + ``latents_mean``).
  * C4 — decode's dtype matches the encode input's dtype.
  * C10 — ``_pad_latent`` survives when the input is smaller than the
    pad width on either axis.
  * C11 — ``_make_window`` returns strictly-positive values on the
    short edge of a partial tile (no seam at the right/bottom border).
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _tiny_vae_config(**overrides):
    """Return a minimal ``AutoencoderKL`` config dict."""
    cfg = {
        "_class_name": "AutoencoderKL",
        "act_fn": "silu",
        "block_out_channels": [32, 64],  # tiny — only for shape tests
        "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D"],
        "force_upcast": False,
        "in_channels": 3,
        "latent_channels": 4,
        "layers_per_block": 1,
        "norm_num_groups": 32,
        "out_channels": 3,
        "sample_size": 128,
        "scaling_factor": 0.13025,
        "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D"],
    }
    cfg.update(overrides)
    return cfg


def _make_codec(**overrides):
    """Build a ``LatentCodec`` with the (random) weights in fp32."""
    from diffusers.models import AutoencoderKL

    from lbm_native.latent_codec import LatentCodec

    cfg = _tiny_vae_config(**overrides)
    vae = AutoencoderKL.from_config(cfg)
    return LatentCodec(vae)


# ---------------------------------------------------------------------------
# C3 — conflicting ``normalize`` config must raise
# ---------------------------------------------------------------------------
def test_codec_rejects_conflicting_normalize():
    """An SD1 config carrying ``latents_mean`` (an SDXL field) must be
    refused even when ``shift_factor != 0`` — the two normalisation
    regimes are mutually exclusive."""
    from diffusers.models import AutoencoderKL

    from lbm_native.latent_codec import LatentCodec

    cfg = _tiny_vae_config(
        shift_factor=0.5,
        latents_mean=[0.0] * 4,
        latents_std=[1.0] * 4,
    )
    vae = AutoencoderKL.from_config(cfg)
    with pytest.raises((ValueError, RuntimeError)):
        LatentCodec(vae, normalize="sd1")


def test_codec_rejects_sdxl_without_latents_std():
    """Symmetric direction of C3: ``normalize='sdxl'`` requires BOTH
    ``latents_mean`` and ``latents_std`` to be present in the config.
    A config that ships with only ``latents_mean`` (some SDXL VAEs do
    this when ``std`` is left to the runtime) must be refused at
    construction time."""
    from diffusers.models import AutoencoderKL

    from lbm_native.latent_codec import LatentCodec

    cfg = _tiny_vae_config(
        latents_mean=[0.0] * 4,
        # latents_std intentionally omitted
    )
    vae = AutoencoderKL.from_config(cfg)
    with pytest.raises((ValueError, RuntimeError)):
        LatentCodec(vae, normalize="sdxl")


# ---------------------------------------------------------------------------
# C4 — decode returns the encoder input's dtype
# ---------------------------------------------------------------------------
def test_decode_returns_input_dtype():
    """``decode`` must return the same dtype as its input ``z``, not
    the dtype of the underlying VAE weights.  We stub the VAE's
    decode call to return a fixed-dtype tensor so the bug surfaces
    even when the numeric result would otherwise match the codec's
    own dtype."""
    from unittest.mock import patch

    import torch

    from lbm_native.latent_codec import LatentCodec

    codec = _make_codec()  # fp32 weights

    class _Stub:
        sample = torch.zeros(1, 3, 32, 32, dtype=torch.float32)

    # The stub pretends to be a ``diffusers.DecoderOutput``; the
    # codec's decode does ``.sample`` on the result.
    z_fp16 = torch.randn(1, 4, 4, 4, dtype=torch.float16)
    with patch.object(codec.vae_model, "decode", return_value=_Stub()):
        pixels = codec.decode(z_fp16)
    assert pixels.dtype == torch.float16


# ---------------------------------------------------------------------------
# C10 — small inputs must not crash _pad_latent
# ---------------------------------------------------------------------------
def test_pad_latent_handles_small_input():
    """Pad a 30×30 latent to 128×128 — the tile is far smaller than the
    pad width, which trips ``reflect`` mode in PyTorch."""
    import torch

    from lbm_native.latent_codec import LatentCodec

    tile = torch.zeros(1, 4, 30, 30)
    out = LatentCodec._pad_latent(tile, tile_h=128, tile_w=128)
    assert out.shape == (1, 4, 128, 128)


# ---------------------------------------------------------------------------
# C11 — partial tile window blends smoothly at the short edge
# ---------------------------------------------------------------------------
def test_make_window_blends_partial_tile():
    """A 16-pixel-tall last tile is shorter than ``2 * overlap_h``
    (which defaults to 32).  The Y dimension must therefore carry a
    non-trivial ramp — strictly inside ``(0, 1)`` at every Y row —
    rather than the buggy behaviour of returning all ones (which would
    leave a hard seam against the penultimate tile)."""
    import torch

    from lbm_native.latent_codec import LatentCodec

    # overlap_h_pix = 16 → 2 * ovh = 32; tile is 16 → partial on Y.
    win = LatentCodec._make_window(H=16, W=128, ovh=16, oVw=16)
    assert win.shape == (1, 16, 128)
    # The X ramp is non-partial here (W=128 > 2 * ovw=32), so its
    # leftmost/rightmost columns are zeros.  We exclude those columns
    # AND the top/bottom rows (which legitimately bound the Y ramp at
    # 0 and 1).  The interior of the partial-Y ramp must be strictly
    # in (0, 1).
    interior = win[:, 1:-1, 16:-16]
    assert torch.all(interior > 0)
    assert torch.all(interior < 1)
    # The partial-Y ramp goes strictly from 0 at the top row to 1 at
    # the bottom row.
    y_column = win[0, :, 16]
    assert float(y_column[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(y_column[-1]) == pytest.approx(1.0, abs=1e-6)
    # Every intermediate Y row must be strictly inside (0, 1).
    for row in y_column[1:-1]:
        assert 0.0 < float(row) < 1.0


# ---------------------------------------------------------------------------
# PA17 — encode/decode must not build an autograd graph
# ---------------------------------------------------------------------------
def test_encode_decode_no_grad():
    """encode/decode must not propagate autograd from grad-enabled inputs.

    Even though ``vae_model.requires_grad_(False)`` is set, calling
    ``encode``/``decode`` without an explicit ``torch.no_grad()`` wrapper
    inside the codec still builds an autograd graph when the *input*
    has ``requires_grad=True`` (the codec is itself a Module, so
    ``requires_grad`` propagates through). The wrapper saves inference
    memory; this test pins that contract using a grad-enabled input so
    the absence of the wrapper is observable.
    """
    import torch

    codec = _make_codec()
    # Use a grad-enabled input — only then does the absence of the
    # ``no_grad`` wrapper surface as ``requires_grad=True`` on the output.
    x = torch.randn(1, 3, 32, 32, requires_grad=True)
    z = torch.randn(1, 4, 4, 4, requires_grad=True)

    assert torch.is_grad_enabled()  # confirm we're not inside no_grad
    encoded = codec.encode(x)
    assert encoded.requires_grad is False
    decoded = codec.decode(z)
    assert decoded.requires_grad is False

    # Calling ``.backward()`` on a no-grad tensor must raise.
    with pytest.raises(RuntimeError):
        encoded.sum().backward()
    with pytest.raises(RuntimeError):
        decoded.sum().backward()