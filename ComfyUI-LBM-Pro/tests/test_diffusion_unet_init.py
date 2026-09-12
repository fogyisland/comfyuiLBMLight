"""Regression tests for :mod:`lbm_native.diffusion_unet` init wiring.

The diamond MRO of ``CondUNet2D`` / ``PlainUNet2D`` (``X → _BaseDiffusersUNet
→ InferenceCore → diffusers.UNet2DConditionModel / UNet2DModel``) makes it
easy to accidentally double-initialise the diffusers parent. The original
implementations explicitly called the diffusers ``__init__`` first, then
``InferenceCore.__init__`` — but ``InferenceCore.__init__`` walks ``super()``
back to the diffusers class and triggers a *second* init with all-default
arguments, silently overwriting every architecture argument.

These tests assert that the architecture arguments actually take effect:
the resulting model has the requested ``block_out_channels``,
``cross_attention_dim``, ``transformer_layers_per_block``, and
``use_linear_projection`` — and that ``proj_in`` is ``nn.Linear`` when
``use_linear_projection=True`` (otherwise cross-attention ``to_k`` /
``to_v`` would silently concat encoder dims and break checkpoint load).
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from lbm_native.diffusion_unet import CondUNet2D, PlainUNet2D


# The exact configuration ``lbm_core.model_factory._assemble_cond_unet``
# uses for all three tasks (relighting / depth / normals).  If any of
# these values changes, this test will catch it.
_JASPERAI_COND_CONFIG = dict(
    in_channels=4,
    out_channels=4,
    down_block_types=["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
    mid_block_type="UNetMidBlock2DCrossAttn",
    up_block_types=["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
    block_out_channels=[320, 640, 1280],
    layers_per_block=2,
    cross_attention_dim=[320, 640, 1280],
    transformer_layers_per_block=[1, 2, 10],
    attention_head_dim=[5, 10, 20],
    use_linear_projection=True,
)


def test_cond_unet2d_keeps_use_linear_projection_true():
    """``use_linear_projection=True`` must reach ``Attention.proj_in``."""
    model = CondUNet2D(**_JASPERAI_COND_CONFIG)
    attn = model.down_blocks[1].attentions[0]
    assert attn.proj_in.__class__ is nn.Linear, (
        "proj_in must be nn.Linear when use_linear_projection=True; got "
        f"{attn.proj_in.__class__.__name__}. This typically means "
        "UNet2DConditionModel.__init__ ran twice — once with the right "
        "kwargs and once with defaults that flipped use_linear_projection "
        "back to False."
    )


def test_cond_unet2d_block_out_channels_not_extended():
    """``block_out_channels`` must stay 3-element, not silently grow to 4."""
    model = CondUNet2D(**_JASPERAI_COND_CONFIG)
    # diffusers UNet2DConditionModel stores the resolved config as a tuple
    assert tuple(model.config.block_out_channels) == (320, 640, 1280)


def test_cond_unet2d_cross_attention_dim_per_level():
    """``cross_attention_dim`` must remain per-level, not collapse to scalar."""
    model = CondUNet2D(**_JASPERAI_COND_CONFIG)
    cad = model.config.cross_attention_dim
    # diffusers either keeps a list or broadcasts — the resolved values
    # applied to each attention layer must match the per-level spec.
    assert tuple(cad) == (320, 640, 1280), (
        f"cross_attention_dim collapsed: got {cad!r}, expected "
        "(320, 640, 1280). The default init round overwrote it with 1280."
    )


def test_cond_unet2d_transformer_layers_per_block_kept():
    """``transformer_layers_per_block=[1, 2, 10]`` must survive init."""
    model = CondUNet2D(**_JASPERAI_COND_CONFIG)
    assert tuple(model.config.transformer_layers_per_block) == (1, 2, 10)


def test_cond_unet2d_cross_attn_to_k_does_not_concat_encoder():
    """With ``use_linear_projection=True``, ``to_k.weight`` has shape
    ``(out_dim, cross_attention_dim_for_this_level)`` — NOT
    ``(out_dim, cross_attention_dim_for_this_level + encoder_hidden_dim)``.

    With jasperai's config, ``down_blocks[1]`` is the second block
    (a ``CrossAttnDownBlock2D``) with output channels 640 and
    ``cross_attention_dim=640`` (the second entry of ``[320, 640, 1280]``).
    jasperai's checkpoint stores ``Linear[640, 640]`` — i.e. NO encoder
    concat.  If ``use_linear_projection`` had silently been flipped
    back to ``False`` by the double-init bug, ``to_k`` would either be
    a 4-D ``Conv2d`` or, in the linear case, ``Linear[640, 1280]``
    (encoder_dim concatenated)."""
    model = CondUNet2D(**_JASPERAI_COND_CONFIG)
    to_k = model.down_blocks[1].attentions[0].transformer_blocks[0].attn2.to_k
    assert tuple(to_k.weight.shape) == (640, 640), (
        f"attn2.to_k.weight has shape {tuple(to_k.weight.shape)}; expected "
        "(640, 640) — i.e. no encoder-dim concatenation. Cross-attn is "
        "either using ``Conv2d`` (because ``use_linear_projection`` "
        "silently flipped back to ``False``) or the encoder dim is "
        "being concatenated."
    )


def test_cond_unet2d_proj_in_weight_is_2d():
    """``proj_in.weight`` must be 2-D (``nn.Linear``), not 4-D
    (``nn.Conv2d``). The checkpoint format is 2-D."""
    model = CondUNet2D(**_JASPERAI_COND_CONFIG)
    proj_in = model.down_blocks[1].attentions[0].proj_in
    assert proj_in.weight.ndim == 2, (
        f"proj_in.weight has {proj_in.weight.ndim} dims; expected 2 "
        "(nn.Linear under use_linear_projection=True)."
    )


def test_cond_unet2d_hard_freeze_disables_grad():
    """``hard_freeze`` (inherited from ``_BaseDiffusersUNet``) must iterate
    parameters that actually exist after the correct init — i.e. it must
    freeze the cross-attn attention parameters too, not just the down
    conv stack."""
    model = CondUNet2D(**_JASPERAI_COND_CONFIG)
    model.hard_freeze()
    # Spot-check: at least one parameter inside a cross-attention block.
    to_k = model.down_blocks[1].attentions[0].transformer_blocks[0].attn2.to_k
    assert not to_k.weight.requires_grad


def test_plain_unet2d_keeps_block_out_channels():
    """``PlainUNet2D`` has the same diamond MRO; the same regression."""
    model = PlainUNet2D(
        in_channels=4,
        out_channels=4,
        block_out_channels=(32, 64),
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
    )
    assert tuple(model.config.block_out_channels) == (32, 64)
