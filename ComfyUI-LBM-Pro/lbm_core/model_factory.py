"""LBM model construction and checkpoint loading for LBM-Pro.

Builds the four collaborators (denoiser, scheduler, codec, aggregator)
and stitches them into a :class:`BridgeSolver`.  This module is the
only place that knows about the legacy checkpoint shape; nodes
consume the resulting solver via its public API.
"""
from __future__ import annotations

from typing import Literal

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models import AutoencoderKL

from lbm_native import (
    BridgeSchedule,
    BridgeSolver,
    ConditionAggregator,
    CondUNet2D,
    LatentCodec,
)


_TASK_SCHEDULE = {
    "relighting": {
        "goal_field": "source_image",
        "discrete_weights": [0.25, 0.25, 0.25, 0.25],
        "noise_jitter": 0.005,
    },
    "depth": {
        "goal_field": "depth",
        "discrete_weights": [0.025, 0.05, 0.025, 0.9],
        "noise_jitter": 0.1,
    },
    "normal": {
        "goal_field": "normals",
        "discrete_weights": [0.05, 0.1, 0.05, 0.8],
        "noise_jitter": 0.1,
    },
}


def _assemble_cond_unet(dtype: torch.dtype) -> CondUNet2D:
    return CondUNet2D(
        in_channels=4,
        out_channels=4,
        center_input_sample=False,
        flip_sin_to_cos=True,
        freq_shift=0,
        down_block_types=[
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
        ],
        mid_block_type="UNetMidBlock2DCrossAttn",
        up_block_types=["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
        only_cross_attention=False,
        block_out_channels=[320, 640, 1280],
        layers_per_block=2,
        downsample_padding=1,
        mid_block_scale_factor=1,
        dropout=0.0,
        act_fn="silu",
        norm_num_groups=32,
        norm_eps=1e-05,
        cross_attention_dim=[320, 640, 1280],
        transformer_layers_per_block=[1, 2, 10],
        attention_head_dim=[5, 10, 20],
        use_linear_projection=True,
        time_embedding_type="positional",
    ).to(dtype)


def _assemble_codec(dtype: torch.dtype) -> LatentCodec:
    cfg = {
        "_class_name": "AutoencoderKL",
        "_diffusers_version": "0.20.0.dev0",
        "act_fn": "silu",
        "block_out_channels": [128, 256, 512, 512],
        "down_block_types": [
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        ],
        "force_upcast": True,
        "in_channels": 3,
        "latent_channels": 4,
        "layers_per_block": 2,
        "norm_num_groups": 32,
        "out_channels": 3,
        "sample_size": 1024,
        "scaling_factor": 0.13025,
        "up_block_types": [
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
        ],
    }
    vae = LatentCodec(AutoencoderKL.from_config(cfg))
    vae.to(dtype)
    return vae


def _assemble_scheduler() -> FlowMatchEulerDiscreteScheduler:
    return FlowMatchEulerDiscreteScheduler.from_config(
        {
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "use_dynamic_shifting": False,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "timestep_spacing": "leading",
        }
    )


def build_lbm_model(
    task: Literal["relighting", "depth", "normal"],
    dtype: torch.dtype,
    bridge_noise_sigma: float,
) -> BridgeSolver:
    """Construct a solver wired with the architecture for ``task``."""
    if task not in _TASK_SCHEDULE:
        raise ValueError(f"Unknown task '{task}'. Expected one of {list(_TASK_SCHEDULE)}")
    spec = _TASK_SCHEDULE[task]
    schedule = BridgeSchedule(
        anchor_field="source_image",
        goal_field=spec["goal_field"],
        timestep_policy="discrete",
        discrete_timesteps=[250, 500, 750, 1000],
        discrete_weights=spec["discrete_weights"],
        noise_jitter=bridge_noise_sigma,
    )
    return BridgeSolver(
        schedule=schedule,
        denoiser=_assemble_cond_unet(dtype),
        sampling_noise_scheduler=_assemble_scheduler(),
        codec=_assemble_codec(dtype),
        aggregator=ConditionAggregator(branches=[]),
    ).to(dtype)


def load_lbm_checkpoint(
    model: BridgeSolver,
    ckpt_path: str,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Load safetensors weights into the solver in-place.

    Accepts ``model: BridgeSolver``; the type signature is the only
    piece of legacy nomenclature retained on this function's API.
    """
    from comfy.utils import load_torch_file

    sd = load_torch_file(ckpt_path, device=device, safe_load=True)
    for name, param in model.named_parameters():
        if name in sd:
            param.data = sd[name].to(dtype=dtype)
