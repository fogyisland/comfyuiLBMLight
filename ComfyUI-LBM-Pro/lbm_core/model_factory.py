"""LBM model construction and checkpoint loading for LBM-Pro.

Centralizes the duplicated architecture blocks from the original
LBM_Relighting.py and LBM_DepthNormal.py node implementations.
"""
from __future__ import annotations

from typing import Literal

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models import AutoencoderKL

from lbm.models.embedders import ConditionerWrapper
from lbm.models.lbm import LBMConfig, LBMModel
from lbm.models.unets import DiffusersUNet2DCondWrapper
from lbm.models.vae import AutoencoderKLDiffusers


_TASK_CONFIG = {
    "relighting": {
        "target_key": "source_image",
        "prob": [0.25, 0.25, 0.25, 0.25],
        "default_sigma": 0.005,
    },
    "depth": {
        "target_key": "depth",
        "prob": [0.025, 0.05, 0.025, 0.9],
        "default_sigma": 0.1,
    },
    "normal": {
        "target_key": "normals",
        "prob": [0.05, 0.1, 0.05, 0.8],
        "default_sigma": 0.1,
    },
}


def _build_unet(dtype: torch.dtype) -> DiffusersUNet2DCondWrapper:
    return DiffusersUNet2DCondWrapper(
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


def _build_vae(dtype: torch.dtype) -> AutoencoderKLDiffusers:
    vae_config = {
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
    vae = AutoencoderKLDiffusers(AutoencoderKL.from_config(vae_config))
    vae.freeze()
    vae.to(dtype)
    return vae


def _build_scheduler() -> FlowMatchEulerDiscreteScheduler:
    scheduler_config = {
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "use_dynamic_shifting": False,
        "beta_schedule": "scaled_linear",
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "timestep_spacing": "leading",
    }
    return FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)


def build_lbm_model(
    task: Literal["relighting", "depth", "normal"],
    dtype: torch.dtype,
    bridge_noise_sigma: float,
) -> LBMModel:
    """Construct an LBM model with the architecture matching the task."""
    if task not in _TASK_CONFIG:
        raise ValueError(f"Unknown task '{task}'. Expected one of {list(_TASK_CONFIG)}")
    cfg = _TASK_CONFIG[task]
    config = {
        "source_key": "source_image",
        "target_key": cfg["target_key"],
        "timestep_sampling": "custom_timesteps",
        "selected_timesteps": [250, 500, 750, 1000],
        "prob": cfg["prob"],
        "bridge_noise_sigma": bridge_noise_sigma,
    }
    return LBMModel(
        LBMConfig(**config),
        denoiser=_build_unet(dtype),
        sampling_noise_scheduler=_build_scheduler(),
        vae=_build_vae(dtype),
        conditioner=ConditionerWrapper(conditioners=[]),
    ).to(dtype)


def load_lbm_checkpoint(
    model: LBMModel,
    ckpt_path: str,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Load safetensors weights into the model in-place."""
    from comfy.utils import load_torch_file

    sd = load_torch_file(ckpt_path, device=device, safe_load=True)
    for name, param in model.named_parameters():
        if name in sd:
            param.data = sd[name].to(dtype=dtype)
