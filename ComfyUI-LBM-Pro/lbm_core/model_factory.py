"""LBM model construction and checkpoint loading for LBM-Pro.

Builds the four collaborators (denoiser, scheduler, codec, aggregator)
and stitches them into a :class:`BridgeSolver`.  This module is the
only place that knows about the legacy checkpoint shape; nodes
consume the resulting solver via its public API.

The checkpoint key remapping lives here too — jasperai's
``model.safetensors`` stores VAE weights under the ``vae.vae_model.*``
prefix (and ``vae.quant_conv.*`` / ``vae.post_quant_conv.*`` for the
quantisation layers), while our rewriter stores the codec under
``codec.*``.  We rewrite keys on load and refuse to silently miss a
large share of the weights.
"""
from __future__ import annotations

import inspect
from typing import Literal

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models import AutoencoderKL

from comfy.utils import load_torch_file

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
        noise_jitter=spec["noise_jitter"],
    )
    return BridgeSolver(
        schedule=schedule,
        denoiser=_assemble_cond_unet(dtype),
        sampling_noise_scheduler=_assemble_scheduler(),
        codec=_assemble_codec(dtype),
        aggregator=ConditionAggregator(branches=[]),
    ).to(dtype)


# jasperai's safetensors file stores the autoencoder under the
# ``vae.vae_model.*`` prefix, with the quantisation convs sitting
# directly under ``vae.quant_conv.*`` and ``vae.post_quant_conv.*``.
# Our ``BridgeSolver`` keeps the codec under ``codec.*``; the
# ``LatentCodec`` itself stores the ``AutoencoderKL`` as
# ``self.vae_model`` (see ``lbm_native/latent_codec.py``), so the
# quantisation convs — which are direct attributes of the
# ``AutoencoderKL`` — need an extra ``vae_model.`` segment when
# remapped under the codec prefix.
_VAE_QUANT_KEYS = ("quant_conv.", "post_quant_conv.")


def _remap_checkpoint_key(key: str) -> str:
    """Translate a checkpoint key into the matching model key, if any.

    Three cases:
      * ``vae.vae_model.X``  → ``codec.vae_model.X`` (codec keeps the
        ``vae_model`` segment because the ``LatentCodec`` stores the
        ``AutoencoderKL`` as ``self.vae_model``).
      * ``vae.quant_conv.X`` / ``vae.post_quant_conv.X``
        → ``codec.vae_model.X`` (these convs are direct attributes
        of the ``AutoencoderKL``; the codec's wrapper adds the
        ``vae_model.`` segment).
      * everything else is passed through unchanged.
    """
    if not key.startswith("vae."):
        return key
    remainder = key[len("vae."):]
    if remainder.startswith("vae_model."):
        return "codec." + remainder
    if any(remainder.startswith(q) for q in _VAE_QUANT_KEYS):
        return "codec.vae_model." + remainder
    return key


def _build_key_index(sd_keys):
    """Group checkpoint keys by their post-remap target for one-pass lookups."""
    index: dict[str, str] = {}
    for k in sd_keys:
        remapped = _remap_checkpoint_key(k)
        # The first checkpoint key that maps to a given model key wins.
        # Subsequent ones are logged as duplicates and ignored.
        if remapped not in index:
            index[remapped] = k
    return index


def load_lbm_checkpoint(
    model: BridgeSolver,
    ckpt_path: str,
    dtype: torch.dtype,
    device: torch.device,
    *,
    min_match_ratio: float = 0.95,
) -> None:
    """Load safetensors weights into the solver in-place.

    The function remaps jasperai's ``vae.*`` keys into the
    ``codec.*`` keys used by the rewritten runtime (including the
    quantisation convs, which live under ``codec.vae_model.*`` in the
    rewritten model) and refuses to silently leave most parameters
    at their initial values.

    Args:
        model: solver to populate.
        ckpt_path: filesystem path of a ``.safetensors`` file.
        dtype: target dtype for the loaded tensors.
        device: device the checkpoint should be uploaded to.
        min_match_ratio: minimum fraction of solver parameters that
            must be matched by checkpoint keys; below this the load
            is treated as a failure and an exception is raised. The
            default of 0.95 is calibrated against a full LBM
            checkpoint — a 2-conv gap (e.g. missing both quantisation
            convs) drops the ratio below this threshold.

    Raises:
        RuntimeError: when the match ratio is too low — the caller
            almost certainly has a checkpoint that does not match
            the architecture.
    """
    _sig = inspect.signature(load_torch_file)
    if "safe_load" in _sig.parameters:
        sd = load_torch_file(ckpt_path, device=device, safe_load=True)
    else:
        sd = load_torch_file(ckpt_path, device=device)

    # Remap every checkpoint key through the same translation the
    # legacy loader applied, then feed the result to
    # ``load_state_dict`` with ``strict=False`` (so missing keys —
    # e.g. ``vae_model.X`` not present in the checkpoint — don't
    # raise).  ``load_state_dict`` updates parameters in-place via
    # ``param.data.copy_()``-style mechanics, preserving the
    # ``Parameter`` object — equivalent to the prior per-param
    # ``param.data = ...`` for inference use.  We dedupe on the
    # post-remap target so the first checkpoint key mapping to any
    # given model key wins, matching the prior ``_build_key_index``
    # behaviour.
    new_sd: dict = {}
    for ckpt_key, tensor in sd.items():
        target = _remap_checkpoint_key(ckpt_key)
        if target not in new_sd:
            new_sd[target] = tensor.to(dtype=dtype, device=device)

    model_params = dict(model.named_parameters())
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    matched = len(new_sd) - len(unexpected)
    total = len(model_params)
    ratio = matched / total if total else 0.0
    if ratio < min_match_ratio:
        sample_missing = sorted(missing)[:8]
        raise RuntimeError(
            "LBM checkpoint load failed: only "
            f"{matched}/{total} ({ratio:.0%}) parameters matched. "
            "The checkpoint likely does not match the expected "
            "architecture. First few missing keys: "
            f"{sample_missing}"
        )
