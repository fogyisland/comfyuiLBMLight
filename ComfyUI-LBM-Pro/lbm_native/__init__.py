"""Independent LBM runtime — fully rewritten from scratch.

This package exposes the building blocks needed to load and run an
LBM checkpoint without depending on the legacy ``lbm`` package.

Public surface:
    BridgeSolver, BridgeSchedule  — solver + schedule
    CondUNet2D, PlainUNet2D       — denoiser wrappers
    LatentCodec, TilePlan         — pixel ↔ latent codec
    ConditionAggregator, BaseCondition  — conditioning branches
    ImageConcatCondition, ImageConcatConfig  — image+mask branch
    InferenceCore, StageConfig    — bookkeeping primitives
    gather_sigmas, predict_clean_state  — module-level helpers

Names are deliberately different from the legacy ``lbm.models.*``
modules so the two implementations can be told apart at a glance.
"""
from __future__ import annotations

from .bridge_solver import BridgeSolver, gather_sigmas, predict_clean_state
from .condition_aggregator import BaseCondition, ConditionAggregator, STACK_AXES
from .diffusion_unet import CondUNet2D, PlainUNet2D
from .image_concat_condition import ImageConcatCondition, ImageConcatConfig
from .inference_core import InferenceCore, StageConfig
from .latent_codec import LatentCodec, TilePlan
from .timestep_policy import BridgeSchedule


__all__ = [
    "BridgeSolver",
    "BridgeSchedule",
    "CondUNet2D",
    "PlainUNet2D",
    "LatentCodec",
    "TilePlan",
    "ConditionAggregator",
    "BaseCondition",
    "ImageConcatCondition",
    "ImageConcatConfig",
    "InferenceCore",
    "StageConfig",
    "gather_sigmas",
    "predict_clean_state",
    "STACK_AXES",
]
