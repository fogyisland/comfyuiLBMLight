"""Bridge solver — the public entry point for training and inference.

Owns the four collaborators (denoiser, codec, aggregator, scheduler)
and exposes:

  * :meth:`training_step`     — one bridge-matching gradient step
  * :meth:`decode_latents_to_pixels` — Euler integration that turns a
    latent batch into a pixel-space image
  * :meth:`sample`            — deprecated alias for backwards
    compatibility with code still using the legacy ``LBMModel.sample``
    name

The naming convention deliberately diverges from the legacy
``LBMModel.sample``/``forward`` style so any caller porting across
implementations has to consciously update their code.
"""
from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from .condition_aggregator import BaseCondition, ConditionAggregator
from .diffusion_unet import CondUNet2D, PlainUNet2D
from .inference_core import InferenceCore
from .latent_codec import LatentCodec
from .timestep_policy import BridgeSchedule


# Module-level dedup set for DeprecationWarning suppression (C18).
# The legacy ``sample`` alias and other deprecation paths fire once per
# call site — we keep that one-shot behaviour per process so the log
# stays clean even when many nodes share a cached model.
_WARNED: set = set()


def _warn_once(key: str, message: str, category: type = DeprecationWarning) -> None:
    """Emit ``message`` at ``category`` at most once per ``key`` per process."""
    if key in _WARNED:
        return
    _WARNED.add(key)
    warnings.warn(message, category, stacklevel=3)


# Module-level helpers (not methods) so other modules can re-use them
# without instantiating a full bridge solver.

def gather_sigmas(
    scheduler: FlowMatchEulerDiscreteScheduler,
    timesteps: torch.Tensor,
    n_dim: int = 4,
    dtype: torch.dtype = torch.float32,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Look up scheduler sigmas at the requested timesteps.

    Returns a tensor right-padded with singleton dimensions so it
    broadcasts cleanly against a latent ``(B, C, H, W)`` tensor.

    The lookup table mapping ``timestep -> sigma index`` is cached on
    the scheduler (``scheduler._sigma_index``) the first time this
    function is called and reused for every subsequent call.  If the
    caller reconfigures the scheduler via ``set_timesteps``, the
    cache must be invalidated by deleting ``scheduler._sigma_index``
    before re-querying — the default cache key is per-process.
    """
    if not hasattr(scheduler, "_sigma_index"):
        schedule_ts = scheduler.timesteps.to(device)
        scheduler._sigma_index = {int(t): i for i, t in enumerate(schedule_ts.tolist())}
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    timesteps = timesteps.to(device)
    pos = scheduler._sigma_index
    flat = sigmas[[pos[int(t)] for t in timesteps.tolist()]].flatten()
    while flat.dim() < n_dim:
        flat = flat.unsqueeze(-1)
    return flat


def predict_clean_state(sample: torch.Tensor, model_output: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Closed-form x_0 under the flow-matching prediction rule.

    Returns the predicted clean state given the current noisy
    ``sample``, the model's predicted bridge direction ``model_output``,
    and the scalar (or broadcastable) ``sigma``.  Implemented as
    ``x_0 = sample - sigma * model_output``.
    """
    return sample - sigma * model_output


class BridgeSolver(InferenceCore):
    """Owns the four collaborators and exposes training/inference.

    The class intentionally does NOT inherit from any custom base
    beyond :class:`InferenceCore`.  This keeps the dependency graph
    shallow and easy to reason about.
    """

    def __init__(
        self,
        schedule: BridgeSchedule,
        denoiser: nn.Module,
        sampling_noise_scheduler: FlowMatchEulerDiscreteScheduler,
        codec: Optional[LatentCodec] = None,
        aggregator: Optional[ConditionAggregator] = None,
    ) -> None:
        super().__init__()
        # Register collaborators as proper submodules so ``.to()``
        # cascades correctly and so the state_dict reflects the
        # architecture (codec.* / denoiser.* / aggregator.*).
        self.schedule = schedule
        self.denoiser = denoiser
        self.codec = codec
        self.aggregator = aggregator
        self.sampling_noise_scheduler = sampling_noise_scheduler
        self.training_iter = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)

    # ------------------------------------------------------------------
    # Deprecated alias
    # ------------------------------------------------------------------
    def sample(self, *args, **kwargs):
        """Deprecated alias for :meth:`decode_latents_to_pixels`.

        Kept on the class (not bound dynamically in ``__init__``) so
        the alias survives ``pickle`` round-trips and ``to(device)``
        operations.  The deprecation warning is suppressed after the
        first emission per process so a long-running graph does not
        spam the console.
        """
        _warn_once(
            "BridgeSolver.sample",
            "BridgeSolver.sample is deprecated; call decode_latents_to_pixels() instead.",
        )
        return self.decode_latents_to_pixels(*args, **kwargs)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def training_step(self, batch: Dict[str, Any], *args, **kwargs) -> Dict[str, torch.Tensor]:
        """One bridge-matching step.

        Mirrors the legacy forward pass but exposes the loss pieces
        individually so callers can build their own logging.
        """
        self.training_iter += 1

        goal = batch[self.schedule.goal_field]
        goal_latents = self._encode(goal)
        anchor = self._resize_anchor(batch[self.schedule.anchor_field], goal.shape[-2:])
        anchor_latents = self._encode(anchor)

        valid_mask = self._build_valid_mask(batch, goal)
        valid_mask_latent = self._shrink_mask_for_latents(valid_mask, anchor_latents)

        guide = self._collect_guide(batch)

        timestep = self._sample_timestep(n=goal_latents.shape[0], device=goal_latents.device)
        sigmas = gather_sigmas(
            self.sampling_noise_scheduler,
            timestep,
            n_dim=4,
            dtype=goal_latents.dtype,
            device=goal_latents.device,
        )
        noisy = self._mix_bridge(anchor_latents, goal_latents, sigmas)

        # Boundary correction: when t equals the first timestep, force
        # the noisy sample to equal the anchor.  This matches the
        # legacy behaviour and keeps the bridge interpolant well-defined.
        first_t = self.sampling_noise_scheduler.timesteps[0]
        for i, t in enumerate(timestep):
            if int(t.item()) == int(first_t):
                noisy[i] = anchor_latents[i]

        prediction = self.denoiser(sample=noisy, timestep=timestep, guide=guide, *args, **kwargs)

        target = anchor_latents - goal_latents
        denoised = predict_clean_state(noisy, prediction, sigmas)

        latent_recon = torch.zeros(goal_latents.shape[0], device=goal_latents.device)
        if self.schedule.latent_loss_kind in ("l2", "l1"):
            latent_recon = self.latent_recon_err(prediction, target.detach(), valid_mask_latent)

        pixel_recon = torch.zeros_like(latent_recon)
        if self.schedule.pixel_loss_weight > 0 and self.codec is not None:
            pixel_recon = self.pixel_recon_err(
                denoised,
                goal.detach(),
                valid_mask,
            )
            total = latent_recon.mean() + self.schedule.pixel_loss_weight * pixel_recon.mean()
        else:
            total = latent_recon.mean()

        return {
            "loss": total,
            "latent_recon_loss": latent_recon.mean(),
            "pixel_recon_loss": pixel_recon.mean(),
            "predicted_hr": denoised,
            "noisy_sample": noisy,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def decode_latents_to_pixels(
        self,
        z: torch.Tensor,
        num_steps: int = 20,
        conditioner_inputs: Optional[Dict[str, Any]] = None,
        max_samples: Optional[int] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> torch.Tensor:
        """Integrate the bridge ODE from ``z`` to a pixel image.

        ``z`` is assumed to be a latent batch (B, C, H/8, W/8).  The
        scheduler is reconfigured in-place to span ``num_steps``
        Euler steps; the callback receives ``(completed, total)``.
        Inference is deterministic — noise injection lives in the
        training loop (``_mix_bridge`` consumes
        ``schedule.noise_jitter``), not at decode time.
        """
        # Reconfigure the scheduler to span ``num_steps`` Euler steps
        # using its built-in training-timestep schedule.  Passing
        # ``num_inference_steps`` keeps sigma/timestep inside the
        # range the UNet was trained on; the previous linspace
        # override pushed sigma out of distribution on every call.
        self.sampling_noise_scheduler.set_timesteps(
            num_inference_steps=num_steps,
            device=z.device,
        )
        sample = z
        guide = self._collect_guide(conditioner_inputs or {}, set_ucg_rate_zero=True)

        if max_samples is not None:
            sample = sample[:max_samples]
            if "guide_pack" in guide:
                guide["guide_pack"] = {
                    k: v[:max_samples] for k, v in guide["guide_pack"].items()
                }

        timesteps = list(self.sampling_noise_scheduler.timesteps)
        for i, t in enumerate(timesteps):
            denoiser_in = self.sampling_noise_scheduler.scale_model_input(sample, t) \
                if hasattr(self.sampling_noise_scheduler, "scale_model_input") else sample
            t_batched = t.to(sample.device).repeat(denoiser_in.shape[0])
            # Cast the timestep to the sample dtype so an fp16/bf16
            # UNet never falls back through an fp64 timestep tensor.
            t_batched = t_batched.to(sample.dtype)
            prediction = self.denoiser(sample=denoiser_in, timestep=t_batched, guide=guide)
            sample = self.sampling_noise_scheduler.step(prediction, t, sample, return_dict=False)[0]
            if progress_cb is not None:
                progress_cb(i + 1, len(timesteps))

        # Numerical safety net: fp16/fp32 drift in the Euler loop
        # can produce a stray nan/inf that would otherwise turn the
        # entire output image black or white.  We clamp to a sane
        # latent range and patch up any non-finite values before
        # handing the sample to the codec.
        sample = torch.nan_to_num(sample, nan=0.0, posinf=4.0, neginf=-4.0)
        sample = sample.clamp(-4.0, 4.0)

        if self.codec is not None:
            return self.codec.decode(sample)
        return sample

    # ------------------------------------------------------------------
    # Convenience shims
    # ------------------------------------------------------------------
    def log_samples(self, batch: Dict[str, Any], max_samples: Optional[int] = None, num_steps: int = 20) -> Dict[str, torch.Tensor]:
        limit = max_samples if max_samples is not None else batch[self.schedule.anchor_field].shape[0]
        slice_ = {k: v[:limit] for k, v in batch.items()}
        anchor = slice_[self.schedule.anchor_field]
        anchor = torch.nn.functional.interpolate(
            anchor,
            size=slice_[self.schedule.goal_field].shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).to(self.dtype)
        z = self._encode(anchor)
        return {
            f"samples_{num_steps}_steps": self.decode_latents_to_pixels(
                z,
                num_steps=num_steps,
                conditioner_inputs=slice_,
                max_samples=limit,
            )
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _encode(self, image: torch.Tensor) -> torch.Tensor:
        if self.codec is None:
            return image
        return self.codec.encode(image)

    @staticmethod
    def _resize_anchor(anchor: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        if anchor.shape[-2:] == target_hw:
            return anchor
        return torch.nn.functional.interpolate(
            anchor,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )

    def _build_valid_mask(self, batch: Dict[str, Any], goal: torch.Tensor) -> torch.Tensor:
        key = self.schedule.mask_field
        if key is None or key not in batch:
            return torch.ones_like(goal).bool()
        mask = batch[key]
        if mask.ndim != 4:
            raise ValueError(
                f"mask must be 4-D (B, 1, H, W); got {tuple(mask.shape)} "
                f"for batch key {key!r}"
            )
        if mask.shape[0] != goal.shape[0]:
            raise ValueError(
                f"mask batch dim must equal goal batch dim "
                f"(got {mask.shape[0]} vs {goal.shape[0]}) for batch key {key!r}"
            )
        if mask.shape[1] != 1:
            mask = mask[:, :1]
        return mask.bool()

    @staticmethod
    def _shrink_mask_for_latents(valid_mask: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        invalid = ~valid_mask[:, 0:1]
        downsample = 8
        pooled = ~torch.nn.functional.max_pool2d(
            invalid.float(),
            downsample,
            downsample,
        ).bool()
        return pooled.repeat(1, latents.shape[1], 1, 1)

    def _collect_guide(
        self,
        batch: Dict[str, Any],
        set_ucg_rate_zero: bool = False,
    ) -> Optional[Dict[str, Any]]:
        if self.aggregator is None:
            return None
        # ``codec`` is deliberately NOT passed: the aggregator does not
        # forward extras to branches.  Branches that need the codec own
        # their own reference (see ImageConcatCondition).
        return self.aggregator(
            batch,
            set_ucg_rate_zero=set_ucg_rate_zero,
        )

    def _mix_bridge(
        self,
        anchor: torch.Tensor,
        goal: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> torch.Tensor:
        """Construct the linear-bridge noisy sample.

        ``anchor`` is the source, ``goal`` is the target.  Sigma
        blends linearly between them, with optional Gaussian jitter
        scaled by the bridge-noise term.
        """
        noise = torch.randn_like(anchor)
        bridge = self.schedule.noise_jitter * (sigmas * (1.0 - sigmas)).sqrt() * noise
        return sigmas * anchor + (1.0 - sigmas) * goal + bridge

    def _sample_timestep(
        self,
        n: int,
        device: torch.device,
        generator: Optional[np.random.Generator] = None,
    ) -> torch.Tensor:
        policy = self.schedule.timestep_policy
        if policy == "uniform":
            idx = torch.randint(
                0,
                len(self.sampling_noise_scheduler.timesteps),
                (n,),
                device="cpu",
            )
            return self.sampling_noise_scheduler.timesteps[idx].to(device=device)
        if policy == "log_normal":
            u = torch.normal(
                mean=self.schedule.logit_mean,
                std=self.schedule.logit_std,
                size=(n,),
                device="cpu",
            )
            u = torch.sigmoid(u)
            indices = (u * len(self.sampling_noise_scheduler.timesteps)).long()
            return self.sampling_noise_scheduler.timesteps[indices].to(device=device)
        # discrete
        idx_np = np.random.choice(
            len(self.schedule.discrete_timesteps),
            n,
            p=self.schedule.discrete_weights,
            generator=generator,
        )
        table = torch.tensor(self.schedule.discrete_timesteps, device=device, dtype=torch.long)
        return table[idx_np]

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------
    def latent_recon_err(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.schedule.latent_loss_kind == "l2":
            return torch.mean(
                ((prediction * valid_mask - target * valid_mask) ** 2)
                .reshape(target.shape[0], -1),
                dim=1,
            )
        if self.schedule.latent_loss_kind == "l1":
            return torch.mean(
                torch.abs(prediction * valid_mask - target * valid_mask)
                .reshape(target.shape[0], -1),
                dim=1,
            )
        raise NotImplementedError(f"latent loss '{self.schedule.latent_loss_kind}' is not implemented")

    def pixel_recon_err(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.codec is None:
            raise RuntimeError("pixel loss requires a codec")
        latent_crop = self.schedule.pixel_loss_max_size // self.codec.downsampling_factor
        input_crop = self.schedule.pixel_loss_max_size
        # Random crop offsets (computed from CPU rng)
        crop_h = max(prediction.shape[2] - latent_crop, 0)
        crop_w = max(prediction.shape[3] - latent_crop, 0)
        input_crop_h = max(target.shape[2] - input_crop, 0)
        input_crop_w = max(target.shape[3] - input_crop, 0)
        offset_h = int(torch.randint(0, crop_h + 1, (1,)).item()) if crop_h > 0 else 0
        offset_w = int(torch.randint(0, crop_w + 1, (1,)).item()) if crop_w > 0 else 0
        offset_in_h = offset_h * self.codec.downsampling_factor
        offset_in_w = offset_w * self.codec.downsampling_factor

        p_crop = prediction[
            :,
            :,
            crop_h - offset_h : crop_h - offset_h + latent_crop,
            crop_w - offset_w : crop_w - offset_w + latent_crop,
        ]
        t_crop = target[
            :,
            :,
            input_crop_h - offset_in_h : input_crop_h - offset_in_h + input_crop,
            input_crop_w - offset_in_w : input_crop_w - offset_in_w + input_crop,
        ]
        m_crop = valid_mask[
            :,
            :,
            input_crop_h - offset_in_h : input_crop_h - offset_in_h + input_crop,
            input_crop_w - offset_in_w : input_crop_w - offset_in_w + input_crop,
        ]
        decoded = self.codec.decode(p_crop).clamp(-1, 1)
        if self.schedule.pixel_loss_kind == "l2":
            return torch.mean(
                ((decoded * m_crop - t_crop * m_crop) ** 2).reshape(t_crop.shape[0], -1),
                dim=1,
            )
        if self.schedule.pixel_loss_kind == "l1":
            return torch.mean(
                torch.abs(decoded * m_crop - t_crop * m_crop).reshape(t_crop.shape[0], -1),
                dim=1,
            )
        if self.schedule.pixel_loss_kind == "lpips":
            return self._lpips_loss(decoded * m_crop, t_crop * m_crop).mean()
        raise NotImplementedError(f"pixel loss '{self.schedule.pixel_loss_kind}' is not implemented")

    def _lpips_loss(self, *args, **kwargs):
        # LPIPS is optional; we only import it on demand to keep the
        # base install footprint small.
        try:
            import lpips  # type: ignore
        except ImportError as e:  # pragma: no cover - depends on env
            raise RuntimeError(
                "pixel_loss_kind='lpips' requires the lpips package; install it with `pip install lpips`"
            ) from e
        if not hasattr(self, "_lpips"):
            self._lpips = lpips.LPIPS(net="vgg")
        return self._lpips(*args, **kwargs)
