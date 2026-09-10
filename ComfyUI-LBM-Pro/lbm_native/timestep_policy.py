"""Schedule dataclass for the bridge-matching training loop.

The class is intentionally narrow: it only carries the fields that
are actually consumed by :mod:`bridge_solver`.  Inference-only
callers can omit training-related fields.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple


TimestepMode = Literal["uniform", "log_normal", "discrete"]


@dataclass(slots=True)
class BridgeSchedule:
    """Static description of how the bridge solver advances.

    Attributes:
        anchor_field:    Batch key carrying the source image.  Used by
                         the bridge interpolant to compute the noise
                         mixing term.
        goal_field:      Batch key carrying the target image.  Used to
                         derive the supervision signal.
        mask_field:      Optional batch key carrying the validity mask.
                         ``None`` disables masking.
        noise_jitter:    Bridge-noise scale (sigma) — controls how
                         much randomness is added on top of the linear
                         interpolant.
        timestep_policy: One of ``"uniform"``, ``"log_normal"``,
                         ``"discrete"``.
        discrete_timesteps: integer timesteps sampled when ``timestep_policy == "discrete"``.
        discrete_weights:   sampling weights for ``discrete_timesteps``;
                            must sum to 1 (within float tolerance).
        latent_loss_kind:   ``"l2"`` or ``"l1"``.
        pixel_loss_kind:    ``"l2"``, ``"l1"``, or ``"lpips"``.
        pixel_loss_max_size: crop size used for the pixel-loss term.
        pixel_loss_weight:   mixing weight of the pixel loss.
        logit_mean, logit_std: parameters for ``"log_normal"``.
    """

    anchor_field: str = "source_image"
    goal_field: str = "target_image"
    mask_field: Optional[str] = None

    noise_jitter: float = 0.001
    timestep_policy: TimestepMode = "uniform"

    discrete_timesteps: Optional[Sequence[int]] = None
    discrete_weights: Optional[Sequence[float]] = None

    logit_mean: float = 0.0
    logit_std: float = 1.0

    latent_loss_kind: Literal["l2", "l1"] = "l2"
    pixel_loss_kind: Literal["l2", "l1", "lpips"] = "l2"
    pixel_loss_max_size: int = 512
    pixel_loss_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.timestep_policy == "log_normal":
            if not isinstance(self.logit_mean, float) or not isinstance(self.logit_std, float):
                raise TypeError("log_normal mode requires float logit_mean/logit_std")
        if self.timestep_policy == "discrete":
            self._validate_discrete()

    def _validate_discrete(self) -> None:
        if not isinstance(self.discrete_timesteps, (list, tuple)) or not isinstance(
            self.discrete_weights, (list, tuple)
        ):
            raise TypeError(
                f"discrete mode requires list/tuple timesteps and weights; "
                f"got {type(self.discrete_timesteps).__name__} / "
                f"{type(self.discrete_weights).__name__}"
            )
        if len(self.discrete_timesteps) != len(self.discrete_weights):
            raise ValueError(
                f"discrete_timesteps / discrete_weights length mismatch "
                f"({len(self.discrete_timesteps)} vs {len(self.discrete_weights)})"
            )
        total = float(sum(self.discrete_weights))
        if abs(total - 1.0) > 1e-3:
            raise ValueError(
                f"discrete_weights must sum to 1.0; got {total}"
            )
