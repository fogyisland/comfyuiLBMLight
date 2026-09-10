"""Primitives shared by every component participating in inference.

Owns the device/dtype bookkeeping that other modules in this package
delegate to.  Anything that holds tensors and needs to be moved across
devices without losing track of its origin dtype should derive from
``InferenceCore``.

Design notes:
  * We deliberately do not subclass ``BaseModel`` or anything else —
    the class is meant to be inherited only by siblings that agree on
    a flat attribute layout.
  * Dtype inference does NOT silently upcast or downcast.  Promote /
    demote decisions belong to the caller; this class only tracks what
    is currently bound.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass(slots=True)
class StageConfig:
    """A neutral config holder.

    Carries nothing more than an input name and a free-form kwargs
    bag.  Subclasses may add typed fields on top; this base stays
    free of any LBM-specific vocabulary so it can be re-used by any
    future stage.
    """

    input_key: str = "image"
    extra: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.extra is None:
            self.extra = {}


class InferenceCore(nn.Module):
    """Tracks device and dtype, provides ergonomic ``move_to`` helpers.

    The class is intentionally thin.  It exists so other modules can
    declare ``self.device`` and ``self.dtype`` once, and call a
    single method to refresh both after a ``.to(...)`` call.
    """

    def __init__(self, config: Optional[StageConfig] = None) -> None:
        super().__init__()
        self.stage_config = config or StageConfig()
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def move_to(self, *, device=None, dtype=None, non_blocking: bool = False) -> "InferenceCore":
        """Apply ``.to(...)`` and remember the resolved device and dtype.

        Accepts keyword arguments to mirror the spirit of ``Module.to``
        without inheriting its quirky positional semantics.  Returns
        ``self`` so calls can be chained.
        """
        kwargs: dict = {}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype
        if non_blocking:
            kwargs["non_blocking"] = True
        if kwargs:
            super().to(**kwargs)
        if device is not None:
            self.device = device if isinstance(device, torch.device) else torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        return self

    def hard_freeze(self) -> None:
        """Disable gradients for every parameter and switch to eval mode."""
        self.eval()
        for piece in self.parameters():
            piece.requires_grad_(False)
