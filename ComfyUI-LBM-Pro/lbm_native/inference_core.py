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
    """Lightweight base class shared by every inference component.

    Provides ergonomic ``move_to`` / ``hard_freeze`` helpers. Device
    and dtype bookkeeping is intentionally delegated to ``nn.Module``
    itself (``self.to(device, dtype)`` already caches both); this class
    does NOT shadow those attributes with custom ``self.device`` /
    ``self.dtype`` fields, because ``nn.Module.__setattr__`` rejects
    ``device`` as a reserved attribute name.
    """

    def __init__(self, config: Optional[StageConfig] = None) -> None:
        # NOTE: We deliberately do NOT call ``super().__init__()`` here.
        # This class participates in a diamond MRO where subclasses
        # mix in a diffusers ``UNet2D*Model``:
        #
        #     Subclass → _BaseDiffusersUNet → InferenceCore → diffusers.UNet…
        #
        # When ``InferenceCore.__init__`` invokes ``super().__init__()``,
        # Python walks MRO from ``InferenceCore`` to the diffusers parent
        # and runs that parent's ``__init__`` — but with **no arguments**,
        # because we no longer have the ``*args, **kwargs`` that the user
        # passed to the subclass constructor.  The diffusers parent then
        # re-runs ``register_to_config`` with every architecture argument
        # restored to its default value, silently overwriting
        # ``block_out_channels``, ``cross_attention_dim``,
        # ``use_linear_projection``, ``transformer_layers_per_block`` and
        # so on.  This is the diamond-init bug that broke checkpoint
        # loading in v0.1.6 (proj_in silently became ``Conv2d``, cross-
        # attn ``to_k`` / ``to_v`` silently concatenated encoder dims,
        # ``block_out_channels`` silently grew from 3 to 4 entries).
        #
        # Subclasses are expected to invoke their diffusers parent's
        # ``__init__`` explicitly with the architecture kwargs, so by
        # the time we get here ``nn.Module.__init__`` has already been
        # called by the time ``super().__init__()`` walked through
        # ``ModelMixin`` / ``Module`` to ``object.__init__``.  Calling
        # it again is redundant — and dangerous here.
        #
        # The downside: ``InferenceCore()`` instantiated directly (no
        # diffusers parent) leaves ``nn.Module.__init__`` uncalled, so
        # ``self._modules`` / ``self._parameters`` are missing and
        # ``.to(...)`` raises ``AttributeError: … has no attribute
        # '_modules'``.  This is acceptable: ``InferenceCore`` is a
        # mixin, never used directly.  The lone bridge solver path
        # (``BridgeSolver(InferenceCore)``) reaches ``nn.Module.__init__``
        # through ``UNet2DConditionModel.__init__``'s own
        # ``super().__init__()`` because ``BridgeSolver`` mixes in a
        # diffusers model.  We guard the trivial pure-InferenceCore
        # case here so direct use still works (e.g. tests / type
        # stubs), but only when no ``super().__init__`` has already
        # been driven through a diamond parent.
        self.stage_config = config or StageConfig()
        # If we are the topmost ``nn.Module`` in the chain
        # (no diffusers parent behind us), wire up the standard
        # ``Module`` machinery ourselves.  Detect by checking for the
        # presence of ``_modules``: a real ``nn.Module`` always has
        # this attribute after init.
        if not hasattr(self, "_modules"):
            nn.Module.__init__(self)

    def move_to(self, *, device=None, dtype=None, non_blocking: bool = False) -> "InferenceCore":
        """Apply ``.to(...)`` and return ``self``.

        Accepts keyword arguments to mirror the spirit of ``Module.to``
        without inheriting its quirky positional semantics.  Returns
        ``self`` so calls can be chained.

        Device / dtype are read back via ``next(self.parameters()).device``
        and ``next(self.parameters()).dtype`` respectively — these are
        kept current by ``nn.Module`` automatically after ``.to(...)``.
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
        return self

    def hard_freeze(self) -> None:
        """Disable gradients for every parameter and switch to eval mode."""
        self.eval()
        for piece in self.parameters():
            piece.requires_grad_(False)
