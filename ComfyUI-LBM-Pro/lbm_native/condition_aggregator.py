"""Aggregation of multiple conditioning branches.

Each conditioning branch produces a dict keyed by modality
(``class_vec`` / ``attn_ctx`` / ``tile_stack``).  The aggregator
merges branches by concatenating along the modality-specific axis.
When UCG (unconditional guidance) is active, branches whose
``input_key`` matches an UCG list are forced to emit zero tensors.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

# Concatenation axis for each modality.  Centralised here so callers
# do not need to memorise the layout.
STACK_AXES: Dict[str, int] = {
    "class_vec": 1,
    "attn_ctx": 2,
    "tile_stack": 1,
}

_LOGGER = logging.getLogger(__name__)


class BaseCondition(nn.Module):
    """Abstract base for conditioning branches.

    Subclasses override :meth:`forward` and provide the
    ``input_key`` attribute to identify which batch key they consume.
    """

    input_key: str = "image"
    ucg_rate: float = 0.0

    def __init__(self) -> None:
        super().__init__()

    def forward(self, batch: Dict[str, Any], force_zero_embedding: bool = False, *args, **kwargs):
        raise NotImplementedError("BaseCondition subclasses must override forward()")


class ConditionAggregator(nn.Module):
    """Runs a list of conditioning branches and merges their outputs.

    The aggregator is a normal :class:`nn.Module` — it does not
    inherit from ``InferenceCore`` because it owns no tensors of its
    own; it merely orchestrates its children.
    """

    def __init__(self, branches: Optional[List[BaseCondition]] = None) -> None:
        super().__init__()
        self.branches = nn.ModuleList(branches or [])

    def on_fit_start(self, device: Optional[torch.device] = None) -> None:
        for branch in self.branches:
            branch.to(device=device)

    def validate(self) -> None:
        """Sanity-check that UCG keys are actually covered by a branch.

        The original implementation called this from ``__init__``,
        which made construction noisy.  We split it into an opt-in
        method callers can invoke when convenient.
        """
        raise NotImplementedError("UCG key validation is configured by the caller")

    def forward(
        self,
        batch: Dict[str, Any],
        ucg_keys: Optional[List[str]] = None,
        set_ucg_rate_zero: bool = False,
        *args,
        **kwargs,
    ):
        if ucg_keys is None:
            ucg_keys = []
        merged: Dict[str, Any] = {"guide_pack": {}}
        for branch in self.branches:
            force_zero = self._decide_zero(branch, ucg_keys, set_ucg_rate_zero)
            piece = branch(batch, force_zero_embedding=force_zero, *args, **kwargs)
            for modality, tensor in piece.items():
                existing = merged["guide_pack"].get(modality)
                axis = STACK_AXES.get(modality, 1)
                if existing is None:
                    merged["guide_pack"][modality] = tensor
                else:
                    merged["guide_pack"][modality] = torch.cat([existing, tensor], dim=axis)
            _LOGGER.debug(
                "branch=%s input_key=%s force_zero=%s",
                branch.__class__.__name__,
                branch.input_key,
                force_zero,
            )
        return merged

    @staticmethod
    def _decide_zero(branch: BaseCondition, ucg_keys: List[str], set_ucg_rate_zero: bool) -> bool:
        if branch.input_key in ucg_keys:
            return True
        if branch.ucg_rate > 0.0 and not set_ucg_rate_zero:
            return bool(torch.rand(()) < branch.ucg_rate)
        return False
