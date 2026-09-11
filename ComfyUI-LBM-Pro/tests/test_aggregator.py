"""Aggregator contract tests.

Covers the tightened ``ConditionAggregator.forward`` signature (no
``**kwargs`` leak into branches), the modality shape/rank validation
performed before ``torch.cat``, and per-instance ``ucg_rate``.
"""
from __future__ import annotations

import inspect

import pytest
import torch

from lbm_native import BaseCondition, ConditionAggregator


class _StrictBranch(BaseCondition):
    """A branch whose ``forward`` accepts no extra kwargs at all."""

    input_key = "img"

    def __init__(self, channels: int = 4, ucg_rate: float = 0.0):
        super().__init__(ucg_rate=ucg_rate)
        self.channels = channels

    def forward(self, batch, force_zero_embedding=False):
        return {"tile_stack": torch.zeros(2, self.channels, 4, 4)}


class _PermissiveBranch(BaseCondition):
    """A branch that would happily swallow any forwarded extras."""

    input_key = "img"

    def forward(self, batch, force_zero_embedding=False, *args, **kwargs):
        return {"tile_stack": torch.zeros(2, 4, 4, 4)}


class _RankBranch(BaseCondition):
    """Emits a ``tile_stack`` with a caller-chosen shape."""

    input_key = "img"

    def __init__(self, shape):
        super().__init__()
        self.shape = tuple(shape)

    def forward(self, batch, force_zero_embedding=False):
        return {"tile_stack": torch.zeros(*self.shape)}


# ---------------------------------------------------------------------------
# C6 — no kwarg leak into branches
# ---------------------------------------------------------------------------
def test_aggregator_does_not_forward_kwargs_to_branches():
    """The aggregator must not splat extras into ``branch(...)``.

    A branch declaring only ``(batch, force_zero_embedding)`` must be
    callable through the aggregator, and the aggregator's own
    signature must expose no ``*args``/``**kwargs`` to forward.
    """
    params = inspect.signature(ConditionAggregator.forward).parameters
    kinds = {p.kind for p in params.values()}
    assert inspect.Parameter.VAR_KEYWORD not in kinds, (
        "ConditionAggregator.forward still accepts **kwargs and would leak them to branches"
    )
    assert inspect.Parameter.VAR_POSITIONAL not in kinds, (
        "ConditionAggregator.forward still accepts *args and would leak them to branches"
    )

    agg = ConditionAggregator(branches=[_StrictBranch(2), _StrictBranch(3)])
    out = agg({"img": torch.zeros(2, 3, 8, 8)})
    assert out["guide_pack"]["tile_stack"].shape == (2, 5, 4, 4)


def test_aggregator_rejects_unexpected_kwarg():
    """``codec=`` (or any other extra) is no longer part of the contract.

    The branch here *would* absorb the extra via its own ``**kwargs``,
    so the ``TypeError`` can only come from the aggregator's own
    tightened signature.
    """
    agg = ConditionAggregator(branches=[_PermissiveBranch()])
    # Sanity: the branch itself tolerates extras, so it is not the raiser.
    assert _PermissiveBranch()({}, codec=object())["tile_stack"].shape == (2, 4, 4, 4)
    with pytest.raises(TypeError):
        agg({"img": torch.zeros(2, 3, 8, 8)}, codec=object())


# ---------------------------------------------------------------------------
# C13 — shape / rank validation before torch.cat
# ---------------------------------------------------------------------------
def test_aggregator_asserts_shape_match():
    """Non-stack axes must agree across branches for the same modality."""
    agg = ConditionAggregator(
        branches=[_RankBranch((2, 4, 8, 8)), _RankBranch((2, 5, 16, 16))]
    )
    with pytest.raises(ValueError, match="tile_stack"):
        agg({"img": torch.zeros(2, 3, 8, 8)})


def test_aggregator_asserts_rank_match():
    """A rank mismatch for the same modality is rejected outright."""
    agg = ConditionAggregator(
        branches=[_RankBranch((2, 4, 8, 8)), _RankBranch((2, 4, 8, 8, 1))]
    )
    with pytest.raises(ValueError, match="rank"):
        agg({"img": torch.zeros(2, 3, 8, 8)})


# ---------------------------------------------------------------------------
# C14 — per-instance ucg_rate
# ---------------------------------------------------------------------------
def test_ucg_rate_per_instance():
    """Two instances must be able to carry independent UCG rates."""
    a = _StrictBranch(ucg_rate=0.3)
    b = _StrictBranch(ucg_rate=0.7)
    assert a.ucg_rate == pytest.approx(0.3)
    assert b.ucg_rate == pytest.approx(0.7)
    # Default stays 0.0 and is not shared state.
    assert _StrictBranch().ucg_rate == pytest.approx(0.0)
    assert a.ucg_rate == pytest.approx(0.3)
