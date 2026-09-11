"""Smoke tests for the rewritten LBM runtime objects.

Tests cover dataclass validation, helper functions, and
``ConditionAggregator`` composition rules.  No GPU is required.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# BridgeSchedule dataclass
# ---------------------------------------------------------------------------
def test_bridge_schedule_defaults():
    from lbm_native import BridgeSchedule

    s = BridgeSchedule()
    assert s.anchor_field == "source_image"
    assert s.goal_field == "target_image"
    assert s.timestep_policy == "uniform"
    assert s.noise_jitter == pytest.approx(0.001)


def test_bridge_schedule_discrete_validates_length():
    from lbm_native import BridgeSchedule

    with pytest.raises(ValueError):
        BridgeSchedule(
            timestep_policy="discrete",
            discrete_timesteps=[100, 200],
            discrete_weights=[0.5, 0.4, 0.1],
        )


def test_bridge_schedule_discrete_validates_sum():
    from lbm_native import BridgeSchedule

    with pytest.raises(ValueError):
        BridgeSchedule(
            timestep_policy="discrete",
            discrete_timesteps=[100, 200],
            discrete_weights=[0.5, 0.8],
        )


def test_bridge_schedule_log_normal_requires_floats():
    from lbm_native import BridgeSchedule

    with pytest.raises(TypeError):
        BridgeSchedule(timestep_policy="log_normal", logit_mean="bad")


def test_bridge_schedule_slots():
    from lbm_native import BridgeSchedule

    s = BridgeSchedule()
    with pytest.raises(AttributeError):
        s.brand_new_field = 1  # slots-based dataclass should reject


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------
def test_predict_clean_state_identity_at_zero_sigma():
    """At sigma=0 the predicted clean state equals the sample."""
    import torch

    from lbm_native import predict_clean_state

    sample = torch.randn(2, 4, 8, 8)
    output = torch.randn_like(sample)
    sigma = torch.zeros(1, 1, 1, 1)
    predicted = predict_clean_state(sample, output, sigma)
    assert torch.allclose(predicted, sample)


def test_predict_clean_state_subtracts_sigma_scaled_output():
    import torch

    from lbm_native import predict_clean_state

    sample = torch.zeros(1, 4, 8, 8)
    output = torch.full((1, 4, 8, 8), 2.0)
    sigma = torch.full((1, 1, 1, 1), 0.25)
    predicted = predict_clean_state(sample, output, sigma)
    assert torch.allclose(predicted, torch.full((1, 4, 8, 8), -0.5))


def test_predict_clean_state_bridge_identity():
    """Algebra identity: predict_clean_state recovers ``goal`` when the
    bridge sample is the midpoint of ``anchor`` and ``goal`` and the
    model output equals ``anchor - goal``.

    Trace:
        noisy = 0.5 * anchor + 0.5 * goal
        prediction = noisy - 0.5 * (anchor - goal)
                   = 0.5*anchor + 0.5*goal - 0.5*anchor + 0.5*goal
                   = goal
    """
    import torch

    from lbm_native import predict_clean_state

    anchor = torch.randn(1, 4, 8, 8)
    goal = torch.randn(1, 4, 8, 8)
    sigma = torch.full((1, 1, 1, 1), 0.5)
    noisy = sigma * anchor + (1.0 - sigma) * goal
    model_output = anchor - goal

    predicted = predict_clean_state(noisy, model_output, sigma)
    assert torch.allclose(predicted, goal, atol=1e-5)


def test_gather_sigmas_returns_broadcastable_shape():
    import torch

    from lbm_native import gather_sigmas

    # Build a tiny scheduler-like object with a ``sigmas`` tensor and
    # ``timesteps`` covering 1000 entries.  We don't need the real
    # diffusers scheduler for this unit test.
    class _Dummy:
        sigmas = torch.linspace(1.0, 0.001, 1000)
        timesteps = torch.linspace(1000, 1, 1000).long()

    scheduler = _Dummy()
    timesteps = torch.tensor([100, 250, 500])
    out = gather_sigmas(scheduler, timesteps, n_dim=4)
    assert out.shape == (3, 1, 1, 1)


# ---------------------------------------------------------------------------
# ImageConcatConfig + ImageConcatCondition
# ---------------------------------------------------------------------------
def test_image_concat_config_requires_keys():
    from lbm_native import ImageConcatConfig

    with pytest.raises(ValueError):
        ImageConcatConfig()


def test_image_concat_condition_returns_tile_stack_modality():
    import torch
    from lbm_native import ImageConcatCondition, ImageConcatConfig

    cfg = ImageConcatConfig(image_keys=["img"])
    branch = ImageConcatCondition(cfg)

    # Fake codec: returns identity-shaped latents
    class _Codec:
        downsampling_factor = 8

        def encode(self, x):
            return torch.randn(x.shape[0], 4, x.shape[2] // 8, x.shape[3] // 8)

    batch = {"img": torch.randn(2, 3, 64, 64)}
    out = branch(batch, codec=_Codec())
    assert "tile_stack" in out
    assert out["tile_stack"].shape == (2, 4, 8, 8)


def test_image_concat_condition_force_zero_matches_shape():
    import torch
    from lbm_native import ImageConcatCondition, ImageConcatConfig

    cfg = ImageConcatConfig(image_keys=["img"])
    branch = ImageConcatCondition(cfg)

    class _Codec:
        downsampling_factor = 8

        def encode(self, x):
            raise AssertionError("force_zero should bypass encode")

    batch = {"img": torch.randn(2, 3, 64, 64)}
    out = branch(batch, codec=_Codec(), force_zero_embedding=True)
    assert out["tile_stack"].shape == (2, 4, 8, 8)


# ---------------------------------------------------------------------------
# ConditionAggregator composition
# ---------------------------------------------------------------------------
def test_condition_aggregator_concats_along_axis():
    """Two branches producing tile_stack outputs should concatenate on dim=1."""
    import torch
    from lbm_native import BaseCondition, ConditionAggregator, STACK_AXES

    class _TileBranch(BaseCondition):
        def __init__(self, channels: int):
            super().__init__()
            self.channels = channels

        def forward(self, batch, force_zero_embedding=False, **kwargs):
            # All branches share a key called "img" — input_key is unique
            return {"tile_stack": torch.zeros(2, self.channels, 4, 4)}

    agg = ConditionAggregator(branches=[_TileBranch(2), _TileBranch(3)])
    assert STACK_AXES["tile_stack"] == 1

    batch = {"img": torch.zeros(2, 3, 8, 8)}
    out = agg(batch)
    assert "guide_pack" in out
    assert out["guide_pack"]["tile_stack"].shape == (2, 5, 4, 4)


def test_condition_aggregator_force_zero_for_ucg():
    """A branch whose input_key matches the UCG list returns zero."""
    import torch
    from lbm_native import BaseCondition, ConditionAggregator

    class _ImgBranch(BaseCondition):
        input_key = "img"

        def forward(self, batch, force_zero_embedding=False, **kwargs):
            if force_zero_embedding:
                return {"tile_stack": torch.zeros(2, 4, 4, 4)}
            return {"tile_stack": torch.ones(2, 4, 4, 4)}

    agg = ConditionAggregator(branches=[_ImgBranch()])
    batch = {"img": torch.zeros(2, 3, 8, 8)}
    out_with_ucg = agg(batch, ucg_keys=["img"])
    assert torch.all(out_with_ucg["guide_pack"]["tile_stack"] == 0)


# ---------------------------------------------------------------------------
# LatentCodec TilePlan helpers
# ---------------------------------------------------------------------------
def test_slice_grid_single_tile_when_short():
    from lbm_native.latent_codec import _slice_grid

    # Short input fits in one tile — single offset
    assert _slice_grid(total=50, tile=128, overlap=16) == [0]


def test_slice_grid_yields_overlapping_offsets():
    from lbm_native.latent_codec import _slice_grid

    starts = _slice_grid(total=400, tile=128, overlap=16)
    # First and last tile must cover both ends
    assert starts[0] == 0
    assert starts[-1] + 128 >= 400
    # Every consecutive pair — including the final, clamped tile — must
    # advance by at most one stride, otherwise the grid leaves a gap.
    # The last offset is clamped to ``total - tile`` so its step is
    # shorter than a full stride; the bound is therefore an upper one.
    stride = 128 - 16
    for i in range(1, len(starts)):
        assert starts[i] - starts[i - 1] <= stride + 1
