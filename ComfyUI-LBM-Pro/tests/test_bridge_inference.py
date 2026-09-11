"""Inference-loop regression tests for BridgeSolver.decode_latents_to_pixels.

Covers the fixes that keep the public inference path correct:

  1. The scheduler uses its training timesteps (not a 1/num_steps
     linspace that puts sigma at noise-floor / model-untrained
     values).
  2. The same input run twice produces identical output (no
     per-step ``randn_like`` jitter).
  3. After ``decode_latents_to_pixels`` runs, the scheduler's
     ``timesteps[0]`` reflects the training start — i.e. the
     cached timestep capture used by ``gather_sigmas`` is the
     fresh value, not a stale leftover from a prior call.
"""
from __future__ import annotations

import pytest
import torch


def _make_solver():
    """Build a minimal BridgeSolver with a fake denoiser.

    The denoiser is just an identity-like stub: it returns zeros
    with the same shape as the sample.  That keeps the Euler
    integration numerically stable without pulling in a real
    UNet.
    """
    from lbm_core.model_factory import _assemble_scheduler
    from lbm_native import BridgeSchedule, BridgeSolver

    scheduler = _assemble_scheduler()
    schedule = BridgeSchedule()

    class _StubDenoiser(torch.nn.Module):
        def forward(self, sample, timestep, guide=None, *args, **kwargs):
            return torch.zeros_like(sample)

    solver = BridgeSolver(
        schedule=schedule,
        denoiser=_StubDenoiser(),
        sampling_noise_scheduler=scheduler,
    )
    return solver, scheduler


def test_decode_uses_training_timesteps():
    """After ``decode_latents_to_pixels`` runs, the scheduler's
    timestep schedule reflects the training-timestep grid the UNet
    was trained on — not the broken linspace override.

    The linspace call (``set_timesteps(sigmas=np.linspace(...))``)
    produced timesteps spanning ``[num_train_timesteps,
    num_train_timesteps / num_steps]`` in even integer steps (e.g.
    ``[1000, 950, 900, ..., 50]`` for ``num_steps=20``).  The
    training-timestep call (``set_timesteps(num_inference_steps=
    ...)``) spans ``[num_train_timesteps, 1]`` and ends at exactly
    ``1.0``.  Asserting on the LAST timestep is the cheapest way
    to tell them apart — both share ``timesteps[0] == 1000``.
    """
    solver, _ = _make_solver()

    z = torch.zeros(1, 4, 8, 8)
    solver.decode_latents_to_pixels(z, num_steps=20)

    timesteps = solver.sampling_noise_scheduler.timesteps
    # Training-timestep schedule ends at 1.0; the linspace override
    # ended at num_train_timesteps / num_steps = 50.
    assert timesteps[-1].item() == pytest.approx(1.0)
    # And the first timestep must be the training start
    # (num_train_timesteps with leading spacing).
    assert timesteps[0].item() == pytest.approx(
        float(solver.sampling_noise_scheduler.config.num_train_timesteps)
    )


def test_decode_is_deterministic():
    """Same input twice must produce identical output.

    The C2 bug added ``torch.randn_like`` to every Euler step, which
    made the inference loop non-deterministic.  Removing it should
    restore reproducibility — running the same call twice must give
    byte-for-byte identical pixels.
    """
    solver, _ = _make_solver()

    torch.manual_seed(0)
    z = torch.randn(1, 4, 8, 8)
    out_a = solver.decode_latents_to_pixels(z.clone(), num_steps=20)

    # Reset RNG state to make sure we're not accidentally inheriting
    # determinism from torch.manual_seed above — the result must not
    # depend on RNG state at all.
    torch.manual_seed(123)
    out_b = solver.decode_latents_to_pixels(z.clone(), num_steps=20)

    assert torch.equal(out_a, out_b)


def test_scheduler_starts_at_training_t():
    """After ``decode_latents_to_pixels`` runs, the scheduler's
    first timestep must be the training start (``num_train_timesteps``
    for the leading-spacing config) — both on the first call and on
    every subsequent call.

    The C9 bug was that ``gather_sigmas``-adjacent code indexed into
    ``scheduler.timesteps[0]`` which, after a prior linspace override,
    could be a stale value.  Caching ``timesteps[0]`` after
    ``set_timesteps`` and re-calling ``set_timesteps`` once per
    inference both ensure the value is always the fresh training
    start.
    """
    solver, _ = _make_solver()

    expected = float(solver.sampling_noise_scheduler.config.num_train_timesteps)

    # First call.
    z = torch.zeros(1, 4, 8, 8)
    solver.decode_latents_to_pixels(z, num_steps=20)
    first_after_first = solver.sampling_noise_scheduler.timesteps[0].item()

    # Second call — must re-set the scheduler so timesteps[0] stays
    # at the training start, not whatever the prior loop left it at.
    solver.decode_latents_to_pixels(z, num_steps=20)
    first_after_second = solver.sampling_noise_scheduler.timesteps[0].item()

    assert first_after_first == pytest.approx(expected)
    assert first_after_second == pytest.approx(expected)
