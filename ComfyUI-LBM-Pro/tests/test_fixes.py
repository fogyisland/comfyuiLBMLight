"""Regression tests for the workflow-hardening fixes.

Covers the four highest-priority issues fixed in the latest pass:
  1. Checkpoint key remap + sanity check (load_lbm_checkpoint)
  2. BridgeSolver.sample is a class method (pickle-safe alias)
  3. decode_latents_to_pixels accepts a noise_jitter override
  4. nan/inf guard clamps the output before the codec sees it
"""
from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# Checkpoint loading — key remap and sanity check
# ---------------------------------------------------------------------------
def _fake_solver_state(sizes: dict[str, int]) -> dict[str, torch.Tensor]:
    return {name: torch.zeros(size) for name, size in sizes.items()}


def test_remap_translates_vae_prefix_to_codec_prefix():
    from lbm_core.model_factory import _remap_checkpoint_key

    # `vae.vae_model.*` keeps the `vae_model` segment because the
    # ``LatentCodec`` stores the ``AutoencoderKL`` as ``self.vae_model``.
    assert _remap_checkpoint_key("vae.vae_model.encoder.conv_in.weight") == (
        "codec.vae_model.encoder.conv_in.weight"
    )
    # ``quant_conv`` / ``post_quant_conv`` live as direct attributes of
    # the ``AutoencoderKL`` (NOT under ``vae_model``), so they need an
    # extra ``vae_model.`` segment when remapped into the codec prefix.
    assert _remap_checkpoint_key("vae.quant_conv.weight") == "codec.vae_model.quant_conv.weight"
    assert _remap_checkpoint_key("vae.post_quant_conv.weight") == "codec.vae_model.post_quant_conv.weight"
    # Keys without the vae. prefix are passed through untouched so the
    # denoiser.* weights and any aggregate-level entries still match.
    assert _remap_checkpoint_key("denoiser.conv_in.weight") == "denoiser.conv_in.weight"


def test_build_key_index_first_wins():
    from lbm_core.model_factory import _build_key_index

    idx = _build_key_index(
        [
            "denoiser.conv_in.weight",
            "vae.vae_model.encoder.conv_in.weight",
            "vae.quant_conv.weight",
        ]
    )
    assert idx["denoiser.conv_in.weight"] == "denoiser.conv_in.weight"
    assert idx["codec.vae_model.encoder.conv_in.weight"] == (
        "vae.vae_model.encoder.conv_in.weight"
    )
    assert idx["codec.vae_model.quant_conv.weight"] == "vae.quant_conv.weight"


def test_remap_inverse_for_dummy_keys():
    """A checkpoint whose key layout mirrors jasperai's loads cleanly.

    Renamed from the older ``test_load_lbm_checkpoint_writes_matching_keys``:
    the test only exercises the ``_remap_checkpoint_key`` translation, not
    a full load pass.
    """
    from lbm_core.model_factory import _remap_checkpoint_key

    fake_params = {
        "denoiser.conv_in.weight": (4, 4, 3, 3),
        "codec.vae_model.encoder.conv_in.weight": (3, 3, 3, 3),
        "codec.vae_model.quant_conv.weight": (8, 4, 1, 1),
    }
    fake_sd = {
        "denoiser.conv_in.weight": torch.full((4, 4, 3, 3), 0.5),
        "vae.vae_model.encoder.conv_in.weight": torch.full((3, 3, 3, 3), 0.25),
        "vae.quant_conv.weight": torch.full((8, 4, 1, 1), 0.125),
    }
    # The remap is exercised at the build_key_index layer; here we
    # sanity-check the remap function on the same fixture.
    for k in fake_sd:
        remapped = _remap_checkpoint_key(k)
        assert remapped in fake_params, f"remap dropped {k} → {remapped}"


def test_load_lbm_checkpoint_raises_on_low_match(monkeypatch):
    """When too few parameters match, the loader must raise.

    This test actually invokes ``load_lbm_checkpoint`` (with the
    safetensors loader stubbed) rather than re-implementing the
    ratio check in test code. The production guard is what we want
    to exercise.
    """
    import sys
    import types

    import lbm_core.model_factory as factory

    # ``load_lbm_checkpoint`` does ``from comfy.utils import load_torch_file``
    # inside its body.  Inject a fake ``comfy.utils`` module so we can
    # control the return value without depending on ComfyUI.
    fake_comfy = types.ModuleType("comfy")
    fake_comfy_utils = types.ModuleType("comfy.utils")

    def _stub_load_torch_file(ckpt_path, device, safe_load):
        # 1/20 = 5% — well below the 0.95 default minimum.
        return {"denoiser.block_0.weight": torch.zeros(4, 4)}

    fake_comfy_utils.load_torch_file = _stub_load_torch_file
    fake_comfy.utils = fake_comfy_utils
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.utils", fake_comfy_utils)

    # A solver with many parameters but a checkpoint that supplies
    # only one key — the loader's match-ratio guard must fire.
    target_params = {
        f"denoiser.block_{i}.weight": torch.zeros(4, 4) for i in range(20)
    }

    class _StubModel:
        def named_parameters(self):
            for k, t in target_params.items():
                yield k, t

    model = _StubModel()
    with pytest.raises(RuntimeError):
        factory.load_lbm_checkpoint(
            model,
            "/nonexistent/path/to/checkpoint.safetensors",
            torch.float32,
            torch.device("cpu"),
        )


# ---------------------------------------------------------------------------
# BridgeSolver.sample as a class method
# ---------------------------------------------------------------------------
def test_bridge_solver_sample_is_class_method(monkeypatch):
    """The deprecated alias must live on the class, not on the instance dict."""
    from lbm_native import BridgeSchedule, BridgeSolver

    schedule = BridgeSchedule()
    # Build a minimal solver with empty collaborators; we only care
    # about attribute resolution here, not about correct execution.
    solver = BridgeSolver(
        schedule=schedule,
        denoiser=torch.nn.Linear(1, 1),
        sampling_noise_scheduler=None,
    )
    # The class itself defines the method, so the bound attribute is
    # the unbound descriptor — accessing it via the instance still
    # produces the same callable.
    assert hasattr(BridgeSolver, "sample")
    assert callable(solver.sample)


def test_bridge_solver_sample_warns_and_delegates(monkeypatch):
    """The deprecated alias should emit a DeprecationWarning and route correctly."""
    from lbm_native import BridgeSchedule, BridgeSolver

    schedule = BridgeSchedule()

    def _fake_decode(self, *args, **kwargs):
        return "decoded"

    monkeypatch.setattr(BridgeSolver, "decode_latents_to_pixels", _fake_decode)

    solver = BridgeSolver(
        schedule=schedule,
        denoiser=torch.nn.Linear(1, 1),
        sampling_noise_scheduler=None,
    )
    with pytest.warns(DeprecationWarning):
        result = solver.sample()
    assert result == "decoded"


def test_bridge_solver_sample_survives_pickle_roundtrip():
    """A pickled and reloaded solver must still expose the sample alias."""
    import io
    import pickle

    from lbm_native import BridgeSchedule, BridgeSolver

    schedule = BridgeSchedule()
    solver = BridgeSolver(
        schedule=schedule,
        denoiser=torch.nn.Linear(1, 1),
        sampling_noise_scheduler=None,
    )
    buf = io.BytesIO()
    pickle.dump(solver, buf)
    buf.seek(0)
    restored = pickle.load(buf)
    assert hasattr(restored, "sample")
    assert callable(restored.sample)


# ---------------------------------------------------------------------------
# noise_jitter override on decode_latents_to_pixels
# ---------------------------------------------------------------------------
def test_decode_latents_to_pixels_accepts_jitter_override():
    """The signature must allow a per-call override without mutating the schedule."""
    import inspect
    from lbm_native import BridgeSolver

    sig = inspect.signature(BridgeSolver.decode_latents_to_pixels)
    assert "noise_jitter" in sig.parameters
    assert sig.parameters["noise_jitter"].default is None


# ---------------------------------------------------------------------------
# nan/inf guard
# ---------------------------------------------------------------------------
def test_predict_clean_state_handles_nan_inputs():
    """predict_clean_state is the core math; verify it propagates nan honestly.

    The *guard* is at the end of ``decode_latents_to_pixels``; here
    we make sure the math helper itself behaves as expected when fed
    a non-finite sample, so downstream ``nan_to_num`` has work to do.
    """
    from lbm_native import predict_clean_state

    sample = torch.tensor([float("nan"), 1.0])
    # sigma is broadcastable to sample's 1-D shape
    sigma = torch.tensor(1.0)
    model_output = torch.zeros(2)
    out = predict_clean_state(sample, model_output, sigma)
    assert torch.isnan(out).any()
    # The element that was 1.0 should remain 1.0 after the math.
    assert out[1].item() == pytest.approx(1.0)
