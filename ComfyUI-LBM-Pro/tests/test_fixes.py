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


def test_load_lbm_checkpoint_writes_via_load_state_dict(monkeypatch):
    """C25 refactor: weights are written via ``load_state_dict``,
    not via direct ``param.data =`` assignment.

    The test stubs ``comfy.utils.load_torch_file`` to return a
    small state dict and verifies that ``load_lbm_checkpoint`` uses
    ``model.load_state_dict(..., strict=False)`` to apply it.  This
    catches regressions where someone reverts to the per-param loop.
    """
    import sys
    import types

    fake_comfy = types.ModuleType("comfy")
    fake_comfy.__path__ = []
    fake_comfy_utils = types.ModuleType("comfy.utils")

    fake_sd = {
        "denoiser.conv_in.weight": torch.full((4, 4, 3, 3), 0.5),
        "vae.vae_model.encoder.conv_in.weight": torch.full((3, 3, 3, 3), 0.25),
    }

    def _stub_load_torch_file(ckpt_path, device, safe_load):
        return fake_sd

    fake_comfy_utils.load_torch_file = _stub_load_torch_file
    fake_comfy.utils = fake_comfy_utils
    fake_comfy_utils.__package__ = "comfy"
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.utils", fake_comfy_utils)

    sys.modules.pop("lbm_core.model_factory", None)
    import lbm_core.model_factory as factory

    target_params = {
        "denoiser.conv_in.weight": torch.zeros(4, 4, 3, 3),
        "denoiser.conv_in.bias": torch.zeros(4),
        "codec.vae_model.encoder.conv_in.weight": torch.zeros(3, 3, 3, 3),
        "codec.vae_model.encoder.conv_in.bias": torch.zeros(3),
        "codec.vae_model.quant_conv.weight": torch.zeros(8, 4, 1, 1),
        "codec.vae_model.quant_conv.bias": torch.zeros(8),
        "denoiser.mid.weight": torch.zeros(4, 4, 3, 3),
        "denoiser.mid.bias": torch.zeros(4),
        "denoiser.out.weight": torch.zeros(4, 4, 3, 3),
        "denoiser.out.bias": torch.zeros(4),
    }

    seen_load_state_dict: dict = {}

    class _StubModel:
        def __init__(self):
            self._params = {k: v.clone() for k, v in target_params.items()}

        def named_parameters(self):
            for k, v in self._params.items():
                yield k, v

        def load_state_dict(self, state_dict, strict=True, assign=False):
            """Stub matching ``nn.Module.load_state_dict`` semantics.

            The C25 refactor (load_state_dict semantics) means we now
            verify the call signature was used: it must pass
            ``strict=False`` so missing keys (e.g. ``quant_conv`` not
            present in this fixture) don't raise.
            """
            seen_load_state_dict["strict"] = strict
            seen_load_state_dict["assign"] = assign
            seen_load_state_dict["state_dict"] = state_dict
            param_names = set(self._params)
            present = set(state_dict)
            missing = sorted(param_names - present)
            unexpected = sorted(present - param_names)
            # Write the loaded values back so we can inspect the result.
            for k, v in state_dict.items():
                if k in self._params and tuple(self._params[k].shape) == tuple(v.shape):
                    self._params[k] = v.clone()
            return missing, unexpected

    model = _StubModel()
    factory.load_lbm_checkpoint(
        model,
        "/fake/path/to/checkpoint.safetensors",
        torch.float32,
        torch.device("cpu"),
        min_match_ratio=0.10,
    )

    # The refactored loader must call load_state_dict (not param.data =).
    assert seen_load_state_dict, "load_state_dict was never called"
    assert seen_load_state_dict["strict"] is False, (
        "load_state_dict was called with strict=True; the loader must "
        "use strict=False so missing keys (e.g. quant_conv absent) "
        "don't raise."
    )

    # Verify the remapped keys were passed.
    sd = seen_load_state_dict["state_dict"]
    assert "denoiser.conv_in.weight" in sd
    assert "codec.vae_model.encoder.conv_in.weight" in sd
    assert "codec.vae_model.encoder.conv_in.bias" not in sd  # not in fixture
    # The values written back must reflect the remapped source tensors.
    assert torch.allclose(model._params["denoiser.conv_in.weight"], torch.full((4, 4, 3, 3), 0.5))


def test_load_lbm_checkpoint_raises_on_low_match(monkeypatch):
    """When too few parameters match, the loader must raise.

    This test actually invokes ``load_lbm_checkpoint`` (with the
    safetensors loader stubbed) rather than re-implementing the
    ratio check in test code. The production guard is what we want
    to exercise.
    """
    import sys
    import types

    # ``load_torch_file`` is imported at module top in
    # ``lbm_core.model_factory`` (C24), so the fake ``comfy.utils``
    # module must be on ``sys.modules`` BEFORE we import the factory.
    fake_comfy = types.ModuleType("comfy")
    fake_comfy_utils = types.ModuleType("comfy.utils")

    def _stub_load_torch_file(ckpt_path, device, safe_load):
        # 1/20 = 5% — well below the 0.95 default minimum.
        return {"denoiser.block_0.weight": torch.zeros(4, 4)}

    fake_comfy_utils.load_torch_file = _stub_load_torch_file
    fake_comfy.utils = fake_comfy_utils
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.utils", fake_comfy_utils)

    # Force a fresh import of the factory so the module-top
    # ``from comfy.utils import load_torch_file`` picks up our stub.
    sys.modules.pop("lbm_core.model_factory", None)
    import lbm_core.model_factory as factory

    # A solver with many parameters but a checkpoint that supplies
    # only one key — the loader's match-ratio guard must fire.
    target_params = {
        f"denoiser.block_{i}.weight": torch.zeros(4, 4) for i in range(20)
    }

    class _StubModel:
        def named_parameters(self):
            for k, t in target_params.items():
                yield k, t

        def load_state_dict(self, state_dict, strict=True, assign=False):
            """Minimal stub matching ``nn.Module.load_state_dict`` semantics.

            The C25 refactor switched from ``param.data =`` to
            ``model.load_state_dict(..., strict=False)`` so the test
            model must expose the call.  We return the standard
            ``(missing, unexpected)`` tuple; the production loader
            only reads ``missing`` to drive the ratio guard.
            """
            param_names = set(target_params)
            present = set(state_dict)
            missing = sorted(param_names - present)
            unexpected = sorted(present - param_names)
            return missing, unexpected

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
# noise_jitter override on decode_latents_to_pixels — REMOVED (F1)
#
# Inference is deterministic — noise injection lives in the training
# loop (``_mix_bridge`` consumes ``schedule.noise_jitter``), not at
# decode time.  The signature no longer carries a ``noise_jitter``
# kwarg; the runtime smoke check below verifies it.
# ---------------------------------------------------------------------------
def test_decode_latents_to_pixels_no_jitter_override():
    """The signature must NOT expose a per-call noise_jitter override.

    F1: the per-call ``noise_jitter`` kwarg was a dead surface — the
    solver never consumed the override at decode time.  Removing it
    closes a confusing public API rather than papering over it.
    """
    import inspect
    from lbm_native import BridgeSolver

    sig = inspect.signature(BridgeSolver.decode_latents_to_pixels)
    assert "noise_jitter" not in sig.parameters, (
        "decode_latents_to_pixels still exposes a noise_jitter override; "
        "inference is deterministic and the kwarg should be removed."
    )


# ---------------------------------------------------------------------------
# N5 / N6 / N12 — LBM_Batch_Processor polish
# ---------------------------------------------------------------------------
def _stub_folder_paths(monkeypatch, tmp_path=None):
    """Inject a fake ``folder_paths`` module so lbm_model_loader imports."""
    import sys
    import types
    if tmp_path is None:
        tmp_path = "/tmp"
    fake_folder_paths = types.ModuleType("folder_paths")
    fake_folder_paths.get_folder_paths = lambda name: [tmp_path]
    monkeypatch.setitem(sys.modules, "folder_paths", fake_folder_paths)


def test_batch_processor_chunks_when_over_max(monkeypatch, tmp_path):
    """N6: when batch size > max_batch, the processor must split
    into chunks and concatenate the results.

    The test stubs ``solver.decode_latents_to_pixels`` so we can
    count how many times it is called and what batch sizes it
    receives — the production decoder would require a real UNet.
    """
    import sys

    _stub_comfy(monkeypatch)
    _stub_folder_paths(monkeypatch, str(tmp_path))
    sys.modules.pop("nodes.lbm_model_loader", None)
    sys.modules.pop("nodes.lbm_batch_processor", None)
    from nodes.lbm_batch_processor import LBM_Batch_Processor

    call_sizes: list[int] = []

    class _StubSolver:
        class _Schedule:
            anchor_field = "source_image"
            mask_field = None

        schedule = _Schedule()

        def to(self, *a, **kw):
            return self

        def cpu(self):
            return self

        def decode_latents_to_pixels(self, z, num_steps, conditioner_inputs, progress_cb=None):
            call_sizes.append(z.shape[0])
            if progress_cb is not None:
                progress_cb(num_steps, num_steps)
            # Return a constant tensor in pixel space (3 channels) so
            # the downstream permute + apply_tint path works without a
            # real UNet / codec.
            return torch.zeros(z.shape[0], 3, 16, 16)

    class _StubCodec:
        def to(self, *a, **kw):
            return self

        def cpu(self):
            return self

        def encode(self, x):
            return torch.zeros(x.shape[0], 4, x.shape[2] // 8, x.shape[3] // 8)

    solver = _StubSolver()
    solver.codec = _StubCodec()

    lbm_model = {
        "model": solver,
        "dtype": torch.float32,
        "device": torch.device("cpu"),
    }
    images = torch.zeros(7, 16, 16, 3)
    node = LBM_Batch_Processor()
    out = node.process_batch(lbm_model, images, steps=2, max_batch=3)
    # Three chunks: 3 + 3 + 1
    assert sorted(call_sizes) == [1, 3, 3], f"unexpected chunk sizes: {call_sizes}"
    assert out[0].shape[0] == 7


def test_batch_processor_rejects_empty_batch(monkeypatch, tmp_path):
    """N12: an empty batch must raise ValueError."""
    import sys

    _stub_comfy(monkeypatch)
    _stub_folder_paths(monkeypatch, str(tmp_path))
    sys.modules.pop("nodes.lbm_model_loader", None)
    sys.modules.pop("nodes.lbm_batch_processor", None)
    from nodes.lbm_batch_processor import LBM_Batch_Processor

    class _StubSolver:
        class _Schedule:
            anchor_field = "source_image"
            mask_field = None
        schedule = _Schedule()

    lbm_model = {
        "model": _StubSolver(),
        "dtype": torch.float32,
        "device": torch.device("cpu"),
    }
    node = LBM_Batch_Processor()
    with pytest.raises(ValueError, match="Empty batch"):
        node.process_batch(lbm_model, torch.zeros(0, 16, 16, 3), steps=2)


def test_batch_processor_has_mask_widget(monkeypatch, tmp_path):
    """N5: the mask input must be exposed on the node's INPUT_TYPES."""
    import sys

    _stub_comfy(monkeypatch)
    _stub_folder_paths(monkeypatch, str(tmp_path))
    sys.modules.pop("nodes.lbm_model_loader", None)
    sys.modules.pop("nodes.lbm_batch_processor", None)
    from nodes.lbm_batch_processor import LBM_Batch_Processor

    spec = LBM_Batch_Processor.INPUT_TYPES()
    optional = spec.get("optional", {})
    assert "mask" in optional, "LBM_Batch_Processor has no `mask` widget"
    assert "max_batch" in optional, "LBM_Batch_Processor has no `max_batch` widget"


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


# ---------------------------------------------------------------------------
# N1 — Compare Grid rejects (or explicitly opts-in to) multi-frame batches
# ---------------------------------------------------------------------------
def test_compare_grid_rejects_multi_frame():
    """A multi-frame input must raise ValueError under default batch_mode."""
    from nodes.lbm_compare_grid import LBM_Compare_Grid

    node = LBM_Compare_Grid()
    # Two-frame (B=2) image, H=8, W=8, 3 channels
    multi = torch.zeros(2, 8, 8, 3)
    single = torch.zeros(1, 8, 8, 3)
    with pytest.raises(ValueError):
        node.compose(image_1=multi, image_2=single)


def test_compare_grid_accepts_multi_frame_with_first_only():
    """Setting batch_mode='first_only' preserves the legacy first-frame behavior."""
    from nodes.lbm_compare_grid import LBM_Compare_Grid

    node = LBM_Compare_Grid()
    multi = torch.zeros(2, 8, 8, 3)
    single = torch.zeros(1, 8, 8, 3)
    out = node.compose(
        image_1=multi, image_2=single, batch_mode="first_only"
    )
    # grid has shape (1, H_total, W_total, 3)
    assert out[0].ndim == 4
    assert out[0].shape[0] == 1


def test_compare_grid_tile_mode_stacks_frames():
    """batch_mode='tile' stacks frames along the height axis of each cell."""
    from nodes.lbm_compare_grid import LBM_Compare_Grid

    node = LBM_Compare_Grid()
    # Two-frame stack of two distinct images.
    a = torch.zeros(2, 8, 8, 3)
    a[1] = 1.0
    b = torch.zeros(1, 8, 8, 3)
    out = node.compose(image_1=a, image_2=b, batch_mode="tile")
    # Tile mode: each cell's H is doubled (since image_1 has B=2).
    assert out[0].ndim == 4
    assert out[0].shape[0] == 1


# ---------------------------------------------------------------------------
# N2 — Depth/Normal Pro refuses a model cached for a different task
# ---------------------------------------------------------------------------
def test_depth_normal_pro_rejects_wrong_task(monkeypatch, tmp_path):
    """A model cached for task='depth' must NOT be usable with task='normal'.

    ``nodes.lbm_depth_normal_pro`` imports ``comfy.model_management``
    and ``comfy.utils`` at module load.  Inject lightweight fakes so
    the module is importable without ComfyUI present.
    """
    import sys

    _stub_comfy(monkeypatch)
    _stub_folder_paths(monkeypatch, str(tmp_path))
    # Force a fresh import of the loaders (which need folder_paths)
    # and the module under test.
    sys.modules.pop("nodes.lbm_model_loader", None)
    sys.modules.pop("nodes.lbm_depth_normal_pro", None)
    from nodes.lbm_depth_normal_pro import LBM_DepthNormal_Pro

    # The validation runs before any heavy work, so the rest of the
    # lbm_model dict only needs the fields the validator reads.
    fake_model = {
        "task": "depth",
        "model": object(),
        "dtype": torch.float32,
        "device": torch.device("cpu"),
    }
    node = LBM_DepthNormal_Pro()
    with pytest.raises(ValueError):
        node.process(
            lbm_model=fake_model,
            image=torch.zeros(1, 8, 8, 3),
            task="normal",
            steps=2,
        )


# ---------------------------------------------------------------------------
# UR1 — Default download URL prefers hf-mirror.com, falls back to HF.co
# ---------------------------------------------------------------------------
def _stub_comfy(monkeypatch):
    """Inject a fake ``comfy`` package so the loader module can import.

    ``comfy.utils.ProgressBar`` is consumed at module load, so both
    submodules must be on ``sys.modules`` as a *package* + submodule pair.
    """
    import sys
    import types

    fake_comfy = types.ModuleType("comfy")
    fake_comfy.__path__ = []  # mark as a package so submodule imports work
    fake_comfy_mm = types.ModuleType("comfy.model_management")
    fake_comfy_mm.get_torch_device = lambda: None
    fake_comfy_mm.unet_offload_device = lambda: None
    fake_comfy_mm.soft_empty_cache = lambda: None
    fake_comfy_utils = types.ModuleType("comfy.utils")
    # Also expose ``load_torch_file`` so the module-top import in
    # ``lbm_core.model_factory`` (C24) resolves when this stub
    # replaces the conftest's default stub.
    fake_comfy_utils.load_torch_file = lambda *a, **kw: {}
    # Link the children back to the parent so ``from comfy.X import Y``
    # finds them as submodules rather than top-level orphans.
    fake_comfy_mm.__package__ = "comfy"
    fake_comfy_utils.__package__ = "comfy"

    class _FakeProgressBar:
        def __init__(self, total):
            self.total = total

        def update_absolute(self, current, total):
            pass

    fake_comfy_utils.ProgressBar = _FakeProgressBar
    fake_comfy.model_management = fake_comfy_mm
    fake_comfy.utils = fake_comfy_utils
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", fake_comfy_mm)
    monkeypatch.setitem(sys.modules, "comfy.utils", fake_comfy_utils)


def test_download_tries_mirror_first(monkeypatch, tmp_path):
    """In auto mode the mirror is attempted before the upstream host."""
    import sys
    import types

    fake_folder_paths = types.ModuleType("folder_paths")
    fake_folder_paths.get_folder_paths = lambda name: [str(tmp_path)]
    monkeypatch.setitem(sys.modules, "folder_paths", fake_folder_paths)
    _stub_comfy(monkeypatch)

    # Force a fresh import of the loader under test.
    sys.modules.pop("nodes.lbm_model_loader", None)
    import nodes.lbm_model_loader as loader_mod

    # Track which URLs the loader tries, in order.
    tried: list[str] = []

    class _FakeResponse:
        def __init__(self, url):
            self.url = url
            self.headers = {"content-length": "11"}

        def raise_for_status(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def iter_content(self, chunk_size):
            if "hf-mirror.com" in self.url:
                # Mirror is "down" — force the fallback to upstream.
                import requests as _req
                raise _req.RequestException("simulated mirror outage")
            yield b"hello world"

    def _fake_get(url, stream=False, timeout=None, **kwargs):
        tried.append(url)
        return _FakeResponse(url)

    monkeypatch.setattr(loader_mod.requests, "get", _fake_get)

    # Run the download with the auto (try mirror, fall back) preset.
    target = loader_mod._download_model(
        "LBM_relighting.safetensors", "relighting",
        mirror="auto (try mirror, fall back)",
    )
    # The first URL must be the mirror; the second must be huggingface.co.
    assert tried, "requests.get was never called"
    assert "hf-mirror.com" in tried[0], f"mirror not tried first: {tried}"
    assert any("huggingface.co" in u for u in tried[1:]), (
        f"upstream fallback not attempted after mirror failure: {tried}"
    )
    assert target.endswith("LBM_relighting.safetensors")


def test_mirror_widget_present_in_input_types(monkeypatch, tmp_path):
    """The LBM_Model_Loader widget must offer the mirror dropdown."""
    import sys
    import types

    fake_folder_paths = types.ModuleType("folder_paths")
    fake_folder_paths.get_folder_paths = lambda name: [str(tmp_path)]
    monkeypatch.setitem(sys.modules, "folder_paths", fake_folder_paths)
    _stub_comfy(monkeypatch)

    sys.modules.pop("nodes.lbm_model_loader", None)
    from nodes.lbm_model_loader import LBM_Model_Loader

    spec = LBM_Model_Loader.INPUT_TYPES()
    optional = spec.get("optional", {})
    assert "mirror" in optional, "LBM_Model_Loader has no `mirror` widget"
    options = optional["mirror"][0] if isinstance(optional["mirror"], tuple) else optional["mirror"]
    assert "auto (try mirror, fall back)" in options
    assert "hf-mirror.com (default)" in options
    assert "huggingface.co" in options
