# Changelog

All notable changes to ComfyUI-LBM-Pro are documented here.

## Nodes

| Internal class | Display name | Category |
|----------------|--------------|----------|
| `LBM_Model_Loader` | `LBM Model Loader` | `🧪BMLab/🔆LBM-Pro` |
| `LBM_Light_Preset` | `LBM Light Preset` | `🧪BMLab/🔆LBM-Pro` |
| `LBM_Relighting_Pro` | `LBM Relighting Pro` | `🧪BMLab/🔆LBM-Pro` |
| `LBM_DepthNormal_Pro` | `LBM Depth/Normal Pro` | `🧪BMLab/🔆LBM-Pro` |
| `LBM_Depth_Visualizer` | `LBM Depth Visualizer` | `🧪BMLab/🔆LBM-Pro` |
| `LBM_Normal_Visualizer` | `LBM Normal Visualizer` | `🧪BMLab/🔆LBM-Pro` |
| `LBM_Compare_Grid` | `LBM Compare Grid` | `🧪BMLab/🔆LBM-Pro` |
| `LBM_Batch_Processor` | `LBM Batch Processor` | `🧪BMLab/🔆LBM-Pro` |

## v0.1.8 — 2026-09-12

### Fixed
- **`Error(s) in loading state_dict for BridgeSolver` on every inference call.** The diamond MRO of `CondUNet2D` / `PlainUNet2D` (`Subclass → _BaseDiffusersUNet → InferenceCore → diffusers.UNet2D*Model`) made the diffusers parent's `__init__` run **twice**: once explicitly (with the user's architecture kwargs), then a second time via `super().__init__()` from `InferenceCore.__init__` — but with NO arguments, so every diffusers default silently overwrote the caller's settings. Concrete symptoms (every one of which jasperai's checkpoint would catch):
  - `block_out_channels=[320, 640, 1280]` → `(320, 640, 1280, 1280)` (4 levels, default)
  - `cross_attention_dim=[320, 640, 1280]` → `1280` (scalar, default)
  - `transformer_layers_per_block=[1, 2, 10]` → `1` (default)
  - `use_linear_projection=True` → `False` (default) — so `Attention.proj_in` became `Conv2d` instead of `Linear`, and cross-attn `to_k` / `to_v` silently concatenated encoder dims. Net result: every cross-attention weight in the checkpoint had a 2× or 4× shape mismatch.

  Fix:
  1. `InferenceCore.__init__` no longer calls `super().__init__()`. The diffusers parent's own `super().__init__()` chain (which lives inside `UNet2DConditionModel.__init__` and `UNet2DModel.__init__`) has already wired up `nn.Module.__init__` by the time `InferenceCore.__init__` runs, so calling it again is redundant — and dangerous here, because the chain now lands on the diffusers `__init__` with no args.
  2. `InferenceCore.__init__` keeps a guard for the bare-`InferenceCore()` case (no diffusers parent behind it): if `_modules` is missing, it explicitly invokes `nn.Module.__init__` so direct usage still works (used by tests / type stubs).
  3. `CondUNet2D.__init__` and `PlainUNet2D.__init__` continue to call the diffusers parent's `__init__` first (with the architecture kwargs) and `InferenceCore.__init__` second (just to populate `stage_config` and the mixin's bookkeeping) — only the second chain through MRO is suppressed now.

  Added `tests/test_diffusion_unet_init.py` with 8 regression tests that pin each of the architecture values to the jasperai spec and verify `proj_in` is `nn.Linear` (not `Conv2d`), `to_k.weight` has shape `(640, 640)` (no encoder concat), and `block_out_channels` stays at 3 entries. All 8 fail on the pre-fix code and pass after the fix.

  Verified end-to-end via a round-trip checkpoint: build the relighting model, dump its `state_dict` in jasperai's key layout, load it back into a fresh model with `load_lbm_checkpoint`, and confirm `load_state_dict(strict=False)` no longer raises any `size mismatch` errors.

### Changed
- **`FlowMatchEulerDiscreteScheduler` warning cleanup.** `_assemble_scheduler` in `model_factory.py` was passing four SD-style keys (`beta_schedule`, `beta_start`, `beta_end`, `timestep_spacing`) that the FlowMatch scheduler silently ignores — but with a warning on every model load. Removed; the scheduler now only sees the three keys it actually consumes (`num_train_timesteps`, `shift`, `use_dynamic_shifting`). Same scheduler behaviour, no log noise.

## v0.1.7 — 2026-09-12

### Fixed
- **`can't set attribute 'device'` on `LBM_Model_Loader` first inference.** `InferenceCore.__init__` was assigning `self.device = torch.device("cpu")` and `self.dtype = torch.float32`; both are reserved attribute names on `torch.nn.Module`, so the very first `CondUNet2D(...)` call inside `build_lbm_model` raised `AttributeError: can't set attribute 'device'`. Removed both attributes — `nn.Module` already tracks device/dtype via `.to(...)` and exposes them through `next(module.parameters()).device` / `.dtype`. `move_to()` simplified to just delegate to `super().to(...)`. Verified in the ComfyUI venv: `InferenceCore()`, `move_to(...)`, `hard_freeze()` all succeed.

## v0.1.6 — 2026-09-12

### Fixed
- **Wrong model filename in download defaults.** The `LBM_Model_Loader` node defaulted to `LBM_relighting.safetensors` / `LBM_depth.safetensors` / `LBM_normals.safetensors` — those filenames do not exist on the jasperai HF repos, so the first-run download failed with `HTTPError: 404 Client Error: Not Found for url: https://huggingface.co/jasperai/LBM_relighting/resolve/main/LBM_relighting.safetensors`. All three repos actually ship a single file named `model.safetensors`. Fixed across `lbm_nodes/lbm_model_loader.py` (default + `_scan_models()` fallback), all 6 example workflows (`01_basic_relighting.json` through `06_batch_processing.json`), `tests/test_fixes.py`, and `README.md` (node-spec table, Reference Models table, and the "Checkpoint files" install-requirement line — now clarifies that the three repos share a single filename and the `task` widget selects which repo to fetch). Verified via HEAD request against `hf-mirror.com`: all three URLs return HTTP 200 with `content-length ≈ 5.02 GB`.

## v0.1.5 — 2026-09-12

### Changed
- **README expansion.** Added a comprehensive **Reference Models** section covering all three jasperai checkpoints (Relighting / Depth / Normals) with HF repo, default filename, file size, goal field, discrete timesteps / weights / noise_jitter, output shape, intended consumer node, and per-model use-case guidance. Added a **Light Preset Gallery** table inside the `LBM Light Preset` node docs showing all 8 presets' actual rgb_tint / intensity / bridge_noise_sigma values plus a "when to pick which preset" cheat sheet. Replaced the 3-diagram "Data Flow Examples" section with per-workflow descriptions for all 6 example JSONs (each with node count, wiring diagram, run instructions, and edit suggestions). Updated Quick Start to enumerate all 6 workflows.

### Fixed
- **Node category rename** (`🧪AILab/🔆LBM-Pro` → `🧪BMLab/🔆LBM-Pro`) on all 8 nodes, CHANGELOG table, and README. The previous category was incorrect for this fork.
- **Example workflow link integrity** (`example_workflows/02_light_presets.json`, `example_workflows/05_compare_grid.json`). Both files had duplicate link IDs in their top-level `links[]` array — the same source-to-target connection was registered 4 times with the same `link_id`, so ComfyUI's workflow loader could only resolve the first destination and the other 3 branches appeared disconnected in the UI. Each link now has a unique id; each of the 4 parallel branches is a distinct top-level link entry. Verified: all 6 workflows load with no dangling / mismatched / duplicate links.



### Changed
- **README expansion.** Added a comprehensive **Reference Models** section covering all three jasperai checkpoints (Relighting / Depth / Normals) with HF repo, default filename, file size, goal field, discrete timesteps / weights / noise_jitter, output shape, intended consumer node, and per-model use-case guidance. Added a **Light Preset Gallery** table inside the `LBM Light Preset` node docs showing all 8 presets' actual rgb_tint / intensity / bridge_noise_sigma values plus a "when to pick which preset" cheat sheet. Replaced the 3-diagram "Data Flow Examples" section with per-workflow descriptions for all 6 example JSONs (each with node count, wiring diagram, run instructions, and edit suggestions). Updated Quick Start to enumerate all 6 workflows.

## v0.1.4 — 2026-09-12

### Fixed
- **`'nodes' is not a package` — root-cause fix.** ComfyUI ships its own top-level `nodes.py` module; the custom-node loader's `__import__("nodes.lbm_model_loader", ...)` call failed because Python resolved `nodes` to that file (not a package). All 8 nodes failed to register on startup, leaving every example workflow with MISSING nodes. Renamed the package directory `nodes/` → `lbm_nodes/`, updated `_NODE_MODULES` paths, all internal `from nodes.X` imports, the README architecture diagram, and the `lbm_core/visualizers.py` docstring reference. After the fix, all 8 nodes register cleanly and the 6 example workflows load without MISSING nodes.

## v0.1.3 — 2026-09-11

### Fixed
- **Audit-driven hardening pass.** Senior-engineer audit surfaced ~70 defects across `lbm_native/`, `lbm_core/`, `lbm_nodes/`, and packaging; all CRITICAL/HIGH items addressed.
- **Inference correctness** (`lbm_native/bridge_solver.py`): training-timestep schedule; per-step random noise removed; deterministic inference; dtype-symmetric timestep casting.
- **VAE codec** (`lbm_native/latent_codec.py`): SDXL/SD1 normalization hardened with explicit `normalize` parameter; encode/decode dtype-symmetric; tile-padding crash fixed; tile-window blending at edges fixed.
- **Checkpoint loader + cache** (`lbm_core/model_factory.py`, `lbm_core/cache.py`): `quant_conv` / `post_quant_conv` key remap fixed (was silently random-init); loader moved outside lock; stampede-dedup via per-key `Future`; `unload` now releases VRAM.
- **Aggregator** (`lbm_native/condition_aggregator.py`): `codec=` no longer leaks through `**kwargs`; per-branch shape assertions; per-instance `ucg_rate`.
- **Nodes** (`lbm_nodes/lbm_*.py`): Compare Grid rejects multi-frame batches by default (opt-in `batch_mode`); Depth/Normal Pro refuses wrong-task model; Model Loader shows download progress bar; ImportError logging.
- **China mirror** (UR1): default download URL is `hf-mirror.com` with `huggingface.co` fallback; widget in `LBM Model Loader`.
- **Packaging**: `[build-system]` block, full `[project]` metadata, `ComfyUI_LBM_Pro` shipped in wheel, `torch` removed from `requirements.txt`, `Pillow`/`transformers` removed (unused/dead), README rewritten with §Installation Requirements + §Troubleshooting, planning docs marked SUPERSEDED, `_load_module_map` dead code removed.
- **Custom-node loader fix** (`__init__.py`): ComfyUI loads custom nodes via `importlib` with only the package directory on `sys.path`, so the top-level `from lbm_core.types import ...` import was raising `ModuleNotFoundError` on first launch and the package registered zero nodes. The package directory is now inserted into `sys.path` before any internal import so `lbm_core`, `lbm_native`, and `lbm_nodes` resolve correctly.

### Tests
- 90 tests passing (66 prior to the audit + 24 new tests across bridge inference, codec, loader, cache, aggregator, and node UX).

## v0.1.2 — 2026-09-11

### Fixed
- Hardening pass (cache lock, nan guard, progress bar, checkpoint key remap, race conditions).

## v0.1.1 — 2026-09-10

### Changed
- `lbm/` → `lbm_native/` rewrite (full refactor of the inference runtime).

## v0.1.0 — 2026-09-10

### Added
- Initial release: 8 specialized nodes (`LBM Model Loader`, `LBM Light Preset`, `LBM Relighting Pro`, `LBM Depth/Normal Pro`, `LBM Depth Visualizer`, `LBM Normal Visualizer`, `LBM Compare Grid`, `LBM Batch Processor`).
- 3 tasks (`relighting`, `depth`, `normals`) with auto-download from Hugging Face.
- Thread-safe model cache with TTL.
