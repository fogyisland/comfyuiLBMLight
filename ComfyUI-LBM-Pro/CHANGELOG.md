# Changelog

All notable changes to ComfyUI-LBM-Pro are documented here.

## Nodes

| Internal class | Display name | Category |
|----------------|--------------|----------|
| `LBM_Model_Loader` | `LBM Model Loader` | `🧪AILab/🔆LBM-Pro` |
| `LBM_Light_Preset` | `LBM Light Preset` | `🧪AILab/🔆LBM-Pro` |
| `LBM_Relighting_Pro` | `LBM Relighting Pro` | `🧪AILab/🔆LBM-Pro` |
| `LBM_DepthNormal_Pro` | `LBM Depth/Normal Pro` | `🧪AILab/🔆LBM-Pro` |
| `LBM_Depth_Visualizer` | `LBM Depth Visualizer` | `🧪AILab/🔆LBM-Pro` |
| `LBM_Normal_Visualizer` | `LBM Normal Visualizer` | `🧪AILab/🔆LBM-Pro` |
| `LBM_Compare_Grid` | `LBM Compare Grid` | `🧪AILab/🔆LBM-Pro` |
| `LBM_Batch_Processor` | `LBM Batch Processor` | `🧪AILab/🔆LBM-Pro` |

## v0.1.3 — 2026-09-11

### Fixed
- **Audit-driven hardening pass.** Senior-engineer audit surfaced ~70 defects across `lbm_native/`, `lbm_core/`, `nodes/`, and packaging; all CRITICAL/HIGH items addressed.
- **Inference correctness** (`lbm_native/bridge_solver.py`): training-timestep schedule; per-step random noise removed; deterministic inference; dtype-symmetric timestep casting.
- **VAE codec** (`lbm_native/latent_codec.py`): SDXL/SD1 normalization hardened with explicit `normalize` parameter; encode/decode dtype-symmetric; tile-padding crash fixed; tile-window blending at edges fixed.
- **Checkpoint loader + cache** (`lbm_core/model_factory.py`, `lbm_core/cache.py`): `quant_conv` / `post_quant_conv` key remap fixed (was silently random-init); loader moved outside lock; stampede-dedup via per-key `Future`; `unload` now releases VRAM.
- **Aggregator** (`lbm_native/condition_aggregator.py`): `codec=` no longer leaks through `**kwargs`; per-branch shape assertions; per-instance `ucg_rate`.
- **Nodes** (`nodes/lbm_*.py`): Compare Grid rejects multi-frame batches by default (opt-in `batch_mode`); Depth/Normal Pro refuses wrong-task model; Model Loader shows download progress bar; ImportError logging.
- **China mirror** (UR1): default download URL is `hf-mirror.com` with `huggingface.co` fallback; widget in `LBM Model Loader`.
- **Packaging**: `[build-system]` block, full `[project]` metadata, `ComfyUI_LBM_Pro` shipped in wheel, `torch` removed from `requirements.txt`, `Pillow`/`transformers` removed (unused/dead), README rewritten with §Installation Requirements + §Troubleshooting, planning docs marked SUPERSEDED, `_load_module_map` dead code removed.
- **Custom-node loader fix** (`__init__.py`): ComfyUI loads custom nodes via `importlib` with only the package directory on `sys.path`, so the top-level `from lbm_core.types import ...` import was raising `ModuleNotFoundError` on first launch and the package registered zero nodes. The package directory is now inserted into `sys.path` before any internal import so `lbm_core`, `lbm_native`, and `nodes` resolve correctly.

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
