# ComfyUI-LBM-Pro

Professional LBM (Latent Bridge Matching) image processing pipeline for ComfyUI.

Built on top of the original [ComfyUI-LBM](https://github.com/1038lab/ComfyUI-LBM), this package adds **8 specialized nodes** with model caching, lighting presets, depth/normal visualization, batch processing, and a comparison grid — designed for production image workflows.

## Features

- **Model caching** — load a 1–2 GB checkpoint once, reuse across many invocations (saves 5–10 s per call)
- **8 lighting presets** (`golden_hour`, `overcast`, `studio_left`, `studio_top`, `sunset`, `night_blue`, `cool_neutral`, `warm_neutral`) + custom builder (azimuth/elevation/intensity/temperature)
- **4 depth colormaps** (viridis, inferno, turbo, gray) with optional invert and auto-normalize
- **Compare Grid** — compose 2–9 images into a single comparison grid (`auto`, `horizontal`, `vertical`, `grid_2x2`, `grid_3x3`)
- **Batch Processor** — apply identical parameters to many images for consistency
- **Normal Visualizer** — validate and re-normalize normal maps

## Nodes

| Node | Category | Purpose |
|------|----------|---------|
| `LBM Model Loader` | `🧪AILab/🔆LBM-Pro` | Load + cache a checkpoint as `LBM_MODEL` |
| `LBM Light Preset` | `🧪AILab/🔆LBM-Pro` | Emit a `LIGHT_PRESET` (preset or custom) |
| `LBM Relighting Pro` | `🧪AILab/🔆LBM-Pro` | Enhanced relighting with light tinting |
| `LBM Depth/Normal Pro` | `🧪AILab/🔆LBM-Pro` | Raw + post-processed depth/normal maps |
| `LBM Depth Visualizer` | `🧪AILab/🔆LBM-Pro` | Colorize depth (viridis/inferno/turbo/gray) |
| `LBM Normal Visualizer` | `🧪AILab/🔆LBM-Pro` | Validate unit-length normals |
| `LBM Compare Grid` | `🧪AILab/🔆LBM-Pro` | Stitch 2–9 images into a grid |
| `LBM Batch Processor` | `🧪AILab/🔆LBM-Pro` | Process image batch with shared params |

## Installation

```bash
cd ComfyUI/custom_nodes
git clone <this-repo-url> ComfyUI-LBM-Pro
cd ComfyUI-LBM-Pro
pip install -r requirements.txt
```

Models auto-download to `ComfyUI/models/diffusion_models/LBM/` on first run (Relighting, Depth, Normals).

## Installation Requirements

- **VRAM**: ≥ 8 GB (relighting @ 1024²) — 24 GB recommended for batch processing
- **Python**: ≥ 3.10
- **PyTorch**: ≥ 2.0 (provided by ComfyUI; do NOT install via pip)
- **ComfyUI**: ≥ 2024 (must be installed first; the `comfy.*` imports assume ComfyUI is on `sys.path`)
- **Checkpoint files**: must end up at `ComfyUI/models/diffusion_models/LBM/LBM_relighting.safetensors` (and the depth / normals variants). The Model Loader auto-downloads on first run.

> **Distribution vs import name**: `pip install` registers the package as `comfyui-lbm-pro` (PyPI convention). The importable module is `ComfyUI_LBM_Pro` (PEP 503 normalization). ComfyUI's scanner imports the underscore form.

## Default download mirror (China)

By default, the Model Loader downloads from `hf-mirror.com` (a Hugging Face mirror accessible from mainland China) and falls back to `huggingface.co` if the mirror is unreachable. Change the **mirror** widget to `"huggingface.co"` to skip the mirror entirely.

## Troubleshooting

- **"ComfyUI not installed" / `ModuleNotFoundError: comfy`** — make sure you `cd ComfyUI/custom_nodes/ComfyUI-LBM-Pro` before `pip install -r requirements.txt`. ComfyUI must be importable from the same Python environment.
- **First-run download hangs or fails** — toggle the **mirror** widget on `LBM Model Loader`: try `"huggingface.co"` if the default mirror is unreachable from your network.
- **`RuntimeError: write permission denied` on `models/diffusion_models/LBM/`** — ComfyUI's models directory is owned by another user. Either `chown` it to match, or set `extra_model_paths.yaml` in ComfyUI to point at a writable directory.
- **Wrong task error from `LBM Depth/Normal Pro`** — the cached model is for a different task (`relighting` / `depth` / `normals`). Load a new model with the matching task, or change the `task` parameter on the node.
- **`RuntimeError: LBM checkpoint load failed: only N/M parameters matched`** — the checkpoint file does not match the expected jasperai layout. Re-download or rename per the convention in Installation Requirements.

## Quick Start

1. Restart ComfyUI.
2. Open `example_workflows/01_basic_relighting.json` from the ComfyUI workflow menu.
3. Replace the `LoadImage` "example.png" with your own image.
4. Run the workflow.

The Relighting model (~1.7 GB) downloads automatically the first time.

## Data Flow Examples

### Basic relighting
```
[LoadImage] ──┐
              ├─→ [LBM Relighting Pro] → [SaveImage]
[LBM Model Loader] ──┘  ↑
[LBM Light Preset] ─────┘
```

### Multi-preset comparison
```
[LoadImage] ──┬─→ [Relighting Pro (golden_hour)] ──┐
              ├─→ [Relighting Pro (studio_top)]  ──┤
[LBM Model Loader] ──────────┬─→ ...                ├─→ [Compare Grid] → [SaveImage]
              ├─→ ...                              │
[LBM Light Preset × 4] ─────┴─→ ...                ┘
```

### Depth visualization pipeline
```
[LoadImage] → [LBM Model Loader (depth)]
              ↓
       [LBM Depth/Normal Pro] → [LBM Depth Visualizer (turbo)] → [SaveImage]
                                [LBM Depth Visualizer (inferno)] → [SaveImage]
```

## Architecture

```
ComfyUI-LBM-Pro/
├── lbm_core/         # Pure logic, no ComfyUI deps
│   ├── types.py      # Custom type constants
│   ├── presets.py    # 8 LightPresets + custom builder
│   ├── cache.py      # Thread-safe LRU-style cache
│   ├── model_factory.py  # Build + load LBM models
│   └── visualizers.py    # Depth colormaps + normal helpers
├── nodes/            # ComfyUI node implementations
├── lbm_native/       # Rewritten inference runtime (native)
├── tests/            # Unit + smoke tests
├── example_workflows/   # 6 ready-to-run .json workflows
└── docs/             # Design + plan documents
```

## Custom Types

The package introduces two custom ComfyUI types:

- **`LBM_MODEL`** — dict containing the loaded `LBMModel`, dtype, task, ckpt path, device
- **`LIGHT_PRESET`** — dict containing preset name, RGB tint, intensity, bridge_noise_sigma, description

These flow between the Model Loader / Light Preset nodes and the consuming nodes.

## Performance Tips

- Reuse a single `LBM Model Loader` output across many `LBM Relighting Pro` nodes — the cache avoids repeated loads.
- Use `bf16` unless you specifically need `fp32`.
- 20–30 steps is a good quality/speed tradeoff.

## Limitations

- **Light presets are post-processing tints**, not model conditioning. The LBM model does not consume external lighting parameters; the tinting is a visual approximation applied after inference. Future LBM versions that support conditioning can replace this implementation.
- The cache TTL is 10 minutes; long-running workflows may benefit from explicit `force_reload`.

## License

GPL-3.0

## Credits

- Original LBM: [gojasper/LBM](https://github.com/gojasper/LBM), [Hugging Face](https://huggingface.co/jasperai/LBM_relighting)
- Original ComfyUI node: [1038lab/ComfyUI-LBM](https://github.com/1038lab/ComfyUI-LBM)
- Paper: "LBM: Latent Bridge Matching for Fast Image-to-Image Translation" — Clément Chadebec, Onur Tasar, Sanjeev Sreetharan, Benjamin Aubin

## Documentation

- Design spec: [`docs/2026-09-10-comfyui-lbm-pro-design.md`](docs/2026-09-10-comfyui-lbm-pro-design.md)
- Implementation plan: [`docs/superpowers/plans/2026-09-10-comfyui-lbm-pro.md`](docs/superpowers/plans/2026-09-10-comfyui-lbm-pro.md)
