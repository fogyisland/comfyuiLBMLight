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
├── lbm/              # Original LBM model code (verbatim)
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
