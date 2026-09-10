# ComfyUI-LBM-Pro User Guide

## Node Reference

### LBM Model Loader

Loads a `.safetensors` checkpoint into memory and caches it. The output is an `LBM_MODEL` that downstream nodes consume.

| Input | Type | Default | Notes |
|-------|------|---------|-------|
| `model_name` | enum | relighting | Files in `models/diffusion_models/LBM/` |
| `task` | enum | relighting | Must match the checkpoint family |
| `precision` | enum | bf16 | fp32 / bf16 / fp16 |
| `bridge_noise_sigma` | float | 0.005 | Range 0.0–0.1 |
| `force_reload` | bool | False | Bypass the cache |

**Output**: `LBM_MODEL`

### LBM Light Preset

Outputs a `LIGHT_PRESET` dict. Either pick from 8 presets or build a custom one.

| Input | Type | Default | Notes |
|-------|------|---------|-------|
| `mode` | enum | preset | `preset` or `custom` |
| `preset_name` | enum | warm_neutral | Only when mode=preset |
| `azimuth_deg` | float | 45 | 0–360 |
| `elevation_deg` | float | 45 | 0–90 (low = high sigma) |
| `intensity` | float | 1.0 | 0–2 |
| `temperature_k` | int | 5500 | 1000–40000 |

**Output**: `LIGHT_PRESET`

### LBM Relighting Pro

Runs the cached relighting model and applies the preset's RGB tint × intensity. Mask input restricts processing to a region.

| Input | Type | Notes |
|-------|------|-------|
| `lbm_model` | LBM_MODEL | from Model Loader |
| `image` | IMAGE | input |
| `steps` | int | 1–100, default 28 |
| `light_preset` | LIGHT_PRESET | optional (defaults to warm_neutral) |
| `mask` | MASK | optional |

**Output**: `IMAGE`

### LBM Depth/Normal Pro

Same architecture as the original node but emits two outputs:
- `raw` — latent in [0, 1]
- `post_processed` — depth inverted (near=bright); normal unchanged

| Input | Type | Notes |
|-------|------|-------|
| `lbm_model` | LBM_MODEL | from Model Loader |
| `image` | IMAGE | input |
| `task` | enum | `depth` / `normal` |
| `steps` | int | 1–100, default 28 |
| `bridge_noise_sigma` | float | 0.0–0.1 |
| `mask` | MASK | optional |

**Outputs**: `IMAGE` (raw), `IMAGE` (post_processed)

### LBM Depth Visualizer

Convert a depth image to a colored visualization.

| Input | Type | Default | Notes |
|-------|------|---------|-------|
| `depth_image` | IMAGE | required | (B, H, W, 1) or (B, H, W, C) — multi-channel is averaged |
| `colormap` | enum | turbo | viridis / inferno / turbo / gray |
| `invert` | bool | False | reverse depth-to-color |
| `auto_normalize` | bool | True | stretch to [0, 1] by min/max |

**Output**: `IMAGE` (RGB float in [0, 1])

### LBM Normal Visualizer

Re-normalize a normal map so every pixel has unit length.

| Input | Type | Default | Notes |
|-------|------|---------|-------|
| `normal_image` | IMAGE | required | (B, H, W, 3) |
| `normalize_range` | bool | True | Rescale [0, 1] → [-1, 1] before normalizing |

**Output**: `IMAGE`

### LBM Compare Grid

Compose 2–9 images into a single grid.

| Input | Type | Default | Notes |
|-------|------|---------|-------|
| `image_1` … `image_9` | IMAGE | first two required | Unused slots are dropped |
| `layout` | enum | auto | `auto`, `horizontal`, `vertical`, `grid_2x2`, `grid_3x3` |
| `padding` | int | 8 | pixels between cells |

**Output**: `IMAGE` (single frame grid)

### LBM Batch Processor

Run a relighting model on a batched `IMAGE` input.

| Input | Type | Notes |
|-------|------|-------|
| `lbm_model` | LBM_MODEL | from Model Loader |
| `images` | IMAGE | batched input |
| `steps` | int | 1–100 |
| `light_preset` | LIGHT_PRESET | optional |

**Output**: `IMAGE` (batched)

## Workflow Examples

See `example_workflows/` for 6 ready-to-run `.json` files:

| File | Description |
|------|-------------|
| `01_basic_relighting.json` | Single image + warm_neutral preset |
| `02_light_presets.json` | Same image × 4 presets → compare grid |
| `03_depth_normal_pipeline.json` | Depth map → turbo + inferno visualizations |
| `04_model_cache_chain.json` | Two Relighting Pro nodes sharing one model cache |
| `05_compare_grid.json` | Same depth × 4 colormaps → 2×2 grid |
| `06_batch_processing.json` | Directory of images → batched relighting |

## Performance Tips

- Models are cached for 10 minutes; reuse the `LBM_Model_Loader` output across nodes.
- Use `bf16` unless you need fp32 precision (saves VRAM).
- Steps 20–30 is a good quality/speed tradeoff.

## Limitations

- Light presets are **post-processing tints**. LBM models do not consume external lighting; the tinting is an approximation.
- Bridge noise sigma is overridden per-call by the preset value.

## License

GPL-3.0
