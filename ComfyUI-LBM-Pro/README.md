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

Eight specialized nodes in category `🧪AILab/🔆LBM-Pro`. They form three logical groups: **Setup** (Loader + Preset, used as inputs to other nodes), **Inference** (Relighting Pro / Depth-Normal Pro / Batch Processor — consume a loaded model and emit images), and **Visualization** (Depth / Normal Visualizer / Compare Grid — post-process the inference outputs).

### `LBM Model Loader` — `LBM_Model_Loader`

**What it does:** Loads an LBM (Latent Bridge Matching) checkpoint from `ComfyUI/models/diffusion_models/LBM/` and emits it as an `LBM_MODEL` dictionary. The model is cached in-process by `(filename, task, precision, mirror)` so subsequent invocations with the same parameters skip the 5–10 s load step. First call auto-downloads the checkpoint (~1.7 GB for relighting, ~1.4 GB for depth/normals) to the standard ComfyUI models directory.

**Inputs:**

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `model_name` | COMBO | `LBM_relighting.safetensors` | All `.safetensors` files in `models/diffusion_models/LBM/` are listed |
| `task` | COMBO | `relighting` | One of `relighting` / `depth` / `normal` — must match the checkpoint |
| `precision` | COMBO | `auto` | `auto` resolves to bf16 on Ampere+, fp16 on Turing, fp32 otherwise |
| `force_reload` | BOOLEAN | `false` | Evicts the cached entry and re-downloads/re-loads |
| `mirror` | COMBO | `auto (try mirror, fall back)` | Download host. See "Default download mirror (China)" below |

**Outputs:** `LBM_MODEL` — a dict containing the loaded solver (`model`), `dtype`, `task`, checkpoint path, and a `resolve_device` callable (re-queries the live device on every call so a GPU hot-swap does not desync).

**How to use:**

1. Add the node. The first run will show a ComfyUI progress bar while it downloads the checkpoint.
2. Wire its `lbm_model` output into any inference node (`LBM Relighting Pro`, `LBM Depth/Normal Pro`, `LBM Batch Processor`).
3. Reuse a single `LBM Model Loader` across many inference nodes — the cache avoids repeated loads. If you change `precision` or `mirror`, a separate cache entry is created.

**Common pitfalls:**

- **Task mismatch.** Loading a `relighting` checkpoint but wiring it into `LBM Depth/Normal Pro` produces a hard error from the consuming node (N2 audit fix). Use the matching task.
- **VRAM.** 8 GB is enough for one 1024² frame; 24 GB is recommended for batch processing.
- **Stale cache after switching GPUs.** Toggle `force_reload` once to drop the entry; the `resolve_device` callable then picks up the new device.

---

### `LBM Light Preset` — `LBM_Light_Preset`

**What it does:** Builds a `LIGHT_PRESET` dictionary that downstream relighting nodes consume. Two modes: pick one of 8 named presets (`golden_hour`, `overcast`, `studio_left`, `studio_top`, `sunset`, `night_blue`, `cool_neutral`, `warm_neutral`) or build a custom preset from azimuth / elevation / intensity / color temperature.

**Inputs:**

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `mode` | COMBO | `preset` | `preset` uses `preset_name`; `custom` uses the 4 numeric parameters |
| `preset_name` | COMBO | `warm_neutral` | All keys of `lbm_core.presets.PRESETS` |
| `azimuth_deg` | FLOAT | 45.0 | Direction of the key light around the subject, 0–360° |
| `elevation_deg` | FLOAT | 45.0 | Height of the key light above the horizon, 0–90° |
| `intensity` | FLOAT | 1.0 | Multiplier on the tint strength, 0.0–2.0 |
| `temperature_k` | INT | 5500 | Color temperature in Kelvin (1000 = candle, 5500 = daylight, 8000 = overcast blue) |

**Outputs:** `LIGHT_PRESET` — dict with `name`, `rgb_tint` (3-tuple), `intensity`, `bridge_noise_sigma`, `description`. Note: presets are **post-processing tints applied to the solver output**; the LBM model does not condition on lighting parameters. See "Limitations" below.

**How to use:**

1. Add the node. Pick a preset (or switch to `custom` and dial your own).
2. Wire its `light_preset` output into the `light_preset` input of `LBM Relighting Pro` or `LBM Batch Processor`.
3. To compare multiple lighting looks in one workflow, add several `LBM Light Preset` nodes (one per look) and feed them into parallel `LBM Relighting Pro` instances — then drop the outputs into `LBM Compare Grid`. The `02_light_presets` example workflow demonstrates this.

---

### `LBM Relighting Pro` — `LBM_Relighting_Pro`

**What it does:** Runs the cached relighting model on an input image, then applies the supplied light preset's tint on top. Returns a single tinted, relit image.

**Inputs:**

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `lbm_model` | `LBM_MODEL` | — | From `LBM Model Loader` (must be a `relighting` model) |
| `image` | `IMAGE` | — | ComfyUI `IMAGE` tensor, shape `(B, H, W, 3)` in `[0, 1]` |
| `steps` | INT | 28 | Euler steps for the bridge solver, 1–100. 20–30 is a good quality/speed tradeoff |
| `light_preset` | `LIGHT_PRESET` | warm_neutral | Optional; defaults to `warm_neutral` if disconnected |
| `mask` | `MASK` | — | Optional inpainting mask. Accepted shapes: `(H, W)`, `(B, H, W)`, or `(B, 1, H, W)`. Always kept in fp32 regardless of model dtype |

**Outputs:** `IMAGE` — relit + tinted image, same shape as input, fp32 in `[0, 1]`.

**How to use:**

1. Add `LBM Model Loader` (set `task=relighting`) and `LoadImage` upstream.
2. Add `LBM Light Preset` (any mode) and connect its `light_preset` to this node.
3. Wire `LBM Model Loader` → `lbm_model`, `LoadImage` → `image`, and the result to `SaveImage` (or into `LBM Compare Grid`).
4. The node calls `solver.cpu()` + `mm.soft_empty_cache()` after each invocation, so VRAM is released between runs.

**Common pitfalls:**

- **Empty batch raises.** A `(0, H, W, 3)` input is a hard error (N12 audit fix), not a silent empty output.
- **Determinism.** Inference is deterministic given the same input image and `steps` — no random seed required.

---

### `LBM Depth/Normal Pro` — `LBM_DepthNormal_Pro`

**What it does:** Runs the cached depth **or** normal model on an input image and emits **two** outputs side-by-side: the raw solver output and a post-processed variant. The post-processing convention is `depth → 1 - raw` (so far objects are bright) and `normal → raw` (already in the expected `[-1, 1]` display range).

**Inputs:**

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `lbm_model` | `LBM_MODEL` | — | From `LBM Model Loader` (must be a `depth` or `normal` model) |
| `image` | `IMAGE` | — | ComfyUI `IMAGE` tensor |
| `task` | COMBO | `depth` | Must match the loaded model's task — otherwise hard error (N2) |
| `steps` | INT | 28 | Euler steps, 1–100 |
| `mask` | `MASK` | — | Optional inpainting mask (same rules as Relighting) |

**Outputs:**

- `raw` (`IMAGE`) — the solver output, scaled to `[0, 1]`. For depth, near=0 / far=1; for normal, encoded RGB in `[-1, 1]` mapped to `[0, 1]`.
- `post_processed` (`IMAGE`) — the display-convention variant. For depth this is `1 - raw` (near=bright, far=dark), the way most depth viewers expect; for normal this equals `raw`.

**How to use:**

1. Add `LBM Model Loader` (set `task=depth` or `task=normal`) and `LoadImage` upstream.
2. Pick the matching `task` on this node.
3. For a quick visualization, wire `raw` into `LBM Depth Visualizer` (or `post_processed` into `LBM Normal Visualizer`).
4. For downstream use (e.g. controlnet depth, normal map as a texture), `post_processed` is the correct output for depth and `raw` (which equals `post_processed`) is the correct output for normal.

**Common pitfalls:**

- **Task mismatch with the model.** Loading a `relighting` model and wiring it here is a hard error: "LBM_DepthNormal_Pro was given a model cached for task='relighting' but the node is configured for task='depth'."
- **`raw` for depth is inverted.** Most display tools expect the `post_processed` output. If you find your depth looks washed out, you forgot the inversion.

---

### `LBM Depth Visualizer` — `LBM_Depth_Visualizer`

**What it does:** Applies a perceptual colormap to a single-channel depth map. Accepts either 1-channel (raw depth) or 3-channel (a depth map that has already been colourised — uses channel 0 only, N18 audit fix) inputs. Always returns a 3-channel `IMAGE` in `[0, 1]`.

**Inputs:**

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `depth_image` | `IMAGE` | — | `(B, H, W, 1)` or `(B, H, W, 3)` |
| `colormap` | COMBO | `turbo` | `viridis` / `inferno` / `turbo` / `gray` |
| `invert` | BOOLEAN | false | Flip near/far after normalization |
| `auto_normalize` | BOOLEAN | true | Stretch the input range to `[0, 1]` per-batch before applying the colormap |

**Outputs:** `IMAGE` — 3-channel colorized depth, `(B, H, W, 3)`, fp32 in `[0, 1]`.

**How to use:**

1. Wire `raw` or `post_processed` from `LBM Depth/Normal Pro` into `depth_image`.
2. Pick a colormap. `turbo` is the most perceptually uniform; `inferno` emphasizes structure; `gray` is the lowest-bias option.
3. To compare 4 colormaps side by side, feed the same depth into 4 visualizer nodes with different `colormap` and pass the outputs into `LBM Compare Grid` — the `05_compare_grid` example workflow shows this.

**Common pitfalls:**

- **3-channel input.** If your upstream is RGB, only the red channel is used (R is channel 0 in `IMAGE` order). To preview a depth map in isolation, you almost always want `invert=false, auto_normalize=true`.
- **Batch size.** All frames in the batch are visualized independently.

---

### `LBM Normal Visualizer` — `LBM_Normal_Visualizer`

**What it does:** Validates and re-normalizes a normal map so every pixel has a unit-length 3-vector, then maps `[-1, 1] → [0, 1]` for display. Use it after `LBM Depth/Normal Pro` (set `task=normal`) when feeding the result into a renderer that expects clean normal maps.

**Inputs:**

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `normal_image` | `IMAGE` | — | `(B, H, W, 3)` — must be 3 channels (hard error otherwise) |
| `input_range` | COMBO | `auto` | `auto` infers from data; `[0,1]` maps to `[-1,1]` first; `[-1,1]` uses as-is |

**Outputs:** `IMAGE` — re-normalized normal map in `[0, 1]`, fp32.

**How to use:**

1. Wire `raw` (which equals `post_processed` for the normal task) from `LBM Depth/Normal Pro` into `normal_image`.
2. Leave `input_range=auto` in the common case — it handles the freshly-generated normal map (`[-1, 1]`) and a pre-displayed `IMAGE` (`[0, 1]`) correctly.
3. Always returns fp32 (N15 audit fix) regardless of the input dtype.

**Common pitfalls:**

- **Wrong channel count.** A 1-channel or 4-channel input raises immediately.
- **`auto` heuristic.** If any sample value is below `-0.5`, the input is treated as already in `[-1, 1]`. If your input has been clamped to `[0, 1]`, choose `[0,1]` explicitly to avoid misclassification.

---

### `LBM Compare Grid` — `LBM_Compare_Grid`

**What it does:** Stitches 2–9 images into a single comparison grid. Layouts are auto (squarest), horizontal, vertical, `grid_2x2` (2–4 images), or `grid_3x3` (3–9 images). Padding is configurable; mismatched input sizes are bilinearly upsampled to the largest frame before composition.

**Inputs:**

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `image_1` / `image_2` | `IMAGE` | — | Required. Both are upsampled to the same canvas size |
| `layout` | COMBO | `auto` | One of `auto` / `horizontal` / `vertical` / `grid_2x2` / `grid_3x3` |
| `image_3` … `image_9` | `IMAGE` | — | Optional. Unconnected slots are skipped |
| `padding` | INT | 8 | Pixel padding between frames and around the canvas, 0–64 |
| `batch_mode` | COMBO | `error` | See below |

**Outputs:** `IMAGE` — `(1, rows*H + (rows+1)*padding, cols*W + (cols+1)*padding, 3)`, fp32.

**`batch_mode` behavior** (when an input has `B > 1`):

| Value | Behavior |
|-------|----------|
| `error` (default) | Hard error — refuses to silently drop frames. Recommended for new workflows |
| `first_only` | Takes only the first frame of each multi-frame input (legacy behavior) |
| `tile` | Stacks frames along the H axis for each multi-frame input — useful for video-frame comparisons |

**How to use:**

1. Wire 2+ image outputs (e.g. from parallel `LBM Relighting Pro` instances) into `image_1`, `image_2`, ...
2. Pick a layout. Use `auto` unless you want a specific shape (e.g. a `horizontal` strip for A/B comparison).
3. Pipe the result into `SaveImage`.

**Common pitfalls:**

- **Default `batch_mode=error`.** If you wire a batched source (e.g. the output of `LBM Batch Processor` with `B > 1`), you must opt in to `first_only` or `tile` — this is intentional (N1 audit fix; previously the node silently dropped B−1 frames per input).
- **2-image `grid_2x2`.** Allowed; the bottom-right cell is empty.

---

### `LBM Batch Processor` — `LBM_Batch_Processor`

**What it does:** Runs the relighting model on a batch of images with **identical** parameters — the same `steps`, the same `light_preset`. This is the right node when you want a consistent visual look across many frames (e.g. an entire directory of photos). The output is the relit + tinted batch, frame-by-frame in the same order as the input.

**Inputs:**

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `lbm_model` | `LBM_MODEL` | — | From `LBM Model Loader` (relighting task) |
| `images` | `IMAGE` | — | Batch tensor `(B, H, W, 3)` in `[0, 1]` |
| `steps` | INT | 28 | Euler steps per frame |
| `light_preset` | `LIGHT_PRESET` | warm_neutral | Optional |
| `mask` | `MASK` | — | Optional inpainting mask |
| `max_batch` | INT | 4 | Max frames per solver invocation. Larger batches are chunked and processed sequentially. Caps VRAM (N6 audit fix) |

**Outputs:** `IMAGE` — relit + tinted batch, same `(B, H, W, 3)` shape, fp32.

**How to use:**

1. Add a directory-loading node (e.g. `LoadImagesFromDirectory`) upstream and connect it to `images`.
2. Add `LBM Model Loader` (relighting) and `LBM Light Preset` and connect both.
3. Set `max_batch` to a value that fits your VRAM: 1–2 on 8 GB cards, 4 on 12–16 GB, 8+ on 24 GB. The node chunks the batch internally and reports progress in the ComfyUI progress bar.
4. Pipe the result to `SaveImage` (writes a numbered series like `batch_relit_00001.png`).

**Common pitfalls:**

- **Empty batch raises.** `B == 0` is a hard error (N12).
- **Mask shape.** The mask's batch dimension must match the images' batch dimension. A `(B, H, W)` mask is broadcast to `(B, 1, H, W)` automatically.
- **`max_batch` is a chunk size, not a count.** Increasing it speeds up the run but uses more VRAM.

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
