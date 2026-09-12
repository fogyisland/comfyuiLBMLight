# ComfyUI-LBM-Pro

Professional LBM (Latent Bridge Matching) image processing pipeline for ComfyUI.

> **中文:** ComfyUI 上的专业 LBM(潜空间桥接匹配)图像处理流水线。

Built on top of the original [ComfyUI-LBM](https://github.com/1038lab/ComfyUI-LBM), this package adds **8 specialized nodes** with model caching, lighting presets, depth/normal visualization, batch processing, and a comparison grid — designed for production image workflows.

> **中文:** 本包基于原始 [ComfyUI-LBM](https://github.com/1038lab/ComfyUI-LBM) 构建,新增 **8 个专用节点**,提供模型缓存、灯光预设、深度/法线可视化、批处理、对比网格等功能,面向生产级图像工作流。

## Features

- **Model caching** — load a 1–2 GB checkpoint once, reuse across many invocations (saves 5–10 s per call)
- **8 lighting presets** (`golden_hour`, `overcast`, `studio_left`, `studio_top`, `sunset`, `night_blue`, `cool_neutral`, `warm_neutral`) + custom builder (azimuth/elevation/intensity/temperature)
- **4 depth colormaps** (viridis, inferno, turbo, gray) with optional invert and auto-normalize
- **Compare Grid** — compose 2–9 images into a single comparison grid (`auto`, `horizontal`, `vertical`, `grid_2x2`, `grid_3x3`)
- **Batch Processor** — apply identical parameters to many images for consistency
- **Normal Visualizer** — validate and re-normalize normal maps

> **中文:**
> - **模型缓存** — 1–2 GB 的 checkpoint 只加载一次,后续调用复用(每次节省 5–10 秒)
> - **8 个灯光预设** (`golden_hour`、`overcast`、`studio_left`、`studio_top`、`sunset`、`night_blue`、`cool_neutral`、`warm_neutral`) + 自定义构建器(方位角/高度角/强度/色温)
> - **4 种深度色图** (viridis、inferno、turbo、gray),支持反转和自动归一化
> - **对比网格** — 将 2–9 张图像拼接成一张对比图(`auto`、`horizontal`、`vertical`、`grid_2x2`、`grid_3x3`)
> - **批处理器** — 对多张图像施加相同参数以保持一致性
> - **法线可视化** — 校验并重新归一化法线贴图

## Nodes

Eight specialized nodes in category `🧪AILab/🔆LBM-Pro`. They form three logical groups: **Setup** (Loader + Preset, used as inputs to other nodes), **Inference** (Relighting Pro / Depth-Normal Pro / Batch Processor — consume a loaded model and emit images), and **Visualization** (Depth / Normal Visualizer / Compare Grid — post-process the inference outputs).

> **中文:** 共有 8 个专用节点,分类位于 `🧪AILab/🔆LBM-Pro`,分为三组:**配置组**(Loader + Preset,作为其它节点的输入)、**推理组**(Relighting Pro / Depth-Normal Pro / Batch Processor — 消费已加载的模型并输出图像)、**可视化组**(Depth / Normal Visualizer / Compare Grid — 对推理结果做后处理)。

### `LBM Model Loader` — `LBM_Model_Loader`

**What it does:** Loads an LBM (Latent Bridge Matching) checkpoint from `ComfyUI/models/diffusion_models/LBM/` and emits it as an `LBM_MODEL` dictionary. The model is cached in-process by `(filename, task, precision, mirror)` so subsequent invocations with the same parameters skip the 5–10 s load step. First call auto-downloads the checkpoint (~1.7 GB for relighting, ~1.4 GB for depth/normals) to the standard ComfyUI models directory.

> **中文:** 从 `ComfyUI/models/diffusion_models/LBM/` 加载 LBM(潜空间桥接匹配)checkpoint,作为 `LBM_MODEL` 字典输出。模型按 `(文件名, 任务, 精度, 镜像源)` 在进程内缓存,后续相同参数调用直接跳过 5–10 秒的加载步骤。首次调用会自动下载 checkpoint(打光约 1.7 GB,深度/法线各约 1.4 GB)到 ComfyUI 的标准模型目录。

**Inputs:**

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `model_name` | COMBO | `LBM_relighting.safetensors` | All `.safetensors` files in `models/diffusion_models/LBM/` are listed |
| `task` | COMBO | `relighting` | One of `relighting` / `depth` / `normal` — must match the checkpoint |
| `precision` | COMBO | `auto` | `auto` resolves to bf16 on Ampere+, fp16 on Turing, fp32 otherwise |
| `force_reload` | BOOLEAN | `false` | Evicts the cached entry and re-downloads/re-loads |
| `mirror` | COMBO | `auto (try mirror, fall back)` | Download host. See "Default download mirror (China)" below |

> **中文 输入表:**

| 名称 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_name` | COMBO | `LBM_relighting.safetensors` | 列出 `models/diffusion_models/LBM/` 中所有 `.safetensors` 文件 |
| `task` | COMBO | `relighting` | 取值 `relighting` / `depth` / `normal` — 必须与 checkpoint 匹配 |
| `precision` | COMBO | `auto` | `auto` 在 Ampere+ 上解析为 bf16,Turing 上为 fp16,其它为 fp32 |
| `force_reload` | BOOLEAN | `false` | 驱逐缓存条目并重新下载/加载 |
| `mirror` | COMBO | `auto (try mirror, fall back)` | 下载主机。详见下文"默认下载镜像(中国)" |

**Outputs:** `LBM_MODEL` — a dict containing the loaded solver (`model`), `dtype`, `task`, checkpoint path, and a `resolve_device` callable (re-queries the live device on every call so a GPU hot-swap does not desync).

> **中文 输出:** `LBM_MODEL` — 字典,包含已加载的 solver(`model`)、`dtype`、`task`、checkpoint 路径,以及一个 `resolve_device` 可调用对象(每次调用重新查询当前设备,避免 GPU 热插拔后设备不同步)。

**How to use:**

1. Add the node. The first run will show a ComfyUI progress bar while it downloads the checkpoint.
2. Wire its `lbm_model` output into any inference node (`LBM Relighting Pro`, `LBM Depth/Normal Pro`, `LBM Batch Processor`).
3. Reuse a single `LBM Model Loader` across many inference nodes — the cache avoids repeated loads. If you change `precision` or `mirror`, a separate cache entry is created.

> **中文 使用步骤:**
> 1. 添加节点。首次运行时会显示 ComfyUI 进度条,等待 checkpoint 下载完成。
> 2. 将 `lbm_model` 输出连接到任意推理节点(`LBM Relighting Pro`、`LBM Depth/Normal Pro`、`LBM Batch Processor`)。
> 3. 让多个推理节点共用一个 `LBM Model Loader` — 缓存会避免重复加载。修改 `precision` 或 `mirror` 会产生独立的缓存条目。

**Common pitfalls:**

- **Task mismatch.** Loading a `relighting` checkpoint but wiring it into `LBM Depth/Normal Pro` produces a hard error from the consuming node (N2 audit fix). Use the matching task.
- **VRAM.** 8 GB is enough for one 1024² frame; 24 GB is recommended for batch processing.
- **Stale cache after switching GPUs.** Toggle `force_reload` once to drop the entry; the `resolve_device` callable then picks up the new device.

> **中文 常见坑:**
> - **任务不匹配。** 加载 `relighting` checkpoint 但连到 `LBM Depth/Normal Pro` 会在消费节点抛硬错误(审计 N2 修复)。请使用对应的任务。
> - **显存。** 单帧 1024² 需要 8 GB;批量处理建议 24 GB。
> - **更换 GPU 后缓存过期。** 勾一次 `force_reload` 丢弃旧条目;`resolve_device` 可调用对象会自动使用新设备。

---

### `LBM Light Preset` — `LBM_Light_Preset`

**What it does:** Builds a `LIGHT_PRESET` dictionary that downstream relighting nodes consume. Two modes: pick one of 8 named presets (`golden_hour`, `overcast`, `studio_left`, `studio_top`, `sunset`, `night_blue`, `cool_neutral`, `warm_neutral`) or build a custom preset from azimuth / elevation / intensity / color temperature.

> **中文:** 构建下游打光节点所需的 `LIGHT_PRESET` 字典。两种模式:从 8 个命名预设中选一个(`golden_hour`、`overcast`、`studio_left`、`studio_top`、`sunset`、`night_blue`、`cool_neutral`、`warm_neutral`),或者用方位角/高度角/强度/色温自定义一个。

**Inputs:**

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `mode` | COMBO | `preset` | `preset` uses `preset_name`; `custom` uses the 4 numeric parameters |
| `preset_name` | COMBO | `warm_neutral` | All keys of `lbm_core.presets.PRESETS` |
| `azimuth_deg` | FLOAT | 45.0 | Direction of the key light around the subject, 0–360° |
| `elevation_deg` | FLOAT | 45.0 | Height of the key light above the horizon, 0–90° |
| `intensity` | FLOAT | 1.0 | Multiplier on the tint strength, 0.0–2.0 |
| `temperature_k` | INT | 5500 | Color temperature in Kelvin (1000 = candle, 5500 = daylight, 8000 = overcast blue) |

> **中文 输入表:**

| 名称 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mode` | COMBO | `preset` | `preset` 使用 `preset_name`;`custom` 使用 4 个数值参数 |
| `preset_name` | COMBO | `warm_neutral` | `lbm_core.presets.PRESETS` 的所有键 |
| `azimuth_deg` | FLOAT | 45.0 | 主光绕主体方向,0–360° |
| `elevation_deg` | FLOAT | 45.0 | 主光在地平线上方的高度,0–90° |
| `intensity` | FLOAT | 1.0 | 色调强度乘数,0.0–2.0 |
| `temperature_k` | INT | 5500 | 色温(单位 K,1000 = 烛光,5500 = 日光,8000 = 阴天蓝光) |

**Outputs:** `LIGHT_PRESET` — dict with `name`, `rgb_tint` (3-tuple), `intensity`, `bridge_noise_sigma`, `description`. Note: presets are **post-processing tints applied to the solver output**; the LBM model does not condition on lighting parameters. See "Limitations" below.

> **中文 输出:** `LIGHT_PRESET` — 字典,包含 `name`、`rgb_tint`(3 元组)、`intensity`、`bridge_noise_sigma`、`description`。**注意:预设是在 solver 输出上叠加的后处理色调**;LBM 模型并不以灯光参数为条件。详见下文"局限性"。

**How to use:**

1. Add the node. Pick a preset (or switch to `custom` and dial your own).
2. Wire its `light_preset` output into the `light_preset` input of `LBM Relighting Pro` or `LBM Batch Processor`.
3. To compare multiple lighting looks in one workflow, add several `LBM Light Preset` nodes (one per look) and feed them into parallel `LBM Relighting Pro` instances — then drop the outputs into `LBM Compare Grid`. The `02_light_presets` example workflow demonstrates this.

> **中文 使用步骤:**
> 1. 添加节点。选一个预设(或切到 `custom` 自行调配)。
> 2. 将 `light_preset` 输出连接到 `LBM Relighting Pro` 或 `LBM Batch Processor` 的 `light_preset` 端口。
> 3. 想在一个工作流里对比多种灯光效果时,加多个 `LBM Light Preset` 节点(每个对应一种风格),分别连到并行的 `LBM Relighting Pro`,再把结果丢进 `LBM Compare Grid`。`02_light_presets` 示例工作流演示了这个套路。

---

### `LBM Relighting Pro` — `LBM_Relighting_Pro`

**What it does:** Runs the cached relighting model on an input image, then applies the supplied light preset's tint on top. Returns a single tinted, relit image.

> **中文:** 对输入图像跑缓存的打光模型,再叠加给定预设的色调。返回一张打光并上色的图像。

**Inputs:**

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `lbm_model` | `LBM_MODEL` | — | From `LBM Model Loader` (must be a `relighting` model) |
| `image` | `IMAGE` | — | ComfyUI `IMAGE` tensor, shape `(B, H, W, 3)` in `[0, 1]` |
| `steps` | INT | 28 | Euler steps for the bridge solver, 1–100. 20–30 is a good quality/speed tradeoff |
| `light_preset` | `LIGHT_PRESET` | warm_neutral | Optional; defaults to `warm_neutral` if disconnected |
| `mask` | `MASK` | — | Optional inpainting mask. Accepted shapes: `(H, W)`, `(B, H, W)`, or `(B, 1, H, W)`. Always kept in fp32 regardless of model dtype |

> **中文 输入表:**

| 名称 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `lbm_model` | `LBM_MODEL` | — | 来自 `LBM Model Loader`(必须是 `relighting` 模型) |
| `image` | `IMAGE` | — | ComfyUI `IMAGE` 张量,形状 `(B, H, W, 3)`,取值 `[0, 1]` |
| `steps` | INT | 28 | 桥接 solver 的 Euler 步数,1–100。20–30 是质量/速度的良好折中 |
| `light_preset` | `LIGHT_PRESET` | warm_neutral | 可选;未连接时默认 `warm_neutral` |
| `mask` | `MASK` | — | 可选重绘遮罩。接受的形状: `(H, W)`、`(B, H, W)`、`(B, 1, H, W)`。始终保持 fp32,与模型 dtype 无关 |

**Outputs:** `IMAGE` — relit + tinted image, same shape as input, fp32 in `[0, 1]`.

> **中文 输出:** `IMAGE` — 打光 + 上色后的图像,形状与输入一致,fp32,取值 `[0, 1]`。

**How to use:**

1. Add `LBM Model Loader` (set `task=relighting`) and `LoadImage` upstream.
2. Add `LBM Light Preset` (any mode) and connect its `light_preset` to this node.
3. Wire `LBM Model Loader` → `lbm_model`, `LoadImage` → `image`, and the result to `SaveImage` (or into `LBM Compare Grid`).
4. The node calls `solver.cpu()` + `mm.soft_empty_cache()` after each invocation, so VRAM is released between runs.

> **中文 使用步骤:**
> 1. 上游添加 `LBM Model Loader`(设 `task=relighting`)和 `LoadImage`。
> 2. 添加 `LBM Light Preset`(任意模式),将其 `light_preset` 连接到本节点。
> 3. 连线: `LBM Model Loader` → `lbm_model`,`LoadImage` → `image`,结果接 `SaveImage`(或丢进 `LBM Compare Grid`)。
> 4. 节点在每次调用后会执行 `solver.cpu()` + `mm.soft_empty_cache()`,运行之间释放显存。

**Common pitfalls:**

- **Empty batch raises.** A `(0, H, W, 3)` input is a hard error (N12 audit fix), not a silent empty output.
- **Determinism.** Inference is deterministic given the same input image and `steps` — no random seed required.

> **中文 常见坑:**
> - **空 batch 会报错。** `(0, H, W, 3)` 的输入是硬错误(审计 N12 修复),不会静默返回空张量。
> - **确定性。** 给定相同输入图像和 `steps` 推理结果完全一致,无需随机种子。

---

### `LBM Depth/Normal Pro` — `LBM_DepthNormal_Pro`

**What it does:** Runs the cached depth **or** normal model on an input image and emits **two** outputs side-by-side: the raw solver output and a post-processed variant. The post-processing convention is `depth → 1 - raw` (so far objects are bright) and `normal → raw` (already in the expected `[-1, 1]` display range).

> **中文:** 用缓存的深度**或**法线模型处理输入图像,同时输出**两个**端口: solver 原始输出 和 后处理变体。后处理规则: `depth → 1 - raw`(近处亮、远处暗);`normal → raw`(已在预期的 `[-1, 1]` 显示范围内)。

**Inputs:**

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `lbm_model` | `LBM_MODEL` | — | From `LBM Model Loader` (must be a `depth` or `normal` model) |
| `image` | `IMAGE` | — | ComfyUI `IMAGE` tensor |
| `task` | COMBO | `depth` | Must match the loaded model's task — otherwise hard error (N2) |
| `steps` | INT | 28 | Euler steps, 1–100 |
| `mask` | `MASK` | — | Optional inpainting mask (same rules as Relighting) |

> **中文 输入表:**

| 名称 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `lbm_model` | `LBM_MODEL` | — | 来自 `LBM Model Loader`(必须是 `depth` 或 `normal` 模型) |
| `image` | `IMAGE` | — | ComfyUI `IMAGE` 张量 |
| `task` | COMBO | `depth` | 必须与加载的模型任务一致,否则硬错误(N2) |
| `steps` | INT | 28 | Euler 步数,1–100 |
| `mask` | `MASK` | — | 可选重绘遮罩(规则同 Relighting) |

**Outputs:**

- `raw` (`IMAGE`) — the solver output, scaled to `[0, 1]`. For depth, near=0 / far=1; for normal, encoded RGB in `[-1, 1]` mapped to `[0, 1]`.
- `post_processed` (`IMAGE`) — the display-convention variant. For depth this is `1 - raw` (near=bright, far=dark), the way most depth viewers expect; for normal this equals `raw`.

> **中文 输出:**
> - `raw`(`IMAGE`) — solver 原始输出,缩放到 `[0, 1]`。深度图近处=0 / 远处=1;法线图编码后的 RGB 原本在 `[-1, 1]`,此处映射为 `[0, 1]`。
> - `post_processed`(`IMAGE`) — 显示约定的变体。深度图等于 `1 - raw`(近亮远暗,这是多数深度查看器的预期);法线图等于 `raw`。

**How to use:**

1. Add `LBM Model Loader` (set `task=depth` or `task=normal`) and `LoadImage` upstream.
2. Pick the matching `task` on this node.
3. For a quick visualization, wire `raw` into `LBM Depth Visualizer` (or `post_processed` into `LBM Normal Visualizer`).
4. For downstream use (e.g. controlnet depth, normal map as a texture), `post_processed` is the correct output for depth and `raw` (which equals `post_processed`) is the correct output for normal.

> **中文 使用步骤:**
> 1. 上游添加 `LBM Model Loader`(设 `task=depth` 或 `task=normal`)和 `LoadImage`。
> 2. 在本节点选对应的 `task`。
> 3. 快速可视化: 把 `raw` 连到 `LBM Depth Visualizer`,或把 `post_processed` 连到 `LBM Normal Visualizer`。
> 4. 下游使用(比如 controlnet 深度、法线作为贴图): 深度用 `post_processed`;法线用 `raw`(此时 `raw` 等于 `post_processed`)。

**Common pitfalls:**

- **Task mismatch with the model.** Loading a `relighting` model and wiring it here is a hard error: "LBM_DepthNormal_Pro was given a model cached for task='relighting' but the node is configured for task='depth'."
- **`raw` for depth is inverted.** Most display tools expect the `post_processed` output. If you find your depth looks washed out, you forgot the inversion.

> **中文 常见坑:**
> - **任务与模型不匹配。** 加载 `relighting` 模型但接这里会硬错误:"LBM_DepthNormal_Pro was given a model cached for task='relighting' but the node is configured for task='depth'。"
> - **`raw` 深度的方向是反的。** 大多数显示工具期望 `post_processed` 输出。如果发现深度看起来发白发淡,说明你漏了反相这一步。

---

### `LBM Depth Visualizer` — `LBM_Depth_Visualizer`

**What it does:** Applies a perceptual colormap to a single-channel depth map. Accepts either 1-channel (raw depth) or 3-channel (a depth map that has already been colourised — uses channel 0 only, N18 audit fix) inputs. Always returns a 3-channel `IMAGE` in `[0, 1]`.

> **中文:** 对单通道深度图应用感知均匀的色图。接受 1 通道(原始深度)或 3 通道(已被上色的深度图,只使用第 0 通道,审计 N18 修复)。始终返回 3 通道 `IMAGE`,取值 `[0, 1]`。

**Inputs:**

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `depth_image` | `IMAGE` | — | `(B, H, W, 1)` or `(B, H, W, 3)` |
| `colormap` | COMBO | `turbo` | `viridis` / `inferno` / `turbo` / `gray` |
| `invert` | BOOLEAN | false | Flip near/far after normalization |
| `auto_normalize` | BOOLEAN | true | Stretch the input range to `[0, 1]` per-batch before applying the colormap |

> **中文 输入表:**

| 名称 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `depth_image` | `IMAGE` | — | `(B, H, W, 1)` 或 `(B, H, W, 3)` |
| `colormap` | COMBO | `turbo` | `viridis` / `inferno` / `turbo` / `gray` |
| `invert` | BOOLEAN | false | 归一化后翻转近/远 |
| `auto_normalize` | BOOLEAN | true | 在应用色图前,把输入范围按 batch 拉伸到 `[0, 1]` |

**Outputs:** `IMAGE` — 3-channel colorized depth, `(B, H, W, 3)`, fp32 in `[0, 1]`.

> **中文 输出:** `IMAGE` — 3 通道上色深度图,`(B, H, W, 3)`,fp32,取值 `[0, 1]`。

**How to use:**

1. Wire `raw` or `post_processed` from `LBM Depth/Normal Pro` into `depth_image`.
2. Pick a colormap. `turbo` is the most perceptually uniform; `inferno` emphasizes structure; `gray` is the lowest-bias option.
3. To compare 4 colormaps side by side, feed the same depth into 4 visualizer nodes with different `colormap` and pass the outputs into `LBM Compare Grid` — the `05_compare_grid` example workflow shows this.

> **中文 使用步骤:**
> 1. 把 `LBM Depth/Normal Pro` 的 `raw` 或 `post_processed` 连到 `depth_image`。
> 2. 选色图。`turbo` 感知最均匀;`inferno` 强调结构;`gray` 偏差最低。
> 3. 想并排对比 4 种色图时, 把同一张深度同时接到 4 个可视化节点(用不同 `colormap`),再把结果丢进 `LBM Compare Grid` — `05_compare_grid` 示例工作流演示了这种用法。

**Common pitfalls:**

- **3-channel input.** If your upstream is RGB, only the red channel is used (R is channel 0 in `IMAGE` order). To preview a depth map in isolation, you almost always want `invert=false, auto_normalize=true`.
- **Batch size.** All frames in the batch are visualized independently.

> **中文 常见坑:**
> - **3 通道输入。** 如果上游是 RGB,只会用红色通道(在 `IMAGE` 顺序中 channel 0)。想独立预览深度图,几乎总是用 `invert=false, auto_normalize=true`。
> - **批大小。** batch 中所有帧独立可视化。

---

### `LBM Normal Visualizer` — `LBM_Normal_Visualizer`

**What it does:** Validates and re-normalizes a normal map so every pixel has a unit-length 3-vector, then maps `[-1, 1] → [0, 1]` for display. Use it after `LBM Depth/Normal Pro` (set `task=normal`) when feeding the result into a renderer that expects clean normal maps.

> **中文:** 校验并重新归一化法线贴图,确保每个像素都是单位长度的 3 维向量,然后把 `[-1, 1]` 映射为 `[0, 1]` 用于显示。在 `LBM Depth/Normal Pro`(设 `task=normal`)之后使用,把结果送入需要干净法线贴图的渲染器。

**Inputs:**

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `normal_image` | `IMAGE` | — | `(B, H, W, 3)` — must be 3 channels (hard error otherwise) |
| `input_range` | COMBO | `auto` | `auto` infers from data; `[0,1]` maps to `[-1,1]` first; `[-1,1]` uses as-is |

> **中文 输入表:**

| 名称 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `normal_image` | `IMAGE` | — | `(B, H, W, 3)` — 必须是 3 通道(否则硬错误) |
| `input_range` | COMBO | `auto` | `auto` 自动推断;`[0,1]` 先映射到 `[-1,1]`;`[-1,1]` 直接使用 |

**Outputs:** `IMAGE` — re-normalized normal map in `[0, 1]`, fp32.

> **中文 输出:** `IMAGE` — 重新归一化的法线图,取值 `[0, 1]`,fp32。

**How to use:**

1. Wire `raw` (which equals `post_processed` for the normal task) from `LBM Depth/Normal Pro` into `normal_image`.
2. Leave `input_range=auto` in the common case — it handles the freshly-generated normal map (`[-1, 1]`) and a pre-displayed `IMAGE` (`[0, 1]`) correctly.
3. Always returns fp32 (N15 audit fix) regardless of the input dtype.

> **中文 使用步骤:**
> 1. 把 `LBM Depth/Normal Pro` 的 `raw`(法线任务下 `raw` 等于 `post_processed`)连到 `normal_image`。
> 2. 通常保持 `input_range=auto` 即可 — 它能正确处理刚生成的法线图(`[-1, 1]`)和已经被显示在 `[0, 1]` 的 `IMAGE`。
> 3. 始终返回 fp32(审计 N15 修复),不受输入 dtype 影响。

**Common pitfalls:**

- **Wrong channel count.** A 1-channel or 4-channel input raises immediately.
- **`auto` heuristic.** If any sample value is below `-0.5`, the input is treated as already in `[-1, 1]`. If your input has been clamped to `[0, 1]`, choose `[0,1]` explicitly to avoid misclassification.

> **中文 常见坑:**
> - **通道数错。** 1 通道或 4 通道输入会立即报错。
> - **`auto` 启发式。** 如果任意采样值低于 `-0.5`,就把输入视为已在 `[-1, 1]`。如果输入已被截断到 `[0, 1]`,请显式选 `[0,1]`,避免误判。

---

### `LBM Compare Grid` — `LBM_Compare_Grid`

**What it does:** Stitches 2–9 images into a single comparison grid. Layouts are auto (squarest), horizontal, vertical, `grid_2x2` (2–4 images), or `grid_3x3` (3–9 images). Padding is configurable; mismatched input sizes are bilinearly upsampled to the largest frame before composition.

> **中文:** 将 2–9 张图像拼成一张对比网格。布局可选 auto(最接近正方形)、horizontal、vertical、`grid_2x2`(2–4 张)、`grid_3x3`(3–9 张)。间距可配;尺寸不一致的输入会双线性上采样到最大帧再拼接。

**Inputs:**

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `image_1` / `image_2` | `IMAGE` | — | Required. Both are upsampled to the same canvas size |
| `layout` | COMBO | `auto` | One of `auto` / `horizontal` / `vertical` / `grid_2x2` / `grid_3x3` |
| `image_3` … `image_9` | `IMAGE` | — | Optional. Unconnected slots are skipped |
| `padding` | INT | 8 | Pixel padding between frames and around the canvas, 0–64 |
| `batch_mode` | COMBO | `error` | See below |

> **中文 输入表:**

| 名称 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `image_1` / `image_2` | `IMAGE` | — | 必填。都会被上采样到同一画布尺寸 |
| `layout` | COMBO | `auto` | 取值 `auto` / `horizontal` / `vertical` / `grid_2x2` / `grid_3x3` |
| `image_3` … `image_9` | `IMAGE` | — | 可选。未连接的槽会被跳过 |
| `padding` | INT | 8 | 帧之间及画布边缘的像素间距,0–64 |
| `batch_mode` | COMBO | `error` | 详见下表 |

**`batch_mode` behavior** (when an input has `B > 1`):

| Value | Behavior |
|-------|----------|
| `error` (default) | Hard error — refuses to silently drop frames. Recommended for new workflows |
| `first_only` | Takes only the first frame of each multi-frame input (legacy behavior) |
| `tile` | Stacks frames along the H axis for each multi-frame input — useful for video-frame comparisons |

> **中文 `batch_mode` 行为**(输入 `B > 1` 时):

| 取值 | 行为 |
|------|------|
| `error`(默认) | 硬错误 — 拒绝静默丢帧。新建工作流推荐此选项 |
| `first_only` | 每个多帧输入只取第一帧(旧行为) |
| `tile` | 把每个多帧输入的帧沿 H 轴堆叠 — 适合视频帧对比 |

**How to use:**

1. Wire 2+ image outputs (e.g. from parallel `LBM Relighting Pro` instances) into `image_1`, `image_2`, ...
2. Pick a layout. Use `auto` unless you want a specific shape (e.g. a `horizontal` strip for A/B comparison).
3. Pipe the result into `SaveImage`.

> **中文 使用步骤:**
> 1. 把 ≥ 2 个图像输出(比如并行的 `LBM Relighting Pro` 实例)接到 `image_1`、`image_2` ...
> 2. 选布局。除非你需要特定形状(比如 A/B 对比的 `horizontal` 长条),否则用 `auto`。
> 3. 把结果接 `SaveImage`。

**Common pitfalls:**

- **Default `batch_mode=error`.** If you wire a batched source (e.g. the output of `LBM Batch Processor` with `B > 1`), you must opt in to `first_only` or `tile` — this is intentional (N1 audit fix; previously the node silently dropped B−1 frames per input).
- **2-image `grid_2x2`.** Allowed; the bottom-right cell is empty.

> **中文 常见坑:**
> - **默认 `batch_mode=error`。** 如果你接的是 batched 数据源(比如 `LBM Batch Processor` 的 `B > 1` 输出),必须显式选 `first_only` 或 `tile` — 这是有意为之(审计 N1 修复;旧版本会静默丢掉每个输入的 B−1 帧)。
> - **`grid_2x2` 放 2 张图。** 允许;右下角为空格。

---

### `LBM Batch Processor` — `LBM_Batch_Processor`

**What it does:** Runs the relighting model on a batch of images with **identical** parameters — the same `steps`, the same `light_preset`. This is the right node when you want a consistent visual look across many frames (e.g. an entire directory of photos). The output is the relit + tinted batch, frame-by-frame in the same order as the input.

> **中文:** 用**完全相同**的参数(同样的 `steps`、同样的 `light_preset`)对一批图像跑打光模型。当你想让很多帧保持一致的视觉效果时(比如整个目录的照片)就用这个节点。输出是按输入顺序逐帧打光 + 上色的 batch。

**Inputs:**

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `lbm_model` | `LBM_MODEL` | — | From `LBM Model Loader` (relighting task) |
| `images` | `IMAGE` | — | Batch tensor `(B, H, W, 3)` in `[0, 1]` |
| `steps` | INT | 28 | Euler steps per frame |
| `light_preset` | `LIGHT_PRESET` | warm_neutral | Optional |
| `mask` | `MASK` | — | Optional inpainting mask |
| `max_batch` | INT | 4 | Max frames per solver invocation. Larger batches are chunked and processed sequentially. Caps VRAM (N6 audit fix) |

> **中文 输入表:**

| 名称 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `lbm_model` | `LBM_MODEL` | — | 来自 `LBM Model Loader`(relighting 任务) |
| `images` | `IMAGE` | — | batch 张量 `(B, H, W, 3)`,取值 `[0, 1]` |
| `steps` | INT | 28 | 每帧的 Euler 步数 |
| `light_preset` | `LIGHT_PRESET` | warm_neutral | 可选 |
| `mask` | `MASK` | — | 可选重绘遮罩 |
| `max_batch` | INT | 4 | 每次 solver 调用的最大帧数。超过则分块顺序处理。控制显存上限(审计 N6 修复) |

**Outputs:** `IMAGE` — relit + tinted batch, same `(B, H, W, 3)` shape, fp32.

> **中文 输出:** `IMAGE` — 打光 + 上色的 batch,形状与输入一致 `(B, H, W, 3)`,fp32。

**How to use:**

1. Add a directory-loading node (e.g. `LoadImagesFromDirectory`) upstream and connect it to `images`.
2. Add `LBM Model Loader` (relighting) and `LBM Light Preset` and connect both.
3. Set `max_batch` to a value that fits your VRAM: 1–2 on 8 GB cards, 4 on 12–16 GB, 8+ on 24 GB. The node chunks the batch internally and reports progress in the ComfyUI progress bar.
4. Pipe the result to `SaveImage` (writes a numbered series like `batch_relit_00001.png`).

> **中文 使用步骤:**
> 1. 上游添加目录加载节点(比如 `LoadImagesFromDirectory`),连到 `images`。
> 2. 添加 `LBM Model Loader`(relighting)和 `LBM Light Preset`,都连进来。
> 3. 根据显存设 `max_batch`: 8 GB 卡 1–2,12–16 GB 卡 4,24 GB 卡 8+。节点内部自动分块,并在 ComfyUI 进度条中报告进度。
> 4. 把结果接 `SaveImage`(会写出带编号的序列,比如 `batch_relit_00001.png`)。

**Common pitfalls:**

- **Empty batch raises.** `B == 0` is a hard error (N12).
- **Mask shape.** The mask's batch dimension must match the images' batch dimension. A `(B, H, W)` mask is broadcast to `(B, 1, H, W)` automatically.
- **`max_batch` is a chunk size, not a count.** Increasing it speeds up the run but uses more VRAM.

> **中文 常见坑:**
> - **空 batch 会报错。** `B == 0` 是硬错误(N12)。
> - **遮罩形状。** 遮罩的 batch 维必须与图像的 batch 维一致。`(B, H, W)` 形状的遮罩会自动广播为 `(B, 1, H, W)`。
> - **`max_batch` 是分块大小,不是批次数。** 调大能加速,但显存占用也会增加。

---

## Installation

```bash
cd ComfyUI/custom_nodes
git clone <this-repo-url> ComfyUI-LBM-Pro
cd ComfyUI-LBM-Pro
pip install -r requirements.txt
```

Models auto-download to `ComfyUI/models/diffusion_models/LBM/` on first run (Relighting, Depth, Normals).

> **中文:** 模型在首次运行时会自动下载到 `ComfyUI/models/diffusion_models/LBM/`(Relighting、Depth、Normals)。

## Installation Requirements

- **VRAM**: ≥ 8 GB (relighting @ 1024²) — 24 GB recommended for batch processing
- **Python**: ≥ 3.10
- **PyTorch**: ≥ 2.0 (provided by ComfyUI; do NOT install via pip)
- **ComfyUI**: ≥ 2024 (must be installed first; the `comfy.*` imports assume ComfyUI is on `sys.path`)
- **Checkpoint files**: must end up at `ComfyUI/models/diffusion_models/LBM/LBM_relighting.safetensors` (and the depth / normals variants). The Model Loader auto-downloads on first run.

> **中文 安装要求:**
> - **显存**: ≥ 8 GB(打光 @ 1024²)— 批量处理建议 24 GB
> - **Python**: ≥ 3.10
> - **PyTorch**: ≥ 2.0(由 ComfyUI 自带,**不要**用 pip 单独安装)
> - **ComfyUI**: ≥ 2024(必须先装好;`comfy.*` 的导入依赖 ComfyUI 在 `sys.path` 中)
> - **Checkpoint 文件**: 必须位于 `ComfyUI/models/diffusion_models/LBM/LBM_relighting.safetensors`(以及 depth / normals 变体)。Model Loader 在首次运行时会自动下载。

> **Distribution vs import name**: `pip install` registers the package as `comfyui-lbm-pro` (PyPI convention). The importable module is `ComfyUI_LBM_Pro` (PEP 503 normalization). ComfyUI's scanner imports the underscore form.

> **中文 分发名 vs 导入名:** `pip install` 注册的包名是 `comfyui-lbm-pro`(PyPI 惯例)。可导入模块名是 `ComfyUI_LBM_Pro`(PEP 503 规范化)。ComfyUI 的扫描器导入下划线形式。

## Default download mirror (China)

By default, the Model Loader downloads from `hf-mirror.com` (a Hugging Face mirror accessible from mainland China) and falls back to `huggingface.co` if the mirror is unreachable. Change the **mirror** widget to `"huggingface.co"` to skip the mirror entirely.

> **中文 默认下载镜像(中国):** 默认情况下,Model Loader 从 `hf-mirror.com`(国内可访问的 Hugging Face 镜像)下载,镜像不可达时回退到 `huggingface.co`。把 **mirror** widget 改成 `"huggingface.co"` 可以完全跳过镜像。

## Troubleshooting

- **"ComfyUI not installed" / `ModuleNotFoundError: comfy`** — make sure you `cd ComfyUI/custom_nodes/ComfyUI-LBM-Pro` before `pip install -r requirements.txt`. ComfyUI must be importable from the same Python environment.
- **First-run download hangs or fails** — toggle the **mirror** widget on `LBM Model Loader`: try `"huggingface.co"` if the default mirror is unreachable from your network.
- **`RuntimeError: write permission denied` on `models/diffusion_models/LBM/`** — ComfyUI's models directory is owned by another user. Either `chown` it to match, or set `extra_model_paths.yaml` in ComfyUI to point at a writable directory.
- **Wrong task error from `LBM Depth/Normal Pro`** — the cached model is for a different task (`relighting` / `depth` / `normals`). Load a new model with the matching task, or change the `task` parameter on the node.
- **`RuntimeError: LBM checkpoint load failed: only N/M parameters matched`** — the checkpoint file does not match the expected jasperai layout. Re-download or rename per the convention in Installation Requirements.

> **中文 故障排查:**
> - **"ComfyUI not installed" / `ModuleNotFoundError: comfy`** — 确认在 `pip install -r requirements.txt` 之前 `cd ComfyUI/custom_nodes/ComfyUI-LBM-Pro`。ComfyUI 必须与本包处于同一 Python 环境。
> - **首次下载卡住或失败** — 切换 `LBM Model Loader` 上的 **mirror** widget: 默认镜像不可达时试 `"huggingface.co"`。
> - **`RuntimeError: write permission denied` 出现在 `models/diffusion_models/LBM/`** — ComfyUI 的模型目录归别的用户所有。`chown` 对齐,或在 ComfyUI 里用 `extra_model_paths.yaml` 指向可写目录。
> - **`LBM Depth/Normal Pro` 报任务错** — 缓存的模型与本节点 `task` 不匹配(`relighting` / `depth` / `normals`)。加载对应任务的模型,或把节点的 `task` 改成一致。
> - **`RuntimeError: LBM checkpoint load failed: only N/M parameters matched`** — checkpoint 文件不符合预期的 jasperai 布局。按"安装要求"一节的约定重新下载或重命名。

## Quick Start

1. Restart ComfyUI.
2. Open `example_workflows/01_basic_relighting.json` from the ComfyUI workflow menu.
3. Replace the `LoadImage` "example.png" with your own image.
4. Run the workflow.

The Relighting model (~1.7 GB) downloads automatically the first time.

> **中文 快速上手:**
> 1. 重启 ComfyUI。
> 2. 从 ComfyUI 工作流菜单打开 `example_workflows/01_basic_relighting.json`。
> 3. 把 `LoadImage` 的 "example.png" 替换成你自己的图像。
> 4. 运行工作流。
>
> Relighting 模型(~1.7 GB)首次运行时会自动下载。

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

> **中文 数据流示例:** 上述三张图分别展示:基础打光、多预设对比、深度可视化流水线。每张图的节点连线一目了然 — `LoadImage` 提供图像,`LBM Model Loader` 提供模型,`LBM Light Preset`(可选)提供灯光风格,推理节点输出图像,`SaveImage` 落盘。批量处理或可视化只需把推理输出分别接到 `LBM Compare Grid` / 多个 `LBM Depth Visualizer` 即可。

## Architecture

```
ComfyUI-LBM-Pro/
├── lbm_core/         # Pure logic, no ComfyUI deps
│   ├── types.py      # Custom type constants
│   ├── presets.py    # 8 LightPresets + custom builder
│   ├── cache.py      # Thread-safe LRU-style cache
│   ├── model_factory.py  # Build + load LBM models
│   └── visualizers.py    # Depth colormaps + normal helpers
├── lbm_nodes/        # ComfyUI node implementations (8 nodes)
├── lbm_native/       # Rewritten inference runtime (native)
├── tests/            # Unit + smoke tests
├── example_workflows/   # 6 ready-to-run .json workflows
└── docs/             # Design + plan documents
```

> **中文 架构:** `lbm_core/` 是纯逻辑(无 ComfyUI 依赖),`lbm_native/` 是重写的推理运行时,`lbm_nodes/` 是 ComfyUI 节点实现(8 个),`tests/` 包含 90 个测试,`example_workflows/` 提供 6 个开箱即用的工作流 JSON,`docs/` 是设计/计划文档。

## Custom Types

The package introduces two custom ComfyUI types:

- **`LBM_MODEL`** — dict containing the loaded `LBMModel`, dtype, task, ckpt path, device
- **`LIGHT_PRESET`** — dict containing preset name, RGB tint, intensity, bridge_noise_sigma, description

These flow between the Model Loader / Light Preset nodes and the consuming nodes.

> **中文 自定义类型:** 本包引入了两个 ComfyUI 自定义类型:
> - **`LBM_MODEL`** — 字典,包含已加载的 `LBMModel`、dtype、task、checkpoint 路径、device
> - **`LIGHT_PRESET`** — 字典,包含预设名、RGB 色调、强度、bridge_noise_sigma、描述
>
> 这两个类型在 Model Loader / Light Preset 节点和消费节点之间流转。

## Performance Tips

- Reuse a single `LBM Model Loader` output across many `LBM Relighting Pro` nodes — the cache avoids repeated loads.
- Use `bf16` unless you specifically need `fp32`.
- 20–30 steps is a good quality/speed tradeoff.

> **中文 性能建议:**
> - 让多个 `LBM Relighting Pro` 共用一个 `LBM Model Loader` 输出 — 缓存避免重复加载。
> - 默认用 `bf16`,除非确实需要 `fp32`。
> - 20–30 步是质量/速度的良好折中。

## Limitations

- **Light presets are post-processing tints**, not model conditioning. The LBM model does not consume external lighting parameters; the tinting is a visual approximation applied after inference. Future LBM versions that support conditioning can replace this implementation.
- The cache TTL is 10 minutes; long-running workflows may benefit from explicit `force_reload`.

> **中文 局限性:**
> - **灯光预设是后处理色调,**不是模型条件。LBM 模型并不消费外部灯光参数;叠加色调是推理之后的视觉近似。未来的 LBM 版本如果支持条件化,可以替换当前实现。
> - 缓存 TTL 是 10 分钟;长时间运行的工作流建议显式 `force_reload`。

## License

GPL-3.0

## Credits

- Original LBM: [gojasper/LBM](https://github.com/gojasper/LBM), [Hugging Face](https://huggingface.co/jasperai/LBM_relighting)
- Original ComfyUI node: [1038lab/ComfyUI-LBM](https://github.com/1038lab/ComfyUI-LBM)
- Paper: "LBM: Latent Bridge Matching for Fast Image-to-Image Translation" — Clément Chadebec, Onur Tasar, Sanjeev Sreetharan, Benjamin Aubin

> **中文 致谢:** 原始 LBM 代码与模型来自 [gojasper/LBM](https://github.com/gojasper/LBM) 和 [Hugging Face](https://huggingface.co/jasperai/LBM_relighting),原始 ComfyUI 节点来自 [1038lab/ComfyUI-LBM](https://github.com/1038lab/ComfyUI-LBM)。论文: "LBM: Latent Bridge Matching for Fast Image-to-Image Translation" — Clément Chadebec, Onur Tasar, Sanjeev Sreetharan, Benjamin Aubin。

## Documentation

- Design spec: [`docs/2026-09-10-comfyui-lbm-pro-design.md`](docs/2026-09-10-comfyui-lbm-pro-design.md)
- Implementation plan: [`docs/superpowers/plans/2026-09-10-comfyui-lbm-pro.md`](docs/superpowers/plans/2026-09-10-comfyui-lbm-pro.md)

> **中文 文档:** 设计规格见 [`docs/2026-09-10-comfyui-lbm-pro-design.md`](docs/2026-09-10-comfyui-lbm-pro-design.md),实施计划见 [`docs/superpowers/plans/2026-09-10-comfyui-lbm-pro.md`](docs/superpowers/plans/2026-09-10-comfyui-lbm-pro.md)。
