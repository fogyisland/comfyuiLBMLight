# 独立 LBM 内核规范（Independent LBM Core Spec）

> **目标**：将 `ComfyUI-LBM-Pro/lbm/` 中的 LBM 模型实现完全重写，使 `lbm/` 包与 `ComfyUI-LBM/` 仓库中的同名包**没有任何共享代码、没有任何导入、没有任何一致的命名约定**。重写后的实现独立完成 LBM 推理所需的一切。

**作用域**：`ComfyUI-LBM-Pro/lbm/`（约 1182 行）→ 完全重写为新包 `ComfyUI-LBM-Pro/lbm_native/`。`lbm_core/`（纯逻辑模块）保持不动。

**已批准的风险**：现有 jasperai checkpoint 的 safetensors 权重可能与新代码不兼容（键名、字段元数据、UNet 配置可能不一致）。这是用户已接受的代价。

**约束**：

1. `nodes/` 和 `lbm_core/` 不允许出现 `import lbm` / `from lbm` —— 所有导入改为 `from lbm_native ...`。
2. `lbm_native/` 内部不允许出现与原版**完全一致**的：类名、文件名、函数名、字段名、docstring 段落、注释句、错误消息。允许接口契约一致（输入/输出 shape、签名）。
3. 运行时外部行为保持兼容：节点的输入参数、输出 shape、UI 字符串、缓存键格式均不变。
4. 不引入新依赖：仍依赖 `torch` 和 `diffusers`（diffusers 提供 `UNet2DConditionModel`、`AutoencoderKL`、`FlowMatchEulerDiscreteScheduler`）。
5. 不允许使用 `pydantic`（原 `lbm_config.py` 用 `pydantic.dataclasses.dataclass`），改用 stdlib `dataclasses`。

---

## 1. 现状摘要（仅用于参照接口契约）

重写前的 LBM 内核由 7 个模块组成：

| 原模块 | 行数 | 角色 |
|--------|------|------|
| `lbm/models/base/base_model.py` | 66 | 抽象基类，封装 device/ddtype 管理 |
| `lbm/models/base/model_config.py` | 7 | 配置基类（仅含 `input_key`） |
| `lbm/models/unets/unet.py` | 148 | UNet2D/UNet2DCond 的 diffusers 包装，前向接受 `conditioning` dict |
| `lbm/models/vae/autoencoderKL.py` | 127 | SD1 VAE 包装，提供 `encode/decode/downsampling_factor/latent_channels` |
| `lbm/models/embedders/conditioners_wrapper.py` | 115 | 多 conditioner 容器，合并 `cond` dict |
| `lbm/models/embedders/latents_concat/...` | 80 | 把 mask/image 编码到 latent 后 concat |
| `lbm/models/lbm/lbm_config.py` | 101 | LBM 训练配置（sampling 策略、loss 等） |
| `lbm/models/lbm/lbm_model.py` | 511 | LBMModel：bridge-matching 训练/采样循环 |

调用方契约（来自 `lbm_core/model_factory.py`）：

- `_build_unet(dtype) → DiffusersUNet2DCondWrapper`（硬编码 cfg，列在 model_factory.py:46-67）
- `_build_vae(dtype) → AutoencoderKLDiffusers`
- `_build_scheduler() → FlowMatchEulerDiscreteScheduler`
- `LBMModel(LBMConfig(...), denoiser, sampling_noise_scheduler, vae, conditioner)`
- `LBMModel.sample(z, num_steps, conditioner_inputs, max_samples=None, verbose=False) → Tensor`（B,3,H,W 像素范围 [-1,1]）
- `model.bridge_noise_sigma` 可被调用方覆盖为 float

---

## 2. 重写方案（架构与命名）

新包 `ComfyUI-LBM-Pro/lbm_native/`，文件命名刻意与原版不同，避免一字不差的复制：

| 新模块 | 原模块替代 | 大致行数 |
|--------|-----------|---------|
| `lbm_native/inference_core.py` | `BaseModel` + `ModelConfig` | ~70 |
| `lbm_native/diffusion_unet.py` | `DiffusersUNet2DCondWrapper`（非 cond 版本） | ~190 |
| `lbm_native/latent_codec.py` | `AutoencoderKLDiffusers` | ~150 |
| `lbm_native/condition_aggregator.py` | `ConditionerWrapper` | ~120 |
| `lbm_native/image_concat_condition.py` | `LatentsConcatEmbedder` | ~90 |
| `lbm_native/timestep_policy.py` | `LBMConfig`（仅 inference 相关字段） | ~120 |
| `lbm_native/bridge_solver.py` | `LBMModel`（重写 `sample` + `forward`，训练字段保留但重写） | ~520 |

总计约 1260 行（与原 ~1182 接近，但实现细节完全不同）。

### 2.1 命名规则（避免与原版雷同）

| 原版 | 新版 |
|------|------|
| `BaseModel` | `InferenceCore` |
| `ModelConfig` | `StageConfig` |
| `DiffusersUNet2DCondWrapper` | `CondUNet2D` |
| `DiffusersUNet2DWrapper` | `PlainUNet2D`（仍提供，但只在新 CondUNet2D 不可用时使用） |
| `AutoencoderKLDiffusers` | `LatentCodec` |
| `ConditionerWrapper` | `ConditionAggregator` |
| `LatentsConcatEmbedder` | `ImageConcatCondition` |
| `BaseConditioner` | `BaseCondition` |
| `LBMConfig` | `BridgeSchedule` |
| `LBMModel` | `BridgeSolver` |
| `sample` | `decode_latents_to_pixels` |
| `forward` | `training_step` |
| `bridge_noise_sigma` | `noise_jitter` |
| `source_key` | `anchor_field` |
| `target_key` | `goal_field` |
| `conditioning` | `guide` |
| `cond` (wrapper dict) | `guide_pack` |
| `vector` / `crossattn` / `concat` | `class_vec` / `attn_ctx` / `tile_stack` |
| `timestep_sampling` | `timestep_policy` |
| `selected_timesteps` | `discrete_timesteps` |
| `prob` | `discrete_weights` |
| `predict_x_0` | `predict_clean_state` |
| `latent_loss` / `pixel_loss` | `latent_recon_err` / `pixel_recon_err` |

### 2.2 调用方更新

`lbm_core/model_factory.py` 改为从 `lbm_native` 导入，并使用新类名：

```python
from lbm_native.bridge_solver import BridgeSolver, BridgeSchedule
from lbm_native.diffusion_unet import CondUNet2D
from lbm_native.latent_codec import LatentCodec
from lbm_native.condition_aggregator import ConditionAggregator
from lbm_native.image_concat_condition import ImageConcatCondition
```

`lbm_core/model_factory.py` 的工厂函数签名 `build_lbm_model(task, dtype, bridge_noise_sigma)` 和 `load_lbm_checkpoint(...)` **保持不变**，内部的实现改用新类。`model_factory.bridge_noise_sigma` 参数改写到 `BridgeSchedule.noise_jitter`。

节点层（`nodes/lbm_relighting_pro.py` 等）调用 `model["model"].sample(z, ...)` → 改为 `model["model"].decode_latents_to_pixels(z, ...)`。但因为这些调用都封装在 `model_factory` 之后，目前只有 `model["model"]` 是 `LBMModel` 实例被传入。为最小化节点改动，新 `BridgeSolver` 必须暴露 `sample` 同名别名方法（带 `DeprecationWarning` 提示），供 `lbm_relighting_pro.py` 调用。

不，最干净的做法：**改节点层调用为 `decode_latents_to_pixels`**。同时检查所有 `bridge_noise_sigma` 引用，改为 `noise_jitter`。

### 2.3 关键接口契约（重写不变）

```
class BridgeSolver(nn.Module):
    schedule: BridgeSchedule
    denoiser: CondUNet2D | PlainUNet2D
    codec: LatentCodec | None
    aggregator: ConditionAggregator | None
    noise_jitter: float  # 初始化后只读
    training_iter: nn.Parameter

    def decode_latents_to_pixels(self, z, num_steps, conditioner_inputs=None,
                                  max_samples=None, progress_cb=None) -> Tensor[B,3,H,W]

    def training_step(self, batch, **kwargs) -> dict[str, Tensor]

    @property
    def dtype(self) -> torch.dtype
    @property
    def device(self) -> torch.device
```

`LatentCodec.encode(x) → Tensor[B,4,H/8,W/8]`、`decode(z) → Tensor[B,3,H,W]`。  
`CondUNet2D.forward(sample, timestep, guide=None) → Tensor`：`guide["guide_pack"]` 含 `class_vec` / `attn_ctx` / `tile_stack` 键。

---

## 3. 与原版的具体差异（保证"无关"）

| 方面 | 原版 | 新版 |
|------|------|------|
| 字典合并维度 | `KEY2CATDIM` 全局常量 | `STACK_AXES` 模块内常量，**键名不同** |
| `BaseModel.to()` 解析 | 用私有 API `torch._C._nn._parse_to` | 用 `torch._C._nn._parse_to` 替代包装，但**逻辑重写**：分三段拆解 device/dtype/non_blocking/convert_to_format |
| VAE tiling | `Tiler` 类工具 + 内部 pad | **重写为简单网格分块**（不用 `Tiler`），每瓦片用 bilinear 重叠线性混合 blend |
| `forward` 函数体行数 | 80+ | 拆为多个私有方法（`_assemble_anchor`、`_draw_interpolant`、`_invoke_denoiser`） |
| timestep 选择 | `torch.randint` + 索引 | 用 `torch.distributions.Categorical(self.discrete_weights).sample()` |
| `BridgeSchedule` | `pydantic.dataclasses.dataclass` | `dataclasses.dataclass(slots=True)`，重写 `__post_init__` |
| `BridgeSolver.sample` 进度条 | `comfy.utils.ProgressBar` | 接受 `progress_cb: Callable[[int, int], None]`，由调用方注入（节点注入 `comfy.utils.ProgressBar` 包装） |
| `_predicted_x_0` | 方法 | **改为函数** `predict_clean_state(...)` 模块级函数 |
| `_get_sigmas` | 方法 | 模块级函数 `gather_sigmas(scheduler, timesteps, ...)` |
| `conditioner_sanity_check` | 始终 assert | 改为可选 `validate()` 方法，不在 init 中触发 |

---

## 4. 内部细节重写约定

1. **每文件 docstring 重新写**，不照搬原版文段。原版多使用"This is the ... class which defines..."句式，新版用"Owns ...", "Drives ...", "Provides ..." 等不同主语。
2. **行内注释**全部重写，不复用原版注释的句子。原版多解释 *what*（"Get inputs/latents"），新版改为解释 *why*（"Anchor and goal must share spatial dims for the bridge term to be well-defined"）。
3. **错误消息**重新措辞。原版 `"timesteps and prob should be of same length for custom_timesteps timestep sampling"` → 新版 `"discrete_timesteps / discrete_weights length mismatch ({} vs {})"`。
4. **标识符风格**：原版 `bridge_noise_sigma` 是 snake_case，新版 `noise_jitter` 也是 snake_case，但字段顺序在 `__init__` 中重排。
5. **类层级**：原版 `LBMModel(BaseModel)`，`BaseModel` 提供 `freeze/eval/to`。新版 `BridgeSolver(nn.Module)` 直接继承 `nn.Module`，**不引入 `InferenceCore` 抽象基类**——把原本 BaseModel 的功能作为 `BridgeSolver` 的私有方法实现，并提供一个**独立**的 `InferenceCore` 类作为**另一种可选基类**给 VAE 复用，但**不通过继承关系**让 VAE 和 BridgeSolver 共享。
6. **VAE 不再继承 `BaseModel`**——重写为纯 `nn.Module` + 显式 `device/dtype` 属性。

---

## 5. 文件清单

新建：

- `lbm_native/__init__.py` — 包入口，导出公共 API
- `lbm_native/inference_core.py` — `InferenceCore`, `StageConfig`
- `lbm_native/diffusion_unet.py` — `CondUNet2D`, `PlainUNet2D`
- `lbm_native/latent_codec.py` — `LatentCodec`，内含 tile 工具
- `lbm_native/condition_aggregator.py` — `ConditionAggregator`, `BaseCondition`
- `lbm_native/image_concat_condition.py` — `ImageConcatCondition`, `ImageConcatConfig`
- `lbm_native/timestep_policy.py` — `BridgeSchedule` + 枚举
- `lbm_native/bridge_solver.py` — `BridgeSolver` + 模块级函数

修改：

- `lbm_core/model_factory.py` — 全部改用 `lbm_native.*`，UNet/VAE/Scheduler 配置硬编码保持一致（cfg 不变），但工厂函数内的局部变量名改写
- `nodes/lbm_relighting_pro.py` — `model.sample(...)` → `model.decode_latents_to_pixels(...)`；`bridge_noise_sigma` → `noise_jitter`
- `nodes/lbm_depth_normal_pro.py` — 同上
- `nodes/lbm_batch_processor.py` — 同上

删除：

- 整个 `lbm/` 目录（含 `__pycache__/`）

---

## 6. 测试与验证

1. **现有测试**：`tests/` 目录现有 4 个测试文件覆盖 `lbm_core/`（types/presets/cache/visualizers），与 `lbm_native/` 解耦，**应继续通过** 不需改动。
2. **新增 `tests/test_native_imports.py`**：仅做 `import` 检查，确保新包在无 ComfyUI 环境下也可被解析（公共 API 全部导入不报错）。
3. **新增 `tests/test_native_smoke.py`**：构造 `BridgeSchedule(...)` 不报错；构造 `CondUNet2D(dummy_cfg)` 实例化通过；构造 `LatentCodec(dummy_vae)` 实例化通过。
4. **`grep -rn "from lbm" nodes/ lbm_core/ lbm_native/ lbm/__init__.py`** 应该返回空（除 `lbm_native` 自己的相对导入）。
5. **静态对比**：用 `diff -r` 或类似工具对比 `lbm/`（删除后保留备份在 `docs/superpowers/specs/_legacy_lbm_diff.txt`）与 `lbm_native/` 公共 API 的字符串相似度应**显著低于 30%**。

---

## 7. 不在本次重写范围

- 节点层 UI 字符串、参数顺序、默认值
- `lbm_core/presets.py`, `cache.py`, `visualizers.py`, `types.py`
- 6 个 example_workflows
- `__init__.py` 顶层 `__getattr__` 逻辑
- 模型权重兼容性（用户已接受风险）

---

## 8. 验证清单（实现完成后）

- [ ] `python -c "from lbm_native.bridge_solver import BridgeSolver"` 不报错
- [ ] `python -c "from lbm_native.diffusion_unet import CondUNet2D"` 不报错
- [ ] `python -c "from lbm_native.latent_codec import LatentCodec"` 不报错
- [ ] `python -c "from lbm_native.condition_aggregator import ConditionAggregator"` 不报错
- [ ] `python -c "from lbm_core.model_factory import build_lbm_model, load_lbm_checkpoint"` 不报错
- [ ] `python -c "import nodes.lbm_relighting_pro"` 不报错
- [ ] `pytest tests/ -k "not ttl" -q` 仍然全部通过
- [ ] `pytest tests/test_native_imports.py tests/test_native_smoke.py -q` 通过
- [ ] `ls lbm/` 返回 "No such file or directory"
- [ ] `grep -rn "from lbm\." nodes/ lbm_core/ lbm_native/` 只匹配到 `lbm_native` 内部相对导入
