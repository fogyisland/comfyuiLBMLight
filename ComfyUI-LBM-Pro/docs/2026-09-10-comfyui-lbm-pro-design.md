# ComfyUI-LBM-Pro 设计规范

**日期**：2026-09-10
**状态**：已批准，待实现
**路径**：`ComfyUI-LBM-Pro/`

## 1. 目标与背景

### 1.1 目标

在当前目录 `D:/ToolDevelop/ComfyuiLBM/` 中构建一个新的 ComfyUI 节点包 **ComfyUI-LBM-Pro**，提供比原 `ComfyUI-LBM`（v1.1.0）**更复杂、更专业**的图像处理流水线，针对专业摄影师/视觉艺术家场景。

### 1.2 背景

原项目 `ComfyUI-LBM/` 仅提供两个基础节点：
- `Relighting (LBM)`：单图重光照
- `Depth / Normal (LBM)`：深度图/法线图生成

主要痛点：
1. 每次调用都重新创建和加载模型（约 5–10 秒开销）
2. 缺少深度/法线的可视化（输出的是 latent 图）
3. 缺少专业光照控制（仅能改 noise sigma）
4. 没有对比工具、批处理、模型复用机制

### 1.3 范围

**包含**：节点注册、模型缓存、5 个工作流示例、文档
**不包含**：HDR/EXR 加载、tile 多尺度、ControlNet 集成、训练/微调

## 2. 架构

### 2.1 顶层目录结构

```
ComfyUI-LBM-Pro/
├── __init__.py                       # 节点注册入口
├── nodes/                            # 所有节点实现
│   ├── __init__.py
│   ├── lbm_model_loader.py
│   ├── lbm_light_preset.py
│   ├── lbm_relighting_pro.py
│   ├── lbm_depth_normal_pro.py
│   ├── lbm_depth_visualizer.py
│   ├── lbm_normal_visualizer.py
│   ├── lbm_compare_grid.py
│   └── lbm_batch_processor.py
├── lbm_core/                         # 核心模块
│   ├── __init__.py
│   ├── cache.py                      # 模型缓存
│   ├── presets.py                    # 光照预设数据库
│   ├── model_factory.py              # LBM 模型构建
│   ├── visualizers.py                # colormap 实现
│   └── types.py                      # 自定义 ComfyUI 类型
├── lbm/                              # 从原项目复用（不修改）
├── example_workflows/                # 5-6 个 .json 示例
├── docs/
├── README.md
├── requirements.txt
└── pyproject.toml
```

### 2.2 核心模块

#### `lbm_core/cache.py` — `LBMModelCache`

```python
class LBMModelCache:
    """单例缓存，避免重复模型加载。"""

    _cache: dict[str, dict[str, Any]] = {}  # key -> {"model": LBMModel, "dtype": torch.dtype, "last_used": timestamp}

    @classmethod
    def get_or_load(cls, key: str, loader_fn: Callable, ttl_seconds: int = 600) -> LBMModel:
        """获取或加载模型。10 分钟无访问自动卸载。"""

    @classmethod
    def unload(cls, key: str) -> bool: ...

    @classmethod
    def clear(cls) -> int: ...  # 返回卸载数量
```

线程安全通过 `threading.Lock` 保证。

#### `lbm_core/presets.py` — 光照预设

```python
@dataclass
class LightPreset:
    name: str
    sh_coeffs: list[float]   # 9 个球谐系数
    intensity: float         # 0–2
    temperature: float       # 色温（K），1000–40000
    description: str

PRESETS: dict[str, LightPreset] = {
    "golden_hour": LightPreset(...),
    "overcast":    LightPreset(...),
    "studio_left": LightPreset(...),
    "studio_top":  LightPreset(...),
    "sunset":      LightPreset(...),
    "night_blue":  LightPreset(...),
    "cool_neutral": LightPreset(...),
    "warm_neutral": LightPreset(...),
}

def build_custom_preset(azimuth_deg: float, elevation_deg: float,
                        intensity: float, temperature_k: float) -> LightPreset:
    """根据方向参数构造预设。"""
```

#### `lbm_core/model_factory.py`

封装原 `LBM_Relighting.create_lbm_model()` 和 `LBM_DepthNormal.create_lbm_model()` 的逻辑。提供：

```python
def build_lbm_model(task: Literal["relighting", "depth", "normal"],
                    dtype: torch.dtype, bridge_noise_sigma: float) -> LBMModel: ...

def load_lbm_checkpoint(model: LBMModel, ckpt_path: str,
                        dtype: torch.dtype) -> None: ...
```

#### `lbm_core/visualizers.py`

```python
COLORMAPS = {"viridis": _viridis, "inferno": _inferno, "turbo": _turbo, "gray": _gray}

def depth_to_colormap(depth: torch.Tensor, colormap: str,
                      invert: bool = False, normalize: bool = True) -> torch.Tensor: ...

def normalize_normal_map(normal: torch.Tensor) -> torch.Tensor: ...
```

#### `lbm_core/types.py`

```python
LBM_MODEL_TYPE = "LBM_MODEL"        # 节点间传递 LBM 模型对象
LIGHT_PRESET_TYPE = "LIGHT_PRESET"  # 节点间传递光照预设
```

## 3. 节点规格

### 3.1 `LBM Model Loader`

**类名**：`LBM_Model_Loader`
**功能**：从磁盘加载 `.safetensors` 并缓存为 `LBM_MODEL` 类型。
**输入**：
- `model_name`（必）：文件名枚举（扫描 `models/diffusion_models/LBM/`）
- `task`（必）：`relighting` / `depth` / `normal`
- `precision`（必）：`fp32` / `bf16` / `fp16`（默认 `bf16`）
- `bridge_noise_sigma`（可选）：默认 0.005

**输出**：`LBM_MODEL`
**类别**：`🧪AILab/🔆LBM-Pro`

**自动下载**：若模型不存在，从 `https://huggingface.co/jasperai/LBM_<task>/resolve/main/model.safetensors` 下载到 `ComfyUI/models/diffusion_models/LBM/`。

### 3.2 `LBM Light Preset`

**类名**：`LBM_Light_Preset`
**功能**：选择或构建光照预设。
**输入**：
- `mode`（必）：`preset` / `custom`
- `preset_name`（preset 模式）：8 个内置预设之一
- `azimuth_deg`、`elevation_deg`、`intensity`、`temperature_k`（custom 模式）

**输出**：`LIGHT_PRESET`
**类别**：`🧪AILab/🔆LBM-Pro`

### 3.3 `LBM Relighting Pro`

**类名**：`LBM_Relighting_Pro`
**功能**：增强版重光照，支持光照预设。
**输入**：
- `lbm_model`（必，`LBM_MODEL`）：来自 Model Loader
- `image`（必，`IMAGE`）
- `light_preset`（可选，`LIGHT_PRESET`）：未连接时使用 `warm_neutral`
- `steps`（必）：1–100，默认 28
- `mask`（可选，`MASK`）

**输出**：`IMAGE`
**类别**：`🧪AILab/🔆LBM-Pro`

**注意**：LBM 原始模型不接受外部光源条件注入。光照预设目前通过**调整 `bridge_noise_sigma` + 多结果平均**的方式模拟不同光照风格（实现细节见 §6）。这不影响节点接口，但用户需理解语义。

### 3.4 `LBM Depth/Normal Pro`

**类名**：`LBM_DepthNormal_Pro`
**功能**：增强版深度/法线生成，输出 raw latent 图像（便于下游可视化）。
**输入**：
- `lbm_model`（必，`LBM_MODEL`）
- `image`（必）
- `task`（必）：`depth` / `normal`
- `steps`（必）：默认 28
- `bridge_noise_sigma`（可选）：默认 0.1
- `mask`（可选）

**输出**：`IMAGE`（raw）、`IMAGE`（后处理：depth 模式下已 invert）
**类别**：`🧪AILab/🔆LBM-Pro`

### 3.5 `LBM Depth Visualizer`

**类名**：`LBM_Depth_Visualizer`
**功能**：将深度图（latent）转为可视化彩图。
**输入**：
- `depth_image`（必，`IMAGE`）
- `colormap`（必）：`viridis` / `inferno` / `turbo` / `gray`（默认 `turbo`）
- `invert`（可选，默认 False）
- `auto_normalize`（可选，默认 True）

**输出**：`IMAGE`（彩色图）

### 3.6 `LBM Normal Visualizer`

**类名**：`LBM_Normal_Visualizer`
**功能**：校验并规范化法线图。
**输入**：
- `normal_image`（必）
- `normalize_range`（可选，默认 True）：确保输出在 [-1, 1]

**输出**：`IMAGE`

### 3.7 `LBM Compare Grid`

**类名**：`LBM_Compare_Grid`
**功能**：将 2–9 张图拼成对比网格。
**输入**：
- `images`：2–9 张 `IMAGE`
- `layout`：`auto` / `horizontal` / `vertical` / `grid_2x2` / `grid_3x3`
- `labels`（可选）：每张图的标签字符串列表
- `padding`（可选，默认 8px）

**输出**：`IMAGE`
**类别**：`🧪AILab/🔆LBM-Pro`

### 3.8 `LBM Batch Processor`

**类名**：`LBM_Batch_Processor`
**功能**：对多张图应用同一组参数（保证一致性）。
**输入**：
- `lbm_model`（必，`LBM_MODEL`）
- `images`（必，`IMAGE`，batch 维度）
- `light_preset`（可选，`LIGHT_PRESET`）
- `steps`（必）：默认 28

**输出**：`IMAGE`（batch）
**类别**：`🧪AILab/🔆LBM-Pro`

## 4. 数据流示例

### 4.1 基础重光照

```
[Load Image] → [LBM Model Loader] → [LBM Relighting Pro] → [Save Image]
                                  ↑
                            [LBM Light Preset]
```

### 4.2 深度+可视化

```
[Load Image] → [LBM Model Loader(task=depth)]
                            ↓
                   [LBM Depth/Normal Pro] → [LBM Depth Visualizer(colormap=turbo)] → [Preview]
```

### 4.3 多光照对比

```
[Load Image] ──────────────┬─→ [LBM Relighting Pro(p1)] ──┐
                           ├─→ [LBM Relighting Pro(p2)] ──┤
[Model Loader] ────────────┼─→ [LBM Relighting Pro(p3)] ──┼─→ [LBM Compare Grid] → [Save]
                           ├─→ [LBM Relighting Pro(p4)] ──┤
[Light Preset p1..p4] ─────┴─→ ...                       ┘
```

## 5. 示例工作流

| 文件 | 描述 |
|------|------|
| `01_basic_relighting.json` | 加载模型 + 重光照 + 保存 |
| `02_light_presets.json` | 同一图 4 种光照对比 |
| `03_depth_normal_pipeline.json` | 深度+法线+colormap 可视化 |
| `04_model_cache_chain.json` | 一个模型多次复用（演示缓存） |
| `05_compare_grid.json` | 法线可视化对比网格 |
| `06_batch_processing.json` | 批量处理 4 张图 |

## 6. 关键实现细节

### 6.1 模型缓存键

```python
cache_key = f"{model_name}|{task}|{precision}"
```

### 6.2 光照预设的"模拟"实现

LBM 模型不接受外部光源条件。本节点通过以下方式**视觉上**模拟不同光照：

| 预设 | RGB tint | intensity | bridge_noise_sigma |
|------|---------|-----------|-------------------|
| `golden_hour` | `(1.15, 0.95, 0.75)` | 1.1 | 0.008 |
| `overcast` | `(0.95, 0.95, 0.95)` | 0.7 | 0.003 |
| `studio_left` | `(1.0, 1.0, 1.0)` | 1.0 | 0.005 |
| `studio_top` | `(1.05, 1.05, 1.0)` | 1.2 | 0.005 |
| `sunset` | `(1.2, 0.85, 0.7)` | 1.0 | 0.010 |
| `night_blue` | `(0.7, 0.85, 1.1)` | 0.6 | 0.020 |
| `cool_neutral` | `(0.95, 0.98, 1.05)` | 0.95 | 0.005 |
| `warm_neutral` | `(1.05, 1.0, 0.95)` | 1.0 | 0.005 |

Custom 模式：根据 `temperature_k` 插值 RGB tint（黑体辐射近似），根据 `intensity` 直接乘 intensity，根据 `elevation_deg` 调整 sigma（低角度=高 sigma）。

> **未来扩展点**：若 LBM 模型后续版本支持条件输入，可替换此实现。

### 6.3 缓存 TTL

10 分钟无访问 → 模型移到 CPU + 释放 GPU 显存（`mm.soft_empty_cache()`）。

### 6.4 错误处理

- 模型文件不存在 → 自动下载（重试 1 次，失败抛 `RuntimeError`）
- 显存不足 → 自动降精度（fp32 → bf16 → fp16）
- 用户提供的 `images` 数量与 `labels` 数量不匹配 → 抛 `ValueError`，提示应一致

### 6.5 类型注册

在 `__init__.py` 中显式注册自定义类型，避免 ComfyUI 版本差异导致识别失败：

```python
from lbm_core.types import LBM_MODEL_TYPE, LIGHT_PRESET_TYPE

NODE_CLASS_MAPPINGS = {...}
NODE_DISPLAY_NAME_MAPPINGS = {...}

# 自定义类型（ComfyUI ≥ 2024 机制）
try:
    from comfy_execution.graph_utils import register_type
    register_type(LBM_MODEL_TYPE, dict)
    register_type(LIGHT_PRESET_TYPE, dict)
except ImportError:
    pass  # 旧版本 ComfyUI 跳过
```

## 7. 测试与验证

### 7.1 单元测试（手动 + 自动混合）

由于依赖 ComfyUI 环境，不强制 Pytest 套件，但每个节点必须：
- 可在 ComfyUI 中**正确注册**（`__init__.py` 中导入无异常）
- `INPUT_TYPES()` 返回合法字典
- `process()` / `light_preset()` 函数签名匹配

### 7.2 冒烟测试

实现一个独立的 `tests/smoke_import.py`：
```python
import sys
sys.path.insert(0, "ComfyUI-LBM-Pro")
from nodes import (LBM_Model_Loader, LBM_Light_Preset, LBM_Relighting_Pro,
                   LBM_DepthNormal_Pro, LBM_Depth_Visualizer, LBM_Normal_Visualizer,
                   LBM_Compare_Grid, LBM_Batch_Processor)
print("All nodes import OK")
```

### 7.3 端到端验证

通过提供 5–6 个 `.json` 示例工作流，让用户在 ComfyUI 中实际运行。

## 8. 风险与缓解

| 风险 | 缓解 |
|------|------|
| LBM 模型不接受外部光照条件 → 光照"模拟"看起来不真实 | 在 README 中明确说明这是 v1 模拟方案，预留 future hook |
| 模型缓存可能持有 GPU 显存过久 | TTL 机制 + 提供 `LBM Clear Cache` 节点（可选） |
| 法线/深度 colormap 实现不准确 | 复用 OpenCV/matplotlib 实现，附单元测试 |
| 自定义 ComfyUI 类型在某些版本失效 | 在 `__init__.py` 显式 register types |

## 9. 实施计划（指向 writing-plans）

将交由 `superpowers:writing-plans` skill 拆解为实现任务列表。
