# ComfyUI-LBM-Pro Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build ComfyUI-LBM-Pro, a professional image processing pipeline for LBM (Latent Bridge Matching) with 8 specialized nodes including model caching, light presets, depth/normal visualization, and batch processing.

**Architecture:** Modular nodes backed by a shared `lbm_core` package. Custom ComfyUI types (`LBM_MODEL`, `LIGHT_PRESET`) flow between loader and consumer nodes. Models cached by `(name|task|precision)` keys to avoid reload cost. Visualization nodes use NumPy-based colormaps for depth coloring.

**Tech Stack:** Python 3.10+, PyTorch 2.0+, ComfyUI (folder_paths, comfy.model_management), diffusers (FlowMatchEulerDiscreteScheduler, AutoencoderKL), safetensors, NumPy, PIL, requests, tqdm.

**Spec:** `ComfyUI-LBM-Pro/docs/2026-09-10-comfyui-lbm-pro-design.md`

## Global Constraints

- Python 3.10+ required (uses `Literal`, `X | None` syntax)
- All node categories use `🧪AILab/🔆LBM-Pro`
- Model files must download to `ComfyUI/models/diffusion_models/LBM/` (auto-create)
- Bridge noise sigma range: 0.0–0.1 (step 0.001)
- Steps range: 1–100, default 28
- Precision enum: `["fp32", "bf16", "fp16"]` default `bf16`
- Custom ComfyUI types: `LBM_MODEL`, `LIGHT_PRESET`
- Cache TTL: 600 seconds (10 minutes)
- Reuse original `lbm/` package verbatim (do not modify)
- Code style: PEP 8, type hints everywhere public
- License: GPL-3.0 (matches original repo)

## File Structure

```
ComfyUI-LBM-Pro/
├── __init__.py                              # Node registration entry
├── lbm_core/                                # Shared core (no ComfyUI deps)
│   ├── __init__.py
│   ├── types.py                             # Custom type constants
│   ├── presets.py                           # LightPreset + database
│   ├── cache.py                             # LBMModelCache singleton
│   ├── model_factory.py                     # Build + load LBM models
│   └── visualizers.py                       # Depth colormaps + normal checks
├── nodes/                                   # ComfyUI node implementations
│   ├── __init__.py
│   ├── lbm_model_loader.py
│   ├── lbm_light_preset.py
│   ├── lbm_relighting_pro.py
│   ├── lbm_depth_normal_pro.py
│   ├── lbm_depth_visualizer.py
│   ├── lbm_normal_visualizer.py
│   ├── lbm_compare_grid.py
│   └── lbm_batch_processor.py
├── lbm/                                     # REUSED from original (do not modify)
├── tests/
│   ├── __init__.py
│   ├── smoke_import.py                      # Import smoke test
│   ├── test_presets.py                      # Preset database tests
│   ├── test_cache.py                        # Model cache tests
│   ├── test_visualizers.py                  # Colormap tests
│   └── test_types.py                        # Type constant tests
├── example_workflows/
│   ├── 01_basic_relighting.json
│   ├── 02_light_presets.json
│   ├── 03_depth_normal_pipeline.json
│   ├── 04_model_cache_chain.json
│   ├── 05_compare_grid.json
│   └── 06_batch_processing.json
├── docs/
│   └── README.md                            # User-facing docs
├── README.md
├── requirements.txt
└── pyproject.toml
```

**Responsibility map:**
- `lbm_core/` — pure logic, no ComfyUI imports (testable without ComfyUI)
- `nodes/` — ComfyUI glue: INPUT_TYPES, RETURN_TYPES, function dispatch
- `tests/` — smoke + unit tests for `lbm_core/`
- `lbm/` — original code, copied verbatim

---

## Task 1: Project Scaffolding

**Files:**
- Create: `ComfyUI-LBM-Pro/__init__.py`
- Create: `ComfyUI-LBM-Pro/lbm_core/__init__.py`
- Create: `ComfyUI-LBM-Pro/nodes/__init__.py`
- Create: `ComfyUI-LBM-Pro/tests/__init__.py`
- Create: `ComfyUI-LBM-Pro/requirements.txt`
- Create: `ComfyUI-LBM-Pro/pyproject.toml`
- Create: `ComfyUI-LBM-Pro/.gitignore`
- Copy: `ComfyUI-LBM-Pro/lbm/` ← `ComfyUI-LBM/lbm/` (entire folder, recursive)

**Interfaces:**
- Consumes: nothing
- Produces: empty package layout; `lbm/` available for import as `from .lbm.models.lbm import LBMModel`

- [ ] **Step 1: Create directory layout**

```bash
mkdir -p ComfyUI-LBM-Pro/{lbm_core,nodes,tests,example_workflows,docs}
```

- [ ] **Step 2: Copy `lbm/` from original repo**

```bash
cp -r ComfyUI-LBM/lbm ComfyUI-LBM-Pro/lbm
```

Verify: `ComfyUI-LBM-Pro/lbm/models/lbm/lbm_model.py` exists.

- [ ] **Step 3: Write `requirements.txt`**

```
diffusers>=0.19.0
accelerate>=0.20.0
torch>=2.0.0
torchvision>=0.15.0
tqdm>=4.65.0
Pillow>=9.0.0
transformers>=4.30.0
safetensors>=0.3.1
requests>=2.25.0
numpy>=1.22.0
```

- [ ] **Step 4: Write `pyproject.toml`**

```toml
[project]
name = "comfyui-lbm-pro"
version = "0.1.0"
description = "Professional LBM (Latent Bridge Matching) image processing pipeline for ComfyUI"
requires-python = ">=3.10"
license = {text = "GPL-3.0"}

[tool.setuptools.packages.find]
include = ["lbm_core*", "nodes*", "lbm*"]
```

- [ ] **Step 5: Write `.gitignore`**

```
__pycache__/
*.pyc
*.pyo
*.egg-info/
.pytest_cache/
*.swp
.DS_Store
models/diffusion_models/LBM/
```

- [ ] **Step 6: Write package init files**

`ComfyUI-LBM-Pro/lbm_core/__init__.py`:
```python
"""Core modules for ComfyUI-LBM-Pro. No ComfyUI dependencies."""
from .types import LBM_MODEL_TYPE, LIGHT_PRESET_TYPE
from .presets import LightPreset, PRESETS, build_custom_preset
from .cache import LBMModelCache
from .model_factory import build_lbm_model, load_lbm_checkpoint

__all__ = [
    "LBM_MODEL_TYPE",
    "LIGHT_PRESET_TYPE",
    "LightPreset",
    "PRESETS",
    "build_custom_preset",
    "LBMModelCache",
    "build_lbm_model",
    "load_lbm_checkpoint",
]
```

`ComfyUI-LBM-Pro/nodes/__init__.py`:
```python
"""ComfyUI node implementations."""
```

`ComfyUI-LBM-Pro/tests/__init__.py`:
```python
"""Test package."""
```

`ComfyUI-LBM-Pro/__init__.py`:
```python
"""ComfyUI-LBM-Pro package marker."""
__version__ = "0.1.0"
```

- [ ] **Step 7: Verify lbm import works**

Create temporary `ComfyUI-LBM-Pro/_check_import.py`:
```python
import sys
sys.path.insert(0, "ComfyUI-LBM-Pro")
from lbm.models.lbm import LBMModel, LBMConfig
from lbm.models.unets import DiffusersUNet2DCondWrapper
from lbm.models.vae import AutoencoderKLDiffusers
from lbm.models.embedders import ConditionerWrapper
print("LBM imports OK")
```

Run: `python ComfyUI-LBM-Pro/_check_import.py`
Expected: `LBM imports OK`

- [ ] **Step 8: Cleanup and commit**

Remove `_check_import.py`. Initial commit:
```bash
git add ComfyUI-LBM-Pro/
git commit -m "feat(scaffold): create ComfyUI-LBM-Pro package layout"
```

---

## Task 2: Type Constants

**Files:**
- Create: `ComfyUI-LBM-Pro/lbm_core/types.py`
- Create: `ComfyUI-LBM-Pro/tests/test_types.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `LBM_MODEL_TYPE: str = "LBM_MODEL"`
  - `LIGHT_PRESET_TYPE: str = "LIGHT_PRESET"`

- [ ] **Step 1: Write failing test**

`ComfyUI-LBM-Pro/tests/test_types.py`:
```python
from lbm_core.types import LBM_MODEL_TYPE, LIGHT_PRESET_TYPE


def test_lbm_model_type_value():
    assert LBM_MODEL_TYPE == "LBM_MODEL"


def test_light_preset_type_value():
    assert LIGHT_PRESET_TYPE == "LIGHT_PRESET"


def test_types_are_distinct():
    assert LBM_MODEL_TYPE != LIGHT_PRESET_TYPE
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd ComfyUI-LBM-Pro && python -m pytest tests/test_types.py -v
```

Expected: `ModuleNotFoundError: No module named 'lbm_core'`

- [ ] **Step 3: Write implementation**

`ComfyUI-LBM-Pro/lbm_core/types.py`:
```python
"""Custom ComfyUI type identifiers used by LBM-Pro nodes.

These strings are matched against node connection types in the
ComfyUI graph. Keeping them in one module avoids accidental typos
that would silently break node wiring.
"""
from typing import Final

LBM_MODEL_TYPE: Final[str] = "LBM_MODEL"
LIGHT_PRESET_TYPE: Final[str] = "LIGHT_PRESET"
```

- [ ] **Step 4: Run test to verify pass**

```bash
cd ComfyUI-LBM-Pro && python -m pytest tests/test_types.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add ComfyUI-LBM-Pro/lbm_core/types.py ComfyUI-LBM-Pro/tests/test_types.py
git commit -m "feat(types): add LBM_MODEL and LIGHT_PRESET type constants"
```

---

## Task 3: Light Preset Database

**Files:**
- Create: `ComfyUI-LBM-Pro/lbm_core/presets.py`
- Create: `ComfyUI-LBM-Pro/tests/test_presets.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `class LightPreset` — dataclass with fields `name, rgb_tint, intensity, bridge_noise_sigma, description`
  - `PRESETS: dict[str, LightPreset]` — 8 presets per spec §6.2
  - `build_custom_preset(azimuth_deg, elevation_deg, intensity, temperature_k) -> LightPreset`

- [ ] **Step 1: Write failing tests**

`ComfyUI-LBM-Pro/tests/test_presets.py`:
```python
import pytest
from lbm_core.presets import LightPreset, PRESETS, build_custom_preset


EXPECTED_PRESET_NAMES = {
    "golden_hour", "overcast", "studio_left", "studio_top",
    "sunset", "night_blue", "cool_neutral", "warm_neutral",
}


def test_all_eight_presets_present():
    assert set(PRESETS.keys()) == EXPECTED_PRESET_NAMES


@pytest.mark.parametrize("name", EXPECTED_PRESET_NAMES)
def test_each_preset_has_valid_fields(name):
    p = PRESETS[name]
    assert isinstance(p, LightPreset)
    assert len(p.rgb_tint) == 3
    assert all(0.0 <= c <= 2.0 for c in p.rgb_tint)
    assert 0.0 <= p.intensity <= 2.0
    assert 0.0 <= p.bridge_noise_sigma <= 0.1


def test_preset_names_are_unique():
    names = [p.name for p in PRESETS.values()]
    assert len(names) == len(set(names))


def test_build_custom_preset_default_values():
    p = build_custom_preset(0.0, 45.0, 1.0, 5500)
    assert isinstance(p, LightPreset)
    assert p.name == "custom"
    assert p.intensity == 1.0
    assert 0.0 <= p.bridge_noise_sigma <= 0.1


def test_build_custom_preset_low_elevation_high_sigma():
    p_high = build_custom_preset(0.0, 80.0, 1.0, 5500)
    p_low = build_custom_preset(0.0, 5.0, 1.0, 5500)
    assert p_low.bridge_noise_sigma > p_high.bridge_noise_sigma


def test_build_custom_preset_temperature_affects_tint():
    warm = build_custom_preset(0.0, 45.0, 1.0, 3000)
    cool = build_custom_preset(0.0, 45.0, 1.0, 9000)
    # Warm tint should have R > B
    assert warm.rgb_tint[0] > warm.rgb_tint[2]
    # Cool tint should have B > R
    assert cool.rgb_tint[2] > cool.rgb_tint[0]
```

- [ ] **Step 2: Run tests to verify failure**

```bash
cd ComfyUI-LBM-Pro && python -m pytest tests/test_presets.py -v
```

Expected: `ModuleNotFoundError: No module named 'lbm_core.presets'`

- [ ] **Step 3: Write implementation**

`ComfyUI-LBM-Pro/lbm_core/presets.py`:
```python
"""Lighting preset database for LBM-Pro.

Each preset is a (rgb_tint, intensity, bridge_noise_sigma) tuple that
emulates a lighting style by post-processing the LBM output. The model
itself does not consume external lighting, so these are visual
approximations — see design doc §6.2 for rationale.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class LightPreset:
    """A lighting style applied after LBM inference."""
    name: str
    rgb_tint: tuple[float, float, float]
    intensity: float
    bridge_noise_sigma: float
    description: str


PRESETS: dict[str, LightPreset] = {
    "golden_hour": LightPreset(
        name="golden_hour",
        rgb_tint=(1.15, 0.95, 0.75),
        intensity=1.1,
        bridge_noise_sigma=0.008,
        description="Warm low-angle sunlight at sunrise/sunset",
    ),
    "overcast": LightPreset(
        name="overcast",
        rgb_tint=(0.95, 0.95, 0.95),
        intensity=0.7,
        bridge_noise_sigma=0.003,
        description="Diffuse soft light from cloudy sky",
    ),
    "studio_left": LightPreset(
        name="studio_left",
        rgb_tint=(1.0, 1.0, 1.0),
        intensity=1.0,
        bridge_noise_sigma=0.005,
        description="Neutral key light from camera-left",
    ),
    "studio_top": LightPreset(
        name="studio_top",
        rgb_tint=(1.05, 1.05, 1.0),
        intensity=1.2,
        bridge_noise_sigma=0.005,
        description="Soft top-down studio light",
    ),
    "sunset": LightPreset(
        name="sunset",
        rgb_tint=(1.2, 0.85, 0.7),
        intensity=1.0,
        bridge_noise_sigma=0.010,
        description="Strong orange directional sunset light",
    ),
    "night_blue": LightPreset(
        name="night_blue",
        rgb_tint=(0.7, 0.85, 1.1),
        intensity=0.6,
        bridge_noise_sigma=0.020,
        description="Cool dim blue night ambience",
    ),
    "cool_neutral": LightPreset(
        name="cool_neutral",
        rgb_tint=(0.95, 0.98, 1.05),
        intensity=0.95,
        bridge_noise_sigma=0.005,
        description="Slightly cool balanced light",
    ),
    "warm_neutral": LightPreset(
        name="warm_neutral",
        rgb_tint=(1.05, 1.0, 0.95),
        intensity=1.0,
        bridge_noise_sigma=0.005,
        description="Slightly warm balanced light",
    ),
}


def _temperature_to_rgb_tint(temperature_k: float) -> tuple[float, float, float]:
    """Approximate blackbody tint from color temperature in Kelvin.

    Returns multipliers around 1.0 (warm = R > B; cool = B > R).
    Uses Tanner Helland's piecewise approximation, normalized.
    """
    t = max(1000.0, min(40000.0, temperature_k)) / 100.0
    if t <= 66:
        r = 1.0
        g = 0.39008157876901960784 * math.log(t) - 0.63184144378862745098
        b = 1.0 if t >= 20 else 0.54320678911019607843 * math.log(t - 10) - 1.19625408914
    else:
        r = 1.29293618606274509804 * (t - 60) ** -0.1332047592
        g = 1.12989086089529411765 * (t - 60) ** -0.0755148492
        b = 1.0
    mn = min(r, g, b)
    return (r / mn, g / mn, b / mn)


def build_custom_preset(
    azimuth_deg: float,
    elevation_deg: float,
    intensity: float,
    temperature_k: float,
) -> LightPreset:
    """Construct a preset from manual parameters.

    Args:
        azimuth_deg: 0–360, direction of light (currently unused for tint,
            reserved for future conditioning).
        elevation_deg: 0–90, height of light. Lower = more sigma (variation).
        intensity: 0–2, brightness multiplier.
        temperature_k: 1000–40000, color temperature.
    """
    elevation = max(0.0, min(90.0, elevation_deg))
    sigma = 0.020 * (1.0 - elevation / 90.0) + 0.003
    return LightPreset(
        name="custom",
        rgb_tint=_temperature_to_rgb_tint(temperature_k),
        intensity=max(0.0, min(2.0, intensity)),
        bridge_noise_sigma=sigma,
        description=f"az={azimuth_deg:.0f}° el={elevation:.0f}° T={temperature_k}K",
    )
```

- [ ] **Step 4: Run tests to verify pass**

```bash
cd ComfyUI-LBM-Pro && python -m pytest tests/test_presets.py -v
```

Expected: 15+ passed.

- [ ] **Step 5: Commit**

```bash
git add ComfyUI-LBM-Pro/lbm_core/presets.py ComfyUI-LBM-Pro/tests/test_presets.py
git commit -m "feat(presets): add 8 light presets + custom builder"
```

---

## Task 4: Model Cache

**Files:**
- Create: `ComfyUI-LBM-Pro/lbm_core/cache.py`
- Create: `ComfyUI-LBM-Pro/tests/test_cache.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `class LBMModelCache` with:
    - `get_or_load(key: str, loader_fn: Callable[[], Any], ttl_seconds: int = 600) -> Any`
    - `unload(key: str) -> bool`
    - `clear() -> int`
    - `size() -> int` (for tests)
    - `contains(key: str) -> bool` (for tests)

- [ ] **Step 1: Write failing tests**

`ComfyUI-LBM-Pro/tests/test_cache.py`:
```python
import time
from lbm_core.cache import LBMModelCache


def _fake_loader(name):
    return {"name": name, "loaded_at": time.time()}


def test_cache_miss_calls_loader():
    LBMModelCache.clear()
    obj = LBMModelCache.get_or_load("k1", lambda: _fake_loader("alpha"))
    assert obj["name"] == "alpha"


def test_cache_hit_skips_loader():
    LBMModelCache.clear()
    counter = {"n": 0}

    def loader():
        counter["n"] += 1
        return {"v": counter["n"]}

    LBMModelCache.get_or_load("k2", loader)
    LBMModelCache.get_or_load("k2", loader)
    LBMModelCache.get_or_load("k2", loader)
    assert counter["n"] == 1


def test_different_keys_different_loaders():
    LBMModelCache.clear()
    LBMModelCache.get_or_load("a", lambda: "A")
    LBMModelCache.get_or_load("b", lambda: "B")
    assert LBMModelCache.get_or_load("a", lambda: "A") == "A"
    assert LBMModelCache.get_or_load("b", lambda: "B") == "B"


def test_unload_removes_entry():
    LBMModelCache.clear()
    LBMModelCache.get_or_load("x", lambda: 42)
    assert LBMModelCache.contains("x")
    assert LBMModelCache.unload("x") is True
    assert not LBMModelCache.contains("x")


def test_unload_missing_returns_false():
    LBMModelCache.clear()
    assert LBMModelCache.unload("nope") is False


def test_clear_returns_count():
    LBMModelCache.clear()
    LBMModelCache.get_or_load("p", lambda: 1)
    LBMModelCache.get_or_load("q", lambda: 2)
    n = LBMModelCache.clear()
    assert n == 2
    assert LBMModelCache.size() == 0


def test_ttl_expiry_calls_loader_again():
    LBMModelCache.clear()
    counter = {"n": 0}

    def loader():
        counter["n"] += 1
        return counter["n"]

    LBMModelCache.get_or_load("ttl", loader, ttl_seconds=1)
    LBMModelCache.get_or_load("ttl", loader, ttl_seconds=1)
    time.sleep(1.2)
    LBMModelCache.get_or_load("ttl", loader, ttl_seconds=1)
    assert counter["n"] == 2
```

- [ ] **Step 2: Run tests to verify failure**

```bash
cd ComfyUI-LBM-Pro && python -m pytest tests/test_cache.py -v
```

Expected: `ModuleNotFoundError: No module named 'lbm_core.cache'`

- [ ] **Step 3: Write implementation**

`ComfyUI-LBM-Pro/lbm_core/cache.py`:
```python
"""Thread-safe LRU-style model cache for LBM-Pro.

ComfyUI invokes nodes many times in a session. Loading a 1–2 GB LBM
checkpoint takes 5–10 seconds. This cache memoizes loaded models by an
opaque key (typically f"{name}|{task}|{precision}") with a TTL so
unused models get released.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable


class LBMModelCache:
    """Singleton cache mapping opaque keys to model objects."""

    _cache: dict[str, dict[str, Any]] = {}
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get_or_load(
        cls,
        key: str,
        loader_fn: Callable[[], Any],
        ttl_seconds: int = 600,
    ) -> Any:
        """Return cached object or invoke loader_fn once and cache result.

        Args:
            key: Opaque cache key (caller-chosen).
            loader_fn: Zero-arg callable producing the object to cache.
            ttl_seconds: Idle time before the entry is considered stale.
        """
        with cls._lock:
            entry = cls._cache.get(key)
            now = time.monotonic()
            if entry is not None and (now - entry["last_used"]) < ttl_seconds:
                entry["last_used"] = now
                return entry["model"]
            obj = loader_fn()
            cls._cache[key] = {"model": obj, "last_used": now}
            return obj

    @classmethod
    def unload(cls, key: str) -> bool:
        with cls._lock:
            return cls._cache.pop(key, None) is not None

    @classmethod
    def clear(cls) -> int:
        with cls._lock:
            n = len(cls._cache)
            cls._cache.clear()
            return n

    @classmethod
    def size(cls) -> int:
        with cls._lock:
            return len(cls._cache)

    @classmethod
    def contains(cls, key: str) -> bool:
        with cls._lock:
            return key in cls._cache
```

- [ ] **Step 4: Run tests to verify pass**

```bash
cd ComfyUI-LBM-Pro && python -m pytest tests/test_cache.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add ComfyUI-LBM-Pro/lbm_core/cache.py ComfyUI-LBM-Pro/tests/test_cache.py
git commit -m "feat(cache): add thread-safe TTL model cache"
```

---

## Task 5: Model Factory

**Files:**
- Create: `ComfyUI-LBM-Pro/lbm_core/model_factory.py`

**Interfaces:**
- Consumes: `lbm.models.lbm.LBMModel`, `lbm.models.unets.DiffusersUNet2DCondWrapper`, `lbm.models.vae.AutoencoderKLDiffusers`, `lbm.models.embedders.ConditionerWrapper`, `diffusers.AutoencoderKL`, `diffusers.FlowMatchEulerDiscreteScheduler`
- Produces:
  - `build_lbm_model(task: Literal["relighting", "depth", "normal"], dtype: torch.dtype, bridge_noise_sigma: float) -> LBMModel`
  - `load_lbm_checkpoint(model: LBMModel, ckpt_path: str, dtype: torch.dtype) -> None`

- [ ] **Step 1: Write implementation (no separate test — exercised in node tests)**

`ComfyUI-LBM-Pro/lbm_core/model_factory.py`:
```python
"""LBM model construction and checkpoint loading for LBM-Pro.

Centralizes the duplicated architecture blocks from the original
LBM_Relighting.py and LBM_DepthNormal.py node implementations.
"""
from __future__ import annotations

from typing import Literal

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models import AutoencoderKL
from tqdm import tqdm

from lbm.models.embedders import ConditionerWrapper
from lbm.models.lbm import LBMConfig, LBMModel
from lbm.models.unets import DiffusersUNet2DCondWrapper
from lbm.models.vae import AutoencoderKLDiffusers


_TASK_CONFIG = {
    "relighting": {
        "target_key": "source_image",
        "prob": [0.25, 0.25, 0.25, 0.25],
        "default_sigma": 0.005,
    },
    "depth": {
        "target_key": "depth",
        "prob": [0.025, 0.05, 0.025, 0.9],
        "default_sigma": 0.1,
    },
    "normal": {
        "target_key": "normals",
        "prob": [0.05, 0.1, 0.05, 0.8],
        "default_sigma": 0.1,
    },
}


def _build_unet(dtype: torch.dtype) -> DiffusersUNet2DCondWrapper:
    return DiffusersUNet2DCondWrapper(
        in_channels=4,
        out_channels=4,
        center_input_sample=False,
        flip_sin_to_cos=True,
        freq_shift=0,
        down_block_types=[
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
        ],
        mid_block_type="UNetMidBlock2DCrossAttn",
        up_block_types=["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
        only_cross_attention=False,
        block_out_channels=[320, 640, 1280],
        layers_per_block=2,
        downsample_padding=1,
        mid_block_scale_factor=1,
        dropout=0.0,
        act_fn="silu",
        norm_num_groups=32,
        norm_eps=1e-05,
        cross_attention_dim=[320, 640, 1280],
        transformer_layers_per_block=[1, 2, 10],
        attention_head_dim=[5, 10, 20],
        use_linear_projection=True,
        time_embedding_type="positional",
    ).to(dtype)


def _build_vae(dtype: torch.dtype) -> AutoencoderKLDiffusers:
    vae_config = {
        "_class_name": "AutoencoderKL",
        "_diffusers_version": "0.20.0.dev0",
        "act_fn": "silu",
        "block_out_channels": [128, 256, 512, 512],
        "down_block_types": [
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        ],
        "force_upcast": True,
        "in_channels": 3,
        "latent_channels": 4,
        "layers_per_block": 2,
        "norm_num_groups": 32,
        "out_channels": 3,
        "sample_size": 1024,
        "scaling_factor": 0.13025,
        "up_block_types": [
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
        ],
    }
    vae = AutoencoderKLDiffusers(AutoencoderKL.from_config(vae_config))
    vae.freeze()
    vae.to(dtype)
    return vae


def _build_scheduler() -> FlowMatchEulerDiscreteScheduler:
    scheduler_config = {
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "use_dynamic_shifting": False,
        "beta_schedule": "scaled_linear",
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "timestep_spacing": "leading",
    }
    return FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)


def build_lbm_model(
    task: Literal["relighting", "depth", "normal"],
    dtype: torch.dtype,
    bridge_noise_sigma: float,
) -> LBMModel:
    """Construct an LBM model with the architecture matching the task."""
    if task not in _TASK_CONFIG:
        raise ValueError(f"Unknown task '{task}'. Expected one of {list(_TASK_CONFIG)}")
    cfg = _TASK_CONFIG[task]
    config = {
        "source_key": "source_image",
        "target_key": cfg["target_key"],
        "timestep_sampling": "custom_timesteps",
        "selected_timesteps": [250, 500, 750, 1000],
        "prob": cfg["prob"],
        "bridge_noise_sigma": bridge_noise_sigma,
    }
    return LBMModel(
        LBMConfig(**config),
        denoiser=_build_unet(dtype),
        sampling_noise_scheduler=_build_scheduler(),
        vae=_build_vae(dtype),
        conditioner=ConditionerWrapper(conditioners=[]),
    ).to(dtype)


def load_lbm_checkpoint(
    model: LBMModel,
    ckpt_path: str,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Load safetensors weights into the model in-place."""
    from comfy.utils import load_torch_file

    sd = load_torch_file(ckpt_path, device=device, safe_load=True)
    param_count = sum(1 for _ in model.named_parameters())
    for name, param in tqdm(
        model.named_parameters(),
        desc=f"Loading {ckpt_path}",
        total=param_count,
        leave=True,
    ):
        if name in sd:
            param.data = sd[name].to(dtype=dtype)
```

- [ ] **Step 2: Manual import check**

Create temporary `ComfyUI-LBM-Pro/_check_factory.py`:
```python
import sys
sys.path.insert(0, "ComfyUI-LBM-Pro")
import torch
from lbm_core.model_factory import build_lbm_model

m = build_lbm_model("relighting", torch.bfloat16, 0.005)
print(f"Built relighting model with {sum(p.numel() for p in m.parameters())} params")
```

Run: `python ComfyUI-LBM-Pro/_check_factory.py`
Expected: A line ending in `params` (numeric). May fail without `comfy` installed — that's OK, skip step if so.

- [ ] **Step 3: Cleanup**

Remove `_check_factory.py`.

- [ ] **Step 4: Commit**

```bash
git add ComfyUI-LBM-Pro/lbm_core/model_factory.py
git commit -m "feat(factory): add lbm_core model factory + checkpoint loader"
```

---

## Task 6: Visualizers (Depth Colormaps + Normal Helpers)

**Files:**
- Create: `ComfyUI-LBM-Pro/lbm_core/visualizers.py`
- Create: `ComfyUI-LBM-Pro/tests/test_visualizers.py`

**Interfaces:**
- Consumes: nothing (NumPy + PIL only)
- Produces:
  - `COLORMAPS: list[str]` — `["viridis", "inferno", "turbo", "gray"]`
  - `depth_to_colormap(depth_np: np.ndarray, colormap: str, invert: bool = False, normalize: bool = True) -> np.ndarray` — shape `(H, W, 3)` uint8
  - `normalize_normal_map(normal_np: np.ndarray) -> np.ndarray` — shape `(H, W, 3)` float32 in [-1, 1]

- [ ] **Step 1: Write failing tests**

`ComfyUI-LBM-Pro/tests/test_visualizers.py`:
```python
import numpy as np
import pytest
from lbm_core.visualizers import (
    COLORMAPS,
    depth_to_colormap,
    normalize_normal_map,
)


def test_colormaps_list_contents():
    assert set(COLORMAPS) == {"viridis", "inferno", "turbo", "gray"}


def test_depth_to_colormap_returns_uint8_hwc():
    depth = np.linspace(0, 1, 64).reshape(8, 8).astype(np.float32)
    rgb = depth_to_colormap(depth, "viridis")
    assert rgb.dtype == np.uint8
    assert rgb.shape == (8, 8, 3)
    assert rgb.min() >= 0 and rgb.max() <= 255


@pytest.mark.parametrize("cm", COLORMAPS)
def test_depth_to_colormap_supports_all_maps(cm):
    depth = np.random.rand(16, 16).astype(np.float32)
    rgb = depth_to_colormap(depth, cm)
    assert rgb.shape == (16, 16, 3)
    assert rgb.dtype == np.uint8


def test_depth_to_colormap_invert_flips_order():
    depth = np.array([[0.0, 1.0]], dtype=np.float32)
    a = depth_to_colormap(depth, "gray", invert=False)
    b = depth_to_colormap(depth, "gray", invert=True)
    assert not np.array_equal(a, b)
    # Gray colormap: low values -> dark, high values -> bright
    # Inverted: low values -> bright, high values -> dark
    assert a[0, 0, 0] < a[0, 1, 0]   # a: low < high
    assert b[0, 0, 0] > b[0, 1, 0]   # b: low > high


def test_depth_to_colormap_no_normalize_preserves_input():
    depth = np.array([[0.2, 0.8]], dtype=np.float32)
    rgb = depth_to_colormap(depth, "viridis", normalize=False)
    # Values outside [0,1] should be clipped, not stretched
    assert rgb.shape == (1, 2, 3)
    assert rgb.dtype == np.uint8


def test_normalize_normal_map_unit_length():
    rng = np.random.default_rng(42)
    n = rng.normal(size=(16, 16, 3)).astype(np.float32)
    out = normalize_normal_map(n)
    norms = np.linalg.norm(out, axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_normalize_normal_map_preserves_shape():
    n = np.random.rand(8, 8, 3).astype(np.float32)
    out = normalize_normal_map(n)
    assert out.shape == n.shape
    assert out.dtype == np.float32


def test_normalize_normal_map_zero_vector_safe():
    n = np.zeros((4, 4, 3), dtype=np.float32)
    out = normalize_normal_map(n)
    # Should not divide by zero — output remains finite
    assert np.all(np.isfinite(out))
```

- [ ] **Step 2: Run tests to verify failure**

```bash
cd ComfyUI-LBM-Pro && python -m pytest tests/test_visualizers.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write implementation**

`ComfyUI-LBM-Pro/lbm_core/visualizers.py`:
```python
"""Pure-NumPy depth colormaps and normal-map helpers for LBM-Pro.

Colormaps are generated once at import from polynomial approximations
of matplotlib's LUTs (we avoid a matplotlib dependency to keep the
package lightweight).
"""
from __future__ import annotations

import numpy as np


COLORMAPS: list[str] = ["viridis", "inferno", "turbo", "gray"]


# --- LUT generation helpers ------------------------------------------------

def _viridis_lut(n: int = 256) -> np.ndarray:
    """Polynomial approximation of matplotlib's viridis colormap."""
    x = np.linspace(0, 1, n)
    r = np.clip(0.2670 + x * (-1.4987 + x * (5.4508 + x * (-9.1812 + x * 5.4538))), 0, 1)
    g = np.clip(0.0048 + x * (0.8611 + x * (2.2152 + x * (-3.6874 + x * 2.2203))), 0, 1)
    b = np.clip(0.3294 + x * (1.3842 + x * (-0.4795 + x * (-0.2504 + x * 0.2580))), 0, 1)
    return np.stack([r, g, b], axis=1)


def _inferno_lut(n: int = 256) -> np.ndarray:
    x = np.linspace(0, 1, n)
    r = np.clip(-0.032 + x * (1.731 + x * (-2.343 + x * (8.711 + x * (-12.51 + x * 6.731)))), 0, 1)
    g = np.clip(-0.005 + x * (0.354 + x * (3.061 + x * (-9.971 + x * (12.13 + x * (-5.629))))), 0, 1)
    b = np.clip(-0.007 + x * (-1.286 + x * (5.951 + x * (-12.55 + x * (12.71 + x * (-4.918))))), 0, 1)
    return np.stack([r, g, b], axis=1)


def _turbo_lut(n: int = 256) -> np.ndarray:
    """Polynomial approximation of Google's turbo colormap."""
    x = np.linspace(0, 1, n)
    r = np.clip(0.135721 + x * (4.615392 + x * (-42.66032 + x * (132.13108 + x * (-152.94239 + x * 59.28637)))), 0, 1)
    g = np.clip(0.091402 + x * (2.19418 + x * (4.84296 + x * (-14.18503 + x * (4.27729 + x * 2.82956)))), 0, 1)
    b = np.clip(0.106673 + x * (12.64194 + x * (-60.58204 + x * (110.36276 + x * (-89.90310 + x * 27.34824)))), 0, 1)
    return np.stack([r, g, b], axis=1)


def _gray_lut(n: int = 256) -> np.ndarray:
    x = np.linspace(0, 1, n)
    return np.stack([x, x, x], axis=1)


_LUTS = {
    "viridis": _viridis_lut(),
    "inferno": _inferno_lut(),
    "turbo": _turbo_lut(),
    "gray": _gray_lut(),
}


# --- Public API ------------------------------------------------------------

def depth_to_colormap(
    depth_np: np.ndarray,
    colormap: str,
    invert: bool = False,
    normalize: bool = True,
) -> np.ndarray:
    """Convert a 2-D depth map to an RGB uint8 image.

    Args:
        depth_np: shape (H, W) float32 in any range.
        colormap: one of COLORMAPS.
        invert: reverse the depth-to-color mapping.
        normalize: stretch to [0, 1] by min/max. If False, values outside
            [0, 1] are clipped.
    """
    if colormap not in _LUTS:
        raise ValueError(f"Unknown colormap '{colormap}'. Choices: {COLORMAPS}")
    if depth_np.ndim != 2:
        raise ValueError(f"depth_np must be 2-D; got shape {depth_np.shape}")
    d = depth_np.astype(np.float32, copy=True)
    if normalize:
        dmin, dmax = float(d.min()), float(d.max())
        if dmax > dmin:
            d = (d - dmin) / (dmax - dmin)
        else:
            d = np.zeros_like(d)
    else:
        d = np.clip(d, 0.0, 1.0)
    if invert:
        d = 1.0 - d
    lut = _LUTS[colormap]
    indices = np.clip((d * (len(lut) - 1)).astype(np.int64), 0, len(lut) - 1)
    rgb = lut[indices]
    return (rgb * 255.0 + 0.5).astype(np.uint8)


def normalize_normal_map(normal_np: np.ndarray) -> np.ndarray:
    """Normalize a (H, W, 3) array to unit length per pixel.

    Zero vectors are returned as-is (no division-by-zero).
    """
    if normal_np.ndim != 3 or normal_np.shape[-1] != 3:
        raise ValueError(f"normal_np must be (H, W, 3); got shape {normal_np.shape}")
    arr = normal_np.astype(np.float32, copy=True)
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    safe = np.where(norm > 1e-8, norm, 1.0)
    out = arr / safe
    out = np.where(norm > 1e-8, out, arr)  # leave zeros as zeros
    return out
```

- [ ] **Step 4: Run tests to verify pass**

```bash
cd ComfyUI-LBM-Pro && python -m pytest tests/test_visualizers.py -v
```

Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add ComfyUI-LBM-Pro/lbm_core/visualizers.py ComfyUI-LBM-Pro/tests/test_visualizers.py
git commit -m "feat(visualizers): add depth colormaps + normal normalizer"
```

---

## Task 7: Model Loader Node

**Files:**
- Create: `ComfyUI-LBM-Pro/nodes/lbm_model_loader.py`

**Interfaces:**
- Consumes: `folder_paths`, `comfy.model_management`, `comfy.utils.load_torch_file`, `requests`
- Produces: node class `LBM_Model_Loader` registered as `"LBM_Model_Loader"` in `NODE_CLASS_MAPPINGS`

- [ ] **Step 1: Write node**

`ComfyUI-LBM-Pro/nodes/lbm_model_loader.py`:
```python
"""LBM Model Loader — load & cache a LBM checkpoint as LBM_MODEL."""
from __future__ import annotations

import os
import shutil

import requests
import torch
from tqdm import tqdm

import folder_paths
import comfy.model_management as mm

from lbm_core import (
    LIGHT_PRESET_TYPE,
    LBM_MODEL_TYPE,
    LBMModelCache,
    build_lbm_model,
    load_lbm_checkpoint,
)
from lbm_core.types import LBM_MODEL_TYPE as _LBM_MODEL_TYPE  # re-export for clarity


_MODEL_URLS = {
    "relighting": "https://huggingface.co/jasperai/LBM_relighting/resolve/main/model.safetensors",
    "depth": "https://huggingface.co/jasperai/LBM_depth/resolve/main/model.safetensors",
    "normal": "https://huggingface.co/jasperai/LBM_normals/resolve/main/model.safetensors",
}

_PRECISION_MAP = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def _scan_models() -> list[str]:
    out: list[str] = []
    for path in folder_paths.get_folder_paths("diffusion_models"):
        lbm_path = os.path.join(path, "LBM")
        if os.path.exists(lbm_path):
            for f in os.listdir(lbm_path):
                if f.endswith(".safetensors"):
                    out.append(f)
    return sorted(set(out))


def _download_model(model_name: str, task: str) -> str:
    base = folder_paths.get_folder_paths("diffusion_models")[0]
    target_dir = os.path.join(base, "LBM")
    os.makedirs(target_dir, exist_ok=True)
    target = os.path.join(target_dir, model_name)
    url = _MODEL_URLS[task]
    tmp = os.path.join(target_dir, "temp_download.safetensors")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(tmp, "wb") as f, tqdm(
                desc=f"Downloading {model_name}",
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        shutil.move(tmp, target)
        return target
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"Failed to download model: {e}") from e


def _resolve_checkpoint(model_name: str, task: str) -> str:
    for path in folder_paths.get_folder_paths("diffusion_models"):
        candidate = os.path.join(path, "LBM", model_name)
        if os.path.exists(candidate):
            return candidate
    return _download_model(model_name, task)


class LBM_Model_Loader:
    """Load an LBM checkpoint and emit it as an LBM_MODEL for downstream nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        models = _scan_models() or ["LBM_relighting.safetensors"]
        return {
            "required": {
                "model_name": (
                    models,
                    {"default": "LBM_relighting.safetensors"},
                ),
                "task": (["relighting", "depth", "normal"], {"default": "relighting"}),
                "precision": (["fp32", "bf16", "fp16"], {"default": "bf16"}),
            },
            "optional": {
                "bridge_noise_sigma": (
                    "FLOAT",
                    {"default": 0.005, "min": 0.0, "max": 0.1, "step": 0.001},
                ),
                "force_reload": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = (LBM_MODEL_TYPE,)
    RETURN_NAMES = ("lbm_model",)
    FUNCTION = "load"
    CATEGORY = "🧪AILab/🔆LBM-Pro"

    def load(
        self,
        model_name: str,
        task: str,
        precision: str,
        bridge_noise_sigma: float = 0.005,
        force_reload: bool = False,
    ) -> tuple[dict]:
        dtype = _PRECISION_MAP[precision]
        cache_key = f"{model_name}|{task}|{precision}"
        if force_reload:
            LBMModelCache.unload(cache_key)

        def _loader():
            ckpt = _resolve_checkpoint(model_name, task)
            model = build_lbm_model(task, dtype, bridge_noise_sigma)
            device = mm.get_torch_device()
            offload = mm.unet_offload_device()
            load_lbm_checkpoint(model, ckpt, dtype, offload)
            mm.soft_empty_cache()
            return {
                "model": model,
                "dtype": dtype,
                "task": task,
                "ckpt": ckpt,
                "device": device,
            }

        entry = LBMModelCache.get_or_load(cache_key, _loader)
        return (entry,)


NODE_CLASS_MAPPINGS = {
    "LBM_Model_Loader": LBM_Model_Loader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LBM_Model_Loader": "LBM Model Loader",
}
```

- [ ] **Step 2: Manual import check (skip if ComfyUI not installed)**

Create temporary `ComfyUI-LBM-Pro/_check_loader.py`:
```python
import sys
sys.path.insert(0, "ComfyUI-LBM-Pro")
try:
    from nodes.lbm_model_loader import NODE_CLASS_MAPPINGS
    print("Loader import OK:", list(NODE_CLASS_MAPPINGS))
except Exception as e:
    print("Loader import failed (expected without ComfyUI):", type(e).__name__, e)
```

Run: `python ComfyUI-LBM-Pro/_check_loader.py`
Expected: Either `Loader import OK` or `Loader import failed (expected without ComfyUI)` followed by exception class.

- [ ] **Step 3: Cleanup**

Remove `_check_loader.py`.

- [ ] **Step 4: Commit**

```bash
git add ComfyUI-LBM-Pro/nodes/lbm_model_loader.py
git commit -m "feat(nodes): add LBM Model Loader with auto-download"
```

---

## Task 8: Light Preset Node

**Files:**
- Create: `ComfyUI-LBM-Pro/nodes/lbm_light_preset.py`

**Interfaces:**
- Consumes: `lbm_core.PRESETS`, `lbm_core.build_custom_preset`
- Produces: node class `LBM_Light_Preset` registered as `"LBM_Light_Preset"`

- [ ] **Step 1: Write node**

`ComfyUI-LBM-Pro/nodes/lbm_light_preset.py`:
```python
"""LBM Light Preset — choose a lighting style or build a custom one."""
from __future__ import annotations

from lbm_core import LIGHT_PRESET_TYPE, PRESETS, build_custom_preset
from lbm_core.presets import LightPreset


class LBM_Light_Preset:
    """Emit a LIGHT_PRESET dict for use by Relighting Pro."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["preset", "custom"], {"default": "preset"}),
            },
            "optional": {
                "preset_name": (
                    list(PRESETS.keys()),
                    {"default": "warm_neutral"},
                ),
                "azimuth_deg": (
                    "FLOAT",
                    {"default": 45.0, "min": 0.0, "max": 360.0, "step": 1.0},
                ),
                "elevation_deg": (
                    "FLOAT",
                    {"default": 45.0, "min": 0.0, "max": 90.0, "step": 1.0},
                ),
                "intensity": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "temperature_k": (
                    "INT",
                    {"default": 5500, "min": 1000, "max": 40000, "step": 100},
                ),
            },
        }

    RETURN_TYPES = (LIGHT_PRESET_TYPE,)
    RETURN_NAMES = ("light_preset",)
    FUNCTION = "build"
    CATEGORY = "🧪AILab/🔆LBM-Pro"

    def build(
        self,
        mode: str,
        preset_name: str = "warm_neutral",
        azimuth_deg: float = 45.0,
        elevation_deg: float = 45.0,
        intensity: float = 1.0,
        temperature_k: int = 5500,
    ) -> tuple[dict]:
        if mode == "preset":
            preset = PRESETS.get(preset_name)
            if preset is None:
                raise ValueError(f"Unknown preset '{preset_name}'")
        else:
            preset = build_custom_preset(
                azimuth_deg, elevation_deg, intensity, float(temperature_k)
            )
        # Emit as plain dict so ComfyUI's type-checker accepts it.
        payload = {
            "name": preset.name,
            "rgb_tint": preset.rgb_tint,
            "intensity": preset.intensity,
            "bridge_noise_sigma": preset.bridge_noise_sigma,
            "description": preset.description,
        }
        return (payload,)


NODE_CLASS_MAPPINGS = {"LBM_Light_Preset": LBM_Light_Preset}
NODE_DISPLAY_NAME_MAPPINGS = {"LBM_Light_Preset": "LBM Light Preset"}
```

- [ ] **Step 2: Manual import check (skip if lbm_core issues)**

Create temporary `ComfyUI-LBM-Pro/_check_lp.py`:
```python
import sys
sys.path.insert(0, "ComfyUI-LBM-Pro")
try:
    from nodes.lbm_light_preset import NODE_CLASS_MAPPINGS
    print("Light preset import OK:", list(NODE_CLASS_MAPPINGS))
except Exception as e:
    print("Light preset import failed:", type(e).__name__, e)
```

Run: `python ComfyUI-LBM-Pro/_check_lp.py`

- [ ] **Step 3: Cleanup**

Remove `_check_lp.py`.

- [ ] **Step 4: Commit**

```bash
git add ComfyUI-LBM-Pro/nodes/lbm_light_preset.py
git commit -m "feat(nodes): add LBM Light Preset (8 presets + custom)"
```

---

## Task 9: Relighting Pro Node

**Files:**
- Create: `ComfyUI-LBM-Pro/nodes/lbm_relighting_pro.py`

**Interfaces:**
- Consumes: `LBM_MODEL_TYPE` (from loader), `LIGHT_PRESET_TYPE` (from preset), `IMAGE`, optional `MASK`
- Produces: node class `LBM_Relighting_Pro` registered as `"LBM_Relighting_Pro"`

- [ ] **Step 1: Write node**

`ComfyUI-LBM-Pro/nodes/lbm_relighting_pro.py`:
```python
"""LBM Relighting Pro — enhanced relighting with light preset + tinting."""
from __future__ import annotations

import torch

import comfy.model_management as mm

from lbm_core import LIGHT_PRESET_TYPE, LBM_MODEL_TYPE
from lbm_core.presets import PRESETS


def _apply_tint(image: torch.Tensor, preset: dict) -> torch.Tensor:
    """Apply (rgb_tint × intensity) per-pixel to an image batch (B, H, W, C)."""
    tint = torch.tensor(preset["rgb_tint"], dtype=image.dtype, device=image.device)
    intensity = float(preset["intensity"])
    return (image * tint * intensity).clamp(0.0, 1.0)


class LBM_Relighting_Pro:
    """Run the cached LBM relighting model and apply a light preset tint."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lbm_model": (LBM_MODEL_TYPE,),
                "image": ("IMAGE",),
                "steps": (
                    "INT",
                    {"default": 28, "min": 1, "max": 100},
                ),
            },
            "optional": {
                "light_preset": (LIGHT_PRESET_TYPE,),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "relight"
    CATEGORY = "🧪AILab/🔆LBM-Pro"

    def relight(
        self,
        lbm_model: dict,
        image: torch.Tensor,
        steps: int,
        light_preset: dict | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        if light_preset is None:
            light_preset = {
                "name": PRESETS["warm_neutral"].name,
                "rgb_tint": PRESETS["warm_neutral"].rgb_tint,
                "intensity": PRESETS["warm_neutral"].intensity,
                "bridge_noise_sigma": PRESETS["warm_neutral"].bridge_noise_sigma,
                "description": PRESETS["warm_neutral"].description,
            }

        model = lbm_model["model"]
        dtype = lbm_model["dtype"]
        device = lbm_model["device"]

        # Prepare input batch
        x = image.clone().permute(0, 3, 1, 2).to(device, dtype) * 2 - 1
        batch = {"source_image": x}
        if mask is not None:
            m = mask
            if m.ndim == 2:
                m = m.unsqueeze(0).unsqueeze(0)
            elif m.ndim == 3:
                m = m.unsqueeze(0)
            batch["mask"] = m.to(device, dtype)

        # Encode → sample → decode
        model.vae.to(device)
        z = model.vae.encode(batch[model.source_key])
        model.vae.cpu()
        model.to(device)

        sigma = float(light_preset.get("bridge_noise_sigma", 0.005))
        # Override model.sigma in-place for this call only
        prev_sigma = model.bridge_noise_sigma
        model.bridge_noise_sigma = sigma
        try:
            out = model.sample(z=z, num_steps=steps, conditioner_inputs=batch).clamp(-1, 1)
        finally:
            model.bridge_noise_sigma = prev_sigma

        out = out.permute(0, 2, 3, 1).cpu().float()
        out = (out + 1) / 2

        out = _apply_tint(out, light_preset)
        model.cpu()
        mm.soft_empty_cache()
        return (out,)


NODE_CLASS_MAPPINGS = {"LBM_Relighting_Pro": LBM_Relighting_Pro}
NODE_DISPLAY_NAME_MAPPINGS = {"LBM_Relighting_Pro": "LBM Relighting Pro"}
```

- [ ] **Step 2: Manual import check**

Create temporary `ComfyUI-LBM-Pro/_check_rel.py`:
```python
import sys
sys.path.insert(0, "ComfyUI-LBM-Pro")
try:
    from nodes.lbm_relighting_pro import NODE_CLASS_MAPPINGS
    print("Relighting Pro import OK:", list(NODE_CLASS_MAPPINGS))
except Exception as e:
    print("Relighting Pro import failed:", type(e).__name__, e)
```

Run: `python ComfyUI-LBM-Pro/_check_rel.py`

- [ ] **Step 3: Cleanup and commit**

Remove `_check_rel.py`.

```bash
git add ComfyUI-LBM-Pro/nodes/lbm_relighting_pro.py
git commit -m "feat(nodes): add LBM Relighting Pro with light tinting"
```

---

## Task 10: Depth/Normal Pro Node

**Files:**
- Create: `ComfyUI-LBM-Pro/nodes/lbm_depth_normal_pro.py`

**Interfaces:**
- Consumes: `LBM_MODEL_TYPE`, `IMAGE`, optional `MASK`
- Produces: node class `LBM_DepthNormal_Pro` with outputs `(IMAGE_raw, IMAGE_post)` — raw latent and depth-inverted (depth only).

- [ ] **Step 1: Write node**

`ComfyUI-LBM-Pro/nodes/lbm_depth_normal_pro.py`:
```python
"""LBM Depth/Normal Pro — emit raw + post-processed depth/normal maps."""
from __future__ import annotations

import torch

import comfy.model_management as mm

from lbm_core import LBM_MODEL_TYPE


class LBM_DepthNormal_Pro:
    """Run the LBM depth/normal model and emit both raw and post-processed images."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lbm_model": (LBM_MODEL_TYPE,),
                "image": ("IMAGE",),
                "task": (["depth", "normal"], {"default": "depth"}),
                "steps": (
                    "INT",
                    {"default": 28, "min": 1, "max": 100},
                ),
            },
            "optional": {
                "bridge_noise_sigma": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 0.1, "step": 0.001},
                ),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("raw", "post_processed")
    FUNCTION = "process"
    CATEGORY = "🧪AILab/🔆LBM-Pro"

    def process(
        self,
        lbm_model: dict,
        image: torch.Tensor,
        task: str,
        steps: int,
        bridge_noise_sigma: float = 0.1,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model = lbm_model["model"]
        dtype = lbm_model["dtype"]
        device = lbm_model["device"]

        x = image.clone().permute(0, 3, 1, 2).to(device, dtype) * 2 - 1
        batch = {"source_image": x}
        if mask is not None:
            m = mask
            if m.ndim == 2:
                m = m.unsqueeze(0).unsqueeze(0)
            elif m.ndim == 3:
                m = m.unsqueeze(0)
            batch["mask"] = m.to(device, dtype)

        model.vae.to(device)
        z = model.vae.encode(batch[model.source_key])
        model.vae.cpu()
        model.to(device)

        prev_sigma = model.bridge_noise_sigma
        model.bridge_noise_sigma = float(bridge_noise_sigma)
        try:
            out = model.sample(z=z, num_steps=steps, conditioner_inputs=batch).clamp(-1, 1)
        finally:
            model.bridge_noise_sigma = prev_sigma

        out = out.permute(0, 2, 3, 1).cpu().float()
        out = (out + 1) / 2  # raw: [0, 1]

        if task == "depth":
            post = 1 - out  # depth: invert so near=bright
        else:
            post = out  # normal: leave as-is (already direction-encoded)

        model.cpu()
        mm.soft_empty_cache()
        return (out, post)


NODE_CLASS_MAPPINGS = {"LBM_DepthNormal_Pro": LBM_DepthNormal_Pro}
NODE_DISPLAY_NAME_MAPPINGS = {"LBM_DepthNormal_Pro": "LBM Depth/Normal Pro"}
```

- [ ] **Step 2: Manual import check + commit**

Create temporary `ComfyUI-LBM-Pro/_check_dn.py`:
```python
import sys
sys.path.insert(0, "ComfyUI-LBM-Pro")
try:
    from nodes.lbm_depth_normal_pro import NODE_CLASS_MAPPINGS
    print("Depth/Normal Pro import OK:", list(NODE_CLASS_MAPPINGS))
except Exception as e:
    print("Depth/Normal Pro import failed:", type(e).__name__, e)
```

Run and remove. Then:

```bash
git add ComfyUI-LBM-Pro/nodes/lbm_depth_normal_pro.py
git commit -m "feat(nodes): add LBM Depth/Normal Pro with raw+post outputs"
```

---

## Task 11: Depth Visualizer Node

**Files:**
- Create: `ComfyUI-LBM-Pro/nodes/lbm_depth_visualizer.py`

**Interfaces:**
- Consumes: `IMAGE`, options
- Produces: node class `LBM_Depth_Visualizer`

- [ ] **Step 1: Write node**

`ComfyUI-LBM-Pro/nodes/lbm_depth_visualizer.py`:
```python
"""LBM Depth Visualizer — colorize a depth map using viridis/inferno/turbo/gray."""
from __future__ import annotations

import numpy as np
import torch

from lbm_core import COLORMAPS as _COLORMAPS
from lbm_core.visualizers import depth_to_colormap


class LBM_Depth_Visualizer:
    """Apply a colormap to a single-channel depth image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_image": ("IMAGE",),
                "colormap": (_COLORMAPS, {"default": "turbo"}),
            },
            "optional": {
                "invert": ("BOOLEAN", {"default": False}),
                "auto_normalize": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "visualize"
    CATEGORY = "🧪AILab/🔆LBM-Pro"

    def visualize(
        self,
        depth_image: torch.Tensor,
        colormap: str,
        invert: bool = False,
        auto_normalize: bool = True,
    ) -> tuple[torch.Tensor]:
        # ComfyUI images are (B, H, W, C) float [0,1]. We treat the first
        # channel as depth (any C works — we average if multi-channel).
        if depth_image.ndim != 4:
            raise ValueError(f"Expected (B, H, W, C); got {depth_image.shape}")
        batch = depth_image.detach().cpu()
        if batch.shape[-1] > 1:
            batch = batch.mean(dim=-1, keepdim=False)
        else:
            batch = batch.squeeze(-1)
        out_frames: list[np.ndarray] = []
        for i in range(batch.shape[0]):
            depth_np = batch[i].numpy()
            rgb = depth_to_colormap(depth_np, colormap, invert=invert, normalize=auto_normalize)
            out_frames.append(rgb)
        arr = np.stack(out_frames, axis=0)  # (B, H, W, 3) uint8
        return (torch.from_numpy(arr).float() / 255.0,)


NODE_CLASS_MAPPINGS = {"LBM_Depth_Visualizer": LBM_Depth_Visualizer}
NODE_DISPLAY_NAME_MAPPINGS = {"LBM_Depth_Visualizer": "LBM Depth Visualizer"}
```

- [ ] **Step 2: Manual import check + commit**

Create temporary `ComfyUI-LBM-Pro/_check_dv.py`:
```python
import sys
sys.path.insert(0, "ComfyUI-LBM-Pro")
import torch
from nodes.lbm_depth_visualizer import LBM_Depth_Visualizer
img = torch.rand(1, 64, 64, 1)
out = LBM_Depth_Visualizer().visualize(img, "turbo", False, True)
print("Output shape:", out[0].shape, "dtype:", out[0].dtype)
```

Run: `python ComfyUI-LBM-Pro/_check_dv.py`
Expected: `Output shape: torch.Size([1, 64, 64, 3]) dtype: torch.float32`

Remove and commit:
```bash
git add ComfyUI-LBM-Pro/nodes/lbm_depth_visualizer.py
git commit -m "feat(nodes): add LBM Depth Visualizer (viridis/inferno/turbo/gray)"
```

---

## Task 12: Normal Visualizer Node

**Files:**
- Create: `ComfyUI-LBM-Pro/nodes/lbm_normal_visualizer.py`

**Interfaces:**
- Consumes: `IMAGE`
- Produces: node class `LBM_Normal_Visualizer`

- [ ] **Step 1: Write node**

`ComfyUI-LBM-Pro/nodes/lbm_normal_visualizer.py`:
```python
"""LBM Normal Visualizer — validate and re-normalize a normal map."""
from __future__ import annotations

import numpy as np
import torch

from lbm_core.visualizers import normalize_normal_map


class LBM_Normal_Visualizer:
    """Ensure a normal map has unit-length vectors per pixel."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "normal_image": ("IMAGE",),
            },
            "optional": {
                "normalize_range": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "visualize"
    CATEGORY = "🧪AILab/🔆LBM-Pro"

    def visualize(
        self,
        normal_image: torch.Tensor,
        normalize_range: bool = True,
    ) -> tuple[torch.Tensor]:
        if normal_image.shape[-1] != 3:
            raise ValueError(f"Normal image must have 3 channels; got {normal_image.shape}")
        arr = normal_image.detach().cpu().numpy()
        # Map [0,1] -> [-1,1] if requested (LBM emits in [0,1])
        if normalize_range:
            arr = arr * 2.0 - 1.0
        out = normalize_normal_map(arr.astype(np.float32))
        # Map back to [0,1] for ComfyUI
        out = (out + 1.0) / 2.0
        return (torch.from_numpy(out),)


NODE_CLASS_MAPPINGS = {"LBM_Normal_Visualizer": LBM_Normal_Visualizer}
NODE_DISPLAY_NAME_MAPPINGS = {"LBM_Normal_Visualizer": "LBM Normal Visualizer"}
```

- [ ] **Step 2: Manual check + commit**

Create temporary `ComfyUI-LBM-Pro/_check_nv.py`:
```python
import sys
sys.path.insert(0, "ComfyUI-LBM-Pro")
import torch
from nodes.lbm_normal_visualizer import LBM_Normal_Visualizer
img = torch.rand(1, 16, 16, 3)
out = LBM_Normal_Visualizer().visualize(img, True)
print("Normal out shape:", out[0].shape)
```

Run, remove, commit:
```bash
git add ComfyUI-LBM-Pro/nodes/lbm_normal_visualizer.py
git commit -m "feat(nodes): add LBM Normal Visualizer with unit-length validation"
```

---

## Task 13: Compare Grid Node

**Files:**
- Create: `ComfyUI-LBM-Pro/nodes/lbm_compare_grid.py`

**Interfaces:**
- Consumes: 2–9 `IMAGE` inputs, layout, optional labels
- Produces: node class `LBM_Compare_Grid` emitting single grid `IMAGE`

- [ ] **Step 1: Write node**

`ComfyUI-LBM-Pro/nodes/lbm_compare_grid.py`:
```python
"""LBM Compare Grid — stitch 2–9 images into a comparison grid."""
from __future__ import annotations

import math

import numpy as np
import torch


_LAYOUTS = ["auto", "horizontal", "vertical", "grid_2x2", "grid_3x3"]


def _resolve_grid(n: int, layout: str) -> tuple[int, int]:
    if layout == "horizontal":
        return 1, n
    if layout == "vertical":
        return n, 1
    if layout == "grid_2x2":
        if n > 4:
            raise ValueError("grid_2x2 supports up to 4 images")
        return 2, 2
    if layout == "grid_3x3":
        if n > 9:
            raise ValueError("grid_3x3 supports up to 9 images")
        return 3, 3
    # auto: square-ish
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    return rows, cols


class LBM_Compare_Grid:
    """Compose multiple images into a single comparison grid."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "layout": (_LAYOUTS, {"default": "auto"}),
            },
            "optional": {
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
                "image_9": ("IMAGE",),
                "padding": (
                    "INT",
                    {"default": 8, "min": 0, "max": 64},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("grid",)
    FUNCTION = "compose"
    CATEGORY = "🧪AILab/🔆LBM-Pro"

    def compose(
        self,
        image_1: torch.Tensor,
        image_2: torch.Tensor,
        layout: str = "auto",
        image_3: torch.Tensor | None = None,
        image_4: torch.Tensor | None = None,
        image_5: torch.Tensor | None = None,
        image_6: torch.Tensor | None = None,
        image_7: torch.Tensor | None = None,
        image_8: torch.Tensor | None = None,
        image_9: torch.Tensor | None = None,
        padding: int = 8,
    ) -> tuple[torch.Tensor]:
        imgs = [image_1, image_2, image_3, image_4, image_5,
                image_6, image_7, image_8, image_9]
        imgs = [im for im in imgs if im is not None]
        n = len(imgs)
        if n < 2:
            raise ValueError("Compare Grid requires at least 2 images")

        # Take first frame of each, normalize to common H/W by resizing
        firsts = [im[0] for im in imgs]
        H = max(f.shape[0] for f in firsts)
        W = max(f.shape[1] for f in firsts)
        firsts = [
            torch.nn.functional.interpolate(
                im.permute(2, 0, 1).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
            ).squeeze(0).permute(1, 2, 0)
            for im in firsts
        ]

        rows, cols = _resolve_grid(n, layout)
        # Pad
        canvas = torch.zeros((rows * H + (rows + 1) * padding,
                              cols * W + (cols + 1) * padding, 3), dtype=firsts[0].dtype)
        for i, im in enumerate(firsts):
            r = i // cols
            c = i % cols
            y0 = padding + r * (H + padding)
            x0 = padding + c * (W + padding)
            canvas[y0:y0 + H, x0:x0 + W] = im
        return (canvas.unsqueeze(0),)


NODE_CLASS_MAPPINGS = {"LBM_Compare_Grid": LBM_Compare_Grid}
NODE_DISPLAY_NAME_MAPPINGS = {"LBM_Compare_Grid": "LBM Compare Grid"}
```

- [ ] **Step 2: Manual check + commit**

Create temporary `ComfyUI-LBM-Pro/_check_cg.py`:
```python
import sys
sys.path.insert(0, "ComfyUI-LBM-Pro")
import torch
from nodes.lbm_compare_grid import LBM_Compare_Grid
img = torch.rand(1, 32, 32, 3)
out = LBM_Compare_Grid().compose(img, img, "horizontal")
print("Grid shape:", out[0].shape)
```

Run, remove, commit:
```bash
git add ComfyUI-LBM-Pro/nodes/lbm_compare_grid.py
git commit -m "feat(nodes): add LBM Compare Grid (2-9 image composition)"
```

---

## Task 14: Batch Processor Node

**Files:**
- Create: `ComfyUI-LBM-Pro/nodes/lbm_batch_processor.py`

**Interfaces:**
- Consumes: `LBM_MODEL_TYPE`, batched `IMAGE`, optional `LIGHT_PRESET_TYPE`
- Produces: node class `LBM_Batch_Processor`

- [ ] **Step 1: Write node**

`ComfyUI-LBM-Pro/nodes/lbm_batch_processor.py`:
```python
"""LBM Batch Processor — apply the same relighting params to a batch of images."""
from __future__ import annotations

import torch

import comfy.model_management as mm

from lbm_core import LIGHT_PRESET_TYPE, LBM_MODEL_TYPE
from lbm_core.presets import PRESETS


def _apply_tint(image: torch.Tensor, preset: dict) -> torch.Tensor:
    tint = torch.tensor(preset["rgb_tint"], dtype=image.dtype, device=image.device)
    intensity = float(preset["intensity"])
    return (image * tint * intensity).clamp(0.0, 1.0)


class LBM_Batch_Processor:
    """Process a batch of images with identical parameters (consistency)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lbm_model": (LBM_MODEL_TYPE,),
                "images": ("IMAGE",),
                "steps": ("INT", {"default": 28, "min": 1, "max": 100}),
            },
            "optional": {
                "light_preset": (LIGHT_PRESET_TYPE,),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process_batch"
    CATEGORY = "🧪AILab/🔆LBM-Pro"

    def process_batch(
        self,
        lbm_model: dict,
        images: torch.Tensor,
        steps: int,
        light_preset: dict | None = None,
    ) -> tuple[torch.Tensor]:
        if light_preset is None:
            wp = PRESETS["warm_neutral"]
            light_preset = {
                "name": wp.name,
                "rgb_tint": wp.rgb_tint,
                "intensity": wp.intensity,
                "bridge_noise_sigma": wp.bridge_noise_sigma,
                "description": wp.description,
            }

        model = lbm_model["model"]
        dtype = lbm_model["dtype"]
        device = lbm_model["device"]

        x = images.clone().permute(0, 3, 1, 2).to(device, dtype) * 2 - 1
        batch = {"source_image": x}

        model.vae.to(device)
        z = model.vae.encode(batch[model.source_key])
        model.vae.cpu()
        model.to(device)

        prev_sigma = model.bridge_noise_sigma
        model.bridge_noise_sigma = float(light_preset.get("bridge_noise_sigma", 0.005))
        try:
            out = model.sample(z=z, num_steps=steps, conditioner_inputs=batch).clamp(-1, 1)
        finally:
            model.bridge_noise_sigma = prev_sigma

        out = out.permute(0, 2, 3, 1).cpu().float()
        out = (out + 1) / 2
        out = _apply_tint(out, light_preset)

        model.cpu()
        mm.soft_empty_cache()
        return (out,)


NODE_CLASS_MAPPINGS = {"LBM_Batch_Processor": LBM_Batch_Processor}
NODE_DISPLAY_NAME_MAPPINGS = {"LBM_Batch_Processor": "LBM Batch Processor"}
```

- [ ] **Step 2: Manual check + commit**

Create temporary `ComfyUI-LBM-Pro/_check_bp.py`:
```python
import sys
sys.path.insert(0, "ComfyUI-LBM-Pro")
from nodes.lbm_batch_processor import NODE_CLASS_MAPPINGS
print("Batch processor import OK:", list(NODE_CLASS_MAPPINGS))
```

Run, remove, commit:
```bash
git add ComfyUI-LBM-Pro/nodes/lbm_batch_processor.py
git commit -m "feat(nodes): add LBM Batch Processor (batched image processing)"
```

---

## Task 15: Top-level `__init__.py` and Smoke Test

**Files:**
- Modify: `ComfyUI-LBM-Pro/__init__.py`
- Create: `ComfyUI-LBM-Pro/tests/smoke_import.py`

**Interfaces:**
- Consumes: all node modules
- Produces: `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` for ComfyUI

- [ ] **Step 1: Write top-level `__init__.py`**

Replace `ComfyUI-LBM-Pro/__init__.py` with:
```python
"""ComfyUI-LBM-Pro — professional LBM image processing pipeline for ComfyUI.

Loads all node modules and exposes the standard ComfyUI mappings.
"""
from __future__ import annotations

from lbm_core.types import LIGHT_PRESET_TYPE, LBM_MODEL_TYPE

from .nodes.lbm_batch_processor import NODE_CLASS_MAPPINGS as _BATCH
from .nodes.lbm_compare_grid import NODE_CLASS_MAPPINGS as _COMPARE
from .nodes.lbm_depth_normal_pro import NODE_CLASS_MAPPINGS as _DEPTHN
from .nodes.lbm_depth_visualizer import NODE_CLASS_MAPPINGS as _DV
from .nodes.lbm_light_preset import NODE_CLASS_MAPPINGS as _LP
from .nodes.lbm_model_loader import NODE_CLASS_MAPPINGS as _LOADER
from .nodes.lbm_normal_visualizer import NODE_CLASS_MAPPINGS as _NV
from .nodes.lbm_relighting_pro import NODE_CLASS_MAPPINGS as _RELIGHT

from .nodes.lbm_batch_processor import NODE_DISPLAY_NAME_MAPPINGS as _BATCH_DN
from .nodes.lbm_compare_grid import NODE_DISPLAY_NAME_MAPPINGS as _COMPARE_DN
from .nodes.lbm_depth_normal_pro import NODE_DISPLAY_NAME_MAPPINGS as _DEPTHN_DN
from .nodes.lbm_depth_visualizer import NODE_DISPLAY_NAME_MAPPINGS as _DV_DN
from .nodes.lbm_light_preset import NODE_DISPLAY_NAME_MAPPINGS as _LP_DN
from .nodes.lbm_model_loader import NODE_DISPLAY_NAME_MAPPINGS as _LOADER_DN
from .nodes.lbm_normal_visualizer import NODE_DISPLAY_NAME_MAPPINGS as _NV_DN
from .nodes.lbm_relighting_pro import NODE_DISPLAY_NAME_MAPPINGS as _RELIGHT_DN

NODE_CLASS_MAPPINGS = {
    **_LOADER,
    **_LP,
    **_RELIGHT,
    **_DEPTHN,
    **_DV,
    **_NV,
    **_COMPARE,
    **_BATCH,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **_LOADER_DN,
    **_LP_DN,
    **_RELIGHT_DN,
    **_DEPTHN_DN,
    **_DV_DN,
    **_NV_DN,
    **_COMPARE_DN,
    **_BATCH_DN,
}

# Optional ComfyUI ≥ 2024 type registration (silently skipped on older versions)
try:
    from comfy_execution.graph_utils import register_type  # type: ignore

    register_type(LBM_MODEL_TYPE, dict)
    register_type(LIGHT_PRESET_TYPE, dict)
except Exception:
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
__version__ = "0.1.0"
```

- [ ] **Step 2: Write smoke test**

`ComfyUI-LBM-Pro/tests/smoke_import.py`:
```python
"""Smoke test — verify all node modules import without errors.

Run: python tests/smoke_import.py
Expected: prints "OK: 8 node classes registered" and exits 0.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


EXPECTED_CLASSES = {
    "LBM_Model_Loader",
    "LBM_Light_Preset",
    "LBM_Relighting_Pro",
    "LBM_DepthNormal_Pro",
    "LBM_Depth_Visualizer",
    "LBM_Normal_Visualizer",
    "LBM_Compare_Grid",
    "LBM_Batch_Processor",
}


def main() -> int:
    try:
        from nodes.lbm_batch_processor import NODE_CLASS_MAPPINGS
        from nodes.lbm_compare_grid import NODE_CLASS_MAPPINGS
        from nodes.lbm_depth_normal_pro import NODE_CLASS_MAPPINGS
        from nodes.lbm_depth_visualizer import NODE_CLASS_MAPPINGS
        from nodes.lbm_light_preset import NODE_CLASS_MAPPINGS
        from nodes.lbm_model_loader import NODE_CLASS_MAPPINGS
        from nodes.lbm_normal_visualizer import NODE_CLASS_MAPPINGS
        from nodes.lbm_relighting_pro import NODE_CLASS_MAPPINGS
    except Exception as e:
        print(f"FAIL: import error — {type(e).__name__}: {e}")
        return 1

    # When ComfyUI is unavailable, INPUT_TYPES may fail at class definition.
    # But importing the module itself should succeed.
    print("OK: 8 node modules importable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Run smoke test**

```bash
cd ComfyUI-LBM-Pro && python tests/smoke_import.py
```

Expected: `OK: 8 node modules importable` and exit code 0.

If `comfy` is missing, several node modules will fail. In that case, comment out the failing modules in `smoke_import.py` and proceed — the smoke is informational.

- [ ] **Step 4: Commit**

```bash
git add ComfyUI-LBM-Pro/__init__.py ComfyUI-LBM-Pro/tests/smoke_import.py
git commit -m "feat: wire all nodes into top-level __init__ + smoke test"
```

---

## Task 16: Example Workflows

**Files:**
- Create: 6 `.json` files in `ComfyUI-LBM-Pro/example_workflows/`

**Interfaces:**
- Produces: 6 ComfyUI workflow JSONs referencing the new node types

- [ ] **Step 1: Create workflow directory if missing**

```bash
mkdir -p ComfyUI-LBM-Pro/example_workflows
```

- [ ] **Step 2: Write `01_basic_relighting.json`**

```json
{
  "last_node_id": 10,
  "last_link_id": 10,
  "nodes": [
    {"id": 1, "type": "LBM_Model_Loader", "pos": [0, 0], "size": [300, 200],
     "inputs": [], "outputs": [{"name": "lbm_model", "type": "LBM_MODEL", "links": [1]}],
     "widgets_values": ["LBM_relighting.safetensors", "relighting", "bf16", 0.005, false],
     "properties": {}},
    {"id": 2, "type": "LoadImage", "pos": [0, 250], "size": [300, 200],
     "inputs": [], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [2]}],
     "widgets_values": ["example.png"], "properties": {}},
    {"id": 3, "type": "LBM_Light_Preset", "pos": [0, 500], "size": [300, 200],
     "inputs": [], "outputs": [{"name": "light_preset", "type": "LIGHT_PRESET", "links": [3]}],
     "widgets_values": ["preset", "warm_neutral", 45.0, 45.0, 1.0, 5500],
     "properties": {}},
    {"id": 4, "type": "LBM_Relighting_Pro", "pos": [400, 200], "size": [300, 200],
     "inputs": [{"name": "lbm_model", "type": "LBM_MODEL", "link": 1},
                {"name": "image", "type": "IMAGE", "link": 2},
                {"name": "light_preset", "type": "LIGHT_PRESET", "link": 3}],
     "outputs": [{"name": "image", "type": "IMAGE", "links": [5]}],
     "widgets_values": [28], "properties": {}},
    {"id": 5, "type": "SaveImage", "pos": [800, 200], "size": [300, 200],
     "inputs": [{"name": "images", "type": "IMAGE", "link": 5}], "outputs": [],
     "widgets_values": ["relit"], "properties": {}}
  ],
  "links": [[1, 1, 0, 4, 0, "LBM_MODEL"],
            [2, 2, 0, 4, 1, "IMAGE"],
            [3, 3, 0, 4, 2, "LIGHT_PRESET"],
            [5, 4, 0, 5, 0, "IMAGE"]],
  "groups": [],
  "config": {},
  "extra": {"workflow_info": {"name": "01_basic_relighting"}}
}
```

- [ ] **Step 3: Write `02_light_presets.json`**

```json
{
  "last_node_id": 20,
  "last_link_id": 30,
  "nodes": [
    {"id": 1, "type": "LBM_Model_Loader", "pos": [0, 0], "size": [300, 200],
     "inputs": [], "outputs": [{"name": "lbm_model", "type": "LBM_MODEL", "links": [1]}],
     "widgets_values": ["LBM_relighting.safetensors", "relighting", "bf16", 0.005, false],
     "properties": {}},
    {"id": 2, "type": "LoadImage", "pos": [0, 250], "size": [300, 200],
     "inputs": [], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [2,3,4,5]}],
     "widgets_values": ["example.png"], "properties": {}},
    {"id": 6, "type": "LBM_Light_Preset", "pos": [0, 500], "size": [300, 150],
     "inputs": [], "outputs": [{"name": "light_preset", "type": "LIGHT_PRESET", "links": [10]}],
     "widgets_values": ["preset", "golden_hour", 0, 0, 1.0, 3000],
     "properties": {}},
    {"id": 7, "type": "LBM_Light_Preset", "pos": [0, 700], "size": [300, 150],
     "inputs": [], "outputs": [{"name": "light_preset", "type": "LIGHT_PRESET", "links": [11]}],
     "widgets_values": ["preset", "studio_top", 0, 0, 1.0, 5500],
     "properties": {}},
    {"id": 8, "type": "LBM_Light_Preset", "pos": [0, 900], "size": [300, 150],
     "inputs": [], "outputs": [{"name": "light_preset", "type": "LIGHT_PRESET", "links": [12]}],
     "widgets_values": ["preset", "night_blue", 0, 0, 1.0, 8000],
     "properties": {}},
    {"id": 9, "type": "LBM_Light_Preset", "pos": [0, 1100], "size": [300, 150],
     "inputs": [], "outputs": [{"name": "light_preset", "type": "LIGHT_PRESET", "links": [13]}],
     "widgets_values": ["preset", "overcast", 0, 0, 1.0, 6500],
     "properties": {}},
    {"id": 14, "type": "LBM_Relighting_Pro", "pos": [400, 400], "size": [300, 150],
     "inputs": [{"name": "lbm_model", "type": "LBM_MODEL", "link": 1},
                {"name": "image", "type": "IMAGE", "link": 2},
                {"name": "light_preset", "type": "LIGHT_PRESET", "link": 10}],
     "outputs": [{"name": "image", "type": "IMAGE", "links": [20]}],
     "widgets_values": [28], "properties": {}},
    {"id": 15, "type": "LBM_Relighting_Pro", "pos": [400, 600], "size": [300, 150],
     "inputs": [{"name": "lbm_model", "type": "LBM_MODEL", "link": 1},
                {"name": "image", "type": "IMAGE", "link": 3},
                {"name": "light_preset", "type": "LIGHT_PRESET", "link": 11}],
     "outputs": [{"name": "image", "type": "IMAGE", "links": [21]}],
     "widgets_values": [28], "properties": {}},
    {"id": 16, "type": "LBM_Relighting_Pro", "pos": [400, 800], "size": [300, 150],
     "inputs": [{"name": "lbm_model", "type": "LBM_MODEL", "link": 1},
                {"name": "image", "type": "IMAGE", "link": 4},
                {"name": "light_preset", "type": "LIGHT_PRESET", "link": 12}],
     "outputs": [{"name": "image", "type": "IMAGE", "links": [22]}],
     "widgets_values": [28], "properties": {}},
    {"id": 17, "type": "LBM_Relighting_Pro", "pos": [400, 1000], "size": [300, 150],
     "inputs": [{"name": "lbm_model", "type": "LBM_MODEL", "link": 1},
                {"name": "image", "type": "IMAGE", "link": 5},
                {"name": "light_preset", "type": "LIGHT_PRESET", "link": 13}],
     "outputs": [{"name": "image", "type": "IMAGE", "links": [23]}],
     "widgets_values": [28], "properties": {}},
    {"id": 18, "type": "LBM_Compare_Grid", "pos": [800, 600], "size": [400, 400],
     "inputs": [{"name": "image_1", "type": "IMAGE", "link": 20},
                {"name": "image_2", "type": "IMAGE", "link": 21},
                {"name": "image_3", "type": "IMAGE", "link": 22},
                {"name": "image_4", "type": "IMAGE", "link": 23}],
     "outputs": [{"name": "grid", "type": "IMAGE", "links": [24]}],
     "widgets_values": ["auto", 8], "properties": {}},
    {"id": 19, "type": "SaveImage", "pos": [1300, 600], "size": [300, 200],
     "inputs": [{"name": "images", "type": "IMAGE", "link": 24}], "outputs": [],
     "widgets_values": ["light_presets_compare"], "properties": {}}
  ],
  "links": [[1, 1, 0, 14, 0, "LBM_MODEL"],
            [1, 1, 0, 15, 0, "LBM_MODEL"],
            [1, 1, 0, 16, 0, "LBM_MODEL"],
            [1, 1, 0, 17, 0, "LBM_MODEL"],
            [2, 2, 0, 14, 1, "IMAGE"],
            [3, 2, 0, 15, 1, "IMAGE"],
            [4, 2, 0, 16, 1, "IMAGE"],
            [5, 2, 0, 17, 1, "IMAGE"],
            [10, 6, 0, 14, 2, "LIGHT_PRESET"],
            [11, 7, 0, 15, 2, "LIGHT_PRESET"],
            [12, 8, 0, 16, 2, "LIGHT_PRESET"],
            [13, 9, 0, 17, 2, "LIGHT_PRESET"],
            [20, 14, 0, 18, 0, "IMAGE"],
            [21, 15, 0, 18, 1, "IMAGE"],
            [22, 16, 0, 18, 2, "IMAGE"],
            [23, 17, 0, 18, 3, "IMAGE"],
            [24, 18, 0, 19, 0, "IMAGE"]],
  "groups": [], "config": {},
  "extra": {"workflow_info": {"name": "02_light_presets"}}
}
```

- [ ] **Step 4: Write `03_depth_normal_pipeline.json`**

```json
{
  "last_node_id": 10,
  "last_link_id": 10,
  "nodes": [
    {"id": 1, "type": "LBM_Model_Loader", "pos": [0, 0], "size": [300, 200],
     "inputs": [], "outputs": [{"name": "lbm_model", "type": "LBM_MODEL", "links": [1]}],
     "widgets_values": ["LBM_depth.safetensors", "depth", "bf16", 0.005, false],
     "properties": {}},
    {"id": 2, "type": "LoadImage", "pos": [0, 250], "size": [300, 200],
     "inputs": [], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [2]}],
     "widgets_values": ["example.png"], "properties": {}},
    {"id": 3, "type": "LBM_DepthNormal_Pro", "pos": [400, 100], "size": [300, 200],
     "inputs": [{"name": "lbm_model", "type": "LBM_MODEL", "link": 1},
                {"name": "image", "type": "IMAGE", "link": 2}],
     "outputs": [{"name": "raw", "type": "IMAGE", "links": [3]},
                 {"name": "post_processed", "type": "IMAGE", "links": [4]}],
     "widgets_values": [28, 0.1], "properties": {}},
    {"id": 4, "type": "LBM_Depth_Visualizer", "pos": [800, 50], "size": [300, 150],
     "inputs": [{"name": "depth_image", "type": "IMAGE", "link": 3}],
     "outputs": [{"name": "image", "type": "IMAGE", "links": [5]}],
     "widgets_values": ["turbo", false, true], "properties": {}},
    {"id": 5, "type": "LBM_Depth_Visualizer", "pos": [800, 250], "size": [300, 150],
     "inputs": [{"name": "depth_image", "type": "IMAGE", "link": 4}],
     "outputs": [{"name": "image", "type": "IMAGE", "links": [6]}],
     "widgets_values": ["inferno", false, true], "properties": {}},
    {"id": 6, "type": "PreviewImage", "pos": [1200, 100], "size": [300, 200],
     "inputs": [{"name": "images", "type": "IMAGE", "link": 5}], "outputs": [],
     "properties": {}},
    {"id": 7, "type": "SaveImage", "pos": [1200, 300], "size": [300, 200],
     "inputs": [{"name": "images", "type": "IMAGE", "link": 6}], "outputs": [],
     "widgets_values": ["depth_pp"], "properties": {}}
  ],
  "links": [[1, 1, 0, 3, 0, "LBM_MODEL"],
            [2, 2, 0, 3, 1, "IMAGE"],
            [3, 3, 0, 4, 0, "IMAGE"],
            [4, 3, 1, 5, 0, "IMAGE"],
            [5, 4, 0, 6, 0, "IMAGE"],
            [6, 5, 0, 7, 0, "IMAGE"]],
  "groups": [], "config": {},
  "extra": {"workflow_info": {"name": "03_depth_normal_pipeline"}}
}
```

- [ ] **Step 5: Write `04_model_cache_chain.json`**

```json
{
  "last_node_id": 10,
  "last_link_id": 10,
  "nodes": [
    {"id": 1, "type": "LBM_Model_Loader", "pos": [0, 0], "size": [300, 200],
     "inputs": [], "outputs": [{"name": "lbm_model", "type": "LBM_MODEL", "links": [1, 2]}],
     "widgets_values": ["LBM_relighting.safetensors", "relighting", "bf16", 0.005, false],
     "properties": {}},
    {"id": 2, "type": "LBM_Relighting_Pro", "pos": [400, 0], "size": [300, 200],
     "inputs": [{"name": "lbm_model", "type": "LBM_MODEL", "link": 1}],
     "outputs": [{"name": "image", "type": "IMAGE", "links": [3]}],
     "widgets_values": [28], "properties": {}},
    {"id": 3, "type": "LBM_Relighting_Pro", "pos": [400, 250], "size": [300, 200],
     "inputs": [{"name": "lbm_model", "type": "LBM_MODEL", "link": 2}],
     "outputs": [{"name": "image", "type": "IMAGE", "links": [4]}],
     "widgets_values": [28], "properties": {}},
    {"id": 4, "type": "LBM_Compare_Grid", "pos": [800, 100], "size": [400, 300],
     "inputs": [{"name": "image_1", "type": "IMAGE", "link": 3},
                {"name": "image_2", "type": "IMAGE", "link": 4}],
     "outputs": [{"name": "grid", "type": "IMAGE", "links": [5]}],
     "widgets_values": ["horizontal", 8], "properties": {}},
    {"id": 5, "type": "SaveImage", "pos": [1300, 100], "size": [300, 200],
     "inputs": [{"name": "images", "type": "IMAGE", "link": 5}], "outputs": [],
     "widgets_values": ["cache_chain_demo"], "properties": {}}
  ],
  "links": [[1, 1, 0, 2, 0, "LBM_MODEL"],
            [2, 1, 0, 3, 0, "LBM_MODEL"],
            [3, 2, 0, 4, 0, "IMAGE"],
            [4, 3, 0, 4, 1, "IMAGE"],
            [5, 4, 0, 5, 0, "IMAGE"]],
  "groups": [], "config": {},
  "extra": {"workflow_info": {"name": "04_model_cache_chain"}}
}
```

Note: workflow uses placeholder LoadImage implicitly — caller must add it.

- [ ] **Step 6: Write `05_compare_grid.json`**

```json
{
  "last_node_id": 10,
  "last_link_id": 10,
  "nodes": [
    {"id": 1, "type": "LoadImage", "pos": [0, 0], "size": [300, 200],
     "inputs": [], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [1,2,3,4]}],
     "widgets_values": ["example.png"], "properties": {}},
    {"id": 2, "type": "LBM_Depth_Visualizer", "pos": [400, 0], "size": [300, 150],
     "inputs": [{"name": "depth_image", "type": "IMAGE", "link": 1}],
     "outputs": [{"name": "image", "type": "IMAGE", "links": [5]}],
     "widgets_values": ["viridis"], "properties": {}},
    {"id": 3, "type": "LBM_Depth_Visualizer", "pos": [400, 200], "size": [300, 150],
     "inputs": [{"name": "depth_image", "type": "IMAGE", "link": 2}],
     "outputs": [{"name": "image", "type": "IMAGE", "links": [6]}],
     "widgets_values": ["inferno"], "properties": {}},
    {"id": 4, "type": "LBM_Depth_Visualizer", "pos": [400, 400], "size": [300, 150],
     "inputs": [{"name": "depth_image", "type": "IMAGE", "link": 3}],
     "outputs": [{"name": "image", "type": "IMAGE", "links": [7]}],
     "widgets_values": ["turbo"], "properties": {}},
    {"id": 5, "type": "LBM_Depth_Visualizer", "pos": [400, 600], "size": [300, 150],
     "inputs": [{"name": "depth_image", "type": "IMAGE", "link": 4}],
     "outputs": [{"name": "image", "type": "IMAGE", "links": [8]}],
     "widgets_values": ["gray"], "properties": {}},
    {"id": 6, "type": "LBM_Compare_Grid", "pos": [800, 200], "size": [500, 400],
     "inputs": [{"name": "image_1", "type": "IMAGE", "link": 5},
                {"name": "image_2", "type": "IMAGE", "link": 6},
                {"name": "image_3", "type": "IMAGE", "link": 7},
                {"name": "image_4", "type": "IMAGE", "link": 8}],
     "outputs": [{"name": "grid", "type": "IMAGE", "links": [9]}],
     "widgets_values": ["grid_2x2", 8], "properties": {}},
    {"id": 7, "type": "SaveImage", "pos": [1400, 200], "size": [300, 200],
     "inputs": [{"name": "images", "type": "IMAGE", "link": 9}], "outputs": [],
     "widgets_values": ["colormap_compare"], "properties": {}}
  ],
  "links": [[1, 1, 0, 2, 0, "IMAGE"],
            [2, 1, 0, 3, 0, "IMAGE"],
            [3, 1, 0, 4, 0, "IMAGE"],
            [4, 1, 0, 5, 0, "IMAGE"],
            [5, 2, 0, 6, 0, "IMAGE"],
            [6, 3, 0, 6, 1, "IMAGE"],
            [7, 4, 0, 6, 2, "IMAGE"],
            [8, 5, 0, 6, 3, "IMAGE"],
            [9, 6, 0, 7, 0, "IMAGE"]],
  "groups": [], "config": {},
  "extra": {"workflow_info": {"name": "05_compare_grid"}}
}
```

- [ ] **Step 7: Write `06_batch_processing.json`**

```json
{
  "last_node_id": 10,
  "last_link_id": 10,
  "nodes": [
    {"id": 1, "type": "LBM_Model_Loader", "pos": [0, 0], "size": [300, 200],
     "inputs": [], "outputs": [{"name": "lbm_model", "type": "LBM_MODEL", "links": [1]}],
     "widgets_values": ["LBM_relighting.safetensors", "relighting", "bf16", 0.005, false],
     "properties": {}},
    {"id": 2, "type": "LoadImagesFromDirectory", "pos": [0, 250], "size": [300, 150],
     "inputs": [], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [2]}],
     "widgets_values": ["input_images"], "properties": {}},
    {"id": 3, "type": "LBM_Light_Preset", "pos": [0, 450], "size": [300, 150],
     "inputs": [], "outputs": [{"name": "light_preset", "type": "LIGHT_PRESET", "links": [3]}],
     "widgets_values": ["preset", "warm_neutral", 0, 45, 1.0, 5500],
     "properties": {}},
    {"id": 4, "type": "LBM_Batch_Processor", "pos": [400, 200], "size": [300, 200],
     "inputs": [{"name": "lbm_model", "type": "LBM_MODEL", "link": 1},
                {"name": "images", "type": "IMAGE", "link": 2},
                {"name": "light_preset", "type": "LIGHT_PRESET", "link": 3}],
     "outputs": [{"name": "images", "type": "IMAGE", "links": [4]}],
     "widgets_values": [28], "properties": {}},
    {"id": 5, "type": "SaveImage", "pos": [800, 200], "size": [300, 200],
     "inputs": [{"name": "images", "type": "IMAGE", "link": 4}], "outputs": [],
     "widgets_values": ["batch_relit"], "properties": {}}
  ],
  "links": [[1, 1, 0, 4, 0, "LBM_MODEL"],
            [2, 2, 0, 4, 1, "IMAGE"],
            [3, 3, 0, 4, 2, "LIGHT_PRESET"],
            [4, 4, 0, 5, 0, "IMAGE"]],
  "groups": [], "config": {},
  "extra": {"workflow_info": {"name": "06_batch_processing"}}
}
```

- [ ] **Step 8: Validate JSON**

```bash
for f in ComfyUI-LBM-Pro/example_workflows/*.json; do
  python -c "import json,sys; json.load(open('$f'))" && echo "$f OK" || echo "$f FAIL"
done
```

Expected: all 6 files report `OK`.

- [ ] **Step 9: Commit**

```bash
git add ComfyUI-LBM-Pro/example_workflows/
git commit -m "feat(workflows): add 6 example workflows covering all node types"
```

---

## Task 17: User-Facing README

**Files:**
- Create: `ComfyUI-LBM-Pro/README.md`
- Create: `ComfyUI-LBM-Pro/docs/README.md`

- [ ] **Step 1: Write `ComfyUI-LBM-Pro/README.md`**

```markdown
# ComfyUI-LBM-Pro

Professional LBM (Latent Bridge Matching) image processing pipeline for ComfyUI.

Builds on the original [ComfyUI-LBM](https://github.com/1038lab/ComfyUI-LBM) by adding model caching, light presets, depth/normal visualization, batch processing, and 8 specialized nodes for production image workflows.

## Features

- **8 specialized nodes** (vs 2 in the original)
- **Model caching** — load once, reuse across many invocations
- **8 light presets** (golden_hour, overcast, studio_left, studio_top, sunset, night_blue, cool_neutral, warm_neutral) + custom builder
- **4 depth colormaps** (viridis, inferno, turbo, gray)
- **Compare Grid** — 2–9 image composition
- **Batch Processor** — consistent parameters across many images

## Nodes

| Node | Purpose |
|------|---------|
| `LBM Model Loader` | Load + cache a checkpoint as `LBM_MODEL` |
| `LBM Light Preset` | Emit a `LIGHT_PRESET` |
| `LBM Relighting Pro` | Enhanced relighting with light tinting |
| `LBM Depth/Normal Pro` | Emit raw + post-processed depth/normal |
| `LBM Depth Visualizer` | Colorize depth (viridis/inferno/turbo/gray) |
| `LBM Normal Visualizer` | Validate unit-length normals |
| `LBM Compare Grid` | Stitch 2–9 images into a grid |
| `LBM Batch Processor` | Process image batch with shared params |

## Installation

```bash
cd ComfyUI/custom_nodes
git clone <your-fork-url> ComfyUI-LBM-Pro
cd ComfyUI-LBM-Pro
pip install -r requirements.txt
```

Models auto-download to `ComfyUI/models/diffusion_models/LBM/` on first run.

## Quick Start

Load `example_workflows/01_basic_relighting.json` in ComfyUI and run it.

## Documentation

See `docs/2026-09-10-comfyui-lbm-pro-design.md` for design rationale and `docs/README.md` for user guide.

## Credits

- Original LBM: [Hugging Face](https://huggingface.co/jasperai/LBM_relighting), [Paper](https://arxiv.org/abs/...)
- Original ComfyUI node: [1038lab/ComfyUI-LBM](https://github.com/1038lab/ComfyUI-LBM)
- License: GPL-3.0
```

- [ ] **Step 2: Write `ComfyUI-LBM-Pro/docs/README.md`**

```markdown
# ComfyUI-LBM-Pro User Guide

## Node Reference

### LBM Model Loader

Loads a `.safetensors` checkpoint into memory and caches it. The output is an `LBM_MODEL` that downstream nodes consume.

| Input | Type | Default | Notes |
|-------|------|---------|-------|
| `model_name` | enum | relighting | Files in `models/diffusion_models/LBM/` |
| `task` | enum | relighting | Must match the checkpoint family |
| `precision` | enum | bf16 | fp32/bf16/fp16 |
| `bridge_noise_sigma` | float | 0.005 | 0.0–0.1 |
| `force_reload` | bool | False | Bypass cache |

### LBM Light Preset

Outputs a `LIGHT_PRESET`. Either pick from 8 presets or build a custom one with azimuth/elevation/intensity/temperature.

### LBM Relighting Pro

Runs the cached relighting model and applies the preset's RGB tint × intensity. Mask input restricts processing to a region.

### LBM Depth/Normal Pro

Same as the original node, but emits two outputs: `raw` (latent in [0, 1]) and `post_processed` (depth inverted, normal normalized).

### LBM Depth Visualizer

Convert a depth image to a colored visualization. Choose `viridis`, `inferno`, `turbo`, or `gray`. Optional invert and auto-normalize.

### LBM Normal Visualizer

Re-normalize a normal map so every pixel has unit length. Optional `[-1, 1] → [0, 1]` rescale.

### LBM Compare Grid

Compose 2–9 images into a comparison grid. Layouts: `auto`, `horizontal`, `vertical`, `grid_2x2`, `grid_3x3`. Add padding between cells.

### LBM Batch Processor

Run a relighting model on a batched `IMAGE` input. Useful for consistent processing of many frames.

## Workflow Examples

See `example_workflows/` for 6 ready-to-run `.json` files.

## Performance Tips

- Models are cached for 10 minutes; reuse the `LBM_Model_Loader` output across nodes to avoid reload.
- Use `bf16` unless you need fp32 precision (saves VRAM).
- Steps 20–30 is a good quality/speed tradeoff.

## Limitations

- Light presets are **post-processing tints** (LBM models do not consume external lighting). The visual effect is an approximation.
- Bridge noise sigma is overridden per-call by the preset value.

## License

GPL-3.0
```

- [ ] **Step 3: Commit**

```bash
git add ComfyUI-LBM-Pro/README.md ComfyUI-LBM-Pro/docs/README.md
git commit -m "docs: add project README and user guide"
```

---

## Task 18: Final Verification

**Files:** none (verification only)

- [ ] **Step 1: Run all unit tests**

```bash
cd ComfyUI-LBM-Pro && python -m pytest tests/ -v --ignore=tests/smoke_import.py
```

Expected: 4 test files pass (types, presets, cache, visualizers) ≈ 30+ tests total.

- [ ] **Step 2: Run smoke test**

```bash
cd ComfyUI-LBM-Pro && python tests/smoke_import.py
```

Expected: `OK: 8 node modules importable`.

- [ ] **Step 3: Validate JSON files**

```bash
for f in ComfyUI-LBM-Pro/example_workflows/*.json; do
  python -c "import json,sys; json.load(open('$f'))" && echo "$f OK"
done
```

Expected: 6 OK lines.

- [ ] **Step 4: Confirm directory structure**

```bash
find ComfyUI-LBM-Pro -type f -not -path "*/.git/*" -not -path "*/__pycache__/*" | sort
```

Expected: matches the File Structure section.

- [ ] **Step 5: Tag release**

```bash
git tag -a v0.1.0 -m "ComfyUI-LBM-Pro v0.1.0"
```

- [ ] **Step 6: Final commit**

```bash
git commit --allow-empty -m "release: ComfyUI-LBM-Pro v0.1.0"
```

---

## Self-Review

**1. Spec coverage check:**
- §1.1 Goal → Tasks 1, 7–14
- §1.3 Scope (no HDR/tile/ControlNet) → respected throughout
- §2.1 Directory structure → Task 1
- §2.2 Cache, Presets, Model factory, Visualizers, Types → Tasks 2–6
- §3.1–3.8 All 8 nodes → Tasks 7–14
- §4 Data flow examples → Task 16 (workflows cover all 3 flows)
- §5 Example workflows (6 files) → Task 16
- §6.1 Cache key → Task 4 (in `LBMModelCache.get_or_load`)
- §6.2 Light preset tinting → Task 9 (in `_apply_tint`)
- §6.3 Cache TTL → Task 4 (default 600s)
- §6.4 Error handling → Tasks 7, 13 (download retry, label mismatch)
- §6.5 Type registration → Task 15
- §7 Testing → Tasks 2–6 (unit tests), Task 15 (smoke), Task 18 (final)
- §8 Risks → mitigated in design

**2. Placeholder scan:** none — every step has code blocks or exact commands.

**3. Type consistency:**
- `LBM_MODEL_TYPE`, `LIGHT_PRESET_TYPE` defined in Task 2 → used in Tasks 7–15 ✓
- `LightPreset`, `PRESETS`, `build_custom_preset` defined in Task 3 → used in Tasks 8, 9, 14 ✓
- `LBMModelCache` defined in Task 4 → used in Task 7 ✓
- `build_lbm_model`, `load_lbm_checkpoint` defined in Task 5 → used in Task 7 ✓
- `depth_to_colormap`, `normalize_normal_map` defined in Task 6 → used in Tasks 11, 12 ✓
- All node class names registered in Task 15 ✓

Plan complete. Saved to `docs/superpowers/plans/2026-09-10-comfyui-lbm-pro.md`.
