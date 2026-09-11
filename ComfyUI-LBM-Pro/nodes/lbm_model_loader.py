"""LBM Model Loader — load & cache a LBM checkpoint as LBM_MODEL."""
from __future__ import annotations

import os
import shutil

import requests
import torch

import folder_paths
import comfy.model_management as mm
from comfy.utils import ProgressBar

from lbm_core import (
    LIGHT_PRESET_TYPE,
    LBM_MODEL_TYPE,
    LBMModelCache,
)
from lbm_core.model_factory import build_lbm_model, load_lbm_checkpoint


# Per-task model repository on Hugging Face. The full URL is built from
# ``_MIRROR_HOSTS`` and ``_MODEL_REPOS`` so the user can pick a mirror
# (or fall back to upstream) via the ``mirror`` widget on the node.
_MODEL_REPOS = {
    "relighting": "jasperai/LBM_relighting",
    "depth": "jasperai/LBM_depth",
    "normal": "jasperai/LBM_normals",
}

# Ordered host list. The first entry is the primary mirror; later entries
# are fallbacks. ``auto`` tries them in this order; explicit choices
# pick a single host.
_MIRROR_HOSTS = ("hf-mirror.com", "huggingface.co")

# Widget labels for the ``mirror`` dropdown. Order is significant —
# the first entry is the implicit choice when the widget is hidden.
_MIRROR_OPTIONS = (
    "hf-mirror.com (default)",
    "huggingface.co",
    "auto (try mirror, fall back)",
)

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


def _candidate_hosts(mirror: str) -> list[str]:
    """Translate the widget value into an ordered list of hostnames."""
    if mirror == "hf-mirror.com (default)":
        return ["hf-mirror.com"]
    if mirror == "huggingface.co":
        return ["huggingface.co"]
    # Default and any future "auto*" labels: try mirror first, fall back.
    return list(_MIRROR_HOSTS)


def _build_urls(task: str, filename: str, mirror: str) -> list[str]:
    """Build the ordered list of candidate download URLs."""
    repo = _MODEL_REPOS[task]
    return [
        f"https://{host}/{repo}/resolve/main/{filename}"
        for host in _candidate_hosts(mirror)
    ]


def _download_model(
    model_name: str,
    task: str,
    mirror: str = "auto (try mirror, fall back)",
) -> str:
    base = folder_paths.get_folder_paths("diffusion_models")[0]
    target_dir = os.path.join(base, "LBM")
    os.makedirs(target_dir, exist_ok=True)
    target = os.path.join(target_dir, model_name)
    tmp = os.path.join(target_dir, "temp_download.safetensors")
    urls = _build_urls(task, model_name, mirror)
    last_exc: Exception | None = None
    for url in urls:
        try:
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                pbar = ProgressBar(total)
                bytes_done = 0
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=64 * 1024):
                        if chunk:
                            f.write(chunk)
                            bytes_done += len(chunk)
                            pbar.update_absolute(bytes_done, total)
            shutil.move(tmp, target)
            return target
        except requests.RequestException as e:
            last_exc = e
            # Clean up a partial file before trying the next URL.
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            continue
    # All candidates failed — propagate the last error with context.
    if os.path.exists(tmp):
        try:
            os.remove(tmp)
        except OSError:
            pass
    raise RuntimeError(
        f"Failed to download model {model_name!r} for task={task!r}; "
        f"tried URLs: {urls}. Last error: {last_exc}"
    ) from last_exc


def _resolve_checkpoint(
    model_name: str,
    task: str,
    mirror: str = "auto (try mirror, fall back)",
) -> str:
    for path in folder_paths.get_folder_paths("diffusion_models"):
        candidate = os.path.join(path, "LBM", model_name)
        if os.path.exists(candidate):
            return candidate
    return _download_model(model_name, task, mirror)


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
                "mirror": (
                    list(_MIRROR_OPTIONS),
                    {"default": "auto (try mirror, fall back)",
                     "tooltip": "Download host. 'auto' tries hf-mirror.com "
                                "first then huggingface.co."},
                ),
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
        mirror: str = "auto (try mirror, fall back)",
    ) -> tuple[dict]:
        dtype = _PRECISION_MAP[precision]
        cache_key = f"{model_name}|{task}|{precision}|{mirror}"
        if force_reload:
            LBMModelCache.unload(cache_key)

        def _loader():
            ckpt = _resolve_checkpoint(model_name, task, mirror)
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
