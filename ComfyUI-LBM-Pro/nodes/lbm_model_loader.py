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
)
from lbm_core.model_factory import build_lbm_model, load_lbm_checkpoint


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
