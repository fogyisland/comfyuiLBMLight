"""Manual model downloader for ComfyUI-LBM-Pro.

The ``LBM_Model_Loader`` node auto-downloads the first time it runs, but in
environments where the ComfyUI process cannot reach ``hf-mirror.com`` /
``huggingface.co`` (corporate proxy, offline, slow link, …) you can pre-stage
the checkpoints by running this script directly.

All three jasperai checkpoints ship as ``model.safetensors`` — the original
``LBM_relighting.safetensors`` / ``LBM_depth.safetensors`` / ``LBM_normals.safetensors``
names do **not** exist on the repos. Each task downloads a separate file but
they share that single filename, so we keep one canonical file per task
under ``models/diffusion_models/LBM/`` and rename them to make the task
explicit:

    models/diffusion_models/LBM/
        LBM_relighting.model.safetensors
        LBM_depth.model.safetensors
        LBM_normals.model.safetensors

The model loader's default is ``model.safetensors`` — if you pre-place the
file under that exact name the picker will find it directly. The renamed
variants above let you keep all three on disk simultaneously and select them
from the dropdown.

Usage
-----

::

    # Download all three (default mirror: hf-mirror.com → huggingface.co fallback):
    python scripts/download_models.py --all

    # Download a single task:
    python scripts/download_models.py --task relighting

    # Force upstream (skip the China mirror):
    python scripts/download_models.py --task depth --mirror huggingface.co

    # Custom ComfyUI install location:
    python scripts/download_models.py --all --comfyui-dir /path/to/ComfyUI
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from typing import Iterable

import requests

# Mirror order matches the LBM_Model_Loader widget default so behaviour is
# consistent with the node.
_MIRRORS = ("hf-mirror.com", "huggingface.co")

# task → (HF repo, on-disk filename under models/diffusion_models/LBM/)
_REPOS = {
    "relighting": ("jasperai/LBM_relighting", "LBM_relighting.model.safetensors"),
    "depth":      ("jasperai/LBM_depth",      "LBM_depth.model.safetensors"),
    "normal":     ("jasperai/LBM_normals",    "LBM_normals.model.safetensors"),
}


def _build_urls(repo: str, mirror: str) -> list[str]:
    if mirror == "hf-mirror.com":
        hosts = ["hf-mirror.com"]
    elif mirror == "huggingface.co":
        hosts = ["huggingface.co"]
    elif mirror == "auto":
        hosts = list(_MIRRORS)
    else:
        raise ValueError(f"Unknown mirror: {mirror!r}")
    return [f"https://{h}/{repo}/resolve/main/model.safetensors" for h in hosts]


def _download(urls: Iterable[str], target: str) -> None:
    os.makedirs(os.path.dirname(target), exist_ok=True)
    tmp = target + ".part"
    last_exc: Exception | None = None
    for url in urls:
        print(f"  trying {url}", flush=True)
        try:
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                done = 0
                chunk_size = 64 * 1024
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            pct = 100.0 * done / total
                            print(f"\r  {done/1e6:7.1f} / {total/1e6:7.1f} MB "
                                  f"({pct:5.1f}%)", end="", flush=True)
                        else:
                            print(f"\r  {done/1e6:7.1f} MB", end="", flush=True)
            print()  # newline after progress
            shutil.move(tmp, target)
            print(f"  ✔ saved → {target} ({os.path.getsize(target)/1e9:.2f} GB)")
            return
        except requests.RequestException as e:
            print(f"  ✘ failed: {e}")
            last_exc = e
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            continue
    if os.path.exists(tmp):
        try:
            os.remove(tmp)
        except OSError:
            pass
    raise RuntimeError(
        f"All mirrors failed for {urls}. Last error: {last_exc}"
    ) from last_exc


def _resolve_lbm_dir(comfyui_dir: str) -> str:
    return os.path.join(comfyui_dir, "models", "diffusion_models", "LBM")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Pre-download LBM checkpoints into a ComfyUI install.",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true",
                   help="Download all three tasks (relighting + depth + normal).")
    g.add_argument("--task", choices=sorted(_REPOS),
                   help="Download a single task.")

    p.add_argument("--mirror", choices=("auto", "hf-mirror.com", "huggingface.co"),
                   default="auto",
                   help="Mirror preference (default: try hf-mirror.com first, "
                        "fall back to huggingface.co).")
    p.add_argument("--comfyui-dir", default=None,
                   help="Path to ComfyUI install root. Defaults to $COMFYUI_DIR "
                        "env var, then ../ComfyUI relative to this repo, then "
                        "H:/ComfyUI on Windows or ~/ComfyUI on POSIX.")

    args = p.parse_args(argv)

    # Resolve ComfyUI dir
    comfyui_dir = (
        args.comfyui_dir
        or os.environ.get("COMFYUI_DIR")
        or _default_comfyui_dir()
    )
    if not os.path.isdir(comfyui_dir):
        print(f"ERROR: ComfyUI directory not found: {comfyui_dir}", file=sys.stderr)
        print("Pass --comfyui-dir or set COMFYUI_DIR.", file=sys.stderr)
        return 2

    lbm_dir = _resolve_lbm_dir(comfyui_dir)
    print(f"ComfyUI dir : {comfyui_dir}")
    print(f"LBM target  : {lbm_dir}")
    print(f"Mirror      : {args.mirror}")
    print()

    tasks = list(_REPOS) if args.all else [args.task]
    for task in tasks:
        repo, filename = _REPOS[task]
        target = os.path.join(lbm_dir, filename)
        if os.path.exists(target):
            print(f"[{task}] already present at {target} "
                  f"({os.path.getsize(target)/1e9:.2f} GB) — skipping")
            continue
        urls = _build_urls(repo, args.mirror)
        print(f"[{task}] downloading {repo} → {os.path.basename(target)}")
        try:
            _download(urls, target)
        except RuntimeError as e:
            print(f"[{task}] FAILED: {e}", file=sys.stderr)
            return 1
        print()

    print("All requested tasks complete.")
    return 0


def _default_comfyui_dir() -> str:
    """Best-effort default ComfyUI location."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(os.path.dirname(here), "ComfyUI"),       # sibling of repo
        r"H:\ComfyUI" if os.name == "nt" else os.path.expanduser("~/ComfyUI"),
        os.path.expanduser("~/ComfyUI"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[0]  # let main() report "not found"


if __name__ == "__main__":
    sys.exit(main())
