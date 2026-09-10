"""Pure-NumPy depth colormaps and normal-map helpers for LBM-Pro.

Colormaps are generated once at import from polynomial approximations
of matplotlib's LUTs (we avoid a matplotlib dependency to keep the
package lightweight).
"""
from __future__ import annotations

import numpy as np


COLORMAPS: list[str] = ["viridis", "inferno", "turbo", "gray"]


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
    out = np.where(norm > 1e-8, out, arr)
    return out
