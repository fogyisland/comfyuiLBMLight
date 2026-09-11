"""Lighting preset database for LBM-Pro.

Each preset is a (rgb_tint, intensity, bridge_noise_sigma) tuple that
emulates a lighting style by post-processing the LBM output. The model
itself does not consume external lighting, so these are visual
approximations — see design doc §6.2 for rationale.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LightPreset:
    """A lighting style applied after LBM inference.

    ``rgb_tint`` per-channel multipliers are clamped to ``[0.0, 1.5]``
    so no single channel can wash out the image — presets that
    mathematically derive a value above 1.5 (e.g. ``_temperature_to_rgb_tint``
    can reach ~1.6× on the warm or cool extreme) are pinned at the
    clamp so behaviour stays predictable across the preset library.
    """
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
    """Approximate RGB tint multipliers from color temperature in Kelvin.

    Warm tints (low K, ~3000) have R > B; cool tints (high K, ~9000) have
    B > R. Output channels center around 1.0 so mean brightness is
    preserved, with each channel clamped to ``[0.0, 1.5]`` so no single
    channel can wash out the image.

    Uses a piecewise linear model derived from blackbody radiation
    reference points (2000K → very orange, 5500K → neutral, 10000K →
    very blue).
    """
    # Clamp input.
    k = max(1000.0, min(40000.0, temperature_k))
    # Reference points (k, R, G, B) — values around 1.0.
    # Interpolated linearly in log-k space.
    log_k = math.log(k)
    if k <= 5500:
        # Warm side: 1000K (deep orange) → 5500K (neutral)
        # Lerp factor from 1000K to 5500K
        t = (log_k - math.log(1000)) / (math.log(5500) - math.log(1000))
        t = max(0.0, min(1.0, t))
        r = 1.45 - 0.45 * t   # 1.45 → 1.0
        g = 0.95 - 0.15 * t   # 0.95 → 0.80
        b = 0.65 + 0.35 * t   # 0.65 → 1.0
    else:
        # Cool side: 5500K (neutral) → 40000K (deep blue)
        t = (log_k - math.log(5500)) / (math.log(40000) - math.log(5500))
        t = max(0.0, min(1.0, t))
        r = 1.0 - 0.40 * t    # 1.0 → 0.60
        g = 0.80 + 0.20 * t   # 0.80 → 1.0
        b = 1.0               # stays 1.0
    # Center on mean=1 so brightness is preserved.
    mean = (r + g + b) / 3.0
    raw = (r / mean, g / mean, b / mean)
    # C27: clamp each channel to [0.0, 1.5] so no preset can wash out
    # a single channel even at the warm/cool extremes.
    return tuple(max(0.0, min(1.5, c)) for c in raw)


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
