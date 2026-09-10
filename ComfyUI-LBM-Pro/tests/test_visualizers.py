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
    assert a[0, 0, 0] < a[0, 1, 0]
    assert b[0, 0, 0] > b[0, 1, 0]


def test_depth_to_colormap_no_normalize_preserves_input():
    depth = np.array([[0.2, 0.8]], dtype=np.float32)
    rgb = depth_to_colormap(depth, "viridis", normalize=False)
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
    assert np.all(np.isfinite(out))
