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
    assert warm.rgb_tint[0] > warm.rgb_tint[2]
    assert cool.rgb_tint[2] > cool.rgb_tint[0]
