from lbm_core.types import LBM_MODEL_TYPE, LIGHT_PRESET_TYPE


def test_lbm_model_type_value():
    assert LBM_MODEL_TYPE == "LBM_MODEL"


def test_light_preset_type_value():
    assert LIGHT_PRESET_TYPE == "LIGHT_PRESET"


def test_types_are_distinct():
    assert LBM_MODEL_TYPE != LIGHT_PRESET_TYPE
