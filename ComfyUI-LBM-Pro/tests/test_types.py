import pytest
from lbm_core.types import LBM_MODEL_TYPE, LIGHT_PRESET_TYPE

_TYPE_CASES = [
    (LBM_MODEL_TYPE, "LBM_MODEL"),
    (LIGHT_PRESET_TYPE, "LIGHT_PRESET"),
]


@pytest.mark.parametrize("constant,expected_value", _TYPE_CASES)
def test_type_constant_value(constant, expected_value):
    assert constant == expected_value


def test_types_are_distinct():
    assert LBM_MODEL_TYPE != LIGHT_PRESET_TYPE
