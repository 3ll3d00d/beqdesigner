import pytest

from model.jriver.mcws import ATMOS_CHANNELS_MIN_VERSION, parse_version


@pytest.mark.parametrize('version_str,expected', [
    ('36.0.14', (36, 0, 14)),
    ('35.0.39', (35, 0, 39)),
    ('35.0.38', (35, 0, 38)),
    ('28', (28,)),
])
def test_parse_version(version_str, expected):
    assert parse_version(version_str) == expected


@pytest.mark.parametrize('version_str,expected', [
    ('36.0.14', True),
    ('35.0.39', True),
    ('35.0.40', True),
    ('35.0.38', False),
    ('35.0.0', False),
    ('35', False),
    ('30.0.5', False),
    ('28.0.28', False),
])
def test_atmos_channels_min_version_gate(version_str, expected):
    assert (parse_version(version_str) >= ATMOS_CHANNELS_MIN_VERSION) is expected
