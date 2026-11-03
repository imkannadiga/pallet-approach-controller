"""The range study's curves (D109) as the pipeline reads them."""

import pathlib

import numpy as np
import pytest
import yaml

from pallet_approach import range_error as RE
from pallet_approach.fusion import PalletFilter

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARAMS = yaml.safe_load((ROOT / 'config' / 'params.yaml').read_text())['/**']['ros__parameters']
STUDY = yaml.safe_load((ROOT.parent / 'docs' / 'studies' / 'range_error_20261102.yaml').read_text())


@pytest.mark.parametrize('sensor', ['lidar', 'camera'])
def test_params_carry_the_study_unchanged(sensor):
    block, study = PARAMS['range_error'][sensor], STUDY[sensor]
    assert block['valid_range'] == study['valid_range_m']
    assert block['sensor_minus_mark'] == study['sensor_minus_mark_m']
    assert block['sigma_m'] == [r[0] for r in study['table']['rows']]
    assert block['sigma_total'] == [r[5] for r in study['table']['rows']]
    assert PARAMS['range_error']['mark_x'] == STUDY['mark_m'][0]


def test_lidar_bias_matches_the_table_and_holds_its_end_values():
    block = PARAMS['range_error']['lidar']
    for m, bias, *_ in STUDY['lidar']['table']['rows']:
        assert RE.lidar_bias(m, block) == pytest.approx(bias, abs=2e-6)
    assert RE.lidar_bias(9.0, block) == pytest.approx(RE.lidar_bias(block['valid_range'][1], block))
    assert RE.sigma_total(0.2, block) == pytest.approx(block['sigma_total'][0])


def test_lidar_correction_moves_each_return_along_its_ray():
    block = PARAMS['range_error']['lidar']
    origin = np.array([0.1, 0.0, 0.3])
    p = np.array([[3.0, 0.0, 0.3], [0.1, 2.0, 0.3]])
    out = RE.correct_lidar(p, origin, block)
    for before, after in zip(p, out):
        r = np.linalg.norm(before - origin)
        assert np.linalg.norm(after - origin) == pytest.approx(
            r - RE.lidar_bias(r - block['sensor_minus_mark'], block))
        d0, d1 = (before - origin) / r, (after - origin) / np.linalg.norm(after - origin)
        assert np.allclose(d0, d1)


def test_the_filter_takes_the_lidars_sigma_from_the_curve():
    f = PalletFilter(PARAMS)
    block = PARAMS['range_error']['lidar']
    assert f.lidar_sigma_range(2.4625) == pytest.approx(RE.sigma_total(2.0, block))
    assert f.lidar_sigma_range(None) == PARAMS['fusion']['lidar_sigma_range']
