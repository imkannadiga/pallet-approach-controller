"""LiDAR-first acquisition (D102 step a) on synthetic LiDAR sweeps of the measured pallet."""

import math

import numpy as np
import pytest

from pallet_approach.lidar_pallet import find_pallet

from pallet_scene import PARAMS, sweep as _sweep


def _stack(n=5, **kw):
    return np.concatenate([_sweep(seed=i, **kw) for i in range(n)])


@pytest.mark.parametrize('face_x,lateral,yaw', [(1.75, 0.0, 0.0), (2.5, 0.1, 0.0), (2.0, -0.15, 15.0),
                                                (2.25, 0.0, -20.0)])
def test_finds_the_short_face(face_x, lateral, yaw):
    r, _ = find_pallet(_stack(face_x=face_x, lateral=lateral, yaw_deg=yaw), PARAMS)
    assert r is not None and r.side == 'short'
    # The face centre: the block fronts' midpoint.
    assert r.centre[0] == pytest.approx(face_x, abs=0.02)
    assert r.centre[1] == pytest.approx(lateral, abs=0.03)
    # The test turns the pallet counter-clockwise by `yaw`, which brings its left
    # edge NEARER: the tape's angle (left edge further = +) is -yaw.
    tape = math.degrees(math.atan2(math.sin(math.pi - r.yaw), math.cos(math.pi - r.yaw)))
    assert tape == pytest.approx(-yaw, abs=2.0)
    assert r.pocket_centre is not None
    assert np.linalg.norm(r.pocket_centre - np.array([face_x, lateral])) < 0.04
    assert r.in_range


def test_a_load_on_top_changes_nothing():
    plain, _ = find_pallet(_stack(face_x=2.0), PARAMS)
    loaded, _ = find_pallet(_stack(face_x=2.0, load=True), PARAMS)
    assert loaded is not None and loaded.side == 'short'
    assert np.allclose(loaded.centre, plain.centre, atol=0.01)


def test_the_long_face_is_classified_long():
    r, _ = find_pallet(_stack(face_x=2.0, side='long'), PARAMS)
    assert r is not None and r.side == 'long'


def test_no_pallet_no_detection():
    r, info = find_pallet(_stack(face_x=2.0, pallet=False), PARAMS)
    assert r is None and info['floor']


def test_outside_the_working_range_is_flagged():
    r, _ = find_pallet(_stack(face_x=3.3), PARAMS)
    assert r is not None and not r.in_range
