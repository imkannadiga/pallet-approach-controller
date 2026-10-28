"""The camera's pockets along the LiDAR's face line (D102 step b), on rendered depth images."""

import math

import numpy as np
import pytest

from pallet_approach.camera_pockets import measure
from pallet_approach.model import Plane

from pallet_scene import CAMERA, FLOOR_Z, INTRINSICS, PARAMS, depth_image

FLOOR = Plane(np.array([0.0, 0.0, 1.0]), -FLOOR_Z)
P = PARAMS['camera_pockets']


def _measure(depth, face_x, lateral, yaw_deg, side='short'):
    # A face pose as the LiDAR would report it: yaw of the outward normal (~pi).
    yaw = math.pi + math.radians(yaw_deg)
    return measure(depth, INTRINSICS, CAMERA, np.array([face_x, lateral]), yaw, FLOOR,
                   PARAMS['pallet']['sides'][side], P)


@pytest.mark.parametrize('face_x,lateral,yaw', [(1.75, 0.0, 0.0), (2.5, 0.12, 0.0), (2.0, -0.1, 15.0)])
def test_pockets_found_where_they_are(face_x, lateral, yaw):
    r = _measure(depth_image(face_x, lateral, yaw, noise=0.003), face_x, lateral, yaw)
    assert r.offset is not None and r.both_open
    assert abs(r.offset) < 0.005                                   # the true face centre is the pocket centre
    assert abs(r.range_residual) < 0.01


def test_a_wrong_lidar_lateral_is_corrected_by_the_camera():
    # The LiDAR's centre is 3 cm off to the side; the camera puts the pockets back.
    r = _measure(depth_image(2.0, 0.0, noise=0.003), 2.0, 0.03, 0.0)
    assert r.offset == pytest.approx(-0.03, abs=0.005)


def test_the_long_face():
    r = _measure(depth_image(2.0, 0.0, side='long', noise=0.003), 2.0, 0.0, 0.0, side='long')
    assert r.offset is not None and abs(r.offset) < 0.006


def test_no_pallet_no_pockets():
    r = _measure(depth_image(2.0, pallet=False), 2.0, 0.0, 0.0)
    assert r.offset is None


def test_a_load_does_not_matter():
    r = _measure(depth_image(2.0, 0.0, load=True, noise=0.003), 2.0, 0.0, 0.0)
    assert r.offset is not None and abs(r.offset) < 0.005


def test_it_is_fast():
    depth = depth_image(2.0, 0.0, noise=0.003)
    times = [_measure(depth, 2.0, 0.0, 0.0).ms for _ in range(20)]
    assert np.median(times) < 10.0                                 # D102 item 9, on this machine


def test_depth_correction_inverts_the_disparity_offset():
    from pallet_approach.camera_pockets import corrected_depth
    beta = 0.0048
    true = np.array([1.0, 2.0, 3.0])
    measured = 1.0 / (1.0 / true + beta)                  # how the ZED gets it wrong (D103)
    assert np.allclose(corrected_depth(measured, beta), true)
    assert np.allclose(corrected_depth(true, 0.0), true)


def test_the_camera_sees_a_turn_the_lidar_missed():
    # The pallet is turned 4 deg; the face handed to the camera says square-on.
    depth = depth_image(2.0, 0.0, yaw_deg=4.0, noise=0.003)
    r = _measure(depth, 2.0, 0.0, 0.0)
    assert abs(math.degrees(r.yaw_residual)) == pytest.approx(4.0, abs=1.0)
    square = _measure(depth_image(2.0, 0.0, noise=0.003), 2.0, 0.0, 0.0)
    assert abs(math.degrees(square.yaw_residual)) < 0.5
