"""Camera and LiDAR in one estimate (D102 step c)."""

import math

import numpy as np

from pallet_approach.fusion import PalletFilter, to_odom, wrap
from pallet_scene import PARAMS

TRUE = np.array([2.0, 0.05])          # pocket pair centre, odom
TRUE_YAW = math.pi + math.radians(2)  # outward normal, slightly turned


def _tangent(yaw):
    return np.array([math.sin(yaw), -math.cos(yaw)])


def _run(steps=60, lidar_lateral_bias=0.0, camera=True, lidar=True, camera_range=0.0, seed=0):
    rng = np.random.default_rng(seed)
    f = PalletFilter(PARAMS)
    t0 = _tangent(TRUE_YAW)
    f.initialise(TRUE + 0.03 * t0, TRUE_YAW + math.radians(1.5), 0.0)    # acquisition: 3 cm, 1.5 deg off
    status = None
    for i in range(1, steps + 1):
        now = i / 15.0                                                    # camera at 15 Hz (D102 item 10)
        if lidar and i % 2 == 0:                                          # LiDAR at ~7.5 Hz here
            c = TRUE + rng.normal(0, 0.004) * np.array([math.cos(TRUE_YAW), math.sin(TRUE_YAW)]) \
                + (lidar_lateral_bias + rng.normal(0, 0.015)) * t0
            f.update_lidar(c, TRUE_YAW + math.radians(rng.normal(0, 1.0)), now)
        if camera:
            f.update_camera(TRUE + rng.normal(0, 0.003) * t0, now)
            f.check_camera(camera_range + rng.normal(0, 0.003), math.radians(rng.normal(0, 0.5)))
        status = f.status(now)
    return f, status


def test_converges_on_the_truth_with_both_sensors():
    _, s = _run()
    t = _tangent(TRUE_YAW)
    assert abs((s.centre - TRUE) @ t) < 0.003                        # sideways: camera-led
    assert abs((s.centre - TRUE) @ np.array([math.cos(TRUE_YAW), math.sin(TRUE_YAW)])) < 0.004
    assert abs(math.degrees(wrap(s.yaw - TRUE_YAW))) < 0.6
    assert s.go_allowed and s.sigma_lateral < 0.004


def test_the_camera_overrules_a_lidar_sideways_bias():
    _, s = _run(lidar_lateral_bias=0.02)
    assert abs((s.centre - TRUE) @ _tangent(TRUE_YAW)) < 0.006       # the LiDAR alone would be 20 mm out


def test_lidar_only_tracks_but_never_allows_go():
    _, s = _run(camera=False)
    assert s.initialised and not s.go_allowed and 'camera' in s.reason


def test_camera_only_tracks_but_never_allows_go():
    _, s = _run(lidar=False)
    assert not s.go_allowed and 'LiDAR' in s.reason


def test_a_range_disagreement_blocks_go_after_five_frames():
    f, s = _run(camera_range=0.03)
    assert s.disagreeing and not s.go_allowed
    for _ in range(3):                                                 # a few frames of agreement end it
        f.check_camera(0.0, 0.0)
    assert not f.disagreeing


def test_an_outlier_is_gated():
    f, _ = _run()
    before = f.x.copy()
    assert not f.update_lidar(TRUE + np.array([0.5, 0.0]), TRUE_YAW, 5.0)
    assert np.allclose(f.x[:2], before[:2], atol=1e-3) and f.rejected >= 1


def test_measurements_from_a_moving_robot_land_in_odom():
    robot = (0.5, -0.2, math.radians(10))
    seen = np.array([1.2, 0.3])                                        # in base_footprint
    odom = to_odom(robot, seen)
    back = np.array([[math.cos(-robot[2]), -math.sin(-robot[2])],
                     [math.sin(-robot[2]), math.cos(-robot[2])]]) @ (odom - robot[:2])
    assert np.allclose(back, seen)
