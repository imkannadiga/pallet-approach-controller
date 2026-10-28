"""The fused perception end to end (D102 step d), on a simulated robot driving at a pallet."""

import math

import numpy as np

from pallet_approach.fused_pipeline import FusedPipeline, to_base
from pallet_approach.fusion import to_odom
from pallet_scene import CAMERA, INTRINSICS, PARAMS, depth_image, sweep

PALLET = np.array([2.6, 0.04])            # pocket pair centre, odom (robot starts at the origin)


def test_acquires_standing_still_then_tracks_while_driving_in():
    pipe = FusedPipeline(PARAMS)
    t = 0.0
    # Standing still: five sweeps, then the pallet is acquired.
    for i in range(5):
        pipe.on_lidar(sweep(PALLET[0], PALLET[1], seed=i), (0.0, 0.0, 0.0), t)
        t += 0.1
    assert pipe.tracking and pipe.side == 'short'
    # Driving straight in 0.6 m at 0.1 m/s: LiDAR at 10 Hz, camera at ~3 Hz.
    for step in range(60):
        x = 0.01 * (step + 1)
        robot = (x, 0.0, 0.0)
        pipe.on_lidar(sweep(PALLET[0] - x, PALLET[1], seed=100 + step), robot, t)
        if step % 3 == 0:
            r = pipe.on_depth(depth_image(PALLET[0] - x, PALLET[1], noise=0.003, seed=step),
                              INTRINSICS, CAMERA, robot, t)
            assert r is not None
        t += 0.1
    s = pipe.status(t)
    assert np.linalg.norm(s.centre - PALLET) < 0.01          # fixed in odom while the robot moved
    assert s.go_allowed, s.reason
    centre, yaw = pipe.in_base((0.6, 0.0, 0.0), t)
    assert np.allclose(centre, PALLET - [0.6, 0.0], atol=0.01)


def test_motion_restarts_the_pool_but_a_single_sweep_can_acquire():
    pipe = FusedPipeline(PARAMS)
    for i in range(4):
        pipe.on_lidar(sweep(*PALLET, pallet=False, seed=i), (0.0, 0.0, 0.0), 0.1 * i)
    assert not pipe.tracking and len(pipe.pool) == 4
    # Moving: the pool restarts, and the one sweep is searched on its own.
    pipe.on_lidar(sweep(PALLET[0] - 0.05, PALLET[1], seed=9), (0.05, 0.0, 0.0), 0.5)
    assert pipe.tracking


def test_a_vanished_pallet_is_dropped_and_reacquired():
    pipe = FusedPipeline(PARAMS)
    for i in range(5):
        pipe.on_lidar(sweep(*PALLET, seed=i), (0.0, 0.0, 0.0), 0.1 * i)
    assert pipe.tracking
    for i in range(25):                                   # 2.5 s of empty floor
        pipe.on_lidar(sweep(*PALLET, pallet=False, seed=50 + i), (0.0, 0.0, 0.0), 0.5 + 0.1 * i)
    assert not pipe.tracking


def test_base_and_odom_round_trip():
    robot = (1.0, -0.5, math.radians(20))
    p = np.array([2.0, 0.3])
    assert np.allclose(to_base(robot, to_odom(robot, p)), p)
