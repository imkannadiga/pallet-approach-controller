"""Shared pytest configuration and fixtures."""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

import synthetic as S  # noqa: E402


@pytest.fixture
def params():
    """Exactly what `perception_node._params()` hands `estimate_pallet`.

    Flat, with the names perception uses at the point of use. If this drifts
    from the node, `estimate_pallet` passes its tests and fails on the robot --
    so `test_params.py` cross-checks both against config/params.yaml.
    """
    return {
        'seed': 0,

        'stride': 2,
        'min_range': 0.40,
        'max_range': 4.00,

        'roi': {'x_min': 0.5, 'x_max': 4.0, 'y_min': -1.5, 'y_max': 1.5,
                'z_min': -0.05, 'z_max': 0.80},
        'voxel_size': 0.02,

        'floor_distance_threshold': 0.02,
        'ransac_iterations': 200,
        'max_floor_tilt': 0.17,
        'floor_margin': 0.03,

        'cluster_eps': 0.05,
        'cluster_min_samples': 8,
        'min_cluster_points': 100,
        'size_tolerance': 0.25,

        # the real pallet, tape-measured, not the EPAL nominal
        'pallet_width': 0.795,
        'pallet_length': 1.195,
        'pallet_height': 0.145,

        'face_distance_threshold': 0.015,
        'max_tilt_from_vertical': 0.26,
        'face_min_points': 60,
        'max_facing_angle': 1.05,

        # The band sits at 0.055 +- 0.02, i.e. 0.035-0.075 above the floor:
        # clear of the 0.03 floor-removal margin below and of the deck board
        # edge above, so it samples block face and nothing else.
        'pocket_center_height': 0.055,
        'pocket_band_half_height': 0.02,
        # points within this of the fitted face plane count as "the face".
        # Measured on the real pallet, its front surface spans 26 mm from deck
        # boards to lower blocks, so a 15 mm RANSAC inlier band misses most of
        # the blocks. 60 mm takes the lot.
        'pocket_slab_half_depth': 0.06,
        'pocket_min_gap_width': 0.15,
        'pocket_max_gap_width': 0.32,
        'pocket_min_points_per_bin': 2,
        'pocket_nominal_width': 0.23,
        'pocket_nominal_height': 0.10,
        'pocket_nominal_spacing': 0.365,
    }


@pytest.fixture
def go_nogo_params():
    return {
        'max_yaw_error': 0.087,
        'max_lateral_offset': 0.050,
        'pocket_width_tolerance': 0.040,
        'pocket_height_tolerance': 0.030,
        'min_pocket_count': 2,
        'overhang_margin': 0.020,
        'pocket_clearance_depth': 0.10,
        'stability_frames': 5,
        'max_position_spread': 0.030,
        'max_yaw_spread': 0.035,
    }


@pytest.fixture
def intrinsics():
    """The ZED's measured intrinsics (docs/SYSTEM.md)."""
    return {'fx': S.FX, 'fy': S.FY, 'cx': S.CX, 'cy': S.CY}


@pytest.fixture
def square_pallet():
    """A pallet 2 m ahead, squarely facing the robot, no noise."""
    return S.pallet_cloud(distance=2.0, yaw=0.0, noise=0.0)


@pytest.fixture
def rotated_pallet():
    """The same pallet rotated 20 degrees, with realistic noise."""
    return S.pallet_cloud(distance=2.0, yaw=math.radians(20), noise=0.005)
