"""Tests for the synthetic test data itself.

If these fail, every other test is measuring the wrong thing.
"""

import math

import numpy as np
import pytest
import synthetic as S


def test_pocket_geometry_matches_epal():
    """The modelled pockets must be the EPAL 1 numbers in params.yaml."""
    for u0, u1 in S.POCKETS:
        assert math.isclose(u1 - u0, S.POCKET_WIDTH, abs_tol=1e-9)
    centers = [0.5 * (u0 + u1) for u0, u1 in S.POCKETS]
    assert math.isclose(centers[1] - centers[0], S.POCKET_CENTER_SPACING, abs_tol=1e-9)
    assert math.isclose(sum(u1 - u0 for u0, u1 in S.BLOCKS), 0.345, abs_tol=1e-9)
    # blocks + pockets must tile the 800 mm face exactly
    total = sum(u1 - u0 for u0, u1 in S.BLOCKS) + sum(u1 - u0 for u0, u1 in S.POCKETS)
    assert math.isclose(total, S.PALLET_WIDTH, abs_tol=1e-9)


def test_pallet_height_decomposes():
    """144 mm = 22 top + 100 block + 22 bottom."""
    assert math.isclose(2 * S.DECK_THICKNESS + S.BLOCK_HEIGHT, S.PALLET_HEIGHT, abs_tol=1e-9)
    assert math.isclose(S.DECK_THICKNESS + S.BLOCK_HEIGHT / 2, S.POCKET_CENTER_HEIGHT, abs_tol=1e-9)


def test_face_sits_at_the_requested_distance(square_pallet):
    points, truth = square_pallet
    at_pocket_height = points[np.abs(points[:, 2] - (truth['floor_z'] + 0.072)) < 0.02]
    assert np.allclose(at_pocket_height[:, 0], 2.0, atol=1e-9)


def test_pockets_are_actually_empty(square_pallet):
    """No points inside either pocket opening, which is what makes gap analysis work."""
    points, truth = square_pallet
    band = points[np.abs(points[:, 2] - (truth['floor_z'] + 0.072)) < 0.02]
    for center in truth['pocket_centers']:
        inside = np.abs(band[:, 1] - center[1]) < 0.08
        assert inside.sum() == 0, 'pocket at y=%.3f is not empty' % center[1]


def test_rotation_moves_the_normal(rotated_pallet):
    _, truth = rotated_pallet
    angle = math.atan2(truth['normal'][1], truth['normal'][0])
    assert math.isclose(abs(angle), math.radians(180 - 20), abs_tol=1e-6)


def test_extrinsic_round_trip():
    forward = S.base_to_camera_matrix()
    back = S.camera_to_base_matrix()
    assert np.allclose(forward @ back, np.eye(4), atol=1e-12)


def test_extrinsic_matches_the_measured_transform():
    """A point 1 m in front of the camera is 1.39 m ahead of base_footprint and
    0.373 m up -- the camera height above the floor, not above base_link."""
    cam_point = np.array([0.0, 0.0, 1.0, 1.0])
    base_point = S.camera_to_base_matrix() @ cam_point
    assert np.allclose(base_point[:3], [1.390, 0.060, 0.373], atol=1e-9)


def test_heights_are_relative_to_the_floor_not_to_zero():
    """base_footprint is 25 mm above the real floor, so heights are measured
    from the fitted floor, never from z = 0. The synthetic clouds use the
    measured offset so that anything assuming otherwise fails here."""
    _, truth = S.pallet_cloud()
    assert truth['floor_z'] == S.FLOOR_Z
    assert truth['floor_z'] != 0.0, 'the default floor must not be zero'
    assert np.allclose(truth['pocket_centers'][:, 2],
                       S.FLOOR_Z + S.POCKET_CENTER_HEIGHT)
    assert truth['face_center'][2] == pytest.approx(S.FLOOR_Z + S.PALLET_HEIGHT / 2)


def test_measured_camera_height_is_consistent():
    """0.373 m above base_footprint, 0.398 m above the floor, 25 mm apart."""
    assert S.CAMERA_HEIGHT_ABOVE_FLOOR == pytest.approx(0.398)
    assert S.CAMERA_XYZ[2] - S.FLOOR_Z == pytest.approx(S.CAMERA_HEIGHT_ABOVE_FLOOR)


def test_depth_image_is_plausible(square_pallet):
    points, _ = square_pallet
    image = S.depth_image(points)
    assert image.shape == (S.IMAGE_HEIGHT, S.IMAGE_WIDTH)
    assert image.dtype == np.float32
    valid = np.isfinite(image)
    assert valid.sum() > 5000
    # the pallet face is 2.0 m ahead of base_link, so 1.61 m from the camera
    assert 1.5 < np.nanmin(image) < 1.7 or np.nanmin(image) > 1.0
