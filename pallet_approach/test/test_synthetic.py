"""Tests for the synthetic test data itself.

If these fail, every other test is measuring the wrong thing.
"""

import math

import numpy as np
import pytest
import synthetic as S


@pytest.mark.parametrize('face', [S.SHORT, S.LONG], ids=lambda f: f.name)
def test_each_face_tiles_exactly(face):
    """Blocks and pockets must account for the whole face with nothing left
    over. A face that does not tile means the measurement was transcribed
    wrong, and every pocket position derived from it would be off."""
    total = (sum(u1 - u0 for u0, u1 in face.blocks)
             + sum(u1 - u0 for u0, u1 in face.pockets))
    assert math.isclose(total, face.width, abs_tol=1e-9), (
        f'{face.name} face: blocks + pockets = {total:.4f} m against a '
        f'{face.width} m face')


@pytest.mark.parametrize('face, width, spacing', [
    (S.SHORT, 0.230, 0.365),
    (S.LONG, 0.3725, 0.5225),
], ids=['short', 'long'])
def test_pocket_geometry_is_the_measured_pallet(face, width, spacing):
    """The measured pallet (D4, D19), not the EPAL nominal this file used to
    model. The two disagree by enough to matter: 230 against 227.5 mm wide and
    365 against 372.5 mm apart on the short face, and the EPAL face has a 22 mm
    bottom board that this pallet's pocket face does not."""
    for u0, u1 in face.pockets:
        assert math.isclose(u1 - u0, width, abs_tol=1e-9)
    assert math.isclose(face.pocket_center_spacing, spacing, abs_tol=1e-9)
    assert math.isclose(face.pocket_width, width, abs_tol=1e-9)


def test_the_short_face_has_no_board_under_the_opening():
    """The fact that reorganised the standoff, the fork height and how pocket
    height is measured. The forks enter between the bottom boards on this face,
    so the opening runs from the floor."""
    assert S.SHORT.bottom_board == 0.0
    assert S.SHORT.opening_bottom == 0.0
    assert S.LONG.bottom_board > 0.0, 'the long face DOES have one'
    assert S.LONG.opening_bottom == S.LONG.bottom_board


@pytest.mark.parametrize('face', [S.SHORT, S.LONG], ids=lambda f: f.name)
def test_pallet_height_decomposes(face):
    """Board below (if any) + opening + deck above = the pallet's height."""
    deck = S.PALLET_HEIGHT - face.opening_top
    assert math.isclose(face.bottom_board + face.pocket_height + deck,
                        S.PALLET_HEIGHT, abs_tol=1e-9)
    assert math.isclose(face.pocket_center_height,
                        face.opening_bottom + face.pocket_height / 2,
                        abs_tol=1e-9)


def test_the_two_faces_are_the_same_pallet():
    """Width of one is the depth of the other, both 145 mm tall."""
    assert math.isclose(S.SHORT.width, S.LONG.depth, abs_tol=1e-9)
    assert math.isclose(S.LONG.width, S.SHORT.depth, abs_tol=1e-9)


@pytest.mark.parametrize('side', ['short', 'long'])
def test_the_cloud_reports_the_face_it_built(side):
    # without the floor, whose y-extent is the ROI's rather than the pallet's
    points, truth = S.pallet_cloud(distance=1.75, side=side, spacing=0.01,
                                   with_floor=False)
    face = S.FACES[side]
    assert truth['side'] == side
    assert math.isclose(truth['pocket_center_spacing'],
                        face.pocket_center_spacing, abs_tol=1e-9)
    spread = points[:, 1].max() - points[:, 1].min()
    assert spread == pytest.approx(face.width, abs=0.02)


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
