"""Tests for calibration.py -- the camera extrinsic correction.

The measurement these encode: the camera's URDF orientation is 0.868 deg out
(pitch +0.698, roll -0.516), measured 2026-10-14 from four bare-floor bags at
two robot headings. See docs/SYSTEM.md.
"""

import math

import numpy as np
import pytest
import synthetic as S

from pallet_approach.model import Plane
from pallet_approach.calibration import (
    apparent_floor_normal,
    correct_transform,
    correction_matrix,
    levelling_rotation,
    measured_floor_normal,
)

MEASURED_PITCH = math.radians(0.698)
MEASURED_ROLL = math.radians(-0.516)

# base_footprint <- camera_optical, as tf reports it
URDF = np.eye(4)
URDF[:3, :3] = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
URDF[:3, 3] = [0.390, 0.060, 0.373]


def test_measured_normal_round_trips_through_pitch_and_roll():
    n = measured_floor_normal(MEASURED_PITCH, MEASURED_ROLL)
    assert math.isclose(math.atan2(-n[0], n[2]), MEASURED_PITCH, abs_tol=1e-12)
    assert math.isclose(math.atan2(n[1], n[2]), MEASURED_ROLL, abs_tol=1e-12)
    assert math.isclose(float(np.linalg.norm(n)), 1.0, abs_tol=1e-12)


def test_a_level_floor_needs_no_correction():
    assert np.allclose(correction_matrix(0.0, 0.0), np.eye(3), atol=1e-12)


def test_correction_is_a_proper_rotation():
    c = correction_matrix(MEASURED_PITCH, MEASURED_ROLL)
    assert np.allclose(c @ c.T, np.eye(3), atol=1e-12)
    assert math.isclose(float(np.linalg.det(c)), 1.0, abs_tol=1e-12)


def test_correction_levels_the_measured_normal():
    """The whole point: the measured tilt must map onto +z."""
    n = measured_floor_normal(MEASURED_PITCH, MEASURED_ROLL)
    c = correction_matrix(MEASURED_PITCH, MEASURED_ROLL)
    assert np.allclose(c @ n, [0.0, 0.0, 1.0], atol=1e-12)


def test_correction_size_matches_the_measurement():
    """0.868 deg total, as reported by floor_plane_check.py."""
    c = correction_matrix(MEASURED_PITCH, MEASURED_ROLL)
    angle = math.degrees(math.acos(np.clip((np.trace(c) - 1) / 2, -1, 1)))
    assert math.isclose(angle, 0.868, abs_tol=0.002)


def test_levelling_rotation_handles_an_already_level_normal():
    assert np.allclose(levelling_rotation(np.array([0.0, 0.0, 1.0])), np.eye(3))


def test_correction_leaves_the_translation_alone():
    """Rotating the whole transform would swing the camera's position by 8 mm.

    The height is known to be right (measured -0.0246 m against -0.025 m from
    the tape), so only the rotation may change.
    """
    c = correction_matrix(MEASURED_PITCH, MEASURED_ROLL)
    out = correct_transform(URDF, c)
    assert np.allclose(out[:3, 3], URDF[:3, 3], atol=1e-15)
    assert not np.allclose(out[:3, :3], URDF[:3, :3])
    # and the mistake it guards against is a real one, not a hypothetical:
    # multiplying the whole 4x4 swings the camera's position about the origin
    c4 = np.eye(4)
    c4[:3, :3] = c
    swung = c4 @ URDF
    moved = float(np.linalg.norm(swung[:3, 3] - URDF[:3, 3]))
    assert moved > 0.005, f'only {1000 * moved:.1f} mm -- is the lever arm right?'


def test_correction_recovers_the_true_orientation():
    """Bake the error into a transform, then correct it back out."""
    c = correction_matrix(MEASURED_PITCH, MEASURED_ROLL)
    mismounted = URDF.copy()
    mismounted[:3, :3] = np.linalg.inv(c) @ URDF[:3, :3]
    fixed = correct_transform(mismounted, c)
    assert np.allclose(fixed[:3, :3], URDF[:3, :3], atol=1e-12)


def test_uncorrected_transform_makes_a_level_floor_look_tilted():
    """Shows the error is real: through the mismounted transform, a level floor
    comes back with exactly the pitch and roll that were measured."""
    c = correction_matrix(MEASURED_PITCH, MEASURED_ROLL)
    mismounted = URDF.copy()
    mismounted[:3, :3] = np.linalg.inv(c) @ URDF[:3, :3]

    n = apparent_floor_normal(assumed=mismounted, true=URDF)
    assert math.isclose(math.degrees(math.atan2(-n[0], n[2])), 0.698, abs_tol=0.01)
    assert math.isclose(math.degrees(math.atan2(n[1], n[2])), -0.516, abs_tol=0.01)

    # with the correction applied it comes back level
    n = apparent_floor_normal(assumed=correct_transform(mismounted, c), true=URDF)
    assert np.allclose(n, [0.0, 0.0, 1.0], atol=1e-12)


@pytest.mark.parametrize('distance,expected_mm', [(1.45, 22), (2.0, 30), (3.0, 45)])
def test_the_error_this_removes(distance, expected_mm):
    """Why it matters: 0.868 deg against a 30 mm pocket-height tolerance."""
    error_mm = 1000 * distance * math.tan(math.radians(0.868))
    assert math.isclose(error_mm, expected_mm, abs_tol=1.5)


def test_params_file_carries_the_measurement():
    """The shipped parameters must be the measured values, and enabled."""
    import pathlib

    import yaml
    params = yaml.safe_load(
        (pathlib.Path(__file__).resolve().parent.parent / 'config' / 'params.yaml').read_text())
    block = params['/**']['ros__parameters']['frames']['extrinsic_correction']
    assert block['enabled'] is True
    assert math.isclose(block['measured_floor_pitch'], MEASURED_PITCH, abs_tol=1e-5)
    assert math.isclose(block['measured_floor_roll'], MEASURED_ROLL, abs_tol=1e-5)


# ------------------------------------------------- why the floor is a Plane

def _fit(points):
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vt[-1]
    if normal[2] < 0:
        normal = -normal
    return Plane(normal=normal, offset=-float(normal @ centroid))


@pytest.mark.parametrize('error_deg', [0.0, 0.868, 3.0])
def test_plane_heights_are_invariant_to_an_extrinsic_rotation(error_deg):
    """The design claim behind `Plane`, and it is exact, not approximate.

    An extrinsic rotation error turns the whole observed scene -- floor and
    pallet together. Rotations are isometries, so the perpendicular distance
    from a point to a plane is unchanged. Measure pocket heights that way and
    they survive an uncalibrated, drifting or freshly-knocked camera mount.
    """
    points, truth = S.pallet_cloud(distance=2.0, noise=0.0)
    probe = truth['pocket_centers'][0]
    true_height = probe[2] - truth['floor_z']
    floor_points = points[np.abs(points[:, 2] - truth['floor_z']) < 0.05]

    wrong = np.linalg.inv(correction_matrix(math.radians(error_deg * 0.8),
                                            math.radians(-error_deg * 0.6)))
    observed_floor = floor_points @ wrong.T
    observed_probe = probe @ wrong.T

    perpendicular = float(_fit(observed_floor).height_of(observed_probe))
    assert math.isclose(perpendicular, true_height, abs_tol=2e-3), (
        f'{error_deg} deg of extrinsic error leaked into a plane-relative height')


def test_a_z_subtraction_is_not_invariant():
    """The counter-example that makes the test above worth having."""
    points, truth = S.pallet_cloud(distance=2.0, noise=0.0)
    probe = truth['pocket_centers'][0]
    true_height = probe[2] - truth['floor_z']
    floor_points = points[np.abs(points[:, 2] - truth['floor_z']) < 0.05]

    wrong = np.linalg.inv(correction_matrix(math.radians(3.0), math.radians(-2.0)))
    z_based = (probe @ wrong.T)[2] - (floor_points @ wrong.T)[:, 2].mean()
    assert abs(z_based - true_height) > 5e-3, (
        'expected a z-subtraction to carry the extrinsic error, but it did not')


def test_plane_height_of_handles_single_points_and_arrays():
    plane = Plane(normal=np.array([0.0, 0.0, 1.0]), offset=0.025)
    assert math.isclose(float(plane.height_of(np.array([2.0, 0.0, 0.047]))), 0.072,
                        abs_tol=1e-12)
    many = plane.height_of(np.array([[2.0, 0.0, 0.047], [2.0, 0.0, 0.119]]))
    assert np.allclose(many, [0.072, 0.144], atol=1e-12)
    assert math.isclose(plane.height_on_axis, -0.025, abs_tol=1e-12)
