"""Tests for approach.py -- the standoff and line-up geometry.

This module is implemented, so these all pass. They are here as the worked
examples: if you change the approach strategy, these numbers say what the old
strategy guaranteed.
"""

import math

import numpy as np
import pytest
import synthetic as S

from pallet_approach.approach import (
    distance_between,
    lateral_offset,
    outward_normal,
    pocket_center,
    range_to_face,
    standoff_pose,
    transform_pose,
    wrap_angle,
    yaw_misalignment,
)
from pallet_approach.model import PalletEstimate, Plane, Pocket, Pose2D

STANDOFF = 1.45
# LINEUP_EXTRA went with the two-leg approach (D38).


def estimate_at(distance=2.0, lateral=0.0, yaw=0.0):
    """A PalletEstimate matching what the synthetic pallet would produce."""
    _, truth = S.pallet_cloud(distance=distance, lateral=lateral, yaw=yaw)
    pockets = tuple(
        Pocket(center=c, width=S.POCKET_WIDTH, height=S.POCKET_HEIGHT)
        for c in truth['pocket_centers']
    )
    return PalletEstimate(
        position=truth['face_center'],
        normal=truth['normal'],
        yaw=math.atan2(truth['normal'][1], truth['normal'][0]),
        pockets=pockets,
        face_inlier_count=2000,
        floor=Plane(normal=np.array([0.0, 0.0, 1.0]), offset=-truth['floor_z']),
    )


# --------------------------------------------------------------- wrap_angle

@pytest.mark.parametrize('angle,expected', [
    (0.0, 0.0),
    (math.pi, math.pi),
    (-math.pi, -math.pi),      # atan2's range includes -pi; same angle as +pi
    (3 * math.pi, math.pi),
    (math.radians(359), math.radians(-1)),
])
def test_wrap_angle(angle, expected):
    assert math.isclose(wrap_angle(angle), expected, abs_tol=1e-9)


def test_wrap_angle_takes_the_short_way_round():
    """+179 and -179 degrees are 2 degrees apart, not 358."""
    error = wrap_angle(math.radians(179) - math.radians(-179))
    assert math.isclose(abs(error), math.radians(2), abs_tol=1e-9)


# ----------------------------------------------------------- outward_normal

def test_outward_normal_flattens_to_the_ground_plane():
    n = outward_normal(np.array([-1.0, 0.0, 0.3]))
    assert math.isclose(n[2], 0.0, abs_tol=1e-12)
    assert math.isclose(float(np.linalg.norm(n)), 1.0, abs_tol=1e-12)


def test_outward_normal_flips_a_normal_pointing_away():
    """RANSAC's sign is arbitrary, so a normal pointing away must be flipped."""
    face = np.array([2.0, 0.0, 0.0])
    away = np.array([1.0, 0.0, 0.0])        # points away from the robot at the origin
    assert outward_normal(away, face_point=face)[0] < 0.0
    toward = np.array([-1.0, 0.0, 0.0])
    assert outward_normal(toward, face_point=face)[0] < 0.0


def test_outward_normal_rejects_a_vertical_normal():
    with pytest.raises(ValueError):
        outward_normal(np.array([0.0, 0.0, 1.0]))


# ------------------------------------------------------------ pocket_center

def test_pocket_center_is_the_midpoint():
    est = estimate_at()
    c = pocket_center(est.pockets)
    assert np.allclose(c[:2], [2.0, 0.0], atol=1e-9)


def test_pocket_center_falls_back_when_pockets_are_missing():
    fallback = np.array([2.0, 0.1, 0.0])
    assert np.allclose(pocket_center((), fallback=fallback), fallback)
    with pytest.raises(ValueError):
        pocket_center(())


# ----------------------------------------------------------------- standoff

def test_standoff_is_on_the_normal_at_the_right_distance():
    est = estimate_at(distance=2.0)
    s = standoff_pose(pocket_center(est.pockets), est.normal, STANDOFF)
    assert math.isclose(s.x, 2.0 - STANDOFF, abs_tol=1e-9)
    assert math.isclose(s.y, 0.0, abs_tol=1e-9)
    assert math.isclose(s.yaw, 0.0, abs_tol=1e-9)      # facing +x, at the pallet


def test_standoff_rejects_a_non_positive_distance():
    est = estimate_at()
    with pytest.raises(ValueError):
        standoff_pose(pocket_center(est.pockets), est.normal, 0.0)


@pytest.mark.parametrize('yaw_deg', [-30, -15, 0, 15, 30])
def test_standoff_distance_is_exact_at_any_pallet_angle(yaw_deg):
    est = estimate_at(distance=2.5, yaw=math.radians(yaw_deg))
    c = pocket_center(est.pockets)
    s = standoff_pose(c, est.normal, STANDOFF)
    assert math.isclose(math.hypot(s.x - c[0], s.y - c[1]), STANDOFF, abs_tol=1e-9)


@pytest.mark.parametrize('yaw_deg', [-30, -15, 0, 15, 30])
def test_robot_at_the_standoff_is_perfectly_aligned(yaw_deg):
    """The whole point of the standoff: no residual yaw or lateral error there."""
    est = estimate_at(distance=2.5, yaw=math.radians(yaw_deg))
    c = pocket_center(est.pockets)
    s = standoff_pose(c, est.normal, STANDOFF)
    assert math.isclose(yaw_misalignment(s.yaw, est.normal), 0.0, abs_tol=1e-9)
    assert math.isclose(lateral_offset(s, c, est.normal), 0.0, abs_tol=1e-9)
    assert math.isclose(range_to_face(s, c, est.normal), STANDOFF, abs_tol=1e-9)


def test_standoff_keeps_the_whole_pocket_opening_in_frame():
    """The binding constraint on the standoff.

    The camera is 0.390 m ahead of base_footprint and 0.398 m above the floor
    (tape-measured 2026-10-14), level, with a 20.76 deg vertical half-FOV.
    Features low on the pallet drop out of the bottom of the image as the robot
    closes in. It is not enough for the pocket *centre* to be visible: the
    GO/NO-GO rules measure the opening height against a 0.030 m tolerance, so
    the bottom lip at 0.022 m above the floor has to be in frame too, or a sound
    pallet reads short and gets a spurious NO-GO.
    """
    camera_ahead = 0.390
    tan_half_fov = (S.IMAGE_HEIGHT / 2) / S.FY
    pocket_bottom = S.DECK_THICKNESS                       # 0.022 m

    camera_range = STANDOFF - camera_ahead
    lowest_visible = S.CAMERA_HEIGHT_ABOVE_FLOOR - camera_range * tan_half_fov
    margin = pocket_bottom - lowest_visible

    assert lowest_visible <= pocket_bottom, (
        f'standoff {STANDOFF} m clips the bottom of the pocket opening')
    # 1.40 m left only 7 mm, which tyre compression or a little pitch under
    # braking would eat. Do not tighten the standoff back below this.
    assert margin >= 0.015, f'only {margin * 1000:.0f} mm of margin on the lip'


def test_the_standoff_is_not_needlessly_far():
    """The other side of it: a standoff much further back loses pocket
    resolution for nothing. 0.100 m of opening should still be ~85 px tall."""
    pixels = S.POCKET_HEIGHT * S.FY / (STANDOFF - 0.390)
    assert pixels > 80, f'pocket opening only {pixels:.0f} px tall at the standoff'


# ---------------------------------------------------------- error quantities

def test_lateral_offset_sign_and_magnitude():
    est = estimate_at(distance=2.0)
    c = pocket_center(est.pockets)
    left = Pose2D(x=0.7, y=0.10, yaw=0.0)
    right = Pose2D(x=0.7, y=-0.10, yaw=0.0)
    assert math.isclose(abs(lateral_offset(left, c, est.normal)), 0.10, abs_tol=1e-9)
    assert lateral_offset(left, c, est.normal) == pytest.approx(
        -lateral_offset(right, c, est.normal))


def test_yaw_misalignment_is_zero_when_square_on():
    est = estimate_at(distance=2.0)
    assert math.isclose(yaw_misalignment(0.0, est.normal), 0.0, abs_tol=1e-9)
    assert math.isclose(yaw_misalignment(math.radians(10), est.normal),
                        math.radians(-10), abs_tol=1e-9)


def test_range_to_face_decreases_as_the_robot_drives_in():
    est = estimate_at(distance=2.0)
    c = pocket_center(est.pockets)
    ranges = [range_to_face(Pose2D(x, 0.0, 0.0), c, est.normal) for x in (0.0, 0.5, 1.0)]
    assert ranges == sorted(ranges, reverse=True)
    assert math.isclose(ranges[0], 2.0, abs_tol=1e-9)


# --------------------------------------------------------------- transforms

def test_transform_pose_composes_correctly():
    """base_link at (5, 5, 90 deg) in odom; a point 1 m ahead is at (5, 6)."""
    out = transform_pose(Pose2D(1.0, 0.0, 0.0), Pose2D(5.0, 5.0, math.pi / 2))
    assert np.allclose([out.x, out.y], [5.0, 6.0], atol=1e-9)
    assert math.isclose(out.yaw, math.pi / 2, abs_tol=1e-9)


def test_transform_pose_identity():
    pose = Pose2D(1.2, -0.3, 0.7)
    out = transform_pose(pose, Pose2D(0.0, 0.0, 0.0))
    assert np.allclose(out.as_array(), pose.as_array(), atol=1e-12)


def test_targets_survive_a_frame_change():
    """Computing a target in base_link then converting to odom must match
    computing it directly in odom. The tracker relies on this: it places each
    estimate in odom with the robot's pose and compares them there (D35)."""
    est = estimate_at(distance=2.0, yaw=math.radians(15))
    base_in_odom = Pose2D(3.0, -1.0, math.radians(40))

    standoff_b = standoff_pose(pocket_center(est.pockets, est.position),
                               est.normal, STANDOFF)
    converted = [transform_pose(standoff_b, base_in_odom)]

    # the same estimate expressed in odom
    cos_a, sin_a = math.cos(base_in_odom.yaw), math.sin(base_in_odom.yaw)
    rot = np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]])
    shift = np.array([base_in_odom.x, base_in_odom.y, 0.0])
    est_odom = PalletEstimate(
        position=rot @ est.position + shift,
        normal=rot @ est.normal,
        yaw=est.yaw + base_in_odom.yaw,
        pockets=tuple(Pocket(rot @ p.center + shift, p.width, p.height) for p in est.pockets),
    )
    direct = [standoff_pose(pocket_center(est_odom.pockets, est_odom.position),
                            est_odom.normal, STANDOFF)]

    for a, b in zip(converted, direct):
        assert np.allclose([a.x, a.y], [b.x, b.y], atol=1e-9)
        assert math.isclose(wrap_angle(a.yaw - b.yaw), 0.0, abs_tol=1e-9)


def test_distance_between_ignores_heading():
    assert math.isclose(distance_between(Pose2D(0, 0, 0), Pose2D(3, 4, 1.0)), 5.0, abs_tol=1e-12)
