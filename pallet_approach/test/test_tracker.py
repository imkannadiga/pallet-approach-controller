"""Tests for tracker.py -- the memory between frames.

Two groups. The association tests are about not being fooled: by a single good
frame, by an outlier, by a second object, by the robot's own motion. The
cross-check tests are about the LiDAR, and the one that matters most is that
"too few returns to judge" never reads as "the sensors agree".
"""

import math

import numpy as np
import pytest

from pallet_approach.approach import wrap_angle as wrap
from pallet_approach.model import PalletEstimate, Plane, Pocket, Pose2D
from pallet_approach.tracker import Tracker, cross_check_face

PARAMS = {
    'window': 10, 'min_window_frames': 5, 'confirm_frames': 5,
    'min_confidence': 0.6, 'max_jump': 0.15, 'max_turn': 0.20,
    'require_same_side': True,
    'floor_z': -0.025, 'pallet_height': 0.145,
    'search_ahead': 0.60, 'search_behind': 1.00, 'half_width': 0.70,
    'floor_margin': 0.03,
    'height_margin': 0.05, 'min_points': 20, 'max_offset': 0.10,
}
FLOOR = Plane(np.array([0.0, 0.0, 1.0]), 0.025)       # floor at z = -0.025


def estimate(distance=3.0, lateral=0.0, yaw_offset=0.0, side='short',
             spacing=0.365):
    """A pallet `distance` ahead, square on unless `yaw_offset` says otherwise."""
    normal = np.array([-math.cos(yaw_offset), -math.sin(yaw_offset), 0.0])
    center = np.array([distance, lateral, 0.050])
    along = np.array([-normal[1], normal[0], 0.0])
    return PalletEstimate(
        position=center,
        normal=normal,
        yaw=math.atan2(normal[1], normal[0]),
        pockets=(Pocket(center=center - along * spacing / 2, width=0.23, height=0.10),
                 Pocket(center=center + along * spacing / 2, width=0.23, height=0.10)),
        face_inlier_count=140,
        floor=FLOOR,
        side=side,
        face_width=0.795,
    )


def lidar_face(distance=3.0, lateral=0.0, heights=(0.05, 0.12), columns=40):
    """Returns spread across a face at `distance`, as the Ouster would see it.

    Two beams by forty azimuth columns, which is roughly what the geometry
    predicts at 3 m (D13 measured the beam spacing; 1.386 deg apart is 72 mm of
    vertical step at 3 m, so one or two land inside a 0.03-0.145 m window).
    """
    y = np.linspace(lateral - 0.35, lateral + 0.35, columns)
    points = [[distance, yi, h - 0.025] for h in heights for yi in y]
    return np.array(points)


def ground_in_front(distance=3.0, columns=200):
    """Floor returns around the face's depth. The case that would make a naive
    cross-check pass on bare ground: these lie almost exactly on the vertical
    plane's own extension, so only the height window excludes them."""
    x = np.linspace(distance - 0.1, distance + 0.1, 20)
    y = np.linspace(-0.5, 0.5, columns // 20)
    grid = np.array([[xi, yi, -0.025] for xi in x for yi in y])
    return grid


def run(tracker, frames, start=0.0, step=0.2, robot=None, lidar=None):
    """Offer `frames` estimates 200 ms apart (perception's 5 Hz)."""
    track = None
    for i, est in enumerate(frames):
        pose = robot(i) if callable(robot) else (robot or Pose2D(0.0, 0.0, 0.0))
        track = tracker.update(est, pose, start + i * step, lidar_points=lidar)
    return track


# ---------------------------------------------------------- confirmation

def test_one_good_frame_is_not_a_confirmed_pallet():
    """The single most important test here, as in test_coordinator."""
    tracker = Tracker(PARAMS)
    tracker.update(estimate(), Pose2D(0.0, 0.0, 0.0), 0.0)
    assert not tracker.confirmed
    assert tracker.status() == 'CONFIRMING'


def test_five_consistent_frames_confirm():
    tracker = Tracker(PARAMS)
    run(tracker, [estimate(3.0 + 0.001 * i) for i in range(5)])
    assert tracker.confirmed
    assert tracker.status() == 'CONFIRMED'
    assert tracker.confidence == pytest.approx(1.0)


def test_confidence_is_zero_until_the_window_means_something():
    """One accepted frame out of one offered is not 100 % confidence."""
    tracker = Tracker(PARAMS)
    for n in range(1, PARAMS['min_window_frames']):
        tracker.update(estimate(), Pose2D(0.0, 0.0, 0.0), 0.2 * n)
        assert tracker.confidence == 0.0, f'{n} frames reported a confidence'
    tracker.update(estimate(), Pose2D(0.0, 0.0, 0.0), 1.0)
    assert tracker.confidence > 0.0


def test_a_blank_frame_counts_against_the_confidence():
    """Otherwise the confidence freezes at its last value when the pallet
    leaves the view, and the coordinator reads a stale 1.0 as current."""
    tracker = Tracker(PARAMS)
    run(tracker, [estimate() for _ in range(5)])
    assert tracker.confidence == pytest.approx(1.0)
    for i in range(3):
        tracker.update(None, Pose2D(0.0, 0.0, 0.0), 2.0 + 0.2 * i)
    assert tracker.confidence < 1.0
    assert not tracker.confirmed
    assert tracker.status() == 'LOST'


def test_an_outlier_breaks_the_consecutive_run_without_losing_the_track():
    tracker = Tracker(PARAMS)
    run(tracker, [estimate() for _ in range(5)])
    before = tracker.track.center.copy()
    tracker.update(estimate(distance=3.5), Pose2D(0.0, 0.0, 0.0), 2.0)
    assert tracker.consecutive == 0
    assert not tracker.confirmed
    assert np.allclose(tracker.track.center, before), (
        'the track must not move to an estimate it rejected')
    assert 'from the track' in tracker.reason


def test_a_pallet_that_appears_to_turn_over_is_rejected():
    """A pallet does not change which face it shows in 200 ms (D36)."""
    tracker = Tracker(PARAMS)
    run(tracker, [estimate(side='short') for _ in range(5)])
    tracker.update(estimate(side='long'), Pose2D(0.0, 0.0, 0.0), 2.0)
    assert tracker.consecutive == 0
    assert 'side changed' in tracker.reason


def test_an_unknown_side_does_not_count_as_a_change():
    """Classification can be switched off (pallet.classify.enabled), and then
    every estimate reads 'unknown'. That is an absence of evidence about the
    side, not evidence that it changed."""
    tracker = Tracker(PARAMS)
    run(tracker, [estimate(side='short') for _ in range(5)])
    tracker.update(estimate(side='unknown'), Pose2D(0.0, 0.0, 0.0), 2.0)
    assert tracker.consecutive == 6
    assert tracker.track.side == 'short', 'and it must not erase what is known'


# -------------------------------------------------- the robot's own motion

def test_a_stationary_pallet_stays_put_while_the_robot_drives():
    """The reason the track lives in odom. The robot closes 1.0 m over five
    frames and the pallet's measured distance shrinks with it; in odom the
    pallet has not moved, and every frame must be accepted."""
    tracker = Tracker(PARAMS)
    for i in range(6):
        travelled = 0.2 * i
        tracker.update(estimate(distance=3.0 - travelled),
                       Pose2D(travelled, 0.0, 0.0), 0.2 * i)
    assert tracker.consecutive == 6
    assert tracker.confirmed
    assert tracker.track.center[0] == pytest.approx(3.0, abs=1e-9)


def test_the_motion_has_to_be_taken_out_or_driving_breaks_the_track():
    """The counterpart: if the robot's pose were ignored, the same drive would
    look like a pallet moving 0.2 m a frame -- past the 0.15 m gate. This test
    is what makes the one above mean something."""
    tracker = Tracker(PARAMS)
    accepted = 0
    for i in range(6):
        tracker.update(estimate(distance=3.0 - 0.2 * i),
                       Pose2D(0.0, 0.0, 0.0), 0.2 * i)   # pose never updated
        accepted += 1 if tracker.consecutive else 0
    assert tracker.consecutive == 0
    assert 'from the track' in tracker.reason


def test_a_turning_robot_is_also_taken_out():
    """Yaw as well as position: the robot turns 10 deg and the pallet's normal
    turns 10 deg the other way in the robot's frame. In odom neither moved."""
    tracker = Tracker(PARAMS)
    for i in range(6):
        turn = math.radians(2.0 * i)
        # The pallet's centre seen from the turned robot: (3, 0) rotated by
        # -turn. (Until 2026-10-20 this held it at (3, 0) in the robot's frame,
        # which slides it 0.52 m along an arc in odom; frame-to-frame gating
        # never noticed, gating against the fused track does.)
        tracker.update(estimate(distance=3.0 * math.cos(turn),
                                lateral=-3.0 * math.sin(turn), yaw_offset=-turn),
                       Pose2D(0.0, 0.0, turn), 0.2 * i)
    assert tracker.consecutive == 6
    assert np.allclose(tracker.track.center[:2], [3.0, 0.0], atol=1e-9)
    assert tracker.track.yaw == pytest.approx(math.pi)


# ----------------------------------------------------- the LiDAR cross-check

def test_the_lidar_agreeing_with_the_camera():
    check = cross_check_face(estimate(3.0), lidar_face(3.0), PARAMS)
    assert check.evaluated and check.agrees
    assert check.offset == pytest.approx(0.0, abs=1e-9)
    assert check.points == 80


def test_the_lidar_seeing_the_face_somewhere_else_disagrees():
    """The failure this check exists for: the camera fits a plane where there
    is nothing, and a second sensor on a different physical principle says so.

    0.4 m is also the case that found the first version of this function wrong.
    The depth window was +-0.15 m, meant to keep a loaded pallet's cargo out --
    and it excluded the disagreement the check exists to catch, so a camera
    hallucinating a face 0.4 m from the real one read as NOT EVALUATED.
    """
    check = cross_check_face(estimate(3.0), lidar_face(3.4), PARAMS)
    assert check.evaluated and not check.agrees
    assert check.blocking
    assert '-400 mm' in check.detail, (
        'the sign says which way: negative is further from the robot than the '
        'camera claimed')


def test_a_small_disagreement_is_tolerated():
    """The LiDAR's own URDF chain is 41 mm out. A tight limit here would be
    measuring our own extrinsics, not checking the camera."""
    check = cross_check_face(estimate(3.0), lidar_face(3.05), PARAMS)
    assert check.evaluated and check.agrees


def test_too_few_returns_is_not_agreement():
    """The most dangerous way for this to be written. At long range or on a
    thin face there may be nothing to compare, and reporting that as agreement
    would make the cross-check vanish exactly when it is hardest."""
    check = cross_check_face(estimate(3.0), lidar_face(3.0, columns=5), PARAMS)
    assert not check.evaluated
    assert not check.agrees
    assert not check.blocking, 'and it must not block a run either'
    assert 'need 20' in check.detail


def test_no_lidar_at_all_is_not_agreement():
    for empty in (None, np.zeros((0, 3))):
        check = cross_check_face(estimate(3.0), empty, PARAMS)
        assert not check.evaluated and not check.blocking


def test_the_floor_cannot_pass_as_a_face():
    """Floor returns in front of the face lie almost exactly on the vertical
    plane's own extension, so without the height window this check would agree
    with the camera about a face that is bare ground."""
    check = cross_check_face(estimate(3.0), ground_in_front(3.0), PARAMS)
    assert not check.evaluated, (
        f'{check.points} floor returns were counted as face returns')


def test_cargo_behind_the_face_does_not_drag_the_statistic():
    """The reason the statistic is the nearest surface and not an average. A
    loaded pallet returns from its cargo as well as its face, and the face is
    what is being checked."""
    cargo = lidar_face(3.0 + 0.5)
    check = cross_check_face(estimate(3.0), np.vstack([cargo, lidar_face(3.0)]),
                             PARAMS)
    assert check.evaluated and check.agrees
    assert check.offset == pytest.approx(0.0, abs=1e-9)


def test_an_obstacle_in_front_of_the_face_is_a_disagreement():
    """Positive offsets mean the LiDAR sees something closer than the camera's
    plane. Between us and the pallet is exactly where that matters, and saying
    so is the right answer rather than a false alarm."""
    check = cross_check_face(estimate(3.0), lidar_face(3.0 - 0.3), PARAMS)
    assert check.evaluated and not check.agrees
    assert '+300 mm' in check.detail


def test_a_disagreement_too_large_to_see_reads_as_not_evaluated():
    """The documented limit of the window. Beyond `search_behind` nothing in
    view is the face, so there is nothing to compare -- and an error that gross
    is caught by the cluster size and ROI checks instead. It must not read as
    agreement."""
    check = cross_check_face(estimate(3.0), lidar_face(3.0 + 2.0), PARAMS)
    assert not check.evaluated
    assert not check.agrees


def test_returns_off_the_end_of_the_face_are_outside_the_window():
    """A doorframe beside the pallet, at the same depth."""
    aside = lidar_face(3.0, lateral=2.0)
    check = cross_check_face(estimate(3.0), np.vstack([aside, lidar_face(3.0)]),
                             PARAMS)
    assert check.points == 80


def test_the_cross_check_follows_a_turned_pallet():
    """The window is measured along the face and perpendicular to it, so a
    pallet at an angle is still covered. Measured in the rotated frame, not by
    widening the box until it works."""
    yaw = math.radians(20.0)
    est = estimate(3.0, yaw_offset=yaw)
    normal = est.normal
    along = np.array([-normal[1], normal[0], 0.0])
    center = est.position
    offsets = np.linspace(-0.35, 0.35, 40)
    points = np.array([center + along * o + np.array([0.0, 0.0, h])
                       for h in (0.0, 0.07) for o in offsets])
    check = cross_check_face(est, points, PARAMS)
    assert check.evaluated and check.agrees


def test_a_disagreeing_lidar_blocks_confirmation():
    """The point of the whole thing: five good camera frames do not confirm a
    pallet the LiDAR says is not there."""
    tracker = Tracker(PARAMS)
    run(tracker, [estimate() for _ in range(6)], lidar=lidar_face(3.4))
    assert tracker.track is not None
    assert not tracker.confirmed
    assert tracker.status() == 'REJECTED'
    assert 'LiDAR disagrees' in tracker.reason


def test_an_unevaluated_cross_check_does_not_block_confirmation():
    """Otherwise no run could ever start at a range where the beams miss."""
    tracker = Tracker(PARAMS)
    run(tracker, [estimate() for _ in range(6)], lidar=lidar_face(3.0, columns=5))
    assert tracker.confirmed


def test_a_fallback_floor_is_used_when_perception_fitted_none():
    """Legitimate at the standoff, where almost no floor is left in frame."""
    est = estimate(3.0)
    no_floor = PalletEstimate(
        position=est.position, normal=est.normal, yaw=est.yaw,
        pockets=est.pockets, face_inlier_count=est.face_inlier_count,
        floor=None, side='short', face_width=0.795)
    check = cross_check_face(no_floor, lidar_face(3.0), PARAMS)
    assert check.evaluated and check.agrees


# ------------------------------------------------------------------ reset

def test_reset_forgets_everything():
    tracker = Tracker(PARAMS)
    run(tracker, [estimate() for _ in range(6)])
    tracker.reset()
    assert tracker.track is None
    assert not tracker.confirmed
    assert tracker.confidence == 0.0
    assert tracker.status() == 'SEARCHING'


# ----------------------------------------------------- fusion (D59)

def test_frame_noise_on_the_normal_averages_out():
    """+-0.04 rad of normal wobble, the size measured in the sim, shrinks in
    the fused yaw by about the square root of the frames averaged."""
    rng = np.random.default_rng(0)
    tracker = Tracker(PARAMS)
    wobble = rng.uniform(-0.04, 0.04, 40)
    for i, w in enumerate(wobble):
        tracker.update(estimate(yaw_offset=w), Pose2D(0.0, 0.0, 0.0), 0.2 * i)
    assert abs(wrap(tracker.track.yaw - math.pi)) < 0.015


def test_closer_frames_count_more():
    tracker = Tracker(PARAMS)
    # The same pallet at (3, 0) in odom, seen from 1.0 m and from 2.0 m away,
    # with an error of +50 mm in the far frame only.
    tracker.update(estimate(distance=2.0, lateral=0.05), Pose2D(1.0, 0.0, 0.0), 0.0)
    tracker.update(estimate(distance=1.0), Pose2D(2.0, 0.0, 0.0), 0.2)
    # weights 1/4 and 1 (the far one decayed by memory): mostly the near one
    assert 0.0 < tracker.track.center[1] < 0.012


def test_the_memory_bounds_how_far_back_the_average_reaches():
    params = dict(PARAMS, fusion_memory=0.5)
    tracker = Tracker(params)
    for i in range(3):
        tracker.update(estimate(lateral=0.10), Pose2D(0.0, 0.0, 0.0), 0.2 * i)
    for i in range(3, 13):
        tracker.update(estimate(lateral=0.0), Pose2D(0.0, 0.0, 0.0), 0.2 * i)
    assert abs(tracker.track.center[1]) < 0.001
