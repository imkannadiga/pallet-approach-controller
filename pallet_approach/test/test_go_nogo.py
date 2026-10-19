"""The safety rules: what the robot refuses to do, and why.

Read the test names as the specification. They were written before the rules
were, and the rules were written to satisfy them. Note how many of them assert a
NO-GO: that asymmetry is deliberate and it is the thing to defend in an
interview. In particular `test_evaluate_refuses_when_data_is_missing` and
`test_no_false_go_in_unsafe_scenarios` are the tests that stand behind the
"0 false GOs" claim, so they must never be relaxed to make a run pass.
"""

import math

import numpy as np
import pytest
import synthetic as S

from pallet_approach import go_nogo
from pallet_approach.model import PalletEstimate, Plane, Pocket, Pose2D


def make_estimate(distance=2.0, lateral=0.0, yaw=0.0, pocket_width=S.POCKET_WIDTH,
                  pocket_height=S.POCKET_HEIGHT, n_pockets=2, inliers=2000):
    _, truth = S.pallet_cloud(distance=distance, lateral=lateral, yaw=yaw)
    pockets = tuple(
        Pocket(center=c, width=pocket_width, height=pocket_height)
        for c in truth['pocket_centers'][:n_pockets]
    )
    return PalletEstimate(
        position=truth['face_center'],
        normal=truth['normal'],
        yaw=math.atan2(truth['normal'][1], truth['normal'][0]),
        pockets=pockets,
        face_inlier_count=inliers,
        floor=Plane(normal=np.array([0.0, 0.0, 1.0]), offset=-truth['floor_z']),
    )


def at_standoff(estimate, standoff=1.45):
    """The robot pose that the approach controller aims for."""
    from pallet_approach.approach import pocket_center, standoff_pose
    return standoff_pose(pocket_center(estimate.pockets, estimate.position),
                         estimate.normal, standoff)


# ----------------------------------------------------------------- each rule

def test_face_quality_passes_with_enough_inliers():
    # 140 is what the real pallet produces at 2 m; see config/params.yaml
    ok, reason = go_nogo.check_face_quality(make_estimate(inliers=140), min_inliers=60)
    assert ok
    assert isinstance(reason, str)


def test_face_quality_fails_on_a_thin_fit():
    ok, reason = go_nogo.check_face_quality(make_estimate(inliers=12), min_inliers=60)
    assert not ok
    assert '12' in reason or '60' in reason, 'the reason must name the numbers'


def test_pocket_count_needs_both_pockets():
    assert go_nogo.check_pocket_count(make_estimate(n_pockets=2), minimum=2)[0]
    assert not go_nogo.check_pocket_count(make_estimate(n_pockets=1), minimum=2)[0]
    assert not go_nogo.check_pocket_count(make_estimate(n_pockets=0), minimum=2)[0]


def test_pocket_dimensions_accept_a_nominal_epal_pallet():
    ok, _ = go_nogo.check_pocket_dimensions(
        make_estimate(), S.POCKET_WIDTH, S.POCKET_HEIGHT, 0.040, 0.030)
    assert ok


@pytest.mark.parametrize('width', [0.14, 0.32])
def test_pocket_dimensions_reject_a_wrong_opening(width):
    ok, reason = go_nogo.check_pocket_dimensions(
        make_estimate(pocket_width=width), S.POCKET_WIDTH, S.POCKET_HEIGHT, 0.040, 0.030)
    assert not ok
    assert reason


def test_pocket_dimensions_reject_a_squashed_opening():
    """A 60 mm high opening will not take a 40 mm fork with any margin."""
    ok, _ = go_nogo.check_pocket_dimensions(
        make_estimate(pocket_height=0.060), S.POCKET_WIDTH, S.POCKET_HEIGHT, 0.040, 0.030)
    assert not ok


def test_yaw_passes_when_square_on():
    estimate = make_estimate()
    ok, _ = go_nogo.check_yaw(estimate, at_standoff(estimate), max_yaw_error=0.087)
    assert ok


def test_yaw_fails_beyond_the_limit():
    """The robot is 10 degrees off square, the limit is 5."""
    estimate = make_estimate()
    pose = at_standoff(estimate)
    rotated = Pose2D(pose.x, pose.y, pose.yaw + math.radians(10))
    ok, reason = go_nogo.check_yaw(estimate, rotated, max_yaw_error=0.087)
    assert not ok
    assert 'deg' in reason.lower(), 'report degrees, not radians'


def test_lateral_offset_passes_on_the_centre_line():
    estimate = make_estimate()
    ok, _ = go_nogo.check_lateral_offset(estimate, at_standoff(estimate),
                                         max_offset=0.050)
    assert ok


def test_lateral_offset_fails_when_off_the_centre_line():
    """0.12 m off centre: a 0.100 m fork in a 0.2275 m pocket has only 0.064 m
    of clearance each side, so this would hit a block."""
    estimate = make_estimate()
    pose = at_standoff(estimate)
    shifted = Pose2D(pose.x, pose.y + 0.12, pose.yaw)
    ok, reason = go_nogo.check_lateral_offset(estimate, shifted, max_offset=0.050)
    assert not ok
    assert reason


def test_overhang_passes_for_a_bare_pallet(square_pallet):
    points, truth = square_pallet
    estimate = make_estimate()
    ok, _ = go_nogo.check_overhang(estimate, points, margin=0.020)
    assert ok


def test_overhang_fails_when_a_box_sticks_out(square_pallet):
    """A box on top, protruding 8 cm past the face, must be caught."""
    points, truth = square_pallet
    rng = np.random.default_rng(0)
    box = rng.uniform([1.92, -0.3, truth['floor_z'] + 0.15],
                      [2.30, 0.3, truth['floor_z'] + 0.45], size=(800, 3))
    ok, reason = go_nogo.check_overhang(make_estimate(), np.concatenate([points, box]),
                                        margin=0.020)
    assert not ok
    assert reason


def test_overhang_refuses_without_data():
    """An uncheckable safety rule is a NO-GO, never a pass."""
    ok, _ = go_nogo.check_overhang(make_estimate(), None, margin=0.020)
    assert not ok


def test_pocket_clearance_passes_for_an_open_pallet(square_pallet):
    points, _ = square_pallet
    ok, _ = go_nogo.check_pocket_clearance(make_estimate(), points, clearance_depth=0.10)
    assert ok


def test_pocket_clearance_fails_with_an_obstruction(square_pallet):
    """Something parked right in front of the left pocket."""
    points, truth = square_pallet
    estimate = make_estimate()
    pocket = estimate.pockets[0].center
    rng = np.random.default_rng(0)
    blocker = rng.uniform([pocket[0] - 0.08, pocket[1] - 0.06, pocket[2] - 0.04],
                          [pocket[0] - 0.01, pocket[1] + 0.06, pocket[2] + 0.04],
                          size=(300, 3))
    ok, reason = go_nogo.check_pocket_clearance(
        estimate, np.concatenate([points, blocker]), clearance_depth=0.10)
    assert not ok
    assert reason


def test_pocket_clearance_refuses_without_data():
    ok, _ = go_nogo.check_pocket_clearance(make_estimate(), None, clearance_depth=0.10)
    assert not ok


def test_stability_passes_on_consistent_estimates():
    history = [make_estimate(distance=2.0 + 0.002 * i) for i in range(5)]
    ok, _ = go_nogo.check_stability(history, 5, 0.030, 0.035)
    assert ok


def test_stability_fails_on_a_jumping_estimate():
    history = [make_estimate(distance=d) for d in (2.0, 2.0, 2.35, 2.0, 2.0)]
    ok, reason = go_nogo.check_stability(history, 5, 0.030, 0.035)
    assert not ok
    assert reason


def test_stability_fails_on_too_few_frames():
    """Not enough evidence yet is a NO-GO, not a pass."""
    ok, _ = go_nogo.check_stability([make_estimate()], 5, 0.030, 0.035)
    assert not ok


# ------------------------------------------------------------------ evaluate

def test_evaluate_says_go_on_a_clean_setup(go_nogo_params, square_pallet):
    points, _ = square_pallet
    estimate = make_estimate()
    history = [make_estimate(distance=2.0 + 0.001 * i) for i in range(5)]
    decision = go_nogo.evaluate(estimate, at_standoff(estimate), history,
                                go_nogo_params, load_points=points, scene_points=points)
    assert decision.go, decision.reason
    assert decision.checks, 'record which rules ran, for the Day 11 tables'


def test_evaluate_refuses_when_data_is_missing(go_nogo_params):
    """No estimate at all is the easiest false GO to write by accident."""
    decision = go_nogo.evaluate(None, Pose2D(0, 0, 0), [], go_nogo_params)
    assert not decision.go
    assert decision.reason


@pytest.mark.parametrize('scenario', ['rotated', 'offset', 'one_pocket', 'thin_face'])
def test_no_false_go_in_unsafe_scenarios(go_nogo_params, square_pallet, scenario):
    """The false-GO test, in miniature. The target for this count is zero.

    Day 11 runs the physical version of this on the robot; this is the one that
    runs on every commit.
    """
    points, _ = square_pallet
    estimate = make_estimate()
    pose = at_standoff(estimate)
    history = [make_estimate() for _ in range(5)]

    if scenario == 'rotated':
        pose = Pose2D(pose.x, pose.y, pose.yaw + math.radians(12))
    elif scenario == 'offset':
        pose = Pose2D(pose.x, pose.y + 0.15, pose.yaw)
    elif scenario == 'one_pocket':
        estimate = make_estimate(n_pockets=1)
        history = [estimate] * 5
    elif scenario == 'thin_face':
        estimate = make_estimate(inliers=20)
        history = [estimate] * 5

    decision = go_nogo.evaluate(estimate, pose, history, go_nogo_params,
                                load_points=points, scene_points=points)
    assert not decision.go, f'false GO in the {scenario} scenario'
    assert decision.reason, 'a NO-GO must always say why'
