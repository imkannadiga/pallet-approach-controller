"""Tests for insertion.py: the blind drive and its two-source stop.

Almost all of this is about D18 -- odometry and the LiDAR both say when to
stop, the earlier one wins, and a disagreement aborts. The cases that matter
are the ones where one source is absent or wrong, because that is when a stop
rule either saves the pallet or drives into it.
"""

import pathlib

import numpy as np
import pytest
import yaml

from pallet_approach.insertion import (
    face_distance_ahead,
    plan_insertion,
    stop_decision,
)

PARAMS_FILE = pathlib.Path(__file__).resolve().parent.parent / 'config' / 'params.yaml'


@pytest.fixture(scope='module')
def yaml_params():
    return yaml.safe_load(PARAMS_FILE.read_text())


@pytest.fixture(scope='module')
def insert(yaml_params):
    """The dict the controller node will hand these functions."""
    shared = yaml_params['/**']['ros__parameters']
    block = dict(yaml_params['approach_controller']['ros__parameters']['insert'])
    lidar = dict(block.pop('lidar'))
    lidar['floor_z'] = shared['floor_z']
    block['lidar'] = lidar
    block['sides'] = shared['pallet']['sides']
    block['carriage_x'] = shared['forks']['carriage_x']
    block['tine_length'] = shared['forks']['tine_length']
    block['standoff'] = shared['standoff_distance']
    return block


# ------------------------------------------------------------------- plan

def test_the_short_face_plan_matches_the_decision_record(insert):
    """D16: standoff 1.75, stop at 0.700, so 1.05 m of travel and 0.9125 m of
    depth -- 76 % of a 1.195 m pallet."""
    plan = plan_insertion(1.75, 'short', insert)
    assert plan.travel == pytest.approx(1.05)
    assert plan.depth == pytest.approx(0.9125)
    assert plan.depth / insert['sides']['short']['depth'] == pytest.approx(0.763, abs=0.005)


def test_the_long_face_stops_further_out(insert):
    """D21: the long face is only 0.795 m deep, so the short side's stop would
    drive the tips 0.118 m out of the back."""
    plan = plan_insertion(1.75, 'long', insert)
    assert plan.stop_face_x > plan_insertion(1.75, 'short', insert).stop_face_x
    assert plan.depth < insert['sides']['long']['depth']


def test_the_plan_uses_the_measured_standoff_not_the_nominal(insert):
    """The closing loop arrives within its range tolerance of 1.75 m, not at
    it. Planning from the nominal would spend that tolerance on the stop."""
    assert plan_insertion(1.72, 'short', insert).travel == pytest.approx(1.02)
    assert plan_insertion(1.78, 'short', insert).travel == pytest.approx(1.08)


def test_a_face_already_inside_the_stop_is_refused(insert):
    with pytest.raises(ValueError, match='at or inside'):
        plan_insertion(0.65, 'short', insert)


def test_an_impossible_travel_is_refused(insert):
    """More than max_travel means the robot is not at a standoff, so the number
    the plan is being built from is wrong."""
    with pytest.raises(ValueError, match='not at a standoff'):
        plan_insertion(4.0, 'short', insert)


def test_an_unknown_side_is_refused(insert):
    """'ambiguous' reaches here if the coordinator ever lets it. Refusing is
    the whole point of D22: there is no defensible geometry to insert against."""
    for side in ('ambiguous', 'unknown', 'nonsense'):
        with pytest.raises(ValueError, match='no geometry'):
            plan_insertion(1.75, side, insert)


def test_a_face_marked_not_insertable_is_refused(insert):
    """D20: whether a site allows long-side entry is policy, not geometry."""
    params = dict(insert)
    params['sides'] = {'long': dict(insert['sides']['long'], insertable=False)}
    with pytest.raises(ValueError, match='not permitted'):
        plan_insertion(1.75, 'long', params)


# ------------------------------------------------------- the LiDAR measurement

def face_returns(distance, columns=30, heights=(0.05, 0.12), floor_z=-0.025):
    y = np.linspace(-0.25, 0.25, columns)
    return np.array([[distance, yi, h + floor_z] for h in heights for yi in y])


def test_the_face_distance_is_measured(insert):
    distance, lateral, detail = face_distance_ahead(face_returns(1.20),
                                                    insert['lidar'])
    assert distance == pytest.approx(1.20, abs=1e-6)
    assert lateral == pytest.approx(0.0, abs=1e-6)
    assert '60 returns' in detail


def test_nothing_measurable_is_none_not_a_large_distance(insert):
    """The distinction that keeps the robot from driving on dead reckoning. A
    missing measurement reported as "the face is far away" is the one answer
    that must never come out of this."""
    for empty in (None, np.zeros((0, 3)), face_returns(1.2, columns=2)):
        distance, lateral, _ = face_distance_ahead(empty, insert['lidar'])
        assert distance is None
        assert np.isnan(lateral)


def test_a_stray_early_return_does_not_stop_the_insertion_short(insert):
    """A low percentile, not the minimum: one return 0.1 m early would
    otherwise fire the stop on every frame it appeared in."""
    points = np.vstack([face_returns(1.20), np.array([[1.10, 0.0, 0.05 - 0.025]])])
    distance, _, _ = face_distance_ahead(points, insert['lidar'])
    assert distance == pytest.approx(1.20, abs=0.01)


def test_the_floor_is_not_the_face(insert):
    floor = np.array([[x, y, -0.025] for x in np.linspace(0.6, 1.5, 20)
                      for y in (-0.2, 0.0, 0.2)])
    assert face_distance_ahead(floor, insert['lidar'])[0] is None


def test_something_above_the_pallet_is_not_the_face(insert):
    """A wall behind a 0.145 m pallet returns from well above it. The corridor
    stops just above the pallet so that those returns cannot become the
    measured face."""
    high = np.array([[1.0, y, 0.9] for y in np.linspace(-0.2, 0.2, 30)])
    assert face_distance_ahead(high, insert['lidar'])[0] is None


def test_the_robots_own_structure_is_excluded(insert):
    near = np.array([[0.30, y, 0.05 - 0.025] for y in np.linspace(-0.2, 0.2, 30)])
    assert face_distance_ahead(near, insert['lidar'])[0] is None


def test_an_obstacle_in_front_of_the_face_is_measured_as_the_surface(insert):
    """And that is correct: the corridor does not try to tell a pallet from a
    person's foot, because both should stop the robot."""
    points = np.vstack([face_returns(1.20), face_returns(0.90)])
    distance, _, _ = face_distance_ahead(points, insert['lidar'])
    assert distance == pytest.approx(0.90, abs=0.01)


def test_the_lateral_drift_is_reported(insert):
    """Logged, not acted on (D7 / ROADMAP stage 6). This is the number lab
    session C needs to decide whether insertion should track the face."""
    shifted = face_returns(1.20) + np.array([0.0, 0.04, 0.0])
    _, lateral, detail = face_distance_ahead(shifted, insert['lidar'])
    assert lateral == pytest.approx(0.04, abs=1e-6)
    assert '+40 mm' in detail


# ------------------------------------------------------------- the stop (D18)

def test_driving_continues_while_both_sources_agree_there_is_room(insert):
    plan = plan_insertion(1.75, 'short', insert)
    decision = stop_decision(0.40, 1.35, 0, plan, insert)
    assert not decision.stop and not decision.abort


def test_the_lidar_can_stop_first(insert):
    """It should, normally: the stop distance is a face distance, so the LiDAR
    measures the condition directly and odometry only infers it."""
    plan = plan_insertion(1.75, 'short', insert)
    decision = stop_decision(1.02, 0.700, 0, plan, insert)
    assert decision.stop and not decision.abort
    assert decision.source == 'lidar'


def test_odometry_can_stop_first(insert):
    plan = plan_insertion(1.75, 'short', insert)
    decision = stop_decision(1.05, 0.715, 0, plan, insert)
    assert decision.stop and not decision.abort
    assert decision.source == 'odometry'


def test_a_disagreement_at_the_stop_aborts(insert):
    """D18's second half, and the reason the two sources are kept separate. One
    of them is wrong, and driving forward is not how to find out which."""
    plan = plan_insertion(1.75, 'short', insert)
    decision = stop_decision(1.05, 0.830, 0, plan, insert)
    assert decision.stop and decision.abort
    assert 'apart' in decision.reason


def test_a_disagreement_inside_the_limit_completes(insert):
    plan = plan_insertion(1.75, 'short', insert)
    decision = stop_decision(1.05, 0.730, 0, plan, insert)
    assert decision.stop and not decision.abort


def test_the_earlier_stop_wins_even_when_it_is_the_lidar_by_a_long_way(insert):
    """An obstacle 0.3 m in front of the planned stop: the LiDAR fires early,
    the sources disagree, and the run aborts rather than pressing on to the
    distance the plan wanted. That ordering is the fail-safe direction."""
    plan = plan_insertion(1.75, 'short', insert)
    decision = stop_decision(0.70, 1.000, 0, plan, insert)
    assert not decision.stop, 'the face is not at the stop distance yet'
    decision = stop_decision(0.70, 0.700, 0, plan, insert)
    assert decision.stop
    assert decision.abort, 'odometry says 1.05 m of face left; something is wrong'


def test_losing_the_lidar_briefly_is_tolerated(insert):
    plan = plan_insertion(1.75, 'short', insert)
    decision = stop_decision(0.40, None, 2, plan, insert)
    assert not decision.stop and not decision.abort
    assert 'no LiDAR' in decision.reason


def test_losing_the_lidar_for_long_aborts(insert):
    """The robot is driving at a pallet it cannot see. Odometry alone is dead
    reckoning, and this is the phase where that is least acceptable."""
    plan = plan_insertion(1.75, 'short', insert)
    decision = stop_decision(0.40, None,
                             insert['max_unmeasured_frames'], plan, insert)
    assert decision.stop and decision.abort
    assert 'no LiDAR measurement' in decision.reason


def test_finishing_on_odometry_with_no_lidar_confirmation_aborts(insert):
    """It stops -- the travel is done -- but the run cannot claim an insertion
    it never measured."""
    plan = plan_insertion(1.75, 'short', insert)
    decision = stop_decision(1.05, None, 1, plan, insert)
    assert decision.stop and decision.abort
    assert 'no measurement to confirm' in decision.reason


def test_the_reported_face_distance_prefers_the_measurement(insert):
    plan = plan_insertion(1.75, 'short', insert)
    assert stop_decision(0.5, 1.21, 0, plan, insert).face_x == pytest.approx(1.21)
    assert stop_decision(0.5, None, 1, plan, insert).face_x == pytest.approx(1.25)


def test_the_stop_never_leaves_the_bumper_touching_the_pallet(insert, yaml_params):
    """The forks are imaginary; the robot is not. Every configured stop has to
    leave a real gap, and it also has to sit where the LiDAR can still see the
    face -- it loses it below a 0.204 m bumper gap (D13)."""
    bumper = yaml_params['/**']['ros__parameters']['robot']['front_bumper_x']
    for side in ('short', 'long'):
        plan = plan_insertion(1.75, side, insert)
        assert plan.stop_face_x - bumper >= 0.204, (
            f'{side}: a {plan.stop_face_x - bumper:.3f} m bumper gap is inside '
            f'the LiDAR blind cone')
