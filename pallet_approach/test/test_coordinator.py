"""Tests for coordinator.py -- the state machine.

Most of these are safety tests. The invariants they pin down:

* the controller may publish in exactly one state, APPROACHING;
* every terminal transition stops the robot;
* a single frame never starts the robot moving;
* cancel works from every running state.
"""

import math

import numpy as np
import pytest

from pallet_approach.coordinator import Coordinator, Outcome, State
from pallet_approach.model import Decision, PalletEstimate

PARAMS = {
    'detecting': {'timeout': 10.0, 'confirm_frames': 5,
                  'max_position_spread': 0.05, 'max_yaw_spread': 0.09},
    'approaching': {'lost_pallet_timeout': 1.0, 'max_estimate_jump': 0.15,
                    'max_estimate_jump_yaw': 0.20},
    'checking': {'settle_time': 0.5, 'frames': 5},
}


def estimate(distance=2.0, yaw=math.pi):
    return PalletEstimate(
        position=np.array([distance, 0.0, 0.072]),   # half the pallet height
        normal=np.array([-math.cos(yaw - math.pi), math.sin(yaw - math.pi), 0.0])
        if yaw != math.pi else np.array([-1.0, 0.0, 0.0]),
        yaw=yaw,
        face_inlier_count=2000,
    )


def confirmed():
    """A coordinator that has just entered APPROACHING."""
    coordinator = Coordinator(PARAMS)
    coordinator.trigger(0.0)
    transition = None
    for i in range(PARAMS['detecting']['confirm_frames']):
        transition = coordinator.on_estimate(estimate(2.0 + 0.001 * i), 0.1 * i)
    assert transition.state is State.APPROACHING
    return coordinator


# ----------------------------------------------------------- the happy path

def test_full_run_reaches_go():
    coordinator = confirmed()
    assert coordinator.on_arrived(1.0).state is State.CHECKING
    for i in range(5):
        coordinator.on_estimate(estimate(), 1.1 + 0.05 * i)
    assert coordinator.ready_to_decide(1.6)
    transition = coordinator.on_decision(Decision(go=True, reason='all checks passed'), 1.7)
    assert transition.state is State.DONE
    assert transition.outcome is Outcome.GO
    assert coordinator.to_idle(1.8).state is State.IDLE


def test_trigger_from_idle_activates_perception_only():
    transition = Coordinator(PARAMS).trigger(0.0)
    assert transition.state is State.DETECTING
    assert transition.perception_active
    assert not transition.controller_active
    assert transition.stop_robot


def test_a_second_trigger_mid_run_is_ignored():
    coordinator = confirmed()
    transition = coordinator.trigger(1.0)
    assert not transition.changed
    assert coordinator.state is State.APPROACHING


# -------------------------------------------------------- the detection gate

def test_one_frame_does_not_start_the_robot():
    """The single most important test in this file."""
    coordinator = Coordinator(PARAMS)
    coordinator.trigger(0.0)
    transition = coordinator.on_estimate(estimate(), 0.0)
    assert transition.state is State.DETECTING
    assert not transition.controller_active


@pytest.mark.parametrize('n', [1, 2, 3, 4])
def test_fewer_than_the_required_frames_never_confirms(n):
    coordinator = Coordinator(PARAMS)
    coordinator.trigger(0.0)
    for i in range(n):
        transition = coordinator.on_estimate(estimate(), 0.1 * i)
    assert transition.state is State.DETECTING
    assert not transition.controller_active


def test_inconsistent_frames_do_not_confirm():
    """Five frames, but one is 40 cm away: that is not a confirmed pallet."""
    coordinator = Coordinator(PARAMS)
    coordinator.trigger(0.0)
    for i, distance in enumerate([2.0, 2.0, 2.4, 2.0, 2.0]):
        transition = coordinator.on_estimate(estimate(distance), 0.1 * i)
    assert transition.state is State.DETECTING
    assert not transition.controller_active
    assert 'inconsistent' in transition.reason


def test_detect_timeout_gives_no_pallet():
    coordinator = Coordinator(PARAMS)
    coordinator.trigger(0.0)
    transition = coordinator.on_tick(10.5)
    assert transition.outcome is Outcome.NO_PALLET
    assert transition.stop_robot
    assert not transition.controller_active


# ------------------------------------------------------- aborting while moving

def test_pallet_lost_while_approaching_aborts():
    coordinator = confirmed()
    transition = coordinator.on_tick(2.0)      # last estimate was at t=0.4
    assert transition.outcome is Outcome.ABORTED
    assert 'lost' in transition.reason
    assert transition.stop_robot
    assert not transition.controller_active


def test_a_single_jump_is_rejected_without_aborting():
    """An outlier should not steer the robot, and should not stop the run either."""
    coordinator = confirmed()
    before = coordinator.accepted.position.copy()
    transition = coordinator.on_estimate(estimate(2.9), 0.5)
    assert transition.state is State.APPROACHING
    assert 'rejected' in transition.reason
    assert np.allclose(coordinator.accepted.position, before)


def test_a_small_correction_is_accepted():
    coordinator = confirmed()
    transition = coordinator.on_estimate(estimate(2.05), 0.5)
    assert transition.state is State.APPROACHING
    assert coordinator.accepted.position[0] == pytest.approx(2.05)


@pytest.mark.parametrize('state_setup', ['detecting', 'approaching', 'checking'])
def test_cancel_works_from_every_running_state(state_setup):
    if state_setup == 'detecting':
        coordinator = Coordinator(PARAMS)
        coordinator.trigger(0.0)
    else:
        coordinator = confirmed()
        if state_setup == 'checking':
            coordinator.on_arrived(1.0)
    transition = coordinator.cancel(2.0)
    assert transition.outcome is Outcome.ABORTED
    assert transition.stop_robot
    assert not transition.controller_active


def test_cancel_when_idle_does_nothing():
    transition = Coordinator(PARAMS).cancel(0.0)
    assert not transition.changed
    assert transition.outcome is None


def test_a_fault_always_stops_the_robot():
    coordinator = confirmed()
    transition = coordinator.on_fault(1.0, 'tf lookup failed')
    assert transition.outcome is Outcome.ABORTED
    assert transition.reason == 'tf lookup failed'
    assert transition.stop_robot


# ------------------------------------------------------------ CHECKING phase

def test_checking_stops_the_robot_on_entry():
    coordinator = confirmed()
    transition = coordinator.on_arrived(1.0)
    assert transition.stop_robot
    assert not transition.controller_active


def test_decision_waits_for_the_chassis_to_settle():
    coordinator = confirmed()
    coordinator.on_arrived(1.0)
    for i in range(5):
        coordinator.on_estimate(estimate(), 1.0 + 0.01 * i)
    assert not coordinator.ready_to_decide(1.2)     # 0.2 s < 0.5 s settle time
    assert coordinator.ready_to_decide(1.6)


def test_decision_waits_for_enough_frames():
    coordinator = confirmed()
    coordinator.history.clear()
    coordinator.on_arrived(1.0)
    coordinator.on_estimate(estimate(), 1.1)
    assert not coordinator.ready_to_decide(2.0)


def test_no_go_is_reported_with_its_reason():
    coordinator = confirmed()
    coordinator.on_arrived(1.0)
    transition = coordinator.on_decision(
        Decision(go=False, reason='yaw 7.2 deg exceeds 5.0 deg limit'), 2.0)
    assert transition.outcome is Outcome.NO_GO
    assert '7.2' in transition.reason
    assert transition.stop_robot


# --------------------------------------------------------------- invariants

def test_controller_is_only_ever_active_while_approaching():
    """Swept over every event from every state."""
    events = [
        ('trigger', lambda c: c.trigger(1.0)),
        ('cancel', lambda c: c.cancel(1.0)),
        ('estimate', lambda c: c.on_estimate(estimate(), 1.0)),
        ('tick', lambda c: c.on_tick(1.0)),
        ('arrived', lambda c: c.on_arrived(1.0)),
        ('decision', lambda c: c.on_decision(Decision(go=True, reason='ok'), 1.0)),
        ('fault', lambda c: c.on_fault(1.0, 'x')),
        ('to_idle', lambda c: c.to_idle(1.0)),
    ]
    for name, event in events:
        for setup in ('idle', 'detecting', 'approaching', 'checking'):
            coordinator = Coordinator(PARAMS)
            if setup == 'detecting':
                coordinator.trigger(0.0)
            elif setup in ('approaching', 'checking'):
                coordinator = confirmed()
                if setup == 'checking':
                    coordinator.on_arrived(0.5)
            transition = event(coordinator)
            if transition.controller_active:
                assert transition.state is State.APPROACHING, (
                    f'{name} from {setup} left the controller active in '
                    f'{transition.state.value}')


def test_every_terminal_transition_stops_the_robot():
    terminals = [
        lambda: Coordinator(PARAMS).__class__(PARAMS).on_tick(0.0),
        lambda: confirmed().cancel(1.0),
        lambda: confirmed().on_fault(1.0, 'x'),
        lambda: confirmed().on_tick(5.0),
    ]
    for make in terminals:
        transition = make()
        if transition.state is State.DONE:
            assert transition.stop_robot
            assert not transition.controller_active
            assert not transition.perception_active
