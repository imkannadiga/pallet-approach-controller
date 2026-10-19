"""Tests for coordinator.py -- the sequence for a whole virtual pickup.

Most of these are safety tests. The invariants they pin down, each asserted
over every event from every state rather than in the one place it is easy to
check:

* the controller may only be told to move in the four DRIVING_STATES;
* every terminal transition stops the robot and deactivates everything;
* a single frame never starts the robot moving;
* cancel, a health failure and a controller abort are honoured from every
  running state;
* an aborted insertion stops where it is and does not drive out.
"""

import pathlib

import pytest
import yaml

from pallet_approach.coordinator import (
    DRIVING_STATES,
    TERMINAL_STATES,
    Coordinator,
    DriveInfo,
    ForkInfo,
    Outcome,
    State,
    TrackInfo,
)
from pallet_approach.model import Decision

PARAMS_FILE = pathlib.Path(__file__).resolve().parent.parent / 'config' / 'params.yaml'


@pytest.fixture(scope='module')
def yaml_params():
    return yaml.safe_load(PARAMS_FILE.read_text())


@pytest.fixture
def params(yaml_params):
    """Exactly what the node hands the machine: flat, with the nested blocks
    it needs carried through as sub-dicts."""
    shared = yaml_params['/**']['ros__parameters']
    block = yaml_params['coordinator']['ros__parameters']
    controller = yaml_params['approach_controller']['ros__parameters']
    closing = dict(shared['closing'])
    closing['target_range'] = shared['standoff_distance']
    closing['carriage_x'] = shared['forks']['carriage_x']
    closing['tine_length'] = shared['forks']['tine_length']
    return {
        'confirm_timeout': block['timeouts']['confirm'],
        'lost_pallet_timeout': block['timeouts']['lost_pallet'],
        'fork_timeout': block['timeouts']['forks'],
        'run_timeout': block['timeouts']['run'],
        'settle_time': block['checking']['settle_time'],
        'checking_frames': block['checking']['frames'],
        'max_retries': block['max_retries'],
        'max_backoffs': block['max_backoffs'],
        'min_retry_backoff': block['min_retry_backoff'],
        'retryable_checks': block['retryable_checks'],
        'entry_face': shared['pallet']['entry_face'],
        'sides': shared['pallet']['sides'],
        'closing': closing,
        'reverse': shared['reverse'],
    }


def track(confirmed=True, side='short', range_=3.5, offset=0.0, heading=0.0,
          confidence=1.0, reason=''):
    return TrackInfo(confirmed=confirmed, confidence=confidence, side=side,
                     range=range_, offset=offset, heading=heading, reason=reason)


def arrived(mode, travelled=1.0):
    return DriveInfo(mode=mode, arrived=True, travelled=travelled)


def forks(state, loaded=False, moving=False, height=0.040):
    return ForkInfo(state=state, height=height, moving=moving, loaded=loaded)


GO = Decision(go=True, reason='all checks passed',
              checks={'yaw': True, 'lateral_offset': True})


def no_go(*failed):
    checks = {'yaw': True, 'lateral_offset': True, 'stability': True,
              'pocket_dimensions': True, 'pocket_count': True,
              'face_quality': True, 'overhang': True, 'pocket_clearance': True}
    for name in failed:
        checks[name] = False
    return Decision(go=False, reason=', '.join(failed) or 'refused', checks=checks)


# -------------------------------------------------------- run helpers

def confirming(params):
    machine = Coordinator(params)
    machine.trigger(0.0)
    return machine


def closing_in(params, **kwargs):
    """A machine that has confirmed a pallet with enough run-up and is closing."""
    machine = confirming(params)
    transition = machine.on_track(track(**kwargs), 1.0)
    assert transition.state is State.CLOSING, transition.reason
    return machine


def at_checking(params):
    machine = closing_in(params)
    for i in range(params['checking_frames']):
        machine.on_track(track(range_=1.75), 2.0 + 0.2 * i)
    transition = machine.on_drive(arrived('CLOSE'), 3.0)
    assert transition.state is State.CHECKING
    return machine


def inserting(params):
    machine = at_checking(params)
    for i in range(params['checking_frames']):
        machine.on_track(track(range_=1.75), 3.1 + 0.2 * i)
    assert machine.on_decision(GO, 4.0).state is State.LOWERING
    transition = machine.on_forks(forks('INSERT_HEIGHT'), 7.0)
    assert transition.state is State.INSERTING
    return machine


# ----------------------------------------------------------- the happy path

def test_a_whole_virtual_pickup(params):
    """Start to finish, with the outcome the scope grew to include (D2)."""
    machine = inserting(params)
    transition = machine.on_drive(arrived('INSERT', travelled=1.05), 25.0)
    assert transition.state is State.LIFTING
    assert transition.fork == 'LIFT'
    transition = machine.on_forks(forks('LIFTED', loaded=True), 27.0)
    assert transition.state is State.BACKING_OUT
    assert transition.drive.mode == 'BACKOUT'
    assert transition.drive.distance == pytest.approx(1.05), (
        'it must back out as far as it went in')
    transition = machine.on_drive(arrived('BACKOUT', travelled=1.05), 50.0)
    assert transition.state is State.DONE
    assert transition.outcome is Outcome.INSERTED
    assert transition.stop_robot and transition.drive is None
    assert machine.to_idle(51.0).state is State.IDLE


def test_the_sequence_waits_for_the_forks_before_inserting(params):
    """A phase whose job is to take time while the robot stands still must not
    be skippable. If the forks are still moving, nothing drives."""
    machine = at_checking(params)
    for i in range(params['checking_frames']):
        machine.on_track(track(range_=1.75), 3.1 + 0.2 * i)
    machine.on_decision(GO, 4.0)
    transition = machine.on_forks(forks('MOVING', moving=True), 5.0)
    assert transition.state is State.LOWERING
    assert transition.drive is None


def test_lifting_waits_for_the_loaded_flag(params):
    """Not for the state name: `loaded` is the simulator's claim that the
    pallet came off the floor, and that is the claim the run reports."""
    machine = inserting(params)
    machine.on_drive(arrived('INSERT', travelled=1.05), 25.0)
    transition = machine.on_forks(forks('LIFTED', loaded=False), 26.0)
    assert transition.state is State.LIFTING
    assert transition.drive is None


# -------------------------------------------------------- the detection gate

def test_one_frame_does_not_start_the_robot(params):
    """The single most important test in this file."""
    machine = confirming(params)
    transition = machine.on_track(track(confirmed=False), 0.2)
    assert transition.state is State.CONFIRMING
    assert transition.drive is None
    assert transition.stop_robot


def test_an_unconfirmed_track_never_leaves_confirming(params):
    machine = confirming(params)
    for i in range(20):
        transition = machine.on_track(track(confirmed=False, reason='2/5 frames'),
                                      0.2 * i)
    assert transition.state is State.CONFIRMING
    assert '2/5' in transition.reason


def test_the_confirm_timeout_gives_no_pallet(params):
    machine = confirming(params)
    transition = machine.on_tick(params['confirm_timeout'] + 0.5)
    assert transition.outcome is Outcome.NO_PALLET
    assert transition.stop_robot and transition.drive is None


def test_the_timeout_reason_carries_the_tracker_s_complaint(params):
    """So an operator can tell "nothing there" from "the LiDAR disagrees"."""
    machine = confirming(params)
    machine.on_track(track(confirmed=False, reason='LiDAR disagrees: 42 returns'),
                     1.0)
    transition = machine.on_tick(params['confirm_timeout'] + 0.5)
    assert 'LiDAR disagrees' in transition.reason


# ------------------------------------------------------------- classifying

def test_an_ambiguous_side_refuses_rather_than_guessing(params):
    """D22/D26 one level up: measuring pocket geometry against the wrong
    nominals is how a damaged pallet passes."""
    machine = confirming(params)
    transition = machine.on_track(track(side='ambiguous', reason='spacing says '
                                        'long, width says short'), 1.0)
    assert transition.outcome is Outcome.NO_GO
    assert 'unclear which face' in transition.reason
    assert 'width says short' in transition.reason
    assert transition.stop_robot


def test_a_face_this_site_forbids_refuses(params):
    """D20: whether long-side entry is allowed is policy, not geometry."""
    forbidden = dict(params)
    forbidden['sides'] = {
        'short': params['sides']['short'],
        'long': dict(params['sides']['long'], insertable=False),
    }
    machine = confirming(forbidden)
    transition = machine.on_track(track(side='long', range_=4.0), 1.0)
    assert transition.outcome is Outcome.NO_GO
    assert 'not permitted' in transition.reason


def test_an_unclassified_side_falls_back_but_says_so(params):
    """Classification can be switched off. Treating "not measured" as the
    default face silently is the thing to avoid."""
    machine = confirming(params)
    transition = machine.on_track(track(side='unknown'), 1.0)
    assert transition.state is State.CLOSING
    assert 'not classified' in transition.reason
    assert machine.side == params['entry_face']


def test_an_unknown_side_name_is_a_fault_not_a_guess(params):
    machine = confirming(params)
    transition = machine.on_track(track(side='diagonal'), 1.0)
    assert transition.outcome is Outcome.ABORTED
    assert 'no geometry configured' in transition.reason


# ----------------------------------------------------------- the run-up

def test_enough_run_up_closes_in_directly(params):
    machine = confirming(params)
    transition = machine.on_track(track(range_=4.0, offset=0.05), 1.0)
    assert transition.state is State.CLOSING
    assert transition.drive.mode == 'CLOSE'


def test_too_little_run_up_reverses_first(params):
    """Athreya's design: variable standoff with a 3 m minimum, and a reverse
    when there is not that much (D39)."""
    machine = confirming(params)
    transition = machine.on_track(track(range_=2.7, offset=0.0), 1.0)
    assert transition.state is State.BACKING_OFF
    assert transition.drive.mode == 'REVERSE'
    # The shortfall plus `run_up_tolerance`, so the robot ends up past the
    # requirement rather than on its edge. Without the overshoot a reverse that
    # stops within its own 10 mm window finishes short, asks for another
    # centimetre, and dies on max_backoffs having done everything right -- the
    # offline harness caught that.
    expected = 0.30 + params['closing']['run_up_tolerance']
    assert transition.drive.distance == pytest.approx(expected, abs=0.01)
    assert '2.70 m of run-up' in transition.reason


def test_a_run_up_a_centimetre_short_is_accepted(params):
    """The acceptance tolerance, from the other side. 2.99 m against a 3.00 m
    requirement is not a reason to reverse again."""
    machine = confirming(params)
    needed = params['closing']['min_run_up']
    transition = machine.on_track(track(range_=needed - 0.01), 1.0)
    assert transition.state is State.CLOSING


def test_a_lateral_offset_lengthens_the_required_run_up(params):
    """The "variable" half of the variable standoff: a robot Nav2 left off the
    centre line needs more distance to correct in, and the amount is geometry."""
    machine = confirming(params)
    square = machine.on_track(track(range_=3.2, offset=0.0), 1.0)
    assert square.state is State.CLOSING

    machine = confirming(params)
    offset = machine.on_track(track(range_=3.2, offset=0.30), 1.0)
    assert offset.state is State.BACKING_OFF, (
        'the same distance with 300 mm of offset is not enough run-up')


def test_a_reverse_that_cannot_be_justified_refuses(params):
    """D42, stated as a refusal rather than a hope. The robot cannot see behind
    itself, so the only ground it may reverse over without a costmap is ground
    its own body was standing on -- and at the start of a run there is none."""
    machine = confirming(params)
    transition = machine.on_track(track(range_=1.9, offset=0.60), 1.0)
    assert transition.outcome is Outcome.NO_GO
    assert 'can be justified' in transition.reason
    assert 'Reposition with Nav2' in transition.reason


def test_the_run_up_is_re_evaluated_after_reversing(params):
    """Against where the robot actually ended up, not where it predicted. The
    reverse is on odometry along the robot's heading, which is not exactly
    along the centre line."""
    machine = confirming(params)
    assert machine.on_track(track(range_=2.8), 1.0).state is State.BACKING_OFF
    machine.on_track(track(range_=3.1), 5.0)
    transition = machine.on_drive(arrived('REVERSE', travelled=0.30), 6.0)
    assert transition.state is State.CLOSING


def test_reversing_twice_and_then_giving_up(params):
    """A loop of reverses would otherwise walk the robot across the lab."""
    machine = confirming(params)
    transition = machine.on_track(track(range_=2.8), 1.0)
    for i in range(params['max_backoffs'] + 2):
        if transition.state is not State.BACKING_OFF:
            break
        machine.on_track(track(range_=2.8), 5.0 + i)
        transition = machine.on_drive(arrived('REVERSE', travelled=0.2),
                                      6.0 + i)
    assert transition.outcome is Outcome.NO_GO
    assert 'reverse(s)' in transition.reason


# ------------------------------------------------------------- the retry

def test_a_retryable_no_go_backs_off_and_tries_again(params):
    """Athreya chose "back off and run the loop again, once" on 2026-10-19."""
    machine = at_checking(params)
    for i in range(params['checking_frames']):
        machine.on_track(track(range_=1.75), 3.1 + 0.2 * i)
    transition = machine.on_decision(no_go('lateral_offset'), 4.0)
    assert transition.state is State.BACKING_OFF
    assert transition.drive.mode == 'REVERSE'
    assert 'retry 1' in transition.reason


def test_only_once(params):
    machine = at_checking(params)
    for i in range(params['checking_frames']):
        machine.on_track(track(range_=1.75), 3.1 + 0.2 * i)
    machine.on_decision(no_go('lateral_offset'), 4.0)
    machine.on_track(track(range_=3.5), 10.0)
    machine.on_drive(arrived('REVERSE'), 11.0)
    for i in range(params['checking_frames']):
        machine.on_track(track(range_=1.75), 12.0 + 0.2 * i)
    machine.on_drive(arrived('CLOSE'), 14.0)
    transition = machine.on_decision(no_go('lateral_offset'), 15.0)
    assert transition.outcome is Outcome.NO_GO


@pytest.mark.parametrize('failure', ['pocket_dimensions', 'pocket_count',
                                     'face_quality', 'overhang',
                                     'pocket_clearance'])
def test_a_property_of_the_pallet_is_not_retried(params, failure):
    """Driving the same approach again would measure the same pallet. Only
    where the robot is standing and how steady the estimate is can change."""
    machine = at_checking(params)
    for i in range(params['checking_frames']):
        machine.on_track(track(range_=1.75), 3.1 + 0.2 * i)
    transition = machine.on_decision(no_go(failure), 4.0)
    assert transition.outcome is Outcome.NO_GO
    assert transition.state is State.DONE


def test_a_mixed_failure_is_not_retried(params):
    """One retryable failure alongside one that is not means the pallet is
    wrong as well as the robot, and the pallet does not change."""
    machine = at_checking(params)
    for i in range(params['checking_frames']):
        machine.on_track(track(range_=1.75), 3.1 + 0.2 * i)
    transition = machine.on_decision(no_go('yaw', 'pocket_count'), 4.0)
    assert transition.outcome is Outcome.NO_GO


def test_the_retry_always_means_a_real_new_approach(params):
    """From the standoff the geometry may ask for very little reverse. A retry
    that moves 20 mm and measures again is not a retry."""
    machine = at_checking(params)
    for i in range(params['checking_frames']):
        machine.on_track(track(range_=3.4, offset=0.0), 3.1 + 0.2 * i)
    transition = machine.on_decision(no_go('yaw'), 4.0)
    assert transition.drive.distance >= params['min_retry_backoff']


# --------------------------------------------------------------- aborting

def test_an_aborted_insertion_stops_where_it_is(params):
    """It does NOT reverse out. The robot's body is at least 0.204 m clear of
    the face at any legal stop, so standing still is safe -- and driving after
    an abort contradicts the one rule with no exceptions. Recovery is the
    operator's, with the reason in the result."""
    machine = inserting(params)
    transition = machine.on_drive(
        DriveInfo(mode='INSERT', abort=True, travelled=0.6,
                  reason='odometry and the LiDAR are 120 mm apart'), 20.0)
    assert transition.state is State.ABORTED
    assert transition.stop_robot
    assert transition.drive is None
    assert '120 mm apart' in transition.reason


def test_a_controller_abort_ends_the_run_from_every_driving_state(params):
    for state, setup in (
        (State.BACKING_OFF, lambda: _in_backoff(params)),
        (State.CLOSING, lambda: closing_in(params)),
        (State.INSERTING, lambda: inserting(params)),
    ):
        machine = setup()
        assert machine.state is state
        mode = machine._expected_mode()
        transition = machine.on_drive(
            DriveInfo(mode=mode, abort=True, reason='diverging'), 30.0)
        assert transition.outcome is Outcome.ABORTED, state
        assert transition.stop_robot


def test_losing_the_pallet_while_closing_aborts(params):
    machine = closing_in(params)
    transition = machine.on_tick(1.0 + params['lost_pallet_timeout'] + 0.5)
    assert transition.outcome is Outcome.ABORTED
    assert 'pallet lost' in transition.reason
    assert transition.stop_robot


def test_losing_the_pallet_while_inserting_does_not_abort(params):
    """By design (D16): the pockets leave the camera's view 0.58 m into the
    travel, so losing the track there is expected. The LiDAR is what watches
    the face from then on, and the controller owns that."""
    machine = inserting(params)
    transition = machine.on_tick(8.0 + params['lost_pallet_timeout'] + 0.5)
    assert transition.state is State.INSERTING
    assert transition.outcome is None


def test_forks_that_never_arrive_abort(params):
    machine = at_checking(params)
    for i in range(params['checking_frames']):
        machine.on_track(track(range_=1.75), 3.1 + 0.2 * i)
    machine.on_decision(GO, 4.0)
    # Keep the pallet in view throughout, or the lost-pallet timeout fires
    # first and the reason would be about the wrong thing.
    deadline = 4.0 + params['fork_timeout'] + 0.5
    now = 4.0
    while now < deadline:
        now += params['lost_pallet_timeout'] / 2.0
        machine.on_track(track(range_=1.75), now)
        transition = machine.on_tick(now)
    assert transition.outcome is Outcome.ABORTED
    assert 'forks did not reach' in transition.reason


def test_losing_the_pallet_while_the_forks_lower_aborts(params):
    """The robot is stopped at the standoff and is about to drive forward into
    the pallet. Perception having gone quiet there is not something to wait
    through."""
    machine = at_checking(params)
    for i in range(params['checking_frames']):
        machine.on_track(track(range_=1.75), 3.1 + 0.2 * i)
    machine.on_decision(GO, 4.0)
    transition = machine.on_tick(4.0 + params['lost_pallet_timeout'] + 0.5)
    assert transition.outcome is Outcome.ABORTED
    assert 'pallet lost' in transition.reason


def test_a_run_that_overruns_aborts(params):
    """An unbounded run is one nobody is watching. Everything else is kept
    healthy so that the run timeout is the thing being tested."""
    machine = closing_in(params)
    now = 1.0
    while now < params['run_timeout'] + 1.0:
        now += params['lost_pallet_timeout'] / 2.0
        machine.on_track(track(range_=2.5), now)
        transition = machine.on_tick(now)
        if transition.outcome is not None:
            break
    assert transition.outcome is Outcome.ABORTED
    assert 'exceeded' in transition.reason


def test_an_unhealthy_stack_aborts(params):
    """D9. The watchdog stops the robot itself; this is the coordinator's side,
    so the run ends with a reason rather than hanging with the wheels held."""
    machine = closing_in(params)
    transition = machine.on_unhealthy(5.0, 'odom: 0.62 s old, limit 0.20 s')
    assert transition.outcome is Outcome.ABORTED
    assert '0.62 s old' in transition.reason
    assert transition.stop_robot and transition.drive is None


def test_an_unhealthy_stack_while_idle_changes_nothing(params):
    machine = Coordinator(params)
    transition = machine.on_unhealthy(1.0, 'lidar: nothing received')
    assert not transition.changed
    assert transition.state is State.IDLE


def test_a_stale_drive_report_from_the_previous_move_is_ignored(params):
    """An `arrived` left over from the reverse would otherwise complete the
    insertion the instant it started."""
    machine = inserting(params)
    transition = machine.on_drive(arrived('REVERSE', travelled=0.3), 9.0)
    assert transition.state is State.INSERTING
    assert transition.outcome is None


# ------------------------------------------------------------- invariants

def _in_backoff(params):
    machine = confirming(params)
    assert machine.on_track(track(range_=2.7), 1.0).state is State.BACKING_OFF
    return machine


SETUPS = {
    'idle': lambda p: Coordinator(p),
    'confirming': confirming,
    'backing_off': _in_backoff,
    'closing': closing_in,
    'checking': at_checking,
    'inserting': inserting,
    'done': lambda p: _finished(p),
}


def _finished(params):
    machine = inserting(params)
    machine.on_drive(arrived('INSERT', travelled=1.05), 25.0)
    machine.on_forks(forks('LIFTED', loaded=True), 27.0)
    machine.on_drive(arrived('BACKOUT', travelled=1.05), 50.0)
    return machine


EVENTS = {
    'trigger': lambda m: m.trigger(60.0),
    'cancel': lambda m: m.cancel(60.0),
    'track': lambda m: m.on_track(track(), 60.0),
    'unconfirmed_track': lambda m: m.on_track(track(confirmed=False), 60.0),
    'tick': lambda m: m.on_tick(60.0),
    'drive_arrived': lambda m: m.on_drive(arrived('CLOSE'), 60.0),
    'drive_abort': lambda m: m.on_drive(
        DriveInfo(mode='CLOSE', abort=True, reason='x'), 60.0),
    'forks': lambda m: m.on_forks(forks('INSERT_HEIGHT'), 60.0),
    'decision_go': lambda m: m.on_decision(GO, 60.0),
    'decision_no_go': lambda m: m.on_decision(no_go('yaw'), 60.0),
    'unhealthy': lambda m: m.on_unhealthy(60.0, 'x'),
    'fault': lambda m: m.on_fault(60.0, 'x'),
    'to_idle': lambda m: m.to_idle(60.0),
}


@pytest.mark.parametrize('event_name', sorted(EVENTS))
@pytest.mark.parametrize('setup_name', sorted(SETUPS))
def test_the_controller_only_drives_in_the_four_driving_states(
        params, setup_name, event_name):
    """Swept over every event from every state. A new state added without
    thinking about DRIVING_STATES cannot drive, which is the right default."""
    machine = SETUPS[setup_name](params)
    transition = EVENTS[event_name](machine)
    if transition.drive is not None:
        assert transition.state in DRIVING_STATES, (
            f'{event_name} from {setup_name} left the controller driving in '
            f'{transition.state.value}')


@pytest.mark.parametrize('event_name', sorted(EVENTS))
@pytest.mark.parametrize('setup_name', sorted(SETUPS))
def test_a_stopped_state_is_never_also_driving(params, setup_name, event_name):
    machine = SETUPS[setup_name](params)
    transition = EVENTS[event_name](machine)
    assert not (transition.stop_robot and transition.drive is not None), (
        f'{event_name} from {setup_name} said stop AND gave a drive command')


@pytest.mark.parametrize('event_name', sorted(EVENTS))
@pytest.mark.parametrize('setup_name', sorted(SETUPS))
def test_every_terminal_transition_stops_everything(params, setup_name,
                                                     event_name):
    machine = SETUPS[setup_name](params)
    transition = EVENTS[event_name](machine)
    if transition.changed and transition.state in TERMINAL_STATES:
        assert transition.stop_robot, f'{event_name} from {setup_name}'
        assert transition.drive is None
        assert not transition.perception_active


@pytest.mark.parametrize('setup_name', ['confirming', 'backing_off', 'closing',
                                        'checking', 'inserting'])
def test_cancel_works_from_every_running_state(params, setup_name):
    machine = SETUPS[setup_name](params)
    transition = machine.cancel(60.0)
    assert transition.outcome is Outcome.ABORTED
    assert transition.stop_robot and transition.drive is None


@pytest.mark.parametrize('setup_name', ['confirming', 'backing_off', 'closing',
                                        'checking', 'inserting'])
def test_a_health_failure_aborts_from_every_running_state(params, setup_name):
    machine = SETUPS[setup_name](params)
    transition = machine.on_unhealthy(60.0, 'depth: 0.4 s old')
    assert transition.outcome is Outcome.ABORTED
    assert transition.stop_robot


def test_a_second_goal_mid_run_is_ignored(params):
    machine = closing_in(params)
    transition = machine.trigger(5.0)
    assert not transition.changed
    assert machine.state is State.CLOSING


def test_an_event_after_a_run_still_reports_its_outcome(params):
    """A second tick after the run ended must not read as "aborted for no
    reason" -- anything watching the latest transition rather than the last
    changed one would see an outcome of None."""
    machine = closing_in(params)
    ended = machine.cancel(5.0)
    assert ended.outcome is Outcome.ABORTED
    again = machine.on_tick(6.0)
    assert not again.changed
    assert again.state is State.ABORTED
    assert again.outcome is Outcome.ABORTED
    assert again.stop_robot and again.drive is None


def test_cancel_when_idle_does_nothing(params):
    transition = Coordinator(params).cancel(0.0)
    assert not transition.changed
    assert transition.outcome is None


def test_repeating_a_state_s_drive_command_does_not_change_it(params):
    """The controller stops if its command goes stale, so the sequence keeps
    saying what it wants. Deriving that from the state rather than remembering
    what was sent means the two cannot drift apart."""
    machine = _in_backoff(params)
    first = machine.on_tick(1.2)
    second = machine.on_tick(1.4)
    assert first.drive == second.drive
    assert first.drive.mode == 'REVERSE'


# ---------------------------------------------------------------- checking

def test_the_decision_waits_for_the_chassis_to_settle(params):
    machine = at_checking(params)
    for i in range(params['checking_frames']):
        machine.on_track(track(range_=1.75), 3.0 + 0.01 * i)
    assert not machine.ready_to_decide(3.2)
    assert machine.ready_to_decide(3.0 + params['settle_time'] + 0.1)


def test_the_decision_waits_for_enough_frames(params):
    machine = closing_in(params)
    machine.history.clear()
    machine.on_drive(arrived('CLOSE'), 3.0)
    machine.on_track(track(range_=1.75), 3.1)
    assert not machine.ready_to_decide(10.0)


def test_only_confirmed_frames_count_towards_the_decision(params):
    """An unconfirmed track is not evidence about the pallet, and the GO/NO-GO
    stability rule is computed over these frames."""
    machine = closing_in(params)
    machine.history.clear()
    machine.on_drive(arrived('CLOSE'), 3.0)
    for i in range(params['checking_frames'] + 2):
        machine.on_track(track(confirmed=False, range_=1.75), 3.1 + 0.1 * i)
    assert not machine.ready_to_decide(10.0)
