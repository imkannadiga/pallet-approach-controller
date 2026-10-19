"""Tests for closing.py: the servo that drives the robot onto the pocket line.

Half of this file is sign conventions, because a sign error here gives a
controller that steers away from the line perfectly smoothly, and the other
half is simulation, because the question "does it get close enough" cannot be
answered by inspection.

The simulated numbers are MEASURED here and compared against limits read out of
`config/params.yaml` -- the pocket slack, the GO/NO-GO tolerances -- rather than
against figures written into the test. A test that hard-codes "under 17 mm"
passes for ever once it has passed; one that asks "inside the slack the pockets
actually leave" fails when somebody narrows a pocket.

The simulation does not model actuator lag, wheel slip or an uneven floor, so
these are lower bounds, not predictions. The real figure comes from lab session
B.
"""

import math
import pathlib
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import simulate as S  # noqa: E402

from pallet_approach.closing import (  # noqa: E402
    DivergenceDetector,
    Phase,
    closing_command,
    commit_range,
    desired_heading,
    geometry_of,
    rear_clearance,
    required_run_up,
    reverse_allowance,
    straight_command,
)

PARAMS_FILE = pathlib.Path(__file__).resolve().parent.parent / 'config' / 'params.yaml'


@pytest.fixture(scope='module')
def yaml_params():
    return yaml.safe_load(PARAMS_FILE.read_text())


@pytest.fixture(scope='module')
def closing(yaml_params):
    """Exactly the dict the controller node will hand `closing_command`."""
    shared = yaml_params['/**']['ros__parameters']
    block = dict(shared['closing'])
    block['target_range'] = shared['standoff_distance']
    block['carriage_x'] = shared['forks']['carriage_x']
    block['tine_length'] = shared['forks']['tine_length']
    return block


@pytest.fixture(scope='module')
def limits(yaml_params):
    return yaml_params['approach_controller']['ros__parameters']['limits']


@pytest.fixture(scope='module')
def slack(yaml_params):
    """Lateral room a 100 mm tine has in the narrowest pocket, metres.

    The whole point of the servo is to arrive inside this. Derived from the
    measured pocket geometry so that narrowing a pocket breaks the tests that
    depend on it.
    """
    sides = yaml_params['/**']['ros__parameters']['pallet']['sides']
    width = yaml_params['/**']['ros__parameters']['forks']['tine_width']
    return min((side['pocket_width'] - width) / 2.0 for side in sides.values())


# ----------------------------------------------------------- the geometry

def test_a_robot_on_the_line_reads_zero():
    geometry = geometry_of(S.truth(S.facing(2.5)))
    assert geometry.range == pytest.approx(2.5)
    assert geometry.offset == pytest.approx(0.0, abs=1e-9)
    assert geometry.heading == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize('offset', [0.3, -0.3, 0.05])
def test_the_offset_sign_survives_a_round_trip(offset):
    """`simulate.facing` and `Geometry.offset` must use the same convention, or
    every test in this file reads backwards."""
    geometry = geometry_of(S.truth(S.facing(3.0, offset=offset)))
    assert geometry.offset == pytest.approx(offset)


@pytest.mark.parametrize('yaw_error', [0.1, -0.1])
def test_the_heading_sign_survives_a_round_trip(yaw_error):
    geometry = geometry_of(S.truth(S.facing(3.0, yaw_error=yaw_error)))
    assert geometry.heading == pytest.approx(yaw_error)


def test_a_robot_left_of_the_line_is_told_to_aim_right(closing):
    """The single sign that matters most. Wrong, and the robot leaves the line
    smoothly and converges on nothing."""
    assert desired_heading(+0.30, closing) < 0.0
    assert desired_heading(-0.30, closing) > 0.0


def test_the_aim_is_capped(closing):
    cap = closing['max_approach_angle']
    assert desired_heading(5.0, closing) == pytest.approx(-cap)
    assert desired_heading(-5.0, closing) == pytest.approx(cap)


def test_the_aim_is_zero_once_the_robot_is_nearly_on_the_line(closing):
    """So it straightens up and arrives square instead of still cutting."""
    inside = closing['straighten_offset'] * 0.5
    assert desired_heading(inside, closing) == 0.0
    assert desired_heading(-inside, closing) == 0.0


def test_the_aim_is_proportional_in_between(closing):
    offset = (closing['straighten_offset'] + closing['max_approach_angle']
              / closing['k_cross']) / 2.0
    assert desired_heading(offset, closing) == pytest.approx(
        -closing['k_cross'] * offset)


def test_a_pallet_turned_sideways_still_gives_a_sensible_line():
    """The centre line follows the pallet, not the room. A robot standing
    square in the room in front of a pallet rotated 20 degrees is off both the
    line and the heading, and both have to come out."""
    pallet = S.Pallet(yaw=math.radians(20.0))
    geometry = geometry_of(S.truth(S.facing(3.0), pallet))
    assert geometry.range == pytest.approx(3.0 * math.cos(math.radians(20.0)))
    assert abs(geometry.offset) > 0.5
    assert geometry.heading == pytest.approx(math.radians(20.0), abs=1e-6)


# ---------------------------------------------------------- the commit point

def test_the_commit_point_is_the_fork_tip_reach(closing, yaml_params):
    forks = yaml_params['/**']['ros__parameters']['forks']
    assert commit_range(closing) == pytest.approx(
        forks['carriage_x'] + forks['tine_length'])
    assert commit_range(closing) == pytest.approx(1.6125)


def test_the_standoff_is_outside_the_commit_point(closing):
    """If this ever failed, every run would begin with the tines through the
    face and the loop below would refuse to move at all."""
    assert closing['target_range'] > commit_range(closing)


@pytest.mark.parametrize('phase', list(Phase))
def test_nothing_moves_past_the_commit_point(closing, limits, phase):
    """A hard stop in the command, not a state. The tines are inside the pallet
    and any steering drags them through the blocks, so this holds whatever the
    state machine believes."""
    inside = commit_range(closing) - 0.10
    geometry = geometry_of(S.truth(S.facing(inside, offset=0.2)))
    command, _, reason = closing_command(geometry, phase, closing, limits)
    assert command.is_zero(), f'{phase} drove inside the fork tip reach'
    if phase is not Phase.ARRIVED:
        assert 'fork tip reach' in reason


# ------------------------------------------------------------- the arrival

def test_reaching_the_standoff_starts_the_square_up(closing, limits):
    geometry = geometry_of(S.truth(S.facing(closing['target_range'],
                                            yaw_error=0.10)))
    command, phase, _ = closing_command(geometry, Phase.CLOSING, closing, limits)
    assert phase is Phase.SQUARING
    assert command.linear == 0.0, 'it must stop driving first'
    assert command.angular > 0.0, 'and turn the way the error points'


def test_squaring_up_finishes_when_square(closing, limits):
    geometry = geometry_of(S.truth(S.facing(closing['target_range'])))
    command, phase, reason = closing_command(geometry, Phase.SQUARING,
                                             closing, limits)
    assert phase is Phase.ARRIVED
    assert command.is_zero()
    assert 'square' in reason


def test_the_arrival_tolerance_is_reachable_by_this_base(closing, limits):
    """Below `min_angular / k_heading` the command is zeroed rather than
    raised, so a tolerance tighter than that is one the base cannot reach and
    the square-up would spin for ever."""
    granularity = limits['min_angular'] / closing['k_heading']
    assert closing['arrival_yaw'] > granularity, (
        f'arrival_yaw {closing["arrival_yaw"]} is inside the base\'s own turn '
        f'granularity of {granularity:.4f} rad, so squaring up cannot finish')


def test_one_noisy_range_reading_does_not_restart_the_drive(closing, limits):
    """Hysteresis. Without it the robot creeps forward one frame at a time
    while it squares up."""
    just_out = closing['target_range'] + closing['range_tolerance'] + 0.01
    geometry = geometry_of(S.truth(S.facing(just_out)))
    _, phase, _ = closing_command(geometry, Phase.SQUARING, closing, limits)
    assert phase is Phase.ARRIVED or phase is Phase.SQUARING

    well_out = closing['target_range'] + closing['resume_range'] + 0.05
    geometry = geometry_of(S.truth(S.facing(well_out)))
    _, phase, _ = closing_command(geometry, Phase.SQUARING, closing, limits)
    assert phase is Phase.CLOSING, 'a real retreat should restart the drive'


def test_arrived_is_an_absorbing_state(closing, limits):
    geometry = geometry_of(S.truth(S.facing(3.0, offset=0.5)))
    command, phase, _ = closing_command(geometry, Phase.ARRIVED, closing, limits)
    assert phase is Phase.ARRIVED
    assert command.is_zero()


# --------------------------------------------------------------- the limits

def test_no_command_exceeds_the_configured_limits(closing, limits):
    """Swept, because the clamp is the last thing between a gain and a wheel."""
    for distance in (1.8, 2.5, 4.0):
        for offset in (-1.0, -0.2, 0.0, 0.05, 0.2, 1.0):
            for yaw in (-0.6, -0.1, 0.0, 0.1, 0.6):
                for phase in list(Phase):
                    geometry = geometry_of(S.truth(
                        S.facing(distance, offset=offset, yaw_error=yaw)))
                    command, _, _ = closing_command(geometry, phase, closing,
                                                    limits)
                    assert abs(command.linear) <= limits['max_linear'] + 1e-12
                    assert abs(command.angular) <= limits['max_angular'] + 1e-12


def test_the_loop_never_reverses(closing, limits):
    """Reversing mid-approach hides the pallet from the camera, and the reverse
    that is allowed is a separate, deliberate phase with its own checks."""
    for offset in (-1.0, 0.0, 1.0):
        for yaw in (-1.5, 0.0, 1.5):
            geometry = geometry_of(S.truth(S.facing(3.0, offset=offset,
                                                    yaw_error=yaw)))
            command, _, _ = closing_command(geometry, Phase.CLOSING, closing,
                                            limits)
            assert command.linear >= 0.0


def test_a_small_angular_command_is_zeroed_not_raised(closing, limits):
    """The decision that gave the servo any fine control at all. The base
    cannot turn slower than min_angular, so a finer correction is dropped
    rather than inflated -- inflating it makes every command hard over and the
    loop bang-bang."""
    tiny = limits['min_angular'] / closing['k_heading'] * 0.5
    geometry = geometry_of(S.truth(S.facing(3.0, yaw_error=tiny)))
    command, _, _ = closing_command(geometry, Phase.CLOSING, closing, limits)
    assert command.angular == 0.0
    assert command.linear > 0.0, 'but it should still be driving'


def test_the_robot_still_moves_when_the_taper_bottoms_out(closing, limits):
    """The linear deadband is raised, unlike the angular one: below min_linear
    a skid steer does not move at all, so a tapered command has to be raised
    back or the robot stops short believing it is still closing."""
    geometry = geometry_of(S.truth(
        S.facing(closing['target_range'] + closing['range_tolerance'] + 0.001)))
    command, phase, _ = closing_command(geometry, Phase.CLOSING, closing, limits)
    assert phase is Phase.CLOSING
    assert command.linear >= limits['min_linear']


# ----------------------------------------------------------- the simulation

def servo(closing, limits):
    """A `simulate.drive` step function running the real closing loop."""
    state = {'phase': Phase.CLOSING}

    def step(estimate, robot):
        command, phase, reason = closing_command(
            geometry_of(estimate), state['phase'], closing, limits)
        state['phase'] = phase
        return ((command.linear, command.angular),
                phase is Phase.ARRIVED, reason)
    return step


@pytest.mark.parametrize('offset', [0.0, 0.10, 0.30, 0.50, 0.80, -0.50])
def test_the_loop_arrives_inside_the_pocket_slack(closing, limits, slack, offset):
    """The question the whole stage exists to answer, measured rather than
    argued. Starting at the run-up the system would itself choose, does the
    robot end up somewhere the forks fit?
    """
    start = required_run_up(offset, closing)
    trace = S.drive(S.facing(start, offset=offset), servo(closing, limits))
    assert trace.arrived, trace.reason
    assert abs(trace.offset) < slack, (
        f'from {offset * 1000:+.0f} mm at {start:.2f} m the run finished '
        f'{trace.offset * 1000:+.0f} mm off the line, outside the '
        f'{slack * 1000:.0f} mm of slack a tine has in the pocket')


@pytest.mark.parametrize('seed', range(5))
@pytest.mark.parametrize('offset,yaw_error', [(0.0, 0.0), (0.30, 0.15),
                                              (-0.50, -0.15), (0.80, 0.0)])
def test_it_still_arrives_with_realistic_measurement_noise(
        closing, limits, yaml_params, slack, offset, yaw_error, seed):
    """10 mm of position noise and 5 mrad of normal noise, which is roughly
    what the first real bags showed (docs/dataset.md: face position within
    5-10 mm, pocket spacing within ~13 mm).

    Checked against the GO/NO-GO tolerances as well as the slack, because a run
    that arrives inside the forks' room but outside the rules it will then be
    judged by is a NO-GO, not a success.
    """
    rules = yaml_params['coordinator']['ros__parameters']['go_nogo']
    start = required_run_up(offset, closing)
    trace = S.drive(S.facing(start, offset=offset, yaw_error=yaw_error),
                    servo(closing, limits), noise=0.010, seed=seed)
    assert trace.arrived, trace.reason
    assert abs(trace.offset) < slack
    assert abs(trace.offset) <= rules['max_lateral_offset'], (
        f'{trace.offset * 1000:+.0f} mm would fail the lateral GO/NO-GO rule')
    assert abs(trace.heading) <= rules['max_yaw_error'], (
        f'{math.degrees(trace.heading):+.2f} deg would fail the yaw GO/NO-GO rule')


def test_the_run_up_formula_matches_what_the_loop_achieves(closing, limits):
    """`required_run_up` claims a model of the servo: linear decay while the
    aim is capped, exponential after. If the model is optimistic the robot sets
    off from too close and arrives outside the pockets, which is the one
    failure that cannot be recovered without reversing.

    So this measures the distance the loop ACTUALLY needs and checks the
    formula is not promising better.
    """
    for offset in (0.10, 0.30, 0.50, 0.80):
        claimed = required_run_up(offset, closing)
        trace = S.drive(S.facing(claimed, offset=offset), servo(closing, limits))
        assert trace.arrived
        assert abs(trace.offset) <= closing['residual_target'] * 1.5, (
            f'the formula allowed {claimed:.2f} m for {offset * 1000:.0f} mm of '
            f'offset and the loop finished {abs(trace.offset) * 1000:.0f} mm out, '
            f'against a {closing["residual_target"] * 1000:.0f} mm target')


def test_too_short_a_run_up_really_does_fail(closing, limits, slack):
    """The counterpart. If 3 m were enough for any offset, the whole
    variable-standoff design would be unnecessary -- and this is the
    measurement Athreya's 3 m minimum was set from."""
    trace = S.drive(S.facing(closing['min_run_up'], offset=0.80),
                    servo(closing, limits))
    assert trace.arrived
    assert abs(trace.offset) > slack, (
        'a 0.80 m offset corrected over 1.25 m of run-up should NOT fit; if it '
        'does, required_run_up is asking for distance nobody needs')


def test_the_decay_length_is_one_over_k_cross(closing, limits):
    """The claim `required_run_up` rests on, measured directly.

    Started inside the capped regime so the decay is purely exponential, and
    stopped early, so what is measured is the decay and not the arrival.
    """
    offset = closing['max_approach_angle'] / closing['k_cross'] * 0.8
    expected = 1.0 / closing['k_cross']

    distance = 0.8
    stop_at = closing['target_range'] + 0.5

    def step(estimate, robot):
        geometry = geometry_of(estimate)
        if geometry.range <= stop_at:
            return (0.0, 0.0), True, 'far enough'
        command, _, reason = closing_command(geometry, Phase.CLOSING,
                                             closing, limits)
        return (command.linear, command.angular), False, reason

    start = stop_at + distance
    trace = S.drive(S.facing(start, offset=offset), step)
    measured = distance / math.log(offset / abs(trace.offset))
    assert measured == pytest.approx(expected, rel=0.35), (
        f'the offset decayed with a length of {measured:.3f} m over {distance} m, '
        f'but required_run_up assumes 1/k_cross = {expected:.3f} m')


def test_a_pallet_at_an_angle_is_approached_along_its_own_line(closing, limits,
                                                               slack):
    """The case the two-leg approach needed a separate rotation for. Here it is
    one behaviour: the line belongs to the pallet, so a turned pallet is just a
    large offset and a large heading error."""
    pallet = S.Pallet(yaw=math.radians(15.0))
    offset, yaw_error = 0.40, math.radians(15.0)
    start = required_run_up(offset, closing)
    trace = S.drive(S.on_line(pallet, start, offset, yaw_error),
                    servo(closing, limits), pallet=pallet)
    assert trace.arrived, trace.reason
    assert abs(trace.offset) < slack
    assert abs(trace.heading) < closing['arrival_yaw'] * 1.5


# ----------------------------------------------------------- the run-up

def test_the_minimum_run_up_is_a_floor_not_a_target(closing):
    assert required_run_up(0.0, closing) == pytest.approx(closing['min_run_up'])
    assert required_run_up(0.001, closing) == pytest.approx(closing['min_run_up'])


def test_more_offset_needs_more_run_up(closing):
    distances = [required_run_up(o, closing) for o in (0.0, 0.2, 0.4, 0.8, 1.2)]
    assert distances == sorted(distances)
    assert distances[-1] > distances[0]


def test_the_run_up_accounts_for_the_capped_regime(closing):
    """The bug a simulation caught: the first version of this function used the
    exponential term alone, claimed 3.04 m was enough to take out 0.50 m of
    offset, and the run finished 250 mm off the line with 11 degrees of yaw.
    While the aim is capped the error falls linearly, which is much slower.
    """
    capped_below = closing['max_approach_angle'] / closing['k_cross']
    exponential_only = (closing['target_range']
                        + math.log(0.50 / closing['residual_target'])
                        / closing['k_cross'])
    assert 0.50 > capped_below, 'this test needs an offset inside the capped regime'
    assert required_run_up(0.50, closing) > exponential_only + 0.5, (
        'the run-up is not accounting for the linear phase')


# ------------------------------------------------------------- divergence

def test_a_shrinking_error_never_trips_the_detector():
    detector = DivergenceDetector(frames=5, min_growth=0.010)
    for offset in (0.30, 0.25, 0.20, 0.10, 0.05, 0.02, 0.01):
        diverging, _ = detector.update(offset)
        assert not diverging


def test_a_growing_error_trips_it():
    detector = DivergenceDetector(frames=5, min_growth=0.010)
    results = [detector.update(0.01 + 0.005 * i)[0] for i in range(7)]
    assert any(results)
    assert not results[0], 'one frame is not a trend'


def test_noise_at_the_standoff_does_not_trip_it():
    """Five frames of 1 mm steps is measurement scatter, not divergence. This
    is why the growth threshold exists alongside the frame count."""
    detector = DivergenceDetector(frames=5, min_growth=0.010)
    for i in range(20):
        diverging, _ = detector.update(0.005 + 0.0002 * i)
        assert not diverging


def test_one_shrinking_frame_resets_the_run():
    detector = DivergenceDetector(frames=3, min_growth=0.001)
    for offset in (0.01, 0.02, 0.03):
        detector.update(offset)
    assert detector.consecutive == 2
    detector.update(0.005)
    assert detector.consecutive == 0


def test_the_sign_of_the_offset_does_not_matter():
    """It is the magnitude that must shrink. A robot crossing the line reads
    +5 mm then -5 mm, which is not growth."""
    detector = DivergenceDetector(frames=2, min_growth=0.001)
    for offset in (0.030, 0.010, -0.010, 0.005, -0.004):
        diverging, _ = detector.update(offset)
        assert not diverging


def test_reset_clears_it():
    detector = DivergenceDetector(frames=2, min_growth=0.001)
    detector.update(0.01)
    detector.update(0.02)
    detector.reset()
    assert detector.consecutive == 0
    assert not detector.update(0.03)[0]


# --------------------------------------------------------- straight driving

def test_straight_driving_stops_at_the_distance():
    limits = {'max_linear': 0.20, 'max_angular': 0.30,
              'min_linear': 0.05, 'min_angular': 0.10}
    command, done, _ = straight_command(0.0, 1.00, 0.05, limits)
    assert not done and command.linear == pytest.approx(0.05)
    command, done, _ = straight_command(0.995, 1.00, 0.05, limits)
    assert done and command.is_zero()


def test_straight_driving_reverses_when_the_distance_is_negative():
    limits = {'max_linear': 0.20, 'max_angular': 0.30,
              'min_linear': 0.05, 'min_angular': 0.10}
    command, done, _ = straight_command(0.0, -1.00, 0.05, limits)
    assert not done
    assert command.linear == pytest.approx(-0.05)


def test_straight_driving_never_steers():
    """D7: during insertion the tines are inside the pallet, so steering drags
    them through the blocks. During the reverse the only thing worth steering
    by is behind the robot where nothing can see it."""
    limits = {'max_linear': 0.20, 'max_angular': 0.30,
              'min_linear': 0.05, 'min_angular': 0.10}
    for travelled in (0.0, 0.5, 0.99):
        for distance in (1.0, -1.0):
            command, _, _ = straight_command(travelled, distance, 0.05, limits)
            assert command.angular == 0.0


def test_straight_driving_respects_the_speed_limit():
    limits = {'max_linear': 0.10, 'max_angular': 0.30,
              'min_linear': 0.05, 'min_angular': 0.10}
    command, _, _ = straight_command(0.0, 1.0, 0.50, limits)
    assert command.linear == pytest.approx(0.10)


# ------------------------------------------------------------ the reverse

REVERSE = {'rear_bumper_x': -0.4625, 'half_width': 0.3498, 'side_margin': 0.10,
           'min_height': 0.03, 'max_height': 1.00, 'floor_z': -0.025,
           'max_unverified': 0.50}


def test_nothing_behind_reads_as_unmeasured_not_as_clear():
    """The distinction this whole function is for. Reversing is the one thing
    this robot does that it cannot see, and "the LiDAR found nothing" has to be
    reported as an absence rather than as a clearance."""
    for empty in (None, np.zeros((0, 3))):
        distance, detail = rear_clearance(empty, REVERSE)
        assert distance is None
        assert detail


def test_an_obstacle_behind_is_measured_from_the_bumper():
    wall = np.array([[-1.5, y, 0.3] for y in np.linspace(-0.2, 0.2, 10)])
    distance, detail = rear_clearance(wall, REVERSE)
    assert distance == pytest.approx(1.5 - 0.4625)
    assert '1.04' in detail


def test_the_nearest_obstacle_wins():
    points = np.vstack([
        np.array([[-2.0, 0.0, 0.3]]),
        np.array([[-0.9, 0.1, 0.3]]),
        np.array([[-3.0, 0.0, 0.3]]),
    ])
    distance, _ = rear_clearance(points, REVERSE)
    assert distance == pytest.approx(0.9 - 0.4625)


def test_the_floor_is_not_an_obstacle():
    floor = np.array([[-1.0 - 0.1 * i, y, -0.025]
                      for i in range(10) for y in (-0.2, 0.0, 0.2)])
    distance, _ = rear_clearance(floor, REVERSE)
    assert distance is None


def test_something_beside_the_robot_is_not_behind_it():
    aside = np.array([[-1.0, 1.2, 0.3], [-1.0, -1.2, 0.3]])
    assert rear_clearance(aside, REVERSE)[0] is None


def test_something_in_front_is_not_behind_either():
    ahead = np.array([[2.0, 0.0, 0.3]])
    assert rear_clearance(ahead, REVERSE)[0] is None


def test_the_corridor_is_wider_than_the_robot():
    """Reversing in a straight line still needs room for the robot's own width
    plus a margin, or it scrapes a doorframe it technically missed."""
    half = REVERSE['half_width'] + REVERSE['side_margin'] - 0.01
    just_inside = np.array([[-1.0, half, 0.3]])
    assert rear_clearance(just_inside, REVERSE)[0] is not None


def test_the_reverse_allowance_is_mostly_ground_the_robot_stood_on():
    """The argument, made into a limit: ground the robot's own body occupied a
    few seconds ago was clear, whatever the LiDAR can or cannot see now."""
    assert reverse_allowance(0.0, REVERSE) == pytest.approx(0.50)
    assert reverse_allowance(1.20, REVERSE) == pytest.approx(1.70)


def test_the_unverified_part_of_the_allowance_is_small():
    """It is the admitted gap in the argument. If this grew, the reverse would
    be relying on nothing at all."""
    assert REVERSE['max_unverified'] <= 0.60
