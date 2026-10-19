"""Tests for forks.py -- a simulator that must not be able to lie.

There is no hardware, so every property these tests pin down is a property of
the claim the simulator makes. The important ones: the lift takes real time,
`loaded` cannot be set except by actually lifting from the insertion height,
and the reported pallet lift is the carriage travel minus the slack above the
tines, which differs between the two faces.
"""

import pathlib

import pytest
import yaml

from pallet_approach.forks import ForkSimulator, State, pallet_lift

PARAMS_FILE = pathlib.Path(__file__).resolve().parent.parent / 'config' / 'params.yaml'


@pytest.fixture(scope='module')
def yaml_params():
    return yaml.safe_load(PARAMS_FILE.read_text())


@pytest.fixture
def params(yaml_params):
    shared = yaml_params['/**']['ros__parameters']
    block = dict(yaml_params['virtual_forks']['ros__parameters'])
    block['tine_thickness'] = shared['forks']['tine_thickness']
    return block


@pytest.fixture
def forks(params):
    return ForkSimulator(params)


def settle(forks, start=0.0, step=0.05, limit=20.0):
    """Run the simulator forward until it stops moving. Returns the time."""
    now = start
    forks.update(now)
    while forks.moving and now - start < limit:
        now += step
        forks.update(now)
    return now - start


# ------------------------------------------------------------------ states

def test_it_starts_parked(forks, params):
    assert forks.state is State.PARKED
    assert forks.height == pytest.approx(params['park_height'])
    assert not forks.loaded


def test_lowering_takes_time(forks, params):
    """A simulator that teleports would let the sequence skip a phase whose
    whole job is to take time while the robot stands still."""
    forks.lower_to_insert(0.0)
    assert forks.moving
    assert forks.height == pytest.approx(params['park_height']), (
        'the height must not jump on the command itself'
    )
    seconds = settle(forks)
    expected = (params['park_height'] - params['insert_height']) / params['speed']
    assert seconds == pytest.approx(expected, abs=0.1)
    assert forks.state is State.INSERT_HEIGHT
    assert forks.height == pytest.approx(params['insert_height'])


def test_the_insertion_height_is_the_one_the_overlay_draws(params, yaml_params):
    """Two numbers for one height is how a simulator and an overlay come to
    disagree about where the forks are."""
    assert params['insert_height'] == pytest.approx(
        yaml_params['/**']['ros__parameters']['forks']['bottom_height'])


def test_lifting_from_the_insertion_height_works(forks, params):
    forks.lower_to_insert(0.0)
    settle(forks)
    assert forks.lift(10.0)
    settle(forks, start=10.0)
    assert forks.state is State.LIFTED
    assert forks.height == pytest.approx(params['insert_height']
                                         + params['lift_travel'])
    assert forks.loaded


def test_lifting_from_park_height_is_refused(forks):
    """The refusal is the point: lifting a carriage that is nowhere near a
    pallet would still hand the sequence a `loaded` flag."""
    assert not forks.lift(0.0)
    assert forks.state is State.PARKED
    assert not forks.loaded


def test_lifting_while_still_lowering_is_refused(forks):
    forks.lower_to_insert(0.0)
    forks.update(0.0)
    forks.update(0.1)
    assert forks.moving
    assert not forks.lift(0.1)


def test_loaded_is_only_set_once_the_lift_finishes(forks):
    """Not when the command is given. The claim is about the pallet having
    come off the floor, which has not happened while the carriage is moving."""
    forks.lower_to_insert(0.0)
    settle(forks)
    forks.lift(10.0)
    forks.update(10.0)
    forks.update(10.1)
    assert forks.moving and not forks.loaded
    settle(forks, start=10.1)
    assert forks.loaded


def test_parking_puts_the_load_down(forks):
    forks.lower_to_insert(0.0)
    settle(forks)
    forks.lift(10.0)
    settle(forks, start=10.0)
    assert forks.loaded
    forks.park(20.0)
    assert not forks.loaded, 'a pallet put down is not being carried'


def test_the_first_update_establishes_the_clock_rather_than_integrating(forks,
                                                                       params):
    """`_updated_at` is None until the first update, never 0.0 -- 0.0 is a
    time. The same sentinel mistake as FrameGate.last_start."""
    forks.lower_to_insert(1000.0)
    forks.update(1000.0)
    assert forks.height == pytest.approx(params['park_height'])
    forks.update(1000.1)
    assert forks.height < params['park_height']


def test_a_backwards_clock_does_not_move_the_forks(forks, params):
    forks.lower_to_insert(10.0)
    forks.update(10.0)
    forks.update(9.0)
    assert forks.height == pytest.approx(params['park_height'])


def test_seconds_to_is_the_time_a_move_would_take(forks, params):
    assert forks.seconds_to(params['insert_height']) == pytest.approx(
        (params['park_height'] - params['insert_height']) / params['speed'])


# -------------------------------------------------------------- pallet lift

def test_the_same_travel_lifts_the_two_faces_differently(params, yaml_params):
    """`lift_travel` is carriage travel, not pallet lift. The tine top takes up
    the slack to the top of the pocket first -- 20 mm on the short face, 45 mm
    on the long -- so the pallet rises by the difference."""
    sides = yaml_params['/**']['ros__parameters']['pallet']['sides']
    short = pallet_lift(sides['short'], params)
    long_ = pallet_lift(sides['long'], params)
    assert short == pytest.approx(0.050)
    assert long_ == pytest.approx(0.025)
    assert short > long_


def test_both_faces_actually_come_off_the_floor(params, yaml_params):
    """The only claim a virtual pickup can make. If the carriage travel were
    less than the slack above the tines, the forks would rise inside the pocket
    and the pallet would not move at all."""
    sides = yaml_params['/**']['ros__parameters']['pallet']['sides']
    for name, side in sides.items():
        assert pallet_lift(side, params) > 0.010, (
            f'{name}: {params["lift_travel"] * 1000:.0f} mm of carriage travel '
            f'lifts the pallet {pallet_lift(side, params) * 1000:.0f} mm')


def test_too_little_travel_reports_no_lift_rather_than_a_negative_one(params,
                                                                      yaml_params):
    sides = yaml_params['/**']['ros__parameters']['pallet']['sides']
    tiny = dict(params, lift_travel=0.005)
    assert pallet_lift(sides['long'], tiny) == 0.0


def test_the_tines_clear_the_opening_at_the_insertion_height(params, yaml_params):
    """The height the forks are lowered to has to fit inside both openings, or
    the insertion drives them into a board. Checked against the measured
    geometry rather than against forks.bottom_height, which is the same number
    and so would prove nothing."""
    sides = yaml_params['/**']['ros__parameters']['pallet']['sides']
    for name, side in sides.items():
        assert params['insert_height'] >= side['opening_bottom']
        assert (params['insert_height'] + params['tine_thickness']
                <= side['opening_top']), (
            f'{name}: the tines do not fit in the opening at the insertion height')
