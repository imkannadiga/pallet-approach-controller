"""config/twist_mux.yaml: exactly one source drives the base (D3, D10, D30).

Arbitration is configuration, not code, which means nothing in this repository
executes it and a typo in a topic name produces a robot that does not move --
or, worse, one that moves when something else wanted it stopped. These checks
are what stands in for running it.

What they cannot check is that twist_mux reads this schema. That is verified on
the robot, in lab session A, and the schema here is the documented Humble one.
"""

import pathlib

import pytest
import yaml

PACKAGE = pathlib.Path(__file__).resolve().parent.parent
MUX_FILE = PACKAGE / 'config' / 'twist_mux.yaml'
PARAMS_FILE = PACKAGE / 'config' / 'params.yaml'


@pytest.fixture(scope='module')
def mux():
    return yaml.safe_load(MUX_FILE.read_text())['twist_mux']['ros__parameters']


@pytest.fixture(scope='module')
def topics():
    params = yaml.safe_load(PARAMS_FILE.read_text())
    return params['/**']['ros__parameters']['topics']


def test_the_priority_order_is_safety_then_us_then_nav2(mux):
    """D10's order, as a comparison rather than as three numbers that happen to
    be right. Nothing we publish may outrank the safety stop."""
    by_name = {name: block['priority'] for name, block in mux['topics'].items()}
    assert by_name['safety'] > by_name['pallet'] > by_name['navigation'], (
        f'priorities {by_name} do not put safety above the pallet controller '
        f'above Nav2')


def test_every_input_has_a_timeout(mux):
    """The base has no command timeout of its own (`ros2 param dump /scout`),
    so a publisher that crashes mid-command would otherwise leave its last
    command driving the robot indefinitely."""
    for name, block in mux['topics'].items():
        assert block.get('timeout', 0) > 0, f'input {name} has no timeout'


def test_the_input_timeouts_are_looser_than_the_watchdog(mux):
    """Deliberate: the watchdog should notice first and stop the robot with a
    reason that reaches the operator. The mux timeout is the backstop for the
    watchdog itself dying, so it has to be the slower of the two."""
    params = yaml.safe_load(PARAMS_FILE.read_text())
    worst = max(params['watchdog']['ros__parameters']['max_age'].values())
    for name, block in mux['topics'].items():
        assert block['timeout'] >= worst, (
            f'input {name} times out in {block["timeout"]} s, before the '
            f'watchdog\'s slowest staleness limit of {worst} s')


def test_the_mux_inputs_are_the_topics_the_nodes_publish(mux, topics):
    """The failure this file exists for: a renamed topic on one side only.

    Nothing connects the YAML to the code at runtime, so a mux listening to
    /cmd_vel_pallet while the controller publishes /cmd_vel_pallet_out is a
    robot that reports it is driving and sits still.
    """
    configured = {block['topic'] for block in mux['topics'].values()}
    assert topics['cmd_vel_pallet'] in configured
    assert topics['cmd_vel_safety'] in configured


def test_the_base_topic_is_not_also_an_input(mux, topics):
    """twist_mux publishes its winner there. An input on the same topic would
    feed the mux its own output."""
    configured = {block['topic'] for block in mux['topics'].values()}
    assert topics['cmd_vel_base'] not in configured


def test_the_nav_lock_blocks_nav2_and_nothing_of_ours(mux, topics):
    """A lock blocks every input at or below its own priority. Set it too high
    and holding it would block our own controller, or even the safety stop --
    the coordinator would then be unable to drive the robot it just locked."""
    locks = mux['locks']
    assert len(locks) == 1, f'expected one lock, found {sorted(locks)}'
    lock = next(iter(locks.values()))
    assert lock['topic'] == topics['nav_lock']
    priorities = {name: block['priority'] for name, block in mux['topics'].items()}
    assert priorities['navigation'] <= lock['priority'], (
        'the lock does not block Nav2, which is its only job')
    assert lock['priority'] < priorities['pallet'] <= priorities['safety'], (
        'the lock also blocks our own controller or the safety stop')


def test_the_lock_expires(mux):
    """A lock with timeout 0 is held for ever once published. If the
    coordinator died mid-run the robot would be locked out of navigation until
    somebody noticed, which is a failure that needs a human to clear."""
    lock = next(iter(mux['locks'].values()))
    assert lock['timeout'] > 0, 'the nav lock never expires'
