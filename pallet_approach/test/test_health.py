"""Tests for health.py -- the thing that decides whether we may drive.

The interesting cases are all about absent information rather than bad
information, because that is where this kind of code goes wrong: a dead topic
reported as fresh, a rate of zero reported before any rate could be known, a
timestamp from a clock that disagrees with ours.
"""

import pytest

from pallet_approach.health import (
    ALWAYS,
    NEVER,
    WHILE_RUNNING,
    HealthMonitor,
    Limits,
    as_float,
)


def monitor(**kwargs):
    limits = {'odom': Limits(max_age=0.20, min_rate=20.0),
              'estimate': Limits(max_age=0.50, min_rate=2.5, when=WHILE_RUNNING)}
    limits.update(kwargs)
    return HealthMonitor(limits, window=2.0)


def feed(m, name, rate, until, start=0.0):
    """Publish `name` at `rate` Hz from `start` to `until`. Returns `until`."""
    t = start
    while t <= until + 1e-9:
        m.seen(name, t)
        t += 1.0 / rate
    return until


# ------------------------------------------------------- absent, not stale

def test_a_source_that_has_never_published_is_not_fresh():
    """The bug this whole file is built around. An age of 0.0 for a topic that
    has produced nothing passes every staleness limit that could be written."""
    m = monitor()
    assert m.age('odom', 5.0) is None
    report = m.check(5.0)
    assert not report.ok
    assert 'nothing received since startup' in report['odom'].detail


def test_zero_is_a_valid_clock_reading_not_a_sentinel():
    """A message that arrives at t = 0.0 is a message, not an absence.

    The same mistake as `FrameGate.last_start = 0.0` and the `False` sentinel
    in the overlay's colour handling: a legal value used to mean "unset".
    """
    m = monitor()
    m.seen('odom', 0.0)
    assert m.age('odom', 0.0) == 0.0
    assert m.check(0.0)['odom'].ok


# ----------------------------------------------------- not yet measurable

def test_a_rate_is_none_until_the_window_holds_enough_history():
    """Reported as 0.0 Hz, an unmeasurable rate would trip the rate limit on
    every run inside the first second."""
    m = monitor()
    m.seen('odom', 0.0)
    assert m.rate('odom', 0.0) is None          # one sample is not a rate
    feed(m, 'odom', 50.0, until=0.3)
    assert m.rate('odom', 0.3) is None          # 0.3 s of a 2 s window
    assert m.check(0.3)['odom'].ok, 'an unmeasurable rate must not fail the check'


def test_a_measurable_rate_is_measured():
    m = monitor()
    feed(m, 'odom', 50.0, until=1.5)
    assert m.rate('odom', 1.5) == pytest.approx(50.0, rel=0.05)


def test_a_source_that_is_alive_but_degraded_fails_on_rate():
    """The case the age limit cannot catch: every message arrives, none is
    stale, and the sensor is producing a fifth of what the pipeline needs."""
    m = monitor()
    feed(m, 'odom', 5.0, until=1.8)
    report = m.check(1.8)
    assert not report.ok
    assert 'Hz' in report['odom'].detail


def test_the_window_slides():
    """An old burst must not keep a dead source looking healthy."""
    m = monitor()
    feed(m, 'odom', 50.0, until=1.5)
    assert m.rate('odom', 10.0) is None, 'samples older than the window stay out'


# -------------------------------------------------------------- clock skew

def test_a_stamp_from_the_future_is_its_own_failure():
    """Measured 2026-10-15: the ZED stamped frames on a clock running 24.5 %
    fast. A negative age satisfies every `age > max_age` test ever written, so
    the stack would have called that tf perfectly fresh."""
    m = HealthMonitor({'tf': Limits(max_age=0.30)})
    m.seen('tf', 100.0)
    report = m.check(10.0)
    assert not report.ok
    assert 'future' in report['tf'].detail


def test_ordinary_clock_jitter_is_not_skew():
    m = HealthMonitor({'tf': Limits(max_age=0.30, max_skew=0.05)})
    m.seen('tf', 10.01)
    assert m.check(10.0)['tf'].ok


# --------------------------------------------------- what is required when

def test_perception_is_not_required_between_runs():
    """Perception is a managed lifecycle node and publishes nothing while idle.
    Requiring it always would leave the stack permanently unhealthy, and D12's
    goal-time check would then refuse every goal."""
    m = monitor()
    feed(m, 'odom', 50.0, until=1.5)
    assert m.check(1.5, ALWAYS).ok
    assert not m.check(1.5, WHILE_RUNNING).ok


def test_perception_is_required_during_a_run():
    m = monitor()
    feed(m, 'odom', 50.0, until=1.5)
    feed(m, 'estimate', 5.0, until=1.5)
    assert m.check(1.5, WHILE_RUNNING).ok


def test_forgetting_a_source_makes_it_unseen_again():
    """Last run's estimates are not evidence about this one."""
    m = monitor()
    feed(m, 'estimate', 5.0, until=1.5)
    m.forget('estimate')
    assert m.age('estimate', 1.5) is None


def test_a_never_source_is_reported_but_cannot_fail_the_stack():
    """For a check that is deliberately switched off -- the tf age under the
    bench `tf_fallback_to_latest` workaround. The number still has to appear,
    or 'we are not enforcing this' becomes invisible."""
    m = monitor(tf=Limits(max_age=0.30, when=NEVER))
    feed(m, 'odom', 50.0, until=1.5)
    report = m.check(1.5)
    assert report.ok
    assert not report['tf'].ok, 'it is still judged'
    assert not report['tf'].required
    assert report.failures == (), 'and still kept out of the failure list'


# ------------------------------------------------------------- mechanics

def test_failures_name_the_source_and_carry_the_numbers():
    """These strings reach the operator as the reason a run stopped."""
    m = monitor()
    feed(m, 'odom', 50.0, until=1.5)
    report = m.check(2.0)
    assert len(report.failures) == 1
    assert report.failures[0].startswith('odom: ')
    assert '0.50' in report.failures[0] and '0.20' in report.failures[0]


def test_a_monitor_with_no_sources_is_refused():
    """It would report `ok` for ever, which is the one answer a watchdog must
    never give by default."""
    with pytest.raises(ValueError):
        HealthMonitor({})


def test_an_unknown_source_name_raises():
    """A typo in a subscription callback would otherwise silently record
    nothing, and the source it was meant to feed would read as dead."""
    with pytest.raises(KeyError):
        monitor().seen('odometry', 1.0)


def test_none_becomes_nan_not_zero():
    """For the message fields, which have no optional type. A reader who
    forgets to check gets something obviously absent rather than a plausible
    wrong number."""
    assert as_float(None) != as_float(None)      # NaN
    assert as_float(0.25) == 0.25


def test_a_source_silent_since_the_run_started_is_fine_inside_its_grace():
    # D58: perception's first estimate needs time after the run activates it.
    monitor = HealthMonitor({'estimate': Limits(max_age=0.5, when=WHILE_RUNNING,
                                                first_message_grace=2.0)})
    assert monitor.check(11.0, WHILE_RUNNING, running_since=10.0).ok
    late = monitor.check(12.5, WHILE_RUNNING, running_since=10.0)
    assert not late.ok and 'nothing received' in late['estimate'].detail


def test_the_grace_covers_only_the_first_message():
    # Once something has arrived, a silence is a stale source, grace or not.
    monitor = HealthMonitor({'estimate': Limits(max_age=0.5, when=WHILE_RUNNING,
                                                first_message_grace=2.0)})
    monitor.seen('estimate', 10.2)
    assert not monitor.check(11.0, WHILE_RUNNING, running_since=10.0).ok


def test_no_run_start_means_no_grace():
    monitor = HealthMonitor({'estimate': Limits(max_age=0.5, when=WHILE_RUNNING,
                                                first_message_grace=2.0)})
    assert not monitor.check(11.0, WHILE_RUNNING).ok


def test_the_estimate_is_required_only_while_the_camera_is_used():
    # D64: from LOWERING on, the insertion is blind by design.
    from pallet_approach.health import phase_for_state
    for state in ('CONFIRMING', 'CLASSIFYING', 'BACKING_OFF', 'CLOSING', 'CHECKING'):
        assert phase_for_state(state) == WHILE_RUNNING
    for state in ('LOWERING', 'INSERTING', 'LIFTING', 'BACKING_OUT', 'IDLE', 'DONE'):
        assert phase_for_state(state) == ALWAYS
    monitor = HealthMonitor({'estimate': Limits(max_age=0.5, when=WHILE_RUNNING),
                             'odom': Limits(max_age=0.2)})
    monitor.seen('estimate', 0.0)
    monitor.seen('odom', 9.9)
    assert monitor.check(10.0, phase_for_state('INSERTING')).ok       # stale estimate, fine
    assert not monitor.check(10.0, phase_for_state('CLOSING')).ok     # same, not fine
    assert not monitor.check(10.5, phase_for_state('INSERTING')).ok   # odometry still required


def test_the_perception_states_are_real_coordinator_states():
    import yaml, pathlib
    from pallet_approach.coordinator import State
    from pallet_approach.health import PERCEPTION_STATES
    params = yaml.safe_load((pathlib.Path(__file__).resolve().parent.parent
                             / 'config' / 'params.yaml').read_text())
    listed = params['watchdog']['ros__parameters']['estimate_required_states']
    names = {s.value for s in State}
    assert set(listed) <= names and set(PERCEPTION_STATES) <= names
    assert set(listed) == set(PERCEPTION_STATES)
