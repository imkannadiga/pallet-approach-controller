"""Tests for controller.py: what every velocity command passes through.

The rotate-drive-rotate law that used to live here went with the two-leg
approach (D38); its replacements are tested in `test_closing.py` and
`test_insertion.py`. What is left is the three things shared by all of them,
and they are shared so that the safety properties are enforced in one place
rather than three. None of these may be relaxed to make a run work.
"""

import pytest

from pallet_approach.controller import (
    ZERO,
    Command,
    RateLimiter,
    apply_deadband,
    clamp_command,
)

LIMITS = {'max_linear': 0.20, 'max_angular': 0.30,
          'min_linear': 0.05, 'min_angular': 0.10}


# ------------------------------------------------------------------- clamp

def test_clamp_respects_the_lab_limits():
    """0.20 m/s is the lab limit for autonomous runs; nothing above it may
    reach the base, whatever a gain asks for."""
    command = clamp_command(Command(5.0, -9.0), LIMITS['max_linear'],
                            LIMITS['max_angular'])
    assert command.linear == LIMITS['max_linear']
    assert command.angular == -LIMITS['max_angular']


def test_clamp_leaves_a_legal_command_alone():
    command = Command(0.1, -0.2)
    assert clamp_command(command, 0.20, 0.30) == command


def test_clamp_is_symmetric():
    assert clamp_command(Command(-5.0, 9.0), 0.20, 0.30) == Command(-0.20, 0.30)


# ---------------------------------------------------------------- deadband

def test_deadband_lifts_small_commands_but_leaves_zero_alone():
    """A skid steer does not move at all below roughly these speeds: the wheels
    load up against static friction while the controller believes it is closing
    the error, so it never arrives. Zero is left alone because zero means stop."""
    lifted = apply_deadband(Command(0.001, -0.001), 0.05, 0.10)
    assert lifted.linear == 0.05
    assert lifted.angular == -0.10
    assert apply_deadband(ZERO, 0.05, 0.10).is_zero()


def test_deadband_keeps_the_sign():
    lifted = apply_deadband(Command(-0.001, 0.001), 0.05, 0.10)
    assert lifted.linear == -0.05
    assert lifted.angular == 0.10


def test_a_zero_minimum_disables_the_deadband():
    """Which is how `closing._finish` gets a zeroed angular command instead of
    an inflated one -- the one change that gave the closing servo any fine
    authority (D40)."""
    assert apply_deadband(Command(0.001, 0.001), 0.0, 0.0) == Command(0.001, 0.001)


# ------------------------------------------------------------ rate limiter

def test_rate_limiter_bounds_acceleration():
    """A jump in the pallet estimate must not become a jerk at the wheels, and
    a skid steer accelerating hard is exactly when its odometry lies."""
    limiter = RateLimiter(max_linear_accel=0.30, max_angular_accel=0.60)
    first = limiter.limit(Command(0.20, 0.30), dt=0.05)
    assert first.linear == pytest.approx(0.015)      # 0.30 * 0.05
    assert first.angular == pytest.approx(0.030)
    second = limiter.limit(Command(0.20, 0.30), dt=0.05)
    assert second.linear == pytest.approx(0.030)


def test_rate_limiter_also_bounds_deceleration_and_resets_to_zero():
    limiter = RateLimiter(0.30, 0.60)
    for _ in range(20):
        limiter.limit(Command(0.20, 0.0), dt=0.05)
    braking = limiter.limit(ZERO, dt=0.05)
    assert 0.0 < braking.linear < 0.20
    limiter.reset()
    assert limiter.previous.is_zero()


def test_rate_limiter_holds_on_a_zero_timestep():
    limiter = RateLimiter(0.30, 0.60)
    limiter.limit(Command(0.1, 0.1), dt=0.05)
    held = limiter.limit(Command(0.2, 0.2), dt=0.0)
    assert held == limiter.previous


def test_rate_limiter_reaches_the_commanded_speed_eventually():
    """It bounds the rate of change, not the value. A limiter that never let
    the command through would be a speed limit by accident."""
    limiter = RateLimiter(0.30, 0.60)
    for _ in range(100):
        command = limiter.limit(Command(0.15, -0.20), dt=0.05)
    assert command.linear == pytest.approx(0.15)
    assert command.angular == pytest.approx(-0.20)


# ------------------------------------------------------------------- ZERO

def test_zero_is_a_command_and_not_an_absence():
    """The base driver has no command timeout, so stopping means sending this,
    repeatedly, rather than stopping sending."""
    assert ZERO.is_zero()
    assert ZERO == Command(0.0, 0.0)
    assert not Command(0.0, 0.1).is_zero()
