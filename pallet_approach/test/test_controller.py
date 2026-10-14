"""Tests for controller.py.

The first group is the control law; the second group is the safety properties,
which are the ones that must never be relaxed to make a run work.
"""

import math

import pytest

from pallet_approach.controller import (
    ZERO,
    Command,
    Phase,
    RateLimiter,
    apply_deadband,
    clamp_command,
    compute_command,
)
from pallet_approach.model import Pose2D

GAINS = {'k_rho': 0.6, 'k_alpha': 1.2, 'k_beta': 0.4}
LIMITS = {'max_linear': 0.20, 'max_angular': 0.30, 'min_linear': 0.05, 'min_angular': 0.10}
TOL = {'position': 0.030, 'yaw': 0.030, 'heading_before_drive': 0.100}


def step(current, target, phase):
    return compute_command(current, target, phase, GAINS, LIMITS, TOL)


def simulate(start, target, dt=0.05, max_steps=4000):
    """Integrate the unicycle model until DONE. Returns (pose, steps, phases)."""
    pose, phase = start, Phase.ROTATE_TO_HEADING
    limiter = RateLimiter(0.30, 0.60)
    phases = []
    for n in range(max_steps):
        command, phase = step(pose, target, phase)
        command = limiter.limit(command, dt)
        assert abs(command.linear) <= LIMITS['max_linear'] + 1e-9
        assert abs(command.angular) <= LIMITS['max_angular'] + 1e-9
        pose = Pose2D(pose.x + command.linear * math.cos(pose.yaw) * dt,
                      pose.y + command.linear * math.sin(pose.yaw) * dt,
                      pose.yaw + command.angular * dt)
        if not phases or phases[-1] != phase:
            phases.append(phase)
        if phase is Phase.DONE:
            return pose, n + 1, phases
    pytest.fail(f'did not converge in {max_steps} steps')


# ------------------------------------------------------------- control law

def test_converges_to_a_target_ahead_and_to_the_side():
    pose, steps, phases = simulate(Pose2D(0.0, 0.4, 0.0), Pose2D(1.3, 0.0, 0.0))
    assert math.hypot(pose.x - 1.3, pose.y) <= TOL['position']
    assert abs(pose.yaw) <= TOL['yaw'] + 0.01
    assert phases == [Phase.ROTATE_TO_HEADING, Phase.DRIVE, Phase.ROTATE_TO_FINAL, Phase.DONE]


@pytest.mark.parametrize('target', [
    Pose2D(1.3, 0.0, 0.0),
    Pose2D(1.0, 0.8, math.radians(-30)),
    Pose2D(0.5, -1.2, math.radians(90)),
    Pose2D(-1.0, 0.3, math.radians(170)),
])
def test_converges_from_varied_geometry(target):
    pose, _, _ = simulate(Pose2D(0.0, 0.0, 0.0), target)
    assert math.hypot(pose.x - target.x, pose.y - target.y) <= TOL['position'] + 0.01
    assert abs(math.atan2(math.sin(pose.yaw - target.yaw),
                          math.cos(pose.yaw - target.yaw))) <= TOL['yaw'] + 0.02


def test_rotates_before_driving():
    """Facing away from the target, the first command must not go forward."""
    command, phase = step(Pose2D(0.0, 0.0, math.pi), Pose2D(1.3, 0.0, 0.0),
                          Phase.ROTATE_TO_HEADING)
    assert command.linear == 0.0
    assert command.angular != 0.0
    assert phase is Phase.ROTATE_TO_HEADING


def test_returns_to_rotating_if_the_target_moves_sideways():
    """Re-estimation can move the target. Driving at the old bearing is wrong."""
    _, phase = step(Pose2D(0.0, 0.0, 0.0), Pose2D(0.5, 1.5, 0.0), Phase.DRIVE)
    assert phase is Phase.ROTATE_TO_HEADING


def test_slows_down_as_it_approaches():
    far, _ = step(Pose2D(0.0, 0.0, 0.0), Pose2D(2.0, 0.0, 0.0), Phase.DRIVE)
    near, _ = step(Pose2D(1.85, 0.0, 0.0), Pose2D(2.0, 0.0, 0.0), Phase.DRIVE)
    assert near.linear < far.linear


def test_never_reverses():
    """Backing up mid-approach would hide the pallet from the camera."""
    for target in (Pose2D(-1.0, 0.0, 0.0), Pose2D(-0.5, -0.5, 1.0)):
        pose, phase = Pose2D(0.0, 0.0, 0.0), Phase.ROTATE_TO_HEADING
        for _ in range(400):
            command, phase = step(pose, target, phase)
            assert command.linear >= 0.0
            pose = Pose2D(pose.x + command.linear * math.cos(pose.yaw) * 0.05,
                          pose.y + command.linear * math.sin(pose.yaw) * 0.05,
                          pose.yaw + command.angular * 0.05)
            if phase is Phase.DONE:
                break


# --------------------------------------------------------------- safety

def test_done_is_sticky_and_zero():
    for _ in range(3):
        command, phase = step(Pose2D(0, 0, 0), Pose2D(9, 9, 0), Phase.DONE)
        assert command.is_zero()
        assert phase is Phase.DONE


def test_clamp_respects_the_lab_limits():
    command = clamp_command(Command(5.0, -9.0), LIMITS['max_linear'], LIMITS['max_angular'])
    assert command.linear == LIMITS['max_linear']
    assert command.angular == -LIMITS['max_angular']


@pytest.mark.parametrize('target', [
    Pose2D(4.0, 0.0, 0.0), Pose2D(0.1, 3.0, 2.0), Pose2D(-3.0, -3.0, -3.0),
])
def test_no_command_ever_exceeds_the_limits(target):
    """Exhaustive over phases: nothing above the configured limit gets out."""
    for phase in Phase:
        for yaw in (0.0, 1.0, -2.5, math.pi):
            command, _ = step(Pose2D(0.0, 0.0, yaw), target, phase)
            assert abs(command.linear) <= LIMITS['max_linear'] + 1e-12
            assert abs(command.angular) <= LIMITS['max_angular'] + 1e-12


def test_deadband_lifts_small_commands_but_leaves_zero_alone():
    lifted = apply_deadband(Command(0.001, -0.001), 0.05, 0.10)
    assert lifted.linear == 0.05
    assert lifted.angular == -0.10
    assert apply_deadband(ZERO, 0.05, 0.10).is_zero()


def test_deadband_never_breaks_the_limits():
    """A deadband larger than the limit would be a configuration error; the
    order clamp-then-deadband must still not emit something unsafe."""
    limits = {'max_linear': 0.20, 'max_angular': 0.30, 'min_linear': 0.05, 'min_angular': 0.10}
    command, _ = compute_command(Pose2D(0, 0, 0), Pose2D(0.031, 0, 0),
                                 Phase.DRIVE, GAINS, limits, TOL)
    assert abs(command.linear) <= limits['max_linear']


def test_rate_limiter_bounds_acceleration():
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
