"""Velocity commands and what every one of them passes through.

1. `clamp_command`: nothing above the limits in params.yaml reaches the base.
2. `ZERO` is a real, published command: the base has no command timeout.
3. `RateLimiter` bounds acceleration.

The control laws live in `closing.py` and `insertion.py` (D38). No ROS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ['Command', 'ZERO', 'clamp_command', 'apply_deadband', 'RateLimiter']


@dataclass(frozen=True)
class Command:
    """A velocity command. Maps straight onto geometry_msgs/Twist."""

    linear: float = 0.0
    angular: float = 0.0

    def is_zero(self) -> bool:
        return self.linear == 0.0 and self.angular == 0.0


ZERO = Command(0.0, 0.0)
"""The stop command. Published explicitly on every stop and in every terminal
state, because nothing else will stop the base."""


def clamp_command(command: Command, max_linear: float, max_angular: float) -> Command:
    """Clamp a command to the configured limits.

    The last gate before the base. `max_linear` is 0.20 m/s, the lab limit for
    autonomous runs; never raise it without being asked.
    """
    return Command(
        linear=max(-max_linear, min(max_linear, command.linear)),
        angular=max(-max_angular, min(max_angular, command.angular)),
    )


def apply_deadband(command: Command, min_linear: float, min_angular: float) -> Command:
    """Raise small non-zero components up to the base's minimum movable speed.

    A skid-steer does not move at all below a certain command: the wheels just
    load up against static friction. Below that threshold the robot sits still
    while the controller believes it is closing the error, so it never arrives.

    Only applied to components that are already non-zero, and the phase logic
    only produces non-zero components while the error is outside tolerance, so
    this cannot cause hunting around the goal.
    """
    linear = command.linear
    angular = command.angular
    if linear != 0.0 and abs(linear) < min_linear:
        linear = math.copysign(min_linear, linear)
    if angular != 0.0 and abs(angular) < min_angular:
        angular = math.copysign(min_angular, angular)
    return Command(linear, angular)


class RateLimiter:
    """Bound how fast a command may change, in m/s^2 and rad/s^2.

    Keeps a jump in the pallet estimate from becoming a jerk at the wheels, and
    keeps the robot's motion smooth enough that the wheel odometry stays
    trustworthy -- a skid-steer slipping under hard acceleration is exactly when
    odometry lies, and the approach depends on it.
    """

    def __init__(self, max_linear_accel: float, max_angular_accel: float):
        self.max_linear_accel = max_linear_accel
        self.max_angular_accel = max_angular_accel
        self.previous = ZERO

    def reset(self) -> None:
        """Forget the previous command. Call on activation and on every stop."""
        self.previous = ZERO

    def limit(self, desired: Command, dt: float) -> Command:
        if dt <= 0.0:
            return self.previous
        max_dv = self.max_linear_accel * dt
        max_dw = self.max_angular_accel * dt
        linear = self.previous.linear + max(-max_dv, min(max_dv, desired.linear - self.previous.linear))
        angular = self.previous.angular + max(-max_dw, min(max_dw, desired.angular - self.previous.angular))
        self.previous = Command(linear, angular)
        return self.previous


# `compute_command`, `_finish`, `_rotate` and `_drive` lived here: the
# rotate-drive-rotate law for driving to a pose. Removed with the two-leg
# approach they served (D38) rather than left unused, because dead code with
# passing tests behind it reads as a design that is still in force. The laws
# that replaced it are `closing.closing_command` (servo onto the pocket centre
# line) and `closing.straight_command` (drive a distance on odometry); both
# finish through `clamp_command` and `apply_deadband` above.
