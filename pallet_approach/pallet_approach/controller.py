"""Pose controller: drive the base to a target pose.

Rotate - drive - rotate, which is the simplest thing that works on a skid-steer
base and the easiest to reason about when something goes wrong in the lab. The
robot turns to face the target, drives to it, then turns to the final heading.

No ROS here: `compute_command` is a pure function of (where I am, where I want
to be, phase), so the whole control law is unit testable at a desk. The node
wrapper owns the clock, the odometry subscription and the publisher.

Three safety properties live in this file and must not be weakened:

1. Every command passes through `clamp_command`, so nothing above the limits in
   params.yaml can ever reach the base.
2. `ZERO` is a real command that gets published, not an absence of commands.
   The base driver has no watchdog -- `ros2 param dump /scout` has no timeout
   parameter at all -- so a stale non-zero command would keep the robot moving.
3. `RateLimiter` bounds acceleration, so a jump in the estimate cannot become a
   jerk at the wheels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from pallet_approach.approach import wrap_angle
from pallet_approach.model import Pose2D

__all__ = ['Phase', 'Command', 'ZERO', 'clamp_command', 'apply_deadband',
           'RateLimiter', 'compute_command']


class Phase(Enum):
    """Where in the rotate-drive-rotate sequence the controller is."""

    ROTATE_TO_HEADING = 'ROTATE_TO_HEADING'
    DRIVE = 'DRIVE'
    ROTATE_TO_FINAL = 'ROTATE_TO_FINAL'
    DONE = 'DONE'


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


def compute_command(
    current: Pose2D,
    target: Pose2D,
    phase: Phase,
    gains: dict,
    limits: dict,
    tolerances: dict,
) -> tuple[Command, Phase]:
    """One control step. Returns the command to send and the next phase.

    Both poses must be in the same frame -- `odom` in the running system, since
    that is where the coordinator stores its targets.

    The sequence:

    * ROTATE_TO_HEADING -- turn in place until the target bears less than
      `heading_before_drive` off the nose. Turning away from the pallet is fine
      here; nothing is moving forward yet.
    * DRIVE -- drive forward, steering gently to hold the bearing. If the
      bearing opens up past twice the threshold, go back to rotating: that
      means the target moved (re-estimation) or the robot slipped.
    * ROTATE_TO_FINAL -- in position; turn to the target heading, which at the
      standoff means square-on to the pallet face.
    * DONE -- `ZERO`, every time it is called.

    Note what is deliberately absent: no integral term, no reverse. An
    integrator would wind up while the base sits in its deadband, and reversing
    mid-approach hides the pallet from the camera.
    """
    if phase is Phase.DONE:
        return ZERO, Phase.DONE

    dx = target.x - current.x
    dy = target.y - current.y
    rho = math.hypot(dx, dy)
    bearing = wrap_angle(math.atan2(dy, dx) - current.yaw)
    final_error = wrap_angle(target.yaw - current.yaw)

    position_tolerance = tolerances['position']
    yaw_tolerance = tolerances['yaw']
    heading_gate = tolerances['heading_before_drive']

    if phase is Phase.ROTATE_TO_HEADING:
        if rho <= position_tolerance:
            # already there; skip straight to the final heading
            return _rotate(final_error, yaw_tolerance, gains['k_beta'],
                           limits, Phase.ROTATE_TO_FINAL, Phase.DONE)
        if abs(bearing) <= heading_gate:
            return _drive(rho, bearing, gains, limits, position_tolerance)
        return _rotate(bearing, heading_gate, gains['k_alpha'],
                       limits, Phase.ROTATE_TO_HEADING, Phase.DRIVE)

    if phase is Phase.DRIVE:
        if rho <= position_tolerance:
            return _rotate(final_error, yaw_tolerance, gains['k_beta'],
                           limits, Phase.ROTATE_TO_FINAL, Phase.DONE)
        if abs(bearing) > 2.0 * heading_gate:
            return _rotate(bearing, heading_gate, gains['k_alpha'],
                           limits, Phase.ROTATE_TO_HEADING, Phase.DRIVE)
        return _drive(rho, bearing, gains, limits, position_tolerance)

    # ROTATE_TO_FINAL
    return _rotate(final_error, yaw_tolerance, gains['k_beta'],
                   limits, Phase.ROTATE_TO_FINAL, Phase.DONE)


def _finish(command: Command, limits: dict) -> Command:
    """Clamp then deadband, in that order, so the deadband is never clipped away."""
    command = clamp_command(command, limits['max_linear'], limits['max_angular'])
    return apply_deadband(command, limits['min_linear'], limits['min_angular'])


def _rotate(error, tolerance, gain, limits, stay, done) -> tuple[Command, Phase]:
    """Turn in place towards `error`, or hand over to `done` once inside tolerance."""
    if abs(error) <= tolerance:
        return ZERO, done
    return _finish(Command(0.0, gain * error), limits), stay


def _drive(rho, bearing, gains, limits, position_tolerance) -> tuple[Command, Phase]:
    """Drive forward, steering gently. Slows down as the target gets close."""
    command = Command(linear=gains['k_rho'] * rho, angular=gains['k_alpha'] * bearing)
    return _finish(command, limits), Phase.DRIVE
