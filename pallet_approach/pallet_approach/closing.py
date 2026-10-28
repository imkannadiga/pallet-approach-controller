"""Closing in on the pallet (D38), and the reverse that makes room to do it.

A continuous servo onto the pocket centre line (C + t n; C = pocket midpoint,
n = outward face normal) while the robot drives in:

* outer loop: aim theta_desired = -k_cross * offset, capped at
  `max_approach_angle` so the pallet stays in view;
* inner loop: omega = k_heading * (theta_desired - theta).

The offset decays with length 1 / k_cross (what `required_run_up` uses). The
line's direction is the face normal, so arriving on it is arriving aligned.
The run-up length, not the gain, limits the lateral error (65 mm of pocket
slack at the standoff). `closing_command` refuses to move once the virtual
tine tips would be inside the pallet (commit point), whatever the state.
Pure functions; the node owns the clock, odometry and publisher.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from pallet_approach.approach import outward_normal, pocket_center, wrap_angle
from pallet_approach.controller import ZERO, Command, apply_deadband, clamp_command
from pallet_approach.model import PalletEstimate

__all__ = ['Geometry', 'Phase', 'geometry_of', 'required_run_up',
           'closing_command', 'desired_heading', 'DivergenceDetector',
           'rear_clearance', 'reverse_allowance', 'straight_command',
           'commit_range']


class Phase(Enum):
    """Where the closing loop is. SQUARING turns in place at the standoff (D41):
    arriving mid-correction can cost 3-4 deg of yaw, and turning about
    `base_footprint` does not change the lateral offset.
    """

    CLOSING = 'CLOSING'
    SQUARING = 'SQUARING'
    ARRIVED = 'ARRIVED'


@dataclass(frozen=True)
class Geometry:
    """Where the robot stands relative to the pocket centre line, in the
    estimate's frame (`base_footprint`). The closing loop uses
    `geometry_from_track` (D59); single-frame judgements use `geometry_of`.
    """

    range: float
    """Distance from `base_footprint` to the face plane along the normal,
    metres. This is the quantity the standoff is quoted in."""

    offset: float
    """Signed lateral displacement from the pocket centre line, metres, in the
    robot's OWN frame.

    Positive means the robot is to the left of the line and has to move towards
    its own -y to get onto it; negative means the line is to its left. Stated
    in the robot's frame on purpose: "left of the line" is ambiguous until you
    say left of what, and the robot faces the pallet, so its left is the
    pallet's right. `test_closing.py` pins the sign down case by case, because
    getting it backwards gives a controller that steers away from the line
    perfectly smoothly.
    """

    heading: float
    """Signed yaw error, radians. Positive when the robot must turn left
    (towards +yaw) to point square at the face."""


def geometry_of(estimate: PalletEstimate) -> Geometry:
    """One estimate reduced to (range, offset, heading). `n` is the normal flipped
    to face the robot; travel is along -n; offset is measured along the robot's
    left, (n_y, -n_x). Sign conventions are stated case by case in the tests.
    """
    center = pocket_center(estimate.pockets, fallback=estimate.position)
    normal = outward_normal(estimate.normal, face_point=center)
    return _geometry(np.asarray(center, dtype=float)[:2], normal[:2])


def geometry_from_track(center_odom, yaw_odom: float, robot) -> Geometry:
    """(range, offset, heading) from the tracker's averaged pallet line in odom and
    the robot's pose now (D59): one frame's normal wobbles +-0.04 rad, +-120 mm
    at 3 m. `robot` is base_footprint's Pose2D in odom.
    """
    dx = float(center_odom[0]) - robot.x
    dy = float(center_odom[1]) - robot.y
    cos_r, sin_r = math.cos(robot.yaw), math.sin(robot.yaw)
    c = np.array([cos_r * dx + sin_r * dy, -sin_r * dx + cos_r * dy])
    yaw = float(yaw_odom) - robot.yaw
    n = np.array([math.cos(yaw), math.sin(yaw)])
    if c @ n > 0.0:
        n = -n      # outward, towards the robot, as outward_normal does
    return _geometry(c, n)


def _geometry(c: np.ndarray, n: np.ndarray) -> Geometry:
    """The algebra shared by both: pocket midpoint `c` and outward normal `n`,
    2D, in the robot's frame."""
    face_range = float(-c @ n)
    left = np.array([n[1], -n[0]])
    offset = float(-c @ left)
    heading = wrap_angle(math.atan2(-n[1], -n[0]))
    return Geometry(range=face_range, offset=offset, heading=heading)


def commit_range(params: dict) -> float:
    """Face distance at which the virtual tine tips reach the face, metres.

    `carriage_x + tine_length`, 1.6125 m on this robot (D15/D16). Closer than
    this the tines are inside the pallet and any steering drags them sideways
    through the blocks.
    """
    return float(params['carriage_x']) + float(params['tine_length'])


def required_run_up(offset: float, params: dict) -> float:
    """Face distance needed before closing in: the larger of `min_run_up` (D75)
    and the standoff plus the distance the servo needs to bring the current
    offset down to `residual_target`. Two regimes add up: at the angle cap the
    offset falls by sin(max_approach_angle) per metre; below
    max_approach_angle / k_cross it decays exponentially with length 1/k_cross.
    `test_closing.py` checks both against simulation.
    """
    reach = max(abs(float(offset)), 1e-4)
    residual = max(float(params['residual_target']), 1e-4)
    k_cross = float(params['k_cross'])
    cap = float(params['max_approach_angle'])

    # Below this offset the proportional term asks for less than the cap, so
    # the loop is in its exponential regime; above it, the aim is pinned at the
    # cap and the error falls at a constant rate instead.
    capped_below = cap / k_cross

    distance = 0.0
    if reach > capped_below:
        distance += (reach - capped_below) / math.sin(cap)
        reach = capped_below
    if reach > residual:
        distance += math.log(reach / residual) / k_cross
    distance += float(params['run_up_margin'])
    return max(float(params['min_run_up']),
               float(params['target_range']) + distance)


def closing_command(
    geometry: Geometry,
    phase: Phase,
    params: dict,
    limits: dict,
) -> tuple[Command, Phase, str]:
    """One step of the closing servo: (command, next phase, reason).

    Gates, in order: past the commit point -> refuse in every phase; ARRIVED ->
    zero; at the standoff -> stop and square up; aim error over
    `max_turn_in_place` -> turn without driving.
    """
    target = float(params['target_range'])
    tolerance = float(params['range_tolerance'])

    if geometry.range <= commit_range(params):
        return ZERO, phase, (
            f'face at {geometry.range:.3f} m is inside the {commit_range(params):.4f} m '
            f'fork tip reach; the closing loop must not drive there')

    if phase is Phase.ARRIVED:
        return ZERO, Phase.ARRIVED, f'arrived, face {geometry.range:.3f} m'

    if phase is Phase.SQUARING:
        # Hysteresis on the way back out: one noisy range reading must not
        # restart the drive, or the robot creeps forward a frame at a time.
        if geometry.range - target > float(params['resume_range']):
            return _closing_step(geometry, params, limits)
        return _square_up(geometry, params, limits)

    if geometry.range - target <= tolerance:
        return _square_up(geometry, params, limits)

    return _closing_step(geometry, params, limits)


def _square_up(geometry: Geometry, params: dict, limits: dict):
    """Turn in place until the robot points square at the face.

    `arrival_yaw` has to be larger than the base's own turn granularity, or
    this never finishes: below `min_angular / k_heading` radians of error the
    command is zeroed (see `_finish`), so a tolerance tighter than that is a
    tolerance the base cannot reach.
    """
    if abs(geometry.heading) <= float(params['arrival_yaw']):
        return ZERO, Phase.ARRIVED, (
            f'at the standoff and square: face {geometry.range:.3f} m, '
            f'offset {geometry.offset * 1000:+.0f} mm, '
            f'yaw {math.degrees(geometry.heading):+.2f} deg')
    command = Command(0.0, float(params['k_heading']) * geometry.heading)
    return (_finish(command, limits), Phase.SQUARING,
            f'squaring up, {math.degrees(geometry.heading):+.1f} deg to go')


def _closing_step(geometry: Geometry, params: dict, limits: dict):
    """Drive forward, steering with the two nested loops."""
    aim = desired_heading(geometry.offset, params)
    error = wrap_angle(aim + geometry.heading)

    if abs(error) > float(params['max_turn_in_place']):
        command = Command(0.0, float(params['k_heading']) * error)
        return (_finish(command, limits), Phase.CLOSING,
                f'turning in place, {math.degrees(error):+.0f} deg to the aim')

    speed = _taper(geometry.range - float(params['target_range']), params, limits)
    command = Command(speed, float(params['k_heading']) * error)
    return (_finish(command, limits), Phase.CLOSING,
            f'closing: {geometry.range:.2f} m, offset {geometry.offset * 1000:+.0f} mm, '
            f'yaw {math.degrees(geometry.heading):+.1f} deg, '
            f'aim {math.degrees(aim):+.1f} deg')


def desired_heading(offset: float, params: dict) -> float:
    """The outer loop's aim (radians, + = left of the line's direction).

    Capped at `max_approach_angle` (keeps the pallet in view) and zero inside
    `straighten_offset`, so the robot arrives square rather than still cutting
    across (at k_cross 2.5 a 35 mm residual would otherwise mean 5 deg).
    """
    magnitude = abs(float(offset))
    if magnitude < float(params['straighten_offset']):
        return 0.0
    aim = -float(params['k_cross']) * float(offset)
    cap = float(params['max_approach_angle'])
    return max(-cap, min(cap, aim))


def _taper(remaining: float, params: dict, limits: dict) -> float:
    """Slow down over the last `taper_range` metres.

    Not for precision -- the deadband means the base cannot creep anyway -- but
    so that the robot is at its slowest at its closest, which is where the
    consequence of a mistake is highest.
    """
    speed = float(params['closing_speed'])
    taper = float(params['taper_range'])
    if taper > 0.0 and remaining < taper:
        fraction = max(remaining / taper, 0.0)
        speed = max(float(limits['min_linear']), speed * fraction)
    return speed


def straight_command(
    travelled: float,
    distance: float,
    speed: float,
    limits: dict,
    tolerance: float = 0.010,
) -> tuple[Command, bool, str]:
    """Drive straight by `distance` (signed) on odometry; `travelled` is positive.
    For backing off, the blind insertion (D7) and backing out: angular is
    exactly zero.
    """
    remaining = abs(distance) - travelled
    if remaining <= tolerance:
        return ZERO, True, f'travelled {travelled:.3f} m of {abs(distance):.3f} m'
    direction = math.copysign(1.0, distance)
    command = Command(direction * abs(speed), 0.0)
    return (_finish(command, limits), False,
            f'{travelled:.3f} / {abs(distance):.3f} m, {remaining:.3f} m to go')


def _finish(command: Command, limits: dict) -> Command:
    """Clamp, then the deadbands: linear is raised to `min_linear` (below it the
    base does not move); angular below `min_angular` is ZEROED, not raised --
    raising made the servo bang-bang. That granularity sets the accuracy (D40).
    """
    command = clamp_command(command, limits['max_linear'], limits['max_angular'])
    command = apply_deadband(command, limits['min_linear'], 0.0)
    angular = (0.0 if abs(command.angular) < float(limits['min_angular'])
               else command.angular)
    return Command(command.linear, angular)


class DivergenceDetector:
    """Is the lateral error growing instead of shrinking?

    Abort when the error has grown for `frames` consecutive steps by at least
    `min_growth` in total, ignoring errors inside `min_offset`. Eight frames, not
    five: under 10 mm of noise five aborted 3 good runs in 12 (test_sequence.py).
    With `window` set (D94) a step is the mean over window / frames seconds, so
    the rule means the same at any frame rate; the growth threshold scales with
    (range / reference_range)^2, like stereo depth error.
    """

    def __init__(self, frames: int, min_growth: float, min_offset: float = 0.0,
                 window: float | None = None, reference_range: float | None = None):
        self.frames = int(frames)
        self.min_growth = float(min_growth)
        self.min_offset = float(min_offset)
        self.window = None if window is None else float(window)
        self.reference_range = reference_range
        self.reset()

    def reset(self) -> None:
        self._last: float | None = None
        self._start: float | None = None
        self.consecutive = 0
        self._history: list[tuple[float, float]] = []

    def growth_needed(self, range_: float | None) -> float:
        if not self.reference_range or range_ is None or not math.isfinite(range_):
            return self.min_growth
        return self.min_growth * max(1.0, (float(range_) / float(self.reference_range)) ** 2)

    def update(self, offset: float, now: float | None = None,
               range_: float | None = None) -> tuple[bool, str]:
        """Offer one frame's lateral offset. Returns (diverging, detail)."""
        if self.window is not None and now is not None:
            return self._update_window(abs(float(offset)), float(now), range_)
        magnitude = abs(float(offset))
        if magnitude < self.min_offset:
            # Already where it needs to be. Growth down here is noise, and the
            # servo is not trying to correct it either.
            self._last = None
            self._start = None
            self.consecutive = 0
            return False, ''
        if self._last is None or magnitude <= self._last:
            self._last = magnitude
            self._start = magnitude
            self.consecutive = 0
            return False, ''
        self.consecutive += 1
        growth = magnitude - (self._start if self._start is not None else magnitude)
        self._last = magnitude
        if self.consecutive >= self.frames and growth >= self.min_growth:
            return True, (f'lateral offset grew for {self.consecutive} frames, '
                          f'{growth * 1000:.0f} mm in total, now '
                          f'{magnitude * 1000:.0f} mm')
        return False, (f'{self.consecutive} growing frame(s), '
                       f'{growth * 1000:.0f} mm')

    def _update_window(self, magnitude: float, now: float, range_) -> tuple[bool, str]:
        """The frame rule on fixed-duration steps: each step's mean |offset| is
        one "frame". Growth is counted from the START of the rising streak, as
        in the frame rule -- a steady drift-off keeps adding up however long it
        lasts. (A first version measured only the last window, and a
        sign-flipped controller drifted 0.69 m off the line uncaught.)"""
        if magnitude < self.min_offset:
            self.reset()
            return False, ''
        width = self.window / self.frames
        if not self._history:
            self._bin_start, self._history = now, [magnitude]
            return False, ''
        if now < self._bin_start + width:
            self._history.append(magnitude)
            return False, ''
        mean = sum(self._history) / len(self._history)
        gap = now - (self._bin_start + width) >= width   # a whole step with no frames
        self._bin_start, self._history = now, [magnitude]
        if gap or self._last is None or mean <= self._last:
            self._last = self._start = mean
            self.consecutive = 0
            return False, ''
        self.consecutive += 1
        self._last = mean
        growth = mean - self._start
        needed = self.growth_needed(range_)
        if self.consecutive >= self.frames and growth >= needed:
            return True, (f'lateral offset grew for {self.consecutive * width:.1f} s, '
                          f'{growth * 1000:.0f} mm in total (limit {needed * 1000:.0f} mm '
                          f'at this range), now {magnitude * 1000:.0f} mm')
        return False, (f'growing for {self.consecutive * width:.1f} s, '
                       f'{growth * 1000:.0f} mm')


def rear_clearance(points: np.ndarray | None, params: dict) -> tuple[float | None, str]:
    """Nearest obstacle behind the rear bumper: (distance from the bumper, detail),
    or None if the corridor (robot width plus margin, `min_height`..`max_height`
    above the floor) is empty. Inside `blind_radius` nothing can be measured; see
    `reverse_allowance`.
    """
    if points is None or len(points) == 0:
        return None, 'no LiDAR data'
    p = np.asarray(points, dtype=float)
    rear = float(params['rear_bumper_x'])
    half = float(params['half_width']) + float(params['side_margin'])
    floor_z = float(params['floor_z'])
    heights = p[:, 2] - floor_z

    behind = (
        (p[:, 0] <= rear)
        & (np.abs(p[:, 1]) <= half)
        & (heights >= float(params['min_height']))
        & (heights <= float(params['max_height']))
    )
    if not behind.any():
        return None, 'nothing in the rear corridor'
    nearest = float(rear - p[behind, 0].max())
    return nearest, (f'nearest return {nearest:.2f} m behind the bumper '
                     f'({int(behind.sum())} in the corridor)')


def reverse_allowance(travelled_forward: float, params: dict) -> float:
    """How far back the robot may drive: the ground it drove forward over (its own
    body was just there) plus `max_unverified` metres nothing has checked (D42).
    """
    return float(travelled_forward) + float(params['max_unverified'])
