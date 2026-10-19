"""Closing in on the pallet, and the reverse that makes room to do it.

This replaces the two-leg rotate-drive-rotate approach (D6) with a servo that
runs continuously while the robot closes (D38). The picture, looking down:

        pallet face
    ════════════════════
            |
            C                  C  = midpoint between the pocket centres
            |                  n  = outward face normal, pointing at the robot
            |  the pocket centre line, C + t*n
            |
            | .  theta         aim this far towards the line...
            o  robot, `offset` to one side of the line

The robot drives forward slowly, and steers with two nested loops:

* the **outer loop** looks at the lateral offset and decides which way to aim:
  `theta_desired = -k_cross * offset`, capped at `max_approach_angle` so the
  robot never crabs so far across that the pallet leaves the camera's view;
* the **inner loop** turns the robot until it is aiming that way:
  `omega = k_heading * (theta_desired - theta_actual)`.

Two gains, each meaning one thing. `k_cross` is the reciprocal of a distance:
heading towards the line at `k_cross * offset` radians makes the offset decay
exponentially with a length of `1 / k_cross` metres, which is the number
`required_run_up` is built on. `k_heading` is how hard the robot chases the
aim it has chosen.

This replaced a pure-pursuit law chasing a carrot on the line. Pure pursuit is
the same algebra with the two gains locked together at `k_cross = k_heading /
lookahead`, and locking them is what made it unusable here: a lookahead short
enough to converge in 1.25 m implied an approach angle of nearly 60 deg, so the
robot dog-legged -- turn in place, drive, turn back -- and arrived square-ish
by luck. Separating the gains lets the decay length and the approach angle be
chosen independently, which is what they are.

There is no separate yaw stage and no final rotation: the line's direction *is*
the face normal, so a robot on the line pointing along it is aligned by
construction.

Why a servo rather than aligning once and driving straight: the measurement
improves as the robot closes, so correcting continuously means every correction
is made with better information than the one before. Aligning at 3 m and
driving blind would freeze in the error that the 3 m measurement had.

**The run-up matters more than the gain.** A servo can only take out as much
lateral error as the distance it has to work in. At the 1.75 m standoff the
pockets leave only 65 mm of slack around a 100 mm tine, so the loop has to
arrive inside that, and `required_run_up` is what says how far back the robot
must start. Athreya set the floor at 3.0 m on 2026-10-19 after a simulation
showed 1.75 m and 2.25 m both finishing outside the slack.

**The commit point is a hard stop in the command, not a state.** Once the face
is closer than the virtual tine tips can reach (`carriage_x + tine_length` =
1.6125 m), the tines are inside the pallet and steering would break them
against the blocks. `closing_command` refuses to move at all there, whatever
the state machine thinks. The standoff sits 0.1375 m outside it, so in a
correct run this never fires -- which is the point of having it.

Pure functions. The node owns the clock, the odometry and the publisher.
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
    """Where the closing loop is. Held by the caller, like `controller.Phase`.

    SQUARING exists because of the base's minimum turn rate. The loop can be
    caught mid-correction when it reaches the standoff -- one 0.2 s frame at
    the 0.30 rad/s limit is 3.4 deg -- and arriving 4 deg out of square spends
    most of the 5 deg GO/NO-GO yaw budget on nothing. Turning in place at the
    standoff costs a second and fixes it, and it does not disturb the lateral
    offset at all, because turning about `base_footprint` does not move
    `base_footprint`.
    """

    CLOSING = 'CLOSING'
    SQUARING = 'SQUARING'
    ARRIVED = 'ARRIVED'


@dataclass(frozen=True)
class Geometry:
    """Where the robot stands relative to the pocket centre line.

    All three are measured in the estimate's own frame, which is
    `base_footprint` -- so the robot is at the origin facing +x and these are
    read off one frame with no odometry involved. That is deliberate: the servo
    should act on what it can see now, not on an integrated position.
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
    """Reduce one estimate to the three numbers the servo needs.

    The sign conventions are the fiddly part and the tests state them case by
    case; the shape of the algebra is:

    * `n` is the outward normal, flipped if necessary to point at the robot, so
      the robot sits at face-distance `-C . n` in front of the face;
    * the direction of travel is `-n`, and the robot's own left is
      `(n_y, -n_x)` in the estimate's frame, which is what `offset` is
      measured along.
    """
    center = pocket_center(estimate.pockets, fallback=estimate.position)
    normal = outward_normal(estimate.normal, face_point=center)
    c = np.asarray(center, dtype=float)[:2]
    n = normal[:2]

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
    """Face distance the robot needs before closing in, metres.

    Two terms, and the larger wins:

    * a floor of `min_run_up` (3.0 m, D39), which is where Athreya set it after
      the simulation;
    * the standoff plus however far this servo needs to bring the lateral error
      it currently has down to `residual_target`.

    The second term is why the standoff is "variable with a 3 m minimum" rather
    than just 3 m: a robot that Nav2 left half a metre off the centre line
    needs more run-up than one left on it, and the amount is geometry rather
    than a guess.

    The servo has **two regimes** and the distance is the sum of both, which
    is the correction that matters most here:

    * while the aim is pinned at `max_approach_angle`, the offset falls at a
      constant `sin(max_approach_angle)` metres per metre travelled -- a large
      offset is taken out linearly, not exponentially;
    * once the offset drops below `max_approach_angle / k_cross` the
      proportional term is asking for less than the cap, and
      `d(offset)/d(distance) = -k_cross * offset` gives exponential decay with
      a length of `1 / k_cross` metres.

    Leaving out the first term is what the first version of this function did,
    and a simulation caught it: it claimed 3.04 m was enough to take out
    0.50 m of offset, and the run finished 250 mm off the line with 11 deg of
    yaw. The honest figure is just over 4 m. `test_closing.py` measures both
    regimes against this formula rather than taking it on trust.
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
    """One step of the closing servo. Returns (command, next phase, reason).

    Gates before any motion, in order:

    1. **Past the commit point** -- refuse entirely, in every phase. The tines
       are through the face and the robot has no business moving there.
    2. **ARRIVED** -- zero, every time it is called.
    3. **At the standoff** -- stop driving and square up.
    4. **Not facing the aim** -- if the inner loop's error is more than
       `max_turn_in_place` off, turn without driving. A robot that has to swing
       20 degrees should do it where it is rather than describe that arc at a
       pallet.
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
    """The outer loop: which way to aim, given the lateral offset, radians.

    Positive means "head to the left of the line's direction". Two shaping
    rules on top of the proportional term:

    **A cap.** `max_approach_angle` is the furthest the robot will crab across.
    Without it a large offset asks for an aim the base would reach by turning
    most of the way sideways, and then the pallet leaves the camera's view --
    the servo would blind itself. With it, a large offset is taken out at a
    constant angle until it is small enough for the proportional term to take
    over, which is also the shortest sensible path.

    **A deadzone.** Inside `straighten_offset` the aim is exactly zero, so the
    robot straightens up and arrives square instead of still cutting across.
    This matters because the arrival yaw is otherwise *determined* by the
    arrival offset: aiming `k_cross * offset` at the line means arriving at
    that angle, and at `k_cross = 2.5` a 35 mm residual is already 5 deg --
    the whole GO/NO-GO yaw budget spent on being almost in the right place.
    Below the deadzone the offset is left alone, which costs the drift of a
    nearly-straight robot over the last half metre: a couple of millimetres.
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
    """Drive straight by `distance` on odometry, forward or back.

    Used for the three phases where there is nothing to servo on: backing off
    to make room, the blind part of the insertion, and backing out afterwards.
    `travelled` is the distance covered so far, always positive; `distance` is
    the target, and its sign is the direction.

    Angular is exactly zero, never a small correction. During insertion that is
    D7 -- the tines are inside the pallet and steering would drag them through
    the blocks -- and during the reverse it is because the only thing worth
    steering by is behind the robot where nothing can see it.
    """
    remaining = abs(distance) - travelled
    if remaining <= tolerance:
        return ZERO, True, f'travelled {travelled:.3f} m of {abs(distance):.3f} m'
    direction = math.copysign(1.0, distance)
    command = Command(direction * abs(speed), 0.0)
    return (_finish(command, limits), False,
            f'{travelled:.3f} / {abs(distance):.3f} m, {remaining:.3f} m to go')


def _finish(command: Command, limits: dict) -> Command:
    """Clamp, then handle the two deadbands differently. This is not the same
    as `controller._finish`, and the difference is the whole behaviour of the
    servo near the line.

    **Linear** is raised to `min_linear`, as everywhere else: below it a skid
    steer does not move at all, so a command reduced below it has to be raised
    back or the robot sits still believing it is closing the error.

    **Angular is ZEROED instead of raised.** The base cannot turn more slowly
    than `min_angular`, so for a correction finer than that the choice is to
    turn too much or not to turn, and not turning is the stable one. Raising it
    instead -- which is what `apply_deadband` does, and what this function did
    first -- makes every command either hard over or hard back, and the servo
    becomes bang-bang with no fine control at all: measured in simulation, the
    residual then scaled with the lookahead and ignored the gain completely,
    which is the signature of a saturated loop.

    The cost is a deadzone on the heading: with `k_heading` at 4.0 and
    `min_angular` at 0.10, an aim error under 1.4 deg produces no command. That
    granularity, and not the gain, is what sets the achievable accuracy
    (D40).
    """
    command = clamp_command(command, limits['max_linear'], limits['max_angular'])
    command = apply_deadband(command, limits['min_linear'], 0.0)
    angular = (0.0 if abs(command.angular) < float(limits['min_angular'])
               else command.angular)
    return Command(command.linear, angular)


class DivergenceDetector:
    """Is the error growing instead of shrinking?

    Athreya chose "abort after N consecutive growing frames" on 2026-10-19 over
    a threshold on the error itself, which is the better choice: a large error
    at 3 m is normal and shrinking, while a small error that keeps growing is
    a controller fighting its own sign error or a pallet estimate walking away.
    The first is safe and the second is not, and only the trend tells them
    apart.

    Both conditions have to hold, because consecutive growth alone fires on
    noise once the error is small: five frames of 1 mm steps at the standoff is
    measurement scatter, not divergence.
    """

    def __init__(self, frames: int, min_growth: float):
        self.frames = int(frames)
        self.min_growth = float(min_growth)
        self.reset()

    def reset(self) -> None:
        self._last: float | None = None
        self._start: float | None = None
        self.consecutive = 0

    def update(self, offset: float) -> tuple[bool, str]:
        """Offer one frame's lateral offset. Returns (diverging, detail)."""
        magnitude = abs(float(offset))
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


def rear_clearance(points: np.ndarray | None, params: dict) -> tuple[float | None, str]:
    """Nearest obstacle behind the rear bumper, metres from the bumper.

    Returns (distance, detail), with None when the LiDAR found nothing in the
    corridor -- which is NOT the same as "it is clear". Reversing is the one
    thing this robot does that it cannot see, and the honest statement of what
    this function provides is:

    * beyond `blind_radius` behind the bumper it is a real measurement;
    * inside it there is nothing to measure, and the only reason the reverse is
      acceptable is that the robot's own body occupied that ground seconds ago
      (see `reverse_allowance`, which turns that argument into a limit).

    The corridor is the robot's width plus a margin, from the rear bumper
    backwards, and only returns between `min_height` and `max_height` above the
    floor count -- the floor itself is not an obstacle and neither is the
    ceiling.
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
    """How far back the robot may be driven, metres.

    The ground immediately behind the robot cannot be checked, so the limit is
    an argument rather than a measurement: `travelled_forward` is ground the
    robot's own body was standing on a few seconds ago, and is therefore known
    to have been clear, plus `max_unverified` metres of ground that nothing has
    checked at all.

    Athreya chose our own reverse controller with a LiDAR check over handing
    the reverse to Nav2 on 2026-10-19. The limit this function states is what
    that choice costs: Nav2 would have had the costmap, which remembers what
    the LiDAR saw while the robot drove past. We do not, so the allowance is
    small and deliberate.
    """
    return float(travelled_forward) + float(params['max_unverified'])
