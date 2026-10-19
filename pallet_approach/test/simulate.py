"""A world the controllers can drive in, for tests.

`synthetic.py` builds point clouds to test perception. This builds the layer
above: a robot that moves when commanded and a pallet that stays where it is,
so a control law can be run end to end at a desk and the result measured rather
than argued about.

What it models, and why each piece is there:

* **Unicycle kinematics.** A skid-steer is not a unicycle, but at 0.10 m/s on a
  flat floor the difference is slip, and slip is what the odometry reports
  anyway. The commands are linear and angular velocity, which is exactly what
  `/cmd_vel` carries.
* **Two rates.** Perception produces an estimate at 5 Hz (D24) and the
  controller publishes at 20 Hz, so the command is held between frames. This
  matters: it is the reason the robot can be caught mid-turn when it reaches
  the standoff, which is what the square-up phase exists for.
* **Measurement noise**, added to the estimate and not to the state. The robot
  really is where it is; the controller only knows what the camera told it. A
  simulation that feeds the controller the truth proves nothing about a
  controller that has to work on measurements.

What it does NOT model: actuator lag, wheel slip, a base that does not reach
its commanded velocity, and a floor that is not flat. So its numbers are a
lower bound on the error, not a prediction -- which is why the gains chosen
from it are the gentler of the two that worked, and why the real figure comes
from lab session B.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from pallet_approach.model import PalletEstimate, Plane, Pocket

FLOOR = Plane(np.array([0.0, 0.0, 1.0]), 0.025)
"""The measured floor: z = -0.025 in base_footprint."""


@dataclass
class Pallet:
    """A pallet standing still, in the world frame.

    Its front face is at `x = 0` facing +x, so the world's x axis is the pocket
    centre line and a robot at (d, 0) facing -x is perfectly aligned at
    face-distance d. Every geometry in the tests is then readable by eye.
    """

    pocket_spacing: float = 0.365
    pocket_width: float = 0.230
    pocket_height: float = 0.100
    center_height: float = 0.050
    yaw: float = 0.0
    """Rotation of the face about the world z axis, radians."""


@dataclass
class Robot:
    """A unicycle, with its pose in the world frame."""

    x: float
    y: float
    yaw: float
    travelled: float = 0.0
    """Path length covered, always positive. What odometry would report."""

    history: list = field(default_factory=list)

    def step(self, linear: float, angular: float, dt: float) -> None:
        self.x += linear * math.cos(self.yaw) * dt
        self.y += linear * math.sin(self.yaw) * dt
        self.yaw += angular * dt
        self.travelled += abs(linear) * dt
        self.history.append((self.x, self.y, self.yaw))

    @property
    def pose(self) -> tuple[float, float, float]:
        return self.x, self.y, self.yaw


def on_line(pallet: Pallet, face_distance: float, offset: float = 0.0,
            yaw_error: float = 0.0) -> Robot:
    """Place a robot at an exact (face_distance, offset, yaw_error).

    Works for a pallet at any angle, which `facing` alone cannot: placing a
    robot at world (d, y) in front of a pallet rotated 15 degrees gives neither
    face-distance d nor offset y, and a test that does it by hand measures the
    arithmetic mistake rather than the controller.

    The three arguments come out of `closing.geometry_of` unchanged, which is
    what makes the tests readable:

    * `face_distance` along the outward normal from the face centre;
    * `offset` along the LINE's own perpendicular, so it is exactly the
      perpendicular distance `Geometry.offset` measures, whatever the robot's
      heading;
    * `yaw_error` positive when the robot must turn left to point square at the
      face, matching `Geometry.heading`.
    """
    normal = _rotation(pallet.yaw) @ np.array([1.0, 0.0])
    square = math.atan2(-normal[1], -normal[0])
    left = _rotation(square) @ np.array([0.0, 1.0])
    position = face_distance * normal + offset * left
    return Robot(x=float(position[0]), y=float(position[1]),
                 yaw=square - yaw_error)


def facing(distance: float, offset: float = 0.0, yaw_error: float = 0.0) -> Robot:
    """`on_line` for a pallet square to the world, which is most tests."""
    return on_line(Pallet(), distance, offset, yaw_error)


def observe(robot: Robot, pallet: Pallet | None = None, noise: float = 0.0,
            yaw_noise: float = 0.005, rng=None) -> PalletEstimate:
    """What perception would report, in `base_footprint`.

    `noise` is the standard deviation of the position error in metres and
    `yaw_noise` of the face normal's angle in radians. The real numbers to put
    here come from `docs/dataset.md`: the face position was within 5-10 mm and
    the pocket spacing within ~13 mm on the first bags.
    """
    pallet = pallet or Pallet()
    rotation = _rotation(-robot.yaw)
    # the face centre and outward normal, in the world
    face_rotation = _rotation(pallet.yaw)
    center_world = np.zeros(2)
    normal_world = face_rotation @ np.array([1.0, 0.0])

    center = rotation @ (center_world - np.array([robot.x, robot.y]))
    normal = rotation @ normal_world
    if noise or yaw_noise:
        rng = rng or np.random.default_rng(0)
        center = center + rng.normal(0.0, noise, 2) if noise else center
        if yaw_noise:
            angle = math.atan2(normal[1], normal[0]) + rng.normal(0.0, yaw_noise)
            normal = np.array([math.cos(angle), math.sin(angle)])

    axis = np.array([-normal[1], normal[0], 0.0])
    center3 = np.array([center[0], center[1], pallet.center_height])
    normal3 = np.array([normal[0], normal[1], 0.0])
    half = pallet.pocket_spacing / 2.0
    return PalletEstimate(
        position=center3,
        normal=normal3,
        yaw=math.atan2(normal[1], normal[0]),
        pockets=(Pocket(center3 - axis * half, pallet.pocket_width,
                        pallet.pocket_height),
                 Pocket(center3 + axis * half, pallet.pocket_width,
                        pallet.pocket_height)),
        face_inlier_count=140,
        floor=FLOOR,
        side='short',
        face_width=0.795,
    )


def _rotation(angle: float) -> np.ndarray:
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return np.array([[cos_a, -sin_a], [sin_a, cos_a]])


def truth(robot: Robot, pallet: Pallet | None = None) -> PalletEstimate:
    """The noiseless observation, for measuring what the run actually achieved.

    The controller never sees this. Scoring a run against the noisy estimate
    the controller happened to get would measure the noise, not the control.
    """
    return observe(robot, pallet, noise=0.0, yaw_noise=0.0)


@dataclass
class Trace:
    """What a simulated run did."""

    arrived: bool
    reason: str
    seconds: float
    range: float
    offset: float
    heading: float
    robot: Robot
    frames: int


def drive(
    robot: Robot,
    step,
    pallet: Pallet | None = None,
    noise: float = 0.0,
    seed: int = 0,
    frame_rate: float = 5.0,
    command_rate: float = 20.0,
    timeout: float = 400.0,
):
    """Run `step` against the world until it stops or `timeout` expires.

    `step(estimate, robot_pose_or_odometry) -> (linear, angular, done, reason)`
    is called at `frame_rate`, and whatever it returns is held at the base
    until the next call -- which is what the real system does, because the
    controller republishes its last command at 20 Hz between estimates.

    Returns a `Trace` measured against the TRUTH, not against the estimates.
    """
    from pallet_approach.closing import geometry_of

    rng = np.random.default_rng(seed)
    dt = 1.0 / command_rate
    frame_dt = 1.0 / frame_rate
    command = (0.0, 0.0)
    next_frame = 0.0
    now = 0.0
    frames = 0
    done, reason = False, 'timed out'

    while now < timeout:
        if now >= next_frame:
            estimate = observe(robot, pallet, noise=noise, rng=rng)
            command_, done, reason = step(estimate, robot)
            frames += 1
            if done:
                break
            command = command_
            next_frame += frame_dt
        robot.step(command[0], command[1], dt)
        now += dt

    final = geometry_of(truth(robot, pallet))
    return Trace(arrived=done, reason=reason, seconds=now, range=final.range,
                 offset=final.offset, heading=final.heading, robot=robot,
                 frames=frames)
