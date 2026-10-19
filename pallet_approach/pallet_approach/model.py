"""Shared data types for the pallet approach stack.

These are the contract between the perception stage and everything downstream.
`perception.py` produces a `PalletEstimate`; `go_nogo.py`, `approach.py`,
`controller.py` and `fork_overlay.py` consume one. Keeping them here means none
of those modules has to import any of the others.

Everything is SI: metres, radians, seconds. Every vector is a NumPy array of
shape (3,) expressed in `base_footprint` unless a field says otherwise:
x forward, y left, z up.

`base_footprint`, not `base_link`: base_link is the chassis body frame, 0.235 m
above the ground (the wheels mount at base_link level, which lifts the body), so
heights there are all large negative offsets. In base_footprint a z is almost a
height above the floor, which is how the pallet geometry, the pocket heights and
the fork heights are all quoted.

*Almost*: base_footprint is not exactly on the floor. Measured 2026-10-14, the
real floor is at **z = -0.025** here -- the URDF's base_footprint joint is 25 mm
out, while its wheel geometry is right to 0.2 mm. Nothing in the pipeline
assumes z = 0 is the floor: every height is taken relative to `floor_height`
below, which perception fits per frame.

The two frames differ by a pure translation in z, so x, y and yaw are identical
in both -- which is why planar driving targets computed here compose directly
with the base_link pose that /odom publishes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Plane:
    """A plane, as ``normal . p + offset = 0``.

    The floor is carried around as a plane rather than as a single height, and
    that is a deliberate design choice worth being able to defend.

    Any error in the camera-to-robot transform rotates the *whole observed
    scene* -- floor and pallet together. A rotation is an isometry, so the
    perpendicular distance from an observed point to an observed plane does not
    change. Measure pocket heights against this plane and they come out exactly
    right even with an uncalibrated, drifting or freshly-knocked camera mount.
    Measure them as a z-coordinate instead and the error leaks straight in,
    because z is a direction in a frame the error has rotated away from.

    Measured on the robot: the camera mount was 0.868 deg out, and a z-based
    pocket height carried 4.5 mm of that while the perpendicular measurement
    carried exactly none. See docs/SYSTEM.md.
    """

    normal: np.ndarray
    """Unit normal, shape (3,). For the floor it points up."""

    offset: float
    """``normal . p + offset = 0`` for every point p on the plane."""

    def __post_init__(self) -> None:
        if np.shape(self.normal) != (3,):
            raise ValueError(f'normal must have shape (3,), got {np.shape(self.normal)}')

    def height_of(self, points: np.ndarray) -> np.ndarray | float:
        """Signed perpendicular distance from `points` to the plane.

        Positive on the side the normal points to, so for the floor this is
        height above the floor. Accepts a single (3,) point or an (N, 3) array.
        This is the isometry-invariant measurement; use it instead of
        subtracting z values.
        """
        p = np.asarray(points, dtype=float)
        return p @ self.normal + self.offset

    @property
    def height_on_axis(self) -> float:
        """Where the plane crosses the frame's vertical axis (x = y = 0).

        Only for logging and for the overlay's fallback -- an extrapolation,
        and therefore sensitive to tilt. Never measure pallet geometry with it.
        """
        return -self.offset / self.normal[2]


@dataclass(frozen=True)
class Pocket:
    """One fork pocket, as measured on the pallet's front face."""

    center: np.ndarray
    """Centre of the pocket opening, shape (3,)."""

    width: float
    """Horizontal opening, metres. EPAL 1 nominal on the 800 mm face: 0.2275."""

    height: float
    """Vertical opening, metres. EPAL 1 nominal: 0.100."""

    def __post_init__(self) -> None:
        if np.shape(self.center) != (3,):
            raise ValueError(f'pocket center must have shape (3,), got {np.shape(self.center)}')


@dataclass(frozen=True)
class PalletEstimate:
    """One frame's estimate of the pallet's front face and its pockets.

    `position` and `normal` describe the plane we would insert forks into;
    `yaw` is the same information as a single angle, which is what the
    controller and the GO/NO-GO rules actually use.
    """

    position: np.ndarray
    """Centre of the fitted front face, shape (3,)."""

    normal: np.ndarray
    """Unit outward normal of the front face, shape (3,); points at the robot."""

    yaw: float
    """Heading of `normal` about base z: ``atan2(normal[1], normal[0])``.

    A pallet squarely facing the robot has ``normal ~ (-1, 0, 0)``, so
    ``yaw ~ pi``. The angle the GO/NO-GO rules care about is the *misalignment*
    between the robot's heading and this normal, not this value on its own.
    """

    pockets: tuple[Pocket, ...] = ()
    """Pockets found in the face, left to right as seen by the robot."""

    face_inlier_count: int = 0
    """Points supporting the face fit. Few points means a weak estimate."""

    floor: Plane | None = None
    """The fitted floor plane, in `base_footprint`. None if none was fitted.

    Its `height_on_axis` should come out near **-0.025**: base_footprint sits
    25 mm above the real floor (measured 2026-10-14). Use `floor.height_of(p)`
    for every pallet height -- see `Plane` for why that is invariant to the
    camera extrinsic and a z-subtraction is not.

    None happens legitimately at the standoff, where only a 4 mm strip of floor
    is still in frame. Fall back to a plane built from the `floor.expected_z`
    parameter there, not to zero.
    """

    side: str = 'unknown'
    """Which face of the pallet this is: ``'short'``, ``'long'``,
    ``'ambiguous'``, or ``'unknown'`` when classification was not run.

    The two faces are different structures and need different pocket nominals,
    fork spacing and insertion stop (D19/D21), so this is not a label -- it
    selects the geometry everything downstream measures against.

    ``'ambiguous'`` is a real answer, not a failure: the estimate is still
    published, with `pockets` empty because there is no defensible set of
    nominals to measure them against. The coordinator owns the decision to stop
    (D22).
    """

    side_detail: dict = field(default_factory=dict)
    """Evidence behind `side`: what each signal picked and how far each side
    was from each measurement. Carried so a refusal can be explained after the
    run rather than being a bare "no"."""

    face_width: float = float('nan')
    """Visible width of the face, measured along the face itself.

    A LOWER bound: occlusion, the ROI crop and the image edge all shorten it and
    nothing lengthens it. That is why it confirms the side rather than deciding
    it.
    """

    stamp: float = 0.0
    """Timestamp of the depth image this came from, seconds."""

    frame_id: str = 'base_footprint'

    @property
    def floor_height(self) -> float:
        """Convenience for logs and messages. Not for measuring geometry."""
        return float('nan') if self.floor is None else self.floor.height_on_axis

    def __post_init__(self) -> None:
        for name in ('position', 'normal'):
            value = getattr(self, name)
            if np.shape(value) != (3,):
                raise ValueError(f'{name} must have shape (3,), got {np.shape(value)}')


@dataclass(frozen=True)
class Decision:
    """The outcome of the GO/NO-GO rules."""

    go: bool
    reason: str
    """Why. For a NO-GO this is the message the operator sees, so make it
    specific: "yaw 7.2 deg beyond 5.0 deg limit", not "misaligned"."""

    checks: dict[str, bool] = field(default_factory=dict)
    """Rule name -> passed. Useful in logs and in the evaluation tables."""


@dataclass(frozen=True)
class Pose2D:
    """A planar pose. Used for driving targets, which live in the `odom` frame."""

    x: float
    y: float
    yaw: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.yaw], dtype=float)
