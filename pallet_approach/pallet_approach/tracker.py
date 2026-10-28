"""The pallet tracker: the same pallet, frame after frame? Pure, no ROS.

Each estimate is placed in `odom` (the robot's pose at the frame) and compared
with the track there, where a stationary pallet is stationary. Confidence is
the accepted fraction of the last `window` frames. The pose is averaged in odom
weighted by 1 / range^2 (D59): one frame's normal wobbles +-0.04 rad. The LiDAR
cross-check asks whether the Ouster sees the face the camera fitted (D13).
`last_estimate` stays the newest single frame for single-frame judgements.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from pallet_approach.approach import pocket_center, transform_point, wrap_angle
from pallet_approach.model import PalletEstimate, Plane, Pose2D

__all__ = ['CrossCheck', 'Track', 'Tracker', 'cross_check_face']


@dataclass(frozen=True)
class CrossCheck:
    """What the LiDAR had to say about the camera's face plane."""

    evaluated: bool
    """False when there were not enough LiDAR returns on the face to judge.

    NOT the same as a disagreement, and recorded separately for the same
    reason `go_nogo` records an unconfigured rule as not evaluated: a check
    that silently passes when it cannot run is worse than one that fails.
    """

    agrees: bool
    offset: float
    """Median signed distance from the LiDAR returns to the camera's face
    plane, metres. Positive means the LiDAR sees the face further from the
    robot than the camera does. NaN when not evaluated."""

    points: int
    detail: str

    @property
    def blocking(self) -> bool:
        """Should this stop a confirmation? Only an actual disagreement does."""
        return self.evaluated and not self.agrees


NOT_EVALUATED = CrossCheck(evaluated=False, agrees=False, offset=float('nan'),
                           points=0, detail='no LiDAR data')


def cross_check_face(
    estimate: PalletEstimate,
    lidar_points: np.ndarray | None,
    params: dict,
) -> CrossCheck:
    """Does the LiDAR see a surface where the camera fitted the face?

    LiDAR returns in a window round the camera's face: half the measured face
    width laterally (the pallet occludes what is behind), from `floor_margin`
    above the fitted floor (else bare floor would "agree"), `search_ahead` /
    `search_behind` in depth. The statistic is the 90th percentile of the signed
    offsets: the nearest surface, robust to cargo behind the face; positive means
    nearer than the camera (also catches an obstacle in front). Fewer than
    `min_points` returns -> not evaluated, which never counts as agreement.
    """
    if lidar_points is None or len(lidar_points) == 0:
        return NOT_EVALUATED

    points = np.asarray(lidar_points, dtype=float)
    normal = np.asarray(estimate.normal, dtype=float)
    center = np.asarray(estimate.position, dtype=float)

    # Signed distance to the face plane along the outward normal, which points
    # at the robot: positive is towards the robot, negative is into the pallet.
    offsets = (points - center) @ normal
    along = (points - center) @ _face_axis(normal)

    floor = estimate.floor or Plane(np.array([0.0, 0.0, 1.0]),
                                    -float(params['floor_z']))
    heights = floor.height_of(points)

    half = float(params['half_width'])
    if np.isfinite(estimate.face_width) and estimate.face_width > 0.0:
        half = min(half, estimate.face_width / 2.0 + 0.05)

    window = (
        (offsets <= float(params['search_ahead']))
        & (offsets >= -float(params['search_behind']))
        & (np.abs(along) <= half)
        & (heights >= float(params['floor_margin']))
        & (heights <= float(params['pallet_height']) + float(params['height_margin']))
    )
    count = int(window.sum())
    if count < int(params['min_points']):
        return CrossCheck(
            evaluated=False, agrees=False, offset=float('nan'), points=count,
            detail=f'only {count} LiDAR returns on the face, need '
                   f'{int(params["min_points"])}')

    surface = float(np.percentile(offsets[window], 90))
    limit = float(params['max_offset'])
    agrees = abs(surface) <= limit
    return CrossCheck(
        evaluated=True, agrees=agrees, offset=surface, points=count,
        detail=(f'{count} returns, nearest surface {surface * 1000:+.0f} mm '
                f'from the camera plane, limit {limit * 1000:.0f} mm'))


def jump_limit(range_: float, params: dict) -> float:
    """How far a new estimate may be from the track and still be the same pallet:
    max_jump + max_jump_per_m2 * range^2, like stereo depth error (D93).
    """
    return (float(params['max_jump'])
            + float(params.get('max_jump_per_m2', 0.0)) * range_ ** 2)


def _face_axis(normal: np.ndarray) -> np.ndarray:
    """Horizontal unit vector along the face, perpendicular to its normal."""
    axis = np.array([-normal[1], normal[0], 0.0])
    length = float(np.linalg.norm(axis))
    if length < 1e-9:
        raise ValueError('face normal is vertical; it is not a wall')
    return axis / length


@dataclass
class Track:
    """What is known about the pallet across frames."""

    center: np.ndarray
    """Pocket-centre midpoint in `odom`, shape (3,), the weighted average over
    the accepted frames (D59); z is the newest frame's, for the overlay."""

    yaw: float
    """Face normal heading in `odom`, the same weighted average (of unit
    vectors, so it wraps correctly)."""

    side: str
    frames: int = 0
    """Accepted frames since the track started."""

    first_seen: float = 0.0
    last_seen: float = 0.0
    last_estimate: PalletEstimate | None = None
    """The newest accepted estimate, in its own frame. The coordinator and the
    closing loop use this and not an average; see the module docstring."""

    cross_check: CrossCheck = field(default=NOT_EVALUATED)

    weight: float = 0.0
    """Sum of the (decayed) frame weights behind `center` and `yaw`."""
    heading_sum: np.ndarray = field(default_factory=lambda: np.zeros(2))
    """Weighted sum of the normal's unit vectors, whose angle is `yaw`."""

    def fuse(self, center: np.ndarray, yaw: float, weight: float,
             memory: float) -> None:
        """Fold one accepted frame into the average (D59)."""
        old = self.weight * memory
        total = old + weight
        self.center = np.array([
            (old * self.center[0] + weight * center[0]) / total,
            (old * self.center[1] + weight * center[1]) / total,
            float(center[2]),
        ])
        self.heading_sum = (self.heading_sum * memory
                            + weight * np.array([math.cos(yaw), math.sin(yaw)]))
        self.yaw = math.atan2(self.heading_sum[1], self.heading_sum[0])
        self.weight = total


class Tracker:
    """One pallet, tracked across frames. Reused across runs via `reset`."""

    def __init__(self, params: dict):
        self.params = dict(params)
        self.track: Track | None = None
        self.offered: deque[bool] = deque(
            maxlen=max(2, int(params['window'])))
        self.consecutive = 0
        self.reason = 'no frames yet'

    # --------------------------------------------------------------- state

    def reset(self) -> None:
        """Forget everything. Called when a run starts or ends."""
        self.track = None
        self.offered.clear()
        self.consecutive = 0
        self.reason = 'no frames yet'

    @property
    def confidence(self) -> float:
        """Fraction of the offered frames in the window that were accepted.

        Zero until the window has been offered `min_window_frames` frames: with
        one accepted frame out of one offered, the honest answer is "not yet",
        not "100 %". The confirmation gate below needs consecutive frames
        anyway, so this only has to avoid being misleading in the log.
        """
        if len(self.offered) < int(self.params['min_window_frames']):
            return 0.0
        return sum(self.offered) / len(self.offered)

    @property
    def confirmed(self) -> bool:
        """Enough consecutive agreeing frames, and no sensor disagreement."""
        if self.track is None:
            return False
        if self.track.cross_check.blocking:
            return False
        return (self.consecutive >= int(self.params['confirm_frames'])
                and self.confidence >= float(self.params['min_confidence']))

    # --------------------------------------------------------------- update

    def update(
        self,
        estimate: PalletEstimate | None,
        robot: Pose2D,
        now: float,
        lidar_points: np.ndarray | None = None,
    ) -> Track | None:
        """Offer one frame. `robot` is the robot's pose in `odom` at its stamp.

        A frame with no pallet in it is an offered frame that was not accepted:
        it counts against the confidence, which is what makes the confidence
        drop when the pallet goes out of view rather than freezing at its last
        value.
        """
        if estimate is None:
            self.offered.append(False)
            self.consecutive = 0
            self.reason = 'no pallet in this frame'
            return self.track

        center = pocket_center(estimate.pockets, fallback=estimate.position)
        in_odom = transform_point(center, robot)
        yaw_in_odom = wrap_angle(estimate.yaw + robot.yaw)

        if self.track is not None:
            jump = float(np.linalg.norm(in_odom[:2] - self.track.center[:2]))
            turn = abs(wrap_angle(yaw_in_odom - self.track.yaw))
            limit = jump_limit(float(np.hypot(center[0], center[1])), self.params)
            if jump > limit:
                self.offered.append(False)
                self.consecutive = 0
                self.reason = (f'{jump * 100:.0f} cm from the track, limit '
                               f'{limit * 100:.0f} cm')
                return self.track
            if turn > float(self.params['max_turn']):
                self.offered.append(False)
                self.consecutive = 0
                self.reason = (f'{math.degrees(turn):.0f} deg from the track, '
                               f'limit '
                               f'{math.degrees(float(self.params["max_turn"])):.0f} deg')
                return self.track
            if (self.params.get('require_same_side', True)
                    and estimate.side not in ('unknown', self.track.side)):
                # The pallet did not turn over between two frames 200 ms apart.
                # One of the two classifications is wrong, and which one is not
                # knowable from here, so neither is trusted.
                self.offered.append(False)
                self.consecutive = 0
                self.reason = (f'side changed from {self.track.side} to '
                               f'{estimate.side}')
                return self.track

        check = cross_check_face(estimate, lidar_points, self.params)

        # 1 / range^2: the normal's error moves the line by range x angle.
        distance = max(float(np.hypot(center[0], center[1])),
                       float(self.params.get('fusion_min_range', 0.5)))
        weight = 1.0 / distance ** 2
        memory = float(self.params.get('fusion_memory', 0.95))
        if self.track is None:
            self.track = Track(center=in_odom, yaw=yaw_in_odom,
                               side=estimate.side, first_seen=now)
            self.track.fuse(in_odom, yaw_in_odom, weight, 0.0)
        else:
            self.track.fuse(in_odom, yaw_in_odom, weight, memory)
            if estimate.side != 'unknown':
                self.track.side = estimate.side
        self.track.frames += 1
        self.track.last_seen = now
        self.track.last_estimate = estimate
        self.track.cross_check = check

        self.offered.append(True)
        self.consecutive += 1
        if check.blocking:
            self.reason = f'LiDAR disagrees: {check.detail}'
        else:
            self.reason = (f'{self.consecutive} consecutive frames, '
                           f'confidence {self.confidence:.2f}')
        return self.track

    def status(self) -> str:
        """One word for the message and the logs."""
        if self.track is None:
            return 'SEARCHING'
        if self.track.cross_check.blocking:
            return 'REJECTED'
        if self.confirmed:
            return 'CONFIRMED'
        if self.consecutive == 0:
            return 'LOST'
        return 'CONFIRMING'
