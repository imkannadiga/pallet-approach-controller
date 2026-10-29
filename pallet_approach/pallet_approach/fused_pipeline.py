"""The fused perception, frame by frame (D102 step d), with no ROS in it.

The node calls `on_lidar` per sweep, `on_depth` per processed depth frame and
`status` to publish. Acquire: LiDAR sweeps pooled while the robot stands still
(searched singly while it moves); the first pallet found starts the filter.
Track: each sweep and each depth frame (pockets along the filter's face) update
the filter. Lose: no accepted sweep for `reacquire_after` s drops the track.
The floor is the LiDAR's last fit in the base frame (valid on a flat floor).
"""

from __future__ import annotations

import bisect
import math
import time
from dataclasses import dataclass, field

import numpy as np

from pallet_approach.camera_pockets import CameraPockets, measure
from pallet_approach.fusion import FusionStatus, PalletFilter, to_odom, wrap
from pallet_approach.lidar_pallet import LidarPallet, find_pallet
from pallet_approach.model import Plane


def pose_at(samples, stamp: float, tolerance: float):
    """The odom pose (x, y, yaw) at `stamp`, interpolated between the samples
    either side of it; `samples` is [(time, (x, y, yaw)), ...] in time order.
    Without a sample within `tolerance` on each side, the nearest one within
    `tolerance`, or None."""
    if not samples:
        return None
    times = [t for t, _ in samples]
    i = bisect.bisect_left(times, stamp)
    before = samples[i - 1] if i > 0 else None
    after = samples[i] if i < len(samples) else None
    if (before and after and stamp - before[0] <= tolerance and after[0] - stamp <= tolerance
            and after[0] > before[0]):
        k = (stamp - before[0]) / (after[0] - before[0])
        (x0, y0, a0), (x1, y1, a1) = before[1], after[1]
        return (x0 + k * (x1 - x0), y0 + k * (y1 - y0), wrap(a0 + k * wrap(a1 - a0)))
    near = min((s for s in (before, after) if s), key=lambda s: abs(s[0] - stamp))
    return tuple(near[1]) if abs(near[0] - stamp) <= tolerance else None


def forward_time(points: np.ndarray, times: np.ndarray, half_angle_deg: float = 30.0) -> float:
    """Median measurement time of the points within `half_angle_deg` of straight
    ahead (base frame), where the pallet is during an approach; 0 when none.

    A LiDAR sweep carries one stamp, the start of the sweep; the Ouster measures
    straight ahead about 51 ms later (measured 2026-10-29), so a pose looked up
    at the stamp places a moving robot's sweep wrongly by speed x 51 ms."""
    ahead = (points[:, 0] > 0) & (np.abs(np.degrees(np.arctan2(points[:, 1], points[:, 0])))
                                   <= half_angle_deg)
    return float(np.median(times[ahead])) if ahead.any() else 0.0


def to_base(robot: tuple[float, float, float], point: np.ndarray) -> np.ndarray:
    """An odom (x, y) point into the base frame at robot pose (x, y, yaw)."""
    x, y, yaw = robot
    c, s = math.cos(yaw), math.sin(yaw)
    d = np.asarray(point) - np.array([x, y])
    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])


@dataclass
class Timing:
    """Milliseconds per call, the last of each, for the node to report."""
    acquisition: float = float('nan')
    lidar: float = float('nan')
    camera: float = float('nan')
    calls: dict = field(default_factory=lambda: {'acquisition': 0, 'lidar': 0, 'camera': 0})


class FusedPipeline:
    def __init__(self, params: dict):
        self.params = params
        self.p = params['fused_pipeline']
        self.filter = PalletFilter(params)
        self.reset()

    def reset(self) -> None:
        self.filter = PalletFilter(self.params)
        self.pool: list[np.ndarray] = []
        self.pool_pose: tuple[float, float, float] | None = None
        self.side = ''
        self.floor: Plane | None = None
        self.last_face: LidarPallet | None = None
        self.last_camera: CameraPockets | None = None
        self.in_range = False
        self.timing = Timing()

    @property
    def tracking(self) -> bool:
        return self.filter.x is not None

    # ---------------------------------------------------------------- LiDAR
    def _moved(self, a, b) -> bool:
        return (math.hypot(a[0] - b[0], a[1] - b[1]) > self.p['still_distance']
                or abs(wrap(a[2] - b[2])) > math.radians(self.p['still_angle_deg']))

    def on_lidar(self, points: np.ndarray, robot: tuple[float, float, float], stamp: float) -> None:
        """One sweep, in the base frame at `stamp`, with the robot's odom pose then."""
        if not self.tracking:
            # Pooled standing still (finds a pallet at 3-3.5 m one sweep does
            # not); moving, each sweep alone, so a far start acquires on the way in.
            moving = self.pool_pose is not None and self._moved(robot, self.pool_pose)
            if moving:
                self.pool = []
            self.pool_pose = robot
            self.pool.append(points)
            self.pool = self.pool[-self.p['acquisition_sweeps']:]
            if not moving and len(self.pool) < self.p['acquisition_sweeps']:
                return
            t0 = time.perf_counter()
            face, info = find_pallet(np.concatenate(self.pool), self.params)
            self.timing.acquisition = 1000 * (time.perf_counter() - t0)
            self.timing.calls['acquisition'] += 1
            if info.get('floor_plane') is not None:
                self.floor = info['floor_plane']
            if face is None:
                self.pool = self.pool[1:]                      # slide, keep trying
                return
            self._start(face, robot, stamp)
            return

        t0 = time.perf_counter()
        face, info = find_pallet(points, self.params)
        self.timing.lidar = 1000 * (time.perf_counter() - t0)
        self.timing.calls['lidar'] += 1
        if info.get('floor_plane') is not None:
            self.floor = info['floor_plane']
        if face is not None and face.side == self.side:
            centre = face.pocket_centre if face.pocket_centre is not None else face.centre
            if self.filter.update_lidar(to_odom(robot, centre), wrap(face.yaw + robot[2]), stamp):
                self.last_face = face
                self.in_range = face.in_range
        if stamp - self.filter.last_lidar > self.p['reacquire_after']:
            self.reset()

    def _start(self, face: LidarPallet, robot, stamp: float) -> None:
        centre = face.pocket_centre if face.pocket_centre is not None else face.centre
        self.filter.initialise(to_odom(robot, centre), wrap(face.yaw + robot[2]), stamp)
        self.side, self.last_face, self.in_range = face.side, face, face.in_range
        self.pool = []

    # ---------------------------------------------------------------- camera
    def on_depth(self, depth: np.ndarray, intrinsics: dict, ground_from_camera: np.ndarray,
                 robot: tuple[float, float, float], stamp: float) -> CameraPockets | None:
        """One depth frame; `ground_from_camera` already tilt-corrected, at `stamp`."""
        if not self.tracking or self.floor is None:
            return None
        self.filter.predict(stamp)
        st = self.filter.status(stamp)
        centre = to_base(robot, st.centre)
        yaw = wrap(st.yaw - robot[2])
        side = self.params['pallet']['sides'][self.side]
        result = measure(depth, intrinsics, ground_from_camera, centre, yaw, self.floor, side,
                         self.params['camera_pockets'], self.params['camera_depth']['disparity_bias'])
        self.timing.camera = result.ms
        self.timing.calls['camera'] += 1
        self.last_camera = result
        if result.offset is not None and result.both_open:
            tangent = np.array([math.sin(yaw), -math.cos(yaw)])
            if self.filter.update_camera(to_odom(robot, centre + result.offset * tangent), stamp):
                self.filter.check_camera(result.range_residual, result.yaw_residual)
        else:
            self.filter.note_camera_miss()
        return result

    # ---------------------------------------------------------------- output
    def status(self, now: float) -> FusionStatus:
        return self.filter.status(now)

    def in_base(self, robot: tuple[float, float, float], now: float):
        """(centre (2,), yaw) of the tracked pallet in the base frame now, or None."""
        if not self.tracking:
            return None
        st = self.filter.status(now)
        return to_base(robot, st.centre), wrap(st.yaw - robot[2])
