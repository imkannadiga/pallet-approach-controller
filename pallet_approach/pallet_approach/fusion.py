"""Camera and LiDAR in one estimate of the pallet (D102 step c).

A Kalman filter on the pallet's pose in odom, x = (px, py, yaw): the pocket
pair's centre on the block-front line and the face's outward normal. The
pallet is static, so prediction only grows the covariance (odometry drift).
The LiDAR measures the full pose (good range and angle, rough lateral); the
camera only the lateral, z = t . (px, py). Each update is gated by its
Mahalanobis distance. GO additionally needs (D102 items 6-7, D104 item 3):
no camera-LiDAR disagreement over 20 mm / 3 deg for 5 frames, both sensors
fresh, and both accepted for the last `healthy_frames` frames.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from pallet_approach.approach import wrap_angle as wrap



def to_odom(robot: tuple[float, float, float], point: np.ndarray) -> np.ndarray:
    """A base-frame (x, y) point into odom, given the robot's odom pose (x, y, yaw)."""
    x, y, yaw = robot
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([x + c * point[0] - s * point[1], y + s * point[0] + c * point[1]])


@dataclass(frozen=True)
class FusionStatus:
    initialised: bool
    centre: np.ndarray            # (2,) odom: the pocket pair's centre on the face
    yaw: float                    # odom: the face's outward normal
    sigma_range: float            # m, along the normal
    sigma_lateral: float          # m, along the face
    sigma_yaw: float              # rad
    lidar_age: float              # s since the last accepted LiDAR measurement
    camera_age: float             # s since the last accepted camera measurement
    disagreeing: bool             # D102 item 6
    go_allowed: bool              # D102 items 6 and 7, and a healthy track (D104 item 3)
    reason: str
    rejected: int                 # measurements refused by the gate
    healthy: bool = False         # both sensors accepted for the last N frames, nothing refused

    @property
    def tangent(self) -> np.ndarray:
        return np.array([math.sin(self.yaw), -math.cos(self.yaw)])


class PalletFilter:
    def __init__(self, params: dict):
        self.p = params['fusion']
        self.x: np.ndarray | None = None
        self.P = np.zeros((3, 3))
        self.stamp = 0.0
        self.last_lidar = -math.inf
        self.last_camera = -math.inf
        self.disagree_run = 0
        self.disagreeing = False
        self.rejected = 0
        n = int(self.p['healthy_frames'])
        self.recent = {'lidar': deque(maxlen=n), 'camera': deque(maxlen=n)}

    # ---------------------------------------------------------------- core
    def initialise(self, centre: np.ndarray, yaw: float, stamp: float,
                   sigma_lateral: float | None = None) -> None:
        """Start from an acquisition (LiDAR face, odom), uncertain as one LiDAR reading."""
        p = self.p
        self.x = np.array([centre[0], centre[1], yaw], float)
        self.P = self._lidar_r(yaw)
        if sigma_lateral is not None:
            t = np.array([math.sin(yaw), -math.cos(yaw)])
            n = np.array([math.cos(yaw), math.sin(yaw)])
            self.P[:2, :2] = (p['lidar_sigma_range'] ** 2 * np.outer(n, n)
                              + sigma_lateral ** 2 * np.outer(t, t))
        self.stamp = stamp
        self.last_lidar = stamp
        self.disagree_run, self.disagreeing, self.rejected = 0, False, 0
        for q in self.recent.values():
            q.clear()

    def predict(self, stamp: float) -> None:
        if self.x is None:
            return
        dt = max(0.0, stamp - self.stamp)
        q = np.diag([self.p['process_sigma_xy'] ** 2, self.p['process_sigma_xy'] ** 2,
                     math.radians(self.p['process_sigma_yaw_deg']) ** 2])
        self.P = self.P + q * dt
        self.stamp = max(self.stamp, stamp)

    def _update(self, z: np.ndarray, h: np.ndarray, r: np.ndarray, gate: float,
                angle_row: int | None = None) -> bool:
        innovation = z - h @ self.x
        if angle_row is not None:
            innovation[angle_row] = wrap(innovation[angle_row])
        s = h @ self.P @ h.T + r
        d2 = float(innovation @ np.linalg.solve(s, innovation))
        if d2 > gate:
            self.rejected += 1
            return False
        k = self.P @ h.T @ np.linalg.inv(s)
        self.x = self.x + k @ innovation
        self.x[2] = wrap(self.x[2])
        self.P = (np.eye(3) - k @ h) @ self.P
        self.P = 0.5 * (self.P + self.P.T)
        return True

    def _lidar_r(self, yaw: float) -> np.ndarray:
        p = self.p
        n = np.array([math.cos(yaw), math.sin(yaw)])
        t = np.array([math.sin(yaw), -math.cos(yaw)])
        r = np.zeros((3, 3))
        r[:2, :2] = p['lidar_sigma_range'] ** 2 * np.outer(n, n) + p['lidar_sigma_lateral'] ** 2 * np.outer(t, t)
        r[2, 2] = math.radians(p['lidar_sigma_yaw_deg']) ** 2
        return r

    # ---------------------------------------------------------------- sensors
    def update_lidar(self, centre: np.ndarray, yaw: float, stamp: float) -> bool:
        """A LiDAR face (odom): position and angle."""
        if self.x is None:
            return False
        self.predict(stamp)
        ok = self._update(np.array([centre[0], centre[1], yaw]), np.eye(3), self._lidar_r(self.x[2]),
                          self.p['gate_chi2_3d'], angle_row=2)
        self.recent['lidar'].append(ok)
        if ok:
            self.last_lidar = stamp
        return ok

    def update_camera(self, pocket_centre: np.ndarray, stamp: float) -> bool:
        """The camera's pocket pair centre (odom): its position ALONG the face only."""
        if self.x is None:
            return False
        self.predict(stamp)
        t = np.array([math.sin(self.x[2]), -math.cos(self.x[2])])
        h = np.array([[t[0], t[1], 0.0]])
        ok = self._update(np.array([t @ pocket_centre]), h,
                          np.array([[self.p['camera_sigma_lateral'] ** 2]]), self.p['gate_chi2_1d'])
        self.recent['camera'].append(ok)
        if ok:
            self.last_camera = stamp
        return ok

    def note_camera_miss(self) -> None:
        """A frame where the camera did not confirm both pockets (D104 item 2): it breaks the run."""
        self.recent['camera'].append(False)

    def check_camera(self, range_residual: float, yaw_residual: float) -> bool:
        """D102 item 6: one frame's camera-minus-face residuals; returns whether
        camera and LiDAR now disagree. A frame with no comparison changes nothing."""
        p = self.p
        if not (math.isfinite(range_residual) and math.isfinite(yaw_residual)):
            return self.disagreeing
        off = (abs(range_residual) > p['disagree_range']
               or abs(math.degrees(yaw_residual)) > p['disagree_yaw_deg'])
        self.disagree_run = self.disagree_run + 1 if off else 0
        self.disagreeing = self.disagree_run >= p['disagree_frames']
        return self.disagreeing

    # ---------------------------------------------------------------- output
    def status(self, now: float) -> FusionStatus:
        p = self.p
        if self.x is None:
            return FusionStatus(False, np.zeros(2), 0.0, math.inf, math.inf, math.inf, math.inf, math.inf,
                                False, False, 'no pallet yet', 0)
        yaw = self.x[2]
        n = np.array([math.cos(yaw), math.sin(yaw)])
        t = np.array([math.sin(yaw), -math.cos(yaw)])
        pxy = self.P[:2, :2]
        lidar_age, camera_age = now - self.last_lidar, now - self.last_camera
        reasons = []
        if lidar_age > p['lidar_max_age']:
            reasons.append(f'LiDAR quiet for {lidar_age:.1f} s')
        if camera_age > p['camera_max_age']:
            reasons.append(f'camera quiet for {camera_age:.1f} s')
        if self.disagreeing:
            reasons.append(f'camera and LiDAR disagree ({self.disagree_run} frames)')
        frames = int(p['healthy_frames'])
        healthy = all(len(q) == frames and all(q) for q in self.recent.values())
        if not healthy and not reasons:
            reasons.append(f'not yet {frames} accepted frames from both sensors in a row')
        return FusionStatus(True, self.x[:2].copy(), float(yaw), float(math.sqrt(n @ pxy @ n)),
                            float(math.sqrt(t @ pxy @ t)), float(math.sqrt(self.P[2, 2])),
                            lidar_age, camera_age, self.disagreeing, not reasons,
                            '; '.join(reasons) if reasons else 'both sensors agree', self.rejected,
                            healthy)
