"""Standoff geometry: where the robot must stand for fork insertion, and how
far off it is. Pure functions, metres and radians.

        pallet face
    ════════════════════
            │ n  (outward face normal, points at the robot)
            C                    C = midpoint between the pocket centres
            │
            ●  S = C + d·n       standoff: where base_footprint stops

Only x and y are used. `base_footprint` and `base_link` differ by a pure z
translation, so a planar pose computed here composes directly with the
base_link pose /odom publishes.
"""

from __future__ import annotations

import math

import numpy as np

from pallet_approach.model import Pocket, Pose2D

__all__ = [
    'wrap_angle',
    'unit',
    'outward_normal',
    'pocket_center',
    'standoff_pose',
    'lateral_offset',
    'range_to_face',
    'yaw_misalignment',
    'transform_pose',
    'transform_point',
    'distance_between',
]


def wrap_angle(angle: float) -> float:
    """Wrap an angle into [-pi, pi] (atan2's range; both ends can come out)."""
    return math.atan2(math.sin(angle), math.cos(angle))


def unit(vector: np.ndarray) -> np.ndarray:
    """Return `vector` scaled to length 1, or raise if it has no length."""
    v = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(v))
    if norm < 1e-9:
        raise ValueError('cannot normalise a zero-length vector')
    return v / norm


def outward_normal(
    normal: np.ndarray,
    face_point: np.ndarray | None = None,
    toward: np.ndarray | None = None,
) -> np.ndarray:
    """Flatten `normal` into the ground plane, pointing from the face at the robot.

    A plane fit's normal has an arbitrary sign. Given `face_point`, the sign is
    chosen to point towards `toward` (default: the origin, i.e. the robot in
    `base_footprint`); without it the caller's sign is kept.
    """
    n = np.asarray(normal, dtype=float)
    planar = unit(np.array([n[0], n[1], 0.0])) if np.linalg.norm(n[:2]) > 1e-9 else None
    if planar is None:
        raise ValueError('face normal is vertical; the face fit is not a wall')
    if face_point is not None:
        observer = np.zeros(3) if toward is None else np.asarray(toward, dtype=float)
        to_observer = np.asarray(observer, dtype=float)[:2] - np.asarray(face_point, dtype=float)[:2]
        if float(np.dot(planar[:2], to_observer)) < 0.0:
            planar = -planar
    return planar


def pocket_center(
    pockets: tuple[Pocket, ...] | list[Pocket],
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    """Midpoint C between the pocket centres, shape (3,).

    This, not the centre of the fitted face, is what the robot aims at: the
    forks have to line up with the pockets, and a pallet with a load pushed to
    one side still has its pockets where they belong. With fewer than two
    pockets there is nothing to average, so `fallback` (normally the face
    centre) is returned and the GO/NO-GO rules will refuse the run anyway.
    """
    if len(pockets) >= 2:
        centers = np.asarray([p.center for p in pockets], dtype=float)
        return centers.mean(axis=0)
    if len(pockets) == 1:
        return np.asarray(pockets[0].center, dtype=float)
    if fallback is None:
        raise ValueError('no pockets and no fallback point')
    return np.asarray(fallback, dtype=float)


def _pose_on_normal(center: np.ndarray, normal: np.ndarray, distance: float) -> Pose2D:
    """Point at `center + distance * normal`, facing back along -normal."""
    c = np.asarray(center, dtype=float)
    n = outward_normal(normal, face_point=c)
    point = c[:2] + distance * n[:2]
    # facing the pallet means heading along -n
    heading = math.atan2(-n[1], -n[0])
    return Pose2D(x=float(point[0]), y=float(point[1]), yaw=wrap_angle(heading))


def standoff_pose(center: np.ndarray, normal: np.ndarray, distance: float) -> Pose2D:
    """S = C + d·n, facing the pallet.

    `distance` is measured from the pocket centre line to `base_footprint`, and
    has to clear the camera's blind region as well as the pallet: the pocket's
    bottom lip drops out of the image before the robot gets close. See
    `coordinator.approach.standoff_distance` in params.yaml for the derivation.
    """
    if distance <= 0.0:
        raise ValueError(f'standoff distance must be positive, got {distance}')
    return _pose_on_normal(center, normal, distance)


def lateral_offset(robot: Pose2D, center: np.ndarray, normal: np.ndarray) -> float:
    """Signed distance from the robot to the pocket centre line, metres.

    Positive means the robot is to the left of the line as seen by someone
    standing at the pallet looking out along n. This is the quantity the
    `max_lateral_offset` rule tests: get it wrong and the forks hit the blocks
    instead of entering the pockets.
    """
    c = np.asarray(center, dtype=float)
    n = outward_normal(normal, face_point=c)
    # the centre line runs along n through C, so the offset is the component of
    # (robot - C) along the in-plane perpendicular to n
    perpendicular = np.array([-n[1], n[0]])
    delta = np.array([robot.x - c[0], robot.y - c[1]])
    return float(np.dot(delta, perpendicular))


def range_to_face(robot: Pose2D, center: np.ndarray, normal: np.ndarray) -> float:
    """Distance from the robot to the face plane along n, metres.

    Positive in front of the face. Compare this with
    `coordinator.approach.standoff_distance` to get the action feedback's
    `distance_to_standoff`.
    """
    c = np.asarray(center, dtype=float)
    n = outward_normal(normal, face_point=c)
    delta = np.array([robot.x - c[0], robot.y - c[1]])
    return float(np.dot(delta, n[:2]))


def yaw_misalignment(robot_yaw: float, normal: np.ndarray) -> float:
    """How far the robot's heading is from square-on to the face, radians.

    Zero when the robot points straight at the pallet. Signed and wrapped, so
    it can be fed to a controller directly. This is the `max_yaw_error` rule.
    """
    n = outward_normal(normal)
    desired = math.atan2(-n[1], -n[0])
    return wrap_angle(desired - robot_yaw)


def transform_pose(pose: Pose2D, frame_in_target: Pose2D) -> Pose2D:
    """Express `pose` (given in frame A) in frame B.

    `frame_in_target` is the pose of A in B -- for example `base_link` in
    `odom`, which is exactly what `/odom` publishes. Composing 2D poses is a
    rotation followed by a translation; writing it out once here keeps the
    nodes free of trigonometry.
    """
    cos_a, sin_a = math.cos(frame_in_target.yaw), math.sin(frame_in_target.yaw)
    return Pose2D(
        x=frame_in_target.x + cos_a * pose.x - sin_a * pose.y,
        y=frame_in_target.y + sin_a * pose.x + cos_a * pose.y,
        yaw=wrap_angle(frame_in_target.yaw + pose.yaw),
    )


def transform_point(point: np.ndarray, frame_in_target: Pose2D) -> np.ndarray:
    """Express a (3,) point given in frame A in frame B. z passes through: the
    frames this is used between differ by a planar motion.
    """
    p = np.asarray(point, dtype=float)
    if p.shape != (3,):
        raise ValueError(f'expected a (3,) point, got {p.shape}')
    cos_a, sin_a = math.cos(frame_in_target.yaw), math.sin(frame_in_target.yaw)
    return np.array([
        frame_in_target.x + cos_a * p[0] - sin_a * p[1],
        frame_in_target.y + sin_a * p[0] + cos_a * p[1],
        p[2],
    ])


def distance_between(a: Pose2D, b: Pose2D) -> float:
    """Planar distance between two poses, metres (heading ignored)."""
    return float(math.hypot(a.x - b.x, a.y - b.y))


def points_then_to_now(points: np.ndarray, then: Pose2D, now: Pose2D) -> np.ndarray:
    """Points seen from the robot at pose `then` (odom), as seen from `now`.

    Planar: x and y move with the robot's change of pose, z does not. Used to
    bring a LiDAR cloud up to date with odometry before comparing the two (D90).
    """
    c0, s0 = math.cos(then.yaw), math.sin(then.yaw)
    wx = then.x + c0 * points[:, 0] - s0 * points[:, 1]
    wy = then.y + s0 * points[:, 0] + c0 * points[:, 1]
    c1, s1 = math.cos(now.yaw), math.sin(now.yaw)
    dx, dy = wx - now.x, wy - now.y
    out = np.array(points, dtype=float, copy=True)
    out[:, 0] = c1 * dx + s1 * dy
    out[:, 1] = -s1 * dx + c1 * dy
    return out

