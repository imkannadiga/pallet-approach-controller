"""Mounting corrections for the camera and the LiDAR.

Camera: the ZED is pitched 0.698 deg and rolled -0.516 deg against the URDF
(measured 2026-10-14 from the floor, `tools/floor_plane_check.py`, docs/SYSTEM.md);
`correction_matrix` + `correct_transform` level it. A floor fit only constrains
pitch, roll and height, not x, y or yaw.
LiDAR: calibrated against the camera (D101, `config/lidar_calibration.yaml`) and
moved with the camera's correction by `lidar_follows_camera`. D74's floor-based
`correct_lidar_transform` is the fallback, off.
Kept here, not in the URDF, because that URDF lives in a shared container.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ['measured_floor_normal', 'levelling_rotation', 'correction_matrix',
           'correct_transform', 'apparent_floor_normal', 'lidar_follows_camera',
           'read_lidar_calibration']


def measured_floor_normal(pitch: float, roll: float) -> np.ndarray:
    """The floor normal implied by a measured pitch and roll, in radians.

    Inverse of how `tools/floor_plane_check.py` reports a plane:
    ``pitch = atan2(-nx, nz)`` and ``roll = atan2(ny, nz)``. Parameterising the
    correction by the *measurement* rather than by the correction itself means
    the number in params.yaml is the number the check script prints, and the
    sign cannot be entered backwards.
    """
    normal = np.array([-math.tan(pitch), math.tan(roll), 1.0])
    return normal / np.linalg.norm(normal)


def levelling_rotation(normal: np.ndarray) -> np.ndarray:
    """The 3x3 rotation that takes `normal` onto +z by the shortest path.

    Rodrigues about the axis ``normal x z``. For a tilt of a fraction of a
    degree this is a nearly-identity rotation, but it is written out exactly
    rather than as a small-angle approximation -- the approximation would be
    fine here and wrong the moment someone reuses this for a real misalignment.
    """
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    target = np.array([0.0, 0.0, 1.0])

    axis = np.cross(n, target)
    sin_theta = float(np.linalg.norm(axis))
    cos_theta = float(np.dot(n, target))

    if sin_theta < 1e-12:                       # already level, or upside down
        return np.eye(3) if cos_theta > 0 else -np.eye(3)

    axis = axis / sin_theta
    skew = np.array([[0.0, -axis[2], axis[1]],
                     [axis[2], 0.0, -axis[0]],
                     [-axis[1], axis[0], 0.0]])
    return np.eye(3) + sin_theta * skew + (1.0 - cos_theta) * (skew @ skew)


def correction_matrix(measured_pitch: float, measured_roll: float) -> np.ndarray:
    """3x3 rotation that levels a floor measured with this pitch and roll."""
    return levelling_rotation(measured_floor_normal(measured_pitch, measured_roll))


def correct_transform(ground_from_camera: np.ndarray,
                      correction: np.ndarray) -> np.ndarray:
    """Apply a correction rotation to a 4x4 ``ground <- camera`` transform.

    Only the rotation block is touched. Multiplying the whole transform would
    swing the translation about the ground origin as well -- 8 mm for a 0.87°
    rotation of a 0.55 m lever arm -- which would move the camera's measured
    position to fix its orientation. The position is known to be right, so it
    stays put.
    """
    out = np.array(ground_from_camera, dtype=float, copy=True)
    out[:3, :3] = np.asarray(correction, dtype=float) @ out[:3, :3]
    return out


def apparent_floor_normal(assumed: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Normal a level floor appears to have when the wrong transform is used.

    The tilt is not a property of one transform -- it is the mismatch between
    the transform you use and the one that is physically true. A level floor has
    normal +z in the true ground frame, so the camera sees ``true_R.T @ z``, and
    applying ``assumed_R`` to that gives what you end up measuring. If the two
    agree you get +z back and the floor looks level.
    """
    camera_frame_normal = np.asarray(true, dtype=float)[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    return np.asarray(assumed, dtype=float)[:3, :3] @ camera_frame_normal


def correct_lidar_transform(ground_from_lidar: np.ndarray,
                            measured_pitch: float, measured_roll: float,
                            measured_floor_height: float,
                            floor_z: float) -> np.ndarray:
    """Level a 4x4 ``ground <- lidar`` transform, then put its floor at `floor_z`.

    Parameterised by the measurement, like the camera's: the floor's pitch and
    roll as the LiDAR sees it, and its height at the ground origin AFTER
    levelling (levelling first, because rotating changes that height by a few
    mm). Unlike the camera, the LiDAR's height is wrong as well, so the
    translation is shifted vertically by ``floor_z - measured_floor_height``.
    """
    out = correct_transform(ground_from_lidar, correction_matrix(measured_pitch, measured_roll))
    out[2, 3] += float(floor_z) - float(measured_floor_height)
    return out


def lidar_follows_camera(ground_from_lidar: np.ndarray, ground_from_camera: np.ndarray,
                         camera_correction: np.ndarray) -> np.ndarray:
    """The calibrated LiDAR (D101) moved with the camera's tilt correction.

    tf gives ground <- camera (URDF) . camera <- lidar (measured); the LiDAR is
    fixed to the camera, so it is rotated about the camera, not the ground
    origin (5 mm at 0.34 m for 0.87 deg). Returns
    corrected(ground <- camera) . (ground <- camera)^-1 . (ground <- lidar).
    """
    g_c = np.asarray(ground_from_camera, dtype=float)
    inv = np.eye(4)
    inv[:3, :3] = g_c[:3, :3].T
    inv[:3, 3] = -g_c[:3, :3].T @ g_c[:3, 3]
    return correct_transform(g_c, camera_correction) @ inv @ np.asarray(ground_from_lidar, dtype=float)


def read_lidar_calibration(path) -> tuple[str, str, np.ndarray]:
    """(parent frame, child frame, 4x4 parent <- child) from a lidar-camera-calib file."""
    import yaml
    with open(path) as f:
        doc = yaml.safe_load(f)
    t = doc['transform']
    x, y, z, w = (float(v) for v in t['rotation_xyzw'])
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    out = np.eye(4)
    out[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                   [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                   [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    out[:3, 3] = [float(v) for v in t['translation']]
    return doc['parent_frame'], doc['child_frame'], out
