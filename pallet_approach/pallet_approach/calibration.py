"""Extrinsic correction for the camera mounting.

The URDF says the ZED's optical frame is level with `base_footprint`. Measured
2026-10-14 against the floor plane, it is not: it is pitched down 0.698° and
rolled -0.516°, 0.868° in total. The LiDAR on the same `bottom_box_link` comes
out level, so the sensor box is fine and the camera on it is not. See
docs/SYSTEM.md for the measurement and `tools/floor_plane_check.py` to repeat it.

Left uncorrected that is 22 mm of vertical error at the standoff and 30 mm at
2 m, against a 30 mm pocket-height tolerance -- so it has to be corrected before
pocket geometry means anything.

The correction lives here rather than in the robot description, because that
URDF is in a container shared with other projects and cannot easily be rebuilt.
Keeping it here also makes it an explicit, re-measurable parameter instead of a
number buried in a xacro.

**What a ground plane can and cannot calibrate.** A plane has three parameters:
two for its normal direction and one for its distance. So observing a known
plane pins down exactly three of the six extrinsic degrees of freedom -- the two
tilt angles and the height. It is blind to the other three: you cannot tell how
a sensor has slid along the plane (x, y) or spun about its normal (yaw), because
none of those change what the plane looks like. So this module corrects pitch
and roll only. The height came out right already (-0.0246 m measured against
-0.025 m from the tape measure, agreeing to 1.7 mm). Yaw, x and y need a
different reference -- an object at a taped position, which is the other half of
the Day 6 task.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ['measured_floor_normal', 'levelling_rotation', 'correction_matrix',
           'correct_transform', 'apparent_floor_normal']


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
