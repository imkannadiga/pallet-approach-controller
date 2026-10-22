"""Masking the robot's own forks out of a depth image: the maths, no ROS.

See depth_self_filter.py for why.
"""

import math

import numpy as np


def optical_to_base(xyz, rpy=(-math.pi / 2, 0.0, -math.pi / 2)):
    """4x4 of the optical frame in base_footprint (the camera's fixed mount)."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rot = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                    [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                    [-sp, cp * sr, cp * cr]])
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = xyz
    return out


def mask_boxes(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float,
               camera_in_base: np.ndarray, boxes) -> np.ndarray:
    """Copy of `depth` with every pixel whose point lies in one of `boxes`
    ((xmin, xmax, ymin, ymax, zmin, zmax) in base_footprint) set to +inf."""
    out = depth.copy()
    if not boxes:
        return out
    ray_x, ray_y = _rays(depth.shape, fx, fy, cx, cy)
    z = depth
    valid = np.isfinite(z)
    x = ray_x * z
    y = ray_y * z
    rot, t = camera_in_base[:3, :3], camera_in_base[:3, 3]
    px = rot[0, 0] * x + rot[0, 1] * y + rot[0, 2] * z + t[0]
    py = rot[1, 0] * x + rot[1, 1] * y + rot[1, 2] * z + t[1]
    pz = rot[2, 0] * x + rot[2, 1] * y + rot[2, 2] * z + t[2]
    inside = np.zeros_like(valid)
    for x0, x1, y0, y1, z0, z1 in boxes:
        inside |= (px >= x0) & (px <= x1) & (py >= y0) & (py <= y1) & (pz >= z0) & (pz <= z1)
    out[valid & inside] = np.inf
    return out


_RAYS = {}


def _rays(shape, fx, fy, cx, cy):
    """Per-pixel (u - cx) / fx and (v - cy) / fy, computed once per camera."""
    key = (shape, fx, fy, cx, cy)
    if key not in _RAYS:
        h, w = shape
        v, u = np.mgrid[0:h, 0:w]
        _RAYS[key] = ((u - cx) / fx, (v - cy) / fy)
    return _RAYS[key]
