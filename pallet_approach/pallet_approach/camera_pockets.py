"""The pockets, measured by the camera along the LiDAR's face line (D102 step b).

No image search: points on the LiDAR's face plane at pocket height, 5 mm apart,
are projected into the depth image. Each reads wood (at the face's depth), hole
(clearly deeper) or unknown; the side's block/pocket layout slid along that
profile gives the pockets' position, refined by parabolic interpolation. The
camera's range and angle against the face are only a cross-check (D102 item 2),
after removing the ZED's disparity offset (D103).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from pallet_approach.model import Plane


@dataclass(frozen=True)
class CameraPockets:
    offset: float | None        # pocket pair centre along the face, from the given face centre, m
    score: float                # template match, normalised: 1 = perfect, 0 = nothing
    contrast: float             # mean profile at blocks minus at pockets (2 = ideal)
    both_open: bool             # each pocket reads mostly hole
    range_residual: float       # median (camera depth - face depth) on wood, m; + = camera reads farther
    yaw_residual: float         # camera's face angle minus the given face's, rad, from the wood's depth trend
    samples: int                # samples with depth
    ms: float


def _template(s: np.ndarray, side: dict, beyond: float) -> np.ndarray:
    w, pw, sp = side['face_width'], side['pocket_width'], side['pocket_spacing']
    t = np.full(s.shape, beyond)
    t[np.abs(s) <= w / 2] = 1.0
    t[np.abs(np.abs(s) - sp / 2) <= pw / 2] = -1.0
    return t


def corrected_depth(z: np.ndarray, disparity_bias: float) -> np.ndarray:
    """The ZED's depth with its constant disparity offset removed (D103).

    1/z_measured = 1/z_true + beta, so z_true = z_measured / (1 - beta z_measured).
    """
    return z / (1.0 - disparity_bias * z)


def measure(depth: np.ndarray, intrinsics: dict, ground_from_camera: np.ndarray,
            centre: np.ndarray, yaw: float, floor: Plane, side: dict, p: dict,
            disparity_bias: float = 0.0) -> CameraPockets:
    """Pocket position along a known face (`centre`, outward-normal `yaw`, ground
    frame, as lidar_pallet reports it), from one depth image."""
    t0 = time.perf_counter()
    tangent = np.array([math.sin(yaw), -math.cos(yaw)])
    half = side['face_width'] / 2 + p['margin']
    s = np.arange(-half, half + 1e-9, p['step'])
    heights = side['band_center'] + np.linspace(-1, 1, p['rows']) * side['band_half_height']
    xy = centre[None, :] + s[:, None] * tangent[None, :]                      # (S, 2)
    # floor height at each xy, plus each band height (z ~ the floor normal)
    n, o = floor.normal, floor.offset
    z_floor = -(n[0] * xy[:, 0] + n[1] * xy[:, 1] + o) / n[2]
    pts = np.empty((len(heights), len(s), 3))
    pts[:, :, :2] = xy[None]
    pts[:, :, 2] = z_floor[None] + heights[:, None]
    r, t = ground_from_camera[:3, :3], ground_from_camera[:3, 3]
    cam = (pts.reshape(-1, 3) - t) @ r                                         # R^T (P - t)
    z = cam[:, 2]
    u = np.round(intrinsics['fx'] * cam[:, 0] / z + intrinsics['cx']).astype(int)
    v = np.round(intrinsics['fy'] * cam[:, 1] / z + intrinsics['cy']).astype(int)
    h, w = depth.shape
    inside = (z > 0.3) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    measured = np.full(z.shape, np.nan)
    measured[inside] = corrected_depth(depth[v[inside], u[inside]].astype(float), disparity_bias)
    diff = measured - z                                                        # + = deeper than the face
    valid = np.isfinite(diff)
    wood = valid & (np.abs(diff) <= p['wood_tolerance'])
    hole = valid & (diff >= p['hole_depth'])
    value = np.where(wood, 1.0, np.where(hole, -1.0, 0.0)).reshape(len(heights), len(s))
    known = valid.reshape(len(heights), len(s)).sum(axis=0)
    profile = np.where(known > 0, value.sum(axis=0) / np.maximum(known, 1), 0.0)

    shifts = np.arange(-p['search'], p['search'] + 1e-9, p['step'])
    scores = np.array([np.mean(profile * _template(s - d, side, p['beyond'])) for d in shifts])
    best = int(np.argmax(scores))
    offset = float(shifts[best])
    if 0 < best < len(scores) - 1:                                             # parabolic refinement
        a, b, c = scores[best - 1], scores[best], scores[best + 1]
        denom = a - 2 * b + c
        if denom < 0:
            offset += 0.5 * p['step'] * (a - c) / denom
    ideal = np.mean(_template(s, side, p['beyond']) ** 2)
    score = float(scores[best] / ideal) if ideal > 0 else 0.0
    x = s - offset
    pocket = np.abs(np.abs(x) - side['pocket_spacing'] / 2) <= side['pocket_width'] / 2 - p['step']
    block = (np.abs(x) <= side['face_width'] / 2 - p['step']) & ~(
        np.abs(np.abs(x) - side['pocket_spacing'] / 2) <= side['pocket_width'] / 2 + p['step'])
    contrast = float(profile[block].mean() - profile[pocket].mean()) if pocket.any() and block.any() else 0.0
    both = all(profile[pocket & (np.sign(x) == sgn)].mean() <= p['open_below']
               for sgn in (-1.0, 1.0) if (pocket & (np.sign(x) == sgn)).any())
    residual = float(np.median(diff[wood])) if wood.any() else float('nan')
    # The camera's face angle relative to the given one: tan = slope of the
    # wood's depth difference along the face.
    s_all = np.broadcast_to(s, (len(heights), len(s))).reshape(-1)
    yaw_residual = float('nan')
    if wood.sum() >= 10 and np.ptp(s_all[wood]) > side['face_width'] / 2:
        slope = np.polyfit(s_all[wood], diff[wood], 1)[0]
        yaw_residual = float(math.atan(slope))
    found = score >= p['min_score'] and contrast >= p['min_contrast']
    return CameraPockets(offset if found else None, score, contrast, bool(both and found), residual,
                         yaw_residual, int(valid.sum()), 1000 * (time.perf_counter() - t0))
