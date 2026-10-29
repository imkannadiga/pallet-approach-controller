"""Putting a measurement back where the question came from.

Perception turns pixels into metres. These turn metres back into pixels, so a
result can be drawn on the image it came from -- which is the only check that
shows whether the geometry is RIGHT rather than merely self-consistent. A
pocket measured correctly lands on the hole in the picture; one measured
wrongly does not, and no table of numbers will tell you which you have.

In the package rather than in a tool because three things need it: the offline
replay, the live viewer, and the fork overlay node on the robot. It lived in
`tools/replay.py` until 2026-10-16, which meant importing it also imported that
file's `matplotlib.use('Agg')` -- so the live viewer silently could not open a
window. Geometry does not belong in a module that chooses a plotting backend.
"""

from __future__ import annotations

import numpy as np


def pocket_corners(pocket, face_normal, floor, nominal=None) -> np.ndarray:
    """The four corners of a pocket opening, in `base_footprint`.

    The opening is a rectangle lying in the face plane: `width` across it and
    `height` up it. `along` is the horizontal direction in the face, which is
    perpendicular to both the floor normal and the face normal -- exactly what
    a cross product gives.

    Pass `nominal` as (width, height) to draw the expected opening instead of
    the measured one, which is how you see a measurement error rather than
    reading it.
    """
    up = np.asarray(floor.normal, dtype=float)
    along = np.cross(up, face_normal)
    along /= np.linalg.norm(along)
    half_w, half_h = ((pocket.width, pocket.height) if nominal is None else nominal)
    half_w, half_h = half_w / 2, half_h / 2
    c = np.asarray(pocket.center, dtype=float)
    return np.array([c + half_w * along + half_h * up,
                     c - half_w * along + half_h * up,
                     c - half_w * along - half_h * up,
                     c + half_w * along - half_h * up])


def face_corners(position, face_normal, width, height, floor) -> np.ndarray:
    """The four corners of the pallet's face, in `base_footprint`: `width`
    across, centred on `position`, from the floor plane up to `height`."""
    up = np.asarray(floor.normal, dtype=float)
    along = np.cross(up, face_normal)
    along /= np.linalg.norm(along)
    base = np.asarray(position, dtype=float) - floor.height_of(position) * up
    half = width / 2
    return np.array([base + half * along + height * up, base - half * along + height * up,
                     base - half * along, base + half * along])


def project(points_ground, ground_to_camera, intrinsics):
    """(N, 3) in base_footprint -> (N, 2) pixels, plus a visibility mask.

    The pinhole projection, `u = fx X / Z + cx`, which is the exact inverse of
    what `perception.depth_to_points` undoes. Points at or behind the image
    plane (Z <= 0) have no projection; they are masked, not clamped, because a
    negative Z silently mirrors a point onto the image and looks like a
    geometry bug rather than the division by zero that it is.

    `fork_overlay.project_points` is the live version of this, drawn on the
    camera stream. This one exists so the offline tool does not depend on it.
    """
    p = np.asarray(points_ground, dtype=float)
    cam = p @ ground_to_camera[:3, :3].T + ground_to_camera[:3, 3]
    z = cam[:, 2]
    visible = z > 1e-6
    pixels = np.full((len(p), 2), np.nan)
    pixels[visible, 0] = intrinsics['fx'] * cam[visible, 0] / z[visible] + intrinsics['cx']
    pixels[visible, 1] = intrinsics['fy'] * cam[visible, 1] / z[visible] + intrinsics['cy']
    return pixels, visible
