"""PointCloud2 to NumPy, without ros2_numpy or point_cloud2.read_points.

Twenty lines of buffer arithmetic instead of a dependency, for the same reason
`nodes/_common.image_to_array` exists: the Humble builds of those helpers are
compiled against NumPy 1.x, which is exactly what breaks inside the ZED
container, and `point_cloud2.read_points` is a Python generator per point --
65 536 of them per Ouster frame at 10 Hz.

This lives in the package rather than in `tools/` because the offline tools and
the running node must agree about what a point is. `tools/_bagutil.py` had its
own copy; a cloud decoded two ways is a class of bug that only shows up as "it
worked on the bag".
"""

from __future__ import annotations

import numpy as np

__all__ = ['DATATYPES', 'unpack_cloud', 'transform_points']

DATATYPES = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
             5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}
"""sensor_msgs/PointField datatype constants."""


def unpack_cloud(msg, fields=('x', 'y', 'z'), finite_only: bool = True) -> np.ndarray:
    """A PointCloud2 as an (N, len(fields)) float array.

    Reads the requested fields by their declared offsets, so it does not care
    what else the cloud carries or in what order -- the Ouster's has nine
    fields (intensity, t, reflectivity, ring, ambient, range) around the three
    we want, and the layout differs between drivers.

    With `finite_only` (the default) the no-return points are dropped. The
    Ouster marks a beam that hit nothing as exactly (0, 0, 0) rather than NaN,
    and a cloud of 40 000 points at the origin would be read by everything
    downstream as a solid obstacle against the robot's own axle, so both
    non-finite and exactly-zero points go.
    """
    if msg.is_bigendian:
        # Nothing in this system is, and silently byte-swapping the wrong way
        # produces coordinates in the 1e30 range that look like a sensor fault.
        raise ValueError('big-endian PointCloud2 is not handled')
    offsets = {}
    for field in msg.fields:
        if field.datatype not in DATATYPES:
            continue
        offsets[field.name] = (field.offset, DATATYPES[field.datatype])
    missing = [name for name in fields if name not in offsets]
    if missing:
        raise ValueError(
            f'cloud has no field(s) {missing}; it has {sorted(offsets)}')

    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(-1, msg.point_step)
    columns = []
    for name in fields:
        offset, dtype = offsets[name]
        size = np.dtype(dtype).itemsize
        columns.append(raw[:, offset:offset + size].copy().view(dtype).ravel())
    points = np.column_stack(columns).astype(float)
    if not finite_only:
        return points
    keep = np.isfinite(points).all(axis=1) & (np.abs(points).sum(axis=1) > 1e-6)
    return points[keep]


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to an (N, 3) array.

    Written out rather than going through homogeneous coordinates and back,
    which allocates an (N, 4) copy of a 65 000-point cloud twice per frame.
    """
    p = np.asarray(points, dtype=float)
    if p.size == 0:
        return p.reshape(0, 3)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f'expected (N, 3) points, got {p.shape}')
    return p @ np.asarray(matrix, dtype=float)[:3, :3].T + np.asarray(matrix)[:3, 3]
