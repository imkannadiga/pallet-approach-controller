"""PointCloud2 to NumPy, without ros2_numpy or point_cloud2.read_points (built
against NumPy 1.x, and a per-point generator). Shared by the nodes and the
offline tools so both decode a cloud the same way.
"""

from __future__ import annotations

import numpy as np

__all__ = ['DATATYPES', 'unpack_cloud', 'transform_points']

DATATYPES = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
             5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}
"""sensor_msgs/PointField datatype constants."""


def unpack_cloud(msg, fields=('x', 'y', 'z'), finite_only: bool = True) -> np.ndarray:
    """A PointCloud2 as an (N, len(fields)) float array, read by declared offsets.

    With `finite_only` (default), non-finite and exactly-(0, 0, 0) points are
    dropped: the Ouster marks a no-return beam as the origin, which would
    otherwise look like an obstacle at the axle.
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
