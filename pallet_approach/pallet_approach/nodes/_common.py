"""Helpers shared by the node wrappers."""

from __future__ import annotations

import numpy as np

# sensor_msgs/Image encoding -> (dtype, channels). Our depth is 32FC1 and the
# rectified colour image is bgr8 (docs/SYSTEM.md).
ENCODINGS = {
    '32FC1': (np.float32, 1), '16UC1': (np.uint16, 1), 'mono8': (np.uint8, 1),
    'mono16': (np.uint16, 1), 'bgr8': (np.uint8, 3), 'rgb8': (np.uint8, 3),
    'bgra8': (np.uint8, 4), 'rgba8': (np.uint8, 4),
}


def image_to_array(msg) -> np.ndarray:
    """sensor_msgs/Image -> NumPy array, without cv_bridge.

    Four lines of buffer arithmetic instead of a dependency whose Humble build
    is compiled against NumPy 1.x -- which is exactly what breaks inside the
    ZED container (docs/SYSTEM.md). `step` is the row stride in bytes and can
    exceed width*channels*itemsize when rows are padded, so slice per row.
    """
    if msg.encoding not in ENCODINGS:
        raise ValueError(f'unhandled encoding {msg.encoding!r}')
    dtype, channels = ENCODINGS[msg.encoding]
    itemsize = np.dtype(dtype).itemsize
    expected_step = msg.width * channels * itemsize

    expected_bytes = msg.height * msg.step
    buffer = np.frombuffer(bytes(msg.data), dtype=np.uint8)[:expected_bytes]
    if msg.step == expected_step:
        array = buffer.view(dtype).reshape(msg.height, msg.width, channels)
    else:
        rows = buffer.reshape(msg.height, msg.step)[:, :expected_step]
        array = np.ascontiguousarray(rows).view(dtype).reshape(
            msg.height, msg.width, channels)

    if channels == 4:
        array = array[:, :, :3]  # Strip the alpha channel (bgra8 -> bgr8)
    if channels == 1:
        array = array[:, :, 0]
    if msg.encoding == 'rgb8':
        array = array[:, :, ::-1]
    return array


def intrinsics_from_camera_info(msg) -> dict:
    """fx, fy, cx, cy from a CameraInfo.

    K is row-major 3x3, so fx = k[0], fy = k[4], cx = k[2], cy = k[5]. The ZED
    publishes rectified images with D all zero, so there is no undistortion to
    do; if D is ever non-zero, the image is not the rectified one and the
    pipeline's assumptions are wrong -- hence the check in the node.
    """
    k = list(msg.k)
    return {'fx': k[0], 'fy': k[4], 'cx': k[2], 'cy': k[5]}


def nested(node, prefix: str, keys) -> dict:
    """Read a nested parameter block declared as `prefix.key` into a dict."""
    return {key: node.get_parameter(f'{prefix}.{key}').value for key in keys}


def points_to_cloud(points, frame_id: str, stamp):
    """(N, 3) NumPy array -> sensor_msgs/PointCloud2, xyz float32.

    Written out rather than using `point_cloud2.create_cloud_xyz32`, for the
    same reason as `image_to_array` above: the Humble build of that helper is
    compiled against NumPy 1.x, which is exactly what breaks inside the ZED
    container (docs/SYSTEM.md).

    Unordered (height 1), dense, little-endian, 12 bytes a point. The
    corresponding reader is `pallet_approach.pointcloud.unpack_cloud`, and
    `test_pointcloud.py` puts a cloud through both.
    """
    from sensor_msgs.msg import PointCloud2, PointField

    array = np.ascontiguousarray(np.asarray(points, dtype=np.float32).reshape(-1, 3))
    cloud = PointCloud2()
    cloud.header.stamp = stamp
    cloud.header.frame_id = frame_id
    cloud.height = 1
    cloud.width = len(array)
    cloud.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    cloud.is_bigendian = False
    cloud.point_step = 12
    cloud.row_step = 12 * len(array)
    cloud.is_dense = True
    cloud.data = array.tobytes()
    return cloud
