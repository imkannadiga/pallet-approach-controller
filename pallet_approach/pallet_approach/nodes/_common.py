"""Helpers shared by the node wrappers."""

from __future__ import annotations

import array as pyarray

import functools
import time
import traceback

import numpy as np

# sensor_msgs/Image encoding -> (dtype, channels). Our depth is 32FC1 and the
# rectified colour image is bgr8 (docs/SYSTEM.md).
ENCODINGS = {
    '32FC1': (np.float32, 1), '16UC1': (np.uint16, 1), 'mono8': (np.uint8, 1),
    'mono16': (np.uint16, 1), 'bgr8': (np.uint8, 3), 'rgb8': (np.uint8, 3),
    'bgra8': (np.uint8, 4), 'rgba8': (np.uint8, 4),
}


def image_to_array(msg) -> np.ndarray:
    """sensor_msgs/Image -> NumPy array, without cv_bridge (its Humble build
    needs NumPy 1.x, docs/SYSTEM.md). Rows may be padded beyond width*channels.
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
    """fx, fy, cx, cy from a CameraInfo (K row-major). The ZED's rectified images have D = 0."""
    k = list(msg.k)
    return {'fx': k[0], 'fy': k[4], 'cx': k[2], 'cy': k[5]}


def nested(node, prefix: str, keys) -> dict:
    """Read a nested parameter block declared as `prefix.key` into a dict."""
    return {key: node.get_parameter(f'{prefix}.{key}').value for key in keys}


def points_to_cloud(points, frame_id: str, stamp):
    """(N, 3) array -> unordered xyz float32 PointCloud2, written out for the same
    NumPy reason as `image_to_array`. Read back by `pointcloud.unpack_cloud`.
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
    # array.array, not bytes: rclpy checks a bytes value element by element in
    # Python (58.5 ms for a 640x360 image, measured 2026-10-22), an array.array
    # of typecode 'B' is copied in C (1.3 ms).
    cloud.data = pyarray.array('B', array.tobytes())
    return cloud


def logged_transition(callback):
    """Log a lifecycle callback's exception, then re-raise.

    rclpy fails the transition silently otherwise (a KeyError in the controller's
    on_configure, 2026-10-20, showed only as "could not configure").
    """
    @functools.wraps(callback)
    def wrapper(self, state):
        try:
            return callback(self, state)
        except Exception:
            self.get_logger().error(
                f'{callback.__name__} raised:\n{traceback.format_exc()}')
            raise
    return wrapper


LIDAR_PARAMS = {
    # D74's floor-based correction: the fallback, off since D101.
    'frames.lidar_correction.enabled': False,
    'frames.lidar_correction.measured_floor_pitch': 0.0,
    'frames.lidar_correction.measured_floor_roll': 0.0,
    'frames.lidar_correction.measured_floor_height': -0.025,
    # D101: the calibrated LiDAR, which follows the camera's tilt correction.
    'frames.lidar_calibration.enabled': False,
    'frames.camera_optical': 'zed2_left_camera_frame_optical',
    'frames.extrinsic_correction.enabled': False,
    'frames.extrinsic_correction.measured_floor_pitch': 0.0,
    'frames.extrinsic_correction.measured_floor_roll': 0.0,
}


def declare_lidar_params(node) -> None:
    """The parameters `lidar_in_ground` reads; one list for every node that uses the LiDAR."""
    for name, default in LIDAR_PARAMS.items():
        if not node.has_parameter(name):
            node.declare_parameter(name, default)


def lidar_in_ground(node, buffer, cloud_frame: str, stamp, timeout) -> np.ndarray:
    """4x4 ``ground <- cloud_frame`` with the LiDAR correction in force: D101
    (calibrated, moved with the camera's tilt correction), else D74, else tf as is.
    Shared by the tracker and the controller so both put the LiDAR in one place.
    """
    from pallet_approach.calibration import (correct_lidar_transform, correction_matrix,
                                             lidar_follows_camera)
    from pallet_approach.ros_conversions import transform_to_matrix
    get = lambda name: node.get_parameter(name).value   # noqa: E731
    ground = get('frames.ground')
    matrix = transform_to_matrix(lookup_waiting(buffer, ground, cloud_frame, stamp, timeout))
    if get('frames.lidar_calibration.enabled'):
        camera = transform_to_matrix(lookup_waiting(
            buffer, ground, get('frames.camera_optical'), stamp, timeout))
        correction = np.eye(3)
        if get('frames.extrinsic_correction.enabled'):
            correction = correction_matrix(get('frames.extrinsic_correction.measured_floor_pitch'),
                                           get('frames.extrinsic_correction.measured_floor_roll'))
        return lidar_follows_camera(matrix, camera, correction)
    if get('frames.lidar_correction.enabled'):
        return correct_lidar_transform(matrix, get('frames.lidar_correction.measured_floor_pitch'),
                                       get('frames.lidar_correction.measured_floor_roll'),
                                       get('frames.lidar_correction.measured_floor_height'),
                                       get('floor_z'))
    return matrix


def lookup_waiting(buffer, target: str, source: str, stamp, timeout=None):
    """`buffer.lookup_transform` that waits by sleeping, not spinning.

    rclpy's own timeout busy-waits (81 % of fork_overlay's CPU, 2026-10-22). Polls
    every 5 ms against a wall-clock deadline, then makes the ordinary lookup, so
    the exceptions are unchanged. `timeout`: a Duration, seconds, or None.
    """
    seconds = 0.0
    if timeout is not None:
        seconds = (timeout.nanoseconds * 1e-9 if hasattr(timeout, 'nanoseconds')
                   else float(timeout))
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and not buffer.can_transform(target, source, stamp):
        time.sleep(0.005)
    return buffer.lookup_transform(target, source, stamp)


class LatestRaw:
    """The newest message of one topic, kept serialized and decoded once, on demand.
    For big messages read only sometimes (a LiDAR cloud is ~1.6 MB at 10 Hz).
    """

    def __init__(self, msg_type):
        self._type = msg_type
        self._raw = None
        self._decoded = None

    def store(self, raw: bytes) -> None:
        self._raw = raw
        self._decoded = None

    def get(self):
        if self._decoded is None and self._raw is not None:
            from rclpy.serialization import deserialize_message
            self._decoded = deserialize_message(self._raw, self._type)
        return self._decoded


class CameraTransformListener:
    """A tf listener that keeps up while its node is busy with depth frames.

    The stock listener spins the whole node and decodes every /tf message in
    Python, and fell 0.15-0.4 s behind on the robot (2026-10-23). This one spins a
    private node holding only the tf subscriptions and decodes a /tf message only
    if it names one of `moving_links`; /tf_static is kept whole. A needed link
    missing from `moving_links` fails the lookup (the safe direction).
    Call `close()` when done: the thread holds a reference to the node.
    """

    def __init__(self, buffer, node, moving_links):
        from threading import Thread

        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
        from rclpy.serialization import deserialize_message
        from tf2_msgs.msg import TFMessage

        self.buffer = buffer
        # rclpy is imported here, not at the top: test_common imports this
        # module without ROS.
        self._decode = lambda raw: deserialize_message(raw, TFMessage)
        # A child_frame_id in CDR is the name followed by its NUL terminator,
        # so 'zed2_left_camera_frame' does not match '..._frame_optical'.
        self._needles = tuple(name.encode() + b'\x00' for name in moving_links)
        self._node = rclpy.create_node(f'{node.get_name()}_tf',
                                       context=node.context,
                                       start_parameter_services=False,
                                       # It never logs, and its rosout
                                       # publisher clashed with the parent's
                                       # ("Publisher already registered").
                                       enable_rosout=False)
        self._node.create_subscription(
            TFMessage, '/tf', self._on_tf,
            QoSProfile(depth=100, durability=DurabilityPolicy.VOLATILE,
                       history=HistoryPolicy.KEEP_LAST),
            raw=True)
        self._node.create_subscription(
            TFMessage, '/tf_static', self._on_tf_static,
            QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                       history=HistoryPolicy.KEEP_LAST))
        self._executor = SingleThreadedExecutor(context=node.context)
        self._executor.add_node(self._node)
        self._thread = Thread(target=self._executor.spin, daemon=True)
        self._thread.start()

    def _on_tf(self, raw: bytes):
        if not any(needle in raw for needle in self._needles):
            return
        for transform in self._decode(raw).transforms:
            self.buffer.set_transform(transform, 'default_authority')

    def _on_tf_static(self, msg):
        for transform in msg.transforms:
            self.buffer.set_transform_static(transform, 'default_authority')

    def close(self):
        self._executor.shutdown()
        self._thread.join(timeout=2.0)
        self._node.destroy_node()


def tracked_topic(source: str) -> str:
    """The TrackedPallet topic for a perception source (D104)."""
    if source not in ('camera', 'fused'):
        raise ValueError(f'perception_source must be camera or fused, not {source!r}')
    return '/pallet_fusion/tracked' if source == 'fused' else '/pallet_tracker/tracked'
