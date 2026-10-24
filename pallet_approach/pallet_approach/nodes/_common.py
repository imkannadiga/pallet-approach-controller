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
    # array.array, not bytes: rclpy checks a bytes value element by element in
    # Python (58.5 ms for a 640x360 image, measured 2026-10-22), an array.array
    # of typecode 'B' is copied in C (1.3 ms).
    cloud.data = pyarray.array('B', array.tobytes())
    return cloud


def logged_transition(callback):
    """Log a lifecycle callback's exception before letting it propagate.

    rclpy's lifecycle catches an exception raised in on_configure (or any
    transition callback), fails the transition, and logs nothing: the node that
    broke stays silent, and the caller only learns the transition failed.
    Measured 2026-10-20 in the Gazebo sim -- a KeyError in the controller's
    on_configure showed up as nothing but the coordinator's "could not
    configure the managed nodes". Re-raised, so the transition still fails
    exactly as it would have; this only makes it say why.
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


def lookup_waiting(buffer, target: str, source: str, stamp, timeout=None):
    """`buffer.lookup_transform`, but waiting by SLEEPING, not spinning.

    rclpy's tf2 `lookup_transform(..., timeout)` waits in a hot loop. Measured
    2026-10-22 with py-spy: 81 % of fork_overlay's CPU was `can_transform`,
    the node at 87 % of a core, every time tf ran behind the image. On the
    Jetson that core is taken from perception. Here the wait polls every 5 ms
    and sleeps in between, which releases the GIL so the listener's own
    thread (`spin_thread=True`, see the nodes) can deliver the transform being
    waited for. Same timeout, same exceptions: after the deadline it makes the
    ordinary lookup, which raises exactly what it always did.

    `timeout` is an rclpy Duration, seconds as a float, or None for no wait.
    The deadline is wall time, so it means the same in sim time and real time.
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
    """The newest message of one topic, kept serialized and decoded on demand.

    For big messages a node only sometimes reads: subscribe with `raw=True`,
    `store` every arrival (a reference, no work), and `get` decodes the newest
    one the first time it is asked for and caches it until the next arrival.
    A 64 x 1024 Ouster cloud is ~1.6 MB at 10 Hz; the tracker reads one per
    estimate and the controller one per tick only while it drives.
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

    Measured on the robot 2026-10-23: perception processes a frame in ~155 ms
    at 5 Hz, and the stock `TransformListener(spin_thread=True)` fell 0.15 to
    0.4 s behind within two seconds of activation, so almost every stamped
    lookup failed with "extrapolation into the future" and the estimate rate
    dropped to 1.8 Hz. A separate process saw the same transforms 44 ms old.
    Two reasons, both in the stock listener:

    * its thread spins the WHOLE node, so it also runs depth callbacks and
      sits behind a 155 ms frame; and
    * it decodes every /tf message in Python -- ~200 a second on the robot
      (odom, map->odom, both ZED links) -- against the interpreter lock that
      perception's NumPy work holds.

    This one spins a private helper node holding only the tf subscriptions,
    and decodes a /tf message only if its raw bytes name one of
    `moving_links` as a child frame. Everything on /tf_static is kept. The
    buffer is then the same buffer, stamped lookups behave exactly as before,
    and a needed link missing from `moving_links` makes the lookup fail --
    no estimate, the safe direction -- rather than silently use stale data.

    Call `close()` when done. Dropping the reference is not enough: the thread
    holds one, so the stock listener was never collected and every
    configure/cleanup cycle left one more thread spinning the node.
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
