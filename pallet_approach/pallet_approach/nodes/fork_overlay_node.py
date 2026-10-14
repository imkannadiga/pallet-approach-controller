"""Draw the virtual forks and detected pockets on the rectified left image.

A plain node, not a lifecycle one: it publishes an image topic and nothing it
does can move the robot, so there is no reason to gate it. It also runs while
idle, which is what makes it usable as a calibration check -- point the robot at
a pallet with nothing else running and look at whether the forks line up.

All the drawing is in `fork_overlay.py`, which is Athreya's file.
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from pallet_approach_interfaces.msg import PalletEstimate as PalletEstimateMsg

from pallet_approach import fork_overlay
from pallet_approach.calibration import correct_transform, correction_matrix
from pallet_approach.model import Plane
from pallet_approach.nodes._common import image_to_array, intrinsics_from_camera_info
from pallet_approach.ros_conversions import estimate_from_msg, transform_to_matrix

SENSOR_QOS = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)

# coordinator state -> the key in the fork_overlay.colors parameter block
STATE_COLORS = {
    'IDLE': 'idle', 'DETECTING': 'detecting', 'APPROACHING': 'approaching',
    'CHECKING': 'approaching', 'DONE': 'idle', 'GO': 'go', 'NO_GO': 'no_go',
}


class ForkOverlayNode(Node):

    def __init__(self):
        super().__init__('fork_overlay')
        self._declare()
        self._intrinsics = None
        self._estimate = None
        self._state = 'IDLE'
        self._warned = False
        self._last_published = 0.0

        prefix = 'frames.extrinsic_correction'
        self._correction = (
            correction_matrix(
                self.get_parameter(f'{prefix}.measured_floor_pitch').value,
                self.get_parameter(f'{prefix}.measured_floor_roll').value)
            if self.get_parameter(f'{prefix}.enabled').value else None)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._publisher = self.create_publisher(Image, '~/image', SENSOR_QOS)

        self.create_subscription(CameraInfo,
                                 self.get_parameter('topics.left_camera_info').value,
                                 self._on_camera_info, SENSOR_QOS)
        self.create_subscription(Image, self.get_parameter('topics.left_image').value,
                                 self._on_image, SENSOR_QOS)
        self.create_subscription(PalletEstimateMsg, '/pallet_perception/estimate',
                                 self._on_estimate, 10)
        self.create_subscription(String, '/coordinator/state', self._on_state, 10)

    def _declare(self):
        self.declare_parameter('topics.left_image',
                               '/zed2/zed_node/rgb/color/rect/image')
        self.declare_parameter('topics.left_camera_info',
                               '/zed2/zed_node/rgb/color/rect/camera_info')
        self.declare_parameter('frames.ground', 'base_footprint')
        self.declare_parameter('frames.camera_optical',
                               'zed2_left_camera_frame_optical')
        self.declare_parameter('frames.tf_timeout', 0.15)
        self.declare_parameter('frames.extrinsic_correction.enabled', True)
        self.declare_parameter('frames.extrinsic_correction.measured_floor_pitch', 0.0)
        self.declare_parameter('frames.extrinsic_correction.measured_floor_roll', 0.0)
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('alpha', 0.45)
        self.declare_parameter('line_thickness', 2)
        self.declare_parameter('draw_pockets', True)
        self.declare_parameter('draw_standoff_marker', True)
        # Where the floor is in base_footprint: -0.025 measured, not zero.
        # Perception's fitted value overrides it when there is one.
        self.declare_parameter('floor_z', -0.025)
        for name, default in (
            ('forks.tine_length', 1.150), ('forks.tine_width', 0.100),
            ('forks.tine_thickness', 0.040), ('forks.spacing', 0.3725),
            ('forks.bottom_height', 0.052), ('forks.carriage_x', 0.0),
        ):
            self.declare_parameter(name, default)
        for name, default in (
            ('colors.idle', [160, 160, 160]), ('colors.detecting', [160, 160, 160]),
            ('colors.approaching', [0, 200, 255]), ('colors.go', [0, 200, 0]),
            ('colors.no_go', [0, 0, 220]),
        ):
            self.declare_parameter(name, default)

    def _params(self) -> dict:
        get = lambda n: self.get_parameter(n).value  # noqa: E731
        return {
            'forks': {k: get(f'forks.{k}') for k in (
                'tine_length', 'tine_width', 'tine_thickness', 'spacing',
                'bottom_height', 'carriage_x')},
            'colors': {k: tuple(int(v) for v in get(f'colors.{k}')) for k in (
                'idle', 'detecting', 'approaching', 'go', 'no_go')},
            'alpha': get('alpha'),
            'line_thickness': get('line_thickness'),
            'draw_pockets': get('draw_pockets'),
            'draw_standoff_marker': get('draw_standoff_marker'),
        }

    def _on_camera_info(self, msg: CameraInfo):
        if self._intrinsics is None:
            self._intrinsics = intrinsics_from_camera_info(msg)

    def _on_estimate(self, msg: PalletEstimateMsg):
        self._estimate = estimate_from_msg(msg)

    def _on_state(self, msg: String):
        self._state = msg.data

    def _on_image(self, msg: Image):
        if self._intrinsics is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        period = 1.0 / float(self.get_parameter('publish_rate').value)
        if now - self._last_published < period:
            return                       # decimate from the camera's ~37 Hz

        camera_frame = msg.header.frame_id or self.get_parameter(
            'frames.camera_optical').value
        ground_frame = self.get_parameter('frames.ground').value
        try:
            # Look up ground <- camera, the same direction perception uses, so
            # the extrinsic correction is applied by the same code path, then
            # invert it. Correcting an already-inverted transform would need the
            # inverse correction too, which is one easy sign error away.
            transform = self._tf_buffer.lookup_transform(
                ground_frame, camera_frame, msg.header.stamp,
                rclpy.duration.Duration(
                    seconds=self.get_parameter('frames.tf_timeout').value))
        except Exception as error:
            self.get_logger().warning(f'tf lookup failed: {error}',
                                      throttle_duration_sec=5.0)
            return

        # Prefer the plane perception actually fitted; fall back to a level
        # plane at the measured floor height when there is no fit, which is the
        # case at the standoff where almost no floor is in frame.
        floor = None if self._estimate is None else self._estimate.floor
        if floor is None:
            floor = Plane(normal=np.array([0.0, 0.0, 1.0]),
                          offset=-float(self.get_parameter('floor_z').value))

        camera_to_ground = transform_to_matrix(transform)
        if self._correction is not None:
            camera_to_ground = correct_transform(camera_to_ground, self._correction)
        ground_to_camera = np.linalg.inv(camera_to_ground)

        try:
            annotated = fork_overlay.render(
                image_to_array(msg), self._estimate,
                STATE_COLORS.get(self._state, 'idle'),
                ground_to_camera, self._intrinsics,
                self._params(), floor)
        except NotImplementedError:
            if not self._warned:
                self.get_logger().error(
                    'fork_overlay.render is not implemented yet -- that file is '
                    'yours to write. Start with tine_corners and project_points.')
                self._warned = True
            return
        except Exception as error:
            self.get_logger().error(f'overlay raised: {error!r}',
                                    throttle_duration_sec=5.0)
            return

        out = Image()
        out.header = msg.header
        out.height, out.width = annotated.shape[0], annotated.shape[1]
        out.encoding = 'bgr8'
        out.is_bigendian = 0
        out.step = out.width * 3
        out.data = annotated.tobytes()
        self._publisher.publish(out)
        self._last_published = now


def main(args=None):
    rclpy.init(args=args)
    node = ForkOverlayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
