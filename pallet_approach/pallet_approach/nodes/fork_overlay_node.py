"""Draw the virtual forks and detected pockets on the rectified left image.

A plain node, not a lifecycle one: it publishes an image topic and nothing it
does can move the robot, so there is no reason to gate it. It also runs while
idle, which is what makes it usable as a calibration check -- point the robot at
a pallet with nothing else running and look at whether the forks line up.

The drawing itself is in `fork_overlay.py`.

**The published image is 640x360 at 5 Hz, and that is not a preference.**
Measured 2026-10-13: one subscriber to a raw ZED image topic from the main PC
dropped the whole ZED node to about 0.2 Hz, and even compressed full-rate
images were unusable remotely. So anything meant to be looked at from another
machine is a separate, downscaled, low-rate topic (CLAUDE.md, "What changed"
item 7). This is that topic, and it is the only image this stack publishes.

It renders at FULL resolution and downscales afterwards, rather than drawing on
a downscaled image. Drawing small would mean scaling the intrinsics, which is
one more place to get a factor of two wrong -- in the one tool whose whole job
is to make a millimetre-scale error visible. A 16.8 px fork-to-pocket gap is
still 8.4 px after halving, which is plenty to see.

The fork height comes from the virtual fork simulator, so the lowering and the
lift are visible on the image rather than only in a log line.
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from pallet_approach_interfaces.msg import ForkState
from pallet_approach_interfaces.msg import PalletEstimate as PalletEstimateMsg

from pallet_approach import fork_overlay
from pallet_approach.calibration import correct_transform, correction_matrix
from pallet_approach.model import Plane
from pallet_approach.nodes._common import image_to_array, intrinsics_from_camera_info
from pallet_approach.ros_conversions import estimate_from_msg, transform_to_matrix

SENSOR_QOS = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)

# Coordinator state -> the key in the fork_overlay.colors parameter block.
# Grouped by what the robot is doing rather than one colour per state: looking
# (grey), driving towards the pallet (amber), committed with the forks in or
# going in (green), and stopped because something was wrong (red).
STATE_COLORS = {
    'IDLE': 'idle',
    'CONFIRMING': 'detecting',
    'CLASSIFYING': 'detecting',
    'BACKING_OFF': 'approaching',
    'CLOSING': 'approaching',
    'CHECKING': 'approaching',
    'LOWERING': 'go',
    'INSERTING': 'inserting',
    'LIFTING': 'inserting',
    'BACKING_OUT': 'go',
    'DONE': 'idle',
    'ABORTED': 'no_go',
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
        self._fork_height = None
        """The live carriage height from the simulator, or None before it has
        said anything -- in which case the configured `forks.bottom_height` is
        drawn. None and not 0.0: a height of zero is a carriage on the floor,
        which is a drawing, not an absence."""

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
        self.create_subscription(ForkState, '/virtual_forks/state',
                                 self._on_fork_state, 10)

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
        self.declare_parameter('debug_width', 640)
        self.declare_parameter('debug_height', 360)
        # Where the floor is in base_footprint: -0.025 measured, not zero.
        # Perception's fitted value overrides it when there is one.
        self.declare_parameter('floor_z', -0.025)
        for name, default in (
            ('forks.tine_length', 1.150), ('forks.tine_width', 0.100),
            ('forks.tine_thickness', 0.040), ('forks.carriage_x', 0.4625),
            # One setting for both faces (D27), so there is nothing per-side
            # to look up here any more.
            ('forks.spacing', 0.365), ('forks.bottom_height', 0.040),
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
            # One fork setting for both faces (D27), so this is a plain read.
            # The tines are built against this, not against the fitted floor:
            # they are bolted to the robot. See fork_overlay.render.
            'floor_z': get('floor_z'),
            'forks': dict(
                {k: get(f'forks.{k}') for k in (
                    'tine_length', 'tine_width', 'tine_thickness', 'spacing',
                    'bottom_height', 'carriage_x')},
                # The LIVE carriage height when the simulator has reported one,
                # so the lowering and the lift are visible on the image and not
                # only in a log line. Falls back to the configured height
                # before it has said anything.
                bottom_height=(get('forks.bottom_height')
                               if self._fork_height is None
                               else self._fork_height)),
            'colors': {k: tuple(int(v) for v in get(f'colors.{k}')) for k in (
                'idle', 'detecting', 'approaching', 'inserting', 'go',
                'no_go')},
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

    def _on_fork_state(self, msg: ForkState):
        self._fork_height = float(msg.height)

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

        annotated = self._downscale(annotated)
        out = Image()
        out.header = msg.header
        out.height, out.width = annotated.shape[0], annotated.shape[1]
        out.encoding = 'bgr8'
        out.is_bigendian = 0
        out.step = out.width * 3
        out.data = annotated.tobytes()
        self._publisher.publish(out)
        self._last_published = now

    def _downscale(self, image):
        """To the debug size, or unchanged if it is already no larger.

        INTER_AREA, because the image is being shrunk: it averages the pixels
        that fall into each output pixel, where INTER_LINEAR samples two of
        them and drops the rest. On an overlay made of thin lines that is the
        difference between a visible fork edge and a dotted one.
        """
        width = int(self.get_parameter('debug_width').value)
        height = int(self.get_parameter('debug_height').value)
        if image.shape[1] <= width and image.shape[0] <= height:
            return image
        import cv2
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


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
