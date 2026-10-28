"""Draw the virtual forks and detected pockets on the rectified left image.

A plain node: it cannot move the robot, and running while idle makes it a
calibration check. The fork height comes from the virtual fork simulator.

Publishes 640x360 at 5 Hz, per the network rule (docs/SYSTEM.md). It renders
at full resolution and downscales afterwards, so the intrinsics are never
rescaled. `dashboard_node.py` adds the HUD on top and is what the browser
stream shows by default.
"""

from __future__ import annotations

import array as pyarray

import numpy as np
import rclpy
from rclpy.serialization import deserialize_message
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer

from pallet_approach_interfaces.msg import ForkState
from pallet_approach_interfaces.msg import PalletEstimate as PalletEstimateMsg

from pallet_approach import fork_overlay
from pallet_approach.calibration import correct_transform, correction_matrix
from pallet_approach.dashboard import STATE_COLOR_KEYS as STATE_COLORS
from pallet_approach.model import Plane
from pallet_approach.nodes._common import (
    CameraTransformListener, image_to_array, intrinsics_from_camera_info, lookup_waiting)
from pallet_approach.rate_limit import Schedule
from pallet_approach.ros_conversions import estimate_from_msg, transform_to_matrix

SENSOR_QOS = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)

# As in perception_node: a stamp this far from our clock is on another clock,
# and a stamped tf lookup against it fails as "extrapolation into the past".
CLOCK_MISMATCH_SECONDS = 5.0


class ForkOverlayNode(Node):

    def __init__(self):
        super().__init__('fork_overlay')
        self._declare()
        self._intrinsics = None
        self._estimate = None
        self._state = 'IDLE'
        self._warned = False
        self._last_published = 0.0
        self._schedule = Schedule(float(self.get_parameter('publish_rate').value))
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
        self._tf_listener = CameraTransformListener(
            self._tf_buffer, self, self.get_parameter('frames.moving_links').value)
        self._publisher = self.create_publisher(Image, '~/image', SENSOR_QOS)

        self.create_subscription(CameraInfo,
                                 self.get_parameter('topics.left_camera_info').value,
                                 self._on_camera_info, SENSOR_QOS)
        # raw=True, and deserialized only once the 5 Hz check passes: a ZED
        # colour frame is 3.7 MB, arrives at ~37 Hz and would otherwise cost a
        # Python conversion for every frame this node then drops (see
        # perception_node, measured 2026-10-22).
        self.create_subscription(Image, self.get_parameter('topics.left_image').value,
                                 self._on_image, SENSOR_QOS, raw=True)
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
        self.declare_parameter('frames.moving_links', ['zed2_left_camera_frame'])
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
            ('colors.approaching', [0, 200, 255]),
            # One literal get() per key in _params(), so test_params sees every read.
            ('colors.inserting', [128, 255, 0]),
            ('colors.go', [0, 200, 0]), ('colors.no_go', [0, 0, 220]),
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
            # Explicit literal calls to get(), not a loop over an f-string
            # key -- see the comment on 'colors.inserting' in _declare above
            # for why that distinction is load-bearing here, not just style.
            'colors': {
                'idle': tuple(int(v) for v in get('colors.idle')),
                'detecting': tuple(int(v) for v in get('colors.detecting')),
                'approaching': tuple(int(v) for v in get('colors.approaching')),
                'inserting': tuple(int(v) for v in get('colors.inserting')),
                'go': tuple(int(v) for v in get('colors.go')),
                'no_go': tuple(int(v) for v in get('colors.no_go')),
            },
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

    def _on_image(self, raw: bytes):
        if self._intrinsics is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        period = 1.0 / float(self.get_parameter('publish_rate').value)
        # Decimate on a schedule (rate_limit.Schedule), not on "time since the
        # last one", which from the ZED's 37 Hz can only give 18.5 fps.
        self._schedule.period = period
        if not self._schedule.due(now):
            return
        msg = deserialize_message(raw, Image)

        camera_frame = msg.header.frame_id or self.get_parameter(
            'frames.camera_optical').value
        ground_frame = self.get_parameter('frames.ground').value

        # Try the stamped lookup; if it fails (lag or a clock-epoch split, which
        # look alike here), fall back to the latest transform. Unconditional,
        # unlike perception's bench-only fallback: this node only draws. The
        # two causes are logged separately, since only the epoch split matters
        # to perception.
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        # NO wait (2026-10-22). This node only draws, and it already falls back
        # to the latest transform; waiting up to frames.tf_timeout per frame
        # for the stamped one held the overlay to ~2 fps in the sim (0.5 s
        # timeout) against a 25 fps target. The stamped lookup is tried first
        # because when it is there it is the accurate one.
        timeout = None
        try:
            # Look up ground <- camera, the same direction perception uses, so
            # the extrinsic correction is applied by the same code path, then
            # invert it. Correcting an already-inverted transform would need the
            # inverse correction too, which is one easy sign error away.
            transform = lookup_waiting(self._tf_buffer, ground_frame, camera_frame, msg.header.stamp, timeout)
        except Exception as stamped_error:
            try:
                transform = lookup_waiting(self._tf_buffer, ground_frame, camera_frame, rclpy.time.Time(), timeout)
            except Exception as error:
                # Neither lookup worked -- the ORIGINAL (stamped) failure is
                # the informative one; the latest-transform attempt failing
                # too usually just means tf has nothing published at all yet.
                self.get_logger().warning(f'tf lookup failed: {error}',
                                          throttle_duration_sec=5.0)
                return
            gap = now - stamp
            if abs(gap) > CLOCK_MISMATCH_SECONDS:
                self.get_logger().warning(
                    f'this frame is stamped {stamp:.3f} but our clock reads '
                    f'{now:.3f}, {gap:.0f} s apart -- the camera and /tf are '
                    f'on different clocks, not just running a bit behind. '
                    f'Using the latest transform for this overlay, which is '
                    f'fine here but NOT fine for perception or the closing '
                    f'loop: run tools/clock_audit.py and fix the clock before '
                    f'driving.',
                    throttle_duration_sec=30.0)
            else:
                self.get_logger().info(
                    f'stamped tf lookup failed ({stamped_error}); used the '
                    f'latest available transform instead, {gap * 1000:+.0f} ms '
                    f'from this frame\'s own stamp',
                    throttle_duration_sec=10.0)

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

        # Shrink FIRST, then draw at the debug size with intrinsics, line
        # width and text scaled by the same factor: 1.1 ms a frame instead of
        # 7.3 ms for drawing at 1280x720 and shrinking after (measured
        # 2026-10-22), which is what makes a 25 fps dashboard affordable on
        # the Jetson. The picture is the same; only the work is smaller.
        image, intrinsics, params = self._at_debug_size(
            image_to_array(msg), self._intrinsics, self._params())
        try:
            annotated = fork_overlay.render(
                image, self._estimate,
                STATE_COLORS.get(self._state, 'idle'),
                ground_to_camera, intrinsics, params, floor)
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

        annotated = self._downscale(annotated)      # a no-op unless the camera is smaller
        out = Image()
        out.header = msg.header
        out.height, out.width = annotated.shape[0], annotated.shape[1]
        out.encoding = 'bgr8'
        out.is_bigendian = 0
        out.step = out.width * 3
        # array.array, not bytes: rclpy checks a bytes value element by element in
        # Python (58.5 ms for a 640x360 image, measured 2026-10-22), an array.array
        # of typecode 'B' is copied in C (1.3 ms).
        out.data = pyarray.array('B', annotated.tobytes())
        self._publisher.publish(out)
        self._last_published = now

    def _at_debug_size(self, image, intrinsics: dict, params: dict):
        """The camera frame shrunk to the debug size, with the intrinsics and
        the drawing sizes scaled to match. Unchanged if it is already small."""
        width = int(self.get_parameter('debug_width').value)
        if image.shape[1] <= width:
            return image, intrinsics, params
        import cv2
        factor = width / image.shape[1]
        height = int(round(image.shape[0] * factor))
        small = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        scaled = {key: (value * factor if key in ('fx', 'fy', 'cx', 'cy') else value)
                  for key, value in intrinsics.items()}
        params = dict(params, text_scale=factor,
                      line_thickness=max(1, round(params.get('line_thickness', 2) * factor)))
        return small, scaled, params

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
