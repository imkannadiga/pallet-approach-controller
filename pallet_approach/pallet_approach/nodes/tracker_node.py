"""Pallet tracker: the memory perception does not have.

Subscribes to one-frame estimates and publishes a track with a confidence and a
LiDAR cross-check. The coordinator reads this node, not perception, so the
"several consistent frames before anything moves" rule is enforced in one
place instead of being re-implemented in the state machine.

Two things are deliberate about the plumbing.

**The LiDAR cloud is decoded lazily.** The raw message is kept and unpacked
only when an estimate arrives, so a 65 536-point cloud is decoded at
perception's 5 Hz rather than the Ouster's 10 Hz, and the cross-check compares
the cloud with the estimate closest to it in time. A cloud older than
`lidar.max_cloud_age` is treated as no LiDAR at all rather than compared
anyway.

**Odometry is looked up at the estimate's stamp**, not taken as the latest.
At 50 Hz and 0.10 m/s the difference is a couple of millimetres, which does not
matter -- but the tracker's whole argument for comparing positions in `odom` is
that the robot's motion has been taken out, and taking out the motion at the
wrong instant quietly undoes it. If no odometry sample is close enough to the
stamp, the frame is rejected with that as the reason.

The LiDAR subscription is BEST_EFFORT: `/ouster/points` publishes BEST_EFFORT
and a RELIABLE subscriber receives nothing at all from it (measured
2026-10-15).
"""

from __future__ import annotations

from collections import deque

import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from pallet_approach.nodes._common import LatestRaw, lookup_waiting
from pallet_approach_interfaces.msg import PalletEstimate as PalletEstimateMsg
from pallet_approach_interfaces.msg import TrackedPallet

from pallet_approach.model import Pose2D
from pallet_approach.calibration import correct_lidar_transform
from pallet_approach.pointcloud import transform_points, unpack_cloud
from pallet_approach.ros_conversions import (
    estimate_from_msg,
    estimate_to_msg,
    pose2d_from_odometry,
    stamp_to_seconds,
    transform_to_matrix,
)
from pallet_approach.tracker import Tracker

RELIABLE = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
BEST_EFFORT = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)

NOT_RUNNING = ('', 'IDLE', 'DONE', 'ABORTED')


class TrackerNode(Node):

    def __init__(self):
        super().__init__('pallet_tracker')
        self._declare()
        self._tracker = Tracker(self._params())
        self._odom: deque[tuple[float, Pose2D]] = deque(maxlen=200)
        self._cloud = LatestRaw(PointCloud2)
        self._state = ''

        self._pub = self.create_publisher(TrackedPallet, '~/tracked', RELIABLE)
        self.create_subscription(PalletEstimateMsg, '/pallet_perception/estimate',
                                 self._on_estimate, RELIABLE)
        self.create_subscription(Odometry, self.get_parameter('topics.odom').value,
                                 self._on_odom, RELIABLE)
        self.create_subscription(PointCloud2, self.get_parameter('topics.lidar').value,
                                 self._on_cloud, BEST_EFFORT, raw=True)
        self.create_subscription(String, '/coordinator/state', self._on_state, RELIABLE)

        self._buffer = Buffer()
        self._listener = TransformListener(self._buffer, self, spin_thread=True)
        self.get_logger().info('tracking; publishing ~/tracked per estimate')

    # -------------------------------------------------------------- params

    def _declare(self):
        self.declare_parameter('topics.odom', '/odom')
        self.declare_parameter('topics.lidar', '/pallet_approach/lidar')
        self.declare_parameter('frames.ground', 'base_footprint')
        self.declare_parameter('frames.tf_timeout', 0.15)
        self.declare_parameter('floor_z', -0.025)
        self.declare_parameter('frames.lidar_correction.enabled', True)
        self.declare_parameter('frames.lidar_correction.measured_floor_pitch', 0.0)
        self.declare_parameter('frames.lidar_correction.measured_floor_roll', 0.0)
        self.declare_parameter('frames.lidar_correction.measured_floor_height', -0.025)
        self.declare_parameter('pallet.pallet_height', 0.145)
        for name, default in (
            ('window', 10), ('min_window_frames', 5), ('confirm_frames', 5),
            ('min_confidence', 0.6), ('max_jump', 0.15), ('max_jump_per_m2', 0.016),
            ('max_turn', 0.20),
            ('require_same_side', True), ('odom_match_tolerance', 0.05),
            ('fusion_memory', 0.95), ('fusion_min_range', 0.5),
            ('lidar.search_ahead', 0.60), ('lidar.search_behind', 1.00),
            ('lidar.half_width', 0.70),
            ('lidar.floor_margin', 0.03), ('lidar.height_margin', 0.05),
            ('lidar.min_points', 20), ('lidar.max_offset', 0.10),
            ('lidar.max_cloud_age', 0.30),
        ):
            self.declare_parameter(name, default)

    def _params(self) -> dict:
        """One flat dict, with the names `tracker.py` uses at the point of use.

        Flattened here rather than in the tracker so that the pure module takes
        a plain dict and stays testable without ROS -- the same split as
        perception's `_params()`.
        """
        get = lambda name: self.get_parameter(name).value   # noqa: E731
        return {
            'window': get('window'),
            'min_window_frames': get('min_window_frames'),
            'confirm_frames': get('confirm_frames'),
            'min_confidence': get('min_confidence'),
            'max_jump': get('max_jump'),
            'max_jump_per_m2': get('max_jump_per_m2'),
            'max_turn': get('max_turn'),
            'require_same_side': get('require_same_side'),
            'fusion_memory': get('fusion_memory'),
            'fusion_min_range': get('fusion_min_range'),
            'floor_z': get('floor_z'),
            'pallet_height': get('pallet.pallet_height'),
            'search_ahead': get('lidar.search_ahead'),
            'search_behind': get('lidar.search_behind'),
            'half_width': get('lidar.half_width'),
            'floor_margin': get('lidar.floor_margin'),
            'height_margin': get('lidar.height_margin'),
            'min_points': get('lidar.min_points'),
            'max_offset': get('lidar.max_offset'),
        }

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # ----------------------------------------------------------- callbacks

    def _on_state(self, msg: String):
        was_running = self._state not in NOT_RUNNING
        self._state = msg.data
        running = self._state not in NOT_RUNNING
        if running != was_running:
            # Both edges: a new run must not inherit the last run's track, and
            # a finished run must not leave one lying around to be inherited.
            self._tracker.reset()
            self.get_logger().info(f'track reset ({self._state or "idle"})')

    def _on_odom(self, msg: Odometry):
        self._odom.append((stamp_to_seconds(msg.header.stamp),
                           pose2d_from_odometry(msg)))

    def _on_cloud(self, raw: bytes):
        """Keep the bytes; decode them when an estimate needs them."""
        self._cloud.store(raw)

    def _pose_at(self, stamp: float) -> Pose2D | None:
        """The odometry sample nearest `stamp`, or None if none is close enough.

        Nearest rather than interpolated: /odom runs at 50 Hz, so the nearest
        sample is within 10 ms, which at the 0.10 m/s closing speed is 1 mm --
        two orders below the 0.15 m association gate. Interpolation would be
        more precision than the input has.
        """
        if not self._odom:
            return None
        stamp_, pose = min(self._odom, key=lambda row: abs(row[0] - stamp))
        tolerance = float(self.get_parameter('odom_match_tolerance').value)
        return pose if abs(stamp_ - stamp) <= tolerance else None

    def _lidar_in_ground(self, stamp: float):
        """The newest cloud in `frames.ground`, or None.

        None covers three different situations -- no cloud yet, a cloud too old
        to compare, and a tf lookup that failed -- and each is logged, because
        "the cross-check did not run" has to be attributable. The caller turns
        None into NOT EVALUATED, never into agreement.
        """
        cloud = self._cloud.get()
        if cloud is None:
            return None
        age = stamp - stamp_to_seconds(cloud.header.stamp)
        if abs(age) > float(self.get_parameter('lidar.max_cloud_age').value):
            self.get_logger().warning(
                f'newest LiDAR cloud is {age:.2f} s from this estimate; not '
                f'cross-checking', throttle_duration_sec=5.0)
            return None
        ground = self.get_parameter('frames.ground').value
        try:
            transform = lookup_waiting(self._buffer, ground, cloud.header.frame_id, cloud.header.stamp,
                rclpy.duration.Duration(
                    seconds=float(self.get_parameter('frames.tf_timeout').value)))
        except Exception as error:
            self.get_logger().warning(
                f'tf {ground} <- {cloud.header.frame_id}: {error}',
                throttle_duration_sec=5.0)
            return None
        matrix = transform_to_matrix(transform)
        if self.get_parameter('frames.lidar_correction.enabled').value:
            matrix = correct_lidar_transform(
                matrix,
                self.get_parameter('frames.lidar_correction.measured_floor_pitch').value,
                self.get_parameter('frames.lidar_correction.measured_floor_roll').value,
                self.get_parameter('frames.lidar_correction.measured_floor_height').value,
                self.get_parameter('floor_z').value)
        return transform_points(unpack_cloud(cloud), matrix)

    def _on_estimate(self, msg: PalletEstimateMsg):
        estimate = estimate_from_msg(msg)
        stamp = stamp_to_seconds(msg.header.stamp)
        now = self._now()

        if estimate is None:
            # An explicitly invalid estimate is evidence too: it counts as an
            # offered frame that was not accepted, which is what makes the
            # confidence fall when the pallet leaves the view instead of
            # freezing at its last value.
            self._tracker.update(None, Pose2D(0.0, 0.0, 0.0), now)
            self._publish(msg, None)
            return

        pose = self._pose_at(stamp)
        if pose is None:
            self._tracker.update(None, Pose2D(0.0, 0.0, 0.0), now)
            self._tracker.reason = 'no odometry sample at the frame stamp'
            self.get_logger().warning(self._tracker.reason,
                                      throttle_duration_sec=2.0)
            self._publish(msg, None)
            return

        points = self._lidar_in_ground(stamp)
        track = self._tracker.update(estimate, pose, now, lidar_points=points)
        self._publish(msg, track)

    # ------------------------------------------------------------- publish

    def _publish(self, source_msg, track):
        out = TrackedPallet()
        out.header.stamp = source_msg.header.stamp
        out.header.frame_id = source_msg.header.frame_id
        out.status = self._tracker.status()
        out.reason = self._tracker.reason
        out.confidence = float(self._tracker.confidence)
        out.consecutive_frames = int(self._tracker.consecutive)
        out.confirmed = bool(self._tracker.confirmed)

        if track is None or track.last_estimate is None:
            out.estimate = estimate_to_msg(None, source_msg.header)
            out.total_frames = 0
            out.lidar_offset = float('nan')
            self._pub.publish(out)
            return

        out.estimate = estimate_to_msg(track.last_estimate, source_msg.header)
        out.total_frames = int(track.frames)
        out.center_odom = Point(x=float(track.center[0]), y=float(track.center[1]),
                                z=float(track.center[2]))
        out.yaw_odom = float(track.yaw)
        check = track.cross_check
        out.lidar_evaluated = bool(check.evaluated)
        out.lidar_agrees = bool(check.agrees)
        out.lidar_offset = float(check.offset)
        out.lidar_points = int(check.points)
        out.lidar_detail = str(check.detail)
        self._pub.publish(out)

        if check.blocking:
            self.get_logger().warning(f'LiDAR disagrees: {check.detail}',
                                      throttle_duration_sec=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = TrackerNode()
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
