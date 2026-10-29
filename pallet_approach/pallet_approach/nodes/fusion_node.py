"""ROS 2 node: the fused perception (D102), running `fused_pipeline` on the live sensors.

Publishes ~/tracked (TrackedPallet, the tracker's message type) and ~/stats
(JSON at 1 Hz: timing, uncertainty, disagreement and freshness). With
`perception_source: camera` nothing reads ~/tracked (shadow); with
`perception:=fused` the coordinator, controller, dashboard and watchdog do
(D104). The pockets' width and height in the estimate are the side's NOMINAL
values, not measurements.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import deque

import numpy as np
import rclpy
import yaml
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import Header, String
from tf2_ros import Buffer

from pallet_approach.calibration import correct_transform, correction_matrix
from pallet_approach.fused_pipeline import FusedPipeline
from pallet_approach.model import PalletEstimate, Pocket
from pallet_approach.nodes._common import (CameraTransformListener, declare_lidar_params, image_to_array,
                                           intrinsics_from_camera_info, lidar_in_ground, lookup_waiting)
from pallet_approach.pointcloud import transform_points, unpack_cloud
from pallet_approach.ros_conversions import (estimate_to_msg, pose2d_from_odometry, stamp_to_seconds,
                                             transform_to_matrix)
from pallet_approach_interfaces.msg import TrackedPallet

RELIABLE = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
NOT_RUNNING = ('', 'IDLE', 'DONE', 'ABORTED')


def _merge(base: dict, over: dict) -> None:
    """Overlay `over` onto `base`, recursing into nested blocks."""
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value


class FusionNode(Node):

    def __init__(self, params_file: str | None = None, overlay_file: str | None = None):
        super().__init__('pallet_fusion')
        self._declare()
        get = lambda n: self.get_parameter(n).value   # noqa: E731
        with open(params_file or _default_params()) as f:
            shared = yaml.safe_load(f)['/**']['ros__parameters']
        # The overlay applies here too: the sim's switches off the real ZED's
        # disparity offset (D103), which made sim depth read +30 mm far.
        if overlay_file:
            with open(overlay_file) as f:
                over = (yaml.safe_load(f) or {}).get('/**', {}).get('ros__parameters', {})
            _merge(shared, over)
        self._shared = shared
        self._pipe = FusedPipeline(shared)
        self._odom: deque = deque(maxlen=400)
        self._intrinsics = None
        self._state = ''
        self._last_depth = -math.inf
        self._last_stamp = 0.0
        self._points = 0
        self._buffer = Buffer()
        self._tf = CameraTransformListener(self._buffer, self, get('frames.moving_links'))
        ec = 'frames.extrinsic_correction'
        self._correction = (correction_matrix(get(f'{ec}.measured_floor_pitch'), get(f'{ec}.measured_floor_roll'))
                            if get(f'{ec}.enabled') else np.eye(3))

        self._pub = self.create_publisher(TrackedPallet, '~/tracked', RELIABLE)
        self._stats_pub = self.create_publisher(String, '~/stats', 1)
        self.create_subscription(CameraInfo, get('topics.depth_camera_info'), self._on_info, qos_profile_sensor_data)
        self.create_subscription(Image, get('topics.depth'), self._on_depth, qos_profile_sensor_data, raw=True)
        self.create_subscription(PointCloud2, get('topics.lidar'), self._on_cloud, qos_profile_sensor_data,
                                 raw=True)
        self.create_subscription(Odometry, get('topics.odom'), self._on_odom, RELIABLE)
        self.create_subscription(String, '/coordinator/state', self._on_state, RELIABLE)
        self.create_timer(1.0, self._publish_stats)
        mode = ('DRIVING the run (perception_source: fused)' if get('perception_source') == 'fused'
                else 'in SHADOW, nothing reads it')
        self.get_logger().info(f'fused perception {mode} ({get("run_when")}); publishing ~/tracked, ~/stats')

    def _declare(self):
        for name, value in (
                ('run_when', 'always'),            # always | running (only while a run is active)
                ('perception_source', 'camera'),   # only for the log line: fused = the run reads us
                ('max_rate', 15.0),                # depth frames per second, D102 item 10
                ('odom_match_tolerance', 0.05),
                ('topics.depth', '/zed2/zed_node/depth/depth_registered'),
                ('topics.depth_camera_info', '/zed2/zed_node/depth/camera_info'),
                ('topics.lidar', '/pallet_approach/lidar'),
                ('topics.odom', '/odom'),
                ('frames.ground', 'base_footprint'),
                ('frames.tf_timeout', 0.15),
                ('frames.moving_links', ['zed2_left_camera_frame']),
                ('floor_z', -0.025)):
            self.declare_parameter(name, value)
        declare_lidar_params(self)
        # The pipeline's nested blocks (lidar_acquisition, camera_pockets,
        # fusion, ...) are read from the YAML files themselves (--params,
        # --overlay; see main) rather than declared key by key.

    # ----------------------------------------------------------- helpers
    def _active(self) -> bool:
        return self.get_parameter('run_when').value == 'always' or self._state not in NOT_RUNNING

    def _pose_at(self, stamp: float):
        if not self._odom:
            return None
        t, pose = min(self._odom, key=lambda row: abs(row[0] - stamp))
        if abs(t - stamp) > self.get_parameter('odom_match_tolerance').value:
            return None
        return (pose.x, pose.y, pose.yaw)

    def _timeout(self):
        return rclpy.duration.Duration(seconds=float(self.get_parameter('frames.tf_timeout').value))

    # ----------------------------------------------------------- callbacks
    def _on_state(self, msg: String):
        if (msg.data in NOT_RUNNING) != (self._state in NOT_RUNNING):
            self._pipe.reset()
        self._state = msg.data

    def _on_odom(self, msg: Odometry):
        self._odom.append((stamp_to_seconds(msg.header.stamp), pose2d_from_odometry(msg)))

    def _on_info(self, msg: CameraInfo):
        self._intrinsics = intrinsics_from_camera_info(msg)

    def _on_cloud(self, raw: bytes):
        if not self._active():
            return
        cloud = deserialize_message(raw, PointCloud2)
        stamp = stamp_to_seconds(cloud.header.stamp)
        robot = self._pose_at(stamp)
        if robot is None:
            return
        try:
            matrix = lidar_in_ground(self, self._buffer, cloud.header.frame_id, cloud.header.stamp,
                                     self._timeout())
        except Exception as error:                                        # noqa: BLE001
            self.get_logger().warning(f'LiDAR tf: {error}', throttle_duration_sec=5.0)
            return
        points = transform_points(unpack_cloud(cloud), matrix)
        self._points = len(points)
        self._last_stamp = max(self._last_stamp, stamp)
        self._pipe.on_lidar(points, robot, stamp)
        self._publish(cloud.header, stamp, robot)

    def _on_depth(self, raw: bytes):
        if not self._active() or self._intrinsics is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._last_depth < 1.0 / self.get_parameter('max_rate').value:
            return
        self._last_depth = now
        msg = deserialize_message(raw, Image)
        stamp = stamp_to_seconds(msg.header.stamp)
        robot = self._pose_at(stamp)
        if robot is None:
            return
        try:
            cam = transform_to_matrix(lookup_waiting(self._buffer, self.get_parameter('frames.ground').value,
                                                     msg.header.frame_id, msg.header.stamp, self._timeout()))
        except Exception as error:                                        # noqa: BLE001
            self.get_logger().warning(f'camera tf: {error}', throttle_duration_sec=5.0)
            return
        cam = correct_transform(cam, self._correction)
        self._last_stamp = max(self._last_stamp, stamp)
        self._pipe.on_depth(image_to_array(msg).astype(np.float32), self._intrinsics, cam, robot, stamp)
        self._publish(msg.header, stamp, robot)

    # ----------------------------------------------------------- output
    def _publish(self, header, stamp: float, robot):
        out = TrackedPallet()
        out.header = Header(stamp=header.stamp, frame_id=self.get_parameter('frames.ground').value)
        st = self._pipe.status(stamp)
        # D104 stages: confirmed = the LiDAR is tracking (enough to drive);
        # status READY = GO allowed; SEARCHING/TRACKING/DISAGREE otherwise.
        tracking = st.initialised and st.lidar_age <= self._shared['fusion']['lidar_max_age']
        out.status = ('DISAGREE' if st.disagreeing else 'READY' if st.go_allowed
                      else 'TRACKING' if tracking else 'SEARCHING')
        out.reason = st.reason
        out.confirmed = bool(tracking)
        out.confidence = float(1.0 if st.go_allowed else 0.5 if tracking else 0.0)
        found = self._pipe.in_base(robot, stamp)
        if found is None:
            out.estimate = estimate_to_msg(None, out.header)
            self._pub.publish(out)
            return
        centre, yaw = found
        side = self._shared['pallet']['sides'][self._pipe.side]
        floor = self._pipe.floor
        if floor is None:
            out.estimate = estimate_to_msg(None, out.header)
            self._pub.publish(out)
            return
        z = float(-(floor.normal[0] * centre[0] + floor.normal[1] * centre[1] + floor.offset) / floor.normal[2])
        tangent = np.array([math.sin(yaw), -math.cos(yaw)])
        pockets = tuple(
            # NOMINAL sizes: the camera measures where the pockets are, not how big.
            Pocket(np.array([*(centre + s * side['pocket_spacing'] / 2 * tangent), z + side['band_center']]),
                   float(side['pocket_width']), float(side['opening_top'] - side['opening_bottom']))
            for s in (1.0, -1.0))                                          # left to right as the robot sees them
        face = self._pipe.last_face
        estimate = PalletEstimate(
            position=np.array([centre[0], centre[1], z + side['band_center']]),
            normal=np.array([math.cos(yaw), math.sin(yaw), 0.0]), yaw=float(yaw), pockets=pockets,
            face_inlier_count=int(face.points) if face else 0, floor=floor, side=self._pipe.side,
            face_width=float(face.width) if face else float('nan'), stamp=stamp,
            frame_id=out.header.frame_id)
        out.estimate = estimate_to_msg(estimate, out.header)
        out.center_odom.x, out.center_odom.y = float(st.centre[0]), float(st.centre[1])
        out.yaw_odom = float(st.yaw)
        cam = self._pipe.last_camera
        # The cross-check proper (D102 item 6), not readiness: that is `status`.
        f = self._shared['fusion']
        out.lidar_evaluated = bool(cam is not None and st.lidar_age <= f['lidar_max_age']
                                   and st.camera_age <= f['camera_max_age'])
        out.lidar_agrees = bool(out.lidar_evaluated and not st.disagreeing)
        out.lidar_offset = float(cam.range_residual) if cam else float('nan')
        out.lidar_points = int(face.points) if face else 0
        out.lidar_detail = f'sigma lateral {1000 * st.sigma_lateral:.1f} mm, range {1000 * st.sigma_range:.1f} mm'
        self._pub.publish(out)

    def _publish_stats(self):
        # Data time, not node time: stamps are compared with stamps.
        st = self._pipe.status(self._last_stamp)
        t = self._pipe.timing
        cam = self._pipe.last_camera
        stats = {
            'tracking': self._pipe.tracking, 'side': self._pipe.side, 'in_range': self._pipe.in_range,
            'go_allowed': st.go_allowed, 'reason': st.reason,
            'ms': {'acquisition': t.acquisition, 'lidar': t.lidar, 'camera': t.camera}, 'calls': t.calls,
            'sigma_mm': {'range': 1000 * st.sigma_range, 'lateral': 1000 * st.sigma_lateral},
            'sigma_yaw_deg': math.degrees(st.sigma_yaw),
            'camera_minus_lidar_mm': 1000 * cam.range_residual if cam else None,
            'camera_yaw_residual_deg': math.degrees(cam.yaw_residual) if cam else None,
            'rejected': st.rejected, 'lidar_points': self._points,
        }
        self._stats_pub.publish(String(data=json.dumps(stats, allow_nan=True)))


def _default_params() -> str:
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory('pallet_approach'), 'config', 'params.yaml')


def main(args=None):
    # The files are plain arguments before --ros-args, not ROS parameters: a
    # node-specific entry in any --params-file beats both a launch dict and a
    # -p override, so as parameters the launch's paths never arrived.
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', default=None, help='params.yaml to read the fused blocks from')
    parser.add_argument('--overlay', default=None, help="merged over its /** block (the sim's)")
    files, _ = parser.parse_known_args(remove_ros_args(args if args is not None else sys.argv)[1:])
    node = FusionNode(files.params, files.overlay)
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
