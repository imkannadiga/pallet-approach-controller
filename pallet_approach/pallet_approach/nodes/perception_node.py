"""Lifecycle node: depth image -> PalletEstimate.

Thin on purpose. Everything it does is: get a depth frame, find the camera
transform for that frame's timestamp, hand both to `perception.estimate_pallet`,
publish the result. All the geometry is in `perception.py`.

It is a lifecycle node so the coordinator can keep it idle between runs: a
720p depth stream at 37 Hz through a Python pipeline is not something to run
continuously for no reason.
"""

from __future__ import annotations

import math

import rclpy
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener

from pallet_approach_interfaces.msg import PalletEstimate as PalletEstimateMsg

from pallet_approach import perception
from pallet_approach.calibration import correct_transform, correction_matrix
from pallet_approach.nodes._common import (
    image_to_array,
    intrinsics_from_camera_info,
)
from pallet_approach.ros_conversions import estimate_to_msg, transform_to_matrix

# The ZED publishes RELIABLE/VOLATILE, so match it. A BEST_EFFORT subscription
# would also work, but being explicit documents the measured QoS.
SENSOR_QOS = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)


class PerceptionNode(LifecycleNode):

    def __init__(self):
        super().__init__('pallet_perception')
        self._declare()
        self._intrinsics = None
        self._tf_buffer = None
        self._tf_listener = None
        self._depth_sub = None
        self._info_sub = None
        self._publisher = None
        self._warned = False
        self._frames = 0
        self._correction = None

    def _declare(self):
        self.declare_parameter('frames.camera_optical', 'zed2_left_camera_frame_optical')
        # Ground level, not the body frame. Note base_footprint is 25 mm above
        # the real floor (measured), so heights are taken relative to the
        # fitted floor plane, never to z = 0.
        self.declare_parameter('frames.ground', 'base_footprint')
        self.declare_parameter('frames.tf_timeout', 0.15)
        # The URDF's camera orientation is 0.868 deg out; see calibration.py.
        self.declare_parameter('frames.extrinsic_correction.enabled', True)
        self.declare_parameter('frames.extrinsic_correction.measured_floor_pitch', 0.0)
        self.declare_parameter('frames.extrinsic_correction.measured_floor_roll', 0.0)
        self.declare_parameter('topics.depth', '/zed2/zed_node/depth/depth_registered')
        self.declare_parameter('topics.depth_camera_info', '/zed2/zed_node/depth/camera_info')
        # the pallet_perception block of params.yaml, as flat dotted names
        for name, default in (
            ('depth.min_range', 0.40), ('depth.max_range', 4.00), ('depth.stride', 2),
            ('roi.x_min', 0.50), ('roi.x_max', 4.00), ('roi.y_min', -1.50),
            ('roi.y_max', 1.50), ('roi.z_min', -0.30), ('roi.z_max', 0.60),
            ('voxel_size', 0.02),
            ('floor.ransac_distance', 0.02), ('floor.ransac_iterations', 200),
            ('floor.max_normal_tilt', 0.17), ('floor.min_inlier_fraction', 0.25),
            ('floor.removal_margin', 0.03),
            ('cluster.eps', 0.05), ('cluster.min_samples', 8),
            ('cluster.min_cluster_points', 100),
            ('face.min_points', 60), ('face.max_tilt_from_vertical', 0.26),
            ('face.ransac_distance', 0.015),
        ):
            self.declare_parameter(name, default)

    def _params(self) -> dict:
        """Build the dict `perception.estimate_pallet` expects.

        params.yaml groups its values (`floor.ransac_distance`, `cluster.eps`)
        because that is how ROS parameters are readable and how they are
        declared. `estimate_pallet` wants them flat, under names that say what
        they are at the point of use. This is the one place that translates, so
        the grouping stays in the config and the perception code stays readable.
        """
        get = lambda name: self.get_parameter(name).value  # noqa: E731
        return {
            'seed': get('seed'),

            'stride': get('depth.stride'),
            'min_range': get('depth.min_range'),
            'max_range': get('depth.max_range'),

            'roi': {key: get(f'roi.{key}') for key in
                    ('x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max')},
            'voxel_size': get('voxel_size'),

            'floor_distance_threshold': get('floor.ransac_distance'),
            'ransac_iterations': get('floor.ransac_iterations'),
            'max_floor_tilt': get('floor.max_normal_tilt'),
            'floor_margin': get('floor.removal_margin'),

            'cluster_eps': get('cluster.eps'),
            'cluster_min_samples': get('cluster.min_samples'),
            'min_cluster_points': get('cluster.min_cluster_points'),
            'size_tolerance': get('cluster.size_tolerance'),

            'face_distance_threshold': get('face.ransac_distance'),
            'max_tilt_from_vertical': get('face.max_tilt_from_vertical'),
            'face_min_points': get('face.min_points'),
            'max_facing_angle': get('face.max_facing_angle'),

            'pallet_width': get('pallet.pallet_width'),
            'pallet_length': get('pallet.pallet_length'),
            'pallet_height': get('pallet.pallet_height'),

            'pocket_center_height': get('pallet.pocket_center_height'),
            'pocket_band_half_height': get('pallet.pocket_band_half_height'),
            'pocket_slab_half_depth': get('pallet.pocket_slab_half_depth'),
            'pocket_min_gap_width': get('pallet.pocket_min_gap_width'),
            'pocket_max_gap_width': get('pallet.pocket_max_gap_width'),
            'pocket_min_points_per_bin': get('pallet.pocket_min_points_per_bin'),
            'pocket_nominal_width': get('pallet.pocket_nominal_width'),
            'pocket_nominal_height': get('pallet.pocket_nominal_height'),
            'pocket_nominal_spacing': get('pallet.pocket_nominal_spacing'),
        }

    # ----------------------------------------------------------- lifecycle

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._publisher = self.create_lifecycle_publisher(
            PalletEstimateMsg, '~/estimate', 10)
        self._info_sub = self.create_subscription(
            CameraInfo, self.get_parameter('topics.depth_camera_info').value,
            self._on_camera_info, SENSOR_QOS)

        prefix = 'frames.extrinsic_correction'
        if self.get_parameter(f'{prefix}.enabled').value:
            pitch = self.get_parameter(f'{prefix}.measured_floor_pitch').value
            roll = self.get_parameter(f'{prefix}.measured_floor_roll').value
            self._correction = correction_matrix(pitch, roll)
            self.get_logger().info(
                'extrinsic correction on: levelling a measured floor tilt of '
                f'pitch {math.degrees(pitch):+.3f} deg, roll {math.degrees(roll):+.3f} deg')
        else:
            self.get_logger().warning(
                'extrinsic correction is DISABLED; the URDF camera orientation '
                'is 0.868 deg out, which is 30 mm of error at 2 m')

        self.get_logger().info('configured; waiting for camera_info')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self._frames = 0
        self._warned = False
        self._depth_sub = self.create_subscription(
            Image, self.get_parameter('topics.depth').value,
            self._on_depth, SENSOR_QOS)
        self.get_logger().info('active')
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        if self._depth_sub is not None:
            self.destroy_subscription(self._depth_sub)
            self._depth_sub = None
        self.get_logger().info(f'inactive after {self._frames} frames')
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self._tf_listener = None
        self._tf_buffer = None
        return TransitionCallbackReturn.SUCCESS

    # ------------------------------------------------------------ callbacks

    def _on_camera_info(self, msg: CameraInfo):
        if self._intrinsics is None:
            self._intrinsics = intrinsics_from_camera_info(msg)
            self.get_logger().info(
                'intrinsics: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}'.format(
                    **self._intrinsics))
            if any(abs(v) > 1e-9 for v in msg.d):
                self.get_logger().warning(
                    'camera_info has non-zero distortion, so this is not the '
                    'rectified stream the pipeline assumes')

    def _on_depth(self, msg: Image):
        if self._intrinsics is None:
            self.get_logger().warning('no camera_info yet, skipping frame',
                                      throttle_duration_sec=5.0)
            return
        try:
            depth = image_to_array(msg)
        except ValueError as error:
            self.get_logger().error(str(error))
            return

        camera_frame = msg.header.frame_id or self.get_parameter(
            'frames.camera_optical').value
        ground_frame = self.get_parameter('frames.ground').value
        timeout = rclpy.duration.Duration(
            seconds=self.get_parameter('frames.tf_timeout').value)
        try:
            # Stamped lookup, not Time(): zed2_camera_center ->
            # zed2_left_camera_frame is republished at 58 Hz by the SDK's
            # self-calibration, so "latest" is not the same as "at this frame".
            transform = self._tf_buffer.lookup_transform(
                ground_frame, camera_frame, msg.header.stamp, timeout)
        except Exception as error:
            self.get_logger().warning(f'tf {ground_frame} <- {camera_frame} failed: {error}',
                                      throttle_duration_sec=2.0)
            return

        camera_to_ground = transform_to_matrix(transform)
        if self._correction is not None:
            camera_to_ground = correct_transform(camera_to_ground, self._correction)

        try:
            estimate = perception.estimate_pallet(
                depth, self._intrinsics, camera_to_ground,
                self._params(), stamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        except NotImplementedError:
            if not self._warned:
                self.get_logger().error(
                    'perception.estimate_pallet is not implemented yet -- that '
                    'file is yours to write. Run: pytest test/test_perception.py')
                self._warned = True
            return
        except Exception as error:  # a perception bug must not take the node down
            self.get_logger().error(f'perception raised: {error!r}',
                                    throttle_duration_sec=2.0)
            return

        self._frames += 1
        header = msg.header
        header.frame_id = ground_frame
        self._publisher.publish(estimate_to_msg(estimate, header))


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
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
