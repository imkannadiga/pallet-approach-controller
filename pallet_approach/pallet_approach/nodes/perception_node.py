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
from rclpy.serialization import deserialize_message
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from rclpy.qos import QoSProfile, ReliabilityPolicy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from tf2_ros import Buffer

from pallet_approach_interfaces.msg import PalletEstimate as PalletEstimateMsg

from pallet_approach import perception
from pallet_approach.perception import flatten_side
from pallet_approach.rate_limit import FrameGate
from pallet_approach.calibration import correct_transform, correction_matrix
from pallet_approach.nodes._common import (
    CameraTransformListener,
    lookup_waiting,
    image_to_array,
    intrinsics_from_camera_info,
    logged_transition,
    points_to_cloud,
)
from pallet_approach.ros_conversions import estimate_to_msg, transform_to_matrix

# The ZED publishes RELIABLE/VOLATILE, so match it. A BEST_EFFORT subscription
# would also work, but being explicit documents the measured QoS.
#
# depth=1 on the depth subscription, deliberately. The node processes the
# NEWEST frame and drops the rest (see _on_depth), so a deeper queue would only
# hand it frames it is going to throw away, having paid to deliver them.
SENSOR_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)

# Beyond this much difference between a frame's stamp and our own clock, the
# two are not the same clock and no amount of waiting will fix it. Well above
# any real transport delay or buffering (tens of milliseconds) and far below
# the kind of gap a wrong clock produces (seconds at least, 1.79e9 in the case
# that prompted this).
CLOCK_MISMATCH_SECONDS = 5.0

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
        self._diag_pub = None
        self._diag_timer = None
        # Rate limiting and timing. See rate_limit.FrameGate and _on_depth.
        self._gate = FrameGate(max_rate=5.0)
        self._latest = None          # the newest depth message not yet processed
        self._last_age = float('nan')
        self._last_side = 'unknown'
        self._last_skip = 'not started'
        self._stamp_skew = float('nan')

    def _declare(self):
        self.declare_parameter('frames.camera_optical', 'zed2_left_camera_frame_optical')
        # Ground level, not the body frame. Note base_footprint is 25 mm above
        # the real floor (measured), so heights are taken relative to the
        # fitted floor plane, never to z = 0.
        self.declare_parameter('frames.ground', 'base_footprint')
        self.declare_parameter('frames.tf_timeout', 0.15)
        self.declare_parameter('frames.moving_links', ['zed2_left_camera_frame'])
        # Bench escape hatch for a camera whose stamps are on another clock.
        # Off by default and it must stay that way for anything that drives;
        # see the comment at the lookup.
        self.declare_parameter('frames.tf_fallback_to_latest', False)
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
            ('face.deck_margin', 0.03),
            ('floor.score_points', 3000),
            ('face.ransac_distance', 0.015),
            ('face.max_facing_angle', 1.05),
            ('cluster.size_tolerance', 0.25),
            ('cluster.max_height_above_pallet', 0.04),
            ('cluster.max_candidates', 3),
            # RANSAC draws random samples, so the same frame can give slightly
            # different planes run to run. A fixed seed makes a replay
            # reproducible, which is the difference between "the bag shows the
            # bug" and "the bag sometimes shows the bug".
            ('seed', 0),
            # How often a frame is actually processed. The ZED publishes depth
            # at ~37 Hz and one frame costs ~90 ms, so most frames are dropped
            # on purpose; see _on_depth.
            ('max_rate', 5.0),
            ('diagnostics_period', 1.0),
        ):
            self.declare_parameter(name, default)

        # Spelled out one per line rather than looped, so test_params.py's
        # static scan of this method can check every one against params.yaml.
        # A parameter declared here and missing from the YAML silently falls
        # back to the default below, and the YAML stops being the single source
        # of truth without anything failing.
        for name, default in (
            ('pallet.pallet_height', 0.145),
            ('pallet.pocket_slab_half_depth', 0.06), ('pallet.pocket_frame_margin', 0.05),
            ('pallet.pocket_min_points_per_bin', 2),
            ('pallet.entry_face', 'short'),

            ('pallet.sides.short.face_width', 0.795),
            ('pallet.sides.short.depth', 1.195),
            ('pallet.sides.short.pocket_width', 0.230),
            ('pallet.sides.short.pocket_spacing', 0.365),
            ('pallet.sides.short.opening_bottom', 0.000),
            ('pallet.sides.short.opening_top', 0.100),
            ('pallet.sides.short.band_center', 0.055),
            ('pallet.sides.short.band_half_height', 0.020),
            ('pallet.sides.short.min_gap_width', 0.15),
            ('pallet.sides.short.max_gap_width', 0.32),
            ('pallet.sides.short.insertable', True),
            ('pallet.sides.short.insertion_stop_face_x', 0.700),

            ('pallet.sides.long.face_width', 1.195),
            ('pallet.sides.long.depth', 0.795),
            ('pallet.sides.long.pocket_width', 0.375),
            ('pallet.sides.long.pocket_spacing', 0.525),
            ('pallet.sides.long.opening_bottom', 0.020),
            ('pallet.sides.long.opening_top', 0.125),
            ('pallet.sides.long.band_center', 0.0725),
            ('pallet.sides.long.band_half_height', 0.025),
            ('pallet.sides.long.min_gap_width', 0.26),
            ('pallet.sides.long.max_gap_width', 0.50),
            ('pallet.sides.long.insertable', True),
            ('pallet.sides.long.insertion_stop_face_x', 0.950),

            ('pallet.classify.enabled', True),
            ('pallet.classify.band_center', 0.060),
            ('pallet.classify.band_half_height', 0.010),
            ('pallet.classify.min_gap_width', 0.15),
            ('pallet.classify.max_gap_width', 0.50),
            ('pallet.classify.spacing_tolerance', 0.06),
            ('pallet.classify.width_tolerance', 0.20),
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
            'max_height_above_pallet': get('cluster.max_height_above_pallet'),
            'max_candidates': get('cluster.max_candidates'),

            'face_distance_threshold': get('face.ransac_distance'),
            'max_tilt_from_vertical': get('face.max_tilt_from_vertical'),
            'face_min_points': get('face.min_points'),
            'face_deck_margin': get('face.deck_margin'),
            'floor_score_points': get('floor.score_points'),
            'max_facing_angle': get('face.max_facing_angle'),

            'pallet_height': get('pallet.pallet_height'),

            'pocket_slab_half_depth': get('pallet.pocket_slab_half_depth'),
            'pocket_frame_margin': get('pallet.pocket_frame_margin'),
            'pocket_min_points_per_bin': get('pallet.pocket_min_points_per_bin'),

            **self._side_params(),
        }

    # ROS parameters are a flat namespace of scalars -- there is no "dict
    # parameter" -- so the per-side geometry that params.yaml nests under
    # pallet.sides.short / .long arrives here as dotted scalar names. This
    # rebuilds the structure `estimate_pallet` wants.
    SIDE_KEYS = ('face_width', 'depth', 'pocket_width', 'pocket_spacing',
                 'opening_bottom', 'opening_top', 'band_center',
                 'band_half_height', 'min_gap_width', 'max_gap_width',
                 'insertion_stop_face_x')
    CLASSIFY_KEYS = ('enabled', 'band_center', 'band_half_height',
                     'min_gap_width', 'max_gap_width', 'spacing_tolerance',
                     'width_tolerance')

    def _side_params(self) -> dict:
        get = lambda name: self.get_parameter(name).value  # noqa: E731
        sides = {}
        for side in ('short', 'long'):
            cfg = {key: get(f'pallet.sides.{side}.{key}') for key in self.SIDE_KEYS}
            cfg['insertable'] = get(f'pallet.sides.{side}.insertable')
            sides[side] = cfg
        expected = get('pallet.entry_face')
        return {
            'sides': sides,
            'classify': {key: get(f'pallet.classify.{key}')
                         for key in self.CLASSIFY_KEYS},
            'entry_face': expected,
            **flatten_side(sides[expected]),
        }

    # ----------------------------------------------------------- lifecycle

    @logged_transition
    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self._tf_buffer = Buffer()
        self._tf_listener = CameraTransformListener(
            self._tf_buffer, self, self.get_parameter('frames.moving_links').value)
        self._publisher = self.create_lifecycle_publisher(
            PalletEstimateMsg, '~/estimate', 10)
        # Diagnostics are a few hundred bytes of text at 1 Hz, so they are safe
        # to read from another machine -- unlike anything carrying image or
        # cloud data, which must never cross the network (a single remote
        # subscriber to raw depth dropped the ZED node to 0.2 Hz).
        self._diag_pub = self.create_lifecycle_publisher(
            DiagnosticArray, '~/diagnostics', 1)
        # Two GO/NO-GO rules need points rather than an estimate --
        # `check_overhang` and `check_pocket_clearance` -- and both treat
        # missing data as a NO-GO rather than a pass, so without these the
        # coordinator refuses every run with "no points to check the load
        # against". The clouds are purpose-built and small: the pallet cluster
        # (a few thousand voxels) and a box around the face, not the ROI. They
        # stay on the Jetson, between two nodes a process apart; nothing
        # subscribes to them from the main PC and nothing should.
        self._cluster_pub = self.create_lifecycle_publisher(
            PointCloud2, '~/cluster_points', 1)
        self._scene_pub = self.create_lifecycle_publisher(
            PointCloud2, '~/scene_points', 1)
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

    @logged_transition
    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self._frames = 0
        self._warned = False
        self._latest = None
        self._gate.reset(self._now())
        # raw=True: the callback gets the serialized bytes, and only the frames
        # actually processed are deserialized. Converting a 1280x720 32FC1
        # image into a Python message costs 4.2 ms on a laptop core (measured
        # 2026-10-22), about twice that on an Orin; at the ZED's 37 Hz that is
        # 16-30 % of a core spent on frames the rate gate then drops.
        self._depth_sub = self.create_subscription(
            Image, self.get_parameter('topics.depth').value,
            self._on_depth, SENSOR_QOS, raw=True)
        period = self.get_parameter('diagnostics_period').value
        if period > 0:
            self._diag_timer = self.create_timer(period, self._publish_diagnostics)
        self.get_logger().info(
            f'active, processing at up to '
            f'{self.get_parameter("max_rate").value:.1f} Hz')
        return super().on_activate(state)

    @logged_transition
    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        if self._depth_sub is not None:
            self.destroy_subscription(self._depth_sub)
            self._depth_sub = None
        if self._diag_timer is not None:
            self.destroy_timer(self._diag_timer)
            self._diag_timer = None
        self.get_logger().info(
            f'inactive after {self._frames} estimates from '
            f'{self._gate.processed} frames ({self._gate.dropped} dropped)')
        return super().on_deactivate(state)

    @logged_transition
    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        if self._tf_listener is not None:
            self._tf_listener.close()
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

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_depth(self, msg: bytes):
        """Keep the newest frame; process at most `max_rate` times a second.

        Depth arrives at ~37 Hz and one frame costs ~90 ms, so the node cannot
        process every frame and must not try. The alternative -- letting rclpy
        queue them -- would not lose a single frame and would be far worse: the
        backlog grows about 26 frames a second, so within ten seconds the
        estimate being published describes where the pallet was seven seconds
        ago, while the robot acts on it. Dropping frames costs nothing here,
        because a stationary pallet looks the same in all of them.

        So: remember only the latest frame, and start work on it when the rate
        limit allows. What gets published is always the most recent thing the
        camera saw, never a backlog.
        """
        self._latest = msg
        self._gate.max_rate = self.get_parameter('max_rate').value
        if not self._gate.offer(self._now()):
            return
        try:
            self._process(deserialize_message(self._latest, Image))
        finally:
            self._gate.finished(self._now())

    def _process(self, msg: Image):
        # Every early return records WHY, and the reason goes out in
        # ~/diagnostics. Without it the symptom is "frames processed, nothing
        # published, frame_ms far too small", which names none of the half
        # dozen things that produce it -- the reason is only in the node's log,
        # throttled, on a machine you are probably not watching.
        if self._intrinsics is None:
            info = self.get_parameter('topics.depth_camera_info').value
            self._last_skip = f'no camera_info yet on {info}'
            self.get_logger().warning(
                f'no camera_info yet on {info}, skipping frame',
                throttle_duration_sec=5.0)
            return
        try:
            depth = image_to_array(msg)
        except ValueError as error:
            self._last_skip = f'undecodable depth image: {error}'
            self.get_logger().error(str(error))
            return

        # Is this frame's stamp even on the same clock we are? Checked before
        # the tf lookup, because a mismatch surfaces there as "extrapolation
        # into the past" with a plausible-looking requested time, and that
        # error names tf rather than the clock. Measured 2026-10-15 on the
        # robot: depth stamped 1835.8 (about half an hour, so an uptime clock)
        # against a tf buffer on Unix epoch, 1.79e9 seconds apart.
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._stamp_skew = self._now() - stamp
        mismatched = abs(self._stamp_skew) > CLOCK_MISMATCH_SECONDS
        if mismatched and not self.get_parameter('frames.tf_fallback_to_latest').value:
            self._last_skip = (
                f'clock mismatch: this frame is stamped {stamp:.3f} but our '
                f'clock reads {self._now():.3f}, {self._stamp_skew:.0f} s apart. '
                f'The image and /tf are on different clocks, so no stamped '
                f'lookup can succeed. Check use_sim_time on both sides and '
                f'whether the camera is stamping with its own clock. '
                f'frames.tf_fallback_to_latest:=true works around it for '
                f'bench testing only.')
            self.get_logger().error(self._last_skip, throttle_duration_sec=10.0)
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
            # Normally the lookup is STAMPED, because
            # zed2_camera_center -> zed2_left_camera_frame is republished at
            # 58 Hz by the SDK's self-calibration, so "latest" is not the same
            # as "at this frame".
            #
            # The exception is a clock mismatch, where the stamp cannot be used
            # at all and the only alternative to stopping is the latest
            # available. That is a BENCH setting, off by default:
            #
            #   harmless here  -- the one moving link is a stereo
            #       self-calibration that drifts by microradians, so "latest"
            #       and "0.2 s ago" are the same number;
            #   not harmless once the robot drives -- composing a pallet pose
            #       with odometry needs to know WHEN the pallet was seen, and
            #       a wrong stamp puts the standoff in the wrong place. It also
            #       makes every recorded bag unusable for replay.
            #
            # So: use it to measure the pipeline while parked, and fix the
            # clock before stage 3.
            lookup_time = (rclpy.time.Time() if mismatched
                           else msg.header.stamp)
            transform = lookup_waiting(self._tf_buffer, ground_frame, camera_frame, lookup_time, timeout)
        except Exception as error:
            self._last_skip = f'tf {ground_frame} <- {camera_frame}: {error}'
            self.get_logger().warning(
                f'tf {ground_frame} <- {camera_frame} failed: {error}',
                throttle_duration_sec=2.0)
            return

        camera_to_ground = transform_to_matrix(transform)
        if self._correction is not None:
            camera_to_ground = correct_transform(camera_to_ground, self._correction)

        decision_points = {}
        try:
            estimate = perception.estimate_pallet(
                depth, self._intrinsics, camera_to_ground,
                self._params(), stamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                decision_points=decision_points)
        except NotImplementedError:
            self._last_skip = 'estimate_pallet is not implemented'
            if not self._warned:
                self.get_logger().error(
                    'perception.estimate_pallet is not implemented yet -- that '
                    'file is yours to write. Run: pytest test/test_perception.py')
                self._warned = True
            return
        except Exception as error:  # a perception bug must not take the node down
            self._last_skip = f'perception raised {error!r}'
            self.get_logger().error(f'perception raised: {error!r}',
                                    throttle_duration_sec=2.0)
            return

        self._last_age = self._now() - stamp

        ground = self.get_parameter('frames.ground').value
        for key, publisher in (('cluster', self._cluster_pub),
                               ('scene', self._scene_pub)):
            points = decision_points.get(key)
            if points is not None:
                publisher.publish(points_to_cloud(points, ground, msg.header.stamp))

        if estimate is None:
            self._last_side = 'none'
            self._last_skip = 'no pallet in frame (pipeline ran to the end)'
            return

        self._last_side = estimate.side
        self._last_skip = ''
        self._frames += 1
        header = msg.header
        header.frame_id = ground_frame
        self._publisher.publish(estimate_to_msg(estimate, header))

    def _publish_diagnostics(self):
        """Rate, timing and drops, as a small text message.

        Published because the bug that matters here is invisible otherwise: a
        pipeline that has quietly fallen to 1 Hz still publishes estimates that
        look perfectly reasonable one at a time. The recording script learned
        the same lesson -- eight bags were recorded at the wrong rate because
        nothing ever printed the achieved one.
        """
        if self._diag_pub is None:
            return
        now = self._now()
        gate = self._gate
        status = DiagnosticStatus(
            name='pallet_perception', hardware_id='zed2',
            level=DiagnosticStatus.OK, message=f'side={self._last_side}')
        if gate.processed == 0:
            status.level = DiagnosticStatus.WARN
            topic = self.get_parameter('topics.depth').value
            publishers = self.count_publishers(topic)
            if gate.dropped == 0:
                # Nothing even arrived. Say which topic and whether anyone is
                # publishing on it, because those are two different faults with
                # two different fixes and "no frames processed" names neither.
                status.message = f'no depth received on {topic}'
                self.get_logger().warning(
                    f'active but no depth on {topic} -- {publishers} publisher(s) '
                    f'visible from here. '
                    + ('Nobody is publishing: is the ZED wrapper running? '
                       if publishers == 0 else
                       'Someone is publishing but nothing arrives: QoS or DDS '
                       'transport mismatch. '),
                    throttle_duration_sec=10.0)
            else:
                # Frames arrived and the gate refused all of them, which at a
                # sane max_rate means the clock is not advancing -- use_sim_time
                # with no /clock being the usual cause.
                status.message = (f'{gate.dropped} frames arrived and all were '
                                  f'dropped; is the clock advancing?')
                self.get_logger().warning(
                    f'{gate.dropped} depth frames dropped and none processed. '
                    f'max_rate={self.get_parameter("max_rate").value}, '
                    f'clock now={now:.3f}. If now is not advancing, check '
                    f'use_sim_time.', throttle_duration_sec=10.0)
            status.values.append(KeyValue(key='depth_topic', value=topic))
            status.values.append(KeyValue(key='depth_publishers',
                                          value=str(publishers)))
        if self._intrinsics is None:
            status.level = max(status.level, DiagnosticStatus.WARN)
            info_topic = self.get_parameter('topics.depth_camera_info').value
            status.values.append(KeyValue(
                key='camera_info',
                value=f'still waiting on {info_topic} '
                      f'({self.count_publishers(info_topic)} publisher(s))'))
        for key, value in (
            ('rate_hz', f'{gate.rate(now):.2f}'),
            ('frame_ms', f'{gate.frame_seconds * 1e3:.1f}'),
            ('estimate_age_ms', f'{self._last_age * 1e3:.0f}'),
            ('processed', str(gate.processed)),
            ('dropped', str(gate.dropped)),
            # Of frames DELIVERED to this node, not of frames the camera
            # produced. The subscription is KEEP_LAST depth 1, so while a frame
            # is being worked on the middleware overwrites the queue and those
            # frames never reach the callback at all -- they are invisible here.
            # Measured 2026-10-15: the camera published ~56 Hz while this node
            # saw ~13 callbacks a second. Compare delivered_hz below against
            # `ros2 topic hz` on the depth topic to see that upstream loss.
            ('drop_fraction_of_delivered', f'{gate.drop_fraction():.2f}'),
            ('delivered_hz', f'{(gate.processed + gate.dropped) / max(now - (gate.window_start or now), 1e-6):.1f}'),
            ('estimates_published', str(self._frames)),
            ('side', self._last_side),
            ('last_skip', self._last_skip or 'none, estimates publishing'),
            ('stamp_skew_s', f'{self._stamp_skew:.3f}'),
            ('tf_mode', 'LATEST (clock mismatch, bench only)'
                        if abs(self._stamp_skew) > CLOCK_MISMATCH_SECONDS
                        else 'stamped'),
        ):
            status.values.append(KeyValue(key=key, value=value))
        if self._last_skip:
            status.level = max(status.level, DiagnosticStatus.WARN)
            status.message = self._last_skip
        array = DiagnosticArray()
        array.header.stamp = self.get_clock().now().to_msg()
        array.status.append(status)
        self._diag_pub.publish(array)
        gate.reset(now)


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
