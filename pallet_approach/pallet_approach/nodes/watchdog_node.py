"""The watchdog: reports whether the stack is healthy, and stops the robot.

1. Publishes `/health`, so the coordinator refuses (D12) or aborts (D9) runs.
2. While a run is in progress, publishes zeros on twist_mux's safety input,
   which outranks the controller and Nav2. This depends on neither the
   coordinator being alive nor correct; the overlap is deliberate.

Camera liveness comes from `depth/camera_info`, never the depth image. The
LiDAR subscription is BEST_EFFORT (a RELIABLE one receives nothing). The stop
latches until the run ends, so a flickering sensor cannot stutter the robot.
"""

from __future__ import annotations

import dataclasses

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, PointCloud2
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from pallet_approach.nodes._common import lookup_waiting, tracked_topic
from pallet_approach_interfaces.msg import Health, TrackedPallet
from pallet_approach_interfaces.msg import PalletEstimate as PalletEstimateMsg

from pallet_approach.health import (
    PERCEPTION_STATES,
    phase_for_state,
    ALWAYS,
    NEVER,
    WHILE_RUNNING,
    HealthMonitor,
    Limits,
    as_float,
)
from pallet_approach.ros_conversions import stamp_to_seconds

RELIABLE = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
BEST_EFFORT = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)

NOT_RUNNING = ('', 'IDLE', 'DONE', 'ABORTED')
"""Coordinator states in which no run is in progress.

Listed as the complement rather than as the driving states, so that a state
added to the machine later is covered by the watchdog from the moment it
exists. Getting this list wrong in the safe direction means arming the watchdog
while the robot is parked, which costs nothing.
"""

SOURCES = ('camera_info', 'estimate', 'odom', 'lidar', 'tf')


class WatchdogNode(Node):

    def __init__(self):
        super().__init__('watchdog')
        self._declare()
        self._state = ''
        self._latched = False
        self._latched_reason = ''
        self._report = None
        self._running_since = None

        self._tf_required = not self.get_parameter('frames.tf_fallback_to_latest').value
        self._monitor = HealthMonitor(self._limits(), self.get_parameter('window').value)
        if not self._tf_required:
            self.get_logger().warning(
                'frames.tf_fallback_to_latest is true, so the tf age is MEASURED '
                'BUT NOT ENFORCED. That is a bench setting for a camera whose '
                'clock is wrong; it must be false for anything that drives.')

        self._health_pub = self.create_publisher(Health, '/health', RELIABLE)
        self._diag_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 10)
        self._stop_pub = self.create_publisher(
            Twist, self.get_parameter('topics.cmd_vel_safety').value, RELIABLE)

        self.create_subscription(CameraInfo,
                                 self.get_parameter('topics.depth_camera_info').value,
                                 lambda _: self._seen('camera_info'), RELIABLE)
        # `estimate` = perception is alive and producing during the run. With
        # the fused perception (D104) that is pallet_fusion's tracked pallet:
        # the old perception node is never activated, and requiring its
        # estimate aborted the first fused sim run at once.
        if self.get_parameter('perception_source').value == 'fused':
            self.create_subscription(TrackedPallet, tracked_topic('fused'),
                                     lambda _: self._seen('estimate'), RELIABLE)
        else:
            self.create_subscription(PalletEstimateMsg, '/pallet_perception/estimate',
                                     lambda _: self._seen('estimate'), RELIABLE)
        self.create_subscription(Odometry, self.get_parameter('topics.odom').value,
                                 lambda _: self._seen('odom'), RELIABLE)
        # raw=True: only the arrival is timed, so the ~1.6 MB cloud is never
        # decoded here.
        self.create_subscription(PointCloud2, self.get_parameter('topics.lidar').value,
                                 lambda _: self._seen('lidar'), BEST_EFFORT, raw=True)
        self.create_subscription(String, '/coordinator/state', self._on_state, RELIABLE)

        self._buffer = Buffer()
        self._listener = TransformListener(self._buffer, self, spin_thread=True)

        self.create_timer(1.0 / float(self.get_parameter('publish_rate').value),
                          self._publish_health)
        self.create_timer(1.0 / float(self.get_parameter('command_rate').value),
                          self._publish_stop)
        self.get_logger().info(
            f'watching {", ".join(SOURCES)}; stop goes to '
            f'{self.get_parameter("topics.cmd_vel_safety").value}')

    # ------------------------------------------------------------- params

    def _declare(self):
        self.declare_parameter('perception_source', 'fused')
        self.declare_parameter('topics.depth_camera_info',
                               '/zed2/zed_node/depth/camera_info')
        self.declare_parameter('topics.odom', '/odom')
        self.declare_parameter('topics.lidar', '/pallet_approach/lidar')
        self.declare_parameter('topics.cmd_vel_safety', '/cmd_vel_safety')
        self.declare_parameter('frames.ground', 'base_footprint')
        self.declare_parameter('frames.camera_optical',
                               'zed2_left_camera_frame_optical')
        self.declare_parameter('frames.tf_fallback_to_latest', False)
        for name, default in (
            ('publish_rate', 2.0), ('command_rate', 20.0), ('window', 2.0),
            ('max_age.camera_info', 0.30), ('min_rate.camera_info', 10.0),
            ('max_age.estimate', 1.00), ('min_rate.estimate', 2.5),
            ('max_age.odom', 0.20), ('min_rate.odom', 20.0),
            ('max_age.lidar', 0.50), ('min_rate.lidar', 5.0),
            ('max_age.tf', 0.30), ('first_estimate_grace', 2.0),
            ('estimate_required_states', list(PERCEPTION_STATES)),
        ):
            self.declare_parameter(name, default)

    def _limits(self) -> dict:
        """D9's staleness limits, with the rate floors from params.yaml.

        `estimate` is the only WHILE_RUNNING source: perception is a managed
        lifecycle node and publishes nothing between runs, so requiring it
        always would make the stack permanently unhealthy and D12's goal-time
        check would refuse every goal.
        """
        def limits(name, when=ALWAYS):
            rate = f'min_rate.{name}'
            floor = (self.get_parameter(rate).value
                     if self.has_parameter(rate) else 0.0)
            return Limits(max_age=float(self.get_parameter(f'max_age.{name}').value),
                          min_rate=float(floor), when=when)
        return {
            'camera_info': limits('camera_info'),
            'estimate': dataclasses.replace(
                limits('estimate', WHILE_RUNNING),
                first_message_grace=float(
                    self.get_parameter('first_estimate_grace').value)),
            'odom': limits('odom'),
            'lidar': limits('lidar'),
            # Reported either way; enforced only when the stamped-lookup path
            # is the one in use. See the warning logged in __init__.
            'tf': limits('tf', ALWAYS if self._tf_required else NEVER),
        }

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # ---------------------------------------------------------- callbacks

    def _seen(self, name: str):
        self._monitor.seen(name, self._now())

    def _on_state(self, msg: String):
        was_running = self._state not in NOT_RUNNING
        self._state = msg.data
        running = self._state not in NOT_RUNNING
        if running and not was_running:
            self._running_since = self._now()
            # Forgotten on BOTH edges. Perception is deactivated after the
            # coordinator leaves the run, so its last estimate can land after
            # the forget below; measured 2026-10-23, the next run then aborted
            # in its first 0.2 s on "estimate: 9.07 s old" instead of getting
            # first_estimate_grace.
            self._monitor.forget('estimate')
        if was_running and not running:
            self._running_since = None
            # The run is over. Release the stop and forget the estimates it
            # produced: last run's perception is not evidence about this one.
            if self._latched:
                self.get_logger().info('run ended; releasing the safety stop')
            self._latched = False
            self._latched_reason = ''
            self._monitor.forget('estimate')

    def _poll_tf(self):
        """Age of the newest ground <- camera_optical transform.

        Recorded as an arrival rather than measured as an age, so one code path
        in `health.py` judges every source. The transform's own stamp is what
        is timed, not the moment we looked: the chain includes the ZED SDK's
        stereo self-calibration, republished at 58 Hz on the CAMERA's clock, so
        this is the check that catches the clock skew that cost an evening on
        2026-10-15 (see tools/clock_audit.py).
        """
        ground = self.get_parameter('frames.ground').value
        optical = self.get_parameter('frames.camera_optical').value
        try:
            transform = lookup_waiting(self._buffer, ground, optical, rclpy.time.Time())
        except Exception:
            return
        self._monitor.seen('tf', stamp_to_seconds(transform.header.stamp))

    # ------------------------------------------------------------ publish

    def _publish_health(self):
        self._poll_tf()
        now = self._now()
        running = self._state not in NOT_RUNNING
        # D64: the estimate is required only in the states that use the camera.
        phase = phase_for_state(
            self._state, self.get_parameter('estimate_required_states').value)
        report = self._monitor.check(now, phase,
                                     running_since=self._running_since)
        self._report = report

        if running and not report.ok and not self._latched:
            self._latched = True
            self._latched_reason = report.summary()
            self.get_logger().error(
                f'SAFETY STOP in {self._state or "a run"}: {self._latched_reason}')

        message = Health()
        message.header.stamp = self.get_clock().now().to_msg()
        message.ok = bool(report.ok)
        message.phase = report.phase
        message.failures = list(report.failures)
        message.names = [s.name for s in report.sources]
        message.ages = [as_float(s.age) for s in report.sources]
        message.rates = [as_float(s.rate) for s in report.sources]
        message.required = [bool(s.required) for s in report.sources]
        message.source_ok = [bool(s.ok) for s in report.sources]
        message.stop_latched = bool(self._latched)
        self._health_pub.publish(message)
        self._diag_pub.publish(self._diagnostics(report))

        if not report.ok and not running:
            self.get_logger().warning(f'unhealthy while idle: {report.summary()}',
                                      throttle_duration_sec=10.0)

    def _publish_stop(self):
        """Zero on the mux's highest-priority input, at the controller's rate.

        Published repeatedly rather than once: the mux drops an input that goes
        quiet for 0.5 s, and letting it drop would hand the base straight back
        to whichever node is still talking -- which, if the coordinator has
        wedged, is the controller we are trying to override.
        """
        if self._latched:
            self._stop_pub.publish(Twist())

    def _diagnostics(self, report) -> DiagnosticArray:
        array = DiagnosticArray()
        array.header.stamp = self.get_clock().now().to_msg()
        for source in report.sources:
            status = DiagnosticStatus()
            status.name = f'pallet_approach: {source.name}'
            status.hardware_id = 'pallet_approach'
            if source.ok:
                status.level = DiagnosticStatus.OK
                status.message = 'ok'
            elif source.required:
                status.level = DiagnosticStatus.ERROR
                status.message = source.detail
            else:
                status.level = DiagnosticStatus.WARN
                status.message = f'{source.detail} (not required in this phase)'
            status.values = [
                KeyValue(key='age_s', value=f'{as_float(source.age):.3f}'),
                KeyValue(key='rate_hz', value=f'{as_float(source.rate):.1f}'),
                KeyValue(key='required', value=str(source.required)),
            ]
            array.status.append(status)
        overall = DiagnosticStatus()
        overall.name = 'pallet_approach: stack'
        overall.hardware_id = 'pallet_approach'
        overall.level = (DiagnosticStatus.OK if report.ok
                         else DiagnosticStatus.ERROR)
        overall.message = report.summary()
        overall.values = [
            KeyValue(key='phase', value=report.phase),
            KeyValue(key='coordinator_state', value=self._state or 'unknown'),
            KeyValue(key='stop_latched', value=str(self._latched)),
            KeyValue(key='latched_reason', value=self._latched_reason),
        ]
        array.status.append(overall)
        return array


def main(args=None):
    rclpy.init(args=args)
    node = WatchdogNode()
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
