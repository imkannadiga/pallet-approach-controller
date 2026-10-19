"""Lifecycle node: the only thing in this stack that moves the robot.

It executes one of four behaviours at a time, chosen by the coordinator over
`~/drive`, and nothing else:

    CLOSE     servo onto the pocket centre line, stop at the standoff (D38)
    REVERSE   straight back, with the rear LiDAR check and the travelled-ground
              allowance (D42)
    INSERT    straight forward, stopping on the earlier of odometry and the
              LiDAR, aborting if they disagree (D18)
    BACKOUT   straight back out of the pallet

The division of labour is deliberate: the coordinator says WHICH behaviour and
how far, and this node owns HOW -- the gains, the limits and the stop rules. So
the sequencing can be tested without a controller and the control laws without
a state machine, which is why `closing.py`, `insertion.py` and `coordinator.py`
have no ROS in them at all.

Five safety properties live here and must not be weakened:

1. **No publishing unless activated.** The publisher exists from `configure`,
   but `_active` gates every write and only the lifecycle state sets it.
2. **Fixed rate, explicit zeros.** The base driver has no command timeout, so a
   command it already has keeps it moving. The timer publishes every tick while
   active, and deactivating sends several explicit zeros.
3. **Stale input means stop.** A drive command older than `target_timeout`, or
   a pallet estimate older than it during CLOSE, stops the robot rather than
   acting on a measurement that may no longer be true.
4. **Somebody is listening.** At configure it checks something subscribes to
   its command topic and refuses otherwise, because with twist_mux absent every
   command would vanish and the stack would report that it was driving (D31).
5. **It never invents a mode.** An unknown mode string stops the robot and says
   so. There is no default behaviour.
"""

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer, TransformListener

from pallet_approach_interfaces.msg import DriveCommand, DriveStatus, TrackedPallet

from pallet_approach import closing, insertion
from pallet_approach.controller import ZERO, Command, RateLimiter
from pallet_approach.pointcloud import transform_points, unpack_cloud
from pallet_approach.ros_conversions import (
    command_to_twist,
    estimate_from_msg,
    pose2d_from_odometry,
    stamp_to_seconds,
    transform_to_matrix,
)

RELIABLE = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
BEST_EFFORT = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)

MODES = ('STOP', 'CLOSE', 'REVERSE', 'INSERT', 'BACKOUT')


class ControllerNode(LifecycleNode):

    def __init__(self):
        super().__init__('approach_controller')
        self._declare()
        self._active = False
        self._pose = None
        self._mode = 'STOP'
        self._command = None
        self._command_at = 0.0
        self._tracked = None
        self._tracked_at = 0.0
        self._cloud = None
        self._limiter = None
        self._timer = None
        self._cmd_pub = None
        self._status_pub = None
        self._last_tick = None
        self._buffer = None
        self._listener = None

        # per-move state, reset on every new command
        self._start_pose = None
        self._travelled = 0.0
        self._phase = closing.Phase.CLOSING
        self._plan = None
        self._unmeasured = 0
        self._divergence = None
        self._done = False
        self._abort = False
        self._reason = 'idle'

    # ------------------------------------------------------------- params

    def _declare(self):
        self.declare_parameter('topics.cmd_vel_pallet', '/cmd_vel_pallet')
        self.declare_parameter('topics.odom', '/odom')
        self.declare_parameter('topics.lidar', '/ouster/points')
        self.declare_parameter('frames.ground', 'base_footprint')
        self.declare_parameter('frames.tf_timeout', 0.15)
        self.declare_parameter('floor_z', -0.025)
        self.declare_parameter('standoff_distance', 1.75)
        self.declare_parameter('robot.width', 0.6996)
        self.declare_parameter('robot.rear_bumper_x', -0.4625)
        self.declare_parameter('forks.carriage_x', 0.4625)
        self.declare_parameter('forks.tine_length', 1.150)
        for name, default in (
            ('limits.max_linear', 0.20), ('limits.max_angular', 0.30),
            ('limits.max_linear_accel', 0.30), ('limits.max_angular_accel', 0.60),
            ('limits.min_linear', 0.05), ('limits.min_angular', 0.10),
            ('closing.k_cross', 2.5), ('closing.k_heading', 4.0),
            ('closing.max_approach_angle', 0.25),
            ('closing.straighten_offset', 0.015),
            ('closing.max_turn_in_place', 0.17),
            ('closing.closing_speed', 0.10), ('closing.taper_range', 0.30),
            ('closing.range_tolerance', 0.03), ('closing.arrival_yaw', 0.035),
            ('closing.resume_range', 0.10), ('closing.min_run_up', 3.0),
            ('closing.residual_target', 0.020), ('closing.run_up_margin', 0.20),
            ('closing.divergence_frames', 5),
            ('closing.min_divergence_growth', 0.010),
            ('reverse.speed', 0.05), ('reverse.max_unverified', 0.50),
            ('reverse.side_margin', 0.10), ('reverse.min_height', 0.03),
            ('reverse.max_height', 1.00), ('reverse.clearance_margin', 0.15),
            ('insert.speed', 0.05), ('insert.stop_tolerance', 0.010),
            ('insert.max_disagreement', 0.05), ('insert.max_travel', 1.20),
            ('insert.max_unmeasured_frames', 5),
            ('insert.lidar.half_width', 0.30), ('insert.lidar.min_height', 0.03),
            ('insert.lidar.max_height', 0.20), ('insert.lidar.min_x', 0.50),
            ('insert.lidar.min_points', 10),
            ('pallet.sides.short.insertion_stop_face_x', 0.700),
            ('pallet.sides.short.depth', 1.195),
            ('pallet.sides.short.insertable', True),
            ('pallet.sides.long.insertion_stop_face_x', 0.950),
            ('pallet.sides.long.depth', 0.795),
            ('pallet.sides.long.insertable', True),
            ('command_rate', 20.0), ('target_timeout', 0.50),
            ('zero_command_repeats', 5),
        ):
            self.declare_parameter(name, default)

    def _get(self, name):
        return self.get_parameter(name).value

    def _group(self, prefix, keys):
        return {key: self._get(f'{prefix}.{key}') for key in keys}

    def _limits(self) -> dict:
        return self._group('limits', ('max_linear', 'max_angular',
                                      'min_linear', 'min_angular'))

    def _closing_params(self) -> dict:
        params = self._group('closing', (
            'k_cross', 'k_heading', 'max_approach_angle', 'straighten_offset',
            'max_turn_in_place', 'closing_speed', 'taper_range',
            'range_tolerance', 'arrival_yaw', 'resume_range', 'min_run_up',
            'residual_target', 'run_up_margin'))
        params['target_range'] = self._get('standoff_distance')
        params['carriage_x'] = self._get('forks.carriage_x')
        params['tine_length'] = self._get('forks.tine_length')
        return params

    def _reverse_params(self) -> dict:
        params = self._group('reverse', (
            'speed', 'max_unverified', 'side_margin', 'min_height',
            'max_height', 'clearance_margin'))
        params['rear_bumper_x'] = self._get('robot.rear_bumper_x')
        params['half_width'] = self._get('robot.width') / 2.0
        params['floor_z'] = self._get('floor_z')
        return params

    def _insert_params(self) -> dict:
        params = self._group('insert', (
            'speed', 'stop_tolerance', 'max_disagreement', 'max_travel',
            'max_unmeasured_frames'))
        params['lidar'] = self._group('insert.lidar', (
            'half_width', 'min_height', 'max_height', 'min_x', 'min_points'))
        params['lidar']['floor_z'] = self._get('floor_z')
        params['carriage_x'] = self._get('forks.carriage_x')
        params['tine_length'] = self._get('forks.tine_length')
        params['sides'] = {
            side: self._group(f'pallet.sides.{side}', (
                'insertion_stop_face_x', 'depth', 'insertable'))
            for side in ('short', 'long')
        }
        return params

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # ----------------------------------------------------------- lifecycle

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        topic = self._get('topics.cmd_vel_pallet')
        self._cmd_pub = self.create_lifecycle_publisher(Twist, topic, RELIABLE)
        self._status_pub = self.create_lifecycle_publisher(
            DriveStatus, '~/status', RELIABLE)
        self.create_subscription(Odometry, self._get('topics.odom'),
                                 self._on_odom, RELIABLE)
        self.create_subscription(DriveCommand, '~/drive', self._on_drive, RELIABLE)
        self.create_subscription(TrackedPallet, '/pallet_tracker/tracked',
                                 self._on_tracked, RELIABLE)
        self.create_subscription(PointCloud2, self._get('topics.lidar'),
                                 self._on_cloud, BEST_EFFORT)

        self._buffer = Buffer()
        self._listener = TransformListener(self._buffer, self)

        if not self._wait_for_a_subscriber(topic):
            self.get_logger().error(
                f'nothing subscribes to {topic}, so every command this node '
                f'sends would be discarded and the robot would never move. '
                f'twist_mux is what should be listening -- start it with the '
                f'stack (launch/pallet_approach.launch.py brings it up) or '
                f'check config/twist_mux.yaml names this topic. Refusing to '
                f'configure.')
            return TransitionCallbackReturn.FAILURE

        others = self.count_publishers(topic) - 1
        if others > 0:
            self.get_logger().warning(
                f'{others} other node(s) also publish {topic}, which is meant '
                f'to be this node\'s private input to twist_mux. Arbitration '
                f'cannot separate two publishers on one input.')

        self._limiter = RateLimiter(self._get('limits.max_linear_accel'),
                                    self._get('limits.max_angular_accel'))
        self.get_logger().info(
            f'configured; commands go to {topic} '
            f'({self.count_subscribers(topic)} subscriber(s)); '
            f'max {self._get("limits.max_linear")} m/s, '
            f'{self._get("limits.max_angular")} rad/s')
        return TransitionCallbackReturn.SUCCESS

    def _wait_for_a_subscriber(self, topic: str, timeout: float = 2.0) -> bool:
        """Is anything listening on `topic`? With a short wait, because
        discovery is not instant -- but an answer either way, never an
        assumption."""
        deadline = self._now() + timeout
        while self._now() < deadline:
            if self.count_subscribers(topic) > 0:
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.count_subscribers(topic) > 0

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self._reset('STOP', 'activated, waiting for a drive command')
        self._command = None
        self._limiter.reset()
        self._last_tick = None
        self._active = True
        period = 1.0 / float(self._get('command_rate'))
        self._timer = self.create_timer(period, self._tick)
        self.get_logger().info('active')
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self._active = False
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._stop()
        self.get_logger().info('inactive, zero command sent')
        return super().on_deactivate(state)

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self._active = False
        self._stop()
        return TransitionCallbackReturn.SUCCESS

    on_error = on_shutdown

    # ------------------------------------------------------------- commands

    def _stop(self):
        """Publish zeros, several times, bypassing the active gate.

        The one place that publishes while not active, and it can only ever
        send zero. Repeated because the command crosses the network and a
        single dropped zero would leave the base moving.
        """
        if self._cmd_pub is None:
            return
        twist = Twist()
        for _ in range(int(self._get('zero_command_repeats'))):
            self._cmd_pub.publish(twist)
        if self._limiter is not None:
            self._limiter.reset()

    def _publish(self, command: Command):
        if self._active and self._cmd_pub is not None:
            self._cmd_pub.publish(command_to_twist(command))

    # ------------------------------------------------------------ callbacks

    def _on_odom(self, msg: Odometry):
        """Accumulate PATH LENGTH, not displacement.

        A move that curves covers more ground than it displaces, and the
        insertion stop is about how far the wheels turned. Publishing both
        (DriveStatus.travelled and .displacement) means a growing gap between
        them shows up as the robot curving when it was told not to.
        """
        pose = pose2d_from_odometry(msg)
        if self._pose is not None:
            self._travelled += math.hypot(pose.x - self._pose.x,
                                          pose.y - self._pose.y)
        self._pose = pose

    def _on_tracked(self, msg: TrackedPallet):
        estimate = estimate_from_msg(msg.estimate)
        if estimate is None:
            return
        self._tracked = (estimate, bool(msg.confirmed))
        self._tracked_at = self._now()

    def _on_cloud(self, msg: PointCloud2):
        self._cloud = msg

    def _on_drive(self, msg: DriveCommand):
        self._command_at = self._now()
        if msg.mode == self._mode and self._command is not None:
            # A repeat of the move in progress, which is how the coordinator
            # keeps the command fresh. Refreshing the timestamp is all it does;
            # restarting the move would reset `travelled` every frame and the
            # robot would drive for ever.
            return
        if msg.mode not in MODES:
            self._reset('STOP', f'unknown drive mode {msg.mode!r}')
            self._abort = True
            self.get_logger().error(
                f'unknown drive mode {msg.mode!r}; stopping. Known: '
                f'{", ".join(MODES)}')
            return
        self._command = msg
        self._reset(msg.mode, f'{msg.mode} requested')
        self.get_logger().info(
            f'{msg.mode}' + (f' {msg.distance:.3f} m' if msg.distance else '')
            + (f' into the {msg.side} face' if msg.side else ''))

    def _reset(self, mode: str, reason: str):
        self._mode = mode
        self._start_pose = self._pose
        self._travelled = 0.0
        self._phase = closing.Phase.CLOSING
        self._plan = None
        self._unmeasured = 0
        self._done = False
        self._abort = False
        self._reason = reason
        self._divergence = closing.DivergenceDetector(
            self._get('closing.divergence_frames'),
            self._get('closing.min_divergence_growth'))
        if self._limiter is not None:
            self._limiter.reset()

    # --------------------------------------------------------------- lidar

    def _lidar_in_ground(self):
        """The newest cloud in `frames.ground`, or None with a logged reason."""
        cloud = self._cloud
        if cloud is None:
            return None
        ground = self._get('frames.ground')
        try:
            transform = self._buffer.lookup_transform(
                ground, cloud.header.frame_id, cloud.header.stamp,
                rclpy.duration.Duration(seconds=float(self._get('frames.tf_timeout'))))
        except Exception as error:
            self.get_logger().warning(f'tf {ground} <- {cloud.header.frame_id}: '
                                      f'{error}', throttle_duration_sec=5.0)
            return None
        return transform_points(unpack_cloud(cloud), transform_to_matrix(transform))

    # ---------------------------------------------------------------- tick

    def _tick(self):
        if not self._active:
            return
        now = self._now()
        dt = 0.0 if self._last_tick is None else now - self._last_tick
        self._last_tick = now

        status = DriveStatus()
        status.header.stamp = self.get_clock().now().to_msg()
        status.mode = self._mode
        status.phase = self._phase.value if self._mode == 'CLOSE' else ''
        status.face_range = float('nan')
        status.lateral_offset = float('nan')
        status.heading_error = float('nan')
        status.rear_clearance = float('nan')

        command = self._decide(now, status)

        status.arrived = bool(self._done)
        status.abort = bool(self._abort)
        status.blocked = bool(command.is_zero() and not self._done
                              and self._mode != 'STOP')
        status.reason = self._reason
        status.travelled = float(self._travelled)
        status.displacement = float(self._displacement())
        self._status_pub.publish(status)

        self._publish(self._limiter.limit(command, dt) if dt > 0.0 else ZERO)

    def _displacement(self) -> float:
        if self._pose is None or self._start_pose is None:
            return 0.0
        return math.hypot(self._pose.x - self._start_pose.x,
                          self._pose.y - self._start_pose.y)

    def _decide(self, now: float, status) -> Command:
        """Work out this tick's command. Returns ZERO for every refusal."""
        if self._mode == 'STOP' or self._command is None:
            self._reason = self._reason if self._mode == 'STOP' else 'no command'
            return ZERO

        if now - self._command_at > float(self._get('target_timeout')):
            self._reason = (f'drive command is {now - self._command_at:.2f} s old; '
                            f'stopping')
            self.get_logger().warning(self._reason, throttle_duration_sec=1.0)
            return ZERO

        if self._done or self._abort:
            return ZERO

        if self._pose is None:
            self._reason = 'no odometry yet'
            return ZERO

        if self._start_pose is None:
            # The move was requested before any odometry had arrived, so it has
            # no origin to measure from. Take this pose as the origin rather
            # than measuring the travel from wherever the robot happened to be
            # when the node woke up.
            self._start_pose = self._pose
            self._travelled = 0.0

        if self._mode == 'CLOSE':
            return self._close(now, status)
        if self._mode in ('REVERSE', 'BACKOUT'):
            return self._straight(status)
        if self._mode == 'INSERT':
            return self._insert(status)

        self._reason = f'unhandled mode {self._mode!r}'
        self._abort = True
        return ZERO

    def _close(self, now: float, status) -> Command:
        if self._tracked is None:
            self._reason = 'no pallet estimate yet'
            return ZERO
        age = now - self._tracked_at
        if age > float(self._get('target_timeout')):
            self._reason = f'pallet estimate is {age:.2f} s old; stopping'
            self.get_logger().warning(self._reason, throttle_duration_sec=1.0)
            return ZERO
        estimate, confirmed = self._tracked
        if not confirmed:
            # Driving on an unconfirmed track is exactly what the tracker
            # exists to prevent.
            self._reason = 'the track is not confirmed'
            return ZERO

        params = self._closing_params()
        geometry = closing.geometry_of(estimate)
        status.face_range = float(geometry.range)
        status.lateral_offset = float(geometry.offset)
        status.heading_error = float(geometry.heading)

        diverging, detail = self._divergence.update(geometry.offset)
        if diverging:
            self._reason = f'diverging: {detail}'
            self._abort = True
            self.get_logger().error(self._reason)
            return ZERO

        command, phase, reason = closing.closing_command(
            geometry, self._phase, params, self._limits())
        self._phase = phase
        self._reason = reason
        if phase is closing.Phase.ARRIVED:
            self._done = True
            self.get_logger().info(reason)
        return command

    def _straight(self, status) -> Command:
        distance = -abs(float(self._command.distance))
        params = self._reverse_params()

        points = self._lidar_in_ground()
        clearance, detail = closing.rear_clearance(points, params)
        if clearance is not None:
            status.rear_clearance = float(clearance)
            remaining = abs(distance) - self._travelled
            if clearance < remaining + float(params['clearance_margin']):
                self._reason = (f'obstacle {clearance:.2f} m behind and '
                                f'{remaining:.2f} m still to reverse: {detail}')
                self._abort = True
                self.get_logger().error(self._reason)
                return ZERO

        command, done, reason = closing.straight_command(
            self._travelled, distance, float(params['speed']), self._limits())
        self._reason = reason
        if done:
            self._done = True
            self.get_logger().info(f'{self._mode} complete: {reason}')
        return command

    def _insert(self, status) -> Command:
        params = self._insert_params()

        if self._plan is None:
            face = self._measured_face(params)
            if face is None:
                self._reason = 'cannot measure the face to plan the insertion'
                return ZERO
            try:
                self._plan = insertion.plan_insertion(
                    face, self._command.side or 'short', params)
            except ValueError as error:
                self._reason = f'insertion refused: {error}'
                self._abort = True
                self.get_logger().error(self._reason)
                return ZERO
            self.get_logger().info(
                f'inserting {self._plan.travel:.3f} m into the '
                f'{self._plan.side} face, stopping with it at '
                f'{self._plan.stop_face_x:.3f} m, tips {self._plan.depth:.3f} m in')

        points = self._lidar_in_ground()
        face_x, lateral, detail = insertion.face_distance_ahead(
            points, params['lidar'])
        self._unmeasured = 0 if face_x is not None else self._unmeasured + 1
        if face_x is not None:
            status.face_range = float(face_x)
            status.lateral_offset = float(lateral)

        decision = insertion.stop_decision(
            self._travelled, face_x, self._unmeasured, self._plan, params)
        self._reason = decision.reason
        if decision.abort:
            self._abort = True
            self._done = True
            self.get_logger().error(f'insertion aborted: {decision.reason}')
            return ZERO
        if decision.stop:
            self._done = True
            self.get_logger().info(f'insertion complete: {decision.reason}')
            return ZERO

        command, _, _ = closing.straight_command(
            self._travelled, self._plan.travel, float(params['speed']),
            self._limits())
        return command

    def _measured_face(self, params) -> float | None:
        """Face distance at the moment the insertion starts.

        The LiDAR first, because it is the sensor that will be watching for the
        whole drive and planning from a different source than the one that
        stops you is how a 50 mm disagreement becomes a surprise. Perception's
        estimate is the fallback, and if neither has an answer the insertion is
        not planned at all.
        """
        points = self._lidar_in_ground()
        face_x, _, detail = insertion.face_distance_ahead(points, params['lidar'])
        if face_x is not None:
            return face_x
        self.get_logger().warning(
            f'no LiDAR face measurement to plan from ({detail}); falling back '
            f'to the camera estimate', throttle_duration_sec=2.0)
        if self._tracked is None:
            return None
        return float(closing.geometry_of(self._tracked[0]).range)


def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._active = False
        node._stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
