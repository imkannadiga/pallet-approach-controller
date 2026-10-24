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

from pallet_approach import closing
from pallet_approach.controller import ZERO, RateLimiter
from pallet_approach.drive import MODES, DriveExecutor, Request, Result, executor_params
from pallet_approach.nodes._common import LatestRaw, logged_transition, lookup_waiting
from pallet_approach.calibration import correct_lidar_transform
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


class ControllerNode(LifecycleNode):

    def __init__(self):
        super().__init__('approach_controller')
        self._declare()
        self._active = False
        self._pose = None
        self._command = None
        self._command_at = 0.0
        self._tracked = None
        self._frames_seen = 0
        self._frames_stepped = 0
        self._tracked_at = 0.0
        self._cloud = LatestRaw(PointCloud2)
        self._limiter = None
        self._timer = None
        self._cmd_pub = None
        self._status_pub = None
        self._last_tick = None
        self._buffer = None
        self._listener = None
        self._start_pose = None
        self._executor = None
        """The per-move logic, in `drive.py`. Separate so that the join between
        the sequence and the control laws can be run at a desk against
        `test/simulate.py` instead of on a robot."""

    # ------------------------------------------------------------- params

    def _declare(self):
        self.declare_parameter('topics.cmd_vel_pallet', '/cmd_vel_pallet')
        self.declare_parameter('topics.odom', '/odom')
        self.declare_parameter('topics.lidar', '/ouster/points')
        self.declare_parameter('frames.ground', 'base_footprint')
        self.declare_parameter('frames.tf_timeout', 0.15)
        self.declare_parameter('floor_z', -0.025)
        self.declare_parameter('frames.lidar_correction.enabled', True)
        self.declare_parameter('frames.lidar_correction.measured_floor_pitch', 0.0)
        self.declare_parameter('frames.lidar_correction.measured_floor_roll', 0.0)
        self.declare_parameter('frames.lidar_correction.measured_floor_height', -0.025)
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
            ('closing.min_divergence_growth', 0.040),
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

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # ----------------------------------------------------------- lifecycle

    @logged_transition
    def on_configure(self, state: State) -> TransitionCallbackReturn:
        topic = self._get('topics.cmd_vel_pallet')

        # Everything a subscription callback can touch is built FIRST, before
        # any subscription exists to call one. This is not just tidiness:
        # `_wait_for_a_subscriber` below spins this node's own executor while
        # it waits, and a subscription created earlier is live during that
        # spin -- so if DriveCommand had been subscribed before `self._executor`
        # existed, a message arriving during the wait (the coordinator
        # publishes a STOP heartbeat at 20 Hz even while idle, so there is
        # always one queued) would call `_on_drive` with `self._executor` still
        # `None`. Measured 2026-10-19: exactly that, an AttributeError that
        # killed the node, which then made the coordinator's NEXT configure
        # attempt fail too ("could not configure the managed nodes") because
        # the service was no longer there to answer it.
        self._limiter = RateLimiter(self._get('limits.max_linear_accel'),
                                    self._get('limits.max_angular_accel'))
        self._executor = DriveExecutor(executor_params(self._get))
        self._buffer = Buffer()
        self._listener = TransformListener(self._buffer, self, spin_thread=True)

        self._cmd_pub = self.create_lifecycle_publisher(Twist, topic, RELIABLE)
        self._status_pub = self.create_lifecycle_publisher(
            DriveStatus, '~/status', RELIABLE)
        self.create_subscription(Odometry, self._get('topics.odom'),
                                 self._on_odom, RELIABLE)
        self.create_subscription(DriveCommand, '~/drive', self._on_drive, RELIABLE)
        self.create_subscription(TrackedPallet, '/pallet_tracker/tracked',
                                 self._on_tracked, RELIABLE)
        self.create_subscription(PointCloud2, self._get('topics.lidar'),
                                 self._on_cloud, BEST_EFFORT, raw=True)

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

    @logged_transition
    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self._executor.start(Request('STOP'))
        self._start_pose = self._pose
        self._command = None
        self._limiter.reset()
        self._last_tick = None
        self._active = True
        period = 1.0 / float(self._get('command_rate'))
        self._timer = self.create_timer(period, self._tick)
        self.get_logger().info('active')
        return super().on_activate(state)

    @logged_transition
    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self._active = False
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._stop()
        self.get_logger().info('inactive, zero command sent')
        return super().on_deactivate(state)

    @logged_transition
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
        """Feed the executor PATH LENGTH, not displacement.

        A move that curves covers more ground than it displaces, and the
        insertion stop is about how far the wheels turned. Publishing both
        (DriveStatus.travelled and .displacement) means a growing gap between
        them shows up as the robot curving when it was told not to.
        """
        pose = pose2d_from_odometry(msg)
        if self._pose is not None and self._executor is not None:
            self._executor.advance(math.hypot(pose.x - self._pose.x,
                                              pose.y - self._pose.y))
        self._pose = pose

    def _on_tracked(self, msg: TrackedPallet):
        estimate = estimate_from_msg(msg.estimate)
        if estimate is None:
            return
        # The fused pallet line in odom (D59) travels with the newest frame:
        # CLOSE steers onto the former, INSERT still reads the latter.
        self._frames_seen += 1
        self._tracked = (estimate, bool(msg.confirmed),
                         (msg.center_odom.x, msg.center_odom.y), float(msg.yaw_odom))
        self._tracked_at = self._now()

    def _on_cloud(self, raw: bytes):
        self._cloud.store(raw)      # decoded only if a move asks for it

    def _on_drive(self, msg: DriveCommand):
        if self._executor is None:
            # Belt and suspenders alongside the construction order in
            # on_configure: this subscription must not be live before
            # self._executor exists, and now it never is -- but a callback
            # assuming its own node's state just because it was called is the
            # exact class of bug that put a None here before, so it costs one
            # guard to make sure it cannot happen a second way.
            return
        self._command_at = self._now()
        if msg.mode == self._executor.mode and self._command is not None:
            # A repeat of the move in progress, which is how the coordinator
            # keeps the command fresh. Refreshing the timestamp is all it does;
            # restarting the move would reset `travelled` every frame and the
            # robot would drive for ever.
            return
        if msg.mode not in MODES:
            self._executor.start(Request('STOP'))
            self._executor.abort = True
            self._executor.reason = f'unknown drive mode {msg.mode!r}'
            self.get_logger().error(
                f'unknown drive mode {msg.mode!r}; stopping. Known: '
                f'{", ".join(MODES)}')
            return
        self._command = msg
        self._executor.start(Request(mode=msg.mode, distance=msg.distance,
                                     side=msg.side))
        self._start_pose = self._pose
        if self._limiter is not None:
            self._limiter.reset()
        self.get_logger().info(
            f'{msg.mode}' + (f' {msg.distance:.3f} m' if msg.distance else '')
            + (f' into the {msg.side} face' if msg.side else ''))

    # --------------------------------------------------------------- lidar

    def _lidar_in_ground(self):
        """The newest cloud in `frames.ground`, or None with a logged reason."""
        cloud = self._cloud.get()
        if cloud is None:
            return None
        ground = self._get('frames.ground')
        try:
            transform = lookup_waiting(self._buffer, ground, cloud.header.frame_id, cloud.header.stamp,
                rclpy.duration.Duration(seconds=float(self._get('frames.tf_timeout'))))
        except Exception as error:
            self.get_logger().warning(f'tf {ground} <- {cloud.header.frame_id}: '
                                      f'{error}', throttle_duration_sec=5.0)
            return None
        matrix = transform_to_matrix(transform)
        if self._get('frames.lidar_correction.enabled'):
            matrix = correct_lidar_transform(
                matrix,
                self._get('frames.lidar_correction.measured_floor_pitch'),
                self._get('frames.lidar_correction.measured_floor_roll'),
                self._get('frames.lidar_correction.measured_floor_height'),
                self._get('floor_z'))
        return transform_points(unpack_cloud(cloud), matrix)


    # ---------------------------------------------------------------- tick

    def _tick(self):
        if not self._active:
            return
        now = self._now()
        dt = 0.0 if self._last_tick is None else now - self._last_tick
        self._last_tick = now

        result, note = self._evaluate(now)

        status = DriveStatus()
        status.header.stamp = self.get_clock().now().to_msg()
        status.mode = self._executor.mode
        status.phase = result.phase
        status.arrived = bool(result.arrived)
        status.abort = bool(result.abort)
        status.blocked = bool(result.blocked and self._executor.mode != 'STOP')
        status.reason = note or result.reason
        status.travelled = float(self._executor.travelled)
        status.displacement = float(self._displacement())
        status.face_range = float(result.face_range)
        status.lateral_offset = float(result.lateral_offset)
        status.heading_error = float(result.heading_error)
        status.rear_clearance = float(result.rear_clearance)
        self._status_pub.publish(status)

        self._publish(self._limiter.limit(result.command, dt) if dt > 0.0 else ZERO)

    def _evaluate(self, now: float):
        """One frame of the executor, plus the refusals only the node can make.

        Returns (result, note); `note` overrides the reason when the node
        itself is the thing refusing, so that "the drive command is stale" does
        not get reported as whatever the control law last said.
        """
        stop = Result(ZERO, False, False, '')

        if self._executor.mode == 'STOP' or self._command is None:
            return stop, self._executor.reason if self._command else 'no command'

        age = now - self._command_at
        if age > float(self._get('target_timeout')):
            note = f'drive command is {age:.2f} s old; stopping'
            self.get_logger().warning(note, throttle_duration_sec=1.0)
            return stop, note

        if self._pose is None:
            return stop, 'no odometry yet'
        if self._start_pose is None:
            # The move was requested before any odometry arrived, so it has no
            # origin to measure from. Take this pose as the origin rather than
            # measuring from wherever the robot was when the node woke up.
            self._start_pose = self._pose

        geometry, note = self._geometry(now)
        # Only the straight moves read the LiDAR (drive.step); CLOSE and STOP
        # never do, so the cloud is not decoded and transformed for them.
        points = (self._lidar_in_ground()
                  if self._executor.mode in ('REVERSE', 'BACKOUT', 'INSERT') else None)
        new_frame = self._frames_seen != self._frames_stepped
        self._frames_stepped = self._frames_seen
        result = self._executor.step(geometry=geometry, lidar_points=points,
                                     new_frame=new_frame)
        if result.arrived and not result.abort:
            self.get_logger().info(f'{self._executor.mode} complete: '
                                   f'{result.reason}',
                                   throttle_duration_sec=5.0)
        if result.abort:
            self.get_logger().error(f'{self._executor.mode} aborted: '
                                    f'{result.reason}',
                                    throttle_duration_sec=5.0)
        return result, note

    def _geometry(self, now: float):
        """This frame's pallet geometry, or (None, why not).

        Only CLOSE needs it; the straight moves do not, and demanding it would
        stop the insertion the moment the pockets left the camera's view, which
        happens by design 0.58 m in (D16).
        """
        if self._executor.mode not in ('CLOSE', 'INSERT'):
            return None, ''
        if self._tracked is None:
            return None, 'no pallet estimate yet'
        age = now - self._tracked_at
        if age > float(self._get('target_timeout')):
            note = f'pallet estimate is {age:.2f} s old'
            if self._executor.mode == 'CLOSE':
                self.get_logger().warning(f'{note}; stopping',
                                          throttle_duration_sec=1.0)
                return None, f'{note}; stopping'
            return None, ''
        estimate, confirmed, center_odom, yaw_odom = self._tracked
        if not confirmed and self._executor.mode == 'CLOSE':
            # Driving on an unconfirmed track is exactly what the tracker
            # exists to prevent.
            return None, 'the track is not confirmed'
        if self._executor.mode == 'CLOSE':
            if self._pose is None:
                return None, 'no odometry yet'
            # D59: the averaged pallet line against where the robot is NOW, so
            # every 20 Hz tick has a current geometry, not one per frame.
            return closing.geometry_from_track(center_odom, yaw_odom, self._pose), ''
        return closing.geometry_of(estimate), ''

    def _displacement(self) -> float:
        if self._pose is None or self._start_pose is None:
            return 0.0
        return math.hypot(self._pose.x - self._start_pose.x,
                          self._pose.y - self._start_pose.y)


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
