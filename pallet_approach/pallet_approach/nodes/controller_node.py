"""Lifecycle node: drive to the target pose. Publishes /cmd_vel only while active.

The safety-critical node. Four rules it enforces, independently of the
coordinator being correct:

1. **No publishing unless activated.** The publisher exists from `configure`,
   but `_active` gates every write, and the lifecycle state is the only thing
   that sets it.
2. **Fixed rate, explicit zeros.** The base driver has no command timeout --
   `ros2 param dump /scout` has no such parameter -- so a command it already
   has keeps it moving. Commands also cross the network to the robot PC. So the
   timer publishes every tick while active, and deactivating sends several
   explicit zeros.
3. **Stale target means stop.** If the coordinator stops refreshing the target,
   the robot stops, rather than driving to a target that may no longer exist.
4. **Somebody is listening.** At configure it checks that something subscribes
   to its command topic, and refuses otherwise.

   This replaces a sole-publisher refusal that belonged to the superseded
   design where Nav2 was shut down for a run (D3). Arbitration took over
   (config/twist_mux.yaml, D30), and that moved our output off /cmd_vel and
   onto /cmd_vel_pallet -- which means that with twist_mux not running, every
   command this node sends goes nowhere and the robot sits still while the
   whole stack reports that it is driving. That is the exact shape of failure
   this project keeps meeting: a reasonable default, chosen silently, never
   reported. So it is reported, and it refuses (D31).
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool

from pallet_approach.controller import ZERO, Phase, RateLimiter, compute_command
from pallet_approach.model import Pose2D
from pallet_approach.ros_conversions import (
    command_to_twist,
    pose2d_from_odometry,
    quaternion_to_yaw,
)

RELIABLE = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)


class ControllerNode(LifecycleNode):

    def __init__(self):
        super().__init__('approach_controller')
        self._declare()
        self._active = False
        self._pose = None
        self._target = None
        self._target_at = 0.0
        self._phase = Phase.ROTATE_TO_HEADING
        self._limiter = None
        self._timer = None
        self._cmd_pub = None
        self._arrived_pub = None
        self._last_tick = None

    def _declare(self):
        self.declare_parameter('topics.cmd_vel_pallet', '/cmd_vel_pallet')
        self.declare_parameter('topics.odom', '/odom')
        for name, default in (
            ('limits.max_linear', 0.20), ('limits.max_angular', 0.30),
            ('limits.max_linear_accel', 0.30), ('limits.max_angular_accel', 0.60),
            ('limits.min_linear', 0.05), ('limits.min_angular', 0.10),
            ('gains.k_rho', 0.6), ('gains.k_alpha', 1.2), ('gains.k_beta', 0.4),
            ('tolerances.position', 0.030), ('tolerances.yaw', 0.030),
            ('tolerances.heading_before_drive', 0.100),
            ('command_rate', 20.0), ('target_timeout', 0.50),
            ('zero_command_repeats', 5),
        ):
            self.declare_parameter(name, default)

    def _group(self, prefix, keys):
        return {k: self.get_parameter(f'{prefix}.{k}').value for k in keys}

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # ----------------------------------------------------------- lifecycle

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        cmd_topic = self.get_parameter('topics.cmd_vel_pallet').value
        self._cmd_pub = self.create_lifecycle_publisher(Twist, cmd_topic, RELIABLE)
        self._arrived_pub = self.create_lifecycle_publisher(Bool, '~/arrived', 10)
        self.create_subscription(Odometry, self.get_parameter('topics.odom').value,
                                 self._on_odom, RELIABLE)
        self.create_subscription(PoseStamped, '~/target', self._on_target, RELIABLE)

        if not self._wait_for_a_subscriber(cmd_topic):
            self.get_logger().error(
                f'nothing subscribes to {cmd_topic}, so every command this node '
                f'sends would be discarded and the robot would never move. '
                f'twist_mux is what should be listening -- start it with the '
                f'stack (launch/pallet_approach.launch.py brings it up) or '
                f'check config/twist_mux.yaml names this topic. Refusing to '
                f'configure.')
            return TransitionCallbackReturn.FAILURE

        others = self.count_publishers(cmd_topic) - 1
        if others > 0:
            self.get_logger().warning(
                f'{others} other node(s) also publish {cmd_topic}, which is '
                f'meant to be this node\'s private input to twist_mux. '
                f'Arbitration cannot separate two publishers on one input.')

        self._limiter = RateLimiter(
            self.get_parameter('limits.max_linear_accel').value,
            self.get_parameter('limits.max_angular_accel').value)
        self.get_logger().info(
            f'configured; commands go to {cmd_topic} '
            f'({self.count_subscribers(cmd_topic)} subscriber(s)); '
            f'max {self.get_parameter("limits.max_linear").value} m/s, '
            f'{self.get_parameter("limits.max_angular").value} rad/s')
        return TransitionCallbackReturn.SUCCESS

    def _wait_for_a_subscriber(self, topic: str, timeout: float = 2.0) -> bool:
        """Is anything listening on `topic`?

        With a short wait, because discovery is not instant and the mux may
        have come up moments before us. Two seconds, then an answer either way
        -- never an assumption.
        """
        deadline = self._now() + timeout
        while self._now() < deadline:
            if self.count_subscribers(topic) > 0:
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.count_subscribers(topic) > 0

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self._phase = Phase.ROTATE_TO_HEADING
        self._target = None
        self._limiter.reset()
        self._last_tick = None
        self._active = True
        period = 1.0 / float(self.get_parameter('command_rate').value)
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

        This is the one place that publishes while not active, and it can only
        ever send zero. Repeating it is deliberate: the command crosses the
        network and a single dropped zero would leave the base moving.
        """
        if self._cmd_pub is None:
            return
        twist = Twist()
        for _ in range(int(self.get_parameter('zero_command_repeats').value)):
            self._cmd_pub.publish(twist)
        self._limiter.reset()

    def _publish(self, command):
        if not self._active:
            return
        self._cmd_pub.publish(command_to_twist(command))

    # ------------------------------------------------------------ callbacks

    def _on_odom(self, msg: Odometry):
        self._pose = pose2d_from_odometry(msg)

    def _on_target(self, msg: PoseStamped):
        """A target from the coordinator, in the odom frame."""
        self._target = Pose2D(x=msg.pose.position.x, y=msg.pose.position.y,
                              yaw=quaternion_to_yaw(msg.pose.orientation))
        self._target_at = self._now()
        if self._phase is Phase.DONE:
            self._phase = Phase.ROTATE_TO_HEADING

    def _tick(self):
        if not self._active:
            return
        now = self._now()
        dt = 0.0 if self._last_tick is None else now - self._last_tick
        self._last_tick = now

        if self._pose is None or self._target is None:
            self._publish(ZERO)
            return
        if now - self._target_at > float(self.get_parameter('target_timeout').value):
            self.get_logger().warning('target is stale, stopping',
                                      throttle_duration_sec=1.0)
            self._publish(ZERO)
            return

        command, phase = compute_command(
            self._pose, self._target, self._phase,
            self._group('gains', ('k_rho', 'k_alpha', 'k_beta')),
            self._group('limits', ('max_linear', 'max_angular',
                                   'min_linear', 'min_angular')),
            self._group('tolerances', ('position', 'yaw', 'heading_before_drive')))

        if phase is Phase.DONE and self._phase is not Phase.DONE:
            self.get_logger().info('reached the target')
            self._arrived_pub.publish(Bool(data=True))
        self._phase = phase
        self._publish(self._limiter.limit(command, dt) if dt > 0.0 else ZERO)


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
