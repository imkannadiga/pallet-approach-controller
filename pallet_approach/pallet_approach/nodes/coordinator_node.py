"""Action server and state machine owner.

The only node that decides anything. It activates and deactivates the other
two, holds the driving targets, and returns GO / NO_GO / NO_PALLET / ABORTED.

The decision logic is in `coordinator.py` (pure state machine) and `go_nogo.py`
(the rules); this file is the ROS shell around them. Every transition it gets
back carries `stop_robot`, `controller_active` and `perception_active`, and this
node's job is simply to make those true -- which is why the safety properties
can be tested without ROS.
"""

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import PoseStamped
from lifecycle_msgs.srv import ChangeState
from lifecycle_msgs.msg import Transition
from nav_msgs.msg import Odometry
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String

from pallet_approach_interfaces.action import PalletTask
from pallet_approach_interfaces.msg import PalletEstimate as PalletEstimateMsg

from pallet_approach import go_nogo
from pallet_approach.approach import (
    pocket_center,
    standoff_pose,
    transform_pose,
)
from pallet_approach.coordinator import Coordinator, Outcome, State
from pallet_approach.model import Decision, Pose2D
from pallet_approach.ros_conversions import (
    estimate_from_msg,
    pose2d_from_odometry,
    pose2d_to_pose,
)

RELIABLE = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
MANAGED = ('pallet_perception', 'approach_controller')


class CoordinatorNode(Node):

    def __init__(self):
        super().__init__('coordinator')
        self._declare()
        self._group = ReentrantCallbackGroup()
        self._machine = Coordinator(self._machine_params())
        self._pose = Pose2D(0.0, 0.0, 0.0)
        self._estimate = None
        self._target = None           # the standoff pose, in odom
        self._arrived = False
        self._goal_handle = None
        self._last_outcome = None
        self._last_reason = ''

        self._target_pub = self.create_publisher(
            PoseStamped, '/approach_controller/target', RELIABLE)
        self._state_pub = self.create_publisher(String, '~/state', RELIABLE)

        self.create_subscription(Odometry, self.get_parameter('topics.odom').value,
                                 self._on_odom, RELIABLE, callback_group=self._group)
        self.create_subscription(PalletEstimateMsg, '/pallet_perception/estimate',
                                 self._on_estimate, RELIABLE, callback_group=self._group)
        self.create_subscription(Bool, '/approach_controller/arrived',
                                 self._on_arrived, RELIABLE, callback_group=self._group)

        self._clients = {
            name: self.create_client(ChangeState, f'/{name}/change_state',
                                     callback_group=self._group)
            for name in MANAGED
        }

        self._action = ActionServer(
            self, PalletTask, self.get_parameter('action_name').value,
            execute_callback=self._execute,
            goal_callback=self._on_goal,
            cancel_callback=self._on_cancel,
            callback_group=self._group)

        self.create_timer(0.05, self._tick, callback_group=self._group)
        self.get_logger().info('coordinator ready; send a goal to start a run')

    # ------------------------------------------------------------- params

    def _declare(self):
        self.declare_parameter('action_name', 'pallet_task')
        self.declare_parameter('standoff_distance', 1.75)
        self.declare_parameter('topics.odom', '/odom')
        self.declare_parameter('frames.odom', 'odom')
        self.declare_parameter('frames.ground', 'base_footprint')
        self.declare_parameter('frames.body', 'base_link')
        for name, default in (
            ('detecting.timeout', 10.0), ('detecting.confirm_frames', 5),
            ('detecting.max_position_spread', 0.050),
            ('detecting.max_yaw_spread', 0.090),
            ('approaching.lost_pallet_timeout', 1.0),
            ('approaching.max_estimate_jump', 0.15),
            ('approaching.max_estimate_jump_yaw', 0.20),
            ('checking.settle_time', 0.5), ('checking.frames', 5),
            ('go_nogo.max_yaw_error', 0.087), ('go_nogo.max_lateral_offset', 0.050),
            ('go_nogo.pocket_width_tolerance', 0.040),
            ('go_nogo.pocket_height_tolerance', 0.030),
            ('go_nogo.min_pocket_count', 2), ('go_nogo.overhang_margin', 0.020),
            ('go_nogo.pocket_clearance_depth', 0.10),
            ('go_nogo.stability_frames', 5), ('go_nogo.max_position_spread', 0.030),
            ('go_nogo.max_yaw_spread', 0.035),
            ('pallet.sides.short.pocket_width', 0.230),
            ('pallet.sides.short.opening_top', 0.100),
            ('pallet.sides.short.opening_bottom', 0.000),
            ('pallet.sides.long.pocket_width', 0.375),
            ('pallet.sides.long.opening_top', 0.125),
            ('pallet.sides.long.opening_bottom', 0.020),
        ):
            self.declare_parameter(name, default)

    def _get(self, name):
        return self.get_parameter(name).value

    def _group_params(self, prefix, keys):
        return {k: self._get(f'{prefix}.{k}') for k in keys}

    def _machine_params(self):
        return {
            'detecting': self._group_params('detecting', (
                'timeout', 'confirm_frames', 'max_position_spread', 'max_yaw_spread')),
            'approaching': self._group_params('approaching', (
                'lost_pallet_timeout', 'max_estimate_jump', 'max_estimate_jump_yaw')),
            'checking': self._group_params('checking', ('settle_time', 'frames')),
        }

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # -------------------------------------------------- lifecycle plumbing

    def _transition(self, node_name: str, transition_id: int) -> bool:
        client = self._clients[node_name]
        if not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error(f'{node_name}/change_state is not available')
            return False
        request = ChangeState.Request()
        request.transition.id = transition_id
        future = client.call_async(request)
        start = self._now()
        while not future.done() and self._now() - start < 5.0:
            rclpy.spin_once(self, timeout_sec=0.05)
        if not future.done():
            self.get_logger().error(f'{node_name} transition {transition_id} timed out')
            return False
        return bool(future.result().success)

    def _apply(self, transition) -> None:
        """Make a Transition from the state machine true in the system.

        Order matters: stop the robot first, then change activations. Stopping
        before deactivating means the zero command goes out even if a lifecycle
        call then fails.
        """
        if transition.stop_robot:
            self._publish_target(None)
        if transition.outcome is not None:
            self._last_outcome = transition.outcome
            self._last_reason = transition.reason
        if transition.changed:
            self._state_pub.publish(String(data=transition.state.value))
            self.get_logger().info(
                f'{transition.state.value}'
                + (f': {transition.reason}' if transition.reason else ''))
            want = {'pallet_perception': transition.perception_active,
                    'approach_controller': transition.controller_active}
            for name, active in want.items():
                self._transition(
                    name,
                    Transition.TRANSITION_ACTIVATE if active
                    else Transition.TRANSITION_DEACTIVATE)

    def _configure_all(self) -> bool:
        return all(self._transition(name, Transition.TRANSITION_CONFIGURE)
                   for name in MANAGED)

    # ------------------------------------------------------------ callbacks

    def _on_odom(self, msg: Odometry):
        self._pose = pose2d_from_odometry(msg)

    def _on_arrived(self, msg: Bool):
        if msg.data:
            self._arrived = True

    def _on_estimate(self, msg: PalletEstimateMsg):
        estimate = estimate_from_msg(msg)
        if estimate is None:
            return
        self._estimate = estimate
        before = self._machine.state
        transition = self._machine.on_estimate(estimate, self._now())
        self._apply(transition)

        if before is State.DETECTING and transition.state is State.APPROACHING:
            # Confirmed. Freeze the targets in odom, where they stay put: odom
            # drifts slowly, the robot frame moves under the robot, and map
            # jumps whenever slam_toolbox closes a loop.
            #
            # The estimate is in base_footprint and self._pose is base_link in
            # odom, and composing them is exact rather than sloppy: the URDF
            # joint between the two is a pure z translation (rpy "0 0 0"), so
            # the frames share x, y and yaw, and these targets are planar.
            center = pocket_center(estimate.pockets, fallback=estimate.position)
            standoff = standoff_pose(center, estimate.normal,
                                     self._get('standoff_distance'))
            self._target = transform_pose(standoff, self._pose)
            self._arrived = False
            self.get_logger().info(
                'standoff in odom: (%.2f, %.2f, %.1f deg)' % (
                    self._target.x, self._target.y,
                    math.degrees(self._target.yaw)))

    def _publish_target(self, target: Pose2D | None):
        if target is None:
            return
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._get('frames.odom')
        msg.pose = pose2d_to_pose(target)
        self._target_pub.publish(msg)

    def _tick(self):
        now = self._now()
        state = self._machine.state

        if state is State.APPROACHING and self._target is not None:
            # Keep publishing: the controller stops if the target goes stale.
            self._publish_target(self._target)
            if self._arrived:
                self._arrived = False
                self._apply(self._machine.on_arrived(now))

        self._apply(self._machine.on_tick(now))

        if self._machine.ready_to_decide(now):
            self._apply(self._machine.on_decision(self._decide(), now))

    def _decide(self) -> Decision:
        params = self._group_params('go_nogo', (
            'max_yaw_error', 'max_lateral_offset', 'pocket_width_tolerance',
            'pocket_height_tolerance', 'min_pocket_count', 'overhang_margin',
            'pocket_clearance_depth', 'stability_frames', 'max_position_spread',
            'max_yaw_spread'))
        try:
            return go_nogo.evaluate(self._estimate, self._pose,
                                    list(self._machine.history), params)
        except NotImplementedError:
            self.get_logger().error(
                'go_nogo.evaluate is not implemented yet -- that file is yours '
                'to write. Returning NO_GO, which is the safe default.')
            return Decision(go=False, reason='go_nogo rules are not implemented')
        except Exception as error:
            self.get_logger().error(f'go_nogo raised: {error!r}')
            return Decision(go=False, reason=f'go_nogo raised {type(error).__name__}')

    # --------------------------------------------------------------- action

    def _on_goal(self, goal_request):
        if self._machine.state in (State.DETECTING, State.APPROACHING, State.CHECKING):
            self.get_logger().warning('rejecting goal: a run is already in progress')
            return GoalResponse.REJECT
        if goal_request.mode not in ('pallet_orientation', ''):
            self.get_logger().warning(f'rejecting unknown mode {goal_request.mode!r}')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _on_cancel(self, goal_handle):
        self.get_logger().info('cancel requested')
        self._apply(self._machine.cancel(self._now()))
        return CancelResponse.ACCEPT

    def _execute(self, goal_handle):
        self._goal_handle = goal_handle
        self._estimate = None
        self._target = None
        self._arrived = False
        self._last_outcome = None
        self._last_reason = ''

        if not self._configure_all():
            goal_handle.abort()
            return self._result(Outcome.ABORTED, 'could not configure the managed nodes')

        self._apply(self._machine.trigger(self._now()))

        while rclpy.ok():
            if self._machine.state is State.DONE:
                break
            if goal_handle.is_cancel_requested:
                self._apply(self._machine.cancel(self._now()))
                break
            goal_handle.publish_feedback(self._feedback())
            self.get_clock().sleep_for(rclpy.duration.Duration(seconds=0.1))

        outcome = self._last_outcome or Outcome.ABORTED
        reason = self._last_reason or 'interrupted'

        self._apply(self._machine.to_idle(self._now()))

        if outcome is Outcome.ABORTED and goal_handle.is_cancel_requested:
            goal_handle.canceled()
        else:
            goal_handle.succeed()
        self._goal_handle = None
        return self._result(outcome, reason)

    def _feedback(self):
        feedback = PalletTask.Feedback()
        feedback.phase = self._machine.state.value
        feedback.distance_to_standoff = float('nan')
        if self._target is not None:
            feedback.distance_to_standoff = float(
                math.hypot(self._target.x - self._pose.x,
                           self._target.y - self._pose.y))
        return feedback

    def _result(self, outcome: Outcome, reason: str):
        result = PalletTask.Result()
        result.outcome = outcome.value
        result.reason = reason
        result.pallet_pose.header.stamp = self.get_clock().now().to_msg()
        result.pallet_pose.header.frame_id = self._get('frames.ground')
        if self._estimate is not None:
            center = pocket_center(self._estimate.pockets, self._estimate.position)
            result.pallet_pose.pose = pose2d_to_pose(
                Pose2D(float(center[0]), float(center[1]), float(self._estimate.yaw)),
                z=float(center[2]))
        self.get_logger().info(f'result: {outcome.value} ({reason})')
        return result


def main(args=None):
    rclpy.init(args=args)
    node = CoordinatorNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
