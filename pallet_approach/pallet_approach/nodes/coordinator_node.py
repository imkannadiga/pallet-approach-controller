"""Action server and owner of the sequence: the ROS shell around `coordinator.py`
(state machine) and `go_nogo.py` (rules). It makes each Transition true:
lifecycle activation, drive commands, fork commands, feedback. Outcomes:
INSERTED, NO_GO, NO_PALLET, ABORTED.

Also owns the health gate (D12: a run is refused while the stack is
unhealthy) and the navigation lock (D30: published for the length of a run so
twist_mux blocks Nav2; it lapses 0.5 s after this node stops).

A run starts from /pallet_approach/start (std_msgs/String, the mode; VOLATILE
depth 1, so nothing latched can start one) or from the /pallet_task action
(D55). A topic-started run is stopped by /pallet_approach/cancel and reports on
/pallet_approach/result (the action's Result type). Both go through
`_start_run`.
"""

from __future__ import annotations

import math
import functools
import threading
import time
from collections import deque

import rclpy
from geometry_msgs.msg import PoseStamped
from lifecycle_msgs.msg import Transition as LifecycleTransition
from lifecycle_msgs.srv import ChangeState, GetState
from nav_msgs.msg import Odometry
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, Empty, String

from pallet_approach_interfaces.action import PalletTask
from pallet_approach_interfaces.msg import (
    CoordinatorStatus,
    DriveCommand,
    DriveStatus,
    ForkState,
    Health,
    TrackedPallet,
)

from pallet_approach import go_nogo
from pallet_approach.nodes._common import tracked_topic
from pallet_approach.approach import pocket_center
from pallet_approach.closing import geometry_of
from pallet_approach.coordinator import (
    MODES,
    PICKUP,
    Coordinator,
    machine_params,
    transitions_to_inactive,
    DriveInfo,
    ForkInfo,
    Outcome,
    State,
    TrackInfo,
)
from pallet_approach.forks import pallet_lift
from pallet_approach.model import Decision, Pose2D
from pallet_approach.pointcloud import unpack_cloud
from pallet_approach.ros_conversions import (
    estimate_from_msg,
    pose2d_from_odometry,
    pose2d_to_pose,
)

RELIABLE = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

# VOLATILE and shallow, on purpose. A latched trigger would start a run the
# moment this node came up, from a message published before it existed; a deep
# queue would start a second run as soon as the first finished.
TRIGGER = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.VOLATILE)

MANAGED = ('pallet_perception', 'approach_controller')


def _serialised(method):
    """Run a callback under the machine lock. The action's `_execute` blocks for a
    whole run, so it takes the lock only around the steps that touch the machine.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._machine_lock:
            return method(self, *args, **kwargs)
    return wrapper


class CoordinatorNode(Node):

    def __init__(self):
        super().__init__('coordinator')
        self._declare()
        # Machine callbacks (tick, trigger, subscriptions) are mutually
        # exclusive: overlapping ticks once issued duplicate lifecycle
        # transitions. Lifecycle clients and the action server get their own
        # reentrant groups, so responses arrive while the machine waits and
        # `_execute` can block for a whole run.
        self._group = MutuallyExclusiveCallbackGroup()
        self._machine_lock = threading.RLock()
        self._service_group = ReentrantCallbackGroup()
        self._action_group = ReentrantCallbackGroup()
        self._machine = Coordinator(self._machine_params())
        self._pose = Pose2D(0.0, 0.0, 0.0)
        self._estimate = None
        self._tracked = None
        self._health = None
        self._cluster_points = None
        self._scene_points = None
        self._checking_estimates: deque = deque(maxlen=20)
        """Estimates taken since CHECKING began. The GO/NO-GO stability rule
        is evaluated over these, and only frames taken with the robot standing
        still belong in it -- five frames while closing span 100 mm of real
        travel, which the rule reads as an unstable estimate."""
        self._goal_handle = None
        self._last_outcome = None
        self._last_reason = ''
        self._reported = True
        """Has the current outcome been published on `~/result` yet? True to
        begin with, because there is no outcome to report before the first
        run."""
        self._run_started_at: float | None = None
        """When the current run was triggered, for `CoordinatorStatus.elapsed`.
        `None`, not 0.0, while no run has ever started -- 0.0 is a clock
        reading `Coordinator.started_at` itself uses internally, and reusing
        it here as an "unset" sentinel is the exact mistake `health.py` and
        `rate_limit.FrameGate` were both written to rule out."""
        self._active_nodes = {name: False for name in MANAGED}

        self._drive_pub = self.create_publisher(
            DriveCommand, '/approach_controller/drive', RELIABLE)
        self._fork_pub = self.create_publisher(
            String, '/virtual_forks/command', RELIABLE)
        self._state_pub = self.create_publisher(String, '~/state', RELIABLE)
        self._status_pub = self.create_publisher(
            CoordinatorStatus, '~/status', RELIABLE)
        self._lock_pub = self.create_publisher(
            Bool, self._get('topics.nav_lock'), RELIABLE)
        self._pose_pub = self.create_publisher(PoseStamped, '~/pallet_pose', RELIABLE)
        self._result_pub = self.create_publisher(
            PalletTask.Result, '/pallet_approach/result', RELIABLE)

        self.create_subscription(String, '/pallet_approach/start',
                                 self._on_start, TRIGGER,
                                 callback_group=self._group)
        self.create_subscription(Empty, '/pallet_approach/cancel',
                                 self._on_cancel_topic, TRIGGER,
                                 callback_group=self._group)

        self.create_subscription(Odometry, self._get('topics.odom'),
                                 self._on_odom, RELIABLE,
                                 callback_group=self._group)
        self._source = self._get('perception_source')
        self.create_subscription(TrackedPallet, tracked_topic(self._source),
                                 self._on_tracked, RELIABLE,
                                 callback_group=self._group)
        self.get_logger().info(f'perception: {self._source} ({tracked_topic(self._source)})')
        self.create_subscription(DriveStatus, '/approach_controller/status',
                                 self._on_drive_status, RELIABLE,
                                 callback_group=self._group)
        self.create_subscription(ForkState, '/virtual_forks/state',
                                 self._on_fork_state, RELIABLE,
                                 callback_group=self._group)
        self.create_subscription(Health, '/health', self._on_health, RELIABLE,
                                 callback_group=self._group)
        # Two of the nine GO/NO-GO rules need points rather than an estimate,
        # and both treat missing data as a NO-GO rather than a pass -- so
        # without these subscriptions every run is refused with "no points to
        # check the load against". Purpose-built and small (see the
        # perception node); they stay on the Jetson.
        self.create_subscription(PointCloud2, '/pallet_perception/cluster_points',
                                 self._on_cluster_points, 1,
                                 callback_group=self._group)
        self.create_subscription(PointCloud2, '/pallet_perception/scene_points',
                                 self._on_scene_points, 1,
                                 callback_group=self._group)

        self._managed_clients = {
            name: self.create_client(ChangeState, f'/{name}/change_state',
                                     callback_group=self._service_group)
            for name in MANAGED
        }
        self._state_clients = {
            name: self.create_client(GetState, f'/{name}/get_state',
                                     callback_group=self._service_group)
            for name in MANAGED
        }

        self._action = ActionServer(
            self, PalletTask, self._get('action_name'),
            execute_callback=self._execute, goal_callback=self._on_goal,
            cancel_callback=self._on_cancel, callback_group=self._action_group)

        self.create_timer(0.05, self._tick, callback_group=self._group)
        self.get_logger().info('coordinator ready; send a goal to start a run')

    # ------------------------------------------------------------- params

    def _declare(self):
        self.declare_parameter('action_name', 'pallet_task')
        self.declare_parameter('topics.odom', '/odom')
        self.declare_parameter('perception_source', 'camera')
        self.declare_parameter('topics.nav_lock', '/pallet_approach/nav_lock')
        self.declare_parameter('frames.odom', 'odom')
        self.declare_parameter('frames.ground', 'base_footprint')
        self.declare_parameter('standoff_distance', 1.75)
        self.declare_parameter('pallet.entry_face', 'short')
        self.declare_parameter('require_healthy_to_start', True)
        for name, default in (
            ('timeouts.confirm', 10.0),
            ('timeouts.lost_pallet', 2.5),
            ('timeouts.forks', 10.0), ('timeouts.run', 240.0),
            ('checking.settle_time', 0.5), ('checking.frames', 5),
            ('max_retries', 1), ('backout.enabled', False), ('min_retry_backoff', 1.00), ('retry_stable_frames', 5),
            ('retry_stable_spread', 0.03),
            ('go_nogo.min_face_inliers', 60),
            ('go_nogo.max_yaw_error', 0.087), ('go_nogo.max_lateral_offset', 0.050),
            ('go_nogo.pocket_width_tolerance', 0.040),
            ('go_nogo.pocket_height_tolerance', 0.030),
            ('go_nogo.min_pocket_count', 2), ('go_nogo.overhang_margin', 0.020),
            ('go_nogo.check_overhang', True), ('go_nogo.check_pocket_clearance', True),
            ('go_nogo.pocket_clearance_depth', 0.10),
            ('go_nogo.stability_frames', 5), ('go_nogo.max_position_spread', 0.030),
            ('go_nogo.max_yaw_spread', 0.035),
            ('closing.k_cross', 2.5), ('closing.max_approach_angle', 0.25),
            ('closing.min_run_up', 3.0), ('closing.residual_target', 0.020),
            ('closing.run_up_margin', 0.20), ('closing.run_up_tolerance', 0.05),
            ('closing.range_tolerance', 0.03),
            ('reverse.max_unverified', 0.50),
            ('forks.bottom_height', 0.040), ('forks.tine_thickness', 0.040),
            ('forks.lift_travel', 0.070),
            # One literal name per line, so test_params.py's scraper sees each.
            ('pallet.sides.short.pocket_width', 0.230),
            ('pallet.sides.short.opening_top', 0.100),
            ('pallet.sides.short.opening_bottom', 0.000),
            ('pallet.sides.short.insertable', True),
            ('pallet.sides.short.depth', 1.195),
            ('pallet.sides.long.pocket_width', 0.375),
            ('pallet.sides.long.opening_top', 0.125),
            ('pallet.sides.long.opening_bottom', 0.020),
            ('pallet.sides.long.insertable', True),
            ('pallet.sides.long.depth', 0.795),
        ):
            self.declare_parameter(name, default)
        self.declare_parameter('retryable_checks',
                               ['yaw', 'lateral_offset', 'stability'])

    def _get(self, name):
        return self.get_parameter(name).value

    def _group_params(self, prefix, keys):
        return {key: self._get(f'{prefix}.{key}') for key in keys}

    def _sides(self) -> dict:
        return {
            side: self._group_params(f'pallet.sides.{side}', (
                'pocket_width', 'opening_top', 'opening_bottom', 'insertable',
                'depth'))
            for side in ('short', 'long')
        }

    def _machine_params(self) -> dict:
        return machine_params(self._get)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # -------------------------------------------------- lifecycle plumbing

    def _transition(self, node_name: str, transition_id: int) -> bool:
        client = self._managed_clients[node_name]
        if not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error(f'{node_name}/change_state is not available')
            return False
        request = ChangeState.Request()
        request.transition.id = transition_id
        future = client.call_async(request)
        # Wait, do not spin: spin_once would move this node to the global
        # executor and leave main()'s executor spinning nothing (sim,
        # 2026-10-20). Wall-time deadline, so it holds if the clock stops.
        deadline = time.monotonic() + 5.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not future.done():
            self.get_logger().error(f'{node_name} transition {transition_id} timed out')
            return False
        return bool(future.result().success)

    def _set_active(self, node_name: str, active: bool) -> bool:
        """Activate or deactivate, and only when it is not already that way.

        Repeating an ACTIVATE on an active node fails the transition, and the
        machine republishes what it wants on every tick, so without this check
        every tick would log a lifecycle error.
        """
        if self._active_nodes.get(node_name) == active:
            return True
        ok = self._transition(
            node_name,
            LifecycleTransition.TRANSITION_ACTIVATE if active
            else LifecycleTransition.TRANSITION_DEACTIVATE)
        if ok:
            self._active_nodes[node_name] = active
        return ok

    def _lifecycle_state(self, node_name: str) -> str | None:
        """The node's current lifecycle state label, or None if it does not
        answer (which is also what a dead process looks like)."""
        client = self._state_clients[node_name]
        if not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error(f'{node_name}/get_state is not available')
            return None
        future = client.call_async(GetState.Request())
        deadline = time.monotonic() + 5.0       # waited, not spun: see _transition
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not future.done():
            self.get_logger().error(f'{node_name}/get_state timed out')
            return None
        return future.result().current_state.label

    def _configure_all(self) -> bool:
        """Bring every managed node to `inactive`, from whatever state it is in."""
        ids = {'configure': LifecycleTransition.TRANSITION_CONFIGURE,
               'deactivate': LifecycleTransition.TRANSITION_DEACTIVATE}
        for name in MANAGED:
            state = self._lifecycle_state(name)
            steps = None if state is None else transitions_to_inactive(state)
            if steps is None:
                self.get_logger().error(
                    f'{name} is {state or "not answering"}; a run cannot start '
                    f'from there')
                return False
            for step in steps:
                if not self._transition(name, ids[step]):
                    self.get_logger().error(f'{name}: {step} from {state} failed')
                    return False
            self._active_nodes[name] = False
        return True

    # ------------------------------------------------------------- applying

    def _apply(self, transition) -> None:
        """Make a Transition from the machine true in the system.

        Order matters: stop the robot first, then change activations. Stopping
        before deactivating means the zero command goes out even if a lifecycle
        call then fails.
        """
        if transition.stop_robot:
            self._send_drive(None)
        if transition.outcome is not None and transition.changed:
            self._last_outcome = transition.outcome
            self._last_reason = transition.reason

        if transition.changed:
            if transition.state is State.CHECKING:
                self._checking_estimates.clear()
            self._state_pub.publish(String(data=transition.state.value))
            self.get_logger().info(
                f'{transition.state.value}'
                + (f': {transition.reason}' if transition.reason else ''))
            # Fused mode (D104): pallet_fusion runs on its own; the old
            # camera pipeline stays inactive and costs nothing.
            self._set_active('pallet_perception',
                             transition.perception_active and self._source == 'camera')
            self._set_active('approach_controller',
                             transition.drive is not None)
            if transition.fork is not None:
                self._fork_pub.publish(String(data=transition.fork))

        if transition.drive is not None:
            self._send_drive(transition.drive)

    def _send_drive(self, request) -> None:
        """Publish the drive command, or a STOP.

        Republished every tick while a move is in progress: the controller
        stops if its command goes stale (`target_timeout`), which is what makes
        a coordinator that dies leave a stationary robot.
        """
        message = DriveCommand()
        message.header.stamp = self.get_clock().now().to_msg()
        if request is None:
            message.mode = 'STOP'
        else:
            message.mode = request.mode
            message.distance = float(request.distance)
            message.side = str(request.side)
        self._drive_pub.publish(message)

    # ------------------------------------------------------------ callbacks

    def _in_odom(self, estimate) -> tuple:
        """This frame's pocket midpoint (or face centre) in odom, using the
        newest odometry pose. Lag is ~0.3 s; at the 0.05 m/s reverse that is a
        constant ~15 mm offset, not scatter, so the D89 stability test is not
        affected by it."""
        point = pocket_center(estimate.pockets, fallback=estimate.position)
        c, s = math.cos(self._pose.yaw), math.sin(self._pose.yaw)
        return (self._pose.x + c * point[0] - s * point[1],
                self._pose.y + s * point[0] + c * point[1])

    @_serialised
    def _on_odom(self, msg: Odometry):
        self._pose = pose2d_from_odometry(msg)

    @_serialised
    def _on_health(self, msg: Health):
        self._health = msg
        if not msg.ok and msg.phase == 'running':
            self._apply(self._machine.on_unhealthy(
                self._now(), '; '.join(msg.failures) or 'unknown'))

    @_serialised
    def _on_start(self, msg: String):
        """The trigger topic. Fire and forget, so every refusal is logged AND
        published on `~/result` -- there is nobody waiting on a return value."""
        mode = (msg.data or '').strip() or PICKUP
        if self._machine.state not in (State.IDLE, State.DONE, State.ABORTED):
            self.get_logger().warning(
                f'ignoring a start on {mode!r}: a run is already in progress '
                f'({self._machine.state.value}). Cancel it first.')
            return
        if mode not in MODES:
            self._refuse(f'unknown mode {mode!r}; known: {", ".join(MODES)}', mode)
            return
        reason = self._unhealthy_reason() if self._get('require_healthy_to_start') else ''
        if reason:
            self._refuse(reason, mode)
            return
        if not self._start_run(mode):
            self._refuse('could not configure the managed nodes', mode)

    @_serialised
    def _on_cancel_topic(self, msg: Empty):
        """Stop a run started from the trigger topic.

        Not a convenience: a topic-started run has no action goal, so without
        this there would be no way to honour a cancel -- and that rule has no
        exceptions.
        """
        self.get_logger().info('cancel requested on the topic')
        self._apply(self._machine.cancel(self._now()))

    def _start_run(self, mode: str) -> bool:
        """Begin a run. The one path both the topic and the action take, so
        the health gate and the configure step cannot apply to only one."""
        self._estimate = None
        self._tracked = None
        self._cluster_points = None
        self._scene_points = None
        self._checking_estimates.clear()
        self._last_outcome = None
        self._last_reason = ''
        self._reported = False
        self._run_started_at = self._now()
        if not self._configure_all():
            return False
        self._apply(self._machine.trigger(self._now(), mode))
        return True

    def _refuse(self, reason: str, mode: str) -> None:
        self.get_logger().error(f'refusing to start {mode!r}: {reason}')
        self._last_outcome = Outcome.ABORTED
        self._last_reason = reason
        self._reported = True
        result = self._result(Outcome.ABORTED, reason)
        # The machine was never triggered, so it still holds the PREVIOUS
        # run's mode and side; report what was asked for, and no side.
        result.mode = mode
        result.side = ''
        self._result_pub.publish(result)

    def _on_cluster_points(self, msg: PointCloud2):
        self._cluster_points = unpack_cloud(msg)

    def _on_scene_points(self, msg: PointCloud2):
        self._scene_points = unpack_cloud(msg)

    @_serialised
    def _on_tracked(self, msg: TrackedPallet):
        estimate = estimate_from_msg(msg.estimate)
        if estimate is None:
            self._apply(self._machine.on_track(
                TrackInfo(confirmed=False, confidence=float(msg.confidence),
                          side='unknown', range=float('nan'), offset=float('nan'),
                          heading=float('nan'), reason=msg.reason),
                self._now()))
            return
        self._estimate = estimate
        self._tracked = msg
        if bool(msg.confirmed) and self._machine.state is State.CHECKING:
            self._checking_estimates.append(estimate)
        geometry = geometry_of(estimate)
        self._apply(self._machine.on_track(
            TrackInfo(confirmed=bool(msg.confirmed),
                      confidence=float(msg.confidence),
                      side=str(estimate.side),
                      range=float(geometry.range),
                      offset=float(geometry.offset),
                      heading=float(geometry.heading),
                      reason=msg.reason,
                      pallet_odom=self._in_odom(estimate),
                      disagreeing=(self._source == 'fused' and msg.status == 'DISAGREE')),
            self._now()))

    @_serialised
    def _on_drive_status(self, msg: DriveStatus):
        self._apply(self._machine.on_drive(
            DriveInfo(mode=msg.mode, arrived=bool(msg.arrived),
                      blocked=bool(msg.blocked), abort=bool(msg.abort),
                      reason=msg.reason, travelled=float(msg.travelled)),
            self._now()))

    @_serialised
    def _on_fork_state(self, msg: ForkState):
        self._apply(self._machine.on_forks(
            ForkInfo(state=msg.state, height=float(msg.height),
                     moving=bool(msg.moving), loaded=bool(msg.loaded)),
            self._now()))

    @_serialised
    def _tick(self):
        now = self._now()
        if self._machine.state not in (State.IDLE, State.DONE, State.ABORTED):
            # Held for the length of the run so twist_mux blocks Nav2 (D30).
            self._lock_pub.publish(Bool(data=True))
        self._apply(self._machine.on_tick(now))
        if self._machine.ready_to_decide(now):
            self._apply(self._machine.on_decision(self._decide(), now))

        # Terminal handling lives here rather than in the action callback, so
        # that a run started from the topic reports its outcome and returns to
        # IDLE exactly like one started from a goal.
        if self._last_outcome is not None and not self._reported:
            self._reported = True
            self._result_pub.publish(
                self._result(self._last_outcome, self._last_reason))
            if self._goal_handle is None:
                self._apply(self._machine.to_idle(now))

        self._publish_status(now)

    def _publish_status(self, now: float) -> None:
        """The live telemetry an action goal's feedback cannot reach: a
        passive observer (the dashboard, rqt, a log) that was not the one who
        sent the goal, or was watching before any goal existed at all.
        Published every tick (D55's `/coordinator/status`) -- cheap, and the
        only way anything downstream can see the CURRENT reason rather than
        only the final one on `~/result`.
        """
        status = CoordinatorStatus()
        status.header.stamp = self.get_clock().now().to_msg()
        status.state = self._machine.state.value
        status.mode = self._machine.mode
        status.side = self._machine.side
        status.reason = self._machine.last_reason
        status.outcome = self._last_outcome.value if self._last_outcome else ''
        status.elapsed = (float('nan') if self._run_started_at is None
                          else float(now - self._run_started_at))
        self._status_pub.publish(status)

    # -------------------------------------------------------------- GO/NO-GO

    def _decide(self) -> Decision:
        params = self._group_params('go_nogo', (
            'min_face_inliers', 'max_yaw_error', 'max_lateral_offset',
            'pocket_width_tolerance', 'pocket_height_tolerance',
            'min_pocket_count', 'overhang_margin', 'check_overhang',
            'check_pocket_clearance',
            'pocket_clearance_depth',
            'stability_frames', 'max_position_spread', 'max_yaw_spread'))
        params['sides'] = self._sides()
        if self._source == 'fused':
            # D104: the end of the decision tree -- a ready track, then alignment.
            tracked = self._tracked
            return go_nogo.evaluate_fused(
                self._estimate, Pose2D(0.0, 0.0, 0.0),
                bool(tracked is not None and tracked.lidar_agrees),
                tracked.reason if tracked is not None else 'no track', params)
        try:
            # The rules work in the estimate's own frame, so the robot is at
            # the origin: perception publishes in base_footprint and the
            # tolerances are about where the robot is relative to the pallet,
            # not about where either is in odom.
            return go_nogo.evaluate(self._estimate, Pose2D(0.0, 0.0, 0.0),
                                    self._estimate_history(), params,
                                    load_points=self._cluster_points,
                                    scene_points=self._scene_points)
        except Exception as error:
            self.get_logger().error(f'go_nogo raised: {error!r}')
            return Decision(go=False, reason=f'go_nogo raised {type(error).__name__}')

    def _estimate_history(self):
        """The estimates taken since CHECKING began, oldest first, for the stability
        rule (never one frame repeated, which could not fail).
        """
        return list(self._checking_estimates)

    # --------------------------------------------------------------- action

    @_serialised
    def _on_goal(self, goal_request):
        if self._machine.state not in (State.IDLE, State.DONE, State.ABORTED):
            self.get_logger().warning('rejecting goal: a run is already in progress')
            return GoalResponse.REJECT
        if (goal_request.mode or '').strip() not in MODES + ('',):
            self.get_logger().warning(
                f'rejecting unknown mode {goal_request.mode!r}; known: '
                f'{", ".join(MODES)}')
            return GoalResponse.REJECT
        if self._get('require_healthy_to_start'):
            reason = self._unhealthy_reason()
            if reason:
                self.get_logger().warning(f'rejecting goal: {reason}')
                return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _unhealthy_reason(self) -> str:
        """D12's gate. Returns '' when it is safe to start."""
        if self._health is None:
            return ('no /health message yet -- the watchdog is not running, so '
                    'nothing is watching the sensors')
        age = self._now() - (float(self._health.header.stamp.sec)
                             + float(self._health.header.stamp.nanosec) * 1e-9)
        if age > 2.0:
            return f'the last /health message is {age:.1f} s old'
        if not self._health.ok:
            return 'stack unhealthy: ' + '; '.join(self._health.failures)
        return ''

    @_serialised
    def _on_cancel(self, goal_handle):
        self.get_logger().info('cancel requested')
        self._apply(self._machine.cancel(self._now()))
        return CancelResponse.ACCEPT

    def _execute(self, goal_handle):
        """The action path. Same run, with feedback and a return value.

        It waits on `_last_outcome` rather than on the state, because `_tick`
        owns the terminal handling and may already have published the result by
        the time this loop looks again.
        """
        self._goal_handle = goal_handle
        mode = (goal_handle.request.mode or '').strip() or PICKUP

        with self._machine_lock:
            started = self._start_run(mode)
        if not started:
            self._goal_handle = None
            goal_handle.abort()
            return self._result(Outcome.ABORTED,
                                'could not configure the managed nodes')

        while rclpy.ok() and self._last_outcome is None:
            if goal_handle.is_cancel_requested:
                with self._machine_lock:
                    self._apply(self._machine.cancel(self._now()))
                break
            with self._machine_lock:
                feedback = self._feedback()
            goal_handle.publish_feedback(feedback)
            self.get_clock().sleep_for(rclpy.duration.Duration(seconds=0.1))

        outcome = self._last_outcome or Outcome.ABORTED
        reason = self._last_reason or 'interrupted'
        cancelled = goal_handle.is_cancel_requested

        with self._machine_lock:
            self._apply(self._machine.to_idle(self._now()))
            result = self._result(outcome, reason)
        # The lock is not republished from here on, so it expires 0.5 s from
        # now and Nav2 can drive again.

        if cancelled:
            goal_handle.canceled()
        else:
            goal_handle.succeed()
        self._goal_handle = None
        return result

    def _feedback(self):
        feedback = PalletTask.Feedback()
        feedback.phase = self._machine.state.value
        feedback.reason = self._machine.last_reason
        feedback.distance_to_standoff = float('nan')
        feedback.lateral_offset = float('nan')
        feedback.confidence = 0.0
        track = self._machine.track
        if track is not None:
            feedback.distance_to_standoff = float(
                track.range - self._get('standoff_distance'))
            feedback.lateral_offset = float(track.offset)
            feedback.confidence = float(track.confidence)
        return feedback

    def _result(self, outcome: Outcome, reason: str):
        result = PalletTask.Result()
        result.outcome = outcome.value
        result.reason = reason
        result.side = self._machine.side
        result.insertion_depth = 0.0
        result.pallet_lift = 0.0
        result.mode = self._machine.mode
        if outcome is Outcome.INSERTED and self._machine.side in self._sides():
            # Report what the run achieved, not that it finished. The lift is
            # the carriage travel minus the slack above the tines, which
            # differs between the faces (D44) -- "the pallet came off the
            # floor" is the claim, and it is a different number per face.
            result.insertion_depth = float(self._machine.insertion_travel)
            result.pallet_lift = float(pallet_lift(
                self._sides()[self._machine.side],
                {'insert_height': self._get('forks.bottom_height'),
                 'tine_thickness': self._get('forks.tine_thickness'),
                 'lift_travel': self._get('forks.lift_travel')}))
        result.pallet_pose.header.stamp = self.get_clock().now().to_msg()
        result.pallet_pose.header.frame_id = self._get('frames.ground')
        if self._estimate is not None:
            center = pocket_center(self._estimate.pockets, self._estimate.position)
            result.pallet_pose.pose = pose2d_to_pose(
                Pose2D(float(center[0]), float(center[1]),
                       float(self._estimate.yaw)),
                z=float(center[2]))
            self._pose_pub.publish(result.pallet_pose)
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
