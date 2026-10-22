"""Composite the fork overlay with a HUD, and publish it for the video stream.

A plain node, like `fork_overlay_node`: it only reads topics and publishes an
image, so there is nothing to gate and it runs from the moment the container
starts. It subscribes to six topics -- the overlay image plus five telemetry
topics -- caches the latest of each with its arrival time, and redraws the
whole frame on its own timer rather than whenever any one of them happens to
update. That decoupling is deliberate: it means the HUD keeps refreshing (and
can show a source going stale) even if one of its inputs stalls, instead of
the whole picture freezing with the one input that died.

The drawing itself is `dashboard.render`, which is pure and takes everything
it needs as arguments -- this file's only job is to keep that function fed.
"""

from __future__ import annotations

import array as pyarray

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from pallet_approach_interfaces.msg import (
    CoordinatorStatus,
    DriveStatus,
    ForkState,
    Health,
    TrackedPallet,
)

from pallet_approach import dashboard
from pallet_approach.nodes._common import image_to_array

RELIABLE = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)


class DashboardNode(Node):

    def __init__(self):
        super().__init__('dashboard')
        self._declare()
        self._snapshot = dashboard.Snapshot()
        self._image = None

        self._publisher = self.create_publisher(Image, '~/image', RELIABLE)

        self.create_subscription(Image, '/fork_overlay/image', self._on_image,
                                 RELIABLE)
        self.create_subscription(CoordinatorStatus, '/coordinator/status',
                                 self._on_coordinator, RELIABLE)
        self.create_subscription(Health, '/health', self._on_health, RELIABLE)
        self.create_subscription(TrackedPallet, '/pallet_tracker/tracked',
                                 self._on_tracked, RELIABLE)
        self.create_subscription(DriveStatus, '/approach_controller/status',
                                 self._on_controller, RELIABLE)
        self.create_subscription(ForkState, '/virtual_forks/state',
                                 self._on_forks, RELIABLE)

        rate = float(self.get_parameter('publish_rate').value)
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            'dashboard ready; /dashboard/image composites /fork_overlay/image '
            'with coordinator, health, tracker, controller and fork telemetry')

    def _declare(self):
        self.declare_parameter('publish_rate', 5.0)
        self.declare_parameter('placeholder_size', [640, 360])
        self.declare_parameter('pad', 8)
        self.declare_parameter('line_height', 15)
        self.declare_parameter('font_scale', 0.40)
        for name, default in (
            ('stale_after.image', 1.0), ('stale_after.coordinator', 1.0),
            ('stale_after.health', 1.5), ('stale_after.tracked', 1.0),
            ('stale_after.controller', 0.5), ('stale_after.forks', 1.0),
        ):
            self.declare_parameter(name, default)
        # The SAME shared `colors` block fork_overlay_node reads (params.yaml
        # `/**`), not a copy: a node block is only visible to the node it is
        # named after, so the one way two differently-named nodes can agree on
        # a palette is to both declare the same key out of the shared block.
        # Declared one literal name at a time, like fork_overlay_node's own
        # copy -- see that node's comment on 'colors.inserting' for why an
        # f-string loop here would be invisible to test_params.py's scraper.
        for name, default in (
            ('colors.idle', [160, 160, 160]), ('colors.detecting', [160, 160, 160]),
            ('colors.approaching', [0, 200, 255]), ('colors.inserting', [128, 255, 0]),
            ('colors.go', [0, 200, 0]), ('colors.no_go', [0, 0, 220]),
        ):
            self.declare_parameter(name, default)

    def _params(self) -> dict:
        get = lambda n: self.get_parameter(n).value  # noqa: E731
        return {
            'placeholder_size': tuple(get('placeholder_size')),
            'pad': get('pad'), 'line_height': get('line_height'),
            'font_scale': get('font_scale'),
            'stale_after': {
                'image': get('stale_after.image'),
                'coordinator': get('stale_after.coordinator'),
                'health': get('stale_after.health'),
                'tracked': get('stale_after.tracked'),
                'controller': get('stale_after.controller'),
                'forks': get('stale_after.forks'),
            },
            'colors': {
                'idle': tuple(int(v) for v in get('colors.idle')),
                'detecting': tuple(int(v) for v in get('colors.detecting')),
                'approaching': tuple(int(v) for v in get('colors.approaching')),
                'inserting': tuple(int(v) for v in get('colors.inserting')),
                'go': tuple(int(v) for v in get('colors.go')),
                'no_go': tuple(int(v) for v in get('colors.no_go')),
            },
        }

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # ----------------------------------------------------------- callbacks

    def _on_image(self, msg: Image):
        self._image = image_to_array(msg)
        self._snapshot.image_at = self._now()

    def _on_coordinator(self, msg: CoordinatorStatus):
        self._snapshot.coordinator = dashboard.CoordinatorInfo(
            state=msg.state, mode=msg.mode, side=msg.side, reason=msg.reason,
            outcome=msg.outcome, elapsed=float(msg.elapsed))
        self._snapshot.coordinator_at = self._now()

    def _on_health(self, msg: Health):
        self._snapshot.health = dashboard.HealthInfo(
            ok=bool(msg.ok), failures=tuple(msg.failures),
            stop_latched=bool(msg.stop_latched))
        self._snapshot.health_at = self._now()

    def _on_tracked(self, msg: TrackedPallet):
        self._snapshot.tracked = dashboard.TrackedInfo(
            status=msg.status, confidence=float(msg.confidence),
            consecutive_frames=int(msg.consecutive_frames),
            total_frames=int(msg.total_frames),
            lidar_evaluated=bool(msg.lidar_evaluated),
            lidar_agrees=bool(msg.lidar_agrees),
            lidar_offset=float(msg.lidar_offset),
            lidar_points=int(msg.lidar_points))
        self._snapshot.tracked_at = self._now()

    def _on_controller(self, msg: DriveStatus):
        self._snapshot.controller = dashboard.ControllerInfo(
            mode=msg.mode, phase=msg.phase, arrived=bool(msg.arrived),
            blocked=bool(msg.blocked), abort=bool(msg.abort),
            face_range=float(msg.face_range),
            lateral_offset=float(msg.lateral_offset),
            heading_error=float(msg.heading_error),
            rear_clearance=float(msg.rear_clearance))
        self._snapshot.controller_at = self._now()

    def _on_forks(self, msg: ForkState):
        self._snapshot.forks = dashboard.ForksInfo(
            state=msg.state, height=float(msg.height),
            target_height=float(msg.target_height), moving=bool(msg.moving),
            loaded=bool(msg.loaded))
        self._snapshot.forks_at = self._now()

    # ---------------------------------------------------------------- tick

    def _tick(self):
        self._snapshot.now = self._now()
        annotated = dashboard.render(self._image, self._snapshot, self._params())

        out = Image()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'dashboard'
        out.height, out.width = annotated.shape[0], annotated.shape[1]
        out.encoding = 'bgr8'
        out.is_bigendian = 0
        out.step = out.width * 3
        # array.array, not bytes: rclpy checks a bytes value element by element in
        # Python (58.5 ms for a 640x360 image, measured 2026-10-22), an array.array
        # of typecode 'B' is copied in C (1.3 ms).
        out.data = pyarray.array('B', annotated.tobytes())
        self._publisher.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = DashboardNode()
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
