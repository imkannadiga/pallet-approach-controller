"""The virtual fork carriage, as a node.

There is no hardware, so this node's only job is to be honest about a thing
that does not exist: it holds a height, moves it at a finite speed, and
publishes what it is doing. The coordinator waits for it exactly as it would
wait for a real carriage, which is the point -- if the sequence can be driven
by this, it can be driven by a carriage.

Commands arrive on `~/command` as plain strings so they can also be given by
hand from a terminal, which is useful on the robot:

    ros2 topic pub --once /virtual_forks/command std_msgs/String "{data: LOWER}"

An unknown command is refused loudly rather than ignored. `LIFT` is refused
unless the forks are at the insertion height, because lifting from travel
height would raise a carriage that is nowhere near a pallet and would still
hand the sequence a `loaded` flag.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from pallet_approach_interfaces.msg import ForkState

from pallet_approach.forks import ForkSimulator

RELIABLE = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
COMMANDS = ('PARK', 'LOWER', 'LIFT')


class ForksNode(Node):

    def __init__(self):
        super().__init__('virtual_forks')
        self._declare()
        self._forks = ForkSimulator({
            'park_height': self.get_parameter('park_height').value,
            'insert_height': self.get_parameter('insert_height').value,
            'lift_travel': self.get_parameter('lift_travel').value,
            'speed': self.get_parameter('speed').value,
            'tine_thickness': self.get_parameter('forks.tine_thickness').value,
        })
        self._pub = self.create_publisher(ForkState, '~/state', RELIABLE)
        self.create_subscription(String, '~/command', self._on_command, RELIABLE)
        rate = float(self.get_parameter('publish_rate').value)
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f'virtual forks at {self._forks.height * 1000:.0f} mm; '
            f'commands: {", ".join(COMMANDS)}')

    def _declare(self):
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('park_height', 0.150)
        self.declare_parameter('insert_height', 0.040)
        self.declare_parameter('lift_travel', 0.070)
        self.declare_parameter('speed', 0.05)
        self.declare_parameter('forks.tine_thickness', 0.040)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_command(self, msg: String):
        command = msg.data.strip().upper()
        now = self._now()
        if command == 'PARK':
            self._forks.park(now)
        elif command == 'LOWER':
            self._forks.lower_to_insert(now)
        elif command == 'LIFT':
            if not self._forks.lift(now):
                self.get_logger().error(
                    f'refusing to LIFT: the forks are {self._forks.describe()}, '
                    f'not at the insertion height. Lifting from here would '
                    f'raise a carriage that is nowhere near a pallet.')
                return
        else:
            self.get_logger().error(
                f'unknown fork command {msg.data!r}; known: {", ".join(COMMANDS)}')
            return
        self.get_logger().info(f'{command}: {self._forks.describe()}')

    def _tick(self):
        self._forks.update(self._now())
        state = ForkState()
        state.header.stamp = self.get_clock().now().to_msg()
        state.height = float(self._forks.height)
        state.target_height = float(self._forks.target)
        state.state = self._forks.state.value
        state.moving = bool(self._forks.moving)
        state.loaded = bool(self._forks.loaded)
        self._pub.publish(state)


def main(args=None):
    rclpy.init(args=args)
    node = ForksNode()
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
