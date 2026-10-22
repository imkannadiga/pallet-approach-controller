"""Drive the forklift's physical carriage from the virtual forks' height.

The stack's virtual_forks node is the carriage the SCOUT does not have: it
publishes `height`, the tine underside above the floor, as it lowers, holds
and lifts. On the simulated forklift that same number moves a real prismatic
joint (Athreya, 2026-10-22: physical lift), so the pallet comes off the floor
when, and as far as, the stack thinks it does. The joint's zero is the mesh's
resting pose, whose tine underside is `rest_height` above the floor.

While the virtual forks are PARKED (the whole approach), the physical carriage
instead travels at `travel_height`, 0.45 m (D71, from D13: travel lift height
at least 0.40 m). The camera sits 0.398 m up where the SCOUT's does, and forks
parked at 0.15 m stick out 1.66 m in front of it, between it and the pockets:
as the truck closed in they hid a growing strip of the face, the side read as
"ambiguous", and the tracker dropped the pallet at ~2.4 m in 5 of 5 runs
(2026-10-22). Raised above the camera they are out of its line of sight. Once
the virtual forks move (lowering, inserting, lifting), the carriage follows
their height; the camera is not used then (D64).
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

from pallet_approach_interfaces.msg import ForkState
from pallet_approach_sim.forklift_model import carriage_target


class ForkLiftBridge(Node):
    def __init__(self):
        super().__init__('sim_fork_lift_bridge')
        self.declare_parameter('rest_height', 0.1318)
        self.declare_parameter('travel_height', 0.45)
        self._rest = float(self.get_parameter('rest_height').value)
        self._travel = float(self.get_parameter('travel_height').value)
        self._pub = self.create_publisher(Float64, '/sim/fork_lift', 10)
        self.create_subscription(ForkState, '/virtual_forks/state', self._on_state, 10)

    def _on_state(self, msg: ForkState):
        self._pub.publish(Float64(data=carriage_target(
            msg.state, float(msg.height), float(msg.target_height),
            self._rest, self._travel)))


def main():
    rclpy.init()
    node = ForkLiftBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
