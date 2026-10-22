"""Drive the forklift's physical carriage from the virtual forks' height.

The stack's virtual_forks node is the carriage the SCOUT does not have: it
publishes `height`, the tine underside above the floor, as it lowers, holds
and lifts. On the simulated forklift that same number moves a real prismatic
joint (Athreya, 2026-10-22: physical lift), so the pallet comes off the floor
when, and as far as, the stack thinks it does. The joint's zero is the mesh's
resting pose, whose tine underside is `rest_height` above the floor.

"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

from pallet_approach_interfaces.msg import ForkState


class ForkLiftBridge(Node):
    def __init__(self):
        super().__init__('sim_fork_lift_bridge')
        self.declare_parameter('rest_height', 0.1318)
        self._rest = float(self.get_parameter('rest_height').value)
        self._pub = self.create_publisher(Float64, '/sim/fork_lift', 10)
        self.create_subscription(ForkState, '/virtual_forks/state', self._on_state, 10)

    def _on_state(self, msg: ForkState):
        self._pub.publish(Float64(data=float(msg.height) - self._rest))

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
