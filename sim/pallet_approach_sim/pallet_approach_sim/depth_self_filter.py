"""Remove the forklift's own forks from the simulated depth image.

The forklift's sensors sit where the SCOUT's do and its body is hidden from
them with visibility flags (D67). Fortress's colour camera and GPU LiDAR honour
that; its depth camera does not (measured 2026-10-22): the two fork blades,
0.46-1.66 m ahead of base_footprint, came out of floor removal as two clusters
bigger than the pallet's face, perception picked one, and every run aborted
with no pallet found. Everything else of the truck in the camera's view is
within the depth camera's 0.3 m minimum range and is already cut.

So, like camera_info_fix, this corrects a Gazebo gap on the ROS side: each
depth pixel whose point falls inside a fork blade (at the carriage's live
height, from /joint_states) becomes +inf, the ZED's "no depth" value, and the
image is republished on the ZED's topic. With no blades configured (the SCOUT)
it passes images through unchanged.

It is also where Gazebo's exact depth is made to look like the ZED's
(`zed_model`, D80; see zed_depth_model.py): a smooth range-dependent warp,
filled pockets and smeared edges, as measured on the robot's bags. That costs
~50 ms a frame, so frames are published at up to `max_rate` (10 Hz) -- more
than perception's 5 Hz, which is all it uses.
"""

import array
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, JointState

from pallet_approach_sim.depth_mask import mask_boxes, optical_to_base
from pallet_approach_sim.zed_depth_model import DepthModel


class DepthSelfFilter(Node):
    def __init__(self):
        super().__init__('sim_depth_self_filter')
        self.declare_parameter('input', '/sim/zed/depth_gazebo')
        self.declare_parameter('output', '/zed2/zed_node/depth/depth_registered')
        self.declare_parameter('camera_xyz', [0.39, 0.06, 0.373])
        self.declare_parameter('hfov', 1.1865)
        # Fork blades, in base_footprint with the carriage at its resting pose:
        # flattened (xmin, xmax, ymin, ymax, zmin, zmax) per blade. Empty: no
        # forks (the SCOUT), pass through.
        self.declare_parameter('blades', [0.0])
        self.declare_parameter('lift_joint', 'fork_lift')
        self.declare_parameter('margin', 0.02)
        self.declare_parameter('zed_model', True)
        self.declare_parameter('max_rate', 10.0)
        self.declare_parameter('seed', 0)
        flat = list(self.get_parameter('blades').value)
        self._blades = ([flat[i:i + 6] for i in range(0, len(flat), 6)]
                        if len(flat) >= 6 else [])
        self._camera = optical_to_base(tuple(self.get_parameter('camera_xyz').value))
        self._hfov = float(self.get_parameter('hfov').value)
        self._margin = float(self.get_parameter('margin').value)
        self._lift_joint = self.get_parameter('lift_joint').value
        self._lift = 0.0
        seed = int(self.get_parameter('seed').value)
        self._model = (DepthModel(seed=seed if seed >= 0 else None)
                       if self.get_parameter('zed_model').value else None)
        self._period = 1.0 / float(self.get_parameter('max_rate').value)
        self._next_due = None
        self._pub = self.create_publisher(Image, self.get_parameter('output').value, 10)
        self.create_subscription(JointState, '/joint_states', self._on_joints, 10)
        # The depth model's warp changes with the view (zed_depth_model.py):
        # progress = metres driven / 0.5 + radians turned / 0.17.
        self._progress = 0.0
        self._odom_last = None
        self.create_subscription(Odometry, '/odom', self._on_odom, 10)
        self.create_subscription(Image, self.get_parameter('input').value, self._on_depth,
                                 10, raw=True)
        self.get_logger().info(
            f'masking {len(self._blades)} fork blade(s) from the depth image; ZED depth '
            f'model {"on" if self._model else "off"}, at up to {1.0 / self._period:.0f} Hz')

    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose
        yaw = 2.0 * math.atan2(p.orientation.z, p.orientation.w)
        if self._odom_last is not None:
            x0, y0, yaw0 = self._odom_last
            turn = (yaw - yaw0 + math.pi) % (2.0 * math.pi) - math.pi
            self._progress += math.hypot(p.position.x - x0, p.position.y - y0) / 0.5
            self._progress += abs(turn) / 0.17
        self._odom_last = (p.position.x, p.position.y, yaw)

    def _on_joints(self, msg: JointState):
        if self._lift_joint in msg.name:
            self._lift = float(msg.position[msg.name.index(self._lift_joint)])

    def _on_depth(self, raw: bytes):
        msg = deserialize_message(raw, Image)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self._model is not None:
            # An average rate, as rate_limit.Schedule does: "one period since
            # the last" turned 10 Hz into 7.5 Hz from Gazebo's 30 Hz frames.
            if self._next_due is not None and stamp < self._next_due:
                return
            start = self._next_due
            if start is None or stamp - start > self._period:
                start = stamp
            self._next_due = start + self._period
        if not self._blades and self._model is None:
            self._pub.publish(msg)
            return
        depth = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width)
        fx = (msg.width / 2) / math.tan(self._hfov / 2)
        if self._blades:
            m = self._margin
            boxes = [(x0 - m, x1 + m, y0 - m, y1 + m, z0 + self._lift - m, z1 + self._lift + m)
                     for x0, x1, y0, y1, z0, z1 in self._blades]
            depth = mask_boxes(depth, fx, fx, msg.width / 2, msg.height / 2, self._camera, boxes)
        if self._model is not None:
            depth = self._model.apply(depth, fx, self._progress)
        msg.data = array.array('B', np.asarray(depth, np.float32).tobytes())
        self._pub.publish(msg)


def main():
    rclpy.init()
    node = DepthSelfFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
