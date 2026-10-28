"""ROS 2 node: bring the LiDAR onto the Jetson once, for every node that reads it.

FastDDS sends a separate copy of each message over the network to every
subscribing process. On the robot the watchdog, the tracker and the controller
each subscribed to /ouster/points from the robot PC, and an OS0-64 cloud is
~3 MB at 10 Hz (~250 Mbit/s a copy). Measured 2026-10-22: the Jetson's
1 Gbit/s link received 520 Mbit/s at idle with two subscribers (three during a
run), and everything else from the robot PC queued behind it -- clouds arrived
0.37 s old and at 7.1 Hz instead of 10, and odometry stalled for up to 1 s.

This node is the only subscriber to `topics.lidar_source` on the Jetson and
republishes each message, still serialized, on `topics.lidar`; the other
nodes read that local copy. Same messages, same stamps: nothing downstream
changes but the topic name. Best effort both ways, depth 1: a slow reader
drops a cloud rather than delaying the next.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2

from pallet_approach.cdr import restamp

BEST_EFFORT = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)


class LidarRelay(Node):

    def _declare(self):
        self.declare_parameter('topics.lidar_source', '/ouster/points')
        self.declare_parameter('topics.lidar', '/pallet_approach/lidar')
        self.declare_parameter('frames.lidar_calibration.enabled', False)
        self.declare_parameter('frames.lidar_calibration.frame', 'lidar_calibrated')

    def __init__(self):
        super().__init__('lidar_relay')
        self._declare()
        source = self.get_parameter('topics.lidar_source').value
        target = self.get_parameter('topics.lidar').value
        self._pub = self.create_publisher(PointCloud2, target, BEST_EFFORT)
        # raw=True: the bytes are passed on as they came, never decoded.
        callback = self._pub.publish
        note = ''
        if self.get_parameter('frames.lidar_calibration.enabled').value:
            frame = self.get_parameter('frames.lidar_calibration.frame').value
            callback = lambda raw: self._pub.publish(restamp(raw, frame))   # noqa: E731
            note = f', re-stamped {frame} (D101)'
        self.create_subscription(PointCloud2, source, callback, BEST_EFFORT, raw=True)
        self.get_logger().info(f'relaying {source} -> {target} (one network copy{note})')


def main(args=None):
    rclpy.init(args=args)
    node = LidarRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
