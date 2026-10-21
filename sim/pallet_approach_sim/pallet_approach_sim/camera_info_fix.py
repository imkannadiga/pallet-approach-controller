"""Republish the simulated ZED's camera_info with the intrinsics it renders with.

Gazebo Fortress's rgbd_camera publishes camera_info for a default 320x240
camera (fx 277, cx 160, cy 120) next to 1280x720 images, and ignores
<lens><intrinsics> when filling it (measured 2026-10-20, ignition-sensors 6.8).
The images themselves ARE rendered at the configured field of view: the
0.795 m pallet face at 3.11 m from the lens measures 243 px, and
949.67 * 0.795 / 3.11 = 243. So only the message is wrong, and this node
replaces its numbers, keeping Gazebo's stamp and rate so that camera_info
still times the camera the way the ZED's does (the watchdog relies on that).

Perception back-projects depth with these numbers and the overlay projects
with them; left alone, both would be off by a factor of 3.4 while looking
plausible.
"""

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo


class CameraInfoFix(Node):
    def __init__(self):
        super().__init__('sim_camera_info_fix')
        self.declare_parameter('width', 1280)
        self.declare_parameter('height', 720)
        self.declare_parameter('hfov', 1.1865)
        self.declare_parameter('input', '/sim/zed/camera_info_gazebo')
        self.declare_parameter('outputs', ['/zed2/zed_node/rgb/color/rect/camera_info',
                                           '/zed2/zed_node/depth/camera_info'])
        w = self.get_parameter('width').value
        h = self.get_parameter('height').value
        fx = (w / 2) / math.tan(self.get_parameter('hfov').value / 2)
        self._w, self._h = w, h
        self._k = [fx, 0.0, w / 2, 0.0, fx, h / 2, 0.0, 0.0, 1.0]
        self._p = [fx, 0.0, w / 2, 0.0, 0.0, fx, h / 2, 0.0, 0.0, 0.0, 1.0, 0.0]
        self._pubs = [self.create_publisher(CameraInfo, topic, 10)
                      for topic in self.get_parameter('outputs').value]
        self.create_subscription(CameraInfo, self.get_parameter('input').value,
                                 self._on_info, 10)
        self.get_logger().info(f'camera_info: {w}x{h}, fx = fy = {fx:.2f}, '
                               f'principal point at the centre')

    def _on_info(self, msg: CameraInfo):
        if (msg.width, msg.height) != (self._w, self._h):
            self.get_logger().error(
                f'Gazebo says {msg.width}x{msg.height}, this node was told '
                f'{self._w}x{self._h}; the images and these intrinsics disagree',
                throttle_duration_sec=10.0)
        msg.k = self._k
        msg.p = self._p
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.d = [0.0] * 5
        msg.distortion_model = 'plumb_bob'
        for pub in self._pubs:
            pub.publish(msg)


def main():
    rclpy.init()
    node = CameraInfoFix()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
