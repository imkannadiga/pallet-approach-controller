"""Republish the camera's one dynamic tf link, the way the ZED SDK does.

On the robot, zed2_camera_center -> zed2_left_camera_frame is not static: the
SDK's stereo self-calibration republishes it at ~58 Hz, stamped now. The
watchdog times that link (it is how a clock skew between camera and Jetson
shows up), so a sim in which the whole camera chain is static would fail
`/health` for a reason the robot never has -- or need the watchdog weakened to
run at all. Neither tests the real stack. This node closes the gap instead.

It publishes on two triggers: a 58 Hz timer, like the SDK, and every camera
frame, stamped with that frame's own stamp. The second exists because this is
a Python node on sim time: its clock is only as current as the last /clock it
has processed, and under load that ran 120-230 ms behind the sensor stamps,
so perception's stamped lookups at the image time failed as "extrapolation
into the future". A transform at exactly the frame's stamp cannot be late for
that frame.

The value comes from the URDF (robot_model.camera_joint_origin), passed in as
parameters by the launch file, so there is one source for it.
"""

import math

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from tf2_ros import TransformBroadcaster


def quaternion_from_rpy(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


class CameraTf(Node):
    def __init__(self):
        super().__init__('sim_camera_tf')
        self.declare_parameter('parent', 'zed2_camera_center')
        self.declare_parameter('child', 'zed2_left_camera_frame')
        self.declare_parameter('xyz', [0.0, 0.0, 0.0])
        self.declare_parameter('rpy', [0.0, 0.0, 0.0])
        self.declare_parameter('rate', 58.0)
        self._broadcaster = TransformBroadcaster(self)
        xyz = self.get_parameter('xyz').value
        q = quaternion_from_rpy(*self.get_parameter('rpy').value)
        self._msg = TransformStamped()
        self._msg.header.frame_id = self.get_parameter('parent').value
        self._msg.child_frame_id = self.get_parameter('child').value
        t = self._msg.transform
        t.translation.x, t.translation.y, t.translation.z = xyz
        t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w = q
        self.declare_parameter('frame_topic', '/sim/zed/camera_info_gazebo')
        self.create_timer(1.0 / self.get_parameter('rate').value, self._tick)
        self.create_subscription(CameraInfo, self.get_parameter('frame_topic').value,
                                 self._on_frame, 10)
        self._last_stamp = None

    def _tick(self):
        now = self.get_clock().now()
        if now.nanoseconds == 0:
            return          # no /clock yet; a zero stamp would look 50 years stale
        self._send(now.to_msg())

    def _on_frame(self, msg: CameraInfo):
        self._send(msg.header.stamp)

    def _send(self, stamp):
        key = (stamp.sec, stamp.nanosec)
        if key == self._last_stamp:
            return          # tf2 warns on a repeated stamp
        self._last_stamp = key
        self._msg.header.stamp = stamp
        self._broadcaster.sendTransform(self._msg)


def main():
    rclpy.init()
    node = CameraTf()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
