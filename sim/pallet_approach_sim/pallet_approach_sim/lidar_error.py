"""Give the sim's LiDAR the real Ouster's range bias, as measured (D109).

The range study of 2026-11-02 (docs/studies/range_error_20261102.yaml; the same
numbers as params.yaml's range_error.lidar, which a test checks) fitted the
LiDAR's error against a tape as bias = a + b m, m = range - sensor_minus_mark,
held at its end values outside the measured range. Its per-position scatter
(4.2 mm) is the tape's (its frame noise is sub-millimetre), so no static error
is added; Gazebo's own 8 mm per-point noise stays, which averages over a face's
~500 returns to the study's 0.2-0.6 mm per frame.

`LidarError.apply` is pure NumPy; the node subscribes to Gazebo's cloud
(/sim/lidar/points_raw), moves each return along its ray, and publishes the
result as /ouster/points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LidarError:
    bias_a: float = -0.000762          # m
    bias_b: float = 0.001038           # m per m
    sensor_minus_mark: float = 0.3141  # m
    valid_range: tuple = (0.993, 3.892)
    tape_sigma: float = 0.0041633      # the study's sigma_position: the tape, not the LiDAR

    def bias(self, r: np.ndarray) -> np.ndarray:
        m = np.clip(r - self.sensor_minus_mark, *self.valid_range)
        return self.bias_a + self.bias_b * m

    def apply(self, xyz: np.ndarray) -> np.ndarray:
        """(N, 3) points in the sensor's frame -> the same rays, biased in range."""
        r = np.linalg.norm(xyz, axis=1)
        out = xyz.copy()
        ok = np.isfinite(r) & (r > 1e-6)
        out[ok] *= ((r[ok] + self.bias(r[ok])) / r[ok])[:, None]
        return out

    def apply_to_buffer(self, data: bytes, point_step: int, offsets: dict) -> bytes:
        """The same, on a PointCloud2's raw bytes with float32 x, y, z at `offsets`."""
        raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, point_step).copy()
        xyz = np.stack([raw[:, offsets[k]:offsets[k] + 4].copy().view(np.float32).reshape(-1)
                        for k in ('x', 'y', 'z')], axis=1).astype(float)
        out = self.apply(xyz).astype(np.float32)
        for i, k in enumerate(('x', 'y', 'z')):
            raw[:, offsets[k]:offsets[k] + 4] = np.ascontiguousarray(out[:, i]).view(np.uint8).reshape(-1, 4)
        return raw.tobytes()


def main(args=None):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import PointCloud2

    class LidarErrorNode(Node):
        def __init__(self):
            super().__init__('sim_lidar_error')
            self.declare_parameter('input', '/sim/lidar/points_raw')
            self.declare_parameter('output', '/ouster/points')
            self._model = LidarError()
            self._pub = self.create_publisher(PointCloud2, self.get_parameter('output').value,
                                              qos_profile_sensor_data)
            self.create_subscription(PointCloud2, self.get_parameter('input').value, self._on_cloud,
                                     qos_profile_sensor_data)
            self.get_logger().info('LiDAR range bias from the range study (D109) on /ouster/points')

        def _on_cloud(self, msg: PointCloud2):
            offsets = {f.name: f.offset for f in msg.fields}
            msg.data = self._model.apply_to_buffer(bytes(msg.data), msg.point_step, offsets)
            self._pub.publish(msg)

    rclpy.init(args=args)
    node = LidarErrorNode()
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
