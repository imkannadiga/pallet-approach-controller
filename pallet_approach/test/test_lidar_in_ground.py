"""The LiDAR's pose is looked up at the latest transform, never at the sweep's stamp."""

import types

import numpy as np
import pytest

rclpy = pytest.importorskip('rclpy')
import rclpy.time  # noqa: E402

from pallet_approach.nodes._common import lidar_in_ground  # noqa: E402


class FakeBuffer:
    """Answers every lookup with an identity transform and records the times asked for."""

    def __init__(self):
        self.times = []

    def can_transform(self, target, source, stamp):
        return True

    def lookup_transform(self, target, source, stamp):
        from geometry_msgs.msg import TransformStamped
        self.times.append(stamp.nanoseconds)
        t = TransformStamped()
        t.transform.rotation.w = 1.0
        return t


def node(calibrated: bool):
    values = {'frames.ground': 'base_footprint', 'frames.camera_optical': 'cam',
              'frames.lidar_calibration.enabled': calibrated,
              'frames.extrinsic_correction.enabled': False,
              'frames.lidar_correction.enabled': False}
    return types.SimpleNamespace(
        get_parameter=lambda name: types.SimpleNamespace(value=values[name]))


@pytest.mark.parametrize('calibrated', [True, False])
def test_a_paused_camera_link_cannot_blind_the_lidar(calibrated):
    buffer = FakeBuffer()
    stamp = rclpy.time.Time(seconds=1793272477, nanoseconds=419_000_000).to_msg()
    matrix = lidar_in_ground(node(calibrated), buffer, 'lidar_calibrated', stamp, 0.0)
    assert buffer.times and all(t == 0 for t in buffer.times)     # Time() = the latest
    assert np.allclose(matrix, np.eye(4))
