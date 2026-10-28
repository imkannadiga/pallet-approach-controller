"""D101: the calibrated LiDAR, its re-stamped clouds, and the launch's static transform."""

import math
import pathlib
import struct

import numpy as np
import pytest
import yaml

from pallet_approach.calibration import (correct_transform, correction_matrix,
                                         lidar_follows_camera, read_lidar_calibration)
from pallet_approach.cdr import frame_id, restamp

CONFIG = pathlib.Path(__file__).resolve().parents[1] / 'config'


def _rot(axis, angle):
    k = np.array(axis, float) / np.linalg.norm(axis)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * kx + (1 - math.cos(angle)) * kx @ kx


def _tf(r, t):
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = r, t
    return out


def test_lidar_moves_rigidly_with_the_corrected_camera():
    g_c = _tf(_rot([0.3, -0.2, 1], 0.4), [0.39, 0.06, 0.373])
    c_l = _tf(_rot([1, 2, 3], 1.1), [0.074, -0.226, -0.243])
    correction = correction_matrix(0.012182, -0.009005)      # the camera's, from params.yaml
    out = lidar_follows_camera(g_c @ c_l, g_c, correction)
    # The camera-LiDAR relationship is untouched: the LiDAR is fixed to the camera.
    corrected_camera = correct_transform(g_c, correction)
    assert np.allclose(np.linalg.inv(corrected_camera) @ out, c_l, atol=1e-12)
    # And it is a real rotation about the camera: the LiDAR position moved, by about
    # 0.87 deg x 0.34 m.
    moved = np.linalg.norm(out[:3, 3] - (g_c @ c_l)[:3, 3])
    assert 0.002 < moved < 0.01


def test_identity_correction_changes_nothing():
    g_c = _tf(_rot([0, 0, 1], 0.2), [0.4, 0.0, 0.4])
    g_l = _tf(_rot([0, 1, 0], 0.1), [0.17, 0.0, 0.6])
    assert np.allclose(lidar_follows_camera(g_l, g_c, np.eye(3)), g_l)


def _cdr_header(frame: str, little=True, tail=b''):
    e = '<' if little else '>'
    name = frame.encode() + b'\x00'
    raw = bytes([0, 1 if little else 0, 0, 0]) + struct.pack(e + 'iI', 123, 456)
    raw += struct.pack(e + 'I', len(name)) + name
    raw += b'\x00' * ((4 - (len(raw) - 4) % 4) % 4)
    return raw + tail


@pytest.mark.parametrize('old,new', [('os_lidar', 'lidar_calibrated'), ('a', 'abcdefgh'),
                                     ('lidar_calibrated', 'x'), ('abc', 'abcd')])
@pytest.mark.parametrize('little', [True, False])
def test_restamp_rewrites_only_the_frame(old, new, little):
    tail = struct.pack('<II', 64, 1024) + b'payload-bytes'
    out = restamp(_cdr_header(old, little, tail), new)
    assert out == _cdr_header(new, little, tail)
    assert frame_id(out) == new
    assert out[4:12] == _cdr_header(old, little)[4:12]           # the stamp is untouched


def test_restamp_matches_rclpy_serialization():
    rclpy_serialization = pytest.importorskip('rclpy.serialization')
    msgs = pytest.importorskip('sensor_msgs.msg')
    cloud = msgs.PointCloud2()
    cloud.header.frame_id = 'os_lidar'
    cloud.header.stamp.sec, cloud.height, cloud.width, cloud.point_step = 7, 1, 2, 16
    cloud.fields = [msgs.PointField(name='x', offset=0, datatype=7, count=1)]
    cloud.data = bytes(range(32))
    out = rclpy_serialization.deserialize_message(
        restamp(rclpy_serialization.serialize_message(cloud), 'lidar_calibrated'), msgs.PointCloud2)
    assert out.header.frame_id == 'lidar_calibrated'
    assert bytes(out.data) == bytes(range(32)) and out.fields[0].name == 'x' and out.width == 2


def test_the_shipped_calibration_file_reads_and_matches_params():
    parent, child, transform = read_lidar_calibration(CONFIG / 'lidar_calibration.yaml')
    params = yaml.safe_load((CONFIG / 'params.yaml').read_text())['/**']['ros__parameters']
    assert parent == params['frames']['camera_optical']
    assert child == params['frames']['lidar_calibration']['frame']
    assert params['frames']['lidar_calibration']['enabled'] is True
    assert params['frames']['lidar_correction']['enabled'] is False   # D74 retired by D101
    r = transform[:3, :3]
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-6)
    # Roughly where the URDF puts it (a few cm), not somewhere absurd.
    assert np.linalg.norm(transform[:3, 3]) == pytest.approx(0.34, abs=0.05)


def test_the_sim_turns_the_calibration_off():
    sim = yaml.safe_load((CONFIG / 'sim_overlay.yaml').read_text())['/**']['ros__parameters']
    assert sim['frames']['lidar_calibration']['enabled'] is False
