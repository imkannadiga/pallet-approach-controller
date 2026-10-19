"""Tests for pointcloud.py, against a fake PointCloud2.

Faked rather than mocked: a little class with the fields a real message has is
enough, and it keeps these tests runnable without ROS like everything else
here. The layout it builds is the Ouster's -- nine fields, xyz first, 48 bytes
a point -- because the bug this code can have is reading the wrong offsets out
of a cloud whose layout is not the one the author imagined.
"""

import numpy as np
import pytest

from pallet_approach.pointcloud import DATATYPES, transform_points, unpack_cloud


class Field:
    def __init__(self, name, offset, datatype, count=1):
        self.name, self.offset, self.datatype, self.count = name, offset, datatype, count


class Cloud:
    """The Ouster's layout: x, y, z, intensity (f32), t (u32), reflectivity,
    ring, ambient (u16), range (u32), padded to 48 bytes."""

    def __init__(self, points, extra=True, bigendian=False):
        points = np.asarray(points, dtype=np.float32)
        self.point_step = 48 if extra else 12
        self.is_bigendian = bigendian
        self.height, self.width = 1, len(points)
        self.fields = [Field('x', 0, 7), Field('y', 4, 7), Field('z', 8, 7)]
        if extra:
            self.fields += [Field('intensity', 12, 7), Field('t', 16, 6),
                            Field('reflectivity', 20, 4), Field('ring', 22, 4),
                            Field('ambient', 24, 4), Field('range', 28, 6)]
        buffer = np.zeros((len(points), self.point_step), dtype=np.uint8)
        buffer[:, 0:12] = points.view(np.uint8).reshape(-1, 12)
        if extra:                       # plausible junk in the other fields
            buffer[:, 12:16] = np.frombuffer(
                np.full(len(points), 1234.5, dtype=np.float32).tobytes(),
                dtype=np.uint8).reshape(-1, 4)
        self.data = buffer.tobytes()


def test_reads_xyz_out_of_a_wide_layout():
    """The point of the offset table: nine fields, and we want three of them."""
    original = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    points = unpack_cloud(Cloud(original))
    assert points.shape == (2, 3)
    assert np.allclose(points, original)


def test_a_bare_xyz_layout_reads_the_same():
    points = unpack_cloud(Cloud([[1.0, 2.0, 3.0]], extra=False))
    assert np.allclose(points, [[1.0, 2.0, 3.0]])


def test_no_return_points_are_dropped():
    """The Ouster marks a beam that hit nothing as exactly (0, 0, 0), not NaN.
    Forty thousand of those would read downstream as a solid obstacle against
    the robot's own axle."""
    points = unpack_cloud(Cloud([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                                 [2.0, 0.0, 0.0]]))
    assert len(points) == 2
    assert np.allclose(points[:, 0], [1.0, 2.0])


def test_non_finite_points_are_dropped():
    points = unpack_cloud(Cloud([[1.0, 0.0, 0.0], [np.nan, 0.0, 0.0],
                                 [np.inf, 1.0, 0.0]]))
    assert len(points) == 1


def test_keeping_everything_is_possible_when_the_caller_wants_it():
    points = unpack_cloud(Cloud([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
                          finite_only=False)
    assert len(points) == 2


def test_extra_fields_can_be_requested_by_name():
    cloud = Cloud([[1.0, 2.0, 3.0]])
    values = unpack_cloud(cloud, fields=('x', 'intensity'), finite_only=False)
    assert values[0, 0] == pytest.approx(1.0)
    assert values[0, 1] == pytest.approx(1234.5)


def test_a_missing_field_raises_rather_than_returning_zeros():
    with pytest.raises(ValueError, match='no field'):
        unpack_cloud(Cloud([[1.0, 2.0, 3.0]]), fields=('x', 'y', 'nope'))


def test_big_endian_is_refused_not_guessed():
    """Byte-swapping the wrong way gives coordinates around 1e30, which look
    like a sensor fault rather than a decoding bug."""
    with pytest.raises(ValueError, match='big-endian'):
        unpack_cloud(Cloud([[1.0, 2.0, 3.0]], bigendian=True))


def test_an_empty_cloud_is_an_empty_array():
    points = unpack_cloud(Cloud([]))
    assert points.shape[0] == 0


def test_the_datatype_table_covers_the_point_field_enum():
    assert sorted(DATATYPES) == list(range(1, 9))


# ----------------------------------------------------------- transform

def test_a_transform_rotates_and_translates():
    matrix = np.array([[0.0, -1.0, 0.0, 1.0],
                       [1.0, 0.0, 0.0, 2.0],
                       [0.0, 0.0, 1.0, 3.0],
                       [0.0, 0.0, 0.0, 1.0]])
    moved = transform_points(np.array([[1.0, 0.0, 0.0]]), matrix)
    assert np.allclose(moved, [[1.0, 3.0, 3.0]])


def test_an_identity_transform_changes_nothing():
    points = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 0.0]])
    assert np.allclose(transform_points(points, np.eye(4)), points)


def test_transforming_an_empty_cloud_keeps_its_shape():
    """A frame with no returns must not crash the node that is about to index
    its columns."""
    assert transform_points(np.zeros((0, 3)), np.eye(4)).shape == (0, 3)


def test_the_wrong_shape_raises():
    with pytest.raises(ValueError, match=r'\(N, 3\)'):
        transform_points(np.zeros((4, 4)), np.eye(4))
