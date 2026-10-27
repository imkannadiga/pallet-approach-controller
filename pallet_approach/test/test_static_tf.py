"""Static transforms are rigid mounting: a bag that arrives without some of them
gets them from the exported reference, and a bag that has its own keeps them."""

import pathlib
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'tools'))
pytest.importorskip('rosbags', reason='the offline tools need rosbags (a venv)')
import _bagutil as bag  # noqa: E402


def edge(parent, child, x):
    return {'parent': parent, 'child': child, 'translation': [x, 0.0, 0.0],
            'rotation': [0.0, 0.0, 0.0, 1.0]}


@pytest.fixture
def reference(tmp_path):
    path = tmp_path / 'static_tf.yaml'
    path.write_text(yaml.safe_dump({'edges': [
        edge('base_link', 'base_footprint', 1.0),
        edge('os_sensor', 'os_lidar', 2.0),
        edge('os_sensor', 'os_imu', 3.0)]}))
    return bag.static_reference(path)


def test_the_reference_is_read_as_4x4_transforms(reference):
    assert len(reference) == 3
    parent, child, matrix = reference[1]
    assert (parent, child) == ('os_sensor', 'os_lidar')
    assert matrix.shape == (4, 4) and matrix[0, 3] == pytest.approx(2.0)


def test_missing_edges_are_filled_in_and_named(reference):
    tree = bag.Tree()
    tree.add('base_link', 'base_footprint', np.eye(4))
    borrowed = bag.fill_static(tree, reference)
    assert sorted(borrowed) == ['os_imu', 'os_lidar']
    assert tree.parent['os_lidar'] == 'os_sensor'


def test_a_bag_that_has_an_edge_is_never_overridden(reference):
    tree = bag.Tree()
    own = np.eye(4)
    own[0, 3] = 9.0
    tree.add('os_sensor', 'os_lidar', own)
    bag.fill_static(tree, reference)
    assert tree.edge['os_lidar'][0, 3] == 9.0       # the bag's own value, not 2.0


def test_nothing_is_borrowed_when_the_bag_is_complete(reference):
    tree = bag.Tree()
    for parent, child, matrix in reference:
        tree.add(parent, child, matrix)
    assert bag.fill_static(tree, reference) == []


def test_an_absent_reference_file_borrows_nothing(tmp_path):
    assert bag.static_reference(tmp_path / 'nope.yaml') == []
    assert bag.fill_static(bag.Tree(), []) == []
