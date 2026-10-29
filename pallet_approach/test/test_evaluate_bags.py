"""The tape-to-truth geometry of tools/evaluate_bags.py. Every accuracy number
it prints is measured against this, so a sign error here would look like a
perception error on the robot."""

import math
import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip('rosbags')
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'tools'))
from _bagutil import BUMPER_X  # noqa: E402
from evaluate_bags import truth_for  # noqa: E402

SIDES = {'short': {'face_width': 0.795, 'pocket_spacing': 0.365, 'pocket_width': 0.23,
                   'opening_bottom': 0.0, 'opening_top': 0.10}}


def test_square_on_puts_the_pockets_either_side_of_the_centre():
    truth = truth_for({'side': 'short', 'dL': 1.29, 'dR': 1.29, 'lat': 0.10}, SIDES)
    assert truth['center'] == pytest.approx([1.29 + BUMPER_X, 0.10])
    assert abs(abs(truth['yaw']) - math.pi) < 1e-9            # facing the robot
    left, right = truth['pockets']
    assert left == pytest.approx([1.29 + BUMPER_X, 0.10 + 0.1825])
    assert right == pytest.approx([1.29 + BUMPER_X, 0.10 - 0.1825])


def test_left_corner_further_away_turns_the_left_pocket_away():
    """dataset.md: a positive angle turns the pallet's left edge away."""
    truth = truth_for({'side': 'short', 'dL': 1.40, 'dR': 1.20, 'lat': 0.0}, SIDES)
    left, right = truth['pockets']
    assert left[0] > right[0] and left[1] > right[1]
    assert np.linalg.norm(left - right) == pytest.approx(0.365)
    theta = math.asin(0.20 / 0.795)
    assert truth['yaw'] == pytest.approx(math.pi - theta)       # normal turned with it
