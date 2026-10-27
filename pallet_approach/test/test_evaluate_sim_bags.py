"""The sim-bag evaluator's error conventions. Without these a table of errors
can look fine and mean the opposite of what its header says."""

import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'tools'))

pytest.importorskip('rosbags', reason='the offline tools need rosbags (a venv)')
import evaluate_sim_bags as E  # noqa: E402


def truth(**over):
    raw = {'name': 't', 'face_x': 2.25, 'face_y': 0.10, 'face_yaw': 0.0}
    raw.update(over)
    return E.Truth(raw)


def test_an_estimate_at_the_truth_has_no_error():
    t = truth()
    e = t.errors(t.centre, t.yaw)
    assert e['range_deck'] == pytest.approx(0.0, abs=1e-9)
    assert e['lateral_deck'] == pytest.approx(0.0, abs=1e-9)
    assert e['yaw'] == pytest.approx(0.0, abs=1e-9)


def test_farther_is_positive_range_and_left_is_positive_lateral():
    t = truth()
    farther = t.errors(t.centre + np.array([0.020, 0.0]), t.yaw)      # +x is away from the robot
    assert farther['range_deck'] == pytest.approx(20.0)
    left = t.errors(t.centre + np.array([0.0, 0.015]), t.yaw)
    assert left['lateral_deck'] == pytest.approx(15.0)


def test_a_turned_face_scores_range_along_its_own_normal():
    t = truth(face_yaw=math.radians(10))
    step = -0.020 * t.normal                  # 20 mm along the outward normal's opposite
    e = t.errors(t.centre + step, t.yaw)
    assert e['range_deck'] == pytest.approx(20.0)
    assert e['lateral_deck'] == pytest.approx(0.0, abs=1e-9)


def test_the_block_face_is_nearer_the_robot_than_the_deck_by_the_protrusion():
    t = truth(block_protrusion=0.02)
    at_blocks = t.errors(t.block_centre(), t.yaw)
    assert at_blocks['range_block'] == pytest.approx(0.0, abs=1e-9)
    assert at_blocks['range_deck'] == pytest.approx(-20.0)           # nearer than the deck


def test_yaw_error_wraps_through_the_branch_cut():
    t = truth()
    e = t.errors(t.centre, t.yaw + math.radians(5) - 2 * math.pi)
    assert e['yaw'] == pytest.approx(5.0)
