"""The shared starting positions (D96): one seed, the same poses in sim and on
the robot. The tests pin the numbers the sim batches already used, the sign of
the tape readings, and the bounds the batch promises."""

import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'tools'))

import start_positions as sp  # noqa: E402


def test_seed_7_reproduces_the_poses_the_sim_batches_used():
    """Before D96 `batch_runs.sh --random 7` drew these with an inline snippet.
    If these move, the sim results from before the change no longer describe
    the same starts."""
    first = sp.start_pose(7, 1)
    second = sp.start_pose(7, 2)
    assert first.face_x == pytest.approx(2.488, abs=5e-4)
    assert first.face_y == pytest.approx(0.271, abs=5e-4)
    assert first.face_yaw == pytest.approx(0.0626, abs=5e-5)
    assert second.face_x == pytest.approx(2.500, abs=5e-4)
    assert second.face_y == pytest.approx(-0.034, abs=5e-4)


def test_the_same_seed_gives_the_same_poses():
    assert sp.start_poses(7, 10) == sp.start_poses(7, 10)


def test_a_different_seed_gives_different_poses():
    assert sp.start_poses(7, 10) != sp.start_poses(8, 10)


@pytest.mark.parametrize('seed', [1, 7, 42])
def test_offset_and_heading_stay_inside_the_promised_bounds(seed):
    for pose in sp.start_poses(seed, 200):
        assert abs(pose.offset) <= sp.MAX_OFFSET
        assert abs(pose.heading_deg) <= sp.MAX_HEADING_DEG


def test_a_positive_heading_turns_the_left_edge_away():
    """The dataset's sign: positive angle, left edge away, dL > dR. At 10 deg
    the corners differ by 0.795 * sin(10 deg) = 13.8 cm (dataset.md, tier 3)."""
    pose = sp.StartPose(run=1, offset=0.0, heading_deg=10.0,
                        face_x=2.5, face_y=0.0, face_yaw=-math.radians(10.0))
    assert pose.d_left > pose.d_right
    assert pose.d_left - pose.d_right == pytest.approx(0.138, abs=1e-3)


def test_a_negative_heading_turns_the_right_edge_away():
    pose = sp.start_pose(7, 1)
    assert pose.heading_deg < 0
    assert pose.d_left < pose.d_right


def test_a_square_pallet_has_equal_corners():
    pose = sp.StartPose(run=1, offset=0.0, heading_deg=0.0,
                        face_x=2.5, face_y=0.1, face_yaw=0.0)
    assert pose.d_left == pytest.approx(pose.d_right)
    assert pose.d_left == pytest.approx(2.5 - sp.BUMPER_X)


def test_lat_is_the_face_centre_offset_positive_to_the_left():
    pose = sp.start_pose(7, 1)
    assert pose.lat == pose.face_y


def test_the_table_has_one_row_per_run_and_the_header_matches():
    table = sp.rows(sp.start_poses(7, 10))
    assert len(table) == 10
    assert all(len(row) == len(sp.HEADER) for row in table)
