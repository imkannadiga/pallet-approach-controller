"""D104: the fused perception's decision tree -- the coordinator's part, the GO
rules, and the filter's healthy-track flag."""

import math

import numpy as np
import pytest

from pallet_approach import go_nogo
from pallet_approach.coordinator import Outcome, State, TrackInfo
from pallet_approach.fusion import PalletFilter
from pallet_approach.model import PalletEstimate, Pocket, Pose2D
from pallet_approach.nodes._common import tracked_topic

from pallet_scene import PARAMS
from test_coordinator import at_checking, closing_in, params, yaml_params  # noqa: F401  (fixtures)


def _track(**kw):
    base = dict(confirmed=True, confidence=1.0, side='short', range=1.75, offset=0.0,
                heading=0.0, reason='camera and LiDAR disagree (5 frames)', pallet_odom=(5.0, 0.0))
    base.update(kw)
    return TrackInfo(**base)


def test_a_disagreement_while_closing_is_a_no_go(params):
    machine = closing_in(params)
    t = machine.on_track(_track(range=2.4, disagreeing=True), 2.0)
    assert t.state is State.DONE and t.outcome is Outcome.NO_GO and t.stop_robot
    assert 'disagree' in t.reason


def test_a_disagreement_while_checking_is_a_no_go(params):
    machine = at_checking(params)
    t = machine.on_track(_track(disagreeing=True), 5.0)
    assert t.outcome is Outcome.NO_GO


def test_camera_mode_never_reports_one(params):
    machine = closing_in(params)                         # TrackInfo's default: not disagreeing
    t = machine.on_track(_track(range=2.4, reason=''), 2.0)
    assert t.state is State.CLOSING


def _estimate(lateral=0.0, yaw_deg=0.0):
    """A pallet at the standoff whose pocket centre line passes `lateral` beside the robot."""
    yaw = math.pi + math.radians(yaw_deg)
    normal = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    # On a turned pallet the centre line runs along its normal, so for it to
    # pass `lateral` beside the robot the centre sits 1.75 tan(yaw) further over.
    centre = np.array([1.75, lateral / math.cos(math.radians(yaw_deg))
                       + 1.75 * math.tan(math.radians(yaw_deg)), 0.05])
    t = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    pockets = (Pocket(centre + 0.1825 * t, 0.23, 0.10), Pocket(centre - 0.1825 * t, 0.23, 0.10))
    return PalletEstimate(position=centre, normal=normal, yaw=yaw, pockets=pockets, side='short')


GO_PARAMS = {'max_yaw_error': 0.052, 'max_lateral_offset': 0.030}


@pytest.mark.parametrize('ready,lateral,yaw,go,first', [
    (True, 0.0, 0.0, True, None),
    (True, 0.025, 2.5, True, None),                 # inside 30 mm and 3 deg
    (True, 0.035, 0.0, False, 'lateral_offset'),
    (True, 0.0, 3.5, False, 'yaw'),
    (False, 0.0, 0.0, False, 'track_ready'),       # the tree's earlier stages failed
])
def test_fused_go_is_a_ready_track_plus_alignment(ready, lateral, yaw, go, first):
    d = go_nogo.evaluate_fused(_estimate(lateral, yaw), Pose2D(0.0, 0.0, 0.0), ready,
                               'not yet 5 accepted frames', GO_PARAMS)
    assert d.go is go
    if first:
        assert d.reason.startswith(first)
    assert set(d.checks) == {'track_ready', 'yaw', 'lateral_offset'}


def test_the_d104_limits_are_in_params():
    g = PARAMS  # the shared block; go_nogo lives under the coordinator
    import pathlib, yaml
    coord = yaml.safe_load((pathlib.Path(__file__).resolve().parents[1] / 'config' / 'params.yaml')
                           .read_text())['coordinator']['ros__parameters']['go_nogo']
    assert coord['max_yaw_error'] == pytest.approx(math.radians(3), abs=0.001)
    assert coord['max_lateral_offset'] == pytest.approx(0.030)
    assert g['perception_source'] == 'camera'           # fused only after the acceptance


def test_a_healthy_track_needs_five_accepted_frames_from_both():
    f = PalletFilter(PARAMS)
    f.initialise(np.array([2.0, 0.0]), math.pi, 0.0)
    t = 0.0
    for i in range(4):
        t += 0.1
        f.update_lidar(np.array([2.0, 0.0]), math.pi, t)
        f.update_camera(np.array([2.0, 0.0]), t)
    assert not f.status(t).healthy and not f.status(t).go_allowed
    t += 0.1
    f.update_lidar(np.array([2.0, 0.0]), math.pi, t)
    f.update_camera(np.array([2.0, 0.0]), t)
    assert f.status(t).healthy and f.status(t).go_allowed
    f.note_camera_miss()                                  # the camera lost a pocket: the run breaks
    assert not f.status(t).go_allowed


def test_the_source_picks_the_topic():
    assert tracked_topic('camera') == '/pallet_tracker/tracked'
    assert tracked_topic('fused') == '/pallet_fusion/tracked'
    with pytest.raises(ValueError):
        tracked_topic('lidar')
