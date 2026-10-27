"""The floor decomposition: a robot-fixed mounting tilt plus a world-fixed slope."""

import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'tools'))
pytest.importorskip('rosbags', reason='the offline tools need rosbags (a venv)')
import floor_plane_check as F  # noqa: E402


def rows_for(mount, world, yaws_deg, noise=0.0, seed=0):
    """What the tool would measure at each heading, from a known truth."""
    rng = np.random.default_rng(seed)
    out = []
    for yaw_deg in yaws_deg:
        yaw = math.radians(yaw_deg)
        c, s = math.cos(yaw), math.sin(yaw)
        g = np.array(mount) + np.array([[c, s], [-s, c]]) @ np.array(world)
        g = g + rng.normal(0, noise, 2)
        out.append({'kind': 'lidar', 'yaw': yaw, 'height': 0.004, 'rms': 0.010,
                    'bag': f'b{len(out)}',
                    'pitch_deg': math.degrees(math.atan(g[0])),
                    'roll_deg': -math.degrees(math.atan(g[1]))})
    return out


MOUNT = (math.tan(math.radians(0.09)), -math.tan(math.radians(-0.13)))
WORLD = (math.tan(math.radians(0.20)), math.tan(math.radians(0.15)))


def test_two_headings_recover_the_truth(capsys):
    F.decompose(rows_for(MOUNT, WORLD, [0, 176]))
    out = capsys.readouterr().out
    assert 'pitch +0.090' in out and 'roll -0.130' in out


def test_four_headings_recover_it_with_a_small_residual(capsys):
    F.decompose(rows_for(MOUNT, WORLD, [0, 90, 176, -90]))
    out = capsys.readouterr().out
    assert 'pitch +0.090' in out and 'roll -0.130' in out
    assert 'consistent' in out and 'NOT consistent' not in out


def test_a_floor_that_changes_between_bags_is_flagged(capsys):
    rows = rows_for(MOUNT, WORLD, [0, 90, 176, -90])
    rows[2]['pitch_deg'] += 0.4          # one bag taken where the floor slopes differently
    F.decompose(rows)
    assert 'NOT consistent' in capsys.readouterr().out


def test_headings_too_close_together_are_refused(capsys):
    F.decompose(rows_for(MOUNT, WORLD, [0, 10]))
    assert 'too close' in capsys.readouterr().out


def test_a_fit_that_is_not_a_floor_is_dropped_not_used(capsys):
    rows = rows_for(MOUNT, WORLD, [0, 90, 176, -90])
    rows[1]['rms'] = 0.07                # a wall in the window
    rows[1]['pitch_deg'] = -10.0         # and the nonsense tilt that comes with it
    F.decompose(rows)
    out = capsys.readouterr().out
    assert 'dropping b1' in out
    assert 'pitch +0.090' in out and 'roll -0.130' in out     # unharmed by it


def test_the_widest_separation_is_measured_between_pairs_across_the_branch_cut(capsys):
    # headings 0, -176, +89, -94 degrees. The widest pair is +89 and -94: 183
    # degrees apart, i.e. 177 once wrapped. Wrapping only the largest RAW
    # difference gave 95 here (265 wrapped), which is what this guards against.
    F.decompose(rows_for(MOUNT, WORLD, [0, -176, 89, -94]))
    assert 'up to 177 deg apart' in capsys.readouterr().out
