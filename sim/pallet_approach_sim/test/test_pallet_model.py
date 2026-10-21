"""The simulated pallet must be the pallet the unit tests are tuned on.

test/synthetic.py in the stack is the reference: the measured short face and
the modelled long face. These slice the Gazebo boxes at a height and compare
what a camera would see on each face with synthetic.py's spans.
"""

import importlib.util
import math
import pathlib
import sys
import xml.etree.ElementTree as ET

import pytest

from pallet_approach_sim import pallet_model

SYNTHETIC = (pathlib.Path(__file__).resolve().parents[3]
             / 'pallet_approach' / 'test' / 'synthetic.py')


@pytest.fixture(scope='module')
def synthetic():
    if not SYNTHETIC.exists():
        pytest.skip(f'{SYNTHETIC} not here (sim package built on its own)')
    spec = importlib.util.spec_from_file_location('synthetic', SYNTHETIC)
    module = importlib.util.module_from_spec(spec)
    sys.modules['synthetic'] = module       # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


def approx_spans(a, b):
    assert len(a) == len(b), (a, b)
    for (a0, a1), (b0, b1) in zip(a, b):
        assert a0 == pytest.approx(b0, abs=1e-6) and a1 == pytest.approx(b1, abs=1e-6), (a, b)


@pytest.mark.parametrize('face', ['short', 'long'])
def test_the_opening_height_has_the_synthetic_blocks(synthetic, face):
    ref = synthetic.FACES[face]
    mid = ref.pocket_center_height
    approx_spans(pallet_model.face_profile(face, mid), ref.blocks)


@pytest.mark.parametrize('face', ['short', 'long'])
def test_the_opening_runs_exactly_where_synthetic_says(synthetic, face):
    ref = synthetic.FACES[face]
    full = [(-ref.width / 2, ref.width / 2)]
    eps = 1e-3
    # just inside the opening: blocks only
    approx_spans(pallet_model.face_profile(face, ref.opening_bottom + eps), ref.blocks)
    approx_spans(pallet_model.face_profile(face, ref.opening_top - eps), ref.blocks)
    # just above it: solid across
    approx_spans(pallet_model.face_profile(face, ref.opening_top + eps), full)
    # below it: a board across the long face, nothing extra on the short one
    if ref.bottom_board > 0:
        approx_spans(pallet_model.face_profile(face, ref.bottom_board / 2), full)


def test_footprint_and_height(synthetic):
    assert pallet_model.WIDTH == pytest.approx(synthetic.SHORT.width)
    assert pallet_model.LENGTH == pytest.approx(synthetic.LONG.width)
    assert pallet_model.HEIGHT == pytest.approx(synthetic.PALLET_HEIGHT)
    tops = max(box.z[1] for box in pallet_model.boxes())
    assert tops == pytest.approx(synthetic.PALLET_HEIGHT)


def test_the_short_face_has_nothing_under_the_pockets():
    # D4: the pockets open from the floor.
    pockets_mid = [(-0.2975, -0.0675), (0.0675, 0.2975)]
    solid = pallet_model.face_profile('short', 0.005)
    for a, b in pockets_mid:
        centre = 0.5 * (a + b)
        assert not any(s0 < centre < s1 for s0, s1 in solid)


def test_the_sdf_parses_and_has_every_box():
    root = ET.fromstring(pallet_model.sdf())
    assert root.find('model/static').text == 'true'
    assert len(root.findall('.//visual')) == len(pallet_model.boxes()) == 1 + 3 + 3 + 9


@pytest.mark.parametrize('face', ['short', 'long'])
@pytest.mark.parametrize('yaw', [0.0, 0.3, -0.4])
def test_placement_puts_the_face_where_asked(face, yaw):
    fx, fy = 3.2, -0.4
    cx, cy, model_yaw = pallet_model.placement(face, fx, fy, yaw)
    # Face centre in the model frame, rotated into the world.
    local = (pallet_model.LENGTH / 2, 0.0) if face == 'short' else (0.0, pallet_model.WIDTH / 2)
    c, s = math.cos(model_yaw), math.sin(model_yaw)
    wx = cx + c * local[0] - s * local[1]
    wy = cy + s * local[0] + c * local[1]
    assert (wx, wy) == pytest.approx((fx, fy), abs=1e-9)
    # Outward normal points back at the robot when yaw is zero.
    normal_local = (1.0, 0.0) if face == 'short' else (0.0, 1.0)
    nx = c * normal_local[0] - s * normal_local[1]
    ny = s * normal_local[0] + c * normal_local[1]
    assert math.atan2(ny, nx) == pytest.approx(
        math.atan2(math.sin(math.pi + yaw), math.cos(math.pi + yaw)), abs=1e-9)


def test_placement_refuses_an_unknown_face():
    with pytest.raises(ValueError):
        pallet_model.placement('top', 3.0, 0.0, 0.0)
