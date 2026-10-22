"""The forklift URDF: the SCOUT's frames and sensor poses, D11's forks, and
what Gazebo needs to drive and lift it."""

import json
import math
import pathlib
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from pallet_approach_sim import forklift_model, pallet_model, robot_model

PACKAGE = pathlib.Path(__file__).resolve().parent.parent
GEOMETRY = json.loads((PACKAGE / 'config' / 'forklift_geometry.json').read_text())
URDF = forklift_model.forklift_urdf(GEOMETRY, 'file:///meshes')


def chain_to(root, link):
    """xyz of `link` from base_footprint, through fixed translations."""
    joints = {j.find('child').get('link'): j for j in root.findall('joint')}
    xyz = np.zeros(3)
    while link != 'base_link':
        j = joints[link]
        xyz += [float(v) for v in j.find('origin').get('xyz').split()]
        link = j.find('parent').get('link')
    return xyz + [0.0, 0.0, GEOMETRY['base_link_z']]


@pytest.mark.parametrize('link, expected', [
    # Read off the SCOUT's URDF chain (scout_sim.urdf.xacro), from base_footprint.
    ('zed2_left_camera_frame', (0.39, 0.06, 0.373)),
    ('zed2_camera_center', (0.40, 0.0, 0.373)),
    ('os_lidar', (0.173, 0.0, 0.6283)),
])
def test_the_sensors_sit_exactly_where_the_scouts_do(link, expected):
    root = ET.fromstring(URDF)
    assert np.allclose(chain_to(root, link), expected, atol=1e-4)


def test_the_frames_the_stack_uses_exist_with_the_scouts_names():
    root = ET.fromstring(URDF)
    links = {l.get('name') for l in root.findall('link')}
    assert {'base_link', 'base_footprint', 'zed2_left_camera_frame_optical',
            'os_lidar'} <= links
    footprint = [j for j in root.findall('joint') if j.find('child').get('link') == 'base_footprint'][0]
    assert footprint.find('origin').get('xyz').split()[2].startswith('-0.21')
    # the dynamic camera joint camera_tf.py republishes
    assert robot_model.camera_joint_origin(URDF)[:2] == ('zed2_camera_center',
                                                        'zed2_left_camera_frame')


def test_the_forks_are_365_mm_apart_and_start_at_the_heel():
    tine = GEOMETRY['tine']
    assert tine['centre_y'] * 2 == pytest.approx(0.365)          # D11
    assert tine['heel_x'] == pytest.approx(0.4625)               # forks.carriage_x
    root = ET.fromstring(URDF)
    blades = [c for c in root.find("link[@name='carriage']").findall('collision')
              if c.find('geometry/box') is not None
              and float(c.find('geometry/box').get('size').split()[0]) > 1.0]
    ys = sorted(float(b.find('origin').get('xyz').split()[1]) for b in blades)
    assert ys == pytest.approx([-0.1825, 0.1825])


def test_the_wheels_hold_the_floor_where_the_scout_has_it():
    # Floor at -0.025 in base_footprint: front wheel bottoms touch it.
    for side in ('front_left', 'front_right'):
        w = GEOMETRY[side]
        assert w['xyz'][2] - w['radius'] == pytest.approx(GEOMETRY['floor_z'], abs=0.002)


def test_gazebo_gets_the_front_wheel_drive_and_the_lift():
    drive = robot_model.Drive(wheel_separation=forklift_model.wheel_separation(GEOMETRY),
                              wheel_radius=0.2847, left_joints=('front_left_wheel',),
                              right_joints=('front_right_wheel',))
    root = ET.fromstring(robot_model.for_gazebo(
        URDF, drive=drive, lift=robot_model.Lift(forklift_model.LIFT_JOINT)))
    diff = root.find(".//plugin[@name='ignition::gazebo::systems::DiffDrive']")
    assert [e.text for e in diff.findall('left_joint')] == ['front_left_wheel']
    lift = root.find(".//plugin[@name='ignition::gazebo::systems::JointPositionController']")
    assert lift.find('joint_name').text == 'fork_lift'
    states = root.find(".//plugin[@name='ignition::gazebo::systems::JointStatePublisher']")
    assert 'fork_lift' in [e.text for e in states.findall('joint_name')]


def test_the_robot_is_hidden_from_its_own_sensors_only():
    sdf = ('<sdf><model name="r"><link name="a"><visual name="v"/>'
           '<sensor name="c" type="camera"><camera/></sensor>'
           '<sensor name="l" type="gpu_lidar"><lidar/></sensor></link></model></sdf>')
    root = ET.fromstring(robot_model.hide_robot_from_sensors(sdf))
    flags = int(root.find('.//visual/visibility_flags').text)
    for path in ('.//camera/visibility_mask', './/lidar/visibility_mask'):
        assert int(root.find(path).text) & flags == 0          # sensors skip the robot
    assert 0xFFFFFFFF & flags                                   # the GUI still shows it


def test_the_forklift_pallet_is_dynamic_and_the_scout_one_static():
    assert ET.fromstring(pallet_model.sdf()).find('model/static').text == 'true'
    dynamic = ET.fromstring(pallet_model.sdf(static=False))
    assert dynamic.find('model/static').text == 'false'
    assert float(dynamic.find('.//inertial/mass').text) > 0


def test_no_link_shares_a_name_with_a_joint():
    # Legal in URDF; SDF refuses the whole model ("name collision").
    root = ET.fromstring(URDF)
    links = {l.get('name') for l in root.findall('link')}
    joints = {j.get('name') for j in root.findall('joint')}
    assert not links & joints


def test_the_meshes_carry_no_texture_coordinates():
    # Fortress's Ogre 2.2 aborts importing an OBJ with UVs (2026-10-22).
    for path in (PACKAGE / 'meshes' / 'forklift').glob('*.obj'):
        text = path.read_text()
        assert '\nvt ' not in text, path.name
        assert 'mtllib forklift.mtl' in text, path.name


def test_the_depth_filter_removes_points_inside_a_blade_and_nothing_else():
    from pallet_approach_sim.depth_mask import mask_boxes, optical_to_base
    camera = optical_to_base((0.39, 0.06, 0.373))
    fx = fy = 949.17
    cx, cy = 640.0, 360.0
    depth = np.full((720, 1280), 2.0, np.float32)    # a wall 2 m ahead
    # One pixel looking at a blade point: base (1.2, 0.1825, 0.12)
    p = np.linalg.inv(camera) @ np.array([1.2, 0.1825, 0.12, 1.0])
    u, v = int(round(fx * p[0] / p[2] + cx)), int(round(fy * p[1] / p[2] + cy))
    depth[v, u] = p[2]
    blade = (0.4625, 1.657, 0.119, 0.246, 0.1068, 0.1412)
    out = mask_boxes(depth, fx, fy, cx, cy, camera, [blade])
    assert np.isinf(out[v, u])
    assert np.isfinite(np.delete(out.ravel(), v * 1280 + u)).all()
    assert np.array_equal(mask_boxes(depth, fx, fy, cx, cy, camera, []), depth)

