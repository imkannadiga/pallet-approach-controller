import xml.etree.ElementTree as ET

import pytest

from pallet_approach_sim import robot_model

# A cut-down expanded URDF with what the sim repo's macros put in: friction on a
# wheel, a Garden-style drive plugin, a LiDAR sensor and a classic IMU plugin.
URDF = """<robot name="r">
  <link name="base_link"/><link name="zed2_camera_center"/>
  <link name="zed2_left_camera_frame"/><link name="os_lidar"/>
  <joint name="zed2_left_camera_joint" type="fixed">
    <origin xyz="-0.01 0.06 0" rpy="0 0 0"/>
    <parent link="zed2_camera_center"/><child link="zed2_left_camera_frame"/>
  </joint>
  <gazebo reference="front_left_wheel_link"><mu1>0.9</mu1></gazebo>
  <gazebo><plugin filename="gz-sim-diff-drive-system" name="gz::sim::systems::DiffDrive"/></gazebo>
  <gazebo reference="os_sensor"><sensor type="gpu_lidar" name="old"/></gazebo>
  <gazebo reference="imu_link"><sensor name="imu" type="imu">
    <plugin filename="libgazebo_ros_imu_sensor.so" name="p"/></sensor></gazebo>
</robot>"""


def test_gazebo_gets_our_plugins_and_sensors_only():
    root = ET.fromstring(robot_model.for_gazebo(URDF))
    plugins = [p.get('filename') for p in root.iter('plugin')]
    assert plugins == ['ignition-gazebo-diff-drive-system',
                       'ignition-gazebo-joint-state-publisher-system']
    sensors = {s.get('name'): s for s in root.iter('sensor')}
    assert set(sensors) == {'zed_left', 'zed_depth', 'ouster'}
    # the wheel friction survives
    assert root.find("gazebo[@reference='front_left_wheel_link']/mu1").text == '0.9'


def test_the_sensors_stamp_the_real_robots_frames():
    root = ET.fromstring(robot_model.for_gazebo(URDF))
    camera = root.find(".//sensor[@name='zed_left']")
    depth = root.find(".//sensor[@name='zed_depth']")
    assert depth.find('gz_frame_id').text == 'zed2_left_camera_frame_optical'
    lidar = root.find(".//sensor[@name='ouster']")
    assert camera.find('gz_frame_id').text == 'zed2_left_camera_frame_optical'
    assert lidar.find('gz_frame_id').text == 'os_lidar'
    # mounted on the x-forward frame Gazebo cameras look along
    assert root.find(".//sensor[@name='zed_left']/..").get('reference') == 'zed2_left_camera_frame'


def test_the_camera_has_the_measured_field_of_view():
    root = ET.fromstring(robot_model.for_gazebo(URDF))
    hfov = float(root.find(".//sensor[@name='zed_left']/camera/horizontal_fov").text)
    fx = 949.6732177734375
    import math
    assert hfov == pytest.approx(2 * math.atan(640 / fx), abs=1e-3)


def test_the_lidar_is_an_os0_64_as_measured():
    root = ET.fromstring(robot_model.for_gazebo(URDF))
    scan = root.find(".//sensor[@name='ouster']/lidar/scan")
    assert scan.find('vertical/samples').text == '64'
    assert scan.find('horizontal/samples').text == '1024'


def test_state_publisher_leaves_the_sdk_joint_to_camera_tf():
    root = ET.fromstring(robot_model.for_state_publisher(URDF))
    joint = root.find("joint[@name='zed2_left_camera_joint']")
    assert joint.get('type') == 'continuous'
    assert not list(root.iter('plugin'))


def test_camera_joint_origin_is_read_from_the_urdf():
    parent, child, xyz, rpy = robot_model.camera_joint_origin(URDF)
    assert (parent, child) == ('zed2_camera_center', 'zed2_left_camera_frame')
    assert xyz == (-0.01, 0.06, 0.0) and rpy == (0.0, 0.0, 0.0)


def test_a_missing_camera_joint_is_an_error_not_a_silent_static_tree():
    with pytest.raises(ValueError):
        robot_model.for_state_publisher('<robot name="r"><link name="a"/></robot>')


def test_the_shipped_xml_files_parse():
    # XML comments may not contain a double hyphen; xacro dies on it at launch.
    import pathlib
    import xml.dom.minidom
    package = pathlib.Path(__file__).resolve().parent.parent
    for path in [*package.glob('urdf/*.xacro'), *package.glob('worlds/*.sdf')]:
        xml.dom.minidom.parse(str(path))


def test_the_camera_has_no_custom_lens_intrinsics():
    # With <lens><intrinsics> ogre2 culled the whole scene after a 0.2 rad turn.
    root = ET.fromstring(robot_model.for_gazebo(URDF))
    assert root.find(".//sensor[@name='zed_left']/camera/lens") is None


def test_colour_and_depth_share_one_view():
    root = ET.fromstring(robot_model.for_gazebo(URDF))
    cams = [root.find(f".//sensor[@name='{n}']/camera") for n in ('zed_left', 'zed_depth')]
    for tag in ('horizontal_fov', 'image/width', 'image/height'):
        assert cams[0].find(tag).text == cams[1].find(tag).text
    assert not list(root.iter('lens'))
