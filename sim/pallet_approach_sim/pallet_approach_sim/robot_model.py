"""Turn the expanded robot URDF into what Gazebo and robot_state_publisher need.

Pure string-in, string-out (no ROS, no xacro), so it is tested on its own.

Two outputs from one URDF:

`for_gazebo`   strips every Gazebo plugin and sensor the sim repo's macros
               carry and adds ours. Theirs are written for Gazebo Garden+
               (`gz-sim-...` filenames, a 32-beam LiDAR with a 0.9 m minimum
               range bridged as a LaserScan, a Gazebo-classic IMU plugin);
               the image runs Fortress, and the LiDAR is not ours.
`for_state_publisher`
               the same tree with the one camera joint the ZED SDK publishes
               dynamically turned into a moving joint, so robot_state_publisher
               leaves it to camera_tf.py exactly as it leaves it to the SDK on
               the robot. (A `floating` joint would not do: kdl_parser turns it
               back into a fixed one, and then two publishers own one frame.)
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

DYNAMIC_CAMERA_JOINT = 'zed2_left_camera_joint'
CAMERA_LINK = 'zed2_left_camera_frame'
CAMERA_OPTICAL = 'zed2_left_camera_frame_optical'
LIDAR_LINK = 'os_lidar'

WHEELS_LEFT = ('front_left_wheel', 'rear_left_wheel')
WHEELS_RIGHT = ('front_right_wheel', 'rear_right_wheel')


@dataclass(frozen=True)
class Camera:
    """The ZED's left camera as an RGB-D sensor.

    Defaults: the measured intrinsics. fx = 949.67 px at 1280 wide is a
    horizontal field of view of 2 atan(640 / 949.67) = 1.1865 rad. Gazebo puts
    the principal point at the image centre (the ZED's is ~12 px left and 10 px
    down of it); everything downstream reads camera_info, so that only moves
    pixels, not geometry.
    """
    width: int = 1280
    height: int = 720
    hfov: float = 1.1865
    rate: float = 15.0
    near: float = 0.3                   # ZED 2i minimum depth, ~0.3 m
    far: float = 20.0
    depth_noise: float = 0.01           # m, gaussian, constant with range


@dataclass(frozen=True)
class Lidar:
    """An OS0-64 laid out as measured on the robot (docs/SYSTEM.md).

    64 x 1024, -44.8 .. +43.9 deg vertical, 10 Hz. The sensor sits on the
    URDF's os_lidar, which is 41 mm higher than the real one was measured to
    be -- the sim inherits the URDF, so in here the URDF is right by
    definition, and the tracker uses the same URDF.
    """
    rows: int = 64
    columns: int = 1024
    vfov_min: float = -0.7819
    vfov_max: float = 0.7662
    rate: float = 10.0
    min_range: float = 0.3
    max_range: float = 75.0
    noise: float = 0.008


@dataclass(frozen=True)
class Drive:
    """Skid steer approximated as a differential drive over all four wheels.

    The separation is the URDF's track, 0.58306. A skid-steer base does not
    turn about that geometry -- the tyres scrub -- so the odometry yaw drifts
    from the truth while turning, as it does on the robot. No velocity or
    acceleration limits here: the controller applies its own, and a limit in
    the plant would hide a controller that did not.
    """
    wheel_separation: float = 0.58306
    wheel_radius: float = 0.16459
    odom_rate: float = 50.0
    left_joints: tuple = WHEELS_LEFT
    right_joints: tuple = WHEELS_RIGHT
    joint_states: tuple = WHEELS_LEFT + WHEELS_RIGHT


@dataclass(frozen=True)
class Lift:
    """A position-controlled prismatic joint (the forklift's carriage).

    Commanded on a Gazebo topic with the joint position in metres; sim's
    fork_lift_bridge.py turns the virtual_forks height into that. Velocity
    commands rather than force: the joint follows the command stiffly
    whatever the load, which is what a hydraulic lift does.
    """
    joint: str
    topic: str = '/model/robot/fork_lift'


def for_gazebo(urdf: str, camera: Camera = Camera(), lidar: Lidar = Lidar(),
               drive: Drive = Drive(), lift: Lift | None = None) -> str:
    root = ET.fromstring(urdf)
    _strip_gazebo_plugins(root)
    blocks = [_drive_xml(drive), _joint_states_xml(drive, lift), _camera_xml(camera),
              _lidar_xml(lidar)]
    if lift is not None:
        blocks.append(_lift_xml(lift))
    for block in blocks:
        root.append(ET.fromstring(block))
    return ET.tostring(root, encoding='unicode')


ROBOT_VISIBILITY = 0x2
"""Visibility flag on every visual of the robot's own model."""
SENSOR_MASK = 0xFFFFFFFF & ~ROBOT_VISIBILITY
"""The camera's and LiDAR's mask: everything except the robot itself."""


def hide_robot_from_sensors(sdf: str) -> str:
    """Make the robot's own visuals invisible to its own camera and LiDAR.

    For the forklift (Athreya, 2026-10-22): its sensors sit exactly where the
    SCOUT's do, which is inside the mast and behind the carriage plate, so
    without this the camera would see the inside of its own carriage. With it,
    both sensors see what the SCOUT's would; the Gazebo window still shows the
    whole truck (its camera's mask includes this flag). Works on the SDF that
    `ign sdf -p` makes of the URDF, because URDF cannot carry either element.
    """
    root = ET.fromstring(sdf)
    for visual in root.iter('visual'):
        flags = visual.find('visibility_flags')
        if flags is None:
            flags = ET.SubElement(visual, 'visibility_flags')
        flags.text = str(ROBOT_VISIBILITY)
    for sensor in root.iter('sensor'):
        body = sensor.find('camera') if sensor.find('camera') is not None else sensor.find('lidar')
        if body is None:
            continue
        mask = body.find('visibility_mask')
        if mask is None:
            mask = ET.SubElement(body, 'visibility_mask')
        mask.text = str(SENSOR_MASK)
    return ET.tostring(root, encoding='unicode')


def for_state_publisher(urdf: str) -> str:
    root = ET.fromstring(urdf)
    _strip_gazebo_plugins(root)
    for joint in root.findall('joint'):
        if joint.get('name') == DYNAMIC_CAMERA_JOINT:
            if joint.get('type') != 'fixed':
                raise ValueError(f'{DYNAMIC_CAMERA_JOINT} is already {joint.get("type")}')
            joint.set('type', 'continuous')
            ET.SubElement(joint, 'axis', {'xyz': '0 0 1'})
            break
    else:
        raise ValueError(f'no joint {DYNAMIC_CAMERA_JOINT!r} in the URDF')
    return ET.tostring(root, encoding='unicode')


def camera_joint_origin(urdf: str):
    """(xyz, rpy) of the dynamic camera joint, for camera_tf.py to republish."""
    root = ET.fromstring(urdf)
    for joint in root.findall('joint'):
        if joint.get('name') == DYNAMIC_CAMERA_JOINT:
            origin = joint.find('origin')
            xyz = tuple(float(v) for v in origin.get('xyz', '0 0 0').split())
            rpy = tuple(float(v) for v in origin.get('rpy', '0 0 0').split())
            return (joint.find('parent').get('link'), joint.find('child').get('link'),
                    xyz, rpy)
    raise ValueError(f'no joint {DYNAMIC_CAMERA_JOINT!r} in the URDF')


def _strip_gazebo_plugins(root):
    """Drop every <gazebo> block that carries a plugin or a sensor.

    Friction and contact settings on the wheels stay: those are the sim repo's
    tuning of the base, and nothing here has a better number.
    """
    for block in list(root.findall('gazebo')):
        if block.find('.//plugin') is not None or block.find('.//sensor') is not None:
            root.remove(block)


def _drive_xml(d: Drive) -> str:
    joints = ''.join(f'<left_joint>{j}</left_joint>' for j in d.left_joints)
    joints += ''.join(f'<right_joint>{j}</right_joint>' for j in d.right_joints)
    return (
        '<gazebo><plugin filename="ignition-gazebo-diff-drive-system" '
        'name="ignition::gazebo::systems::DiffDrive">'
        f'{joints}'
        f'<wheel_separation>{d.wheel_separation}</wheel_separation>'
        f'<wheel_radius>{d.wheel_radius}</wheel_radius>'
        f'<odom_publish_frequency>{d.odom_rate}</odom_publish_frequency>'
        '<topic>/model/robot/cmd_vel</topic>'
        '<odom_topic>/model/robot/odometry</odom_topic>'
        '<tf_topic>/model/robot/tf</tf_topic>'
        '<frame_id>odom</frame_id>'
        '<child_frame_id>base_link</child_frame_id>'
        '</plugin></gazebo>')


def _joint_states_xml(d: Drive, lift: Lift | None = None) -> str:
    names = d.joint_states + ((lift.joint,) if lift is not None else ())
    joints = ''.join(f'<joint_name>{j}</joint_name>' for j in names)
    return ('<gazebo><plugin filename="ignition-gazebo-joint-state-publisher-system" '
            'name="ignition::gazebo::systems::JointStatePublisher">'
            f'<topic>/model/robot/joint_states</topic>{joints}</plugin></gazebo>')


def _lift_xml(lift: Lift) -> str:
    return ('<gazebo><plugin filename="ignition-gazebo-joint-position-controller-system" '
            'name="ignition::gazebo::systems::JointPositionController">'
            f'<joint_name>{lift.joint}</joint_name><topic>{lift.topic}</topic>'
            '<use_velocity_commands>true</use_velocity_commands>'
            '<p_gain>8.0</p_gain><cmd_max>0.8</cmd_max><cmd_min>-0.8</cmd_min>'
            '</plugin></gazebo>')


def camera_intrinsics(c: Camera):
    """(fx, fy, cx, cy) of an ideal pinhole with this field of view."""
    fx = (c.width / 2) / math.tan(c.hfov / 2)
    return fx, fx, c.width / 2, c.height / 2


def _camera_xml(c: Camera) -> str:
    # Attached to the left camera frame (x forward), which is what a Gazebo
    # camera looks along; the headers carry the OPTICAL frame, which is the
    # convention the images are in. Same as the ZED.
    #
    # TWO sensors, a camera and a depth_camera, at the same pose with the same
    # field of view -- not one rgbd_camera. With rgbd_camera, both its images
    # went blank (colour all sky, depth all inf) whenever the robot had turned
    # more than ~0.15 rad, and came back when it turned back; the LiDAR on the
    # same body was unaffected. Measured 2026-10-20 on an RTX 3080.
    #
    # Fortress publishes camera_info with a default 320x240 camera's
    # intrinsics next to 1280x720 images, so camera_info_fix.py corrects it on
    # the ROS side. Do not add <lens><intrinsics> here: Fortress ignores it for
    # camera_info.
    common = (
        '<always_on>1</always_on><visualize>false</visualize>'
        f'<update_rate>{c.rate}</update_rate>'
        f'<gz_frame_id>{CAMERA_OPTICAL}</gz_frame_id>')
    image = (f'<horizontal_fov>{c.hfov}</horizontal_fov>'
             f'<image><width>{c.width}</width><height>{c.height}</height>')
    return (
        f'<gazebo reference="{CAMERA_LINK}">'
        f'<sensor name="zed_left" type="camera">{common}'
        '<topic>/sim/zed_rgb/image</topic>'
        f'<camera>{image}<format>R8G8B8</format></image>'
        f'<clip><near>0.05</near><far>{c.far}</far></clip></camera></sensor>'
        f'<sensor name="zed_depth" type="depth_camera">{common}'
        '<topic>/sim/zed/depth_image</topic>'
        f'<camera>{image}<format>R_FLOAT32</format></image>'
        f'<clip><near>{c.near}</near><far>{c.far}</far></clip>'
        f'<noise><type>gaussian</type><mean>0</mean><stddev>{c.depth_noise}</stddev></noise>'
        '</camera></sensor></gazebo>')


def _lidar_xml(l: Lidar) -> str:
    return (
        f'<gazebo reference="{LIDAR_LINK}"><sensor name="ouster" type="gpu_lidar">'
        '<always_on>1</always_on><visualize>false</visualize>'
        f'<update_rate>{l.rate}</update_rate>'
        '<topic>/sim/lidar</topic>'
        f'<gz_frame_id>{LIDAR_LINK}</gz_frame_id>'
        '<lidar><scan>'
        f'<horizontal><samples>{l.columns}</samples><resolution>1</resolution>'
        '<min_angle>-3.14159265</min_angle><max_angle>3.14159265</max_angle></horizontal>'
        f'<vertical><samples>{l.rows}</samples><resolution>1</resolution>'
        f'<min_angle>{l.vfov_min}</min_angle><max_angle>{l.vfov_max}</max_angle></vertical>'
        '</scan>'
        f'<range><min>{l.min_range}</min><max>{l.max_range}</max><resolution>0.001</resolution></range>'
        f'<noise><type>gaussian</type><mean>0</mean><stddev>{l.noise}</stddev></noise>'
        '</lidar></sensor></gazebo>')
