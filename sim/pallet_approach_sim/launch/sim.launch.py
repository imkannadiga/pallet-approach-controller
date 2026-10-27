"""Gazebo (Fortress) with the lab SCOUT and the measured pallet.

Publishes exactly the topics and tf the robot does -- the ZED's depth, image
and camera_info, /ouster/points, /odom, odom -> base_link, the URDF's static
tree and the ZED's one dynamic camera link -- and listens on /cmd_vel. The
stack is started separately, unchanged, against those topics (sim/run_sim.sh).

    ros2 launch pallet_approach_sim sim.launch.py
    ros2 launch pallet_approach_sim sim.launch.py face_x:=2.5 face_y:=0.3 face_yaw:=0.2
    ros2 launch pallet_approach_sim sim.launch.py face:=long
    ros2 launch pallet_approach_sim sim.launch.py gui:=true

Pallet placement: the robot spawns at the origin looking along +x. face_x and
face_y put the centre of the chosen face on the floor; face_yaw turns the face
away from looking straight back at the robot (radians, counter-clockwise).
"""

import json
import math
import os
import subprocess
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from pallet_approach_sim import forklift_model, pallet_model, robot_model


def _setup(context):
    share = get_package_share_directory('pallet_approach_sim')
    arg = lambda name: LaunchConfiguration(name).perform(context)   # noqa: E731

    camera = robot_model.Camera(
        width=int(arg('camera_width')), height=int(arg('camera_height')),
        rate=float(arg('camera_rate')), depth_noise=float(arg('depth_noise')))
    robot = arg('robot')
    work = tempfile.mkdtemp(prefix='pallet_sim_')
    extra = []
    if robot == 'forklift':
        with open(os.path.join(share, 'config', 'forklift_geometry.json')) as f:
            geometry = json.load(f)
        expanded = forklift_model.forklift_urdf(
            geometry, 'file://' + os.path.join(share, 'meshes', 'forklift'))
        drive = robot_model.Drive(
            wheel_separation=forklift_model.wheel_separation(geometry),
            wheel_radius=geometry['front_left']['radius'],
            left_joints=('front_left_wheel',), right_joints=('front_right_wheel',),
            joint_states=('front_left_wheel', 'front_right_wheel',
                          'rear_left_wheel', 'rear_right_wheel'))
        gz_urdf = robot_model.for_gazebo(expanded, camera=camera, drive=drive,
                                         lift=robot_model.Lift(forklift_model.LIFT_JOINT))
        # URDF cannot carry visibility flags; the SDF Gazebo would make of it can.
        urdf_file = os.path.join(work, 'robot.urdf')
        with open(urdf_file, 'w') as f:
            f.write(gz_urdf)
        sdf = subprocess.run(['ign', 'sdf', '-p', urdf_file], check=True,
                             capture_output=True, text=True).stdout
        robot_description = robot_model.hide_robot_from_sensors(sdf)
        robot_file = os.path.join(work, 'robot.sdf')
        extra.append(Node(package='pallet_approach_sim', executable='fork_lift_bridge',
                          output='screen',
                          parameters=[{'rest_height': geometry['tine']['bottom_above_floor']},
                                      {'use_sim_time': True}]))
        tine = geometry['tine']
        blades = []
        for side in (1, -1):
            centre = side * tine['centre_y']
            blades += [tine['heel_x'], tine['tip_x'], centre - tine['width'] / 2,
                       centre + tine['width'] / 2, tine['bottom_z'], tine['top_z']]
        pallet_static = False
    elif robot == 'scout':
        blades = [0.0]          # no forks: the depth filter passes frames through
        import xacro   # only needed here, and only inside the sim image
        expanded = xacro.process_file(
            os.path.join(share, 'urdf', 'scout_sim.urdf.xacro')).toxml()
        robot_description = robot_model.for_gazebo(expanded, camera=camera)
        robot_file = os.path.join(work, 'robot.urdf')
        pallet_static = True
    else:
        raise ValueError(f"robot must be 'forklift' or 'scout', not {robot!r}")
    rsp_urdf = robot_model.for_state_publisher(expanded)
    parent, child, xyz, rpy = robot_model.camera_joint_origin(expanded)

    face = arg('face')
    face_x, face_y, face_yaw = (float(arg(n)) for n in ('face_x', 'face_y', 'face_yaw'))
    px, py, pyaw = pallet_model.placement(face, face_x, face_y, face_yaw)

    pallet_file = os.path.join(work, 'pallet.sdf')
    with open(robot_file, 'w') as f:
        f.write(robot_description)
    with open(pallet_file, 'w') as f:
        f.write(pallet_model.sdf(
            static=pallet_static, block_protrusion=float(arg('block_protrusion')),
            load_height=float(arg('load_height')),
            load_overhang=float(arg('load_overhang'))))

    world = os.path.join(share, 'worlds', {'room': 'pallet_room.sdf',
                                           'empty': 'pallet_lab.sdf'}[arg('world')])
    gz_cmd = ['ign', 'gazebo', '-r', '-v', '2', world]
    if arg('gui').lower() not in ('true', '1'):
        # Server only, rendering offscreen for the camera and LiDAR.
        gz_cmd[2:2] = ['-s', '--headless-rendering']

    sim_time = {'use_sim_time': True}
    return [
        LogInfo(msg=(
            f'[truth] {face} face centre at ({face_x:.3f}, {face_y:.3f}) m, yaw {face_yaw:+.3f} rad '
            f'({math.degrees(face_yaw):+.1f} deg); pallet model at ({px:.3f}, {py:.3f}), '
            f'yaw {pyaw:+.3f}; block protrusion {float(arg("block_protrusion")):.3f} m, '
            f'load {float(arg("load_height")):.2f} m tall, overhang {float(arg("load_overhang")):.3f} m. '
            f'Robot at the origin. Files in {work}')),
        ExecuteProcess(cmd=gz_cmd, output='screen'),
        Node(package='ros_gz_bridge', executable='parameter_bridge', name='sim_bridge',
             output='screen',
             parameters=[{'config_file': os.path.join(share, 'config', 'bridge.yaml')},
                         sim_time]),
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             output='screen',
             parameters=[{'robot_description': rsp_urdf}, sim_time]),
        Node(package='pallet_approach_sim', executable='camera_tf', output='screen',
             parameters=[{'parent': parent, 'child': child,
                          'xyz': list(xyz), 'rpy': list(rpy)}, sim_time]),
        Node(package='pallet_approach_sim', executable='camera_info_fix', output='screen',
             parameters=[{'width': camera.width, 'height': camera.height,
                          'hfov': camera.hfov}, sim_time]),
        Node(package='ros_gz_sim', executable='create', output='screen',
             arguments=['-world', 'pallet_lab', '-file', robot_file, '-name', 'robot',
                        '-z', '0.25']),
        Node(package='ros_gz_sim', executable='create', output='screen',
             arguments=['-world', 'pallet_lab', '-file', pallet_file, '-name', 'pallet',
                        '-x', str(px), '-y', str(py), '-Y', str(pyaw)]),
        Node(package='pallet_approach_sim', executable='depth_self_filter', output='screen',
             parameters=[{'blades': blades, 'hfov': camera.hfov,
                          'zed_model': arg('depth_model') == 'zed',
                          'seed': int(arg('seed'))}, sim_time]),
        *extra,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('robot', default_value='forklift',
                              description="'forklift' (default) or 'scout'"),
        DeclareLaunchArgument('gui', default_value='false',
                              description='open the Gazebo window (needs X11)'),
        DeclareLaunchArgument('face', default_value='short',
                              description='which pallet face the robot is looking at'),
        DeclareLaunchArgument('face_x', default_value='3.5',
                              description='face centre, metres ahead of the robot'),
        DeclareLaunchArgument('face_y', default_value='0.10',
                              description='face centre, metres to the robot\'s left'),
        DeclareLaunchArgument('face_yaw', default_value='0.05',
                              description='face turned away from square, radians CCW'),
        DeclareLaunchArgument('camera_width', default_value='1280'),
        DeclareLaunchArgument('camera_height', default_value='720'),
        # 30 Hz, near the ZED's ~37, so the 25 fps dashboard is reachable in
        # the sim too. A weak laptop may need camera_rate:=15 (dashboard then
        # tops out at 15).
        DeclareLaunchArgument('camera_rate', default_value='30.0',
                              description='Hz; perception uses 5, the dashboard 25'),
        # D80: the ZED's measured error is a smooth warp plus filled pockets
        # and smeared edges (zed_depth_model.py), not per-pixel noise, so
        # Gazebo's own gaussian noise is off by default.
        DeclareLaunchArgument('depth_model', default_value='zed',
                              description="'zed' (measured ZED effects) or 'none'"),
        DeclareLaunchArgument('depth_noise', default_value='0.0',
                              description='extra gaussian depth noise, metres'),
        DeclareLaunchArgument('seed', default_value='0',
                              description='depth model seed; -1 for a fresh one'),
        DeclareLaunchArgument('block_protrusion', default_value='0.0',
                              description='metres the outer blocks stand proud of the deck '
                                          'at the short ends (the real pallet: ~0.02)'),
        DeclareLaunchArgument('load_height', default_value='0.0',
                              description='metres of load on the deck; 0 = none'),
        DeclareLaunchArgument('load_overhang', default_value='0.0',
                              description='metres the load overhangs the pallet on every side'),
        DeclareLaunchArgument('world', default_value='room',
                              description="'room' (walls and distractors) or 'empty'"),
        OpaqueFunction(function=_setup),
    ])
