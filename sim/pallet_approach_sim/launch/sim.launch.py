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

import math
import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from pallet_approach_sim import pallet_model, robot_model


def _setup(context):
    share = get_package_share_directory('pallet_approach_sim')
    arg = lambda name: LaunchConfiguration(name).perform(context)   # noqa: E731

    import xacro   # only needed here, and only inside the sim image
    expanded = xacro.process_file(
        os.path.join(share, 'urdf', 'scout_sim.urdf.xacro')).toxml()

    camera = robot_model.Camera(
        width=int(arg('camera_width')), height=int(arg('camera_height')),
        rate=float(arg('camera_rate')), depth_noise=float(arg('depth_noise')))
    gz_urdf = robot_model.for_gazebo(expanded, camera=camera)
    rsp_urdf = robot_model.for_state_publisher(expanded)
    parent, child, xyz, rpy = robot_model.camera_joint_origin(expanded)

    face = arg('face')
    face_x, face_y, face_yaw = (float(arg(n)) for n in ('face_x', 'face_y', 'face_yaw'))
    px, py, pyaw = pallet_model.placement(face, face_x, face_y, face_yaw)

    work = tempfile.mkdtemp(prefix='pallet_sim_')
    robot_file = os.path.join(work, 'scout.urdf')
    pallet_file = os.path.join(work, 'pallet.sdf')
    with open(robot_file, 'w') as f:
        f.write(gz_urdf)
    with open(pallet_file, 'w') as f:
        f.write(pallet_model.sdf())

    world = os.path.join(share, 'worlds', 'pallet_lab.sdf')
    gz_cmd = ['ign', 'gazebo', '-r', '-v', '2', world]
    if arg('gui').lower() not in ('true', '1'):
        # Server only, rendering offscreen for the camera and LiDAR.
        gz_cmd[2:2] = ['-s', '--headless-rendering']

    sim_time = {'use_sim_time': True}
    return [
        LogInfo(msg=(
            f'[truth] {face} face centre at ({face_x:.3f}, {face_y:.3f}) m, yaw {face_yaw:+.3f} rad '
            f'({math.degrees(face_yaw):+.1f} deg); pallet model at ({px:.3f}, {py:.3f}), '
            f'yaw {pyaw:+.3f}. Robot at the origin. Files in {work}')),
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
             arguments=['-world', 'pallet_lab', '-file', robot_file, '-name', 'scout',
                        '-z', '0.25']),
        Node(package='ros_gz_sim', executable='create', output='screen',
             arguments=['-world', 'pallet_lab', '-file', pallet_file, '-name', 'pallet',
                        '-x', str(px), '-y', str(py), '-Y', str(pyaw)]),
    ]


def generate_launch_description():
    return LaunchDescription([
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
        DeclareLaunchArgument('camera_rate', default_value='15.0',
                              description='Hz; perception uses 5, the watchdog wants 10'),
        DeclareLaunchArgument('depth_noise', default_value='0.01',
                              description='gaussian depth noise, metres'),
        OpaqueFunction(function=_setup),
    ])
