"""Launch the whole stack, idle, on the Jetson.

Nothing moves when this starts. The two lifecycle nodes come up unconfigured,
the overlay starts drawing immediately (it is harmless and doubles as a
calibration check), and the coordinator waits for an action goal:

    ros2 action send_goal /pallet_task \\
        pallet_approach_interfaces/action/PalletTask \\
        "{mode: pallet_orientation}" --feedback

Arguments:
    params:=<path>     override config/params.yaml
    overlay:=false     skip the fork overlay
    log_level:=debug
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node


def generate_launch_description():
    share = get_package_share_directory('pallet_approach')
    default_params = os.path.join(share, 'config', 'params.yaml')

    params = LaunchConfiguration('params')
    overlay = LaunchConfiguration('overlay')
    log_level = LaunchConfiguration('log_level')
    arguments = ['--ros-args', '--log-level', log_level]

    return LaunchDescription([
        DeclareLaunchArgument('params', default_value=default_params,
                              description='parameter file'),
        DeclareLaunchArgument('overlay', default_value='true',
                              description='run the fork overlay'),
        DeclareLaunchArgument('log_level', default_value='info'),

        # Managed by the coordinator. They come up unconfigured, so the
        # controller cannot publish cmd_vel until a run actually starts.
        LifecycleNode(
            package='pallet_approach', executable='pallet_perception',
            name='pallet_perception', namespace='', output='screen',
            parameters=[params], arguments=arguments,
        ),
        LifecycleNode(
            package='pallet_approach', executable='approach_controller',
            name='approach_controller', namespace='', output='screen',
            parameters=[params], arguments=arguments,
        ),
        Node(
            package='pallet_approach', executable='coordinator',
            name='coordinator', output='screen',
            parameters=[params], arguments=arguments,
        ),
        Node(
            package='pallet_approach', executable='fork_overlay',
            name='fork_overlay', output='screen',
            parameters=[params], arguments=arguments,
            condition=IfCondition(overlay),
        ),
    ])
