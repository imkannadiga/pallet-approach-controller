"""Launch the whole stack, idle, on the Jetson.

Nothing moves when this starts. The two lifecycle nodes come up unconfigured,
the watchdog and the overlay start immediately (both are read-only until
something goes wrong), twist_mux starts so that there is somewhere for commands
to go, and the coordinator waits for an action goal:

    ros2 action send_goal /pallet_task pallet_approach_interfaces/action/PalletTask "{mode: pickup}" --feedback

Order matters in one place: the approach controller refuses to configure if
nothing subscribes to its command topic (D31), and twist_mux is what subscribes.
It is started here, in the same launch file, so that the normal way of running
the stack cannot produce a robot that reports it is driving and does not move.

Arguments:
    params:=<path>       override config/params.yaml
    twist_mux:=false     do not start the mux (then nothing drives the base)
    overlay:=false       skip the fork overlay
    watchdog:=false      skip the watchdog -- for bench work only
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
    mux_params = os.path.join(share, 'config', 'twist_mux.yaml')

    params = LaunchConfiguration('params')
    overlay = LaunchConfiguration('overlay')
    watchdog = LaunchConfiguration('watchdog')
    mux = LaunchConfiguration('twist_mux')
    log_level = LaunchConfiguration('log_level')
    arguments = ['--ros-args', '--log-level', log_level]

    return LaunchDescription([
        DeclareLaunchArgument('params', default_value=default_params,
                              description='parameter file'),
        DeclareLaunchArgument('overlay', default_value='true',
                              description='run the fork overlay'),
        DeclareLaunchArgument('watchdog', default_value='true',
                              description='run the health watchdog and safety stop'),
        DeclareLaunchArgument('twist_mux', default_value='true',
                              description='run twist_mux; without it nothing drives'),
        DeclareLaunchArgument('log_level', default_value='info'),

        # Arbitration: safety stop > pallet controller > Nav2 (D3, D10, D30).
        # twist_mux publishes `cmd_vel_out`, which is remapped onto the topic
        # the scout driver subscribes to.
        Node(
            package='twist_mux', executable='twist_mux', name='twist_mux',
            output='screen', parameters=[mux_params],
            remappings=[('cmd_vel_out', '/cmd_vel')],
            arguments=arguments, condition=IfCondition(mux),
        ),

        # Independent of the coordinator on purpose: it watches the sensors
        # itself and drives the mux's highest-priority input, so a coordinator
        # that has wedged still leaves a stationary robot.
        Node(
            package='pallet_approach', executable='watchdog', name='watchdog',
            output='screen', parameters=[params], arguments=arguments,
            condition=IfCondition(watchdog),
        ),

        # No hardware behind it; the coordinator waits for it exactly as it
        # would wait for a real carriage.
        Node(
            package='pallet_approach', executable='virtual_forks',
            name='virtual_forks', output='screen',
            parameters=[params], arguments=arguments,
        ),

        # Passive: it publishes a track per estimate and drives nothing. It
        # resets on both edges of a run, so one run never inherits another's
        # track.
        Node(
            package='pallet_approach', executable='pallet_tracker',
            name='pallet_tracker', output='screen',
            parameters=[params], arguments=arguments,
        ),

        # Managed by the coordinator. They come up unconfigured, so the
        # controller cannot publish a command until a run actually starts.
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
