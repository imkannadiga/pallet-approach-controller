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
    cmd_vel:=/cmd_vel_dry   DRY RUN: arbitrate as normal, but send the winning
                         command somewhere the base is not listening. The whole
                         sequence runs on a real pallet with the wheels
                         disconnected. See below.
    twist_mux:=false     do not start the mux (then the controller refuses to
                         configure, so a goal fails immediately)
    video_stream:=false  do not start MediaMTX or the video streamers (no
                         GStreamer/MediaMTX needed then -- useful on a
                         desk/CI box that has neither installed)
    encoder:=x264        forwarded to video_stream.launch.py: auto | nvenc |
                         x264 | vp8
    overlay:=false       skip the fork overlay
    dashboard:=false     skip the HUD dashboard (needs overlay:=true)
    watchdog:=false      skip the watchdog -- for bench work only
    log_level:=debug

DRY RUN, and this is the way to bring the stack up on the robot for the first
time:

    ros2 launch pallet_approach pallet_approach.launch.py cmd_vel:=/cmd_vel_dry

Everything runs -- perception, the tracker, the health checks, the full state
machine, the controller, the forks, the overlay -- and `twist_mux` arbitrates
exactly as it will in a real run. The only difference is where its output goes,
so the base never receives a command and the robot cannot move. Watch it with:

    ros2 topic echo /cmd_vel_dry
    ros2 topic echo /approach_controller/status
    ros2 topic echo /coordinator/status
    ros2 topic echo /health

Or watch it as video: run video_stream.launch.py alongside this file and open
http://<jetson-ip>:8889/dashboard in a browser -- the dashboard shows every
one of those topics composited onto the camera image, which is a much faster
way to watch a dry run than six terminals of `ros2 topic echo`.

The robot stays still, so the closing loop never arrives and the run ends on
the 240 s timeout with the pallet still 3 m away. That is the expected outcome
and it is not a failure: what the dry run proves is that every node comes up,
discovers the others, reads the right topics, agrees about tf and the clock,
and produces sane commands for a real pallet. Which is precisely the layer that
no offline test covers.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node


def generate_launch_description():
    share = get_package_share_directory('pallet_approach')
    default_params = os.path.join(share, 'config', 'params.yaml')
    mux_params = os.path.join(share, 'config', 'twist_mux.yaml')

    params = LaunchConfiguration('params')
    cmd_vel = LaunchConfiguration('cmd_vel')
    overlay = LaunchConfiguration('overlay')
    dashboard = LaunchConfiguration('dashboard')
    watchdog = LaunchConfiguration('watchdog')
    video_stream = LaunchConfiguration('video_stream')
    encoder = LaunchConfiguration('encoder')
    mux = LaunchConfiguration('twist_mux')
    log_level = LaunchConfiguration('log_level')
    arguments = ['--ros-args', '--log-level', log_level]

    return LaunchDescription([
        DeclareLaunchArgument('params', default_value=default_params,
                              description='parameter file'),
        DeclareLaunchArgument(
            'cmd_vel', default_value='/cmd_vel',
            description='where the mux sends the winning command; set to '
                        '/cmd_vel_dry for a dry run with the wheels '
                        'disconnected'),
        DeclareLaunchArgument('overlay', default_value='true',
                              description='run the fork overlay'),
        DeclareLaunchArgument(
            'dashboard', default_value='true',
            description='composite health/tracker/controller/fork telemetry '
                        'onto the overlay image; needs overlay:=true'),
        DeclareLaunchArgument('watchdog', default_value='true',
                              description='run the health watchdog and safety stop'),
        DeclareLaunchArgument('twist_mux', default_value='true',
                              description='run twist_mux; without it nothing drives'),
        DeclareLaunchArgument(
            'video_stream', default_value='true',
            description='start MediaMTX and the WebRTC video streams'),
        DeclareLaunchArgument('encoder', default_value='x264',
                              description='forwarded to video_stream.launch.py'),
        DeclareLaunchArgument('log_level', default_value='info'),

        # Arbitration: safety stop > pallet controller > Nav2 (D3, D10, D30).
        # twist_mux publishes `cmd_vel_out`, which is remapped onto the topic
        # the scout driver subscribes to -- or, for a dry run, onto one it is
        # not. Nothing else in the stack changes, which is the point: the dry
        # run exercises the same arbitration the real run will.
        Node(
            package='twist_mux', executable='twist_mux', name='twist_mux',
            output='screen', parameters=[mux_params],
            remappings=[('cmd_vel_out', cmd_vel)],
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
        # Subscribes to fork_overlay's image, so it has nothing to draw on if
        # overlay:=false -- it still runs and still publishes the HUD panel on
        # a placeholder (dashboard.render's "NO CAMERA FEED" path), which is
        # more useful than crashing but is not what dashboard:=true is for.
        # The launch file does not enforce overlay:=true here; it is on the
        # operator to not turn one off without the other.
        Node(
            package='pallet_approach', executable='dashboard',
            name='dashboard', output='screen',
            parameters=[params], arguments=arguments,
            condition=IfCondition(dashboard),
        ),

        # MediaMTX plus both video streams (dashboard, raw camera). A separate
        # launch file (video_stream.launch.py) because it can also be run on
        # its own against an already-running stack; included here so that `up`
        # genuinely starts everything in one command, which is the point of
        # having one entry point at all.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(share, 'launch', 'video_stream.launch.py')]),
            launch_arguments={'encoder': encoder}.items(),
            condition=IfCondition(video_stream),
        ),
    ])
