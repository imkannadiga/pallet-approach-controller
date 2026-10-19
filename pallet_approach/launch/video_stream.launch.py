"""Start MediaMTX and the video streamers together, on the Jetson.

    ros2 launch pallet_approach video_stream.launch.py
    ros2 launch pallet_approach video_stream.launch.py encoder:=x264 width:=960 height:=540
    ros2 launch pallet_approach video_stream.launch.py raw_camera:=false

Then on the laptop (connected to the robot's Wi-Fi):

    http://<jetson-ip>:8889/dashboard   <- everything: forks, pockets, health,
                                            tracker, controller, forks, why
    http://<jetson-ip>:8889/zed         <- the raw rectified left image, for
                                            checking the camera itself

Two independent GStreamer pipelines, each its own `video_streamer` instance,
publishing under different MediaMTX paths -- exactly what `video_stream.py`'s
own docstring anticipated ("one instance per stream (camera, later the fork
overlay)") before there was a second stream to point at. `dashboard` is the
one that answers "what is the robot doing and why"; `zed` is the one for
checking the camera and the fork-overlay calibration itself, with nothing
drawn on it. Run this file on the Jetson, alongside
`pallet_approach.launch.py`, not instead of it -- this file only streams
images that something else has to be producing.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # The filename here has to match what actually ships: config/mediamtx.yaml
    # (not .yml -- the first version of this file named the wrong extension
    # and would have failed to find the file the moment anyone ran it from an
    # installed share/ directory instead of the source tree).
    config = os.path.join(get_package_share_directory('pallet_approach'),
                          'config', 'mediamtx.yaml')

    args = [
        DeclareLaunchArgument('encoder', default_value='auto'),     # auto | nvenc | x264 | vp8
        DeclareLaunchArgument('width', default_value='1280'),
        DeclareLaunchArgument('height', default_value='720'),
        DeclareLaunchArgument('fps', default_value='15'),
        DeclareLaunchArgument('bitrate_kbps', default_value='3000'),
        DeclareLaunchArgument(
            'raw_camera', default_value='true',
            description='also stream the unannotated camera feed at /zed'),
    ]

    mediamtx = ExecuteProcess(cmd=['mediamtx', config], output='screen', name='mediamtx')

    # The dashboard stream: dashboard_node composites fork_overlay's image
    # with health, the tracker, the controller and the forks, and republishes
    # it on /dashboard/image (pallet_approach.launch.py has to be running for
    # that topic to exist at all -- this file only streams it). Fixed
    # width/height rather than the shared LaunchConfiguration args above,
    # because dashboard.frame_size computes an exact size from
    # config/params.yaml's `dashboard` block (640x360 image + a fixed-height
    # HUD panel), and feeding GStreamer a different size would stretch it.
    # test_dashboard.py checks this literal pair against that computation, so
    # changing the dashboard's font_scale or line_height means updating this
    # number too, deliberately -- not silently.
    dashboard_stream = Node(
        package='pallet_approach',
        executable='video_streamer',
        name='video_streamer_dashboard',
        output='screen',
        parameters=[{
            'image_topic': '/dashboard/image',
            'stream_path': 'dashboard',
            'encoder': LaunchConfiguration('encoder'),
            'width': 640,
            'height': 520,
            # 5 fps: dashboard_node redraws at config/params.yaml's
            # dashboard.publish_rate (5 Hz), so encoding faster than that
            # would just send duplicate frames.
            'fps': 5,
            # Mostly static text and a slowly-changing camera frame compress
            # far better than the live raw feed; 1200 kbps is generous for
            # that content at 640x520.
            'bitrate_kbps': 1200,
        }],
    )

    # The raw camera, unannotated -- for checking the camera itself (exposure,
    # focus, the fork-overlay calibration against a clean picture) separately
    # from anything this stack has drawn on it. Optional: two concurrent
    # encodes is real Jetson load, and most of the time the dashboard is the
    # one that answers the actual question.
    camera_stream = Node(
        package='pallet_approach',
        executable='video_streamer',
        name='video_streamer_zed',
        output='screen',
        condition=IfCondition(LaunchConfiguration('raw_camera')),
        parameters=[{
            'image_topic': '/zed2/zed_node/rgb/color/rect/image',
            'stream_path': 'zed',
            'encoder': LaunchConfiguration('encoder'),
            'width': LaunchConfiguration('width'),
            'height': LaunchConfiguration('height'),
            'fps': LaunchConfiguration('fps'),
            'bitrate_kbps': LaunchConfiguration('bitrate_kbps'),
        }],
    )

    return LaunchDescription(args + [mediamtx, dashboard_stream, camera_stream])
