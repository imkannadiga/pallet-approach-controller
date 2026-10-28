from glob import glob

from setuptools import find_packages, setup

package_name = 'pallet_approach'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        # mediamtx.yaml and video_stream.launch.py are already caught by the
        # two globs above -- these two extra entries named 'mediamtx.yml'
        # (wrong extension; the file is .yaml) pointed at a path that does not
        # exist, which would have failed the package build the first time
        # anyone actually ran it from an installed share/ directory rather
        # than from the source tree.
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Athreya',
    maintainer_email='athreyahosahalliprakash@gmail.com',
    description='Pallet perception and fork-insertion approach on a SCOUT 2.0.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pallet_perception = pallet_approach.nodes.perception_node:main',
            'approach_controller = pallet_approach.nodes.controller_node:main',
            'coordinator = pallet_approach.nodes.coordinator_node:main',
            'watchdog = pallet_approach.nodes.watchdog_node:main',
            'pallet_tracker = pallet_approach.nodes.tracker_node:main',
            'virtual_forks = pallet_approach.nodes.forks_node:main',
            'fork_overlay = pallet_approach.nodes.fork_overlay_node:main',
            'dashboard = pallet_approach.nodes.dashboard_node:main',
            'video_streamer = pallet_approach.nodes.video_streamer:main',
            'lidar_relay = pallet_approach.nodes.lidar_relay_node:main',
            'pallet_fusion = pallet_approach.nodes.fusion_node:main',
        ],
    },
)
