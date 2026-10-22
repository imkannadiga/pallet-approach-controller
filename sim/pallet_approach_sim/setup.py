from glob import glob

from setuptools import find_packages, setup

package_name = 'pallet_approach_sim'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/urdf', glob('urdf/*.xacro')),
        ('share/' + package_name + '/worlds', glob('worlds/*.sdf')),
        ('share/' + package_name + '/config', glob('config/*.json')),
        ('share/' + package_name + '/meshes/forklift', glob('meshes/forklift/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Athreya',
    maintainer_email='athreyahosahalliprakash@gmail.com',
    description='Gazebo simulation of the lab SCOUT and the measured pallet.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_tf = pallet_approach_sim.camera_tf:main',
            'camera_info_fix = pallet_approach_sim.camera_info_fix:main',
            'fork_lift_bridge = pallet_approach_sim.fork_lift_bridge:main',
        ],
    },
)
