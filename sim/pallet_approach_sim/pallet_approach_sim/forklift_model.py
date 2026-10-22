"""The forklift, as a URDF the rest of the sim treats exactly like the SCOUT.

Athreya's decisions, 2026-10-22 (sensor placement and the URDF only):

* The OpenRobotics Forklift from Gazebo Fuel (CC BY 4.0, Cole Biesemeyer),
  split into body, carriage, forks and wheels by sim/tools/build_forklift.py
  (D10). Masses, joints and plugins are added here; it ships with none.
* Forks 0.365 m apart, the short-side pocket centres (D11).
* The camera and the LiDAR exactly where they are on the SCOUT, relative to
  base_footprint, with the SCOUT's frame names -- so the stack's extrinsics,
  the LiDAR cross-check and the insertion stop are unchanged. The forklift's
  own body is hidden from both sensors (robot_model.hide_robot_from_sensors),
  so they see what the SCOUT's would. This replaces D12/D14's mast camera and
  mast-top 2D LiDAR.
* base_footprint placed so the fork heel is 0.4625 m ahead of it, where the
  virtual forks start (forks.carriage_x); base_footprint 25 mm above the floor
  and base_link 0.21 m above it, as on the SCOUT.
* Differential drive on the front wheels for now; the rear wheels are passive
  and slide when it turns in place, because the closing loop turns in place
  and a rear-steered (D16) truck cannot. Front axle at x = 0.048, so it pivots
  4.8 cm ahead of where the SCOUT does.
* Physical forks: the carriage is a prismatic joint that really lifts the
  (now dynamic) pallet, commanded from the virtual_forks height.
"""

from __future__ import annotations

import math

# The SCOUT's sensor mounts, from base_link, read off its URDF chain
# (stuttgart_scout macros via scout_sim.urdf.xacro), all pure translations.
CAMERA_MOUNT = (0.40, 0.0, 0.148)        # zed2_camera_link
LIDAR_MOUNT = (0.173, 0.0, 0.3321)       # os_sensor_base_link

BODY_MASS = 400.0
# Between the axles, nearer the front: with it over the rear wheels (the first
# version, -0.6) the driven front wheels carried too little weight and slipped
# -- commanded a 38 deg turn, the truck did not turn at all (measured
# 2026-10-22), and odometry, which counts wheel turns, believed it had.
BODY_COM_X = -0.25
# Front (driven) wheels grip; rear wheels are casters in effect: nearly no
# friction, so they slide sideways when the truck turns in place.
FRONT_MU = 1.2
REAR_MU = 0.02
CARRIAGE_MASS = 60.0
LIFT_JOINT = 'fork_lift'
LIFT_TOP = 1.5                           # m of travel above the mesh's resting pose


def forklift_urdf(geometry: dict, mesh_uri: str) -> str:
    """URDF string. `mesh_uri` is the directory holding the OBJ meshes, as a
    `file://` (or `package://`) prefix without the trailing slash."""
    g = geometry
    base_z = g['base_link_z']
    tine = g['tine']
    parts = ['<?xml version="1.0"?>', '<robot name="forklift">']

    def link(name, visual=None, collisions=(), mass=None, com=(0, 0, 0), box=None):
        out = [f'<link name="{name}">']
        if visual:
            out.append(f'<visual><origin xyz="0 0 0"/><geometry><mesh filename="{mesh_uri}/{visual}"/>'
                       '</geometry></visual>')
        for xyz, rpy, geom in collisions:
            out.append(f'<collision><origin xyz="{_v(xyz)}" rpy="{_v(rpy)}"/>'
                       f'<geometry>{geom}</geometry></collision>')
        if mass:
            ix, iy, iz = _box_inertia(mass, box or (0.2, 0.2, 0.2))
            out.append(f'<inertial><origin xyz="{_v(com)}"/><mass value="{mass}"/>'
                       f'<inertia ixx="{ix:.4f}" ixy="0" ixz="0" iyy="{iy:.4f}" iyz="0" '
                       f'izz="{iz:.4f}"/></inertial>')
        out.append('</link>')
        return ''.join(out)

    def fixed(name, parent, child, xyz, rpy=(0, 0, 0)):
        return (f'<joint name="{name}" type="fixed"><parent link="{parent}"/>'
                f'<child link="{child}"/><origin xyz="{_v(xyz)}" rpy="{_v(rpy)}"/></joint>')

    # --- body: one box for the chassis, one for the mast; the wheels carry it.
    lo, hi = g['body']['min'], g['body']['max']
    chassis = ((hi[0] - 0.25 + lo[0]) / 2, 0.0, 0.55 - base_z)
    chassis_size = (hi[0] - 0.25 - lo[0], hi[1] - lo[1] - 0.25, 0.75)
    mast = (hi[0] - 0.10, 0.0, 1.10 - base_z)
    parts.append(link(
        'base_link', 'body.obj',
        collisions=[(chassis, (0, 0, 0), _box(chassis_size)),
                    (mast, (0, 0, 0), _box((0.18, hi[1] - lo[1] - 0.2, 1.9)))],
        mass=BODY_MASS, com=(BODY_COM_X, 0.0, 0.35 - base_z), box=(1.9, 1.0, 1.2)))
    parts.append('<link name="base_footprint"/>')
    parts.append(fixed('base_footprint_joint', 'base_link', 'base_footprint', (0, 0, -base_z)))

    # --- wheels: front driven, rear passive (they slide when it turns in place).
    for name in ('front_left', 'front_right', 'rear_left', 'rear_right'):
        w = g[name]
        x, y, z = w['xyz']
        radius = w['radius'] if name.startswith('front') else z - g['floor_z']
        # Rear: radius to the floor, not the mesh's 0.238, whose centre is 16 mm
        # too low for the front wheels' 0.285 -- a 0.7 deg pitch at the camera.
        geom = (f'<cylinder radius="{radius:.4f}" length="0.20"/>' if name.startswith('front')
                else f'<sphere radius="{radius:.4f}"/>')
        parts.append(link(f'{name}_wheel_link', f'{name}.obj',
                          collisions=[((0, 0, 0), (math.pi / 2, 0, 0) if name.startswith('front')
                                       else (0, 0, 0), geom)],
                          mass=20.0, box=(radius * 2, 0.2, radius * 2)))
        parts.append(f'<joint name="{name}_wheel" type="continuous"><parent link="base_link"/>'
                     f'<child link="{name}_wheel_link"/><origin xyz="{_v((x, y, z - base_z))}"/>'
                     '<axis xyz="0 1 0"/></joint>')

    for name, mu in (('front_left', FRONT_MU), ('front_right', FRONT_MU),
                     ('rear_left', REAR_MU), ('rear_right', REAR_MU)):
        parts.append(f'<gazebo reference="{name}_wheel_link"><mu1>{mu}</mu1><mu2>{mu}</mu2>'
                     '</gazebo>')

    # --- carriage and forks, on the lift.
    length = tine['tip_x'] - tine['heel_x']
    blade_z = (tine['bottom_z'] + tine['top_z']) / 2 - base_z
    blade = (length, tine['width'], tine['top_z'] - tine['bottom_z'])
    plate = (tine['heel_x'] - 0.03, 0.0, 0.70 + g['floor_z'] - base_z)
    collisions = [((tine['heel_x'] + length / 2, side * tine['centre_y'], blade_z), (0, 0, 0),
                   _box(blade)) for side in (1, -1)]
    collisions.append((plate, (0, 0, 0), _box((0.06, 1.0, 1.0))))
    parts.append(
        '<link name="carriage">'
        + ''.join(f'<visual><geometry><mesh filename="{mesh_uri}/{m}"/></geometry></visual>'
                  for m in ('carriage.obj', 'tine_left.obj', 'tine_right.obj'))
        + ''.join(f'<collision><origin xyz="{_v(xyz)}" rpy="{_v(rpy)}"/><geometry>{geom}'
                  '</geometry></collision>' for xyz, rpy, geom in collisions)
        + f'<inertial><origin xyz="{_v((tine["heel_x"] + 0.2, 0, blade_z + 0.3))}"/>'
          f'<mass value="{CARRIAGE_MASS}"/><inertia ixx="5" ixy="0" ixz="0" iyy="5" iyz="0" izz="5"/>'
          '</inertial></link>')
    parts.append(f'<joint name="{LIFT_JOINT}" type="prismatic"><parent link="base_link"/>'
                 '<child link="carriage"/><origin xyz="0 0 0"/><axis xyz="0 0 1"/>'
                 f'<limit lower="{-tine["bottom_above_floor"]:.4f}" upper="{LIFT_TOP}" '
                 'effort="20000" velocity="0.5"/></joint>')

    # --- sensors, exactly where the SCOUT has them, with its frame names.
    parts.append('<link name="zed2_camera_link"/>')
    parts.append(fixed('zed2_camera_joint', 'base_link', 'zed2_camera_link', CAMERA_MOUNT))
    parts.append('<link name="zed2_camera_center"/>')
    parts.append(fixed('zed2_camera_center_joint', 'zed2_camera_link', 'zed2_camera_center',
                       (0, 0, 0.015)))
    parts.append('<link name="zed2_left_camera_frame"/>')
    parts.append(fixed('zed2_left_camera_joint', 'zed2_camera_center', 'zed2_left_camera_frame',
                       (-0.01, 0.06, 0)))
    parts.append('<link name="zed2_left_camera_frame_optical"/>')
    parts.append(fixed('zed2_left_camera_optical_joint', 'zed2_left_camera_frame',
                       'zed2_left_camera_frame_optical', (0, 0, 0),
                       (-math.pi / 2, 0, -math.pi / 2)))
    parts.append('<link name="os_sensor_base_link"/>')
    parts.append(fixed('os_sensor_mount_joint', 'base_link', 'os_sensor_base_link', LIDAR_MOUNT))
    parts.append('<link name="os_sensor"/>')
    parts.append(fixed('os_sensor_joint', 'os_sensor_base_link', 'os_sensor', (0, 0, 0.05)))
    parts.append('<link name="os_lidar"/>')
    parts.append(fixed('os_lidar_joint', 'os_sensor', 'os_lidar', (0, 0, 0.03618)))

    parts.append('</robot>')
    return '\n'.join(parts)


def wheel_separation(geometry: dict) -> float:
    return abs(geometry['front_left']['xyz'][1] - geometry['front_right']['xyz'][1])


def _v(values) -> str:
    return ' '.join(f'{float(v):.5f}' for v in values)


def _box(size) -> str:
    return f'<box size="{_v(size)}"/>'


def _box_inertia(mass, size):
    x, y, z = size
    return (mass * (y * y + z * z) / 12, mass * (x * x + z * z) / 12, mass * (x * x + y * y) / 12)
