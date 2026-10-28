"""Shared bag reading for the offline tools.

No ROS: `rosbags` reads the files directly, so everything here runs on a laptop
in a plain virtualenv. Humble does not embed message definitions in its bags --
that arrived in Iron -- so a typestore has to be supplied, which is what
`typestore()` is for.
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# Re-exported. The cloud decoder lives in the package, not here: the offline
# tools and the running node have to agree about what a point is, and this file
# used to hold a second copy of it.
from pallet_approach.pointcloud import unpack_cloud  # noqa: E402,F401

try:
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
except ImportError:                                           # pragma: no cover
    sys.exit('pip install rosbags numpy pyyaml  (in a venv)')

GROUND = 'base_footprint'
BUMPER_X = 0.4625          # base_footprint to the front bumper (D23); tape distances start here
CAMERA = 'zed2_left_camera_frame_optical'

CONFIG = pathlib.Path(__file__).resolve().parent.parent / 'config'
STATIC_TF = CONFIG / 'static_tf.yaml'
"""The real robot's static transforms, exported once from a good bag
(tools/export_static_tf.py). They are rigid mounting, so a bag need not carry them."""


def typestore(name: str = 'ROS2_HUMBLE'):
    return get_typestore(getattr(Stores, name))


def matrix_from(translation, xyzw) -> np.ndarray:
    """4x4 from a translation (3,) and a quaternion (x, y, z, w)."""
    x, y, z, w = xyzw
    m = np.eye(4)
    m[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    m[:3, 3] = translation
    return m


def quat_matrix(translation, rotation) -> np.ndarray:
    """4x4 from geometry_msgs Vector3 and Quaternion."""
    return matrix_from([translation.x, translation.y, translation.z],
                       [rotation.x, rotation.y, rotation.z, rotation.w])


class Tree:
    """Static tf tree with the up-then-down lookup tf2 performs.

    `base_footprint` is a *child* of `base_link`, not an ancestor of the
    sensors, so climbing parent links from a camera frame never reaches it. A
    lookup has to go up to the common ancestor and back down the other branch.
    """

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.edge: dict[str, np.ndarray] = {}

    def add(self, parent: str, child: str, matrix: np.ndarray) -> None:
        self.parent[child] = parent
        self.edge[child] = matrix

    def _to_root(self, frame: str) -> np.ndarray:
        matrix, node = np.eye(4), frame
        while node in self.parent:
            matrix = self.edge[node] @ matrix
            node = self.parent[node]
        return matrix

    def lookup(self, target: str, source: str) -> np.ndarray:
        """target <- source, as a 4x4."""
        known = set(self.parent) | set(self.parent.values())
        for frame in (target, source):
            if frame not in known:
                raise KeyError(f'{frame} is not in this bag\'s tf tree')
        return np.linalg.inv(self._to_root(target)) @ self._to_root(source)


def static_reference(path: pathlib.Path | None = None) -> list:
    """[(parent, child, 4x4)] from the static-transform reference file, or []."""
    import yaml
    path = STATIC_TF if path is None else path
    if not path.exists():
        return []
    doc = yaml.safe_load(path.read_text())
    out = []
    for e in doc.get('edges', []):
        out.append((e['parent'], e['child'], matrix_from(e['translation'], e['rotation'])))
    return out


def fill_static(tree, reference: list | None = None) -> list[str]:
    """Add the static edges a bag is missing from the reference; return their children.

    Only edges the bag LACKS are added -- what a bag does carry is never
    overridden, so a bag that has its own static tree is untouched. Static
    transforms do not change from bag to bag, and a recorder sometimes joins the
    latched /tf_static topic before it has discovered every publisher.

    This is the REAL robot's tree. A sim bag carries its own and so gets nothing
    added; a sim bag that somehow lacked one would wrongly get this one.
    """
    reference = static_reference() if reference is None else reference
    borrowed = []
    for parent, child, matrix in reference:
        if child not in tree.parent:
            tree.add(parent, child, matrix)
            borrowed.append(child)
    return borrowed


def read(path: pathlib.Path, store='ROS2_HUMBLE', max_frames: int = 5) -> dict:
    """Everything the offline tools need from one bag.

    Returns a dict with `tree`, `intrinsics`, `depth` (list of arrays),
    `image` (list of BGR arrays), `clouds` ({frame: points}) and `odom_yaw`.
    """
    ts = typestore(store)
    out = {'tree': Tree(), 'intrinsics': None, 'depth': [], 'image': [],
           'clouds': {}, 'odom_yaw': float('nan'), 'depth_frame': None}
    yaws = []

    with AnyReader([path], default_typestore=ts) as reader:
        conns = list(reader.connections)

        for conn, _, raw in reader.messages(
                connections=[c for c in conns if c.topic in ('/tf', '/tf_static')]):
            for tr in reader.deserialize(raw, conn.msgtype).transforms:
                out['tree'].add(tr.header.frame_id, tr.child_frame_id,
                                quat_matrix(tr.transform.translation,
                                            tr.transform.rotation))

        for conn in [c for c in conns if c.topic == '/odom']:
            for _, _, raw in reader.messages(connections=[conn]):
                q = reader.deserialize(raw, conn.msgtype).pose.pose.orientation
                yaws.append(math.atan2(2 * (q.w * q.z + q.x * q.y),
                                       1 - 2 * (q.y * q.y + q.z * q.z)))

        for conn in [c for c in conns if 'depth' in c.topic and 'camera_info' in c.topic]:
            for _, _, raw in reader.messages(connections=[conn]):
                k = list(reader.deserialize(raw, conn.msgtype).k)
                out['intrinsics'] = {'fx': k[0], 'fy': k[4], 'cx': k[2], 'cy': k[5]}
                break
            break

        for conn in [c for c in conns
                     if c.msgtype.endswith('msg/Image') and 'depth' in c.topic]:
            for _, _, raw in reader.messages(connections=[conn]):
                msg = reader.deserialize(raw, conn.msgtype)
                if msg.encoding != '32FC1':
                    break
                out['depth_frame'] = msg.header.frame_id
                out['depth'].append(np.frombuffer(bytes(msg.data), np.float32)
                                    .reshape(msg.height, msg.width))
                if len(out['depth']) >= max_frames:
                    break
            break

        # The ZED publishes the rectified colour image as bgra8 -- four
        # channels, not three. Alpha is unused; drop it and return RGB.
        channels = {'bgr8': 3, 'rgb8': 3, 'bgra8': 4, 'rgba8': 4}
        for conn in [c for c in conns if c.msgtype.endswith('msg/Image')
                     and 'rect/image' in c.topic]:
            for _, _, raw in reader.messages(connections=[conn]):
                msg = reader.deserialize(raw, conn.msgtype)
                n = channels.get(msg.encoding)
                if n is None:
                    break
                rows = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, -1)
                img = rows[:, :msg.width * n].reshape(msg.height, msg.width, n)
                rgb = img[:, :, :3]
                out['image'].append(rgb[:, :, ::-1] if msg.encoding.startswith('bgr')
                                    else rgb)
                if len(out['image']) >= max_frames:
                    break
            break

        for conn in [c for c in conns if c.msgtype.endswith('msg/PointCloud2')]:
            for _, _, raw in reader.messages(connections=[conn]):
                msg = reader.deserialize(raw, conn.msgtype)
                out['clouds'].setdefault(msg.header.frame_id, []).append(
                    unpack_cloud(msg))
                break
    # Static transforms are rigid mounting; fill in any this bag arrived without.
    out['borrowed'] = fill_static(out['tree'])
    out['clouds'] = {f: np.concatenate(v) for f, v in out['clouds'].items() if v}
    if yaws:
        out['odom_yaw'] = float(np.median(yaws))
    return out


def _merge(base: dict, over: dict) -> dict:
    """`over` on top of `base`, key by key, as ROS applies an overlay file."""
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_params(package_root: pathlib.Path, overlay: str | None = None) -> dict:
    """The flat parameter dict `perception.estimate_pallet` expects.

    Mirrors `nodes/perception_node.py::_params()`, so the offline tools run the
    pipeline with exactly the parameters the robot would use. If the two drift
    apart, `test_params.py` catches it.

    `overlay` names a second file in `config/` applied on top, as the launch
    file does: 'sim_overlay.yaml' for bags recorded in the Gazebo sim, whose
    camera is mounted exactly as the URDF says and so must not get the real
    robot's tilt correction.
    """
    import yaml
    raw = yaml.safe_load((package_root / 'config' / 'params.yaml').read_text())
    if overlay:
        _merge(raw, yaml.safe_load((package_root / 'config' / overlay).read_text()))
    shared = raw['/**']['ros__parameters']
    block = raw['pallet_perception']['ros__parameters']
    return {
        'seed': block['seed'],
        'stride': block['depth']['stride'],
        'min_range': block['depth']['min_range'],
        'max_range': block['depth']['max_range'],
        'roi': dict(block['roi']),
        'voxel_size': block['voxel_size'],
        'floor_distance_threshold': block['floor']['ransac_distance'],
        'ransac_iterations': block['floor']['ransac_iterations'],
        'max_floor_tilt': block['floor']['max_normal_tilt'],
        'floor_margin': block['floor']['removal_margin'],
        'cluster_eps': block['cluster']['eps'],
        'cluster_min_samples': block['cluster']['min_samples'],
        'min_cluster_points': block['cluster']['min_cluster_points'],
        'size_tolerance': block['cluster']['size_tolerance'],
        'max_height_above_pallet': block['cluster']['max_height_above_pallet'],
        'max_candidates': block['cluster']['max_candidates'],
        'pallet_height': shared['pallet']['pallet_height'],
        'face_distance_threshold': block['face']['ransac_distance'],
        'max_tilt_from_vertical': block['face']['max_tilt_from_vertical'],
        'face_min_points': block['face']['min_points'],
        'face_deck_margin': block['face']['deck_margin'],
        'floor_score_points': block['floor']['score_points'],
        'max_facing_angle': block['face']['max_facing_angle'],
        'pocket_slab_half_depth': shared['pallet']['pocket_slab_half_depth'],
        'pocket_frame_margin': shared['pallet']['pocket_frame_margin'],
        'pocket_min_points_per_bin': shared['pallet']['pocket_min_points_per_bin'],
        'extrinsic_correction': dict(shared['frames']['extrinsic_correction']),
        **side_params(shared['pallet']),
    }


def side_params(pallet: dict) -> dict:
    """Per-side pallet geometry, plus the expected side's values flattened.

    The pipeline's parameter dict is flat -- `pocket_center_height` and friends
    name one number each, because at the point of use there is only one side in
    play. But the two faces of this pallet need different numbers (D19/D21), and
    which face the robot is looking at is not known until perception has
    classified it. So: `sides` carries both, `classify` says how to tell them
    apart, and the flat keys hold `entry_face`'s values as the starting point.
    `estimate_pallet` swaps in the classified side's values once it knows.
    """
    sides = pallet['sides']
    expected = sides[pallet['entry_face']]
    return {
        'sides': {name: dict(cfg) for name, cfg in sides.items()},
        'classify': dict(pallet['classify']),
        'entry_face': pallet['entry_face'],
        **flatten_side(expected),
    }


def flatten_side(side: dict) -> dict:
    """One side's geometry under the flat names the pipeline stages take."""
    return {
        'pallet_width': side['face_width'],
        'pallet_length': side['depth'],
        'pocket_center_height': side['band_center'],
        'pocket_band_half_height': side['band_half_height'],
        'pocket_min_gap_width': side['min_gap_width'],
        'pocket_max_gap_width': side['max_gap_width'],
        'pocket_nominal_width': side['pocket_width'],
        'pocket_nominal_height': side['opening_top'] - side['opening_bottom'],
        'pocket_nominal_spacing': side['pocket_spacing'],
        'pocket_opening_bottom': side['opening_bottom'],
        'pocket_opening_top': side['opening_top'],
    }


def lidar_to_ground(tree, lidar_frame: str, camera_frame: str, shared: dict, mode: str = 'auto') -> np.ndarray:
    """4x4 ``ground <- lidar`` for a bag, the way the robot's nodes compute it.

    `shared` is the `/**` parameter block (params.yaml, plus any overlay). Old
    bags carry the LiDAR's URDF frame, so the calibration (D101) is applied here
    from config/lidar_calibration.yaml rather than read from the bag's tf.

    mode: 'auto' -- whatever params.yaml has in force; 'calibrated', 'd74' or
    'urdf' to force one, for comparisons.
    """
    from pallet_approach.calibration import (correct_lidar_transform, correction_matrix,
                                             lidar_follows_camera, read_lidar_calibration)
    frames = shared['frames']
    if mode == 'auto':
        mode = ('calibrated' if frames.get('lidar_calibration', {}).get('enabled')
                else 'd74' if frames.get('lidar_correction', {}).get('enabled') else 'urdf')
    if mode == 'calibrated':
        _, _, cam_T_lidar = read_lidar_calibration(CONFIG / 'lidar_calibration.yaml')
        g_c = tree.lookup(GROUND, camera_frame)
        ec = frames.get('extrinsic_correction', {})
        c = (correction_matrix(ec['measured_floor_pitch'], ec['measured_floor_roll'])
             if ec.get('enabled') else np.eye(3))
        return lidar_follows_camera(g_c @ cam_T_lidar, g_c, c)
    to_ground = tree.lookup(GROUND, lidar_frame)
    if mode == 'd74':
        lc = frames['lidar_correction']
        to_ground = correct_lidar_transform(to_ground, lc['measured_floor_pitch'], lc['measured_floor_roll'],
                                            lc['measured_floor_height'], shared['floor_z'])
    return to_ground


def read_clouds(path: pathlib.Path, count: int) -> list:
    """The first `count` LiDAR clouds as (frame_id, (N, 3) array)."""
    from pallet_approach.pointcloud import unpack_cloud
    out = []
    with AnyReader([path], default_typestore=typestore()) as reader:
        conns = [c for c in reader.connections if c.msgtype.endswith('msg/PointCloud2')]
        for conn, _, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            out.append((msg.header.frame_id, unpack_cloud(msg)))
            if len(out) >= count:
                break
    return out


def shared_params(overlay: str | None = None) -> dict:
    """params.yaml's `/**` block, nested as written, with an overlay file merged on top."""
    import yaml
    raw = yaml.safe_load((CONFIG / 'params.yaml').read_text())
    if overlay:
        _merge(raw, yaml.safe_load((CONFIG / overlay).read_text()))
    return raw['/**']['ros__parameters']


def camera_to_ground(data: dict, correction: dict) -> np.ndarray:
    """4x4 ground <- camera optical for a bag read by `read`, tilt-corrected if
    `correction` (an `extrinsic_correction` block) is enabled."""
    from pallet_approach.calibration import correct_transform, correction_matrix
    matrix = data['tree'].lookup(GROUND, data['depth_frame'] or CAMERA)
    if correction.get('enabled'):
        matrix = correct_transform(matrix, correction_matrix(
            correction['measured_floor_pitch'], correction['measured_floor_roll']))
    return matrix


def lidar_sweeps(path: pathlib.Path, data: dict, shared: dict, count: int,
                 mode: str = 'auto') -> list[np.ndarray]:
    """The first `count` LiDAR sweeps of a bag in the ground frame, LiDAR pose as `lidar_to_ground`."""
    clouds = read_clouds(path, count)
    if not clouds:
        return []
    to_ground = lidar_to_ground(data['tree'], clouds[0][0], data['depth_frame'] or CAMERA, shared, mode)
    return [c @ to_ground[:3, :3].T + to_ground[:3, 3] for _, c in clouds]
