"""The pallet face as the LiDAR sees it, against the camera's two estimates.

    python3 tools/lidar_face.py ~/bags/pallet_s_pos2_cond1 [--clouds 40]
    python3 tools/lidar_face.py ~/bags/pallet_s_pos1_cond1 --borrow-tf-from ~/bags/pallet_s_pos2_cond1

Why: neither bag has tape truth, and the two camera estimates (the current
pipeline and `track_prototype.py`) disagree on range by about 20 mm. The
Ouster is an independent sensor, so what it says about where the face is can
arbitrate, within its own known error.

How: each cloud is levelled with the D74 mounting correction and put in
`base_footprint`. Returns in a window round the camera's face (lateral: the
face width; vertical: just above the floor margin to just above the deck; in
depth: +-0.35 m) are fitted with a line by total least squares, dropping
returns more than 2 cm off it, three times. A beam crossing the face gives a
row of points on that line, so a few beams are enough: one gets the face
distance and its yaw with no camera in the fit at all.

What it cannot do: the LiDAR's forward position is only known to about 14 mm
(D90 measured it reading that much too far on the pallet's front edge, by
tape, once), so a camera-vs-LiDAR gap inside that is not a finding.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import _bagutil as bag  # noqa: E402
import track_prototype as T  # noqa: E402
from pallet_approach import perception as P  # noqa: E402
from pallet_approach.calibration import (  # noqa: E402
    correct_lidar_transform, correct_transform, correction_matrix)
from pallet_approach.pointcloud import unpack_cloud  # noqa: E402
from rosbags.highlevel import AnyReader  # noqa: E402


def read_clouds(path: pathlib.Path, count: int):
    """The first `count` LiDAR clouds as (frame_id, (N, 3) array)."""
    out = []
    with AnyReader([path], default_typestore=bag.typestore()) as reader:
        conns = [c for c in reader.connections if c.msgtype.endswith('msg/PointCloud2')]
        for conn, _, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            out.append((msg.header.frame_id, unpack_cloud(msg)))
            if len(out) >= count:
                break
    return out


def fit_face_line(points_xy: np.ndarray, rounds: int = 3, tolerance: float = 0.02):
    """Total-least-squares line through the face returns, with trimming.

    Returns (point_on_line, unit_direction, kept_count, rms) or None.
    """
    pts = points_xy
    for _ in range(rounds):
        if len(pts) < 8:
            return None
        centre = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - centre, full_matrices=False)
        direction = vt[0]
        normal = np.array([-direction[1], direction[0]])
        residual = (pts - centre) @ normal
        keep = np.abs(residual) < tolerance
        if keep.all():
            break
        pts = pts[keep]
    if len(pts) < 8:
        return None
    centre = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centre, full_matrices=False)
    direction = vt[0]
    normal = np.array([-direction[1], direction[0]])
    return centre, direction, len(pts), float(np.sqrt(np.mean(((pts - centre) @ normal) ** 2)))


def lidar_face(cloud_ground: np.ndarray, centre_xy, face_x_axis: np.ndarray,
               floor, params: dict, h_lo: float = 0.04, h_hi: float | None = None):
    """The face line from one cloud, or None. `centre_xy` and `face_x_axis`
    (the unit direction ALONG the face) only place the search window."""
    half = params['pallet_width'] / 2.0
    rel = cloud_ground[:, :2] - centre_xy
    along = rel @ face_x_axis
    normal = np.array([-face_x_axis[1], face_x_axis[0]])
    across = rel @ normal
    height = floor.height_of(cloud_ground)
    window = ((np.abs(along) <= half) & (np.abs(across) <= 0.35)
              & (height >= h_lo)
              & (height <= (params['pallet_height'] + 0.03 if h_hi is None else h_hi)))
    if window.sum() < 8:
        return None
    pts = cloud_ground[window][:, :2]
    # The face is the NEAREST surface in the window (cargo or the back of the
    # pallet may also return). Start from the nearest 25 % by depth across.
    depth = (pts - centre_xy) @ normal
    near = depth >= np.percentile(depth, 60)       # normal points at the robot
    return fit_face_line(pts[near])


def x_at(centre, direction, y: float) -> float:
    """x of the line at lateral position y."""
    return float(centre[0] + (y - centre[1]) * direction[0] / direction[1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('bag', type=pathlib.Path)
    ap.add_argument('--clouds', type=int, default=40)
    ap.add_argument('--frames', type=int, default=120)
    ap.add_argument('--borrow-tf-from', type=pathlib.Path)
    ap.add_argument('--sim', action='store_true',
                    help='a Gazebo bag: apply config/sim_overlay.yaml (no tilt corrections)')
    args = ap.parse_args()

    overlay = 'sim_overlay.yaml' if args.sim else None
    params = bag.load_params(HERE.parent, overlay=overlay)
    import yaml
    raw = yaml.safe_load((HERE.parent / 'config' / 'params.yaml').read_text())
    if overlay:
        bag._merge(raw, yaml.safe_load((HERE.parent / 'config' / overlay).read_text()))
    shared = raw['/**']['ros__parameters']
    lc = shared['frames']['lidar_correction']
    floor_z = shared['floor_z']

    data = bag.read(args.bag, max_frames=args.frames)
    if args.borrow_tf_from:
        donor = bag.read(args.borrow_tf_from, max_frames=1)['tree']
        for child, parent in donor.parent.items():
            if child not in data['tree'].parent:
                data['tree'].add(parent, child, donor.edge[child])
    cam_to_ground = data['tree'].lookup(bag.GROUND, data['depth_frame'] or bag.CAMERA)
    c = params['extrinsic_correction']
    if c.get('enabled'):
        cam_to_ground = correct_transform(cam_to_ground, correction_matrix(
            c['measured_floor_pitch'], c['measured_floor_roll']))
    intr = data['intrinsics']
    geo = T.Geometry.from_params(params)

    # --- camera: baseline and prototype, medians over the bag's frames
    frames = data['depth'][::max(len(data['depth']) // 20, 1)]
    est = [P.estimate_pallet(f, intr, cam_to_ground, params) for f in frames]
    usable = [e for e in est if e is not None]
    if not usable:
        print('the current pipeline finds no pallet in this bag')
        return 1
    ref = usable[len(usable) // 2]
    prior = T.prior_from(ref)
    centre = prior.centre
    axis = prior.tangent
    floor = ref.floor
    base_x = [np.mean([p.center for p in e.pockets], axis=0)[0] for e in usable if len(e.pockets) == 2]
    proto_x, proto_yaw = [], []
    p = prior
    for f in frames:
        r = T.track(f, intr, cam_to_ground, p, geo)
        if r.ok:
            proto_x.append(r.mid[0]); proto_yaw.append(math.degrees(r.yaw))
            p = T.prior_from_result(r, p)
    y_mid = float(centre[1])

    # --- LiDAR
    xs, yaws, counts, rms = [], [], [], []
    for frame_id, cloud in read_clouds(args.bag, args.clouds):
        to_ground = data['tree'].lookup(bag.GROUND, frame_id)
        if lc.get('enabled', True):
            to_ground = correct_lidar_transform(
                to_ground, lc['measured_floor_pitch'], lc['measured_floor_roll'],
                lc['measured_floor_height'], floor_z)
        g = cloud @ to_ground[:3, :3].T + to_ground[:3, 3]
        fit = lidar_face(g, centre, axis, floor, params)
        if fit is None:
            continue
        point, direction, n, r = fit
        xs.append(x_at(point, direction, y_mid))
        d = direction if direction[1] > 0 else -direction
        yaws.append(math.degrees(math.atan2(-d[0], d[1])) if False else math.degrees(
            math.atan2(d[0], d[1])))
        counts.append(n); rms.append(r)

    print(f'{args.bag.name}: face at about x = {centre[0]:.2f} m, lateral {y_mid:+.2f} m')
    print(f'  LiDAR   : {len(xs)} of {args.clouds} clouds gave a face line; '
          f'{int(np.median(counts)) if counts else 0} returns on it (median), '
          f'rms {1000 * np.median(rms) if rms else float("nan"):.1f} mm')
    if xs:
        print(f'            x at the pocket midpoint: median {np.median(xs):.3f} m, '
              f'spread (std) {1000 * np.std(xs):.1f} mm')
    if base_x:
        print(f'  camera, current pipeline : {np.median(base_x):.3f} m  (std {1000 * np.std(base_x):.1f} mm, '
              f'{len(base_x)} frames)')
    if proto_x:
        print(f'  camera, prototype tracker: {np.median(proto_x):.3f} m  (std {1000 * np.std(proto_x):.1f} mm, '
              f'{len(proto_x)} frames)')
    if xs and base_x:
        print(f'  LiDAR minus current pipeline : {1000 * (np.median(xs) - np.median(base_x)):+.1f} mm')
    if xs and proto_x:
        print(f'  LiDAR minus prototype tracker: {1000 * (np.median(xs) - np.median(proto_x)):+.1f} mm')
    truth_file = args.bag / 'truth.yaml'
    if truth_file.exists() and xs:
        truth = yaml.safe_load(truth_file.read_text())
        # The deck face is at face_x; blocks standing proud are nearer by their protrusion.
        print(f'  TRUTH (sim): deck face x = {float(truth["face_x"]):.3f} m, block face x = '
              f'{float(truth["face_x"]) - float(truth.get("block_protrusion", 0)):.3f} m;  LiDAR minus deck face: '
              f'{1000 * (np.median(xs) - float(truth["face_x"])):+.1f} mm')
    print("  (on the real robot the LiDAR's own forward error is about 14 mm, D90: differences inside that are not a finding)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
