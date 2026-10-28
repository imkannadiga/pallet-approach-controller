"""The pallet face as the LiDAR sees it, against the camera pipeline's estimate.

    python3 tools/lidar_face.py ~/bags/pallet_s_205 [--clouds 40] [--lidar-pose d74]

Each cloud goes into `base_footprint` with the LiDAR pose in force (D101, or as
chosen); returns in a window round the camera's face, between the floor margin
and just above the deck, are fitted with a trimmed total-least-squares line.
Reports the face distance and yaw from both sensors, and for sim bags the truth.
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
from pallet_approach import perception as P  # noqa: E402


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
    # The face is the nearest surface in the window: fit the nearest 40 %.
    depth = (pts - centre_xy) @ normal
    near = depth >= np.percentile(depth, 60)       # normal points at the robot
    return fit_face_line(pts[near])


def x_at(centre, direction, y: float) -> float:
    """x of the line at lateral position y."""
    return float(centre[0] + (y - centre[1]) * direction[0] / direction[1])


def _tape_deg(yaw: float) -> float:
    """A face yaw (square-on ~ pi) as the tape's angle, degrees, + when dL > dR."""
    return math.degrees(math.atan2(math.sin(math.pi - yaw), math.cos(math.pi - yaw)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('bag', type=pathlib.Path)
    ap.add_argument('--clouds', type=int, default=40)
    ap.add_argument('--frames', type=int, default=120)
    ap.add_argument('--borrow-tf-from', type=pathlib.Path)
    ap.add_argument('--lidar-pose', choices=('auto', 'calibrated', 'd74', 'urdf'), default='auto',
                    help="the LiDAR's pose: what params.yaml has in force (default), the "
                         'camera-LiDAR calibration (D101), the D74 floor correction, or the URDF as is')
    ap.add_argument('--sim', action='store_true',
                    help='a Gazebo bag: apply config/sim_overlay.yaml (no tilt corrections)')
    args = ap.parse_args()

    overlay = 'sim_overlay.yaml' if args.sim else None
    params = bag.load_params(HERE.parent, overlay=overlay)
    shared = bag.shared_params(overlay)

    data = bag.read(args.bag, max_frames=args.frames)
    if args.borrow_tf_from:
        donor = bag.read(args.borrow_tf_from, max_frames=1)['tree']
        for child, parent in donor.parent.items():
            if child not in data['tree'].parent:
                data['tree'].add(parent, child, donor.edge[child])
    cam_to_ground = bag.camera_to_ground(data, params['extrinsic_correction'])
    intr = data['intrinsics']

    # --- camera: the pipeline's estimates, medians over the bag's frames
    frames = data['depth'][::max(len(data['depth']) // 20, 1)]
    est = [P.estimate_pallet(f, intr, cam_to_ground, params) for f in frames]
    usable = [e for e in est if e is not None]
    if not usable:
        print('the camera pipeline finds no pallet in this bag')
        return 1
    ref = usable[len(usable) // 2]
    pairs = [e for e in usable if len(e.pockets) == 2]
    centre = (np.mean([p.center[:2] for p in ref.pockets], axis=0) if len(ref.pockets) == 2
              else ref.position[:2])
    axis = np.array([ref.normal[1], -ref.normal[0]])     # along the face
    floor = ref.floor
    base_x = [np.mean([p.center for p in e.pockets], axis=0)[0] for e in pairs]
    base_yaw = [_tape_deg(e.yaw) for e in pairs]
    y_mid = float(centre[1])

    # --- LiDAR
    xs, yaws, counts, rms = [], [], [], []
    for frame_id, cloud in bag.read_clouds(args.bag, args.clouds):
        to_ground = bag.lidar_to_ground(data['tree'], frame_id, data['depth_frame'] or bag.CAMERA,
                                        shared, args.lidar_pose)
        g = cloud @ to_ground[:3, :3].T + to_ground[:3, 3]
        fit = lidar_face(g, centre, axis, floor, params)
        if fit is None:
            continue
        point, direction, n, r = fit
        xs.append(x_at(point, direction, y_mid))
        d = direction if direction[1] > 0 else -direction
        yaws.append(math.degrees(math.atan2(d[0], d[1])))        # tape convention: + when dL > dR
        counts.append(n)
        rms.append(r)

    print(f'{args.bag.name}: face at about x = {centre[0]:.2f} m, lateral {y_mid:+.2f} m')
    print(f'  LiDAR   : {len(xs)} of {args.clouds} clouds gave a face line; '
          f'{int(np.median(counts)) if counts else 0} returns on it (median), '
          f'rms {1000 * np.median(rms) if rms else float("nan"):.1f} mm')
    if xs:
        print(f'            x at the pocket midpoint: median {np.median(xs):.3f} m, '
              f'spread (std) {1000 * np.std(xs):.1f} mm')
    if base_x:
        print(f'  camera pipeline: {np.median(base_x):.3f} m  (std {1000 * np.std(base_x):.1f} mm, '
              f'{len(base_x)} frames)')
    for label, values in (('LiDAR line', yaws), ('camera', base_yaw)):
        if values:
            print(f'  yaw, {label:10s}: {np.median(values):+.2f} deg  (std {np.std(values):.2f})')
    if xs and base_x:
        print(f'  LiDAR minus camera: {1000 * (np.median(xs) - np.median(base_x)):+.1f} mm')
    truth_file = args.bag / 'truth.yaml'
    if truth_file.exists() and xs:
        import yaml
        truth = yaml.safe_load(truth_file.read_text())
        # The deck face is at face_x; blocks standing proud are nearer by their protrusion.
        print(f'  TRUTH (sim): deck face x = {float(truth["face_x"]):.3f} m, block face x = '
              f'{float(truth["face_x"]) - float(truth.get("block_protrusion", 0)):.3f} m;  LiDAR minus deck face: '
              f'{1000 * (np.median(xs) - float(truth["face_x"])):+.1f} mm')
    return 0


if __name__ == '__main__':
    sys.exit(main())
