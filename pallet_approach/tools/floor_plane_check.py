#!/usr/bin/env python3
"""Fit the floor plane in camera and LiDAR bags and report it in `base_footprint`.

    python3 floor_plane_check.py ~/bags/floor_cam_pose1 ~/bags/floor_lidar_pose1 ...

Answers: is the floor level (extrinsic and base_footprint right)? Do camera
and LiDAR agree? Is a common tilt robot-fixed (mounting) or world-fixed
(sloped floor)? The last needs bags at different headings, taken from /odom.
Reads depth images and point clouds with `rosbags`; no ROS.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np

try:
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
except ImportError:
    sys.exit('pip install rosbags numpy pyyaml  (in a venv)')

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _bagutil import GROUND, Tree, fill_static, quat_matrix  # noqa: E402
from pallet_approach.calibration import correct_transform, correction_matrix  # noqa: E402
DTYPES = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
          5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}


# ---------------------------------------------------------------- bag reading

def read_bag(path: pathlib.Path, typestore):
    """Return (tree, points_by_frame, odom_yaw)."""
    tree = Tree()
    clouds: dict[str, list[np.ndarray]] = {}
    yaws: list[float] = []
    intrinsics = {}

    with AnyReader([path], default_typestore=typestore) as reader:
        conns = list(reader.connections)

        # tf first: every transform we see, static or not. The robot is parked,
        # so taking the last value of each edge is enough.
        for conn, _, raw in reader.messages(
                connections=[c for c in conns if c.topic in ('/tf', '/tf_static')]):
            msg = reader.deserialize(raw, conn.msgtype)
            for tr in msg.transforms:
                tree.add(tr.header.frame_id, tr.child_frame_id,
                         quat_matrix(tr.transform.translation, tr.transform.rotation))

        for conn, _, raw in reader.messages(
                connections=[c for c in conns if c.topic.endswith('/odom')
                             or c.topic == '/odom']):
            msg = reader.deserialize(raw, conn.msgtype)
            q = msg.pose.pose.orientation
            yaws.append(math.atan2(2 * (q.w * q.z + q.x * q.y),
                                   1 - 2 * (q.y * q.y + q.z * q.z)))

        for conn in conns:
            if 'camera_info' in conn.topic:
                for _, _, raw in reader.messages(connections=[conn]):
                    msg = reader.deserialize(raw, conn.msgtype)
                    k = list(msg.k)
                    intrinsics = {'fx': k[0], 'fy': k[4], 'cx': k[2], 'cy': k[5]}
                    break
                break

        # depth images
        depth_conns = [c for c in conns
                       if c.msgtype.endswith('msg/Image') and 'depth' in c.topic]
        for conn in depth_conns:
            n = 0
            for _, _, raw in reader.messages(connections=[conn]):
                msg = reader.deserialize(raw, conn.msgtype)
                if msg.encoding != '32FC1' or not intrinsics:
                    break
                d = np.frombuffer(bytes(msg.data), dtype=np.float32).reshape(
                    msg.height, msg.width)
                clouds.setdefault(msg.header.frame_id, []).append(
                    backproject(d, intrinsics))
                n += 1
                if n >= 5:
                    break

        # point clouds
        for conn in [c for c in conns if c.msgtype.endswith('msg/PointCloud2')]:
            n = 0
            for _, _, raw in reader.messages(connections=[conn]):
                msg = reader.deserialize(raw, conn.msgtype)
                clouds.setdefault(msg.header.frame_id, []).append(unpack_cloud(msg))
                n += 1
                if n >= 5:
                    break

    yaw = float(np.median(yaws)) if yaws else float('nan')
    return tree, {f: np.concatenate(v) for f, v in clouds.items() if v}, yaw


def backproject(depth: np.ndarray, k: dict, stride: int = 3) -> np.ndarray:
    d = depth[::stride, ::stride]
    v, u = np.indices(d.shape)
    v = v * stride
    u = u * stride
    m = np.isfinite(d) & (d > 0.3) & (d < 8.0)
    z = d[m]
    return np.column_stack([(u[m] - k['cx']) * z / k['fx'],
                            (v[m] - k['cy']) * z / k['fy'], z])


def unpack_cloud(msg) -> np.ndarray:
    offs = {f.name: (f.offset, DTYPES[f.datatype]) for f in msg.fields}
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    buf = buf.reshape(-1, msg.point_step)
    out = []
    for name in ('x', 'y', 'z'):
        off, dt = offs[name]
        size = np.dtype(dt).itemsize
        out.append(buf[:, off:off + size].copy().view(dt).ravel())
    p = np.column_stack(out).astype(float)
    return p[np.isfinite(p).all(axis=1) & (np.abs(p).sum(axis=1) > 1e-6)]


# ------------------------------------------------------------------ plane fit

def fit_plane(points: np.ndarray, iterations: int = 6):
    """Robust least-squares plane, trimmed. Returns (normal, offset, rms, n)."""
    keep = np.ones(len(points), dtype=bool)
    normal = np.array([0.0, 0.0, 1.0])
    offset = 0.0
    for _ in range(iterations):
        p = points[keep]
        if len(p) < 50:
            break
        centroid = p.mean(axis=0)
        _, _, vt = np.linalg.svd(p - centroid, full_matrices=False)
        normal = vt[-1]
        if normal[2] < 0:
            normal = -normal
        offset = -float(normal @ centroid)
        residual = points @ normal + offset
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-9
        keep = np.abs(residual) < 2.5 * scale
    residual = points[keep] @ normal + offset
    return normal, offset, float(residual.std()), int(keep.sum())


def describe(normal, offset) -> dict:
    """Tilt of a plane whose normal should be +z, and its height below the origin."""
    n = normal / np.linalg.norm(normal)
    if n[2] < 0:
        n = -n
    return {
        'normal': n,
        # the plane passes through z = -offset on the sensor's vertical axis
        'height': -offset / n[2],
        'pitch_deg': math.degrees(math.atan2(-n[0], n[2])),
        'roll_deg': math.degrees(math.atan2(n[1], n[2])),
        'tilt_deg': math.degrees(math.acos(min(1.0, abs(n[2])))),
    }


# ----------------------------------------------------------------------- main

def analyse(path: pathlib.Path, typestore, rmin: float, rmax: float,
            correction=None, donor=None):
    tree, clouds, yaw = read_bag(path, typestore)
    if donor is not None:
        # A bag can arrive with part of /tf_static missing (the recorder joined
        # the latched topic before it had discovered every publisher). The
        # missing edges are fixed to the robot, so a sibling bag has them.
        borrowed = [c for c in donor.parent if c not in tree.parent]
        for child in borrowed:
            tree.add(donor.parent[child], child, donor.edge[child])
        if borrowed:
            print(f'  {path.name}: borrowed static transforms for {", ".join(borrowed)} '
                  f'-- an ASSUMPTION that nothing was moved')
    # Whatever is still missing comes from the exported reference (rigid mounting).
    filled = fill_static(tree)
    if filled:
        print(f'  {path.name}: filled in {len(filled)} static transform(s) from '
              f'config/static_tf.yaml ({", ".join(filled[:3])}...)')
    results = []
    for frame, pts in clouds.items():
        try:
            to_ground = tree.lookup(GROUND, frame)
        except KeyError as err:
            print(f'  {frame}: {err}')
            continue
        # The correction applies to the camera chain only -- the LiDAR measured
        # level, so applying a camera correction to it would introduce the very
        # error we are removing.
        if correction is not None and 'os_' not in frame:
            to_ground = correct_transform(to_ground, correction)
        g = pts @ to_ground[:3, :3].T + to_ground[:3, 3]
        radius = np.hypot(g[:, 0], g[:, 1])
        floor = g[(radius > rmin) & (radius < rmax) & (g[:, 2] > -0.25) & (g[:, 2] < 0.25)]
        if len(floor) < 500:
            continue
        normal, offset, rms, n = fit_plane(floor)
        d = describe(normal, offset)
        d.update(frame=frame, rms=rms, n=n, yaw=yaw,
                 kind='lidar' if 'os_' in frame else 'camera',
                 span=(float(radius.min()), float(radius[radius < rmax].max())))
        results.append(d)
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bags', nargs='+', type=pathlib.Path)
    ap.add_argument('--rmin', type=float, default=1.0, help='metres from the robot')
    ap.add_argument('--rmax', type=float, default=4.0)
    ap.add_argument('--typestore', default='ROS2_HUMBLE')
    ap.add_argument('--borrow-tf-from', type=pathlib.Path,
                    help='fill in static transforms a bag is missing from this bag')
    ap.add_argument('--correct', action='store_true',
                    help="apply the camera extrinsic correction from "
                         "config/params.yaml, to check the residual tilt")
    ap.add_argument('--correct-pitch', type=float,
                    help='override the measured floor pitch, in DEGREES')
    ap.add_argument('--correct-roll', type=float,
                    help='override the measured floor roll, in DEGREES')
    args = ap.parse_args()

    correction = None
    if args.correct or args.correct_pitch is not None or args.correct_roll is not None:
        pitch, roll = args.correct_pitch, args.correct_roll
        if pitch is None or roll is None:
            import yaml
            params = yaml.safe_load(
                (pathlib.Path(__file__).resolve().parent.parent
                 / 'config' / 'params.yaml').read_text())
            block = params['/**']['ros__parameters']['frames']['extrinsic_correction']
            pitch = math.degrees(block['measured_floor_pitch']) if pitch is None else pitch
            roll = math.degrees(block['measured_floor_roll']) if roll is None else roll
        correction = correction_matrix(math.radians(pitch), math.radians(roll))
        print(f'applying the camera extrinsic correction: levelling a measured '
              f'floor tilt of pitch {pitch:+.3f} deg, roll {roll:+.3f} deg')
        print('(the LiDAR measured level, so it is left alone)\n')

    typestore = get_typestore(getattr(Stores, args.typestore))
    print(f'floor plane in {GROUND}, using points {args.rmin}-{args.rmax} m out\n')
    print(f'{"bag":26} {"sensor":7} {"pitch":>8} {"roll":>8} {"tilt":>7} '
          f'{"height":>8} {"rms":>7} {"yaw":>8}  points')
    print('-' * 100)

    donor = read_bag(args.borrow_tf_from, typestore)[0] if args.borrow_tf_from else None
    rows = []
    for bag in args.bags:
        for r in analyse(bag, typestore, args.rmin, args.rmax, correction, donor):
            r['bag'] = bag.name
            rows.append(r)
            print(f'{bag.name:26} {r["kind"]:7} {r["pitch_deg"]:+8.3f} '
                  f'{r["roll_deg"]:+8.3f} {r["tilt_deg"]:7.3f} '
                  f'{r["height"]:+8.4f} {r["rms"]:7.4f} '
                  f'{math.degrees(r["yaw"]):+8.1f}  {r["n"]}')

    print('\npitch > 0 means the floor rises ahead of the robot.')
    print('height is where the plane crosses the robot\'s vertical axis; it should')
    print(f'be the floor height in {GROUND}.')

    decompose(rows)
    return 0


RMS_LIMIT = {'camera': 0.004, 'lidar': 0.020}
"""Metres. Floor fits above these are not floors; see `decompose`."""


def decompose(rows) -> None:
    """Split the measured tilt into a robot-fixed and a world-fixed part.

    The floor gradient measured in `base_footprint` is the sum of a mounting
    error, fixed to the robot, and the floor's own slope, fixed to the world and
    so rotating with the robot's heading. Two bags at different headings give
    two equations and separate them. The headings come from `/odom`, so the turn
    needs no measuring -- it only has to be large.

    With more than two headings this is a least-squares fit over ALL of them,
    and the residual is the self-check: if one constant mounting tilt and one
    constant slope explain every heading, the residual is small. A large one
    means the floor's slope changed between where the robot stood for the
    different bags (the robot moved while turning), or the odometry's heading is
    off. Exactly two bags is the old exact solve, unchanged.
    """
    for kind in ('camera', 'lidar'):
        group = [r for r in rows if r['kind'] == kind and not math.isnan(r['yaw'])]
        # A floor fit with a large residual is not a floor: at a heading where
        # the sensor faces a wall or furniture a metre away, the 1-4 m window
        # holds that instead (a camera fit of pitch -10 deg, height +0.38 m,
        # rms 70 mm came from exactly that). Good fits sit at 1-2 mm (camera) and
        # about 10 mm (LiDAR, whose beams are sparse); these limits are
        # heuristic, and what is dropped is printed.
        limit = RMS_LIMIT[kind]
        dropped = [r for r in group if r['rms'] > limit]
        for r in dropped:
            print(f'\n{kind}: dropping {r["bag"]} -- floor fit rms {r["rms"] * 1000:.1f} mm '
                  f'(limit {limit * 1000:.0f} mm): the window held something other than the floor')
        group = [r for r in group if r['rms'] <= limit]
        if len(group) < 2:
            continue
        yaws = [r['yaw'] for r in group]
        # The angle BETWEEN two headings, each pair wrapped to 0-180, then the widest.
        separation = max(min(abs(math.degrees(a - b)) % 360, 360 - abs(math.degrees(a - b)) % 360)
                         for a in yaws for b in yaws)
        if separation < 30:
            print(f'\n{kind}: headings only {separation:.0f} deg apart -- too close '
                  f'to separate a mounting error from a floor slope')
            continue

        def gradient(r):
            return np.array([math.tan(math.radians(r['pitch_deg'])),
                             -math.tan(math.radians(r['roll_deg']))])

        def rot(angle):
            c, s_ = math.cos(angle), math.sin(angle)
            return np.array([[c, s_], [-s_, c]])

        # gradient_i = mount + R(yaw_i) @ world, linear in (mount, world).
        design = np.vstack([np.hstack([np.eye(2), rot(r['yaw'])]) for r in group])
        target = np.concatenate([gradient(r) for r in group])
        solution, *_ = np.linalg.lstsq(design, target, rcond=None)
        mount, world = solution[:2], solution[2:]
        residual = design @ solution - target

        def size(g):
            return math.degrees(math.atan(float(np.linalg.norm(g))))

        print(f'\n{kind.upper()}  ({len(group)} headings, up to {separation:.0f} deg apart)')
        print(f'   mounting tilt, robot-fixed   {size(mount):6.3f} deg   '
              f'pitch {math.degrees(math.atan(mount[0])):+.3f}  '
              f'roll {-math.degrees(math.atan(mount[1])):+.3f}')
        print(f'   floor slope, world-fixed     {size(world):6.3f} deg')
        if len(group) > 2:
            # Judged against what is being separated: the residual has to be
            # small next to the floor slope, or the split is not trustworthy.
            worst = float(np.max(np.abs(np.degrees(np.arctan(residual)))))
            slope = size(world)
            verdict = ('consistent' if worst < 0.5 * slope else
                       'NOT consistent: the floor or the heading differs between bags')
            print(f'   fit residual                 worst {worst:.3f} deg against a {slope:.3f} deg '
                  f'slope ({verdict})')
        heights = [r['height'] for r in group]
        print(f'   floor height                 {np.mean(heights):+.4f} m   '
              f'spread {1000 * (max(heights) - min(heights)):.1f} mm')

    print('\nBoth sensors must see the same world-fixed slope -- there is one floor.')
    print('A mounting tilt on one sensor and not the other is that sensor\'s')
    print('extrinsic error, and it is the number to calibrate out.')


if __name__ == '__main__':
    sys.exit(main())
