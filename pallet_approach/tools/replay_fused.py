#!/usr/bin/env python3
"""Draw every stage of the fused perception (D102) on one recorded bag.

    python3 tools/replay_fused.py ~/bags/pallet_s_129 --out docs/images/fused_pallet_s_129.png

The companion of replay.py, which draws the camera-only pipeline. Calls each
stage itself, with the parameters and the D101 LiDAR pose from params.yaml, so
the figure shows exactly what the robot computes: the LiDAR finds the face, the
camera measures the pockets along it. The robot stands still in these bags, so
the filter adds nothing to a single frame and is not drawn. No ROS.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import _bagutil as bag  # noqa: E402
from pallet_approach import camera_pockets as CP  # noqa: E402
from pallet_approach import lidar_pallet as LP  # noqa: E402
from pallet_approach.fused_pipeline import camera_correction  # noqa: E402


def truth(name: str, shared: dict):
    rows = yaml.safe_load((HERE.parent.parent / 'docs' / 'bag_truth.yaml').read_text()) or {}
    row = rows.get(name)
    if not row or not row.get('measured'):
        return None
    w = shared['pallet']['sides'][row.get('side', 'short')]['face_width']
    angle = math.asin(max(-1.0, min(1.0, (row['dL'] - row['dR']) / w)))
    return np.array([(row['dL'] + row['dR']) / 2 + bag.BUMPER_X, row.get('lat', 0.0)]), angle


def project(points, ground_from_camera, k):
    r, t = ground_from_camera[:3, :3], ground_from_camera[:3, 3]
    cam = (np.asarray(points, float) - t) @ r
    z = cam[:, 2]
    return np.stack([k['fx'] * cam[:, 0] / z + k['cx'], k['fy'] * cam[:, 1] / z + k['cy']], 1), z > 0.1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('bag', type=pathlib.Path)
    ap.add_argument('--sweeps', type=int, default=5, help='LiDAR sweeps pooled, as acquisition does standing still')
    ap.add_argument('--frame', type=int, default=0)
    ap.add_argument('--out', type=pathlib.Path, default=pathlib.Path('fused_stages.png'))
    args = ap.parse_args()

    shared = bag.shared_params()
    data = bag.read(args.bag, max_frames=args.frame + 1)
    depth, image, k = data['depth'][args.frame], data['image'][args.frame], data['intrinsics']
    cam = bag.camera_to_ground(data, shared['frames']['extrinsic_correction'])
    cloud = np.concatenate(bag.lidar_sweeps(args.bag, data, shared, args.sweeps))

    # ---- LiDAR stages, as find_pallet runs them
    p = dict(shared['lidar_acquisition'], floor_z=shared['floor_z'])
    floor = LP.fit_floor(cloud, p, np.random.default_rng(0))
    h = floor.height_of(cloud)
    lo, hi = p['band']
    sel = ((h >= lo) & (h <= hi + p['surface_extra_height']) & (cloud[:, 0] > p['search_x'][0])
           & (cloud[:, 0] < p['search_x'][1]) & (np.abs(cloud[:, 1]) < p['search_y']))
    near = LP.first_surface(cloud[sel], h[sel], p)
    band = near[floor.height_of(near) <= hi]
    face, info = LP.find_pallet(cloud, shared)
    if face is None:
        print(f'no pallet in the LiDAR: {info}')
        return 1
    side = shared['pallet']['sides'][face.side]
    tangent = np.array([math.sin(face.yaw), -math.cos(face.yaw)])

    # ---- camera stage, along the LiDAR's face
    cpp = shared['camera_pockets']
    beta, offset_c = camera_correction(shared)
    res = CP.measure(depth, k, cam, face.centre, face.yaw, floor, side, cpp, beta, offset_c)
    half = side['face_width'] / 2 + cpp['margin']
    s = np.arange(-half, half + 1e-9, cpp['step'])
    rows = side['band_center'] + np.linspace(-1, 1, cpp['rows']) * side['band_half_height']
    xy = face.centre[None, :] + s[:, None] * tangent[None, :]
    zf = -(floor.normal[0] * xy[:, 0] + floor.normal[1] * xy[:, 1] + floor.offset) / floor.normal[2]
    pts = np.concatenate([np.column_stack([xy, zf + r]) for r in rows])
    uv, ok = project(pts, cam, k)
    hgt, wid = depth.shape
    inside = ok & (uv[:, 0] >= 0) & (uv[:, 0] < wid) & (uv[:, 1] >= 0) & (uv[:, 1] < hgt)
    expected = ((pts - cam[:3, 3]) @ cam[:3, :3])[:, 2]
    measured = np.full(len(pts), np.nan)
    ui, vi = uv[inside].astype(int).T
    measured[inside] = CP.corrected_depth(depth[vi, ui].astype(float), beta, offset_c)
    diff = measured - expected
    label = np.where(np.abs(diff) <= cpp['wood_tolerance'], 1.0, np.where(diff >= cpp['hole_depth'], -1.0, 0.0))
    label[~np.isfinite(diff)] = np.nan
    lab = label.reshape(len(rows), len(s))
    known = np.isfinite(lab).sum(0)
    profile = np.where(known > 0, np.nansum(lab, 0) / np.maximum(known, 1), 0.0)
    offset = res.offset if res.offset is not None else 0.0
    pocket_centre = face.centre + offset * tangent

    t = truth(args.bag.name, shared)

    # ---- figure
    fig = plt.figure(figsize=(17, 11))
    g = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.75], hspace=0.38, wspace=0.22)
    title = f'{args.bag.name}: the fused perception, stage by stage ({args.sweeps} LiDAR sweeps pooled, depth frame {args.frame})'
    fig.suptitle(title, fontsize=14, y=0.995)

    ax = fig.add_subplot(g[0, 0])
    m = (np.abs(cloud[:, 1]) < 2.0) & (cloud[:, 0] > -0.5) & (cloud[:, 0] < 4.5)
    sc = ax.scatter(cloud[m, 1], cloud[m, 0], c=h[m], s=0.3, cmap='viridis', vmin=-0.05, vmax=0.6)
    ax.add_patch(plt.Rectangle((-p['search_y'], p['search_x'][0]), 2 * p['search_y'],
                               p['search_x'][1] - p['search_x'][0], fill=False, ls='--', ec='k', lw=0.8))
    fig.colorbar(sc, ax=ax, fraction=0.04, label='height above the LiDAR\'s floor fit (m)')
    ax.set_title(f'1. LiDAR, calibrated (D101)\n{len(cloud):,} returns; dashed: search area', fontsize=11)
    ax.set(xlabel='y, left (m)', ylabel='x, ahead (m)', xlim=(2, -2), ylim=(-0.5, 4.5), aspect='equal')

    ax = fig.add_subplot(g[0, 1])
    sl = cloud[sel]
    ax.scatter(sl[:, 1], sl[:, 0], s=0.8, c='0.75', label=f'pallet-height slice ({lo:.2f}-{hi:.3f} m): {len(sl):,}')
    ax.scatter(band[:, 1], band[:, 0], s=2, c='C3', label=f'nearest vertical surface per bearing: {len(band):,}')
    ax.set_title('2. Slice at pallet height; keep the nearest\nvertical surface per bearing', fontsize=11)
    ax.legend(loc='lower left', fontsize=8, markerscale=4)
    ax.set(xlabel='y, left (m)', ylabel='x, ahead (m)', xlim=(1.0, -1.0), aspect='equal',
           ylim=(face.centre[0] - 0.5, face.centre[0] + 1.3))

    ax = fig.add_subplot(g[0, 2])
    fz = floor.height_of(band)
    ax.scatter(band[:, 1], band[:, 0], s=3, c=fz, cmap='viridis', vmin=0, vmax=hi)
    ends = np.array([face.centre - face.width / 2 * tangent, face.centre + face.width / 2 * tangent])
    ax.plot(ends[:, 1], ends[:, 0], 'k-', lw=2, label=f'face line: width {face.width:.3f} m -> {face.side} side')
    for sgn in (-1, 1):
        c = face.centre + ((face.pocket_offset or 0) + sgn * side['pocket_spacing'] / 2) * tangent
        a = c - side['pocket_width'] / 2 * tangent
        b = c + side['pocket_width'] / 2 * tangent
        ax.plot([a[1], b[1]], [a[0], b[0]], color='C1', lw=5, alpha=0.7,
                label='pockets from the LiDAR\'s block/gap template' if sgn < 0 else None)
    if t is not None:
        tc, ta = t
        tt = np.array([math.sin(ta), math.cos(ta)])        # along the face (outward normal yaw = pi - angle)
        te = np.array([tc - side['face_width'] / 2 * tt, tc + side['face_width'] / 2 * tt])
        ax.plot(te[:, 1], te[:, 0], 'g--', lw=1.2, label='tape-measured face')
    ax.set_title(f'3. Face line (RANSAC): {face.points} returns, rms {1000 * face.rms:.1f} mm\n'
                 f'{face.distance:.3f} m away, yaw {math.degrees(face.yaw - math.copysign(math.pi, face.yaw)):+.1f} deg',
                 fontsize=11)
    ax.legend(loc='lower left', fontsize=8)
    ax.set(xlabel='y, left (m)', ylabel='x, ahead (m)', xlim=(0.6, -0.6), aspect='equal',
           ylim=(face.centre[0] - 0.35, face.centre[0] + 0.45))

    ax = fig.add_subplot(g[1, 0:2])
    ax.imshow(image[:, :, ::-1])
    colors = {1.0: 'lime', -1.0: 'red', 0.0: 'yellow'}
    for val, col in colors.items():
        mk = inside & (label == val)
        ax.scatter(uv[mk, 0], uv[mk, 1], s=4, c=col,
                   label={1.0: 'wood (depth on the face)', -1.0: 'hole (8+ cm deeper)', 0.0: 'neither'}[val])
    ax.set_title(f'4. Camera, only where the pockets must be: the LiDAR\'s face line at {len(rows)} heights '
                 f'across the pocket band, every {1000 * cpp["step"]:.0f} mm\n'
                 f'depth corrected by the range study (D109: beta {beta}, offset {1000 * offset_c:.1f} mm)', fontsize=11)
    ax.legend(loc='upper right', fontsize=9, markerscale=3)
    pc = uv[inside]
    ax.set(xlim=(pc[:, 0].min() - 120, pc[:, 0].max() + 120), ylim=(pc[:, 1].max() + 140, pc[:, 1].min() - 200))
    ax.axis('off')

    ax = fig.add_subplot(g[1, 2])
    ax.imshow(image[:, :, ::-1])
    for sgn in (-1, 1):
        c = pocket_centre + sgn * side['pocket_spacing'] / 2 * tangent
        zc = -(floor.normal[0] * c[0] + floor.normal[1] * c[1] + floor.offset) / floor.normal[2]
        corners = []
        for dx, dz in ((-1, 0), (1, 0), (1, 1), (-1, 1), (-1, 0)):
            q = c + dx * side['pocket_width'] / 2 * tangent
            corners.append([q[0], q[1], zc + side['opening_bottom'] + dz * (side['opening_top'] - side['opening_bottom'])])
        cuv, _ = project(corners, cam, k)
        ax.plot(cuv[:, 0], cuv[:, 1], 'lime', lw=2)
    fc = []
    for dx, dz in ((-1, 0), (1, 0), (1, 1), (-1, 1), (-1, 0)):
        q = face.centre + dx * face.width / 2 * tangent
        zq = -(floor.normal[0] * q[0] + floor.normal[1] * q[1] + floor.offset) / floor.normal[2]
        fc.append([q[0], q[1], zq + dz * shared['pallet']['pallet_height']])
    fuv, _ = project(fc, cam, k)
    ax.plot(fuv[:, 0], fuv[:, 1], 'w-', lw=1)
    ax.set(xlim=(fuv[:, 0].min() - 80, fuv[:, 0].max() + 80), ylim=(fuv[:, 1].max() + 60, fuv[:, 1].min() - 80))
    ax.axis('off')
    ax.set_title('6. Result: face (LiDAR) and pockets\n(camera offset along it) on the image', fontsize=11)

    ax = fig.add_subplot(g[2, :])
    ax.step(s, profile, where='mid', color='k', lw=1.2, label='profile: mean of wood (+1) / hole (-1) over the heights')
    ax.plot(s, CP._template(s - offset, side, cpp['beyond']) * 0.9, color='C1', lw=2, alpha=0.8,
            label=f'best block/pocket template: shift {1000 * offset:+.0f} mm, score {res.score:.2f}, '
                  f'contrast {res.contrast:.2f}')
    ax.axhline(cpp['open_below'], color='C3', ls=':', lw=1, label='a pocket reads open below this')
    ax.set(xlabel='position along the face (m, + = robot\'s left; drawn as the camera sees it)',
           ylabel='wood / hole', ylim=(-1.15, 1.6))
    ax.invert_xaxis()
    ax.legend(loc='upper center', fontsize=9, ncol=3, frameon=False)
    lines = [f'5. Template match along the face: both pockets open {res.both_open}',
             f'camera - LiDAR range {1000 * res.range_residual:+.1f} mm, yaw {math.degrees(res.yaw_residual):+.2f} deg, '
             f'{res.ms:.1f} ms on the laptop']
    if t is not None:
        pc_err = pocket_centre - t[0]                      # the short face's layout is symmetric
        lines.append(f'pocket pair centre vs tape: dx {1000 * pc_err[0]:+.0f} mm, dy {1000 * pc_err[1]:+.0f} mm '
                     f'(tape good to ~10 mm)')
    ax.set_title('\n'.join(lines), fontsize=11)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=100, bbox_inches='tight')
    print(f'wrote {args.out}')
    print(f'LiDAR: {face.side}, {face.distance:.3f} m, width {face.width:.3f}, rms {1000 * face.rms:.1f} mm, '
          f'pocket offset {face.pocket_offset}; camera: offset {res.offset}, score {res.score:.2f}, '
          f'open {res.both_open}, range {1000 * res.range_residual:+.1f} mm')
    if t is not None:
        print(f'tape: centre {t[0]}, angle {math.degrees(t[1]):+.1f} deg; fused pocket centre {pocket_centre}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
