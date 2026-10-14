#!/usr/bin/env python3
"""Replay a bag through the perception pipeline and draw every stage.

Tests tell you a function is correct against a case you thought of. This tells
you what the pipeline is doing to a real frame, which is a different question
and usually the one you need when something returns None on the robot.

    python3 tools/replay.py ~/bags/pallet_2m_0deg --truth-x 2.0
    python3 tools/replay.py ~/bags/pallet_2m_0deg --frame 3 --out /tmp/stages.png

No ROS: reads the bag with `rosbags`, uses the parameters from
`config/params.yaml` and the extrinsic correction from the same file, so it runs
the pipeline exactly as the robot would.

It calls each stage itself rather than going through `estimate_pallet`, which is
the one duplication worth having: it can then draw the intermediate results and
keep going after a stage returns nothing, instead of reporting a single None.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PACKAGE = HERE.parent
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(HERE))

import _bagutil as bag                                            # noqa: E402
from pallet_approach import perception as P                       # noqa: E402
from pallet_approach.calibration import correct_transform, correction_matrix  # noqa: E402
from pallet_approach.model import Plane                           # noqa: E402

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
except ImportError:                                               # pragma: no cover
    sys.exit('pip install matplotlib  (in a venv)')


def run(depth, intrinsics, to_ground, params):
    """Every stage, collected. Never raises on a stage returning nothing."""
    s = {'to_ground': to_ground, 'intrinsics': intrinsics}
    rng = np.random.default_rng(params['seed'])

    s['raw'] = P.depth_to_points(depth, intrinsics['fx'], intrinsics['fy'],
                                 intrinsics['cx'], intrinsics['cy'],
                                 params['stride'], params['min_range'],
                                 params['max_range'])
    s['ground'] = P.transform_points(s['raw'], to_ground)
    roi = params['roi']
    s['cropped'] = P.crop_to_roi(s['ground'], roi['x_min'], roi['x_max'],
                                 roi['y_min'], roi['y_max'],
                                 roi['z_min'], roi['z_max'])
    s['voxels'] = P.voxel_downsample(s['cropped'], params['voxel_size'])
    if len(s['voxels']) < 3:
        return s

    normal, offset, inliers = P.fit_plane_ransac(
        s['voxels'], params['floor_distance_threshold'],
        params['ransac_iterations'], rng=rng)
    if normal[2] < 0:
        normal, offset = -normal, -offset
    s['floor'] = Plane(normal, offset)
    s['floor_inliers'] = inliers
    s['floor_tilt'] = math.degrees(math.acos(min(1.0, abs(normal[2]))))
    if math.radians(s['floor_tilt']) > params['max_floor_tilt']:
        s['stopped'] = 'floor plane is too steep to be the floor'
        return s

    s['standing'] = P.remove_floor(s['voxels'], s['floor'], params['floor_margin'])
    s['labels'] = P.cluster_points(s['standing'], params['cluster_eps'],
                                   params['cluster_min_samples'])
    s['pallet'] = P.select_pallet_cluster(
        s['standing'], s['labels'], params['pallet_width'], params['pallet_height'],
        size_tolerance=params['size_tolerance'], floor=s['floor'],
        min_points=params['min_cluster_points'])
    if s['pallet'] is None:
        s['stopped'] = 'no cluster looked like a pallet'
        return s

    face = P.fit_front_face(s['pallet'], params['face_distance_threshold'],
                            params['max_tilt_from_vertical'], rng=rng)
    if face is None:
        s['stopped'] = 'no acceptable front face'
        return s
    s['face_normal'], s['face_centroid'], s['face_mask'] = face
    s['yaw'] = P.face_yaw(s['face_normal'])
    s['inliers'] = s['pallet'][s['face_mask']]

    # The slab, not the RANSAC inliers. Measured on the real pallet, its front
    # surface spans 26 mm from the deck boards to the lower blocks, so a 15 mm
    # inlier band samples mostly deck and starves the gap analysis. This
    # mirrors what estimate_pallet does.
    offset = (s['pallet'] - s['face_centroid']) @ s['face_normal']
    s['face'] = s['pallet'][np.abs(offset) < params['pocket_slab_half_depth']]
    s['slab_half'] = params['pocket_slab_half_depth']

    s['pockets'] = P.find_pockets(
        s['face'], s['face_normal'], s['face_centroid'], s['floor'],
        params['pocket_band_half_height'], params['pocket_min_gap_width'],
        params['pocket_max_gap_width'], params['pocket_center_height'],
        bin_width=params['voxel_size'],
        min_points_per_bin=params['pocket_min_points_per_bin'])
    return s


def pocket_corners(pocket, face_normal, floor, nominal=None) -> np.ndarray:
    """The four corners of a pocket opening, in `base_footprint`.

    The opening is a rectangle lying in the face plane: `width` across it and
    `height` up it. `along` is the horizontal direction in the face, which is
    perpendicular to both the floor normal and the face normal -- exactly what
    a cross product gives.

    Pass `nominal` as (width, height) to draw the expected opening instead of
    the measured one, which is how you see a measurement error rather than
    reading it.
    """
    up = np.asarray(floor.normal, dtype=float)
    along = np.cross(up, face_normal)
    along /= np.linalg.norm(along)
    half_w, half_h = ((pocket.width, pocket.height) if nominal is None else nominal)
    half_w, half_h = half_w / 2, half_h / 2
    c = np.asarray(pocket.center, dtype=float)
    return np.array([c + half_w * along + half_h * up,
                     c - half_w * along + half_h * up,
                     c - half_w * along - half_h * up,
                     c + half_w * along - half_h * up])


def project(points_ground, ground_to_camera, intrinsics):
    """(N, 3) in base_footprint -> (N, 2) pixels, plus a visibility mask.

    The pinhole projection, `u = fx X / Z + cx`, which is the exact inverse of
    what `perception.depth_to_points` undoes. Points at or behind the image
    plane (Z <= 0) have no projection; they are masked, not clamped, because a
    negative Z silently mirrors a point onto the image and looks like a
    geometry bug rather than the division by zero that it is.

    `fork_overlay.project_points` is the live version of this, drawn on the
    camera stream. This one exists so the offline tool does not depend on it.
    """
    p = np.asarray(points_ground, dtype=float)
    cam = p @ ground_to_camera[:3, :3].T + ground_to_camera[:3, 3]
    z = cam[:, 2]
    visible = z > 1e-6
    pixels = np.full((len(p), 2), np.nan)
    pixels[visible, 0] = intrinsics['fx'] * cam[visible, 0] / z[visible] + intrinsics['cx']
    pixels[visible, 1] = intrinsics['fy'] * cam[visible, 1] / z[visible] + intrinsics['cy']
    return pixels, visible


def report(s, params, truth_x) -> None:
    print(f'{"stage":26} {"points":>9}  notes')
    print('-' * 72)
    print(f'{"depth_to_points":26} {len(s["raw"]):9d}')
    print(f'{"crop_to_roi":26} {len(s["cropped"]):9d}')
    print(f'{"voxel_downsample":26} {len(s["voxels"]):9d}  at {params["voxel_size"]} m')
    if 'floor' not in s:
        print('stopped: too few points to fit a floor')
        return
    print(f'{"fit_plane_ransac":26} {int(s["floor_inliers"].sum()):9d}  '
          f'tilt {s["floor_tilt"]:.2f} deg, height '
          f'{s["floor"].height_on_axis:+.4f} m')
    if 'standing' in s:
        print(f'{"remove_floor":26} {len(s["standing"]):9d}  '
              f'margin {params["floor_margin"]} m')
        sizes = sorted((int((s['labels'] == l).sum())
                        for l in set(s['labels'].tolist()) if l >= 0), reverse=True)
        print(f'{"cluster_points":26} {len(sizes):9d}  clusters, largest {sizes[:5]}')
    if s.get('pallet') is not None:
        print(f'{"select_pallet_cluster":26} {len(s["pallet"]):9d}')
    if 'face' in s:
        print(f'{"fit_front_face":26} {len(s["inliers"]):9d}  '
              f'face x {s["face_centroid"][0]:.4f} m, '
              f'yaw {math.degrees(s["yaw"]):+.2f} deg')
        print(f'{"  face slab +-%.0f mm" % (1000 * s["slab_half"]):26} '
              f'{len(s["face"]):9d}  what find_pockets works on')
        if truth_x is not None:
            print(f'{"":26} {"":9}  tape measure {truth_x:.3f} m -> '
                  f'error {1000 * (s["face_centroid"][0] - truth_x):+.0f} mm')
    if 'pockets' in s:
        print(f'{"find_pockets":26} {len(s["pockets"]):9d}  pockets')
        for i, q in enumerate(s['pockets']):
            h = f'{q.height:.4f}' if np.isfinite(q.height) else 'nan'
            print(f'{"":26} {"":9}  [{i}] width {q.width:.4f} height {h} '
                  f'centre y {q.center[1]:+.4f}')
        if len(s['pockets']) >= 2:
            d = float(np.linalg.norm(s['pockets'][0].center - s['pockets'][1].center))
            print(f'{"":26} {"":9}  spacing {d:.4f} m (EPAL 0.3725)')
    if 'stopped' in s:
        print(f'\nSTOPPED: {s["stopped"]}')


def draw(s, depth, image, params, out, truth_x):
    fig = plt.figure(figsize=(17, 16), facecolor='white')
    grid = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.24,
                            height_ratios=[1.0, 1.0, 0.85, 1.5])
    nice = dict(fontsize=10, loc='left')

    # 1 - the input
    ax = fig.add_subplot(grid[0, 0])
    if image is not None:
        ax.imshow(image)
    ax.set_title('1  rectified left image (the input)', **nice)
    ax.axis('off')

    # 2 - the depth map
    ax = fig.add_subplot(grid[0, 1])
    m = ax.imshow(depth, cmap='turbo', vmin=params['min_range'], vmax=params['max_range'])
    valid = 100 * float(np.isfinite(depth).mean())
    ax.set_title(f'2  depth, 32FC1  ({valid:.0f} % valid)', **nice)
    ax.axis('off')
    fig.colorbar(m, ax=ax, fraction=0.035, label='m')

    # 3 - cropped cloud from above
    ax = fig.add_subplot(grid[0, 2])
    p = s['cropped']
    ax.scatter(p[:, 1], p[:, 0], c=p[:, 2], s=0.4, cmap='viridis', linewidths=0)
    ax.set_title(f'3  crop_to_roi, from above  ({len(p)} pts)', **nice)
    ax.set_xlabel('y, left +  [m]')
    ax.set_ylabel('x, forward  [m]')
    ax.invert_xaxis()
    ax.set_aspect('equal')

    # 4 - height above the fitted floor
    ax = fig.add_subplot(grid[1, 0])
    if 'floor' in s:
        p = s['voxels']
        h = s['floor'].height_of(p)
        sc = ax.scatter(p[:, 1], p[:, 0], c=h, s=1.2, cmap='coolwarm',
                        vmin=-0.05, vmax=0.25, linewidths=0)
        fig.colorbar(sc, ax=ax, fraction=0.035, label='m above floor')
        ax.set_title(f'4  floor fitted: tilt {s["floor_tilt"]:.2f}°, '
                     f'z {s["floor"].height_on_axis:+.3f} m', **nice)
        ax.set_xlabel('y  [m]')
        ax.set_ylabel('x  [m]')
        ax.invert_xaxis()
        ax.set_aspect('equal')

    # 5 - clusters
    ax = fig.add_subplot(grid[1, 1])
    if 'labels' in s:
        p, lab = s['standing'], s['labels']
        ax.scatter(p[lab < 0][:, 1], p[lab < 0][:, 0], c='0.8', s=1, linewidths=0,
                   label='noise')
        for i, l in enumerate(sorted(set(lab.tolist()) - {-1})):
            q = p[lab == l]
            ax.scatter(q[:, 1], q[:, 0], s=2, linewidths=0,
                       color=plt.cm.tab10(i % 10), label=f'{l} ({len(q)})')
        ax.legend(fontsize=6, markerscale=3, loc='upper right')
        ax.set_title(f'5  remove_floor + cluster_points  '
                     f'({len(p)} pts)', **nice)
        ax.set_xlabel('y  [m]')
        ax.set_ylabel('x  [m]')
        ax.invert_xaxis()
        ax.set_aspect('equal')

    # 6 - the chosen pallet and its face
    ax = fig.add_subplot(grid[1, 2])
    if s.get('pallet') is not None:
        q = s['pallet']
        ax.scatter(q[:, 1], q[:, 0], s=3, c='0.6', linewidths=0, label='cluster')
        if 'face' in s:
            f = s['face']
            ax.scatter(f[:, 1], f[:, 0], s=6, c='tab:red', linewidths=0,
                       label=f'face slab ({len(f)})')
            i = s['inliers']
            ax.scatter(i[:, 1], i[:, 0], s=4, c='tab:orange', linewidths=0,
                       label=f'RANSAC inliers ({len(i)})')
            c, n = s['face_centroid'], s['face_normal']
            ax.arrow(c[1], c[0], -0.25 * n[1], -0.25 * n[0], color='tab:blue',
                     width=0.008, label='face normal')
        if truth_x is not None:
            ax.axhline(truth_x, color='k', ls='--', lw=1, label=f'tape {truth_x} m')
        ax.legend(fontsize=7, markerscale=2)
        ax.set_title('6  select_pallet_cluster + fit_front_face', **nice)
        ax.set_xlabel('y  [m]')
        ax.set_ylabel('x  [m]')
        ax.invert_xaxis()
        ax.set_aspect('equal')

    # 7 - the face, flattened: this is what find_pockets sees
    ax = fig.add_subplot(grid[2, :2])
    if 'face' in s:
        up = np.asarray(s['floor'].normal)
        along = np.cross(up, s['face_normal'])
        along /= np.linalg.norm(along)
        sc = (s['face'] - s['face_centroid']) @ along
        h = s['floor'].height_of(s['face'])
        band = params['pocket_band_half_height']
        centre = params['pocket_center_height']
        ax.axhspan(centre - band, centre + band, color='tab:orange', alpha=0.15,
                   label='pocket band')
        ax.scatter(sc, h, s=8, c='tab:red', linewidths=0)
        for q in s.get('pockets', ()):
            off = float((q.center - s['face_centroid']) @ along)
            ax.add_patch(Rectangle((off - q.width / 2, centre - 0.05), q.width, 0.10,
                                   fill=False, ec='tab:green', lw=2))
        ax.set_title('7  the face, unrolled — what find_pockets works on '
                     '(green = pockets found)', **nice)
        ax.set_xlabel('along the face  [m]')
        ax.set_ylabel('above the floor  [m]')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

    # 8 - the gap histogram
    ax = fig.add_subplot(grid[2, 2])
    if 'face' in s:
        in_band = np.abs(h - centre) <= band
        if in_band.any():
            lo = sc.min()
            nb = max(1, int(np.ceil((sc.max() - lo) / params['voxel_size'])))
            idx = np.clip(((sc[in_band] - lo) / params['voxel_size']).astype(int), 0, nb - 1)
            counts = np.bincount(idx, minlength=nb)
            ax.bar(lo + params['voxel_size'] * (np.arange(nb) + 0.5), counts,
                   width=params['voxel_size'] * 0.9, color='tab:red')
            ax.axhline(params['pocket_min_points_per_bin'] - 0.5, color='k', ls='--',
                       lw=1, label='wood threshold')
            ax.legend(fontsize=7)
        ax.set_title('8  occupancy at pocket height (gaps = pockets)', **nice)
        ax.set_xlabel('along the face  [m]')
        ax.set_ylabel('points per bin')
        ax.grid(alpha=0.25, axis='y')

    # 9 - the measured pockets, projected back onto the image they came from
    ax = fig.add_subplot(grid[3, :])
    if image is not None:
        ax.imshow(image)
    ax.axis('off')
    drawn = 0
    extent = []
    if 'pockets' in s and image is not None:
        to_camera = np.linalg.inv(s['to_ground'])
        nominal = (params['pocket_nominal_width'], params['pocket_nominal_height'])
        for i, q in enumerate(s['pockets']):
            px, ok = project(pocket_corners(q, s['face_normal'], s['floor']),
                             to_camera, s['intrinsics'])
            if not ok.all():
                continue
            ax.add_patch(Polygon(px, closed=True, facecolor='tab:green',
                                 edgecolor='lime', lw=2.5, alpha=0.45,
                                 label='measured' if i == 0 else None))
            nx, nok = project(pocket_corners(q, s['face_normal'], s['floor'], nominal),
                              to_camera, s['intrinsics'])
            if nok.all():
                ax.add_patch(Polygon(nx, closed=True, fill=False, edgecolor='white',
                                     lw=1.6, ls='--',
                                     label='expected' if i == 0 else None))
            ax.annotate(f'{q.width * 1000:.0f} x {q.height * 1000:.0f} mm',
                        xy=(px[:, 0].mean(), px[:, 1].max() + 18),
                        color='white', fontsize=10, ha='center',
                        bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.65))
            extent.append(px)
            drawn += 1
        if drawn:
            ax.legend(fontsize=9, loc='upper right', framealpha=0.85)
    # Crop to the pallet. At full frame the openings are a couple of hundred
    # pixels across and the fit cannot be judged by eye, which is the only
    # reason this panel exists.
    if extent and image is not None:
        box = np.concatenate(extent)
        cx, cy = box[:, 0].mean(), box[:, 1].mean()
        half = max(np.ptp(box[:, 0]), np.ptp(box[:, 1])) * 1.25 + 40
        aspect = 3.2
        ax.set_xlim(max(0, cx - half * aspect / 2),
                    min(image.shape[1], cx + half * aspect / 2))
        ax.set_ylim(min(image.shape[0], cy + half / 2 + 30), max(0, cy - half / 2))
    elif image is not None:
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
    ax.set_title(f'9  the {drawn} detected pocket(s) projected back onto the image '
                 f'\u2014 solid = measured, dashed = expected '
                 f'{1000 * params["pocket_nominal_width"]:.0f} x '
                 f'{1000 * params["pocket_nominal_height"]:.0f} mm', **nice)

    note = s.get('stopped', '')
    fig.suptitle('perception pipeline on a real bag' + (f'  —  STOPPED: {note}' if note else ''),
                 fontsize=13, y=0.975)
    fig.savefig(out, dpi=110, bbox_inches='tight')
    print(f'\nwrote {out}')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bag', type=pathlib.Path)
    ap.add_argument('--frame', type=int, default=0)
    ap.add_argument('--out', type=pathlib.Path, default=pathlib.Path('pipeline_stages.png'))
    ap.add_argument('--truth-x', type=float, help='tape-measured face distance, m')
    ap.add_argument('--no-correction', action='store_true',
                    help='skip the measured extrinsic correction, to see its effect')
    args = ap.parse_args()

    params = bag.load_params(PACKAGE)
    data = bag.read(args.bag, max_frames=args.frame + 1)
    if not data['depth']:
        raise SystemExit('no 32FC1 depth image in that bag')
    depth = data['depth'][min(args.frame, len(data['depth']) - 1)]
    image = data['image'][min(args.frame, len(data['image']) - 1)] if data['image'] else None

    to_ground = data['tree'].lookup(bag.GROUND, data['depth_frame'] or bag.CAMERA)
    correction = params['extrinsic_correction']
    if correction.get('enabled') and not args.no_correction:
        to_ground = correct_transform(to_ground, correction_matrix(
            correction['measured_floor_pitch'], correction['measured_floor_roll']))
        print('extrinsic correction: on')
    else:
        print('extrinsic correction: OFF')

    s = run(depth, data['intrinsics'], to_ground, params)
    report(s, params, args.truth_x)
    draw(s, depth, image, params, args.out, args.truth_x)
    return 0


if __name__ == '__main__':
    sys.exit(main())
