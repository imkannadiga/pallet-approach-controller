#!/usr/bin/env python3
"""Measure the ZED's depth bias on flat-board bags, as one disparity offset (D103).

    python3 tools/fit_depth_bias.py ~/bags/deck_80 ~/bags/deck_105 ... [--reference tape|lidar]

A stereo camera computes depth from disparity, z = f B / d. A small constant
error in d -- a fraction of a pixel, from the rectification or the matcher --
gives 1/z_measured = 1/z_true + beta, so the depth error grows with the
SQUARE of the distance: z_measured ~ z_true - beta z_true^2. That is what the
board bags show (-10 mm at 1.5 m, -19 at 2.5, -47 at 3.5), and it is
corrected by z_true = z_measured / (1 - beta z_measured), one number.

For each bag: the board centre's distance along the camera's optical axis
from the ZED's depth image (median over the board patch), and the same from
the reference -- the tape (dL, dR in docs/bag_truth.yaml) or the calibrated
LiDAR (D101). beta is fitted by least squares on 1/z.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import yaml

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import _bagutil as bag  # noqa: E402


def board_patch(points: np.ndarray, x0: float) -> np.ndarray:
    """Returns on the board: near the expected x, straight ahead, off the floor."""
    return points[(np.abs(points[:, 0] - x0) < 0.25) & (np.abs(points[:, 1]) < 0.2)
                  & (points[:, 2] > 0.15) & (points[:, 2] < 0.6)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('bags', nargs='+', type=pathlib.Path)
    ap.add_argument('--reference', choices=('tape', 'lidar'), default='lidar')
    args = ap.parse_args()
    shared = bag.shared_params()
    ec = shared['frames']['extrinsic_correction']
    truths = yaml.safe_load((HERE.parent.parent / 'docs' / 'bag_truth.yaml').read_text())
    z_cam, z_ref, names = [], [], []
    for path in args.bags:
        row = truths[path.name]
        tape_x = (row['dL'] + row['dR']) / 2 + bag.BUMPER_X
        data = bag.read(path, max_frames=10)
        cam = bag.camera_to_ground(data, ec)
        k = data['intrinsics']
        zs = []
        for depth in data['depth']:
            v, u = np.mgrid[0:depth.shape[0]:4, 0:depth.shape[1]:4]
            z = depth[::4, ::4].astype(float)
            ok = np.isfinite(z) & (z > 0.3)
            pc = np.stack([(u[ok] - k['cx']) * z[ok] / k['fx'], (v[ok] - k['cy']) * z[ok] / k['fy'], z[ok]], 1)
            ground = pc @ cam[:3, :3].T + cam[:3, 3]
            on = board_patch(ground, tape_x)
            if len(on) > 20:
                # back into the camera frame: the optical-axis depth of the board patch
                zs.append(np.median(((on - cam[:3, 3]) @ cam[:3, :3])[:, 2]))
        if args.reference == 'tape':
            ref_x = tape_x
        else:
            lid = np.concatenate(bag.lidar_sweeps(path, data, shared, 10))
            ref_x = float(np.median(board_patch(lid, tape_x)[:, 0]))
        # the reference board point straight ahead, as an optical-axis depth
        ref_point = np.array([ref_x, 0.0, cam[2, 3]])
        z_ref.append(float(((ref_point - cam[:3, 3]) @ cam[:3, :3])[2]))
        z_cam.append(float(np.median(zs)))
        names.append(path.name)
    z_cam, z_ref = np.array(z_cam), np.array(z_ref)
    beta = float(np.mean(1 / z_cam - 1 / z_ref))          # least squares on 1/z with one offset
    corrected = z_cam / (1 - beta * z_cam)
    print(f'reference: {args.reference}')
    print(f"{'bag':10s} {'ref z':>6s} {'ZED - ref':>9s} {'corrected - ref':>15s}  (mm)")
    for n, zc, zr, zk in zip(names, z_cam, z_ref, corrected):
        print(f'{n:10s} {zr:6.3f} {1000 * (zc - zr):+9.1f} {1000 * (zk - zr):+15.1f}')
    print(f'\nbeta = {beta:.5f} 1/m  (1/z_measured = 1/z_true + beta)')
    print(f'residual after correction: rms {1000 * np.sqrt(np.mean((corrected - z_ref) ** 2)):.1f} mm, '
          f'max {1000 * np.max(np.abs(corrected - z_ref)):.1f} mm')
    return 0


if __name__ == '__main__':
    sys.exit(main())
