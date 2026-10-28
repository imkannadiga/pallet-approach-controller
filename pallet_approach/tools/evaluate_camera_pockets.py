#!/usr/bin/env python3
"""Score the camera's pockets along the LiDAR's face line (D102 step b), bag by bag.

    python3 tools/evaluate_camera_pockets.py ~/bags/pallet_s_*

Per bag: the LiDAR finds the face once (five sweeps pooled, as acquisition
does standing still); then every depth frame measures the pockets along it.
Truth (docs/bag_truth.yaml) puts the pocket pair's centre on the face centre:
(dL + dR) / 2 ahead of the bumper, `lat` to the side. `lat` is only taped on
the _lat_ bags; on the rest it is 0 by setup, so a few cm there is the setup.

Columns: found = frames with both pockets; dy = pocket-pair centre sideways
error (mm, median) for the camera and, for comparison, the LiDAR's own pocket
search; scatter = frame-to-frame std of the camera's (mm); cam-lidar range =
median camera depth minus LiDAR face on the wood (mm); ms = per frame.
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
from pallet_approach.camera_pockets import measure  # noqa: E402
from pallet_approach.lidar_pallet import find_pallet  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('bags', nargs='+', type=pathlib.Path)
    ap.add_argument('--frames', type=int, default=20)
    ap.add_argument('--disparity-bias', type=float, default=None,
                    help='override camera_depth.disparity_bias (0 = no depth correction)')
    args = ap.parse_args()
    shared = bag.shared_params()
    if args.disparity_bias is None:
        args.disparity_bias = shared['camera_depth']['disparity_bias']
    truths = yaml.safe_load((HERE.parent.parent / 'docs' / 'bag_truth.yaml').read_text()) or {}
    ec = shared['frames']['extrinsic_correction']
    print(f"{'bag':22s} {'found':>7s} {'cam dy':>7s} {'lidar dy':>8s} {'scatter':>7s} "
          f"{'cam-lidar':>9s} {'score':>6s} {'ms':>5s}")
    for path in args.bags:
        data = bag.read(path, max_frames=args.frames)
        sweeps = bag.lidar_sweeps(path, data, shared, 5)
        if not sweeps or not data['depth']:
            print(f'{path.name:22s} missing LiDAR or depth')
            continue
        face, info = find_pallet(np.concatenate(sweeps), shared)
        if face is None:
            print(f'{path.name:22s} the LiDAR found no pallet')
            continue
        cam = bag.camera_to_ground(data, ec)
        side = shared['pallet']['sides'][face.side]
        results = [measure(d, data['intrinsics'], cam, face.centre, face.yaw, info['floor_plane'], side,
                           shared['camera_pockets'], args.disparity_bias) for d in data['depth']]
        ok = [r for r in results if r.offset is not None and r.both_open]
        line = f'{path.name:22s} {len(ok):3d}/{len(results):<3d}'
        if not ok:
            print(line + f"  best score {max(r.score for r in results):.2f}, contrast {max(r.contrast for r in results):.2f}")
            continue
        centres = np.array([face.centre + r.offset * face.tangent for r in ok])
        row = truths.get(path.name)
        truth_y = row.get('lat', 0.0) if row and row.get('measured') else float('nan')
        lid = face.pocket_centre
        cam_dy = 1000 * (np.median(centres[:, 1]) - truth_y)
        lid_dy = 1000 * (lid[1] - truth_y) if lid is not None else float('nan')
        scatter = 1000 * np.std([r.offset for r in ok])
        rr = 1000 * np.median([r.range_residual for r in ok])
        print(line + f' {cam_dy:+7.0f} {lid_dy:+8.0f} {scatter:7.1f} {rr:+9.0f} '
                     f'{np.median([r.score for r in ok]):6.2f} {np.median([r.ms for r in results]):5.1f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
