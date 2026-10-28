#!/usr/bin/env python3
"""Score the LiDAR-first acquisition (D102 step a) against the tape, bag by bag.

    python3 tools/evaluate_lidar_acquisition.py ~/bags/pallet_s_* ~/bags/pallet_l_* ~/bags/empty
    python3 tools/evaluate_lidar_acquisition.py ~/bags/pallet_s_205 --stack 5

Each LiDAR sweep is run on its own (or `--stack` consecutive sweeps together,
as acquisition would with the robot standing still), in `base_footprint` with
the LiDAR pose params.yaml has in force (the D101 calibration). Truth comes
from docs/bag_truth.yaml: the face centre is (dL + dR) / 2 ahead of the bumper,
`lat` to the side, and the angle from dL - dR over the face width. Bags with no
truth row (`empty`) are scored for false detections only.

Columns: found = sweeps with a pallet; side; in range = the D97 working range
said yes; dx / dy = face centre error (mm, median); pocket dy = the pocket
pair's centre, sideways (mm, median) and how often the pockets were found;
yaw = error (deg, median); width (m); ms = time per call.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time

import numpy as np
import yaml

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import _bagutil as bag  # noqa: E402
from pallet_approach.lidar_pallet import find_pallet  # noqa: E402


def truth_for(row: dict, shared: dict):
    side = row.get('side', 'short')
    width = shared['pallet']['sides'][side]['face_width']
    angle = math.asin(max(-1.0, min(1.0, (row['dL'] - row['dR']) / width)))
    return np.array([(row['dL'] + row['dR']) / 2 + bag.BUMPER_X, row.get('lat', 0.0)]), angle, side


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('bags', nargs='+', type=pathlib.Path)
    ap.add_argument('--clouds', type=int, default=30)
    ap.add_argument('--stack', type=int, default=1, help='sweeps stacked per call')
    ap.add_argument('--lidar-pose', default='auto', choices=('auto', 'calibrated', 'd74', 'urdf'))
    args = ap.parse_args()
    shared = bag.shared_params()
    truths = yaml.safe_load((HERE.parent.parent / 'docs' / 'bag_truth.yaml').read_text()) or {}

    print(f"{'bag':22s} {'found':>7s} {'side':>6s} {'in rng':>6s} {'dx':>5s} {'dy':>5s} "
          f"{'pocket dy':>12s} {'yaw':>6s} {'width':>6s} {'ms':>5s}")
    for path in args.bags:
        data = bag.read(path, max_frames=1)
        pts = bag.lidar_sweeps(path, data, shared, args.clouds, args.lidar_pose)
        if not pts:
            print(f'{path.name:22s} no LiDAR')
            continue
        calls = [np.concatenate(pts[i:i + args.stack]) for i in range(0, len(pts) - args.stack + 1, args.stack)]
        results, ms = [], []
        for cloud in calls:
            t0 = time.perf_counter()
            r, _ = find_pallet(cloud, shared)
            ms.append(1000 * (time.perf_counter() - t0))
            results.append(r)
        found = [r for r in results if r is not None]
        row = truths.get(path.name)
        line = f'{path.name:22s} {len(found):3d}/{len(results):<3d}'
        if not found:
            print(line + f"{'-':>7s}{'':>39s} {np.median(ms):5.0f}")
            continue
        sides = {r.side for r in found}
        in_rng = sum(r.in_range for r in found)
        line += f" {'/'.join(sorted(sides)):>6s} {in_rng:3d}/{len(found):<2d}"
        if row is None or not row.get('measured', False):
            print(line + f"{'(no truth)':>32s} {np.median([r.width for r in found]):6.3f} {np.median(ms):5.0f}")
            continue
        centre, angle, side = truth_for(row, shared)
        dx = 1000 * np.median([r.centre[0] - centre[0] for r in found])
        dy = 1000 * np.median([r.centre[1] - centre[1] for r in found])
        pk = [r.pocket_centre for r in found if r.pocket_centre is not None]
        pdy = f'{1000 * np.median([q[1] - centre[1] for q in pk]):+5.0f} {len(pk):2d}/{len(found):<2d}' if pk else '    - ' + f'0/{len(found)}'
        tape_yaw = [math.degrees(math.atan2(math.sin(math.pi - r.yaw), math.cos(math.pi - r.yaw))) for r in found]
        dyaw = np.median(tape_yaw) - math.degrees(angle)
        print(line + f' {dx:+5.0f} {dy:+5.0f} {pdy:>12s} {dyaw:+6.1f} {np.median([r.width for r in found]):6.3f} '
                     f'{np.median(ms):5.0f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
