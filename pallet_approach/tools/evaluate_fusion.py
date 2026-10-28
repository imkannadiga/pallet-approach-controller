#!/usr/bin/env python3
"""The whole fused chain (D102 a-c) on recorded bags, robot standing still.

    python3 tools/evaluate_fusion.py ~/bags/pallet_s_* ~/bags/empty

LiDAR acquisition on five pooled sweeps; then every LiDAR sweep (single,
`find_pallet`) and every depth frame (`camera_pockets.measure` against the
filter's current face, D103 depth correction on) go through the filter in
time order, with the disagreement check running. The robot is still, so odom
is base_footprint. Reports the final estimate against the tape, whether GO
is allowed, and why not.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np
import yaml

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import _bagutil as bag  # noqa: E402
from pallet_approach.camera_pockets import measure  # noqa: E402
from pallet_approach.fusion import PalletFilter  # noqa: E402
from pallet_approach.lidar_pallet import find_pallet  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('bags', nargs='+', type=pathlib.Path)
    ap.add_argument('--seconds', type=float, default=4.0, help='of data replayed per bag')
    args = ap.parse_args()
    shared = bag.shared_params()
    ec = shared['frames']['extrinsic_correction']
    truths = yaml.safe_load((HERE.parent.parent / 'docs' / 'bag_truth.yaml').read_text()) or {}
    print(f"{'bag':22s} {'dx':>5s} {'dy':>5s} {'yaw':>6s} {'s_lat':>6s} {'lidar':>7s} {'camera':>7s} "
          f"{'GO':>3s}  reason")
    for path in args.bags:
        n_frames = int(args.seconds * 4)
        data = bag.read(path, max_frames=n_frames)
        sweeps = bag.lidar_sweeps(path, data, shared, int(args.seconds * 10) + 5)
        face, info = find_pallet(np.concatenate(sweeps[:5]), shared)
        if face is None:
            print(f'{path.name:22s} no pallet acquired')
            continue
        cam = bag.camera_to_ground(data, ec)
        side = shared['pallet']['sides'][face.side]
        f = PalletFilter(shared)
        start = face.pocket_centre if face.pocket_centre is not None else face.centre
        f.initialise(start, face.yaw, 0.0)
        events = [(i / 10.0, 'lidar', s) for i, s in enumerate(sweeps[5:])]
        events += [(i / 4.0, 'camera', d) for i, d in enumerate(data['depth'])]
        accepted = {'lidar': 0, 'camera': 0}
        tried = {'lidar': 0, 'camera': 0}
        for t, kind, item in sorted(events, key=lambda e: e[0]):
            tried[kind] += 1
            if kind == 'lidar':
                r, _ = find_pallet(item, shared)
                if r is not None:
                    c = r.pocket_centre if r.pocket_centre is not None else r.centre
                    accepted['lidar'] += f.update_lidar(c, r.yaw, t)
            else:
                st = f.status(t)
                m = measure(item, data['intrinsics'], cam, st.centre, st.yaw, info['floor_plane'], side,
                            shared['camera_pockets'], shared['camera_depth']['disparity_bias'])
                if m.offset is not None and m.both_open:
                    accepted['camera'] += f.update_camera(st.centre + m.offset * st.tangent, t)
                    f.check_camera(m.range_residual, m.yaw_residual)
        end = max(e[0] for e in events)
        s = f.status(end)
        row = truths.get(path.name)
        if row and row.get('measured'):
            w = side['face_width']
            tx = (row['dL'] + row['dR']) / 2 + bag.BUMPER_X
            ty = row.get('lat', 0.0)
            tyaw = math.asin(max(-1, min(1, (row['dL'] - row['dR']) / w)))
            fyaw = math.atan2(math.sin(math.pi - s.yaw), math.cos(math.pi - s.yaw))
            err = f'{1000 * (s.centre[0] - tx):+5.0f} {1000 * (s.centre[1] - ty):+5.0f} {math.degrees(fyaw - tyaw):+6.1f}'
        else:
            err = f"{'-':>5s} {'-':>5s} {'-':>6s}"
        print(f"{path.name:22s} {err} {1000 * s.sigma_lateral:6.1f} {accepted['lidar']:3d}/{tried['lidar']:<3d} "
              f"{accepted['camera']:3d}/{tried['camera']:<3d} {'yes' if s.go_allowed else 'no':>3s}  {s.reason}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
