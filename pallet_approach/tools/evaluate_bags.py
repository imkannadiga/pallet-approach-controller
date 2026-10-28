#!/usr/bin/env python3
"""How well does perception find the pockets on real bags? Every frame, every bag.

    python3 tools/evaluate_bags.py ~/bags/pallet_s_129 ~/bags/pallet_s_205 ...
    python3 tools/evaluate_bags.py ~/bags/*  --truth ../docs/bag_truth.yaml --csv out.csv

`replay.py` draws one frame's stages; this answers the question the robot runs
into instead: over all the frames of a bag, how often are both pockets found,
how far are their centres from where the tape says they are, and how much do
they move from one frame to the next while nothing moves at all.

The pipeline is `perception.estimate_pallet` with `config/params.yaml` and the
same extrinsic correction the robot uses, so a number here is a number about
the robot's perception. No ROS.

**Truth** comes from `docs/bag_truth.yaml`: the tape readings of
`docs/dataset.md` (bumper to the left and right front corners, and the lateral
offset), from which the face centre, its angle and both pocket centres follow
with the measured pallet geometry. A bag without truth still gets the detection
rate and the frame-to-frame scatter -- the scatter needs no tape -- but no
accuracy. A truth marked `measured: false` (a target distance, not a tape
reading) is reported, and flagged, but must not be quoted as accuracy.

Columns, per bag:

    found      frames with a pallet / frames
    2 pockets  frames with both pockets
    mid dx/dy  pocket midpoint error against the tape, median and p95 (mm);
               dx along the robot's x, dy sideways
    yaw        face direction error, median and p95 (deg)
    spacing    measured centre spacing minus 0.365 m, median (mm)
    width      measured pocket width minus 0.230 m, median (mm)
    height     pocket centre above the fitted floor minus 0.050 m, median (mm)
    scatter    frame-to-frame standard deviation of the midpoint, x / y (mm)
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PACKAGE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PACKAGE))
import _bagutil as bag  # noqa: E402
from pallet_approach import perception as P  # noqa: E402
from pallet_approach.approach import wrap_angle as wrap  # noqa: E402




def truth_for(entry: dict, sides: dict) -> dict:
    """Face centre, yaw and pocket centres in base_footprint, from tape.

    `dL`/`dR`: bumper line to the left/right front corner, metres, measured
    perpendicular to the bumper. `lat`: face centre's offset, positive = left.
    A positive angle turns the pallet's left edge away (dataset.md).
    """
    side = sides[entry.get('side', 'short')]
    width, spacing = float(side['face_width']), float(side['pocket_spacing'])
    d_left, d_right = float(entry['dL']), float(entry['dR'])
    theta = math.asin(max(-1.0, min(1.0, (d_left - d_right) / width)))
    center = np.array([(d_left + d_right) / 2.0 + bag.BUMPER_X, float(entry.get('lat', 0.0))])
    left = np.array([math.sin(theta), math.cos(theta)])     # along the face, to the left
    normal = np.array([-math.cos(theta), math.sin(theta)])  # towards the robot
    return {
        'center': center,
        'yaw': math.atan2(normal[1], normal[0]),
        'pockets': (center + left * spacing / 2.0, center - left * spacing / 2.0),
        'measured': bool(entry.get('measured', True)),
        'spacing': spacing,
        'width': float(side['pocket_width']),
        'height': (float(side['opening_bottom']) + float(side['opening_top'])) / 2.0,
    }


def evaluate(path: pathlib.Path, params: dict, truth: dict | None,
             max_frames: int) -> dict:
    data = bag.read(path, max_frames=max_frames)
    if not data['depth'] or data['intrinsics'] is None:
        return {'bag': path.name, 'frames': 0, 'error': 'no depth or camera_info'}
    to_ground = bag.camera_to_ground(data, params['extrinsic_correction'])

    rows, seconds = [], []
    for depth in data['depth']:
        start = time.perf_counter()
        est = P.estimate_pallet(depth, data['intrinsics'], to_ground, params)
        seconds.append(time.perf_counter() - start)
        rows.append(est)

    found = [e for e in rows if e is not None]
    pairs = [e for e in found if len(e.pockets) == 2]
    out = {'bag': path.name, 'frames': len(rows), 'found': len(found),
           'pairs': len(pairs), 'ms': 1000.0 * float(np.median(seconds)),
           'sides': ','.join(sorted({e.side for e in found})) or '-'}
    if not pairs:
        return out

    left = np.array([e.pockets[0].center[:2] for e in pairs])
    right = np.array([e.pockets[1].center[:2] for e in pairs])
    mid = (left + right) / 2.0
    out['spacing'] = np.linalg.norm(left - right, axis=1)
    out['width'] = np.array([[p.width for p in e.pockets] for e in pairs]).ravel()
    out['height'] = np.array([[float(e.floor.height_of(p.center)) if e.floor else np.nan
                               for p in e.pockets] for e in pairs]).ravel()
    out['scatter'] = mid.std(axis=0) if len(mid) > 1 else np.full(2, np.nan)
    out['yaw'] = np.array([e.yaw for e in pairs])
    if truth is not None:
        out['mid_err'] = mid - truth['center']
        out['pocket_err'] = np.concatenate([left - truth['pockets'][0],
                                            right - truth['pockets'][1]])
        out['yaw_err'] = np.degrees([wrap(y - truth['yaw']) for y in out['yaw']])
        out['measured'] = truth['measured']
    out['truth'] = truth
    return out


def mm(values, stat='median') -> str:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return '-'
    if stat == 'median':
        return f'{1000 * np.median(values):+.0f}'
    return f'{1000 * np.percentile(np.abs(values), 95):.0f}'


def report(results: list[dict], nominal: dict) -> list[dict]:
    header = (f'{"bag":24s} {"found":>9s} {"2 pockets":>9s} {"mid dx":>11s} '
              f'{"mid dy":>11s} {"yaw":>11s} {"spacing":>7s} {"width":>6s} '
              f'{"height":>6s} {"scatter":>9s} {"ms":>5s}')
    print(header)
    print('-' * len(header))
    table = []
    for r in results:
        if r.get('error'):
            print(f'{r["bag"]:24s} {r["error"]}')
            continue
        row = {'bag': r['bag'], 'frames': r['frames'], 'found': r['found'],
               'pairs': r['pairs'], 'ms': round(r['ms'])}
        line = (f'{r["bag"]:24s} {r["found"]:>4d}/{r["frames"]:<4d} '
                f'{r["pairs"]:>4d}/{r["frames"]:<4d}')
        if r['pairs']:
            if 'mid_err' in r:
                dx, dy = r['mid_err'][:, 0], r['mid_err'][:, 1]
                line += (f' {mm(dx):>5s}/{mm(dx, "p95"):<5s} {mm(dy):>5s}/{mm(dy, "p95"):<5s}'
                         f' {np.median(r["yaw_err"]):+5.1f}/{np.percentile(np.abs(r["yaw_err"]), 95):<4.1f}')
                row.update(mid_dx_median_mm=mm(dx), mid_dx_p95_mm=mm(dx, 'p95'),
                           mid_dy_median_mm=mm(dy), mid_dy_p95_mm=mm(dy, 'p95'),
                           yaw_median_deg=round(float(np.median(r['yaw_err'])), 2),
                           truth='tape' if r['measured'] else 'TARGET ONLY')
            else:
                line += f' {"no truth":>11s} {"":>11s} {"":>11s}'
            line += (f' {mm(r["spacing"] - nominal["spacing"]):>7s}'
                     f' {mm(r["width"] - nominal["width"]):>6s}'
                     f' {mm(r["height"] - nominal["height"]):>6s}'
                     f' {1000 * r["scatter"][0]:4.0f}/{1000 * r["scatter"][1]:<4.0f}')
            row.update(spacing_median_mm=mm(r['spacing'] - nominal['spacing']),
                       width_median_mm=mm(r['width'] - nominal['width']),
                       height_median_mm=mm(r['height'] - nominal['height']),
                       scatter_x_mm=round(1000 * float(r['scatter'][0]), 1),
                       scatter_y_mm=round(1000 * float(r['scatter'][1]), 1))
        line += f' {r["ms"]:5.0f}'
        if r.get('truth') is not None and not r['truth']['measured']:
            line += '  (truth is a TARGET, not tape)'
        print(line)
        table.append(row)
    return table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('bags', nargs='+', type=pathlib.Path)
    ap.add_argument('--truth', type=pathlib.Path,
                    default=PACKAGE.parent / 'docs' / 'bag_truth.yaml')
    ap.add_argument('--max-frames', type=int, default=200,
                    help='depth frames read per bag (memory: ~3.7 MB each)')
    ap.add_argument('--csv', type=pathlib.Path, help='also write the table here')
    args = ap.parse_args()

    import yaml
    params = bag.load_params(PACKAGE)
    raw = yaml.safe_load((PACKAGE / 'config' / 'params.yaml').read_text())
    sides = raw['/**']['ros__parameters']['pallet']['sides']
    entries = (yaml.safe_load(args.truth.read_text()) or {}) if args.truth.exists() else {}

    results = []
    for path in args.bags:
        if not (path / 'metadata.yaml').exists():
            continue
        entry = entries.get(path.name)
        truth = truth_for(entry, sides) if entry and 'dL' in entry else None
        print(f'... {path.name}', file=sys.stderr)
        results.append(evaluate(path, params, truth, args.max_frames))

    short = sides['short']
    nominal = {'spacing': short['pocket_spacing'], 'width': short['pocket_width'],
               'height': (short['opening_bottom'] + short['opening_top']) / 2.0}
    table = report(results, nominal)
    if args.csv and table:
        keys = sorted({k for row in table for k in row}, key=lambda k: (k != 'bag', k))
        with args.csv.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(table)
    return 0


if __name__ == '__main__':
    sys.exit(main())
