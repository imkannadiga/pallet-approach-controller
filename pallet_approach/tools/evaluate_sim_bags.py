"""The camera pipeline on sim bags, scored against the exact truth in truth.yaml.

    python3 tools/evaluate_sim_bags.py ~/bags/sim/sim_* [--step 4]

The sim is the one place with exact ground truth (the launch arguments,
recorded beside each bag). Its depth is a fitted model (D80, D105), so this
says nothing about real depth. Errors of the pocket midpoint, in the pallet's
own axes: range (+ = farther than the true face; against the block face when
the blocks stand proud, else the deck), lateral (+ = robot's left), yaw (deg).
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
from pallet_approach import perception as P  # noqa: E402
from pallet_approach.approach import wrap_angle as wrap  # noqa: E402


class Truth:
    def __init__(self, raw: dict):
        self.raw = raw
        self.name = raw['name']
        self.centre = np.array([float(raw['face_x']), float(raw['face_y'])])
        self.yaw = math.pi + float(raw['face_yaw'])          # outward normal, at the robot
        self.normal = np.array([math.cos(self.yaw), math.sin(self.yaw)])
        self.tangent = np.array([self.normal[1], -self.normal[0]])
        self.protrusion = float(raw.get('block_protrusion', 0.0))
        self.load = float(raw.get('load_height', 0.0))
        self.overhang = float(raw.get('load_overhang', 0.0))

    def block_centre(self) -> np.ndarray:
        """The pocket midpoint on the BLOCK face: nearer the robot by the protrusion."""
        return self.centre + self.protrusion * self.normal

    def errors(self, mid_xy, yaw: float) -> dict:
        out = {}
        for label, ref in (('deck', self.centre), ('block', self.block_centre())):
            delta = np.asarray(mid_xy, dtype=float) - ref
            out[f'range_{label}'] = -float(delta @ self.normal) * 1000.0
            out[f'lateral_{label}'] = float(delta @ self.tangent) * 1000.0
        out['yaw'] = math.degrees(wrap(yaw - self.yaw))
        return out


def summary(rows: list[dict], key: str) -> str:
    values = np.array([r[key] for r in rows], dtype=float)
    return 'n/a' if values.size == 0 else f'{np.median(values):+.1f}'


def evaluate(path: pathlib.Path, params: dict, step: int) -> dict | None:
    truth_file = path / 'truth.yaml'
    if not truth_file.exists():
        return None
    truth = Truth(yaml.safe_load(truth_file.read_text()))
    data = bag.read(path, max_frames=400)
    if not data['depth'] or data['intrinsics'] is None:
        return {'name': truth.name, 'error': 'no depth or camera_info'}
    to_ground = data['tree'].lookup(bag.GROUND, data['depth_frame'] or bag.CAMERA)
    intr = data['intrinsics']
    frames = data['depth'][::step]

    rows, ms = [], []
    for depth in frames:
        t0 = time.perf_counter()
        est = P.estimate_pallet(depth, intr, to_ground, params)
        ms.append((time.perf_counter() - t0) * 1000)
        if est is not None and len(est.pockets) == 2:
            mid = np.mean([p.center[:2] for p in est.pockets], axis=0)
            rows.append(truth.errors(mid, est.yaw))
    return {'name': truth.name, 'frames': len(frames), 'pairs': len(rows),
            'ms': float(np.median(ms)), 'rows': rows, 'truth': truth}


def report(results: list[dict]) -> None:
    print(f'{"bag":18s} {"pose":22s} {"pairs":>7s} {"range":>8s} {"lat":>7s} {"yaw":>6s} {"ms":>5s}')
    for r in results:
        if 'error' in r:
            print(f'{r["name"]:18s} {r["error"]}')
            continue
        t = r['truth']
        pose = f'x{t.centre[0]:.2f} y{t.centre[1]:+.2f}'
        if t.protrusion:
            pose += f' p{t.protrusion * 100:.0f}'
        if t.load:
            pose += f' L{t.load:.2f}'
        key = 'block' if t.protrusion else 'deck'
        rows = r['rows']
        print(f'{r["name"]:18s} {pose:22s} {r["pairs"]:>3d}/{r["frames"]:<3d} '
              f'{summary(rows, "range_" + key):>8s} {summary(rows, "lateral_" + key):>7s} '
              f'{summary(rows, "yaw"):>6s} {r["ms"]:5.1f}')
    print('\nmm and deg, median over frames; range + = farther than the true face '
          '(the block face for bags with blocks proud, p.., else the deck).')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('bags', nargs='+', type=pathlib.Path)
    ap.add_argument('--step', type=int, default=4, help='use every Nth depth frame')
    args = ap.parse_args()
    params = bag.load_params(HERE.parent, overlay='sim_overlay.yaml')
    results = []
    for path in args.bags:
        print(f'... {path.name}', file=sys.stderr)
        try:
            r = evaluate(path, params, args.step)
        except Exception as error:                              # noqa: BLE001
            r = {'name': path.name, 'error': f'{type(error).__name__}: {error}'}
        if r is not None:
            results.append(r)
    report(results)
    return 0


if __name__ == '__main__':
    sys.exit(main())
