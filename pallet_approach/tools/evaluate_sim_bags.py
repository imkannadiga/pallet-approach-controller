"""The current pipeline and the strip-and-template tracker, on sim bags, against truth.

    python3 tools/evaluate_sim_bags.py ~/bags/sim/sim_* [--step 4]

A sim bag has exact ground truth: the pallet was placed at the launch arguments
recorded in `truth.yaml`. So this is the one place the two methods can be scored
on ACCURACY and not only on agreement with each other. What it does not say
anything about is real depth: the sim's depth is a fitted model (sim/README.md),
and beyond about 2.4 m it extrapolates.

Per bag it reports, for each method, the errors of the pocket midpoint against
the truth in the pallet's own axes:

    range    how much FARTHER than the true plane the estimate is (negative =
             nearer). Scored against the DECK face, and against the BLOCK face
             when the bag has blocks standing proud of the deck.
    lateral  along the face, positive to the robot's left.
    yaw      the face normal's direction, degrees.

The tracker runs two ways, so what it does on its own is separate from what it
inherits:

    prior = the current pipeline's first estimate  (only if that found pockets)
    prior = the truth moved by 5 cm along the normal, 5 cm sideways, 2 degrees
            (what a tracker with a good prior would hold; independent of the
            current pipeline entirely, including its floor fit)
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
import track_prototype as T  # noqa: E402
from pallet_approach import perception as P  # noqa: E402
from pallet_approach.model import Plane  # noqa: E402

SIM_FLOOR = Plane(np.array([0.0, 0.0, 1.0]), 0.0248)
"""The sim's floor in base_footprint: z = -0.0248 (wheel geometry, sim/README.md)."""


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


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
    geo = T.Geometry.from_params(params)
    frames = data['depth'][::step]

    base_rows, ms = [], []
    base_est = []
    for depth in frames:
        t0 = time.perf_counter()
        est = P.estimate_pallet(depth, intr, to_ground, params)
        ms.append((time.perf_counter() - t0) * 1000)
        base_est.append(est)
        if est is not None and len(est.pockets) == 2:
            mid = np.mean([p.center[:2] for p in est.pockets], axis=0)
            base_rows.append(truth.errors(mid, est.yaw))
    out = {'name': truth.name, 'frames': len(frames),
           'base_found': sum(e is not None for e in base_est),
           'base_pairs': len(base_rows), 'base_ms': float(np.median(ms)),
           'base': base_rows, 'truth': truth}

    def run_tracker(prior):
        rows, times = [], []
        track_floor = prior.floor
        T.track(frames[0], intr, to_ground, prior, geo)                # warm-up
        for depth in frames:
            res = T.track(depth, intr, to_ground, prior, geo)
            times.append(res.seconds * 1000)
            if res.ok:
                rows.append(truth.errors(res.mid[:2], res.yaw))
                prior = T.prior_from_result(res, T.Prior(prior.centre, prior.normal, track_floor))
        return rows, float(np.median(times))

    first = next((e for e in base_est if e is not None and len(e.pockets) == 2), None)
    if first is not None:
        out['proto_a'], out['proto_a_ms'] = run_tracker(T.prior_from(first))
    else:
        out['proto_a'], out['proto_a_ms'] = [], float('nan')
    c = truth.centre + 0.05 * truth.normal + 0.05 * truth.tangent
    angle = math.radians(2.0)
    n2 = np.array([math.cos(truth.yaw + angle), math.sin(truth.yaw + angle)])
    out['proto_b'], out['proto_b_ms'] = run_tracker(T.Prior(c, n2, SIM_FLOOR))
    return out


def report(results: list[dict]) -> None:
    print(f'{"bag":18s} {"pose":22s} {"CURRENT PIPELINE":^40s} {"TRACKER, truth+5cm prior":^34s}')
    print(f'{"":18s} {"":22s} {"pairs":>7s} {"range":>8s} {"lat":>7s} {"yaw":>6s} {"ms":>5s}'
          f' {"ok":>5s} {"range":>8s} {"lat":>7s} {"yaw":>6s} {"ms":>5s}')
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
        # Score against the block face when the blocks stand proud, else the deck.
        key = 'block' if t.protrusion else 'deck'
        b, p = r['base'], r['proto_b']
        print(f'{r["name"]:18s} {pose:22s} {r["base_pairs"]:>3d}/{r["frames"]:<3d} '
              f'{summary(b, "range_" + key):>8s} {summary(b, "lateral_" + key):>7s} '
              f'{summary(b, "yaw"):>6s} {r["base_ms"]:5.1f}'
              f' {len(p):>3d}/{r["frames"]:<1d} {summary(p, "range_" + key):>8s} '
              f'{summary(p, "lateral_" + key):>7s} {summary(p, "yaw"):>6s} {r["proto_b_ms"]:5.1f}')
    print('\nrange/lat in mm (median over frames; range: + = estimate farther than the true face), '
          'yaw in degrees.\nScored against the block face for bags with blocks proud (p..), '
          'the deck otherwise.')
    print('Tracker started from the TRUTH moved 5 cm / 5 cm / 2 deg, not from the current pipeline.')
    a = [r for r in results if 'proto_a' in r and r['proto_a']]
    if a:
        print(f'Tracker started from the current pipeline\'s first estimate: ran on {len(a)} of '
              f'{len(results)} bags (the rest had no two-pocket estimate to start from).')


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
