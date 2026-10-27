"""PROTOTYPE, not wired into the robot: strip-and-correlate pocket tracking.

Does a cheaper, template-based tracker find the pockets as well as the current
pipeline, and does it survive what the current one cannot (a load on the deck)?

    python3 tools/track_prototype.py ~/bags/pallet_s_pos2_cond1 [--frames 280] [--step 4]

How the prototype works, given a prior (where the pallet was last frame):

1. Project a box around the expected pocket band into the depth image and
   unproject only that strip of pixels (a few percent of the image).
2. Keep points in the pocket band (heights above the fitted floor) and fit a
   vertical line  d = a + b*u  to them, coarse to fine, where u runs along the
   face and d across it. That gives the face's range and yaw.
3. Count points near that line in 5 mm slots along the face, and slide a
   template over the counts: wood (+1) where a block is, hole (-1) where a
   pocket is. The best shift is the pallet's position along the face; the
   pocket centres follow from the template.

It uses the baseline's own floor plane and prior, so what is timed here is the
per-frame cost AFTER a tracker exists. It does not replace acquiring the pallet
in the first place.

Everything is reported against the CURRENT pipeline on the same frames, not
against tape: neither bag has tape truth (docs/bag_truth.yaml). Agreement with
the baseline and frame-to-frame scatter are the only accuracy-like numbers it
can give, and they say nothing about a bias both methods share.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time
from dataclasses import dataclass

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import _bagutil as bag  # noqa: E402
from pallet_approach import perception as P  # noqa: E402
from pallet_approach.calibration import correct_transform, correction_matrix  # noqa: E402
from pallet_approach.model import Plane  # noqa: E402

BIN = 0.005
"""Slot width along the face, metres."""

SPAN = 0.65
"""Half-length of the profile, metres. The face is 0.40 either side of centre."""


@dataclass(frozen=True)
class Geometry:
    face_width: float
    pocket_width: float
    pocket_spacing: float
    band_lo: float
    band_hi: float
    deck: float

    @classmethod
    def from_params(cls, params: dict) -> 'Geometry':
        centre = params['pocket_center_height']
        half = params['pocket_band_half_height']
        return cls(face_width=params['pallet_width'],
                   pocket_width=params['pocket_nominal_width'],
                   pocket_spacing=params['pocket_nominal_spacing'],
                   band_lo=centre - half, band_hi=centre + half,
                   deck=params['pallet_height'])

    @property
    def band_centre(self) -> float:
        return 0.5 * (self.band_lo + self.band_hi)


@dataclass(frozen=True)
class Prior:
    centre: np.ndarray          # (2,) pocket midpoint, base_footprint x, y
    normal: np.ndarray          # (2,) unit, pointing at the robot
    floor: Plane

    @property
    def tangent(self) -> np.ndarray:
        """Along the face, positive to the robot's left."""
        return np.array([self.normal[1], -self.normal[0]])


@dataclass
class Result:
    ok: bool
    reason: str = ''
    mid: np.ndarray | None = None
    pockets: np.ndarray | None = None      # (2, 3): left, right
    yaw: float = float('nan')
    quality: dict | None = None
    seconds: float = 0.0
    window_pixels: int = 0


def kernel(geo: Geometry) -> np.ndarray:
    """Wood +1, pocket -1, beyond the face ends -0.3, on the slot grid."""
    half = int(round((geo.face_width / 2 + 0.10) / BIN))
    u = np.arange(-half, half + 1) * BIN
    k = np.full(u.shape, -0.3)
    k[np.abs(u) <= geo.face_width / 2] = 1.0
    for sign in (-1.0, 1.0):
        k[np.abs(u - sign * geo.pocket_spacing / 2) <= geo.pocket_width / 2] = -1.0
    return k


def window(shape, intr, to_cam, prior, geo, half_width=0.55, half_depth=0.15, pad=4):
    """Pixel box (r0, r1, c0, c1) around the expected pocket band, or None."""
    n, t = prior.normal, prior.tangent
    fn, fo = prior.floor.normal, prior.floor.offset
    corners = []
    for u in (-half_width, half_width):
        for d in (-half_depth, half_depth):
            for h in (geo.band_lo - 0.02, geo.band_hi + 0.02):
                x, y = prior.centre + u * t + d * n
                z = (h - fo - fn[0] * x - fn[1] * y) / fn[2]
                corners.append([x, y, z, 1.0])
    cam = (to_cam @ np.array(corners).T).T[:, :3]
    if (cam[:, 2] < 0.2).any():
        return None
    px = intr['fx'] * cam[:, 0] / cam[:, 2] + intr['cx']
    py = intr['fy'] * cam[:, 1] / cam[:, 2] + intr['cy']
    c0, c1 = int(math.floor(px.min())) - pad, int(math.ceil(px.max())) + pad
    r0, r1 = int(math.floor(py.min())) - pad, int(math.ceil(py.max())) + pad
    r0, c0 = max(r0, 0), max(c0, 0)
    r1, c1 = min(r1, shape[0]), min(c1, shape[1])
    return (r0, r1, c0, c1) if r1 - r0 > 2 and c1 - c0 > 2 else None


def track(depth, intr, to_ground, prior: Prior, geo: Geometry,
          kern: np.ndarray | None = None, debug: dict | None = None,
          slab: float = 0.02) -> Result:
    start = time.perf_counter()
    kern = kernel(geo) if kern is None else kern
    to_cam = np.linalg.inv(to_ground)
    box = window(depth.shape, intr, to_cam, prior, geo)
    if box is None:
        return Result(False, 'the expected strip is outside the image')
    r0, r1, c0, c1 = box

    z = depth[r0:r1, c0:c1].astype(np.float64)
    v, u_px = np.mgrid[r0:r1, c0:c1]
    ok = np.isfinite(z) & (z > 0.3) & (z < 5.0)
    z, v, u_px = z[ok], v[ok], u_px[ok]
    cam = np.column_stack(((u_px - intr['cx']) * z / intr['fx'],
                           (v - intr['cy']) * z / intr['fy'], z))
    ground = cam @ to_ground[:3, :3].T + to_ground[:3, 3]
    height = prior.floor.height_of(ground)
    band = (height >= geo.band_lo) & (height <= geo.band_hi)
    ground = ground[band]
    n, t = prior.normal, prior.tangent
    rel = ground[:, :2] - prior.centre
    u, d = rel @ t, rel @ n

    a = b = 0.0
    for w in (0.10, 0.05, 0.03, 0.02):
        sel = (np.abs(d - (a + b * u)) < w) & (np.abs(u) < 0.55)
        if sel.sum() < 30:
            return Result(False, f'only {int(sel.sum())} points near the face',
                          seconds=time.perf_counter() - start,
                          window_pixels=(r1 - r0) * (c1 - c0))
        b, a = np.polyfit(u[sel], d[sel], 1)

    perp = (d - a - b * u) / math.sqrt(1.0 + b * b)
    near = np.abs(perp) < slab
    slots = int(round(2 * SPAN / BIN))
    index = ((u[near] + SPAN) / BIN).astype(int)
    index = index[(index >= 0) & (index < slots)]
    counts = np.bincount(index, minlength=slots).astype(float)
    if counts.sum() < 30:
        return Result(False, 'too few points on the face',
                      seconds=time.perf_counter() - start,
                      window_pixels=(r1 - r0) * (c1 - c0))

    scores = np.correlate(counts, kern, mode='same')
    best = int(np.argmax(scores))
    if 0 < best < slots - 1:
        y0, y1, y2 = scores[best - 1], scores[best], scores[best + 1]
        denominator = y0 - 2 * y1 + y2
        best_f = best + (0.5 * (y0 - y2) / denominator if denominator < 0 else 0.0)
    else:
        best_f = float(best)
    shift = -SPAN + (best_f + 0.5) * BIN
    if debug is not None:
        debug.update(counts=counts, scores=scores, shift=shift, a=a, b=b)

    # Quality: how much of the face evidence sits on wood, and how much better
    # the peak is than any peak a whole pocket pitch away (the "wrong period"
    # failure a sliding template can have).
    wood = kern > 0
    half = len(kern) // 2
    lo, hi = max(best - half, 0), min(best + half + 1, slots)
    klo = lo - (best - half)
    seg, kseg = counts[lo:hi], kern[klo:klo + (hi - lo)]
    wood_frac = float(seg[kseg > 0].sum() / max(seg.sum(), 1.0))
    far = np.abs(np.arange(slots) - best) * BIN > 0.10
    second = float(scores[far].max()) if far.any() else 0.0
    quality = {'wood_frac': wood_frac,
               'peak_ratio': second / scores[best] if scores[best] > 0 else float('inf'),
               'points': int(counts.sum()), 'score': float(scores[best])}

    def place(offset: float) -> np.ndarray:
        x, y = prior.centre + offset * t + (a + b * offset) * n
        fn, fo = prior.floor.normal, prior.floor.offset
        zc = (geo.band_centre - fo - fn[0] * x - fn[1] * y) / fn[2]
        return np.array([x, y, zc])

    face_normal = (n - b * t) / math.sqrt(1.0 + b * b)
    return Result(True, mid=place(shift),
                  pockets=np.array([place(shift + geo.pocket_spacing / 2),
                                    place(shift - geo.pocket_spacing / 2)]),
                  yaw=math.atan2(face_normal[1], face_normal[0]),
                  quality=quality, seconds=time.perf_counter() - start,
                  window_pixels=(r1 - r0) * (c1 - c0))


def prior_from(estimate) -> Prior:
    """A tracker's prior: the previous estimate's pocket midpoint (or, with no
    pockets, its face centre) and face normal."""
    mid = (np.mean([p.center[:2] for p in estimate.pockets], axis=0)
           if estimate.pockets else np.asarray(estimate.position[:2], dtype=float))
    n = np.asarray(estimate.normal[:2], dtype=float)
    return Prior(mid, n / np.linalg.norm(n), estimate.floor)


def prior_from_result(result: Result, previous: Prior) -> Prior:
    yaw = result.yaw
    return Prior(result.mid[:2], np.array([math.cos(yaw), math.sin(yaw)]), previous.floor)


def inject_load(depth, intr, to_ground, prior: Prior, geo: Geometry,
                overhang: float, top: float) -> np.ndarray:
    """A synthetic load: a flat vertical front on the deck, from the deck top up
    to `top` metres above the floor, `overhang` metres in front of the pallet
    face and as wide as the face. Written into a COPY of a real depth frame, so
    the pallet, the floor and the noise are all real; only the load is not.
    """
    out = depth.copy()
    n, t = prior.normal, prior.tangent
    fn, fo = prior.floor.normal, prior.floor.offset
    to_cam = np.linalg.inv(to_ground)
    # The load front is the vertical plane  n . (p_xy - centre) = overhang.
    # In the camera frame: plane normal and offset from the ground-frame ones.
    n3 = np.array([n[0], n[1], 0.0])
    point = np.array([prior.centre[0] + overhang * n[0], prior.centre[1] + overhang * n[1], 0.0])
    r_cam = to_cam[:3, :3]
    n_cam = r_cam @ n3
    p_cam = r_cam @ point + to_cam[:3, 3]
    d0 = float(n_cam @ p_cam)
    rows, cols = depth.shape
    v, u = np.mgrid[0:rows, 0:cols]
    ray = np.stack(((u - intr['cx']) / intr['fx'], (v - intr['cy']) / intr['fy'],
                    np.ones_like(u, dtype=float)), axis=-1)
    denominator = ray @ n_cam
    with np.errstate(divide='ignore', invalid='ignore'):
        hit = d0 / denominator
    ok = np.isfinite(hit) & (hit > 0.3) & (hit < 5.0)
    p_cam_hit = ray * hit[..., None]
    ground = p_cam_hit @ to_ground[:3, :3].T + to_ground[:3, 3]
    height = prior.floor.height_of(ground.reshape(-1, 3)).reshape(rows, cols)
    along = (ground[..., :2] - prior.centre) @ t
    inside = (ok & (height >= geo.deck) & (height <= top)
              & (np.abs(along) <= geo.face_width / 2))
    closer = inside & (~np.isfinite(out) | (hit < out))
    out[closer] = hit[closer]
    return out


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def pairs_of(estimate):
    return estimate is not None and len(estimate.pockets) == 2


def stat(values, fn) -> str:
    values = np.asarray(values, dtype=float)
    return 'n/a' if values.size == 0 else f'{fn(values):.1f}'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('bag', type=pathlib.Path)
    ap.add_argument('--frames', type=int, default=280)
    ap.add_argument('--step', type=int, default=4, help='use every Nth frame')
    ap.add_argument('--slab', type=float, default=0.02,
                    help='half-thickness of the face slab that is counted, metres')
    ap.add_argument('--plot', type=pathlib.Path,
                    help='save the along-the-face point counts and the fitted template here')
    ap.add_argument('--borrow-tf-from', type=pathlib.Path,
                    help='take missing static transforms from this bag (an assumption)')
    args = ap.parse_args()

    params = bag.load_params(HERE.parent)
    geo = Geometry.from_params(params)
    kern = kernel(geo)
    data = bag.read(args.bag, max_frames=args.frames)
    if args.borrow_tf_from:
        donor = bag.read(args.borrow_tf_from, max_frames=1)['tree']
        for child, parent in donor.parent.items():
            if child not in data['tree'].parent:
                data['tree'].add(parent, child, donor.edge[child])
    frame = data['depth_frame'] or bag.CAMERA
    to_ground = data['tree'].lookup(bag.GROUND, frame)
    c = params['extrinsic_correction']
    if c.get('enabled'):
        to_ground = correct_transform(to_ground, correction_matrix(
            c['measured_floor_pitch'], c['measured_floor_roll']))
    intr = data['intrinsics']
    frames = data['depth'][::args.step]
    print(f'{args.bag.name}: {len(frames)} frames (every {args.step}th of '
          f'{len(data["depth"])}), fx {intr["fx"]:.1f} px')

    # ---------------------------------------------------------- baseline
    base, base_ms = [], []
    for depth in frames:
        t0 = time.perf_counter()
        base.append(P.estimate_pallet(depth, intr, to_ground, params))
        base_ms.append((time.perf_counter() - t0) * 1000)
    ref = [i for i, e in enumerate(base) if pairs_of(e)]
    usable = [i for i, e in enumerate(base) if e is not None]
    print(f'\nBASELINE (current pipeline): found {len(usable)}'
          f', two pockets {len(ref)} of {len(frames)}; median {np.median(base_ms):.1f} ms/frame')
    if not usable:
        return 1
    if not ref:
        print('  (no frame has two pockets: the tracker starts from the baseline\'s FACE, '
              'and the comparison against baseline pockets is skipped)')
    start = (ref or usable)[0]

    # ---------------------------------------- 1. tracking, same frames
    prior = prior_from(base[start])
    track(frames[start], intr, to_ground, prior, geo, kern)        # warm-up, untimed
    rows, track_ms, window_px = [], [], []
    for i in range(len(frames)):
        res = track(frames[i], intr, to_ground, prior, geo, kern, slab=args.slab)
        track_ms.append(res.seconds * 1000)
        window_px.append(res.window_pixels)
        rows.append((i, res))
        if res.ok:
            prior = prior_from_result(res, prior)
    ok_rows = [(i, r) for i, r in rows if r.ok]
    print(f'\nTRACKER (prototype, prior = last result): ok {len(ok_rows)} of {len(rows)} frames; '
          f'median {np.median(track_ms):.1f} ms/frame '
          f'(window {np.median(window_px) / frames[0].size * 100:.1f} % of the image)')
    pair = [(i, r) for i, r in ok_rows if pairs_of(base[i])]
    if pair:
        dx = np.array([(r.mid - np.mean([p.center for p in base[i].pockets], axis=0))
                       for i, r in pair]) * 1000
        yaw = np.degrees([wrap(r.yaw - base[i].yaw) for i, r in pair])
        left = np.array([(r.pockets[0] - base[i].pockets[0].center) for i, r in pair]) * 1000
        right = np.array([(r.pockets[1] - base[i].pockets[1].center) for i, r in pair]) * 1000
        print(f'  vs baseline on {len(pair)} frames: midpoint x {np.median(dx[:, 0]):+.1f} mm '
              f'(median), y {np.median(dx[:, 1]):+.1f} mm; |diff| p95 x '
              f'{np.percentile(np.abs(dx[:, 0]), 95):.1f}, y {np.percentile(np.abs(dx[:, 1]), 95):.1f} mm')
        print(f'  yaw difference median {np.median(yaw):+.2f} deg, p95 |diff| '
              f'{np.percentile(np.abs(yaw), 95):.2f} deg')
        print(f'  pocket-centre y difference: left {np.median(left[:, 1]):+.1f}, '
              f'right {np.median(right[:, 1]):+.1f} mm (median)')
        mids_p = np.array([r.mid for _, r in pair]) * 1000
        mids_b = np.array([np.mean([p.center for p in base[i].pockets], axis=0)
                           for i, _ in pair]) * 1000
        yaw_p = np.degrees([r.yaw for _, r in pair])
        yaw_b = np.degrees([base[i].yaw for i, _ in pair])
        print(f'  frame-to-frame scatter (std): midpoint x/y  baseline '
              f'{mids_b[:, 0].std():.1f}/{mids_b[:, 1].std():.1f} mm   prototype '
              f'{mids_p[:, 0].std():.1f}/{mids_p[:, 1].std():.1f} mm;  yaw  baseline '
              f'{np.degrees(np.std([wrap(math.radians(y) - math.radians(yaw_b[0])) for y in yaw_b])):.2f}'
              f'   prototype {np.degrees(np.std([wrap(math.radians(y) - math.radians(yaw_p[0])) for y in yaw_p])):.2f} deg')
        q = np.array([[r.quality['wood_frac'], r.quality['peak_ratio'], r.quality['points']]
                      for _, r in pair])
        print(f'  quality at the true pose: wood fraction {q[:, 0].min():.2f}..{q[:, 0].max():.2f}, '
              f'peak ratio {q[:, 1].min():.2f}..{q[:, 1].max():.2f}, '
              f'points {int(q[:, 2].min())}..{int(q[:, 2].max())}')

    if args.plot is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        picks = [i for i, r in rows if r.ok][:: max(len(ok_rows) // 3, 1)][:3]
        fig, axes = plt.subplots(len(picks), 1, figsize=(9, 2.6 * len(picks)), squeeze=False)
        for ax, i in zip(axes[:, 0], picks):
            dbg = {}
            track(frames[i], intr, to_ground, prior_from(base[i]) if base[i] else prior, geo, kern, dbg, slab=args.slab)
            if not dbg:
                continue
            axis = -SPAN + (np.arange(len(dbg['counts'])) + 0.5) * BIN
            ax.bar(axis * 100, dbg['counts'], width=BIN * 100, color='0.55')
            top = max(dbg['counts'].max(), 1)
            s = dbg['shift']
            for sign in (-1, 1):
                lo = s + sign * geo.pocket_spacing / 2 - geo.pocket_width / 2
                ax.axvspan(lo * 100, (lo + geo.pocket_width) * 100, color='tab:red', alpha=0.25)
            ax.axvline(s * 100, color='tab:red', lw=0.8)
            ax.set_title(f'frame {i}: points on the face by position along it '
                         f'(red = fitted pockets, centre {s * 100:+.1f} cm)', fontsize=9)
            ax.set_xlabel('cm along the face, + = robot\'s left')
            ax.set_ylabel('points')
        fig.tight_layout()
        fig.savefig(args.plot, dpi=110)
        print(f'\nwrote {args.plot}')

    # --------------------------------- 2. how wrong may the prior be?
    pool = ref or usable
    probe = [i for i in pool[:: max(len(pool) // 4, 1)]][:4]
    print('\nPRIOR ERROR: does the tracker still land on the same answer as with the right prior?'
          f'\n  (4 frames; "same" = within 20 mm of the unperturbed tracker answer)')
    cases = ([('lateral', dy, 0.0, 0.0) for dy in (-0.30, -0.20, -0.10, 0.10, 0.20, 0.30)]
             + [('range', 0.0, dx, 0.0) for dx in (-0.30, -0.15, -0.10, -0.06, -0.03, 0.03, 0.06, 0.10, 0.15, 0.30)]
             + [('yaw deg', 0.0, 0.0, dyaw) for dyaw in (-15, -10, -5, 5, 10, 15)])
    for label, dy, dx, dyaw in cases:
        landed = 0
        for i in probe:
            p0 = prior_from(base[i])
            truth = track(frames[i], intr, to_ground, p0, geo, kern)
            c = math.cos(math.radians(dyaw)); s = math.sin(math.radians(dyaw))
            n2 = np.array([c * p0.normal[0] - s * p0.normal[1], s * p0.normal[0] + c * p0.normal[1]])
            bad = Prior(p0.centre + np.array([dx, dy]), n2, p0.floor)
            res = track(frames[i], intr, to_ground, bad, geo, kern)
            if truth.ok and res.ok and np.linalg.norm(res.mid[:2] - truth.mid[:2]) < 0.020:
                landed += 1
        value = dy if label == 'lateral' else dx if label == 'range' else dyaw
        print(f'  {label:8s} {value:+6.2f}: same answer on {landed} of {len(probe)} frames')

    # ------------------------------------------- 3. a synthetic load
    print('\nLOADED PALLET (synthetic load written into real frames; the tracker is compared with its own answer on the unloaded frame)')
    for name, overhang, top in (('flush, 0.5 m tall', 0.0, 0.5),
                                ('overhanging 8 cm, 0.5 m tall', 0.08, 0.5),
                                ('flush, 0.3 m tall', 0.0, 0.3),
                                ('flush, 0.18 m tall', 0.0, 0.18)):
        b_found = b_pair = t_ok = 0
        err = []
        for i in probe:
            p0 = prior_from(base[i])
            clean = track(frames[i], intr, to_ground, p0, geo, kern)
            loaded = inject_load(frames[i], intr, to_ground, p0, geo, overhang, top)
            e = P.estimate_pallet(loaded, intr, to_ground, params)
            b_found += e is not None
            b_pair += pairs_of(e)
            res = track(loaded, intr, to_ground, p0, geo, kern)
            if res.ok and clean.ok:
                t_ok += 1
                err.append(np.linalg.norm(res.mid[:2] - clean.mid[:2]) * 1000)
        print(f'  {name:30s} baseline: found {b_found}/{len(probe)}, two pockets {b_pair}/{len(probe)}'
              f'  |  tracker: ok {t_ok}/{len(probe)}, answer moved by {stat(err, np.median)} mm vs no load')

    # ------------------------------ 4. nothing there: does it say so?
    print('\nNOTHING THERE: the same tracker with a prior moved off the pallet '
          '(whatever else is in the room)')
    positives = [track(frames[i], intr, to_ground, prior_from(base[i]), geo, kern) for i in probe]
    pos = [(r.quality['wood_frac'], r.quality['peak_ratio']) for r in positives if r.ok]
    neg_ok, neg_reject, neg = 0, 0, []
    for i in probe:
        p0 = prior_from(base[i])
        for dy in (-1.2, -0.9, 0.9, 1.2):
            for dx in (-0.4, 0.0, 0.4):
                res = track(frames[i], intr, to_ground,
                            Prior(p0.centre + np.array([dx, dy]), p0.normal, p0.floor), geo, kern)
                if res.ok:
                    neg_ok += 1
                    neg.append((res.quality['wood_frac'], res.quality['peak_ratio']))
                else:
                    neg_reject += 1
    print(f'  rejected outright (too few points / outside image): {neg_reject} of {neg_ok + neg_reject}')
    if pos:
        print(f'  with the pallet:    wood fraction {min(p[0] for p in pos):.2f}..{max(p[0] for p in pos):.2f}, '
              f'peak ratio {min(p[1] for p in pos):.2f}..{max(p[1] for p in pos):.2f}')
    if neg:
        print(f'  without the pallet: wood fraction {min(p[0] for p in neg):.2f}..{max(p[0] for p in neg):.2f}, '
              f'peak ratio {min(p[1] for p in neg):.2f}..{max(p[1] for p in neg):.2f}  ({len(neg)} windows that returned a result)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
