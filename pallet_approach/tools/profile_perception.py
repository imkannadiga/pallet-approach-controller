"""Where a perception frame's time goes, stage by stage, on a real bag frame.

    python3 tools/profile_perception.py ~/bags/pallet_2m_0deg [--runs 20] [--threads 1]

Run it on the Jetson (inside the container) for the numbers that matter: a
laptop core is roughly twice an Orin A78AE core. `--threads 1` pins NumPy's
BLAS to one thread, which is what each node effectively gets when the whole
stack is running.
"""
import argparse
import os
import pathlib
import sys

ap = argparse.ArgumentParser()
ap.add_argument('bag', type=pathlib.Path)
ap.add_argument('--runs', type=int, default=20)
ap.add_argument('--threads', type=int, default=0)
args = ap.parse_args()
if args.threads:
    for var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[var] = str(args.threads)

import time  # noqa: E402
from collections import defaultdict  # noqa: E402

import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import _bagutil as bag  # noqa: E402
from pallet_approach import perception as P  # noqa: E402

STAGES = ('depth_to_points', 'transform_points', 'crop_to_roi', 'voxel_downsample',
          'fit_plane_ransac', 'remove_floor', 'cluster_points',
          'select_pallet_cluster', 'fit_front_face', 'face_extent', 'find_pockets',
          'classify_side')
spent = defaultdict(float)
for name in STAGES:
    original = getattr(P, name)

    def timed(*a, _f=original, _n=name, **k):
        t = time.perf_counter()
        try:
            return _f(*a, **k)
        finally:
            spent[_n] += time.perf_counter() - t
    setattr(P, name, timed)

params = bag.load_params(HERE.parent)
data = bag.read(args.bag, max_frames=1)
depth = data['depth'][0]
to_ground = bag.camera_to_ground(data, params['extrinsic_correction'])

P.estimate_pallet(depth, data['intrinsics'], to_ground, params)   # warm-up
spent.clear()
totals = []
for _ in range(args.runs):
    t = time.perf_counter()
    est = P.estimate_pallet(depth, data['intrinsics'], to_ground, params)
    totals.append(time.perf_counter() - t)
totals = np.array(totals) * 1000
print(f'{depth.shape[1]}x{depth.shape[0]} depth, {args.runs} runs: median '
      f'{np.median(totals):.1f} ms, p90 {np.percentile(totals, 90):.1f} ms; '
      f'pallet {"found" if est is not None else "NOT found"}')
for name, s in sorted(spent.items(), key=lambda kv: -kv[1]):
    print(f'  {name:22s} {s / args.runs * 1000:6.1f} ms')
