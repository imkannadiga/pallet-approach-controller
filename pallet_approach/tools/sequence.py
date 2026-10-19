#!/usr/bin/env python3
"""Run a whole virtual pickup offline and print what happened.

The offline replay CLAUDE.md asks for before anything drives, as a command you
can run in a second:

    python3 tools/sequence.py                       # a clean run from 3.5 m
    python3 tools/sequence.py --offset 0.4          # off the centre line
    python3 tools/sequence.py --range 2.5           # too close; it reverses
    python3 tools/sequence.py --noise 0.01 --seeds 10
    python3 tools/sequence.py --yaw 15              # the pallet turned
    python3 tools/sequence.py --no-lidar            # the insertion refuses
    python3 tools/sequence.py --break-gain          # a sign error in the servo
    python3 tools/sequence.py --mode approach_only  # stop at the decision

It drives the real state machine, the real tracker, the real control laws, the
real fork simulator and the real GO/NO-GO rules through `test/simulate.py`'s
world. What it does NOT exercise is the ROS layer -- discovery, QoS, lifecycle,
tf and the clock -- which is where every failure on the real robot has come
from so far. A clean run here is necessary and not sufficient.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
PACKAGE = HERE.parent
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(PACKAGE / 'test'))

import sequence as Q  # noqa: E402
import simulate as S  # noqa: E402


def one(args, seed: int):
    params = Q.load()
    if args.break_gain:
        params['drive']['closing']['k_cross'] *= -1
    pallet = S.Pallet(yaw=math.radians(args.yaw))
    rig = Q.Rig(params, pallet=pallet, noise=args.noise, seed=seed,
                lidar=not args.no_lidar)
    robot = S.on_line(pallet, args.range, args.offset,
                      math.radians(args.yaw_error))
    return rig.run(robot, mode=args.mode)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--range', type=float, default=3.5,
                        help='starting face distance, m (default 3.5)')
    parser.add_argument('--offset', type=float, default=0.10,
                        help='starting offset from the pocket centre line, m')
    parser.add_argument('--yaw-error', type=float, default=0.0,
                        help="the robot's starting yaw error, degrees")
    parser.add_argument('--yaw', type=float, default=0.0,
                        help='how far the PALLET is turned, degrees')
    parser.add_argument('--noise', type=float, default=0.0,
                        help='per-frame position noise, m (try 0.005)')
    parser.add_argument('--seeds', type=int, default=1,
                        help='how many runs to do, with different noise draws')
    parser.add_argument('--no-lidar', action='store_true',
                        help='no LiDAR at all, to see the insertion refuse')
    parser.add_argument('--break-gain', action='store_true',
                        help='flip the servo gain, to see the divergence abort')
    parser.add_argument('--mode', default='pickup',
                        choices=('pickup', 'approach_only'),
                        help='approach_only stops at the GO/NO-GO decision')
    parser.add_argument('--quiet', action='store_true',
                        help='one line per run instead of the full trace')
    args = parser.parse_args()

    outcomes = {}
    for seed in range(args.seeds):
        run = one(args, seed)
        outcomes[run.outcome] = outcomes.get(run.outcome, 0) + 1
        if args.quiet:
            print(f'seed {seed:>3}  {run.outcome:<9} '
                  f'{abs(run.offset) * 1000:5.1f} mm  '
                  f'{math.degrees(run.heading):+5.2f} deg  '
                  f'{run.seconds:5.1f} s  {run.reason[:60]}')
        else:
            if args.seeds > 1:
                print(f'--- seed {seed} ---')
            print(run.describe())
            print(f'{"":8}  arrived {abs(run.offset) * 1000:.1f} mm off the line, '
                  f'{math.degrees(run.heading):+.2f} deg out of square')
            if run.insertion_depth:
                print(f'{"":8}  inserted {run.insertion_depth:.3f} m, '
                      f'forks {"loaded" if run.loaded else "empty"}')
            print()

    if args.seeds > 1:
        print('outcomes: ' + ', '.join(f'{count} x {name}'
                                       for name, count in sorted(outcomes.items())))
    wanted = 'GO' if args.mode == 'approach_only' else 'INSERTED'
    return 0 if wanted in outcomes else 1


if __name__ == '__main__':
    sys.exit(main())
