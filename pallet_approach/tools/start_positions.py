"""The shared starting positions for the acceptance test (D96).

One seed gives the same N starting poses for the Gazebo sim and for the real
robot. Each pose is written two ways:

* the sim's `face_x face_y face_yaw` launch arguments, for `batch_runs.sh`;
* the tape readings for the real pallet: `dL`, `dR` and `lat`, in
  `docs/bag_truth.yaml`'s convention (metres, bumper to front corner, positive
  lat = pallet to the left).

The formula is the one `batch_runs.sh --random` used before D96, so seed 7
gives the same ten poses it always did.

Geometry, base_footprint frame (x forward, y left). The robot sits at the
origin. The face centre is at `(fx, fy)` and the face's normal points back at
the robot. Its yaw `fyaw` is the rotation of the face, so `fyaw = -psi`, where
`psi` is the heading error: the pallet appears turned by `psi`. `psi > 0`
turns the pallet's left edge away from the robot, so `dL > dR`, the dataset's
sign convention.

Usage:
    python3 tools/start_positions.py --seed 7 --count 10
    python3 tools/start_positions.py --seed 7 --csv docs/starts_seed7.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from dataclasses import dataclass

DISTANCE = 2.5
"""Metres from the robot origin (base_footprint) to the face centre. The
batch's old comment called this "from the face"; the number has always been
the origin distance, so it is kept as it was."""

MAX_OFFSET = 0.15
"""Metres, the largest sideways offset of the robot from the pallet's centre line."""

MAX_HEADING_DEG = 10.0
"""Degrees, the largest heading error either way."""

BUMPER_X = 0.4625
"""Metres, base_footprint to the front bumper (D23). Tape distances are
measured from the bumper, so `dL = corner_x - BUMPER_X`."""

HALF_FACE = 0.795 / 2
"""Metres, half the short (pocket) face, the one the acceptance test uses."""


@dataclass(frozen=True)
class StartPose:
    run: int
    offset: float
    """Robot's sideways offset from the pallet's centre line, metres."""
    heading_deg: float
    """Heading error, degrees; the pallet appears turned by this much."""
    face_x: float
    face_y: float
    face_yaw: float

    @property
    def d_left(self) -> float:
        """Tape: bumper to the pallet's left front corner, metres."""
        return self._corner(+1)[0] - BUMPER_X

    @property
    def d_right(self) -> float:
        """Tape: bumper to the pallet's right front corner, metres."""
        return self._corner(-1)[0] - BUMPER_X

    @property
    def lat(self) -> float:
        """Tape: the face centre's sideways offset, positive = pallet to the left."""
        return self.face_y

    def _corner(self, side: int) -> tuple[float, float]:
        # The face's tangent points left at yaw 0 and turns with the face.
        tx, ty = -math.sin(self.face_yaw), math.cos(self.face_yaw)
        return (self.face_x + side * HALF_FACE * tx,
                self.face_y + side * HALF_FACE * ty)


def start_pose(seed: int, run: int) -> StartPose:
    """The pose for run `run` (1-based) of `seed`. Same inputs, same output."""
    rng = random.Random(seed * 1000 + run)
    offset = rng.uniform(-MAX_OFFSET, MAX_OFFSET)
    psi = math.radians(rng.uniform(-MAX_HEADING_DEG, MAX_HEADING_DEG))
    d = DISTANCE
    face_x = d * math.cos(psi) - offset * math.sin(psi)
    face_y = -d * math.sin(psi) - offset * math.cos(psi)
    return StartPose(run=run, offset=offset, heading_deg=math.degrees(psi),
                     face_x=face_x, face_y=face_y, face_yaw=-psi)


def start_poses(seed: int, count: int) -> list[StartPose]:
    return [start_pose(seed, run) for run in range(1, count + 1)]


HEADER = ['run', 'face_x', 'face_y', 'face_yaw', 'dL', 'dR', 'lat',
          'offset_m', 'heading_deg']


def rows(poses: list[StartPose]) -> list[list]:
    return [[p.run, f'{p.face_x:.4f}', f'{p.face_y:.4f}', f'{p.face_yaw:.4f}',
             f'{p.d_left:.4f}', f'{p.d_right:.4f}', f'{p.lat:+.4f}',
             f'{p.offset:+.3f}', f'{p.heading_deg:+.1f}'] for p in poses]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--csv', help='also write the table to this file')
    args = parser.parse_args(argv)

    table = [HEADER] + rows(start_poses(args.seed, args.count))
    for line in table:
        print('  '.join(f'{str(cell):>10}' for cell in line))
    if args.csv:
        with open(args.csv, 'w', newline='') as handle:
            csv.writer(handle).writerows(table)
        print(f'wrote {args.csv}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
