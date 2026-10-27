#!/usr/bin/env python3
"""What can the Ouster actually see of a 145 mm pallet, and how close?

The LiDAR is the cross-check sensor for the pallet face (stage 2) and the
distance stop during insertion (stages 5-6). Both uses stand or fall on one
question the datasheet cannot answer: the lowest beams leave the sensor at a
fixed angle, the sensor sits 0.61 m up, and the robot's own body is in the way,
so there is a forward cone the LiDAR cannot see into at all. This measures it.

Nothing here is taken from the datasheet. The beam altitude angles are read off
a real cloud (the organised 64 x 1024 grid: one row per beam), and the beam
origin's height above the floor is *fitted* from where those beams land on a
bare floor, which is why a bare-floor bag is the required input -- the URDF
chain to `os_lidar` is 38 mm out (docs/SYSTEM.md) and using it would push every
number below in the same wrong direction.

    python3 pallet_approach/tools/lidar_coverage.py ~/bags/floor_lidar_pose1 \
                                                    ~/bags/floor_lidar_pose2

Needs: /pallet_approach/lidar (or /ouster/points) and /tf_static, recorded on bare floor.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _bagutil import GROUND, Tree, fill_static, quat_matrix, typestore  # noqa: E402
from pallet_approach.pointcloud import DATATYPES  # noqa: E402

try:
    from rosbags.highlevel import AnyReader
except ImportError:                                           # pragma: no cover
    sys.exit('pip install rosbags numpy pyyaml  (in a venv)')

# A bag from the Jetson carries the relay's copy; one from the robot PC, the raw
# topic. Same messages, same stamps. The relay is preferred when both exist.
CLOUD_TOPICS = ('/pallet_approach/lidar', '/ouster/points')

# [measured] docs/SYSTEM.md: the floor in base_footprint, from the camera at two
# headings. base_footprint itself is 25 mm above the floor, so this is not 0.
FLOOR_Z = -0.0246

# [measured] from the deployed URDF's base_link collision box, 0.925 x 0.38 x
# 0.21 centred on base_link: the front face is half the length ahead of
# base_footprint. This is the part of the robot that must not touch the pallet.
BUMPER_X = 0.4625

# Straight ahead, where the pallet is. Wide enough to hold ~57 of the 1024
# azimuth columns, narrow enough that the floor under it is flat.
AHEAD_DEG = 10.0

# The beam origin's height is fitted from the ground intercepts of this many of
# the steepest beams. A beam at 5 deg lands 7 m away, where floor slope and a
# long lever arm swamp the height it is meant to measure; the steep ones land
# under a metre out and measure it directly.
STEEPEST_FOR_HEIGHT = 8


def read_grid(bag: pathlib.Path, store: str) -> tuple[dict, np.ndarray, str]:
    """One organised cloud as {field: (64, 1024)}, plus base_footprint <- cloud."""
    tree, grid, frame = Tree(), None, ''
    with AnyReader([bag], default_typestore=typestore(store)) as reader:
        conns = list(reader.connections)
        for conn, _, raw in reader.messages(
                connections=[c for c in conns if c.topic in ('/tf', '/tf_static')]):
            for tr in reader.deserialize(raw, conn.msgtype).transforms:
                tree.add(tr.header.frame_id, tr.child_frame_id,
                         quat_matrix(tr.transform.translation, tr.transform.rotation))
        names = {c.topic for c in conns}
        cloud_topic = next((t for t in CLOUD_TOPICS if t in names), None)
        for conn in [c for c in conns if c.topic == cloud_topic]:
            for _, _, raw in reader.messages(connections=[conn]):
                msg = reader.deserialize(raw, conn.msgtype)
                if msg.height < 2:
                    sys.exit(f'{bag.name}: {cloud_topic} is not organised '
                             f'(height {msg.height}); per-beam angles need the grid')
                offsets = {f.name: (f.offset, DATATYPES[f.datatype])
                           for f in msg.fields}
                flat = np.frombuffer(bytes(msg.data), np.uint8).reshape(-1,
                                                                        msg.point_step)
                grid = {}
                for name in ('x', 'y', 'z', 'range'):
                    if name not in offsets:
                        continue
                    offset, dtype = offsets[name]
                    size = np.dtype(dtype).itemsize
                    grid[name] = (flat[:, offset:offset + size].copy().view(dtype)
                                  .ravel().reshape(msg.height, msg.width).astype(float))
                frame = msg.header.frame_id
                break
            break
    if grid is None:
        sys.exit(f'{bag.name}: none of {", ".join(CLOUD_TOPICS)}')
    fill_static(tree)          # rigid mounting the bag may have arrived without
    return grid, tree.lookup(GROUND, frame), frame


def beam_altitudes(grid: dict) -> np.ndarray:
    """Per-row altitude angle in degrees, in the sensor's own frame."""
    radius = np.hypot(grid['x'], grid['y'])
    elevation = np.degrees(np.arctan2(grid['z'], radius))
    valid = grid.get('range', radius) > 0
    out = np.full(grid['x'].shape[0], np.nan)
    for row in range(out.size):
        mask = valid[row] & (radius[row] > 0.1)
        if mask.sum() > 20:
            out[row] = np.median(elevation[row][mask])
    return out


def survey(bag: pathlib.Path, store: str) -> dict:
    """Beam angles, the fitted origin height, and the lowest beam usable ahead."""
    grid, transform, frame = read_grid(bag, store)
    altitude = beam_altitudes(grid)
    valid = grid.get('range', np.hypot(grid['x'], grid['y'])) > 0

    points = np.stack([grid['x'], grid['y'], grid['z']], -1) @ transform[:3, :3].T \
        + transform[:3, 3]
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    ahead = valid & (np.abs(np.degrees(np.arctan2(y, x))) < AHEAD_DEG) & (x > 0)
    on_floor = ahead & (np.abs(z - FLOOR_Z) < 0.06)

    lidar_x = float(transform[0, 3])
    intercepts, heights = {}, []
    for row in range(altitude.size):
        if on_floor[row].sum() < 20 or not np.isfinite(altitude[row]) \
                or altitude[row] >= 0:
            continue
        reach = float(np.median(x[row][on_floor[row]]))
        intercepts[row] = reach
        # h = (reach - lidar_x) * tan(|altitude|): the beam drops h over that run.
        heights.append((reach - lidar_x) * np.tan(np.radians(-altitude[row])))

    steep = sorted(intercepts, key=lambda r: intercepts[r])
    near = steep[:STEEPEST_FOR_HEIGHT]
    heights = [(intercepts[r] - lidar_x) * np.tan(np.radians(-altitude[r]))
               for r in near]
    return {
        'bag': bag.name, 'frame': frame, 'rows': altitude.size,
        'columns': grid['x'].shape[1], 'altitude': altitude,
        'lidar_x': lidar_x, 'urdf_z': float(transform[2, 3]),
        'height': float(np.mean(heights)), 'height_spread': float(np.std(heights)),
        'intercepts': intercepts, 'fitted_from': len(near),
        'lowest_row': steep[0] if steep else None,
        'blind_radius': intercepts[steep[0]] if steep else float('nan'),
        'ahead_columns': int(ahead[altitude.size // 2].sum()),
        'ahead_per_row': {r: int(ahead[r].sum()) for r in range(altitude.size)},
    }


def pallet_table(height: float, lidar_x: float, altitude: np.ndarray,
                 lowest_row: int, deck: float, pocket_top: float) -> None:
    """Which beams cross the pallet's front face, as the robot closes in."""
    usable = [(r, altitude[r]) for r in range(altitude.size)
              if np.isfinite(altitude[r]) and altitude[r] < 0 and r <= lowest_row]

    def beams_at(face_x: float) -> list[float]:
        out = []
        for _, alt in usable:
            h = height - (face_x - lidar_x) * np.tan(np.radians(-alt))
            if 0.0 <= h <= deck:
                out.append(h)
        return sorted(out, reverse=True)

    print(f'\n  {"face at x":>10} {"bumper gap":>11} {"beams":>6} {"in pockets":>11}'
          f'  heights (m)')
    for face_x in (2.5, 2.0, 1.7, 1.45, 1.2, 1.0, 0.9, 0.8, 0.75, 0.7, 0.666, 0.6):
        hs = beams_at(face_x)
        pockets = sum(1 for h in hs if h <= pocket_top)
        print(f'  {face_x:>10.3f} {face_x - BUMPER_X:>11.3f} {len(hs):>6} '
              f'{pockets:>11}  ' + ' '.join(f'{h:.3f}' for h in hs))

    # The deck is a horizontal target, so the lowest beam reaches further over it
    # than it does over the floor; that extra reach is what survives at the end
    # of insertion, when the face itself is already inside the blind cone.
    print('\n  last distances at which the face stays visible '
          '(x of base_footprint, then bumper gap):')
    for want in (1, 2, 3):
        limit = None
        for face_x in np.arange(0.4, 2.5, 0.001):
            if len(beams_at(float(face_x))) >= want:
                limit = float(face_x)
                break
        if limit is not None:
            print(f'    {want} beam{"s" if want > 1 else " "} on the face: '
                  f'x >= {limit:.3f} m  (gap {limit - BUMPER_X:.3f} m)')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bags', nargs='+', type=pathlib.Path,
                    help='bare-floor bags with the LiDAR cloud and /tf_static')
    ap.add_argument('--store', default='ROS2_HUMBLE')
    ap.add_argument('--package-root', type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parents[1],
                    help='for config/params.yaml (pallet geometry)')
    args = ap.parse_args()

    import yaml
    shared = yaml.safe_load(
        (args.package_root / 'config' / 'params.yaml').read_text())['/**']['ros__parameters']
    pallet = shared['pallet']
    deck = pallet['pallet_height']
    # The opening's top edge on the face we expect to meet. Per side since
    # 2026-10-14 (D19): the two faces of this pallet open to different heights,
    # 0.100 m on the short face and 0.125 m on the long one.
    pocket_top = pallet['sides'][pallet['entry_face']]['opening_top']

    results = [survey(bag, args.store) for bag in args.bags]

    for r in results:
        alt = r['altitude']
        finite = np.isfinite(alt)
        spacing = np.diff(np.sort(alt[finite]))
        print(f"\n=== {r['bag']}  ({r['frame']}, {r['rows']} beams x "
              f"{r['columns']} columns) ===")
        print(f"  vertical FOV   {np.nanmin(alt):+.2f} .. {np.nanmax(alt):+.2f} deg "
              f"({np.nanmax(alt) - np.nanmin(alt):.2f} deg over {finite.sum()} beams)")
        print(f"  beam spacing   {spacing.mean():.3f} deg mean, "
              f"{spacing.min():.3f} min, {spacing.max():.3f} max")
        print(f"  axis at x      {r['lidar_x']:+.3f} m in {GROUND}")
        print(f"  origin height  {r['height']:.4f} m above the floor "
              f"(fitted from the {r['fitted_from']} steepest of "
              f"{len(r['intercepts'])} ground intercepts, "
              f"spread {r['height_spread'] * 1000:.1f} mm)")
        print(f"                 the URDF chain claims "
              f"{r['urdf_z'] - FLOOR_Z:.4f} m -- "
              f"{(r['urdf_z'] - FLOOR_Z - r['height']) * 1000:+.0f} mm out")
        print(f"  lowest beam usable straight ahead: row {r['lowest_row']} at "
              f"{alt[r['lowest_row']]:+.2f} deg")
        print(f"  FORWARD BLIND RADIUS on the floor: "
              f"{r['blind_radius']:.3f} m from {GROUND}")
        blocked = [row for row in range(r['rows'])
                   if np.isfinite(alt[row]) and alt[row] < alt[r['lowest_row']]
                   and r['ahead_per_row'][row] < r['ahead_columns'] // 4]
        if blocked:
            print(f"  beams below it with no returns ahead (the robot's own body "
                  f"is in the way): rows {blocked}, "
                  f"{', '.join(f'{alt[b]:+.2f}' for b in blocked)} deg")

    base = results[0]
    if len(results) > 1:
        spread = [r['height'] for r in results]
        print(f"\nacross {len(results)} bags: origin height "
              f"{np.mean(spread):.4f} m +- {np.std(spread) * 1000:.1f} mm, "
              f"blind radius "
              f"{np.mean([r['blind_radius'] for r in results]):.3f} m "
              f"+- {np.std([r['blind_radius'] for r in results]) * 1000:.0f} mm")

    print(f"\n=== a {deck * 1000:.0f} mm pallet, pockets up to "
          f"{pocket_top * 1000:.0f} mm, seen from {base['bag']} ===")
    pallet_table(base['height'], base['lidar_x'], base['altitude'],
                 base['lowest_row'], deck, pocket_top)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
