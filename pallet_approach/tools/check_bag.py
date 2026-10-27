"""Is this bag usable? Run it right after each recording, while the pallet is still there.

    python3 tools/check_bag.py ~/bags/board_205 [--needs depth,lidar]
    python3 tools/check_bag.py ~/bags/pallet_s_129 --moving     # a run-up, not a static bag

Written after `pallet_s_pos1_cond1` arrived without static transforms and the
problem was found afterwards, when the pallet was gone. Everything here is a
yes/no that can be fixed on the spot by recording again.

Checks: duration; messages and rate per topic; depth encoding and the fraction
of valid pixels; the LiDAR cloud (relay or raw topic) and its rate; static tf
present and `base_footprint` reachable from the camera and the LiDAR; odometry
(a static bag must not move); and whether a ground-truth row exists in
docs/bag_truth.yaml. Exit status 1 if anything FAILs.
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
from rosbags.highlevel import AnyReader  # noqa: E402

LIDAR_TOPICS = ('/pallet_approach/lidar', '/ouster/points')
RESULTS = []


def say(level: str, text: str) -> None:
    RESULTS.append(level)
    print(f'  [{level:4s}] {text}')


def odometry_path(path: pathlib.Path) -> tuple[float, float, int]:
    """Total path length and net displacement (metres) from /odom."""
    xs = []
    with AnyReader([path], default_typestore=bag.typestore()) as reader:
        conns = [c for c in reader.connections if c.topic == '/odom']
        for conn, _, raw in reader.messages(connections=conns):
            p = reader.deserialize(raw, conn.msgtype).pose.pose.position
            xs.append((p.x, p.y))
    if len(xs) < 2:
        return 0.0, 0.0, len(xs)
    pts = np.array(xs)
    return (float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()),
            float(np.linalg.norm(pts[-1] - pts[0])), len(xs))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('bag', type=pathlib.Path)
    ap.add_argument('--needs', default='depth,lidar',
                    help='comma list of what must be present: depth, lidar, image')
    ap.add_argument('--moving', action='store_true',
                    help='the robot is expected to move (a run-up); skip the stationary check')
    args = ap.parse_args()
    needs = {n.strip() for n in args.needs.split(',') if n.strip()}
    path = args.bag
    print(f'{path.name}')

    with AnyReader([path], default_typestore=bag.typestore()) as reader:
        duration = reader.duration / 1e9
        conns = {c.topic: c for c in reader.connections}
        counts = {t: c.msgcount for t, c in conns.items()}
    print(f'  duration {duration:.1f} s, {sum(counts.values())} messages')
    for topic, n in sorted(counts.items()):
        print(f'    {topic:55s} {n:5d} msgs  {n / duration if duration else 0:6.1f} Hz')
    if duration < 3:
        say('FAIL', f'only {duration:.1f} s recorded')

    # ---- depth
    depth_topic = next((t for t in conns if 'depth' in t and conns[t].msgtype.endswith('msg/Image')), None)
    if 'depth' in needs:
        if depth_topic is None:
            say('FAIL', 'no depth image topic')
        else:
            rate = counts[depth_topic] / duration
            say('ok' if rate >= 3 else 'FAIL', f'depth {rate:.1f} Hz ({counts[depth_topic]} frames)')
    if 'image' in needs and not any(conns[t].msgtype.endswith('msg/Image') and 'rect/image' in t for t in conns):
        say('FAIL', 'no colour image topic')

    # ---- LiDAR
    lidar_topic = next((t for t in LIDAR_TOPICS if t in conns), None)
    if 'lidar' in needs:
        if lidar_topic is None:
            say('FAIL', 'no LiDAR cloud (neither /pallet_approach/lidar nor /ouster/points)')
        else:
            rate = counts[lidar_topic] / duration
            n_clouds = counts[lidar_topic]
            # A moving bag needs the full rate. A static one only needs enough
            # clouds to average: 40 clouds at 6.8 Hz is fine for a flat board.
            if rate >= 7:
                level, note = 'ok', ''
            elif not args.moving and n_clouds >= 20:
                level, note = 'WARN', f' -- below 7 Hz but {n_clouds} clouds is enough for a static bag'
            else:
                level, note = 'FAIL', ' -- THROTTLED; check the relay and the network'
            say(level, f'LiDAR {rate:.1f} Hz on {lidar_topic}{note}')
            if lidar_topic == '/ouster/points':
                say('WARN', 'raw /ouster/points: fine only if this was recorded ON the robot PC; '
                            'from the Jetson it is a second network copy')

    # ---- tf, first frames
    try:
        data = bag.read(path, max_frames=1)
    except Exception as error:                                  # noqa: BLE001
        say('FAIL', f'cannot read the bag: {type(error).__name__}: {error}')
        data = None
    if data is not None:
        frame = data['depth_frame'] or bag.CAMERA
        borrowed = data.get('borrowed', [])
        if borrowed:
            say('ok', f'static tf: {len(borrowed)} edge(s) not in this bag, filled in from '
                      f'config/static_tf.yaml ({", ".join(borrowed[:4])}{"..." if len(borrowed) > 4 else ""}); '
                      f'fine unless the hardware was changed since it was exported')
        try:
            data['tree'].lookup(bag.GROUND, frame)
            say('ok', f'tf: {bag.GROUND} reachable from {frame}')
        except KeyError as error:
            say('FAIL', f'tf: {error} (the dynamic transforms are missing, or config/static_tf.yaml '
                        f'is absent or incomplete)')
        if data['intrinsics'] is None and 'depth' in needs:
            say('FAIL', 'no depth camera_info')
        if data['depth']:
            d = data['depth'][0]
            valid = np.isfinite(d) & (d > 0.3) & (d < 5.0)
            frac = float(valid.mean())
            say('ok' if frac > 0.4 else 'WARN',
                f'depth: {d.shape[1]}x{d.shape[0]}, {100 * frac:.0f}% valid pixels, '
                f'median {np.median(d[valid]):.2f} m' if valid.any() else 'depth: no valid pixels')
        if 'lidar' in needs and data['clouds']:
            frame_id, cloud = next(iter(data['clouds'].items()))
            try:
                data['tree'].lookup(bag.GROUND, frame_id)
                say('ok', f'tf: {bag.GROUND} reachable from the LiDAR frame {frame_id} '
                          f'({len(cloud)} points in the first cloud)')
            except KeyError as error:
                say('FAIL', f'tf (LiDAR): {error}')

    # ---- odometry
    if '/odom' not in counts:
        say('WARN', 'no /odom')
    else:
        travelled, net, n = odometry_path(path)
        if args.moving:
            say('ok' if travelled > 0.05 else 'WARN',
                f'odometry: moved {travelled:.2f} m ({net:.2f} m net) -- expected for a run-up')
        else:
            say('ok' if travelled < 0.02 else 'FAIL',
                f'odometry: moved {travelled * 1000:.0f} mm' + ('' if travelled < 0.02
                    else ' -- the robot moved during a bag that should be static'))

    # ---- ground truth
    truth_file = HERE.parent.parent / 'docs' / 'bag_truth.yaml'
    truth = yaml.safe_load(truth_file.read_text()) if truth_file.exists() else {}
    row = (truth or {}).get(path.name)
    if row is None:
        say('WARN', f'no row for {path.name} in docs/bag_truth.yaml -- write the tape readings now')
    elif row.get('measured') is True:
        say('ok', 'ground truth: measured')
    else:
        say('WARN', 'ground truth row exists but measured is not true (a target, not a reading)')

    fails = RESULTS.count('FAIL')
    print(f'  => {"RECORD AGAIN" if fails else "usable"} ({fails} fail, {RESULTS.count("WARN")} warn)')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
