#!/usr/bin/env python3
"""Timing and accuracy report for one recorded pickup, for slides.

    python3 tools/run_report.py ~/bags/robot_fused_run5 [--tz +0100] [--csv-dir DIR]

Reads the run's bag (.db3 directly; the stack's own topics plus odometry, tf,
the relayed LiDAR and the ZED's camera_info) and, if present beside it,
tegrastats.log. Prints the stage timeline, motion per stage, sensor rates and
ages, the fusion node's processing times, the end-to-end perception latency,
the estimate at the decision, and Jetson load per stage. Every number comes
from the recording; nothing is assumed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import re
import sqlite3

import numpy as np
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

MSGS = pathlib.Path(__file__).resolve().parents[2] / 'pallet_approach_interfaces' / 'msg'


def typestore():
    ts = get_typestore(Stores.ROS2_HUMBLE)
    add = {}
    for f in MSGS.glob('*.msg'):
        add.update(get_types_from_msg(f.read_text(), f'pallet_approach_interfaces/msg/{f.stem}'))
    # The result topic carries the action's Result message.
    for f in (MSGS.parent / 'action').glob('*.action'):
        goal, result, feedback = re.split(r'^---\s*$', f.read_text(), flags=re.M)
        for part, text in (('Goal', goal), ('Result', result), ('Feedback', feedback)):
            add.update(get_types_from_msg(text, f'pallet_approach_interfaces/action/{f.stem}_{part}'))
    ts.register(add)
    return ts


class Bag:
    def __init__(self, path: pathlib.Path):
        self.ts = typestore()
        self.con = sqlite3.connect(next(path.glob('*.db3')))
        self.topics = {r[1]: (r[0], r[2]) for r in self.con.execute('select id, name, type from topics')}

    def read(self, name):
        if name not in self.topics:
            return
        tid, typ = self.topics[name]
        for t, d in self.con.execute('select timestamp, data from messages where topic_id=? order by timestamp',
                                     (tid,)):
            yield t * 1e-9, self.ts.deserialize_cdr(d, typ)


def stamp(h) -> float:
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def stats(x) -> str:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return 'n/a'
    return f'median {np.median(x):7.1f}  p95 {np.percentile(x, 95):7.1f}  max {np.max(x):7.1f}'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('bag', type=pathlib.Path)
    ap.add_argument('--tz', default='+0100', help="the Jetson's UTC offset, for tegrastats.log")
    args = ap.parse_args()
    bag = Bag(args.bag)

    # ------------------------------------------------------------ timeline
    states = [(t, m.data) for t, m in bag.read('/coordinator/state')]
    status = [(t, m) for t, m in bag.read('/coordinator/status')]
    if not states:
        states, last = [], None
        for t, m in status:
            if m.state != last:
                states.append((t, m.state)); last = m.state
    run = [(t, s) for t, s in states]
    start = next(t for t, s in run if s == 'CONFIRMING')
    end = next((t for t, s in run if s in ('DONE', 'ABORTED', 'IDLE') and t > start), run[-1][0])
    print('== Timeline (coordinator state changes; trigger at 0.00 s)')
    odom = np.array([(stamp(m.header), m.pose.pose.position.x, m.pose.pose.position.y,
                      m.twist.twist.linear.x, m.twist.twist.angular.z) for _, m in bag.read('/odom')])
    segments = []
    for (t0, s0), (t1, _) in zip(run, run[1:] + [(end, 'END')]):
        if t0 < start or t0 >= end:
            continue
        segments.append((s0, t0, min(t1, end)))
    print(f'{"stage":12s} {"start s":>8s} {"dur s":>7s} {"share":>6s} {"driven m":>9s} {"mean |v|":>9s} {"max |v|":>8s} {"turned deg":>10s}')
    total = end - start
    for s0, t0, t1 in segments:
        m = (odom[:, 0] >= t0) & (odom[:, 0] <= t1)
        seg = odom[m]
        driven = float(np.sum(np.hypot(np.diff(seg[:, 1]), np.diff(seg[:, 2])))) if len(seg) > 1 else 0.0
        turned = float(np.degrees(np.sum(np.abs(seg[:-1, 4] * np.diff(seg[:, 0]))))) if len(seg) > 1 else 0.0
        v = np.abs(seg[:, 3]) if len(seg) else np.array([0.0])
        print(f'{s0:12s} {t0 - start:8.2f} {t1 - t0:7.2f} {100 * (t1 - t0) / total:5.1f}% {driven:9.3f} '
              f'{v.mean():9.3f} {v.max():8.3f} {turned:10.1f}')
    print(f'{"TOTAL":12s} {"":8s} {total:7.2f}')
    result = [m for _, m in bag.read('/pallet_approach/result')]
    if result:
        r = result[-1]
        print(f'result: {r.outcome}  side {r.side}  insertion {r.insertion_depth:.3f} m  lift {r.pallet_lift:.3f} m  '
              f'-- {r.reason}')

    # ----------------------------------------------------- sensors in, ages
    print('\n== Sensors: rate, and age on arrival at the Jetson (receive time - capture stamp, ms)')
    for name, label in (('/zed2/zed_node/depth/camera_info', 'ZED depth (camera_info)'),
                        ('/pallet_approach/lidar', 'Ouster LiDAR (relay)'),
                        ('/odom', 'wheel odometry')):
        rows = [(t, stamp(m.header)) for t, m in bag.read(name)]
        if not rows:
            continue
        r = np.array(rows)
        rate = (len(r) - 1) / (r[-1, 0] - r[0, 0])
        gaps = np.diff(r[:, 1])
        print(f'{label:26s} {rate:6.1f} Hz   age {stats(1000 * (r[:, 0] - r[:, 1]))}   '
              f'longest gap {1000 * gaps.max():6.0f} ms')

    # ------------------------------------------------- the fused perception
    print('\n== Fused perception (pallet_fusion)')
    tr = [(t, m) for t, m in bag.read('/pallet_fusion/tracked')]
    if tr:
        r = np.array([(t, stamp(m.header)) for t, m in tr])
        print(f'tracked pallet out: {(len(r) - 1) / (r[-1, 0] - r[0, 0]):.1f} Hz; '
              f'end-to-end age when it reaches the stack (capture -> fused estimate): {stats(1000 * (r[:, 0] - r[:, 1]))} ms')
    sts = [(t, json.loads(m.data)) for t, m in bag.read('/pallet_fusion/stats')]
    if sts:
        for key, label in (('lidar', 'LiDAR face fit per sweep'), ('camera', 'camera pockets per frame'),
                           ('acquisition', 'acquisition (pooled sweeps)')):
            vals = [d['ms'][key] for _, d in sts if d.get('ms', {}).get(key) is not None]
            print(f'  {label:30s} {stats(vals)} ms   (1 Hz samples of the latest call)')
        rng = [d.get('camera_minus_lidar_mm') for _, d in sts if start <= _ <= end]
        print(f'  camera - LiDAR range during the run: {stats([v for v in rng if v is not None])} mm')

    # ---------------------------------------------------- at the decision
    print('\n== At the decision (CHECKING)')
    chk = next((t for t, s in run if s == 'CHECKING'), None)
    if chk is not None and tr:
        near = [m for t, m in tr if chk - 0.5 <= t <= chk + 0.6 and m.estimate.valid]
        if near:
            e = near[-1].estimate
            print(f'  face {e.position.x:.3f} m ahead, lateral {1000 * e.position.y:+.0f} mm, '
                  f'yaw {np.degrees(e.yaw - np.pi * np.sign(e.yaw)):+.2f} deg (0 = square); '
                  f'cross-check {"agrees" if near[-1].lidar_agrees else "DISAGREES"} '
                  f'{1000 * near[-1].lidar_offset:+.1f} mm; {near[-1].lidar_detail}')
    for t, m in bag.read('/approach_controller/status'):
        pass

    # -------------------------------------------------------------- control
    print('\n== Control')
    for name, label in (('/cmd_vel_pallet', 'controller commands'), ('/cmd_vel', 'twist_mux output to the base'),
                        ('/approach_controller/drive', 'coordinator -> controller')):
        rows = [t for t, _ in bag.read(name)]
        if len(rows) > 1:
            d = np.diff(rows)
            print(f'  {label:30s} {(len(rows) - 1) / (rows[-1] - rows[0]):5.1f} Hz, longest gap {1000 * d.max():5.0f} ms')

    # ------------------------------------------------------------ Jetson load
    tg = args.bag / 'tegrastats.log'
    if tg.exists():
        print('\n== Jetson load per stage (tegrastats, 0.5 s samples)')
        tz = dt.timezone(dt.timedelta(hours=int(args.tz[:3]), minutes=int(args.tz[0] + args.tz[3:])))
        samples = []
        for line in tg.read_text().splitlines():
            m = re.match(r'(\d\d-\d\d-\d{4} \d\d:\d\d:\d\d) RAM (\d+)/(\d+)MB.*?CPU \[([^\]]*)\].*?GR3D_FREQ (\d+)%', line)
            if not m:
                continue
            t = dt.datetime.strptime(m.group(1), '%m-%d-%Y %H:%M:%S').replace(tzinfo=tz).timestamp()
            cores = [int(c.split('%')[0]) for c in m.group(4).split(',') if c != 'off']
            p = re.search(r'VIN_SYS_5V0 (\d+)mW', line)
            samples.append((t, np.mean(cores), max(cores), int(m.group(5)), int(m.group(2)),
                            int(p.group(1)) if p else np.nan, len(cores)))
        s = np.array(samples)
        print(f'  online cores: {int(s[0, 6])};  whole recording: CPU mean {s[:, 1].mean():.0f}%, '
              f'GPU {s[:, 3].mean():.0f}%, RAM {s[:, 4].mean() / 1024:.1f} GB')
        print(f'  {"stage":12s} {"CPU mean":>9s} {"CPU max core":>13s} {"GPU":>5s} {"RAM GB":>7s}')
        for s0, t0, t1 in [('before run', s[0, 0], start)] + segments:
            m = (s[:, 0] >= t0) & (s[:, 0] <= t1 + 0.5)
            if m.any():
                print(f'  {s0:12s} {s[m, 1].mean():8.0f}% {s[m, 2].max():12.0f}% {s[m, 3].mean():4.0f}% '
                      f'{s[m, 4].mean() / 1024:7.1f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
