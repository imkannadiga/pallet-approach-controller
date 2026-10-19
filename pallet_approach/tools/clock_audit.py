#!/usr/bin/env python3
"""Measure every publisher's clock against this machine's, and against each other.

Written after an afternoon of guessing at a timestamp problem from its
symptoms. The symptoms -- "extrapolation into the past", an absurd estimate
age, a ratio of 1.2458 derived by hand from two diagnostics messages -- each
narrowed it a little and none of them settled it. This settles it, by measuring
the one thing that matters directly: for every topic, how its header stamps
advance relative to wall time, and where they sit relative to the system clock.

Run it inside the container on the machine you care about:

    python3 tools/clock_audit.py                       # the usual suspects
    python3 tools/clock_audit.py --seconds 30
    python3 tools/clock_audit.py --topics /a /b /tf
    python3 tools/clock_audit.py --all                 # everything with a header

What to look for, in the slope column:

    1.000   healthy: that publisher's clock runs at the same rate as ours
    != 1    its clock runs at a different RATE -- not an offset, not drift.
            Nothing physical does this; it means the stamps are computed
            rather than read from a clock.

and in the offset column:

    ~0      same clock, same epoch
    large   different epoch (a steady/uptime clock, or sim time)

Two publishers can only be compared in a stamped tf lookup if BOTH columns
agree. A tf chain that mixes them cannot be resolved at any requested time,
which is exactly the failure this was written for.

ROS is needed, so this runs on the robot, not on a laptop.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, ReliabilityPolicy
    from rosidl_runtime_py.utilities import get_message
except ImportError:                                           # pragma: no cover
    sys.exit('needs ROS 2 on the path: run this inside the container')

DEFAULT_TOPICS = [
    '/zed2/zed_node/depth/depth_registered',
    '/zed2/zed_node/depth/camera_info',
    '/zed2/zed_node/rgb/color/rect/image',
    '/zed2/zed_node/imu/data',
    '/zed2/zed_node/odom',
    '/tf',
    '/tf_static',
    '/odom',
    '/ouster/points',
]


def stamps_in(msg):
    """Every header stamp a message carries, in seconds. Mostly one."""
    if hasattr(msg, 'transforms'):                 # tf2_msgs/TFMessage
        return [t.header.stamp.sec + t.header.stamp.nanosec * 1e-9
                for t in msg.transforms]
    header = getattr(msg, 'header', None)
    if header is None or not hasattr(header, 'stamp'):
        return []
    return [header.stamp.sec + header.stamp.nanosec * 1e-9]


def frames_in(msg):
    if hasattr(msg, 'transforms'):
        return [f'{t.header.frame_id}->{t.child_frame_id}' for t in msg.transforms]
    return ['']


def fit(xs, ys):
    """Least-squares slope and intercept of ys against xs, no numpy needed."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return float('nan'), float('nan')
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope, my - slope * mx


class ClockAudit(Node):

    def __init__(self, topics, per_frame):
        super().__init__('clock_audit')
        self.samples = defaultdict(list)           # key -> [(wall, stamp)]
        self.per_frame = per_frame
        self.start = time.time()
        available = dict(self.get_topic_names_and_types())
        self.subs = []
        for topic in topics:
            types = available.get(topic)
            if not types:
                print(f'  {topic}: not advertised, skipping')
                continue
            try:
                msg_type = get_message(types[0])
            except Exception as error:             # noqa: BLE001
                print(f'  {topic}: cannot load {types[0]} ({error})')
                continue
            # BEST_EFFORT deliberately: a BEST_EFFORT subscriber can receive
            # from a RELIABLE publisher, but not the other way round. Asking
            # for RELIABLE got "incompatible QoS ... No messages will be
            # received" from /ouster/points, which publishes BEST_EFFORT -- a
            # diagnostic tool that silently misses a topic is worse than no
            # tool. tf_static is additionally latched.
            qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT)
            if topic == '/tf_static':
                qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
            self.subs.append(self.create_subscription(
                msg_type, topic,
                lambda msg, t=topic: self._on(t, msg), qos))
            print(f'  {topic}: {types[0]}')

    def _on(self, topic, msg):
        wall = time.time()
        for stamp, frame in zip(stamps_in(msg), frames_in(msg)):
            if stamp == 0.0:
                continue                           # unstamped, nothing to compare
            key = f'{topic} [{frame}]' if (self.per_frame and frame) else topic
            self.samples[key].append((wall, stamp))


def report(samples, seconds):
    print(f'\n{"topic":<52} {"n":>6} {"slope":>8} {"offset s":>14} {"rate Hz":>8}')
    print('-' * 92)
    rows = []
    for key, pairs in sorted(samples.items()):
        if len(pairs) < 5:
            print(f'{key:<52} {len(pairs):>6}  too few samples')
            continue
        walls = [w for w, _ in pairs]
        stamps = [s for _, s in pairs]
        # A latched topic delivers its whole backlog the instant you subscribe,
        # so every sample shares a wall time while the stamps are from whenever
        # each was first published. Fitting stamps against wall time then
        # divides a real spread by nearly zero and reports an absurd rate --
        # this tool said /tf_static ran at 1354x before this check existed, and
        # that was the tool being wrong, not the robot.
        span = walls[-1] - walls[0]
        if span < 1.0:
            print(f'{key:<52} {len(pairs):>6} {"latched":>8} '
                  f'{stamps[-1] - walls[-1]:>14.3f} {"-":>8}'
                  f'  <- all arrived at once; rate not measurable, offset is')
            continue
        slope, _ = fit(walls, stamps)
        offset = stamps[-1] - walls[-1]
        rate = (len(pairs) - 1) / (walls[-1] - walls[0]) if walls[-1] > walls[0] else 0
        flag = ''
        if abs(slope - 1.0) > 0.001:
            flag = f'  <- RATE IS {slope:.4f}x, stamps are computed, not read'
        elif abs(offset) > 1.0:
            flag = '  <- different epoch (uptime or sim clock?)'
        print(f'{key:<52} {len(pairs):>6} {slope:>8.4f} {offset:>14.3f} {rate:>8.1f}{flag}')
        rows.append((key, slope, offset))
    print('-' * 92)

    healthy = [k for k, s, o in rows if abs(s - 1.0) <= 0.001 and abs(o) <= 1.0]
    broken = [(k, s, o) for k, s, o in rows if k not in healthy]
    print(f'\n{len(healthy)} topic(s) agree with this machine\'s clock.')
    if not broken:
        print('All of them. The clocks are not the problem.')
        return
    print(f'{len(broken)} do not:\n')
    for key, slope, offset in broken:
        if abs(slope - 1.0) > 0.001:
            print(f'  {key}\n    runs at {slope:.4f}x real time. A clock cannot do '
                  f'this -- an oscillator is wrong in parts per million, not '
                  f'{abs(slope - 1) * 100:.1f} %. Its stamps are being CALCULATED '
                  f'(frame count, tick counter, sample index) with a wrong '
                  f'divisor somewhere, not read from any clock.')
        else:
            print(f'  {key}\n    same rate, but {offset:+.0f} s away: a different '
                  f'epoch. Uptime clock or simulated time.')
    print('\nA stamped tf lookup can only join publishers that agree in BOTH')
    print('columns. If the broken ones include anything in the camera->base')
    print('chain, no requested time can satisfy it.')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seconds', type=float, default=15.0)
    ap.add_argument('--topics', nargs='*', default=None)
    ap.add_argument('--all', action='store_true',
                    help='every advertised topic, not just the usual suspects')
    ap.add_argument('--per-frame', action='store_true',
                    help='split /tf by parent->child, to see which broadcaster is which')
    args = ap.parse_args()

    rclpy.init()
    probe = Node('clock_audit_probe')
    time.sleep(1.0)                                # let discovery settle
    if args.all:
        topics = sorted(t for t, _ in probe.get_topic_names_and_types())
    else:
        topics = args.topics or DEFAULT_TOPICS
    probe.destroy_node()

    print('subscribing:')
    node = ClockAudit(topics, args.per_frame)
    if not node.subs:
        print('\nnothing to listen to')
        return 1
    print(f'\nlistening for {args.seconds:g} s...')
    end = time.time() + args.seconds
    while time.time() < end and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
    samples = dict(node.samples)
    node.destroy_node()
    rclpy.shutdown()
    report(samples, args.seconds)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
