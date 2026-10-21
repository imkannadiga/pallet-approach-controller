"""Perception's output rate and latency (now minus the frame stamp), sim time.

    python3 estimate_timing.py [seconds]
Needs perception active, i.e. a run in progress.
"""
import sys
import time

import rclpy
from rclpy.parameter import Parameter

from pallet_approach_interfaces.msg import PalletEstimate

rclpy.init()
node = rclpy.create_node('estimate_timing', parameter_overrides=[
    Parameter('use_sim_time', value=True)])
samples = []


def on_estimate(msg):
    now = node.get_clock().now().nanoseconds * 1e-9
    stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    samples.append((now, now - stamp, msg.valid))


node.create_subscription(PalletEstimate, '/pallet_perception/estimate', on_estimate, 10)
t0 = time.monotonic()
while time.monotonic() - t0 < float(sys.argv[1] if len(sys.argv) > 1 else 10):
    rclpy.spin_once(node, timeout_sec=0.1)
if len(samples) < 2:
    print(f'{len(samples)} estimates')
    sys.exit(1)
gaps = [b[0] - a[0] for a, b in zip(samples, samples[1:])]
lat = sorted(s[1] for s in samples)
print(f'{len(samples)} estimates, {len(samples) / (samples[-1][0] - samples[0][0]):.2f} Hz sim, '
      f'gap max {max(gaps):.2f} s, latency median {lat[len(lat) // 2]:.2f} max {lat[-1]:.2f} s, '
      f'valid {sum(s[2] for s in samples)}/{len(samples)}')
