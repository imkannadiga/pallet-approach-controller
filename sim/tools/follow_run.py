"""Start a run and print each coordinator state change until the result.

    python3 follow_run.py <mode> [timeout_s]

Runs inside the stack container (run_sim.sh run <mode>). Exit code 0 only if a
result arrived; the result line is the last line printed.
"""
import sys
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.parameter import Parameter
from std_msgs.msg import String

from pallet_approach_interfaces.action import PalletTask
from pallet_approach_interfaces.msg import CoordinatorStatus


def main():
    mode = sys.argv[1]
    timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 1800.0
    rclpy.init()
    node = rclpy.create_node('follow_run', parameter_overrides=[
        Parameter('use_sim_time', value=True)])
    last, result, odom = [None], [], [None]

    def on_status(m):
        if (m.state, m.reason) != last[0]:
            last[0] = (m.state, m.reason)
            o = odom[0]
            where = (f'({o.pose.pose.position.x:+.3f}, {o.pose.pose.position.y:+.3f})'
                     if o else '')
            print(f'[{m.elapsed:7.2f}s] {m.state:12s} {where} {m.reason}', flush=True)

    node.create_subscription(CoordinatorStatus, '/coordinator/status', on_status, 10)
    node.create_subscription(PalletTask.Result, '/pallet_approach/result', result.append, 10)
    node.create_subscription(Odometry, '/odom', lambda m: odom.__setitem__(0, m), 10)
    start = node.create_publisher(String, '/pallet_approach/start', 1)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 3.0:           # discovery before the trigger
        rclpy.spin_once(node, timeout_sec=0.1)
    start.publish(String(data=mode))
    while not result and time.monotonic() - t0 < timeout:
        rclpy.spin_once(node, timeout_sec=0.1)
    if not result:
        print('RESULT none (timed out)', flush=True)
        sys.exit(1)
    r = result[-1]
    p = r.pallet_pose.pose.position
    print(f'RESULT {r.outcome} mode={r.mode} side={r.side!r} reason={r.reason!r} '
          f'insertion={r.insertion_depth:.3f} lift={r.pallet_lift:.3f} '
          f'pallet=({p.x:.3f}, {p.y:.3f})', flush=True)


if __name__ == '__main__':
    main()
