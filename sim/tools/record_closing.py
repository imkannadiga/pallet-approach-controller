"""Record what the closing loop sees and commands, for a whole run.

    python3 record_closing.py out.csv [seconds]
Run it in the stack container alongside a run (run_sim.sh run ... in another
shell). Columns: sim time, commanded linear and angular (/cmd_vel_pallet), and
the controller's own range, lateral offset and heading error.
"""
import csv
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.parameter import Parameter

from pallet_approach_interfaces.msg import DriveStatus

rclpy.init()
node = rclpy.create_node('record_closing', parameter_overrides=[
    Parameter('use_sim_time', value=True)])
rows, status = [], [None]
node.create_subscription(DriveStatus, '/approach_controller/status',
                         lambda m: status.__setitem__(0, m), 10)


def on_cmd(msg):
    s = status[0]
    if s is None or s.mode != 'CLOSE':
        return
    t = node.get_clock().now().nanoseconds * 1e-9
    rows.append((round(t, 3), round(msg.linear.x, 4), round(msg.angular.z, 4),
                 round(s.face_range, 4), round(s.lateral_offset, 4),
                 round(s.heading_error, 4), s.phase))


node.create_subscription(Twist, '/cmd_vel_pallet', on_cmd, 10)
t0 = time.monotonic()
while time.monotonic() - t0 < float(sys.argv[2] if len(sys.argv) > 2 else 120):
    rclpy.spin_once(node, timeout_sec=0.05)
with open(sys.argv[1], 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['t', 'linear', 'angular', 'range', 'offset', 'heading', 'phase'])
    w.writerows(rows)
print(f'{len(rows)} rows -> {sys.argv[1]}')
