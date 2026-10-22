"""Frame rates along the video path, and the sim's real-time factor.

    python3 stream_rates.py [seconds]
Run inside the stack container. Counts camera, fork_overlay and dashboard
frames over wall-clock time -- what a viewer actually sees.
"""
import sys
import time

import rclpy
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image

TOPICS = ('/zed2/zed_node/rgb/color/rect/image', '/fork_overlay/image', '/dashboard/image')
seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
rclpy.init()
node = rclpy.create_node('stream_rates')
counts, clock = {t: 0 for t in TOPICS}, []
for topic in TOPICS:
    node.create_subscription(Image, topic, lambda _m, t=topic: counts.__setitem__(t, counts[t] + 1),
                             qos_profile_sensor_data, raw=True)
node.create_subscription(Clock, '/clock', lambda m: clock.append(
    (time.time(), m.clock.sec + m.clock.nanosec * 1e-9)), 10)
start = time.time()
while time.time() - start < seconds:
    rclpy.spin_once(node, timeout_sec=0.05)
if len(clock) > 1:
    print(f'real-time factor {(clock[-1][1] - clock[0][1]) / (clock[-1][0] - clock[0][0]):.2f}')
for topic, n in counts.items():
    print(f'{topic}: {n / seconds:.1f} fps')
