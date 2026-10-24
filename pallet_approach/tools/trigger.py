"""Start or cancel a run, and confirm the coordinator heard it.

    python3 trigger.py approach_only | pickup | cancel

Why not `ros2 topic pub --once`: in Humble it sends its one message and exits
without waiting for the subscriber to be matched, and /pallet_approach/start is
VOLATILE on purpose (a latched start must never begin a run at startup). From
another machine discovery takes a second or two, so the message is often sent
to nobody and the coordinator logs nothing at all. This waits for the
coordinator's subscription, publishes, then waits for /coordinator/status to
show the run starting (or the cancel taking effect) and says what happened.
If nothing subscribes it says so: then this shell cannot see the stack over
DDS at all (ROS_DOMAIN_ID, RMW, or the FastDDS profile differ).
"""

import sys
import time

import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Empty, String

from pallet_approach_interfaces.msg import CoordinatorStatus

TRIGGER = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.VOLATILE)
MATCH_TIMEOUT = 10.0
REACTION_TIMEOUT = 5.0


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    what = sys.argv[1]
    rclpy.init()
    node = rclpy.create_node('pallet_approach_trigger')
    status = []
    node.create_subscription(CoordinatorStatus, '/coordinator/status', status.append, 10)
    if what == 'cancel':
        pub = node.create_publisher(Empty, '/pallet_approach/cancel', TRIGGER)
        msg = Empty()
    else:
        pub = node.create_publisher(String, '/pallet_approach/start', TRIGGER)
        msg = String(data=what)

    # Wait for the subscription to be matched AND for one status message, so
    # "the state changed" is measured against the state before sending.
    deadline = time.monotonic() + MATCH_TIMEOUT
    while ((pub.get_subscription_count() == 0 or not status)
           and time.monotonic() < deadline):
        rclpy.spin_once(node, timeout_sec=0.1)
    if pub.get_subscription_count() == 0:
        print(f'NOTHING SUBSCRIBES to {pub.topic_name} after {MATCH_TIMEOUT:.0f} s: the '
              'coordinator is not reachable from this shell over DDS. Check that it '
              'runs (docker logs), and that ROS_DOMAIN_ID (10), RMW_IMPLEMENTATION '
              'and FASTRTPS_DEFAULT_PROFILES_FILE match the stack container.')
        return 1

    before = status[-1].state if status else None
    if what == 'cancel' and before in ('IDLE', 'DONE', 'ABORTED'):
        print(f'coordinator is {before}: no run to cancel')
        return 0
    pub.publish(msg)
    print(f'sent {what!r} to {pub.get_subscription_count()} subscriber(s)')
    deadline = time.monotonic() + REACTION_TIMEOUT
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if status and status[-1].state != before:
            s = status[-1]
            print(f'coordinator: {s.state} ({s.reason})' + (f', outcome {s.outcome}' if s.outcome else ''))
            return 0
    s = status[-1] if status else None
    print('no state change within 5 s' + (f'; coordinator is {s.state}: {s.reason}' if s else
                                         '; no /coordinator/status received either') +
          '. See the stack log for a refusal (unhealthy sensor, unknown mode).')
    return 1


if __name__ == '__main__':
    sys.exit(main())
