"""Conversions between the plain-Python types and ROS messages.

All the message handling lives here so that `perception.py`, `go_nogo.py`,
`approach.py`, `controller.py` and `coordinator.py` never import a ROS message
and stay testable without a ROS installation.
"""

from __future__ import annotations

import math

import numpy as np
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
from nav_msgs.msg import Odometry

from pallet_approach_interfaces.msg import PalletEstimate as PalletEstimateMsg
from pallet_approach_interfaces.msg import Pocket as PocketMsg

from pallet_approach.controller import Command
from pallet_approach.model import PalletEstimate, Plane, Pocket, Pose2D

__all__ = ['yaw_to_quaternion', 'quaternion_to_yaw', 'pose2d_from_odometry',
           'pose2d_to_pose', 'estimate_to_msg', 'estimate_from_msg',
           'command_to_twist', 'transform_to_matrix', 'stamp_to_seconds']


def yaw_to_quaternion(yaw: float) -> Quaternion:
    """A planar rotation as a quaternion. Only z and w are non-zero."""
    return Quaternion(x=0.0, y=0.0, z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))


def quaternion_to_yaw(q: Quaternion) -> float:
    """Extract the yaw from a quaternion, ignoring roll and pitch.

    The standard formula. On a wheeled robot on a flat floor roll and pitch are
    noise, so dropping them is correct here -- but say so out loud rather than
    pretending a quaternion only holds yaw.
    """
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def pose2d_from_odometry(msg: Odometry) -> Pose2D:
    """The robot's planar pose from /odom, which is odom -> base_link."""
    return Pose2D(x=msg.pose.pose.position.x,
                  y=msg.pose.pose.position.y,
                  yaw=quaternion_to_yaw(msg.pose.pose.orientation))


def pose2d_to_pose(pose: Pose2D, z: float = 0.0) -> Pose:
    return Pose(position=Point(x=pose.x, y=pose.y, z=z),
                orientation=yaw_to_quaternion(pose.yaw))


def estimate_to_msg(estimate: PalletEstimate | None, header) -> PalletEstimateMsg:
    """Pack an estimate for publication. None becomes `valid = False`.

    Publishing an explicit invalid message rather than nothing lets the
    coordinator tell "perception is running and sees nothing" apart from
    "perception has died", which are different failures with different
    responses.
    """
    msg = PalletEstimateMsg()
    msg.header = header
    if estimate is None:
        msg.valid = False
        return msg
    msg.valid = True
    msg.position = Point(x=float(estimate.position[0]),
                         y=float(estimate.position[1]),
                         z=float(estimate.position[2]))
    msg.normal = Vector3(x=float(estimate.normal[0]),
                         y=float(estimate.normal[1]),
                         z=float(estimate.normal[2]))
    msg.yaw = float(estimate.yaw)
    msg.face_inlier_count = int(estimate.face_inlier_count)
    msg.side = str(estimate.side)
    msg.side_reason = str(estimate.side_detail.get('reason', ''))
    msg.face_width = float(estimate.face_width)
    # NaN, not 0.0, when there is no floor fit. Zero would read as "the floor
    # is at base_footprint's own origin", which is the one thing it is NOT:
    # measured 2026-10-14, the real floor sits 25 mm below at -0.025. Until
    # this line existed the field was never assigned at all and every estimate
    # published 0.0 (caught by test_interfaces.py).
    msg.floor_height = float(estimate.floor_height)
    if estimate.floor is not None:
        msg.floor_valid = True
        msg.floor_normal = Vector3(x=float(estimate.floor.normal[0]),
                                   y=float(estimate.floor.normal[1]),
                                   z=float(estimate.floor.normal[2]))
        msg.floor_offset = float(estimate.floor.offset)
    msg.pockets = [
        PocketMsg(center=Point(x=float(p.center[0]), y=float(p.center[1]),
                               z=float(p.center[2])),
                  width=float(p.width), height=float(p.height))
        for p in estimate.pockets
    ]
    return msg


def estimate_from_msg(msg: PalletEstimateMsg) -> PalletEstimate | None:
    """Unpack an estimate. An invalid message becomes None."""
    if not msg.valid:
        return None
    return PalletEstimate(
        position=np.array([msg.position.x, msg.position.y, msg.position.z]),
        normal=np.array([msg.normal.x, msg.normal.y, msg.normal.z]),
        yaw=float(msg.yaw),
        pockets=tuple(
            Pocket(center=np.array([p.center.x, p.center.y, p.center.z]),
                   width=float(p.width), height=float(p.height))
            for p in msg.pockets
        ),
        face_inlier_count=int(msg.face_inlier_count),
        floor=(Plane(normal=np.array([msg.floor_normal.x, msg.floor_normal.y,
                                      msg.floor_normal.z]),
                     offset=float(msg.floor_offset))
               if msg.floor_valid else None),
        stamp=stamp_to_seconds(msg.header.stamp),
        frame_id=msg.header.frame_id,
        # Round-tripped: the tracker republishes every estimate, and defaults
        # here made every pallet look unclassified (sim, 2026-10-20).
        side=msg.side or 'unknown',
        side_detail={'reason': msg.side_reason} if msg.side_reason else {},
        face_width=float(msg.face_width),
    )


def command_to_twist(command: Command) -> Twist:
    """A Command as geometry_msgs/Twist. /cmd_vel is unstamped Twist."""
    twist = Twist()
    twist.linear.x = float(command.linear)
    twist.angular.z = float(command.angular)
    return twist


def transform_to_matrix(transform) -> np.ndarray:
    """A geometry_msgs/TransformStamped as a 4x4 homogeneous matrix.

    Written out rather than pulling in tf_transformations, which is not in the
    container and would be one more dependency for four lines of algebra.
    """
    t = transform.transform.translation
    q = transform.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    matrix = np.eye(4)
    matrix[:3, :3] = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    matrix[:3, 3] = [t.x, t.y, t.z]
    return matrix


def stamp_to_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9
