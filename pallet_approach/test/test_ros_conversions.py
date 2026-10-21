"""Estimate <-> message round trip. Needs the built interfaces, so it runs in
the container (`colcon test` or the sim's stack image) and skips elsewhere."""

import math

import numpy as np
import pytest

pytest.importorskip('pallet_approach_interfaces.msg')
from std_msgs.msg import Header  # noqa: E402

from pallet_approach.model import PalletEstimate, Plane, Pocket  # noqa: E402
from pallet_approach.ros_conversions import estimate_from_msg, estimate_to_msg  # noqa: E402


def test_every_field_the_coordinator_reads_survives_the_round_trip():
    # The tracker republishes each estimate, so whatever this drops, the
    # coordinator never sees. It used to drop the side.
    estimate = PalletEstimate(
        position=np.array([3.5, 0.1, 0.07]), normal=np.array([-1.0, 0.0, 0.0]),
        yaw=math.pi, face_inlier_count=300,
        pockets=(Pocket(np.array([3.5, 0.28, 0.05]), 0.23, 0.10),
                 Pocket(np.array([3.5, -0.08, 0.05]), 0.23, 0.10)),
        floor=Plane(np.array([0.0, 0.0, 1.0]), 0.025),
        side='short', side_detail={'reason': 'spacing and width agree'},
        face_width=0.79)
    header = Header(frame_id='base_footprint')
    back = estimate_from_msg(estimate_to_msg(estimate, header))
    assert back.side == 'short'
    assert back.side_detail['reason'] == 'spacing and width agree'
    assert back.face_width == pytest.approx(0.79)
    assert np.allclose(back.position, estimate.position)
    assert len(back.pockets) == 2
