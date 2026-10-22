"""The frame gate: what the node does when depth arrives faster than it can work.

These are the cases that decide whether the robot acts on fresh geometry or on
a backlog, so they are worth pinning even though the class is small.
"""

import pytest

from pallet_approach.rate_limit import FrameGate


def test_first_frame_is_always_accepted():
    gate = FrameGate(max_rate=5.0)
    assert gate.offer(now=100.0)


def test_frames_inside_the_interval_are_dropped_not_queued():
    """The whole point: a frame that arrives too soon is discarded, not kept.

    Keeping it would mean the next estimate describes an older scene than the
    camera could have provided, which is the failure this class exists to
    prevent.
    """
    gate = FrameGate(max_rate=5.0)          # 200 ms apart
    assert gate.offer(0.0)
    gate.finished(0.05)
    assert not gate.offer(0.10)
    assert not gate.offer(0.19)
    assert gate.offer(0.20)
    assert gate.dropped == 2
    assert gate.processed == 1


def test_a_frame_arriving_mid_work_is_dropped():
    """A multi-threaded executor can deliver one while the last is in flight.
    Processing both at once would make `frame_seconds` meaningless."""
    gate = FrameGate(max_rate=5.0)
    assert gate.offer(0.0)
    assert not gate.offer(1.0), 'accepted a frame while still busy'
    gate.finished(0.09)
    assert gate.offer(1.0)


def test_rate_zero_means_no_limit():
    gate = FrameGate(max_rate=0.0)
    for i in range(5):
        assert gate.offer(i * 1e-6)
        gate.finished(i * 1e-6)
    assert gate.dropped == 0


def test_measured_rate_counts_only_processed_frames():
    gate = FrameGate(max_rate=10.0)
    gate.reset(0.0)
    # multiplied, not accumulated: 0.1 added four times lands at
    # 0.30000000000000004, and a frame offered 0.09999999999999998 later is
    # inside a 0.1 s interval. Real arrivals are 27 ms apart against a 200 ms
    # interval, so the boundary never comes up on the robot -- but a test that
    # walks straight into it is testing float addition, not the gate.
    for i in range(5):
        now = i * 0.1
        assert gate.offer(now)
        gate.finished(now + 0.05)
    # five frames in the 0.4 s between the first and last start
    assert gate.rate(0.4) == pytest.approx(12.5)
    assert gate.frame_seconds == pytest.approx(0.05)


def test_drop_fraction_is_of_everything_seen():
    gate = FrameGate(max_rate=5.0)
    gate.offer(0.0)
    gate.finished(0.05)
    for i in range(9):
        gate.offer(0.05 + i * 0.001)
    assert gate.drop_fraction() == pytest.approx(0.9)


def test_the_robot_rate_keeps_the_estimate_inside_the_watchdog_limit():
    """D9 stops the robot on an estimate older than 0.50 s. At the configured
    5 Hz, a frame waits at most one interval plus one frame time, so the
    measured 90-120 ms pipeline has a wide margin -- and this test says how wide
    before anyone raises the limit or slows the pipeline.
    """
    interval = 1.0 / 5.0
    worst_frame = 0.150          # generous against 96 ms measured on a laptop
    assert interval + worst_frame < 0.50


def test_a_schedule_reaches_25_fps_from_a_37_hz_camera():
    from pallet_approach.rate_limit import Schedule
    schedule = Schedule(25.0)
    frames = [i / 37.0 for i in range(37 * 10)]
    passed = sum(schedule.due(t) for t in frames)
    assert 248 <= passed <= 252          # 10 s at 25 fps, not 185 (18.5 fps)


def test_a_schedule_does_not_burst_after_a_pause():
    from pallet_approach.rate_limit import Schedule
    schedule = Schedule(25.0)
    assert schedule.due(0.0)
    assert schedule.due(5.0)             # after a 5 s gap: one frame...
    assert not schedule.due(5.001)       # ...not a catch-up burst
