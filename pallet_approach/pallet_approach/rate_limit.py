"""Deciding which frames to process when they arrive faster than they can be.

Depth arrives at ~37 Hz and perception takes ~90 ms or more. Queueing makes
the estimate ever older; processing whatever comes next gives an unplanned
rate. So: keep only the newest frame and start work at most `max_rate` times a
second. Kept outside the node so `tools/live.py` runs the same policy.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FrameGate:
    """Rate limiter with counters, for a producer faster than its consumer.

        if gate.offer(now):
            result = expensive(frame)
            gate.finished(now_after)

    Takes the time as an argument, so the caller's clock (ROS or wall) counts.
    """

    max_rate: float
    """Hz. Zero or negative means no limit, which is only useful in tests."""

    processed: int = 0
    dropped: int = 0
    last_start: float | None = None
    """When the most recent accepted frame started, or None before the first.

    None rather than 0.0: a monotonic clock legitimately reads 0.0, and a
    falsy sentinel that is also a valid value makes the first frame after
    t = 0 behave differently from every other. A test caught exactly that.
    """

    busy: bool = False
    """True between `offer` returning True and the matching `finished`."""

    frame_seconds: float = 0.0
    """How long the last accepted frame took. The number to watch."""

    window_start: float | None = None
    next_due: float | None = None

    @property
    def interval(self) -> float:
        return 1.0 / self.max_rate if self.max_rate > 0 else 0.0

    def offer(self, now: float) -> bool:
        """Should this frame be processed? Counts the ones that should not."""
        if self.busy:
            # Re-entrant arrival: a callback fired while the previous frame was
            # still being worked on. Single-threaded rclpy will not do this, but
            # a multi-threaded executor would, and silently processing two
            # frames at once would make `frame_seconds` meaningless.
            self.dropped += 1
            return False
        # A thousandth of an interval early still counts as due, or a source at
        # exactly max_rate loses half its frames to floating-point noise.
        if (self.next_due is not None
                and now < self.next_due - self.interval * 0.001):
            self.dropped += 1
            return False
        # Advance from the previous due time, not this frame, so the average is
        # max_rate (otherwise 10 Hz in, 3.3 Hz gate gave 2.5 Hz). After a pause
        # or an overrun, restart from this frame, so no burst.
        start = self.next_due
        if start is None or now - start > self.interval:
            start = now
        self.next_due = start + self.interval
        self.last_start = now
        self.busy = True
        if self.window_start is None:
            self.window_start = now
        return True

    def finished(self, now: float) -> None:
        self.busy = False
        self.processed += 1
        self.frame_seconds = now - self.last_start

    def rate(self, now: float) -> float:
        """Frames actually processed per second since the window opened."""
        if self.window_start is None:
            return 0.0
        window = now - self.window_start
        return self.processed / window if window > 0 else 0.0

    def drop_fraction(self) -> float:
        seen = self.processed + self.dropped
        return self.dropped / seen if seen else 0.0

    def reset(self, now: float | None = None) -> None:
        self.processed = self.dropped = 0
        self.last_start = None
        self.next_due = None
        self.busy = False
        self.window_start = now


class Schedule:
    """Let frames through at an AVERAGE rate, from a faster input.

    "At least 1 / rate since the last one" can only produce the input rate
    divided by a whole number: from a 37 Hz camera, asking for anything between
    18.5 and 37 Hz gives 18.5. This advances a due time by one period per frame
    let through instead, so the average is exactly `rate`. After a pause longer than one period it restarts from
    the current frame, so a pause does not come out as a burst.
    """

    def __init__(self, rate: float):
        self.period = 1.0 / float(rate)
        self.next_due = None

    def due(self, now: float) -> bool:
        if self.next_due is not None and now < self.next_due:
            return False
        start = self.next_due
        if start is None or now - start > self.period:
            start = now         # first frame, or the input paused: restart
        self.next_due = start + self.period
        return True
