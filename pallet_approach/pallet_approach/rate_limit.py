"""Deciding which frames to process when they arrive faster than they can be.

Depth arrives from the ZED at about 37 Hz. One frame through the perception
pipeline costs about 90 ms on a laptop and more on the Jetson, so roughly four
frames in five cannot be looked at. That is not a problem to solve, it is a
situation to handle correctly, and there are two ways to get it wrong:

* **Queue them.** Nothing is discarded, and the backlog grows about 26 frames a
  second. Within ten seconds the estimate being published describes where the
  pallet was seven seconds ago -- while the robot drives on it. The staleness
  watchdog would fire constantly and be right to.
* **Process whatever arrives next.** The estimate is fresh, but the rate is
  whatever the pipeline happens to manage, which drifts with scene complexity
  and with whatever else the Jetson is doing.

What this does instead: keep only the newest frame, and start work at most
`max_rate` times a second. The published estimate is then never older than one
processing period, and the rate is a number that was chosen rather than one
that emerged.

The gate is here rather than inline in the node so that the offline harness
(`tools/live.py`) can exercise the real policy against a recorded bag, instead
of a reimplementation of it that might drift.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FrameGate:
    """Rate limiter with counters, for a producer faster than its consumer.

    Usage is `offer()` per arriving frame; it returns True at most `max_rate`
    times a second and False otherwise, counting the refusals.

        gate = FrameGate(max_rate=5.0)
        ...
        if gate.offer(now):
            result = expensive(frame)
            gate.finished(now_after)

    `offer` takes the time rather than reading a clock so that the caller's
    clock is the one that counts -- the ROS clock in the node, which may be
    simulated, and wall time in the offline harness.
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
        # A hair under the interval still counts as "due". Without this, a
        # source whose period is close to `interval` drops about half its
        # frames to floating-point noise: set max_rate to the camera's own
        # 37 Hz and arrivals land 27.027 ms apart against a 27.027 ms interval,
        # where the comparison goes either way on the last bit and the achieved
        # rate silently halves. The tolerance is a thousandth of the interval,
        # far below any real jitter.
        if (self.last_start is not None
                and now - self.last_start < self.interval * 0.999):
            self.dropped += 1
            return False
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
