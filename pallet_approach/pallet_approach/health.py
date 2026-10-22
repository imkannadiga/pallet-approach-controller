"""Is the stack healthy enough to drive?

D9 and D12 in `docs/decisions.md`: the robot stops if a sensor goes stale, and
the coordinator refuses a goal if the stack is unhealthy when it arrives. Both
need the same answer to the same question, so the question is answered once,
here, with no ROS in sight.

Two things this file is careful about, both learned the hard way on this
project:

**Never seen is not stale.** A source that has produced nothing since startup
has no age at all. Reporting it as age 0.0 makes a dead topic look perfectly
fresh, which is the same class of bug as the `last_start = 0.0` sentinel in
`rate_limit.py` and the `False`-as-sentinel in `fork_overlay`. So the age of an
unseen source is `None`, and `None` fails every limit.

**Not yet measurable is not zero.** A rate needs two messages and enough
elapsed time to mean anything. Immediately after activation there is neither,
and a rate reported as 0.0 Hz would trip the rate limit on every run before the
first second was out. So the rate is `None` until the window holds enough
history, and a `None` rate is not judged -- the age limit covers that interval,
which is what it is for.

What is *required* changes with the run. The camera, the LiDAR, odometry and tf
must be healthy before a goal is accepted; a perception estimate cannot be,
because perception is deactivated between runs and publishes nothing. So each
source carries the set of phases in which it is required, and `check` is told
which phase the system is in.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

__all__ = ['Limits', 'SourceState', 'Report', 'HealthMonitor', 'as_float',
           'ALWAYS', 'WHILE_RUNNING', 'NEVER']

ALWAYS = 'always'
"""Required whenever the stack is up: the sensors and tf."""

WHILE_RUNNING = 'running'
"""Required only while a run is in progress: anything our own stack produces."""

NEVER = 'never'
"""Measured and reported, but never allowed to make the stack unhealthy.

For a check that is deliberately disabled: the tf age when the bench
`tf_fallback_to_latest` workaround is on, for instance. The point of having it
rather than dropping the source is that the number still appears in /health and
in the diagnostics, so "we are not enforcing this" is visible instead of
invisible -- the recurring failure on this project is the reasonable default
nobody mentions.
"""


@dataclass(frozen=True)
class Limits:
    """What counts as healthy for one source."""

    max_age: float
    """Seconds since the last message before the source is stale. D9 sets these
    at roughly five missed messages at the measured rate."""

    min_rate: float = 0.0
    """Hz below which the source counts as degraded, or 0 to not check.

    Derived from what the pipeline CONSUMES, not from what the sensor nominally
    produces: depth arrives at ~37 Hz and perception looks at 5 of them, so a
    camera at 12 Hz is fine and one at 2 Hz is not."""

    when: str = ALWAYS
    """`ALWAYS`, `WHILE_RUNNING` or `NEVER`."""
    first_message_grace: float = 0.0
    """Seconds after a run starts during which a source that has not yet
    produced anything still counts as healthy (D58, 2026-10-20).

    For `estimate`: perception is activated by the run itself and needs a
    frame plus up to a second of processing before its first estimate, while
    the watchdog judges at 2 Hz. Without a grace, a run aborted whenever the
    first check came first -- 2 runs in 4 in the Gazebo sim. Only ever-silent
    sources get it: once one message has arrived, the ordinary age limit
    applies, so a perception that stops mid-run is caught exactly as before."""

    max_skew: float = 0.05
    """How far a message may be stamped into the FUTURE before that is itself
    the failure, seconds.

    Only reachable for a source whose timestamp comes from another machine's
    clock -- the tf chain, whose camera link is republished at 58 Hz on the
    ZED's clock. A negative age passes every staleness limit ever written, so
    the 24.5 % fast camera clock measured on 2026-10-15 would have read as
    perfectly fresh tf. 50 ms is loose enough for ordinary clock jitter between
    two synced machines and tight enough that real skew is named as skew."""


@dataclass(frozen=True)
class SourceState:
    """One source's condition at one instant."""

    name: str
    age: float | None
    """Seconds since the last message; `None` if nothing has ever arrived."""

    rate: float | None
    """Hz over the window; `None` while the window is too short to measure."""

    required: bool
    ok: bool
    detail: str
    """Empty when ok, otherwise what is wrong, with the numbers in it."""


@dataclass(frozen=True)
class Report:
    """The whole stack's condition at one instant."""

    ok: bool
    """Every source required in this phase is inside its limits."""

    sources: tuple[SourceState, ...]
    phase: str

    @property
    def failures(self) -> tuple[str, ...]:
        """One line per required source that is outside its limits."""
        return tuple(f'{s.name}: {s.detail}'
                     for s in self.sources if s.required and not s.ok)

    def summary(self) -> str:
        if self.ok:
            return 'healthy'
        return '; '.join(self.failures)

    def __getitem__(self, name: str) -> SourceState:
        for source in self.sources:
            if source.name == name:
                return source
        raise KeyError(name)


class HealthMonitor:
    """Tracks arrival times per source and answers whether we may drive.

    Pure bookkeeping: the caller supplies every timestamp, so a whole run can
    be replayed through it in a test without a clock.
    """

    def __init__(self, limits: dict[str, Limits], window: float = 2.0):
        if not limits:
            raise ValueError('a health monitor with no sources would always be healthy')
        self.limits = dict(limits)
        self.window = float(window)
        self._arrivals: dict[str, deque[float]] = {
            name: deque(maxlen=512) for name in limits}
        self._last: dict[str, float | None] = {name: None for name in limits}

    def seen(self, name: str, now: float) -> None:
        """Record a message arriving on `name`."""
        if name not in self.limits:
            raise KeyError(f'unknown source {name!r}; known: {sorted(self.limits)}')
        self._last[name] = now
        arrivals = self._arrivals[name]
        arrivals.append(now)
        while arrivals and now - arrivals[0] > self.window:
            arrivals.popleft()

    def forget(self, name: str) -> None:
        """Drop a source's history, so it reads as never seen again.

        Called when the thing that produces it is deactivated: a perception
        estimate from the previous run is not evidence about this one.
        """
        self._last[name] = None
        self._arrivals[name].clear()

    def age(self, name: str, now: float) -> float | None:
        last = self._last[name]
        return None if last is None else now - last

    def rate(self, name: str, now: float) -> float | None:
        """Hz over the window, or None while it cannot be measured.

        Needs two arrivals and at least half a window of history. The half is
        the point: with a 2 s window a 5 Hz source has 5 samples after one
        second, which measures the rate perfectly well, and waiting for the
        full window would leave a gap where only the age limit applies.
        """
        arrivals = self._arrivals[name]
        while arrivals and now - arrivals[0] > self.window:
            arrivals.popleft()
        if len(arrivals) < 2:
            return None
        span = arrivals[-1] - arrivals[0]
        if span < self.window / 2.0:
            return None
        return (len(arrivals) - 1) / span

    def check(self, now: float, phase: str = ALWAYS,
              running_since: float | None = None) -> Report:
        """Judge every source. `phase` is ALWAYS or WHILE_RUNNING.

        A source is required when its own `when` says so and the phase allows
        it. `NEVER` sources are judged and reported exactly like the others --
        they simply cannot make `Report.ok` false.
        """
        states = []
        for name, limits in self.limits.items():
            required = (limits.when == ALWAYS
                        or (limits.when == WHILE_RUNNING and phase == WHILE_RUNNING))
            age = self.age(name, now)
            rate = self.rate(name, now)
            ok, detail = True, ''
            waited = None if running_since is None else now - running_since
            if (age is None and waited is not None
                    and waited < limits.first_message_grace):
                detail = (f'waiting for the first message, {waited:.1f} s of '
                          f'{limits.first_message_grace:.1f} s')
            elif age is None:
                ok = False
                detail = 'nothing received since startup'
            elif age < -limits.max_skew:
                ok = False
                detail = (f'stamped {-age:.2f} s in the future: the clocks '
                          f'disagree, so no age here means anything')
            elif age > limits.max_age:
                ok = False
                detail = f'{age:.2f} s old, limit {limits.max_age:.2f} s'
            elif limits.min_rate > 0.0 and rate is not None and rate < limits.min_rate:
                ok = False
                detail = f'{rate:.1f} Hz, limit {limits.min_rate:.1f} Hz'
            states.append(SourceState(name=name, age=age, rate=rate,
                                      required=required, ok=ok, detail=detail))
        return Report(ok=all(s.ok for s in states if s.required),
                      sources=tuple(states), phase=phase)


def as_float(value: float | None) -> float:
    """`None` as NaN, for a message field that has no optional type.

    NaN rather than -1 or 0: both of those are readable as measurements, and a
    reader who forgets to check gets a plausible wrong number instead of one
    that is obviously absent.
    """
    return float('nan') if value is None else float(value)


PERCEPTION_STATES = ('CONFIRMING', 'CLASSIFYING', 'BACKING_OFF', 'CLOSING', 'CHECKING')
"""Coordinator states in which the camera is in use (D64)."""


def phase_for_state(state: str, perception_states=PERCEPTION_STATES) -> str:
    """WHILE_RUNNING only in the states that use the camera; ALWAYS otherwise.

    D64 (Athreya, 2026-10-22): perception's estimate is required to be fresh
    only while the coordinator is using it. From LOWERING on, the camera
    cannot see the pockets (they leave its view ~0.9 m ahead, D16) and the
    insertion runs on odometry with the LiDAR as its stop, both of which stay
    required in every state. Requiring the estimate there aborted 5 of 5
    pickups in the Gazebo sim half a metre into the insertion.
    """
    return WHILE_RUNNING if state in tuple(perception_states) else ALWAYS
