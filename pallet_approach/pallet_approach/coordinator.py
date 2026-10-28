"""The state machine for a whole virtual pickup. No ROS: each event returns a
`Transition` (state, stop, drive, forks, perception); the node acts on it.

    IDLE -> CONFIRMING (-> NO_PALLET) -> CLASSIFYING (-> NO_GO)
         -> [BACKING_OFF] -> CLOSING -> CHECKING (-> NO_GO, one retry)
         -> LOWERING -> INSERTING -> LIFTING -> BACKING_OUT -> DONE
    Any state: cancel, unhealthy stack, lost pallet, controller abort -> ABORTED

Safety properties are tests, not comments: the controller drives only in
DRIVING_STATES; every terminal state stops the robot; one frame cannot start
it; an aborted insertion stops where it is and does not reverse out (D49).
"""

from __future__ import annotations


import numpy as np

from collections import deque
from dataclasses import dataclass
from enum import Enum

from pallet_approach.model import Decision

__all__ = ['State', 'Outcome', 'DriveRequest', 'TrackInfo', 'DriveInfo',
           'ForkInfo', 'Transition', 'Coordinator', 'DRIVING_STATES',
           'TERMINAL_STATES', 'MODES', 'PICKUP']

PICKUP = 'pickup'
"""The whole thing: approach, check, lower, insert, lift, back out.

It is the only mode. An approach-only mode existed for the first driving
session and was removed on 2026-10-27 (D95); a dry run now uses the
wheels-disconnected launch (`cmd_vel:=/cmd_vel_dry`), not a shorter sequence.
"""

MODES = (PICKUP,)


class State(Enum):
    IDLE = 'IDLE'
    CONFIRMING = 'CONFIRMING'
    CLASSIFYING = 'CLASSIFYING'
    BACKING_OFF = 'BACKING_OFF'
    CLOSING = 'CLOSING'
    CHECKING = 'CHECKING'
    LOWERING = 'LOWERING'
    INSERTING = 'INSERTING'
    LIFTING = 'LIFTING'
    BACKING_OUT = 'BACKING_OUT'
    DONE = 'DONE'
    ABORTED = 'ABORTED'


DRIVING_STATES = (State.BACKING_OFF, State.CLOSING, State.INSERTING,
                  State.BACKING_OUT)
"""The only states in which the controller may be told to move.

Listed once, here, and asserted over every event from every state in
`test_coordinator.py`. A new state added without thinking about this list
cannot drive, which is the right default.
"""

TERMINAL_STATES = (State.DONE, State.ABORTED)


class Outcome(Enum):
    INSERTED = 'INSERTED'
    """The whole sequence completed: forks in, lifted, backed out."""

    NO_GO = 'NO_GO'
    NO_PALLET = 'NO_PALLET'
    ABORTED = 'ABORTED'


@dataclass(frozen=True)
class DriveRequest:
    """What the controller should be doing. `None` means stop."""

    mode: str
    """CLOSE, REVERSE, INSERT or BACKOUT."""

    distance: float = 0.0
    """Metres, for REVERSE and BACKOUT. Always positive."""

    side: str = ''
    """Which face, for INSERT."""


@dataclass(frozen=True)
class TrackInfo:
    """What the tracker says, reduced to what the sequence needs."""

    confirmed: bool
    confidence: float
    side: str
    range: float
    offset: float
    heading: float
    reason: str = ''
    pallet_odom: tuple | None = None     # this frame's pocket midpoint in odom (D89)
    disagreeing: bool = False            # fused perception: camera and LiDAR disagree (D104 item 3)


@dataclass(frozen=True)
class DriveInfo:
    """What the controller says about the move it was given."""

    mode: str
    arrived: bool = False
    blocked: bool = False
    abort: bool = False
    reason: str = ''
    travelled: float = 0.0


@dataclass(frozen=True)
class ForkInfo:
    """What the fork simulator says."""

    state: str
    height: float
    moving: bool
    loaded: bool


@dataclass(frozen=True)
class Transition:
    """What the node should do after an event."""

    state: State
    stop_robot: bool
    drive: DriveRequest | None
    """What to command the controller. `None` means stop and deactivate it;
    this is the only way the controller is ever allowed to move."""

    fork: str | None
    """PARK, LOWER, LIFT, or None to leave the forks alone."""

    perception_active: bool
    outcome: Outcome | None = None
    reason: str = ''
    changed: bool = False

    @property
    def controller_active(self) -> bool:
        return self.drive is not None



def transitions_to_inactive(state: str) -> list[str] | None:
    """Lifecycle transitions that bring a managed node to `inactive` from its
    current state label; None when a run cannot start from it. A run leaves its
    nodes `inactive`, and re-sending `configure` from there kills the process
    (rclpy raises inside the service), so only what is missing is sent.
    """
    return {
        'unconfigured': ['configure'],
        'inactive': [],
        'active': ['deactivate'],
    }.get(state)


def machine_params(get) -> dict:
    """Everything `Coordinator` reads, by params.yaml name. The node passes its
    get_parameter and test/sequence.py a YAML lookup, so the harness runs on
    exactly the dict the robot builds.
    """
    def group(prefix, keys):
        return {key: get(f'{prefix}.{key}') for key in keys}

    closing = group('closing', (
        'k_cross', 'max_approach_angle', 'min_run_up', 'residual_target',
        'run_up_margin', 'run_up_tolerance'))
    closing['target_range'] = get('standoff_distance')
    closing['range_tolerance'] = get('closing.range_tolerance')
    return {
        'confirm_timeout': get('timeouts.confirm'),
        'lost_pallet_timeout': get('timeouts.lost_pallet'),
        'fork_timeout': get('timeouts.forks'),
        'run_timeout': get('timeouts.run'),
        'settle_time': get('checking.settle_time'),
        'checking_frames': get('checking.frames'),
        'max_retries': get('max_retries'),
        'backout_enabled': get('backout.enabled'),
        'retry_stable_frames': get('retry_stable_frames'),
        'retry_stable_spread': get('retry_stable_spread'),
        'min_retry_backoff': get('min_retry_backoff'),
        'retryable_checks': list(get('retryable_checks')),
        'entry_face': get('pallet.entry_face'),
        'sides': {
            side: group(f'pallet.sides.{side}', (
                'pocket_width', 'opening_top', 'opening_bottom', 'insertable',
                'depth'))
            for side in ('short', 'long')
        },
        'closing': closing,
        'reverse': {'max_unverified': get('reverse.max_unverified')},
    }

class Coordinator:
    """One instance per node, reused across runs."""

    def __init__(self, params: dict):
        self.params = params
        self.state = State.IDLE
        self.started_at = 0.0
        self.entered_at = 0.0
        self.last_track_at = 0.0
        self.track: TrackInfo | None = None
        self.history: deque = deque(maxlen=20)
        self.side = ''
        self.mode = PICKUP
        self.retries = 0
        self.forward_travel = 0.0
        self.retry_backoff = False
        self.reversed = 0.0
        self.insertion_travel = 0.0
        self.pending_backoff = 0.0
        self.last_reason = ''
        self.outcome: Outcome | None = None

    # ------------------------------------------------------------ plumbing

    def _at(self, state: State, now: float, *, stop, drive, fork, perception,
            outcome=None, reason='') -> Transition:
        self.state = state
        self.entered_at = now
        self.last_reason = reason
        self.outcome = outcome
        return Transition(state=state, stop_robot=stop, drive=drive, fork=fork,
                          perception_active=perception, outcome=outcome,
                          reason=reason, changed=True)

    def _stay(self, reason: str = '') -> Transition:
        """No state change: the current state's permissions, with the outcome carried
        along so a late event after the end still reports why the run ended.
        """
        return Transition(
            state=self.state,
            stop_robot=self.state not in DRIVING_STATES,
            drive=self._drive_for(self.state),
            fork=None,
            perception_active=self.state not in (State.IDLE, *TERMINAL_STATES),
            outcome=self.outcome if self.state in TERMINAL_STATES else None,
            reason=reason or self.last_reason,
            changed=False,
        )

    def _drive_for(self, state: State) -> DriveRequest | None:
        """The drive command that belongs to a state, republished on every tick.

        The controller stops if its command goes stale (`target_timeout`), so
        the sequence has to keep saying what it wants. Deriving it from the
        state rather than remembering what was sent means the two cannot drift
        apart.
        """
        if state is State.BACKING_OFF:
            return DriveRequest('REVERSE', distance=self.pending_backoff)
        if state is State.CLOSING:
            return DriveRequest('CLOSE')
        if state is State.INSERTING:
            return DriveRequest('INSERT', side=self.side)
        if state is State.BACKING_OUT:
            return DriveRequest('BACKOUT', distance=self.insertion_travel)
        return None

    def _finish(self, now: float, outcome: Outcome, reason: str) -> Transition:
        """Every run ends here. Robot stopped, everything deactivated."""
        state = State.DONE if outcome is Outcome.INSERTED else (
            State.ABORTED if outcome is Outcome.ABORTED else State.DONE)
        self.track = None
        return self._at(state, now, stop=True, drive=None, fork=None,
                        perception=False, outcome=outcome, reason=reason)

    def _abort(self, now: float, reason: str) -> Transition:
        return self._finish(now, Outcome.ABORTED, reason)

    # -------------------------------------------------------------- events

    def trigger(self, now: float, mode: str = PICKUP) -> Transition:
        """A goal arrived. Only legal from IDLE or a terminal state.

        `mode` must be PICKUP. An unknown mode is refused here rather than
        defaulted to.
        """
        if self.state not in (State.IDLE, *TERMINAL_STATES):
            return self._stay('a run is already in progress')
        if mode not in MODES:
            return self._stay(f'unknown mode {mode!r}; known: '
                              f'{", ".join(MODES)}')
        self.mode = mode
        self.started_at = now
        self.last_track_at = now
        self.outcome = None
        self.track = None
        self.history.clear()
        self.side = ''
        self.retries = 0
        self.forward_travel = 0.0
        self.retry_backoff = False
        self.reversed = 0.0
        self.insertion_travel = 0.0
        self.pending_backoff = 0.0
        return self._at(State.CONFIRMING, now, stop=True, drive=None,
                        fork='PARK', perception=True,
                        reason=f'looking for a pallet ({mode})')

    def cancel(self, now: float) -> Transition:
        if self.state in (State.IDLE, *TERMINAL_STATES):
            return self._stay('nothing to cancel')
        return self._abort(now, 'cancelled by the operator')

    def on_unhealthy(self, now: float, reason: str) -> Transition:
        """The watchdog reports the stack is no longer fit to drive (D9)."""
        if self.state in (State.IDLE, *TERMINAL_STATES):
            return self._stay(f'unhealthy while idle: {reason}')
        return self._abort(now, f'stack unhealthy: {reason}')

    def on_fault(self, now: float, reason: str) -> Transition:
        """Anything unexpected: a tf failure, a lifecycle transition that did
        not take, an exception in a callback."""
        if self.state in (State.IDLE, *TERMINAL_STATES):
            return self._stay(reason)
        return self._abort(now, reason)

    def on_track(self, track: TrackInfo, now: float) -> Transition:
        """The tracker published. Called at perception's rate."""
        self.track = track
        if track.confirmed:
            self.last_track_at = now
            self.history.append(track)

        if self.state is State.CONFIRMING:
            if not track.confirmed:
                return self._stay(track.reason or 'confirming')
            return self._classify(now)

        # D104 item 3: once the approach has begun, camera and LiDAR telling
        # different stories ends the run. Only the fused perception reports
        # this; it has already waited the 5 frames.
        if track.disagreeing and self.state in (State.CLOSING, State.CHECKING):
            return self._finish(now, Outcome.NO_GO,
                                f'camera and LiDAR disagree about the pallet: {track.reason}')

        if self.state is State.BACKING_OFF and self._backed_up_enough():
            return self._classify(now)
        return self._stay()

    def on_drive(self, drive: DriveInfo, now: float) -> Transition:
        """The controller reported on the move it was given."""
        if self.state not in DRIVING_STATES:
            return self._stay()
        if drive.mode != self._expected_mode():
            # The controller is still reporting the previous move. Ignoring it
            # matters: an `arrived` left over from the reverse would otherwise
            # complete the insertion the instant it started.
            return self._stay()

        if drive.abort:
            return self._abort(now, f'{drive.mode}: {drive.reason}')

        if self.state is State.BACKING_OFF:
            self.reversed = max(self.reversed, float(drive.travelled))
            if self._backed_up_enough():
                return self._classify(now)
            if drive.arrived:
                why = f' ({self.track.reason})' if self.track and self.track.reason else ''
                return self._finish(
                    now, Outcome.NO_GO,
                    f'backed up {self.reversed:.2f} m, as far as the ground it has '
                    f'covered allows, without a stable estimate{why}')
            return self._stay(drive.reason)

        if not drive.arrived:
            return self._stay(drive.reason)
        if self.state is State.CLOSING:
            self.forward_travel += float(drive.travelled)
            # Clear the history: frames taken while CLOSING span real travel
            # and fail the stability rule, which is about standing still.
            self.history.clear()
            return self._at(State.CHECKING, now, stop=True, drive=None,
                            fork=None, perception=True,
                            reason='at the standoff, checking')
        if self.state is State.INSERTING:
            self.insertion_travel = drive.travelled
            return self._at(State.LIFTING, now, stop=True, drive=None,
                            fork='LIFT', perception=True,
                            reason=f'inserted {drive.travelled:.3f} m, lifting')
        if self.state is State.BACKING_OUT:
            return self._finish(
                now, Outcome.INSERTED,
                f'virtual pickup complete: inserted '
                f'{self.insertion_travel:.3f} m into the {self.side} face and '
                f'backed out')
        return self._stay()

    def on_forks(self, forks: ForkInfo, now: float) -> Transition:
        """The fork simulator published."""
        if self.state is State.LOWERING:
            if forks.moving or forks.state != 'INSERT_HEIGHT':
                return self._stay(f'forks {forks.state}')
            return self._at(State.INSERTING, now, stop=False,
                            drive=DriveRequest('INSERT', side=self.side),
                            fork=None, perception=True,
                            reason=f'forks at {forks.height * 1000:.0f} mm, inserting')
        if self.state is State.LIFTING:
            if not forks.loaded:
                return self._stay(f'forks {forks.state}')
            if not self.params.get('backout_enabled', True):
                # D91: the forks are imaginary, so there is nothing to carry
                # out, and the back-out is the one move the robot makes with
                # only the LiDAR behind it. The run ends lifted, stopped where
                # the insertion left it.
                return self._finish(
                    now, Outcome.INSERTED,
                    f'virtual pickup complete: inserted '
                    f'{self.insertion_travel:.3f} m into the {self.side} face and '
                    f'lifted; back-out disabled, stopped at the pallet')
            return self._at(State.BACKING_OUT, now, stop=False,
                            drive=DriveRequest('BACKOUT',
                                               distance=self.insertion_travel),
                            fork=None, perception=True,
                            reason='lifted, backing out')
        return self._stay()

    def on_decision(self, decision: Decision, now: float) -> Transition:
        """The GO/NO-GO rules returned a verdict."""
        if self.state is not State.CHECKING:
            return self._stay()
        if decision.go:
            return self._at(State.LOWERING, now, stop=True, drive=None,
                            fork='LOWER', perception=True,
                            reason=f'GO: {decision.reason}')
        if self._retryable(decision) and self.retries < int(self.params['max_retries']):
            self.retries += 1
            return self._retry(now, f'NO-GO ({decision.reason}); '
                                    f'retry {self.retries}')
        return self._finish(now, Outcome.NO_GO, decision.reason)

    def on_tick(self, now: float) -> Transition:
        """Called on a timer. Owns every timeout.

        Separate from the event callbacks on purpose: if the tracker or the
        controller stops publishing entirely, no callback fires, and without a
        timer the machine would wait for ever with the robot still moving.
        """
        if self.state in (State.IDLE, *TERMINAL_STATES):
            return self._stay()

        if self.state is State.CONFIRMING:
            if now - self.started_at > float(self.params['confirm_timeout']):
                return self._finish(
                    now, Outcome.NO_PALLET,
                    f'no pallet confirmed within '
                    f'{float(self.params["confirm_timeout"]):.0f} s'
                    + (f' ({self.track.reason})' if self.track else ''))
            return self._stay()

        # Every state after CONFIRMING needs the pallet to still be there. The
        # exception is the insertion and the back-out, where the pockets have
        # left the camera's view by design (D16) and losing the track is
        # expected rather than a fault.
        if self.state not in (State.INSERTING, State.LIFTING, State.BACKING_OUT):
            quiet = now - self.last_track_at
            if quiet > float(self.params['lost_pallet_timeout']):
                why = (f' (tracker: {self.track.reason})'
                       if self.track is not None and self.track.reason else '')
                return self._abort(now, f'pallet lost: no confirmed track for '
                                        f'{quiet:.1f} s{why}')

        if self.state in (State.LOWERING, State.LIFTING):
            waited = now - self.entered_at
            if waited > float(self.params['fork_timeout']):
                return self._abort(
                    now, f'the forks did not reach their target in {waited:.1f} s')

        if now - self.started_at > float(self.params['run_timeout']):
            return self._abort(
                now, f'run exceeded {float(self.params["run_timeout"]):.0f} s')
        return self._stay()

    def to_idle(self, now: float) -> Transition:
        """Return to IDLE once the result has been reported."""
        return self._at(State.IDLE, now, stop=True, drive=None, fork=None,
                        perception=False, reason='idle')

    # ------------------------------------------------------------ internals

    def _expected_mode(self) -> str:
        request = self._drive_for(self.state)
        return request.mode if request else ''

    def _classify(self, now: float) -> Transition:
        """Decide the face, then whether there is room to close in.

        Reached from CONFIRMING and again after every reverse, so that the
        run-up is re-evaluated against the geometry the robot actually ends up
        with rather than against the one it predicted.
        """
        track = self.track
        if track is None or not track.confirmed:
            # The tracker's own reason, so the abort says WHY: on the robot
            # (2026-10-23) this fired right after a reverse with nothing to
            # tell a side flip from a jump from a LiDAR disagreement.
            why = f' ({track.reason})' if track is not None and track.reason else ''
            return self._abort(now, f'the track was lost before classifying{why}')

        if track.side == 'ambiguous':
            return self._finish(
                now, Outcome.NO_GO,
                f'unclear which face this is: {track.reason}. Measuring pocket '
                f'geometry against the wrong nominals is how a damaged pallet '
                f'passes (D22/D26).')

        if track.side == 'unknown':
            # Classification is switched off (pallet.classify.enabled). Fall
            # back to the configured expectation, and SAY SO, rather than
            # silently treating "not measured" as the default face.
            side = str(self.params['entry_face'])
            note = f'side not classified; assuming the {side} face'
        else:
            side = track.side
            note = f'{side} face'

        if side not in self.params['sides']:
            return self._abort(now, f'no geometry configured for side {side!r}')
        if not self.params['sides'][side].get('insertable', True):
            return self._finish(
                now, Outcome.NO_GO,
                f'insertion into the {side} face is not permitted here '
                f'(pallet.sides.{side}.insertable is false)')
        self.side = side

        # D89: no reverse before an approach. The robot drives in from wherever
        # Nav2 left it and is measured at the standoff, where perception is at
        # its best; backing up happens only after a NO-GO (_retry). Closer than
        # the standoff it cannot be measured there at all, so that is refused.
        closing = self.params['closing']
        standoff = float(closing['target_range'])
        if track.range < standoff - float(closing.get('range_tolerance', 0.03)):
            return self._finish(
                now, Outcome.NO_GO,
                f'{note}, but only {track.range:.2f} m away, inside the '
                f'{standoff:.2f} m standoff where the approach is checked. '
                f'Reposition with Nav2 and run again.')
        return self._at(State.CLOSING, now, stop=False,
                        drive=DriveRequest('CLOSE'), fork=None,
                        perception=True,
                        reason=(f'{note}, {track.range:.2f} m away with '
                                f'{track.offset * 1000:+.0f} mm of offset; closing in'))

    def _retry(self, now: float, reason: str) -> Transition:
        """Back up with perception running, then approach again (D50, D89).

        At least `min_retry_backoff`, then until the estimate is stable (see
        `_backed_up_enough`). Never further than the ground this run has
        covered: the distance driven forward plus the reverse D42 allows over
        unchecked ground. Reaching that without a stable estimate is a NO-GO.
        """
        if self.track is None:
            return self._abort(now, 'nothing to retry against: no track')
        minimum = float(self.params['min_retry_backoff'])
        limit = self.forward_travel + float(self.params['reverse']['max_unverified'])
        if limit < minimum:
            return self._finish(
                now, Outcome.NO_GO,
                f'{reason}; cannot back up the {minimum:.2f} m a retry needs: the '
                f'run covered only {self.forward_travel:.2f} m (+ '
                f'{float(self.params["reverse"]["max_unverified"]):.2f} m allowed unchecked)')
        self.retry_backoff = True
        self.reversed = 0.0
        self.history.clear()
        self.pending_backoff = limit
        return self._at(State.BACKING_OFF, now, stop=False,
                        drive=DriveRequest('REVERSE', distance=limit),
                        fork=None, perception=True,
                        reason=(f'{reason}; backing up at least {minimum:.2f} m '
                                f'(at most {limit:.2f} m) until the estimate is stable'))

    def _backed_up_enough(self) -> bool:
        """Far enough back, and the pallet holding still in odom (D89).

        Stable = the last `retry_stable_frames` confirmed frames' pocket
        midpoints, in odom, all within `retry_stable_spread` of their mean --
        the GO/NO-GO stability limit. In odom because the robot is moving.
        """
        if not self.retry_backoff or self.reversed < float(self.params['min_retry_backoff']):
            return False
        n = int(self.params['retry_stable_frames'])
        points = [t.pallet_odom for t in list(self.history)[-n:]
                  if t.confirmed and t.pallet_odom is not None]
        if len(points) < n:
            return False
        xy = np.asarray(points, dtype=float)
        if np.max(np.linalg.norm(xy - xy.mean(axis=0), axis=1)) > float(
                self.params['retry_stable_spread']):
            return False
        self.retry_backoff = False
        return True

    def _retryable(self, decision: Decision) -> bool:
        """NO-GOs another approach could change: where the robot stands and how steady
        the estimate is (`retryable_checks`), not properties of the pallet.
        """
        retryable = set(self.params['retryable_checks'])
        failed = {name for name, passed in decision.checks.items()
                  if passed is False}
        return bool(failed) and failed <= retryable

    def ready_to_decide(self, now: float) -> bool:
        """Has the robot been stopped in CHECKING long enough to settle?

        Evaluating the rules while the chassis is still rocking would measure
        the suspension rather than the pallet.
        """
        if self.state is not State.CHECKING:
            return False
        if now - self.entered_at < float(self.params['settle_time']):
            return False
        return len(self.history) >= int(self.params['checking_frames'])

    def describe(self) -> str:
        return f'{self.state.value}: {self.last_reason}'
