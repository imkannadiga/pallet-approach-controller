"""The state machine for a whole virtual pickup.

No ROS. The machine is driven by events and returns, for each one, what state
to be in, whether to stop the robot, what the controller should be doing and
what the forks should be doing. The node wrapper turns that into lifecycle
transitions, drive commands, action feedback and velocity commands.

    IDLE
     | goal
     v
    CONFIRMING    several consistent frames, LiDAR agreeing       -> NO_PALLET
     v
    CLASSIFYING   which face, and how much run-up this offset needs
     |  not enough room -> BACKING_OFF -> back here
     |  unclear side, or a face this site forbids -> NO_GO
     v
    CLOSING       servo onto the pocket centre line, stop at the standoff
     v
    CHECKING      stopped; the GO/NO-GO rules            -> NO_GO (or one retry)
     v  GO
    LOWERING      forks to pocket height
     v
    INSERTING     straight in, stopping on the earlier of odometry and LiDAR
     v
    LIFTING       forks up, pallet off the floor
     v
    BACKING_OUT   straight back to the standoff
     v
    DONE          outcome INSERTED, control back to Nav2

    Any state: cancel, unhealthy stack, lost pallet, controller abort
               -> robot stopped -> ABORTED

Keeping it pure is what makes the safety properties testable, and the ones that
matter are stated as tests rather than as comments:

* the controller is only ever commanded to drive in four states;
* every terminal state stops the robot and deactivates everything;
* a single frame cannot start the robot moving;
* a cancel, a health failure and a controller abort are honoured from every
  running state;
* **an aborted insertion stops where it is.** It does not reverse out. The
  robot's body is at least 0.204 m clear of the face at any legal stop (D13,
  D16), so standing still is safe, and driving after an abort contradicts the
  one rule that has no exceptions. Recovery is the operator's, with the reason
  in the result.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

from pallet_approach.closing import required_run_up
from pallet_approach.model import Decision

__all__ = ['State', 'Outcome', 'DriveRequest', 'TrackInfo', 'DriveInfo',
           'ForkInfo', 'Transition', 'Coordinator', 'DRIVING_STATES',
           'TERMINAL_STATES']


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

    GO = 'GO'
    """Reserved for a run asked to stop at the GO/NO-GO decision."""

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
        self.retries = 0
        self.backoffs = 0
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
        """No state change. Mirrors the current state's permissions exactly,
        so a state's behaviour is defined in one place and repeating an event
        cannot quietly change it.

        It carries the outcome too. Without that, a second event arriving after
        a run has ended reports `state = ABORTED` with `outcome = None`, which
        reads as "aborted for no reason" to anything looking at the latest
        transition rather than the last changed one. Found by a test that
        ticked past the end of a run.
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

    def trigger(self, now: float) -> Transition:
        """A goal arrived. Only legal from IDLE or a terminal state."""
        if self.state not in (State.IDLE, *TERMINAL_STATES):
            return self._stay('a run is already in progress')
        self.started_at = now
        self.last_track_at = now
        self.outcome = None
        self.track = None
        self.history.clear()
        self.side = ''
        self.retries = 0
        self.backoffs = 0
        self.insertion_travel = 0.0
        self.pending_backoff = 0.0
        return self._at(State.CONFIRMING, now, stop=True, drive=None,
                        fork='PARK', perception=True,
                        reason='looking for a pallet')

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

        if not drive.arrived:
            return self._stay(drive.reason)

        if self.state is State.BACKING_OFF:
            return self._classify(now)
        if self.state is State.CLOSING:
            # Clear the history: the GO/NO-GO stability rule is evaluated over
            # these frames, and the ones collected while CLOSING were taken
            # with the robot moving at up to 0.10 m/s. Five of those span
            # 100 mm of real travel, which the rule reads as an unstable
            # estimate and refuses -- measured in test_sequence.py, where a
            # perfectly good noisy run came back
            # "stability: face position moved 118 mm across 5 frames".
            # `ready_to_decide` then waits for five frames taken standing
            # still, which is what the rule is actually about.
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
                return self._abort(now, f'pallet lost: no confirmed track for '
                                        f'{quiet:.1f} s')

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
            return self._abort(now, 'the track was lost before classifying')

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

        needed = required_run_up(track.offset, self.params['closing'])
        # A tolerance on the ACCEPTANCE, because the reverse stops within its
        # own 10 mm window and odometry path length is never shorter than the
        # displacement. Without it a reverse that finishes at 2.99 m against a
        # 3.00 m requirement asks for another 10 mm, and the run dies on
        # `max_backoffs` having done everything right -- which is what
        # test_sequence.py caught.
        tolerance = float(self.params['closing']['run_up_tolerance'])
        if track.range >= needed - tolerance:
            return self._at(State.CLOSING, now, stop=False,
                            drive=DriveRequest('CLOSE'), fork=None,
                            perception=True,
                            reason=(f'{note}, {track.range:.2f} m away with '
                                    f'{track.offset * 1000:+.0f} mm of offset; '
                                    f'{needed:.2f} m of run-up needed, closing in'))

        # Reverse a little further than the shortfall, so the robot ends up
        # comfortably past the requirement rather than on its edge.
        return self._start_backoff(now, needed - track.range + tolerance,
                                   note, needed)

    def _start_backoff(self, now: float, distance: float, note: str,
                       needed: float) -> Transition:
        """Reverse to make room, or refuse if it cannot be justified."""
        self.backoffs += 1
        if self.backoffs > int(self.params['max_backoffs']):
            return self._finish(
                now, Outcome.NO_GO,
                f'still only {self.track.range:.2f} m from the pallet after '
                f'{self.backoffs - 1} reverse(s); {needed:.2f} m is needed')

        allowance = float(self.params['reverse']['max_unverified'])
        if distance > allowance:
            # The reverse cannot be justified from here (D42). Nav2 has the
            # costmap and could plan it; we do not, and guessing is not an
            # option when the robot cannot see behind itself.
            return self._finish(
                now, Outcome.NO_GO,
                f'{distance:.2f} m of reverse is needed to get {needed:.2f} m of '
                f'run-up, and only {allowance:.2f} m can be justified over '
                f'ground nothing has checked. Reposition with Nav2 and run '
                f'again.')

        self.pending_backoff = distance
        return self._at(State.BACKING_OFF, now, stop=False,
                        drive=DriveRequest('REVERSE', distance=distance),
                        fork=None, perception=True,
                        reason=(f'{note}, but only {self.track.range:.2f} m of '
                                f'run-up against {needed:.2f} m needed; '
                                f'reversing {distance:.2f} m'))

    def _retry(self, now: float, reason: str) -> Transition:
        """Back off and run the closing loop again.

        Athreya chose one retry on 2026-10-19. The reverse distance is the
        run-up the geometry asks for from where the robot now stands, which is
        normally short -- it is at the standoff, so it needs to get back to
        3 m.
        """
        track = self.track
        if track is None:
            return self._abort(now, 'nothing to retry against: no track')
        needed = required_run_up(track.offset, self.params['closing'])
        distance = max(needed - track.range, float(self.params['min_retry_backoff']))
        self.backoffs = 0
        self.pending_backoff = distance
        return self._at(State.BACKING_OFF, now, stop=False,
                        drive=DriveRequest('REVERSE', distance=distance),
                        fork=None, perception=True,
                        reason=f'{reason}; reversing {distance:.2f} m to try again')

    def _retryable(self, decision: Decision) -> bool:
        """Which NO-GOs are worth another approach.

        Only the ones another approach could change: where the robot is
        standing, and how steady the estimate is. A pocket that measures the
        wrong size, a face with no support behind the fit, an obstruction in
        the pocket or a load overhanging the face are all properties of the
        pallet, and driving the same approach again would measure them again.
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
