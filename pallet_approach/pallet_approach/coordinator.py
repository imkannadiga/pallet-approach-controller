"""The state machine: IDLE -> DETECTING -> APPROACHING -> CHECKING -> DONE.

No ROS here either. This class is driven by events and returns, for each one,
what state to be in, whether to stop the robot, and whether the controller may
run. The node wrapper turns that into lifecycle transitions, action feedback and
velocity commands.

Keeping it pure is what makes the safety properties testable:

* every terminal state stops the robot;
* the controller is permitted in exactly one state, APPROACHING;
* a cancel is honoured from any running state;
* DETECTING needs several consistent frames, so one lucky frame cannot start
  the robot moving;
* losing the pallet or getting unstable estimates while moving aborts.

    IDLE --trigger--> DETECTING --confirmed--> APPROACHING --arrived--> CHECKING --> DONE
                          |                        |                        |
                   timeout: NO_PALLET     cancel / lost / unstable      cancel
                          |                   : ABORTED                     |
                          +-------------> robot stopped, back to IDLE <------+
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from pallet_approach.model import Decision, PalletEstimate

__all__ = ['State', 'Outcome', 'Transition', 'Coordinator']


class State(Enum):
    IDLE = 'IDLE'
    DETECTING = 'DETECTING'
    APPROACHING = 'APPROACHING'
    CHECKING = 'CHECKING'
    DONE = 'DONE'


class Outcome(Enum):
    GO = 'GO'
    NO_GO = 'NO_GO'
    NO_PALLET = 'NO_PALLET'
    ABORTED = 'ABORTED'


@dataclass(frozen=True)
class Transition:
    """What the node should do after an event."""

    state: State
    stop_robot: bool
    """Publish a zero command. True in every terminal state, and on entry to
    CHECKING. Never assume the base will stop by itself: it has no watchdog."""

    controller_active: bool
    """Whether the controller node may publish. True only in APPROACHING."""

    perception_active: bool
    """Whether the perception node should be running."""

    outcome: Outcome | None = None
    reason: str = ''
    changed: bool = False
    """True when this event changed the state, so the node only logs and
    switches lifecycle states on real transitions."""


class Coordinator:
    """The state machine. One instance per node, reused across runs."""

    def __init__(self, params: dict):
        self.params = params
        self.state = State.IDLE
        self.started_at = 0.0
        self.entered_at = 0.0
        self.last_estimate_at = 0.0
        self.confirm_buffer: deque[PalletEstimate] = deque(
            maxlen=max(2, int(params['detecting']['confirm_frames'])))
        self.history: deque[PalletEstimate] = deque(maxlen=20)
        self.accepted: PalletEstimate | None = None

    # ------------------------------------------------------------ helpers

    def _at(self, state: State, now: float, *, stop, controller, perception,
            outcome=None, reason='', changed=True) -> Transition:
        if changed:
            self.state = state
            self.entered_at = now
        return Transition(state=state, stop_robot=stop, controller_active=controller,
                          perception_active=perception, outcome=outcome,
                          reason=reason, changed=changed)

    def _stay(self, now: float, reason: str = '') -> Transition:
        """No state change. Mirrors the current state's permissions."""
        return Transition(
            state=self.state,
            stop_robot=self.state in (State.IDLE, State.CHECKING, State.DONE),
            controller_active=self.state is State.APPROACHING,
            perception_active=self.state in (State.DETECTING, State.APPROACHING,
                                             State.CHECKING),
            reason=reason,
            changed=False,
        )

    def _terminal(self, now: float, outcome: Outcome, reason: str) -> Transition:
        """Every run ends here: robot stopped, both nodes deactivated."""
        self.accepted = None
        self.confirm_buffer.clear()
        return self._at(State.DONE, now, stop=True, controller=False,
                        perception=False, outcome=outcome, reason=reason)

    def _spread(self, estimates) -> tuple[float, float]:
        """Position and yaw spread (max - min) across a set of estimates."""
        positions = np.array([e.position[:2] for e in estimates])
        position_spread = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
        yaws = [e.yaw for e in estimates]
        reference = yaws[0]
        unwrapped = [reference + math.atan2(math.sin(y - reference), math.cos(y - reference))
                     for y in yaws]
        return position_spread, float(max(unwrapped) - min(unwrapped))

    # ------------------------------------------------------------- events

    def trigger(self, now: float) -> Transition:
        """A goal arrived. Only legal from IDLE or DONE."""
        if self.state in (State.DETECTING, State.APPROACHING, State.CHECKING):
            return self._stay(now, reason='a run is already in progress')
        self.started_at = now
        self.last_estimate_at = now
        self.confirm_buffer.clear()
        self.history.clear()
        self.accepted = None
        return self._at(State.DETECTING, now, stop=True, controller=False,
                        perception=True, reason='looking for a pallet')

    def cancel(self, now: float) -> Transition:
        """The operator cancelled. Honoured from any running state."""
        if self.state in (State.IDLE, State.DONE):
            return self._stay(now, reason='nothing to cancel')
        return self._terminal(now, Outcome.ABORTED, 'cancelled by the operator')

    def on_estimate(self, estimate: PalletEstimate, now: float) -> Transition:
        """Perception produced an estimate."""
        if self.state is State.DETECTING:
            self.confirm_buffer.append(estimate)
            self.last_estimate_at = now
            self.history.append(estimate)

            needed = int(self.params['detecting']['confirm_frames'])
            if len(self.confirm_buffer) < needed:
                return self._stay(now, reason=f'{len(self.confirm_buffer)}/{needed} frames')

            position_spread, yaw_spread = self._spread(self.confirm_buffer)
            if (position_spread > self.params['detecting']['max_position_spread']
                    or yaw_spread > self.params['detecting']['max_yaw_spread']):
                # inconsistent: keep looking rather than driving at a guess
                return self._stay(
                    now,
                    reason=f'inconsistent: {position_spread * 100:.1f} cm / '
                           f'{math.degrees(yaw_spread):.1f} deg spread')

            self.accepted = estimate
            return self._at(State.APPROACHING, now, stop=False, controller=True,
                            perception=True, reason='pallet confirmed')

        if self.state is State.APPROACHING:
            previous = self.accepted
            if previous is not None:
                jump = float(np.linalg.norm(estimate.position[:2] - previous.position[:2]))
                yaw_jump = abs(math.atan2(math.sin(estimate.yaw - previous.yaw),
                                          math.cos(estimate.yaw - previous.yaw)))
                if (jump > self.params['approaching']['max_estimate_jump']
                        or yaw_jump > self.params['approaching']['max_estimate_jump_yaw']):
                    # one outlier is not a reason to stop, but it is not a
                    # reason to steer at it either: drop it and keep going on
                    # the last good target. The lost-pallet timeout catches a
                    # run of them.
                    return self._stay(now, reason=f'rejected a {jump * 100:.0f} cm jump')
            self.accepted = estimate
            self.last_estimate_at = now
            self.history.append(estimate)
            return self._stay(now)

        if self.state is State.CHECKING:
            self.accepted = estimate
            self.last_estimate_at = now
            self.history.append(estimate)
            return self._stay(now)

        return self._stay(now)

    def on_tick(self, now: float) -> Transition:
        """Called on a timer. Owns every timeout.

        Separating this from the estimate callback matters: if perception stops
        publishing entirely, no callback fires, and without a timer tick the
        machine would wait for ever with the robot still moving.
        """
        if self.state is State.DETECTING:
            if now - self.started_at > self.params['detecting']['timeout']:
                return self._terminal(
                    now, Outcome.NO_PALLET,
                    f'no pallet confirmed within '
                    f'{self.params["detecting"]["timeout"]:.0f} s')
            return self._stay(now)

        if self.state is State.APPROACHING:
            quiet = now - self.last_estimate_at
            if quiet > self.params['approaching']['lost_pallet_timeout']:
                return self._terminal(
                    now, Outcome.ABORTED,
                    f'pallet lost: no estimate for {quiet:.1f} s')
            return self._stay(now)

        return self._stay(now)

    def on_arrived(self, now: float) -> Transition:
        """The controller reports it reached the standoff pose."""
        if self.state is not State.APPROACHING:
            return self._stay(now)
        return self._at(State.CHECKING, now, stop=True, controller=False,
                        perception=True, reason='at the standoff, checking')

    def ready_to_decide(self, now: float) -> bool:
        """Has the robot been stopped in CHECKING long enough to settle?

        Evaluating the rules while the chassis is still rocking would measure
        the suspension, not the pallet.
        """
        if self.state is not State.CHECKING:
            return False
        if now - self.entered_at < self.params['checking']['settle_time']:
            return False
        return len(self.history) >= int(self.params['checking']['frames'])

    def on_decision(self, decision: Decision, now: float) -> Transition:
        """The GO/NO-GO rules returned a verdict."""
        if self.state is not State.CHECKING:
            return self._stay(now)
        outcome = Outcome.GO if decision.go else Outcome.NO_GO
        return self._terminal(now, outcome, decision.reason)

    def on_fault(self, now: float, reason: str) -> Transition:
        """Anything unexpected: a tf failure, a perception exception, a lifecycle
        transition that did not take. Always ends the run with the robot stopped.
        """
        if self.state in (State.IDLE, State.DONE):
            return self._stay(now, reason=reason)
        return self._terminal(now, Outcome.ABORTED, reason)

    def to_idle(self, now: float) -> Transition:
        """Return to IDLE after the result has been reported."""
        return self._at(State.IDLE, now, stop=True, controller=False,
                        perception=False, reason='idle')
