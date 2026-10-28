"""The virtual forks: a carriage this robot does not have.

It holds a height, moves it at a finite speed, and sets `loaded` only when
asked to lift from the insertion height, never because the state machine
said LIFTING. Heights are of the tine underside above the floor
(`forks.bottom_height`). `lift_travel` is carriage travel; the tine first
closes the gap to the pocket top, so `pallet_lift` reports how far the pallet
actually came off the floor.
"""

from __future__ import annotations

import math
from enum import Enum

__all__ = ['State', 'ForkSimulator', 'pallet_lift']


class State(Enum):
    PARKED = 'PARKED'
    MOVING = 'MOVING'
    INSERT_HEIGHT = 'INSERT_HEIGHT'
    LIFTED = 'LIFTED'


def pallet_lift(side: dict, params: dict) -> float:
    """How far the pallet comes off the floor, metres, for one face.

    Zero rather than negative when the carriage travel is less than the slack
    above the tines: the forks then rise inside the pocket and the pallet does
    not move, which is a failed lift and should read as one.
    """
    slack = (float(side['opening_top'])
             - (float(params['insert_height']) + float(params['tine_thickness'])))
    return max(0.0, float(params['lift_travel']) - slack)


class ForkSimulator:
    """Height, motion and the loaded flag. No ROS, no clock of its own."""

    def __init__(self, params: dict):
        self.params = dict(params)
        self.height = float(params['park_height'])
        self.target = self.height
        self.state = State.PARKED
        self.loaded = False
        self._updated_at: float | None = None
        self._destination = self.state

    # ---------------------------------------------------------- commands

    def park(self, now: float) -> None:
        """Back to travel height. Clears `loaded`: a pallet put down is not
        carried any more, and leaving the flag set is how a simulator starts
        telling comfortable lies."""
        self._go(float(self.params['park_height']), State.PARKED, now)
        self.loaded = False

    def lower_to_insert(self, now: float) -> None:
        self._go(float(self.params['insert_height']), State.INSERT_HEIGHT, now)
        self.loaded = False

    def lift(self, now: float) -> bool:
        """Raise by `lift_travel`. Returns False, and does nothing, unless the
        forks are at the insertion height.

        The refusal is the point. Lifting from park height would raise a
        carriage that is nowhere near a pallet, and the sequence would still
        get a `loaded` flag out of it.
        """
        if self.state is not State.INSERT_HEIGHT:
            return False
        self._go(float(self.params['insert_height'])
                 + float(self.params['lift_travel']), State.LIFTED, now)
        return True

    def _go(self, target: float, destination: State, now: float) -> None:
        self.target = float(target)
        self._destination = destination
        self._updated_at = now
        if abs(self.target - self.height) > 1e-9:
            self.state = State.MOVING

    # ------------------------------------------------------------ update

    def update(self, now: float) -> None:
        """Advance the height towards the target at `speed`.

        The first call after a command establishes the clock rather than
        integrating from it -- `_updated_at` is None until then, never 0.0,
        because 0.0 is a time.
        """
        if self._updated_at is None:
            self._updated_at = now
            return
        dt = now - self._updated_at
        self._updated_at = now
        if dt <= 0.0 or self.state is not State.MOVING:
            return
        step = float(self.params['speed']) * dt
        remaining = self.target - self.height
        if abs(remaining) <= step:
            self.height = self.target
            self.state = self._destination
            if self.state is State.LIFTED:
                self.loaded = True
        else:
            self.height += math.copysign(step, remaining)

    # ------------------------------------------------------------- state

    @property
    def moving(self) -> bool:
        return self.state is State.MOVING

    @property
    def at_insert_height(self) -> bool:
        return self.state is State.INSERT_HEIGHT

    def seconds_to(self, target: float) -> float:
        """How long a move to `target` would take, for the sequence's timeout."""
        return abs(float(target) - self.height) / float(self.params['speed'])

    def describe(self) -> str:
        return (f'{self.state.value} at {self.height * 1000:.0f} mm'
                + (f' -> {self.target * 1000:.0f} mm' if self.moving else '')
                + (' (loaded)' if self.loaded else ''))
