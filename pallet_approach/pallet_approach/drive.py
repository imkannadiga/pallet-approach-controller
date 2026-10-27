"""Executing one drive request: the controller's logic, without ROS.

`closing.py` and `insertion.py` hold the control laws; this holds the small
amount of state that running them needs -- which mode is in progress, how far
it has travelled, which phase the servo is in, the insertion plan -- and
decides, each frame, what the base should be told.

It is a separate file from `nodes/controller_node.py` for the same reason
`coordinator.py` is separate from `nodes/coordinator_node.py`: the join between
the sequence and the control laws is where the interesting mistakes live, and
testing it through a ROS node means testing it on a robot. With this split, the
whole virtual pickup runs at a desk against `test/simulate.py`, and the node is
a shell that subscribes, publishes and calls `step`.

Everything comes in as arguments: the geometry for this frame, the LiDAR cloud
in `base_footprint`, the odometry travel. Nothing is read from a clock.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pallet_approach import closing, insertion
from pallet_approach.controller import ZERO, Command

__all__ = ['Request', 'Result', 'DriveExecutor', 'MODES']

MODES = ('STOP', 'CLOSE', 'REVERSE', 'INSERT', 'BACKOUT')


@dataclass(frozen=True)
class Request:
    """What to do. The same shape as `coordinator.DriveRequest`, and the same
    fields as the `DriveCommand` message, so nothing has to be translated."""

    mode: str
    distance: float = 0.0
    side: str = ''


@dataclass(frozen=True)
class Result:
    """What the executor decided this frame."""

    command: Command
    arrived: bool
    abort: bool
    reason: str
    phase: str = ''
    face_range: float = float('nan')
    lateral_offset: float = float('nan')
    heading_error: float = float('nan')
    rear_clearance: float = float('nan')

    @property
    def blocked(self) -> bool:
        """Not driving, and not because the move finished."""
        return self.command.is_zero() and not self.arrived


def executor_params(get) -> dict:
    """Everything `DriveExecutor` reads, from parameter names.

    `get(name)` returns one parameter by its params.yaml name, such as
    'closing.k_cross'. The controller node passes its own get_parameter; the
    tests pass a lookup into params.yaml and build a real DriveExecutor from the
    result, so a key the executor reads and this function forgets fails at a
    desk instead of inside a lifecycle callback. That is not hypothetical: on
    2026-10-20 the Gazebo sim found `divergence_frames` and
    `min_divergence_growth` declared, set in params.yaml, and never passed on,
    so the controller raised KeyError on every configure -- which rclpy's
    lifecycle swallows, leaving only "could not configure the managed nodes"
    on the coordinator.
    """
    def group(prefix, keys):
        return {key: get(f'{prefix}.{key}') for key in keys}

    closing_params = group('closing', (
        'k_cross', 'k_heading', 'max_approach_angle', 'straighten_offset',
        'max_turn_in_place', 'closing_speed', 'taper_range',
        'range_tolerance', 'arrival_yaw', 'resume_range', 'min_run_up',
        'residual_target', 'run_up_margin',
        'divergence_frames', 'min_divergence_growth', 'divergence_window'))
    closing_params['target_range'] = get('standoff_distance')
    closing_params['carriage_x'] = get('forks.carriage_x')
    closing_params['tine_length'] = get('forks.tine_length')

    reverse_params = group('reverse', (
        'speed', 'max_unverified', 'side_margin', 'min_height',
        'max_height', 'clearance_margin'))
    reverse_params['rear_bumper_x'] = get('robot.rear_bumper_x')
    reverse_params['half_width'] = get('robot.width') / 2.0
    reverse_params['floor_z'] = get('floor_z')

    insert_params = group('insert', (
        'speed', 'stop_tolerance', 'max_disagreement', 'max_travel',
        'max_unmeasured_frames'))
    insert_params['lidar'] = group('insert.lidar', (
        'half_width', 'min_height', 'max_height', 'min_x', 'min_points',
        'edge_bin', 'edge_min_returns'))
    insert_params['lidar']['floor_z'] = get('floor_z')
    insert_params['carriage_x'] = get('forks.carriage_x')
    insert_params['tine_length'] = get('forks.tine_length')
    insert_params['sides'] = {
        side: group(f'pallet.sides.{side}', (
            'insertion_stop_face_x', 'depth', 'insertable'))
        for side in ('short', 'long')
    }

    return {
        'limits': group('limits', ('max_linear', 'max_angular',
                                   'min_linear', 'min_angular')),
        'closing': closing_params,
        'reverse': reverse_params,
        'insert': insert_params,
    }


class DriveExecutor:
    """One drive request at a time. `start` resets everything per move."""

    def __init__(self, params: dict):
        self.params = params
        self.mode = 'STOP'
        self.side = ''
        self.distance = 0.0
        self.travelled = 0.0
        self.phase = closing.Phase.CLOSING
        self.plan: insertion.Plan | None = None
        self.unmeasured = 0
        self.done = False
        self.abort = False
        self.reason = 'idle'
        self._divergence = self._new_detector()

    def _new_detector(self):
        c = self.params['closing']
        return closing.DivergenceDetector(
            c['divergence_frames'], c['min_divergence_growth'], c['residual_target'],
            window=c.get('divergence_window'), reference_range=c.get('target_range'))

    def start(self, request: Request) -> None:
        """Begin a new move. Called when the requested mode changes.

        A repeat of the move in progress must NOT come here: restarting would
        zero `travelled` on every frame and the robot would drive for ever.
        """
        self.mode = request.mode
        self.side = request.side
        self.distance = abs(float(request.distance))
        self.travelled = 0.0
        self.phase = closing.Phase.CLOSING
        self.plan = None
        self.unmeasured = 0
        self.done = False
        self.abort = False
        self.reason = f'{request.mode} requested'
        self._divergence = self._new_detector()

    def advance(self, metres: float) -> None:
        """Add odometry path length. Always positive, whatever the direction:
        the stop rules are about how far the wheels turned."""
        self.travelled += abs(float(metres))

    # ---------------------------------------------------------------- step

    def step(
        self,
        geometry: closing.Geometry | None = None,
        lidar_points: np.ndarray | None = None,
        new_frame: bool = True,
        now: float | None = None,
    ) -> Result:
        """Decide this frame's command.

        `geometry` is this frame's pallet geometry, or None when there is no
        confirmed estimate. `new_frame` says whether a camera frame has arrived
        since the last call: the divergence detector counts FRAMES (8 at 5 Hz,
        1.6 s), and since D59 the geometry is recomputed on every 20 Hz tick,
        which would otherwise make it four times as quick to fire. `lidar_points` is the newest cloud in
        `base_footprint`, or None.
        """
        if self.mode == 'STOP':
            return Result(ZERO, False, False, self.reason)
        if self.done or self.abort:
            return Result(ZERO, self.done, self.abort, self.reason)
        if self.mode == 'CLOSE':
            return self._close(geometry, new_frame, now)
        if self.mode in ('REVERSE', 'BACKOUT'):
            return self._straight(lidar_points)
        if self.mode == 'INSERT':
            return self._insert(geometry, lidar_points)
        self.abort = True
        self.reason = f'unknown drive mode {self.mode!r}'
        return Result(ZERO, False, True, self.reason)

    # ------------------------------------------------------------- the modes

    def _close(self, geometry, new_frame: bool = True, now: float | None = None) -> Result:
        if geometry is None:
            self.reason = 'no confirmed pallet estimate'
            return Result(ZERO, False, False, self.reason,
                          phase=self.phase.value)

        diverging, detail = (self._divergence.update(geometry.offset, now, geometry.range)
                             if new_frame else (False, ''))
        if diverging:
            self.abort = True
            self.reason = f'diverging: {detail}'
            return Result(ZERO, False, True, self.reason, phase=self.phase.value,
                          face_range=geometry.range, lateral_offset=geometry.offset,
                          heading_error=geometry.heading)

        command, phase, reason = closing.closing_command(
            geometry, self.phase, self.params['closing'], self.params['limits'])
        self.phase = phase
        self.reason = reason
        self.done = phase is closing.Phase.ARRIVED
        return Result(command, self.done, False, reason, phase=phase.value,
                      face_range=geometry.range, lateral_offset=geometry.offset,
                      heading_error=geometry.heading)

    def _straight(self, lidar_points) -> Result:
        params = self.params['reverse']
        clearance, detail = closing.rear_clearance(lidar_points, params)
        remaining = self.distance - self.travelled
        if clearance is not None and clearance < remaining + float(
                params['clearance_margin']):
            self.abort = True
            self.reason = (f'obstacle {clearance:.2f} m behind and '
                           f'{remaining:.2f} m still to reverse: {detail}')
            return Result(ZERO, False, True, self.reason,
                          rear_clearance=clearance)

        command, done, reason = closing.straight_command(
            self.travelled, -self.distance, float(params['speed']),
            self.params['limits'])
        self.done = done
        self.reason = reason
        return Result(command, done, False, reason,
                      rear_clearance=float('nan') if clearance is None else clearance)

    def _insert(self, geometry, lidar_points) -> Result:
        params = self.params['insert']

        if self.plan is None:
            face = self._face_to_plan_from(geometry, lidar_points)
            if face is None:
                self.reason = 'cannot measure the face to plan the insertion'
                return Result(ZERO, False, False, self.reason)
            try:
                self.plan = insertion.plan_insertion(face, self.side, params)
            except ValueError as error:
                self.abort = True
                self.reason = f'insertion refused: {error}'
                return Result(ZERO, False, True, self.reason)
            self.reason = (f'inserting {self.plan.travel:.3f} m into the '
                           f'{self.plan.side} face')

        face_x, lateral, _ = insertion.face_distance_ahead(
            lidar_points, params['lidar'])
        self.unmeasured = 0 if face_x is not None else self.unmeasured + 1

        decision = insertion.stop_decision(
            self.travelled, face_x, self.unmeasured, self.plan, params)
        self.reason = decision.reason
        measured = dict(
            face_range=decision.face_x,
            lateral_offset=float('nan') if face_x is None else lateral)

        if decision.abort:
            self.abort = True
            self.done = True
            return Result(ZERO, True, True, decision.reason, **measured)
        if decision.stop:
            self.done = True
            return Result(ZERO, True, False, decision.reason, **measured)

        command, _, _ = closing.straight_command(
            self.travelled, self.plan.travel, float(params['speed']),
            self.params['limits'])
        return Result(command, False, False, decision.reason, **measured)

    def _face_to_plan_from(self, geometry, lidar_points) -> float | None:
        """Face distance to plan the insertion from.

        The LiDAR first, because it is the sensor that will be watching for the
        whole drive: planning from one source and stopping on another is how a
        50 mm disagreement becomes a surprise at the moment of the stop rather
        than a known offset at the start (D45). The camera is the fallback, and
        if neither has an answer the insertion is not planned at all.
        """
        face_x, _, _ = insertion.face_distance_ahead(
            lidar_points, self.params['insert']['lidar'])
        if face_x is not None:
            return face_x
        return None if geometry is None else float(geometry.range)
