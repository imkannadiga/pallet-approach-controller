"""A whole virtual pickup, run at a desk.

Wires together everything that is not ROS: the coordinator's state machine, the
real tracker, the real drive executor, the real fork simulator and the real
GO/NO-GO rules, driving the robot in `simulate.py`'s world.

This is the harness the project's third recurring bug class needed -- the join
between two things that each test clean. Three runtime failures in a row came
from joins nothing checked (a parameter declared on one side only, a message
field assigned but not declared, a tool reading a key that had moved), and the
join between the sequence and the control laws is the biggest one left: the
coordinator says CLOSE and the executor has to be in a state where CLOSE means
something.

What it still does not include: the ROS layer itself -- discovery, QoS,
lifecycle transitions, tf and the clock. Every failure on the robot so far has
come from exactly that layer, so passing here is necessary and nowhere near
sufficient. It is the offline replay required before anything drives,
not a substitute for lab session B.
"""

from __future__ import annotations

import math
import pathlib
import sys
from dataclasses import dataclass, field

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import simulate as S  # noqa: E402

from pallet_approach import go_nogo  # noqa: E402
from pallet_approach.closing import geometry_of  # noqa: E402
from pallet_approach.coordinator import (  # noqa: E402
    machine_params,
    Coordinator,
    DriveInfo,
    ForkInfo,
    State,
    TrackInfo,
)
from pallet_approach.drive import DriveExecutor, Request, executor_params  # noqa: E402
from pallet_approach.forks import ForkSimulator  # noqa: E402
from pallet_approach.model import Pose2D  # noqa: E402
from pallet_approach.tracker import Tracker  # noqa: E402

PARAMS_FILE = pathlib.Path(__file__).resolve().parent.parent / 'config' / 'params.yaml'


def load() -> dict:
    """Every block of params.yaml, flattened the way the nodes flatten it.

    Built from the real file so that a parameter change is felt here. The
    flattening duplicates what each node's `_params()` does, which is itself
    checked against the YAML by `test_params.py` -- so the chain from the file
    to this harness is covered end to end even though no ROS is involved.
    """
    raw = yaml.safe_load(PARAMS_FILE.read_text())
    shared = raw['/**']['ros__parameters']
    controller = raw['approach_controller']['ros__parameters']
    coordinator = raw['coordinator']['ros__parameters']
    forks = raw['virtual_forks']['ros__parameters']
    tracker = raw['pallet_tracker']['ros__parameters']

    # The machine and the executor get EXACTLY what their nodes build, from
    # the same functions: `machine_params` and `executor_params` take a getter,
    # and this one reads params.yaml the way ROS would -- the shared block
    # under the node's own. Copying whole YAML blocks instead (as this did
    # until 2026-10-20) handed the harness keys the nodes never passed on.
    def getter(node_block):
        flat = _flatten(raw['/**']['ros__parameters'])
        flat.update(_flatten(node_block))
        return flat.__getitem__

    machine = machine_params(getter(coordinator))
    drive = executor_params(getter(controller))

    tracker_params = dict(
        {k: v for k, v in tracker.items() if k != 'lidar'},
        floor_z=shared['floor_z'],
        pallet_height=shared['pallet']['pallet_height'],
        **tracker['lidar'])

    return {
        'machine': machine,
        'drive': drive,
        'forks': dict(forks, insert_height=shared['forks']['bottom_height'],
                      lift_travel=shared['forks']['lift_travel'],
                      tine_thickness=shared['forks']['tine_thickness']),
        'tracker': tracker_params,
        'go_nogo': dict(coordinator['go_nogo'], sides=shared['pallet']['sides']),
        'shared': shared,
    }



def _flatten(block, prefix=''):
    flat = {}
    for key, value in (block or {}).items():
        if isinstance(value, dict):
            flat.update(_flatten(value, f'{prefix}{key}.'))
        else:
            flat[f'{prefix}{key}'] = value
    return flat


@dataclass
class Event:
    """One state change, for the trace."""

    seconds: float
    state: str
    reason: str


@dataclass
class Run:
    """What a simulated run did."""

    outcome: str
    reason: str
    seconds: float
    states: list = field(default_factory=list)
    trace: list = field(default_factory=list)
    range: float = float('nan')
    offset: float = float('nan')
    heading: float = float('nan')
    travelled: float = 0.0
    insertion_depth: float = 0.0
    loaded: bool = False
    commands: list = field(default_factory=list)

    def visited(self, name: str) -> bool:
        return name in self.states

    def describe(self) -> str:
        lines = [f'{event.seconds:7.2f}s  {event.state:<12} {event.reason}'
                 for event in self.trace]
        lines.append(f'{"":8}  outcome {self.outcome}: {self.reason}')
        return '\n'.join(lines)


def lidar_for(robot: S.Robot, pallet: S.Pallet, columns: int = 40,
              rear: np.ndarray | None = None) -> np.ndarray:
    """A plausible Ouster frame in `base_footprint`: the pallet face and
    whatever else the caller wants behind the robot.

    Two beams across the face, which is roughly what the geometry gives between
    1 and 4 m (D13). Deliberately simple: this harness is about the sequence,
    and `test_tracker.py` and `test_insertion.py` are where the LiDAR windows
    themselves are tested.
    """
    geometry = geometry_of(S.truth(robot, pallet))
    normal = np.array([math.cos(geometry.heading + math.pi),
                       math.sin(geometry.heading + math.pi), 0.0])
    # the face centre in the robot's frame
    estimate = S.truth(robot, pallet)
    center = np.asarray(estimate.position, dtype=float)
    axis = np.array([-estimate.normal[1], estimate.normal[0], 0.0])
    offsets = np.linspace(-0.30, 0.30, columns)
    points = [center + axis * o + np.array([0.0, 0.0, h - estimate.position[2]])
              for h in (0.05 + S.FLOOR.height_on_axis, 0.11 + S.FLOOR.height_on_axis)
              for o in offsets]
    cloud = np.array(points)
    del normal
    if rear is not None and len(rear):
        cloud = np.vstack([cloud, rear])
    return cloud


class Rig:
    """Everything but ROS, stepped at the real rates."""

    def __init__(self, params: dict, pallet: S.Pallet | None = None,
                 noise: float = 0.0, seed: int = 0, rear: np.ndarray | None = None,
                 lidar: bool = True, obstruction: np.ndarray | None = None):
        self.params = params
        self.pallet = pallet or S.Pallet()
        self.machine = Coordinator(params['machine'])
        self.tracker = Tracker(params['tracker'])
        self.executor = DriveExecutor(params['drive'])
        self.forks = ForkSimulator(params['forks'])
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.rear = rear
        self.lidar = lidar
        self.obstruction = obstruction
        """Extra points added to the cluster and the scene, in the ROBOT's
        frame at the moment of the decision -- for the tests that put a load
        over the face or a box in a pocket."""
        self.estimates: list = []
        """Estimates taken since CHECKING began, mirroring what the
        coordinator node keeps. Frames taken while closing span real travel and
        the stability rule reads that as an unstable estimate."""
        self.requested = None
        self.commands: list = []
        """Every (linear, angular) actually sent, so a test can assert that no
        run ever exceeded the limits rather than trusting the clamp."""

    def run(self, robot: S.Robot, frame_rate: float = 5.0,
            command_rate: float = 20.0, limit: float = 400.0,
            mode: str = 'pickup') -> Run:
        dt = 1.0 / command_rate
        frame_dt = 1.0 / frame_rate
        command = (0.0, 0.0)
        now, next_frame = 0.0, 0.0
        states, trace = [], []
        last_state = None
        odom = Pose2D(robot.x, robot.y, robot.yaw)

        self._note(self.machine.trigger(now, mode), states, trace, now)

        while now < limit and self.machine.state not in (State.DONE,
                                                         State.ABORTED):
            if now >= next_frame:
                command = self._frame(robot, odom, now, states, trace)
                next_frame += frame_dt
            before = (robot.x, robot.y)
            robot.step(command[0], command[1], dt)
            self.executor.advance(math.hypot(robot.x - before[0],
                                             robot.y - before[1]))
            self.forks.update(now)
            odom = Pose2D(robot.x, robot.y, robot.yaw)
            now += dt
            if self.machine.state is not last_state:
                last_state = self.machine.state

        final = geometry_of(S.truth(robot, self.pallet))
        return Run(
            outcome=(self.machine.outcome.value if self.machine.outcome
                     else 'TIMEOUT'),
            reason=self.machine.last_reason, seconds=now, states=states,
            trace=trace, range=final.range, offset=final.offset,
            heading=final.heading, travelled=robot.travelled,
            insertion_depth=self.machine.insertion_travel,
            loaded=self.forks.loaded, commands=list(self.commands))

    # ------------------------------------------------------------ one frame

    def _frame(self, robot, odom, now, states, trace):
        estimate = S.observe(robot, self.pallet, noise=self.noise, rng=self.rng)
        cloud = (lidar_for(robot, self.pallet, rear=self.rear)
                 if self.lidar else None)

        self.tracker.update(estimate, odom, now, lidar_points=cloud)
        track = self.tracker.track
        geometry = geometry_of(estimate)
        if self.machine.state is State.CHECKING and self.tracker.confirmed:
            self.estimates.append(estimate)

        self._note(self.machine.on_track(TrackInfo(
            confirmed=self.tracker.confirmed,
            confidence=self.tracker.confidence,
            side=estimate.side if track else 'unknown',
            range=geometry.range, offset=geometry.offset,
            heading=geometry.heading, reason=self.tracker.reason), now),
            states, trace, now)

        self._note(self.machine.on_forks(ForkInfo(
            state=self.forks.state.value, height=self.forks.height,
            moving=self.forks.moving, loaded=self.forks.loaded), now),
            states, trace, now)

        if self.machine.ready_to_decide(now):
            # The point sets two of the nine rules need, which perception
            # publishes on the robot and which a harness that leaves them out
            # cannot reach GO without -- both treat missing data as a NO-GO.
            cluster = S.pallet_points(robot, self.pallet, estimate=estimate)
            decision = go_nogo.evaluate(
                estimate, Pose2D(0.0, 0.0, 0.0),
                self.estimates[-int(self.params['go_nogo']['stability_frames']):],
                self.params['go_nogo'],
                load_points=np.vstack([cluster, self.obstruction])
                if self.obstruction is not None else cluster,
                scene_points=np.vstack([cluster, self.obstruction])
                if self.obstruction is not None else cluster)
            self._note(self.machine.on_decision(decision, now), states, trace, now)

        self._note(self.machine.on_tick(now), states, trace, now)

        request = self.machine._drive_for(self.machine.state)
        if request is None:
            self.executor.start(Request('STOP'))
            self.requested = None
            return (0.0, 0.0)

        if (self.requested is None or request.mode != self.requested.mode):
            self.executor.start(Request(request.mode, request.distance,
                                        request.side))
            self.requested = request

        result = self.executor.step(
            geometry=geometry if self.tracker.confirmed else None,
            lidar_points=cloud)
        self.commands.append((result.command.linear, result.command.angular))
        self._note(self.machine.on_drive(DriveInfo(
            mode=request.mode, arrived=result.arrived, blocked=result.blocked,
            abort=result.abort, reason=result.reason,
            travelled=self.executor.travelled), now), states, trace, now)
        return (result.command.linear, result.command.angular)

    def _note(self, transition, states, trace, now):
        if transition.changed:
            states.append(transition.state.value)
            trace.append(Event(now, transition.state.value, transition.reason))
            if transition.state is State.CHECKING:
                self.estimates.clear()
            if transition.fork == 'LOWER':
                self.forks.lower_to_insert(now)
            elif transition.fork == 'LIFT':
                self.forks.lift(now)
            elif transition.fork == 'PARK':
                self.forks.park(now)
