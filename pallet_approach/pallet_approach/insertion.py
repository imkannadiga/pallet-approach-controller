"""Driving the virtual forks into the pockets, and knowing when to stop.

The one phase where the robot deliberately drives at something it cannot fully
see. The pockets leave the camera's view 0.58 m into the travel (D16), so the
last 0.47 m is blind to the camera; the LiDAR holds the face for all of it,
which is what makes a measured stop possible at all (D13).

Two sources say when to stop and they are kept separate on purpose:

* **odometry** -- the robot has travelled the planned distance;
* **the LiDAR** -- the face is at the planned stop distance.

D18: whichever says stop first stops the robot, and if they disagree by more
than `max_disagreement` at that moment the run aborts instead of completing.
Fail-safe in the direction of not touching the pallet, and a disagreement that
large means one of the two is wrong, which is not a condition to drive forward
in.

Angular is exactly zero throughout (D7). The tines are inside the pallet and
steering would drag them sideways through the blocks; the LiDAR's lateral
reading is logged rather than acted on, and whether that should change is a
decision for lab session C with a drift number in hand.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ['Plan', 'plan_insertion', 'face_distance_ahead', 'Stop',
           'stop_decision']


@dataclass(frozen=True)
class Plan:
    """What one insertion is meant to do, fixed when it starts."""

    start_face_x: float
    """Face distance measured at the moment insertion began, metres.

    The MEASURED one, not the configured standoff: the closing loop arrives
    within `range_tolerance` of 1.75 m, not exactly at it, and planning the
    travel from the nominal would spend that tolerance on the stop distance.
    """

    stop_face_x: float
    """Face distance to stop at. Per side (D21): 0.700 m entering the 795 mm
    face, 0.950 m entering the 1195 mm one, because the long face is only
    0.795 m deep and the short side's stop would drive the tips out of the
    back."""

    travel: float
    """How far to drive, metres. `start_face_x - stop_face_x`."""

    depth: float
    """How far the tine tips end up inside the pallet, metres. Reported so a
    run can say what it achieved rather than that it finished."""

    side: str


def plan_insertion(start_face_x: float, side: str, params: dict) -> Plan:
    """Work out the travel for one insertion, or refuse to plan it.

    Refuses rather than clamps in three cases, each of which means the robot is
    not where the sequence believes it is:

    * the face is already at or inside the stop distance -- there is nothing to
      drive;
    * the travel is longer than `max_travel`, which means the robot is further
      from the pallet than any standoff would put it;
    * the resulting depth would push the tips out of the back of the pallet.
    """
    sides = params['sides']
    if side not in sides:
        raise ValueError(f'no geometry configured for side {side!r}; '
                         f'have {sorted(sides)}')
    geometry = sides[side]
    if not geometry.get('insertable', True):
        raise ValueError(f'insertion into the {side} face is not permitted '
                         f'(pallet.sides.{side}.insertable is false)')

    stop = float(geometry['insertion_stop_face_x'])
    travel = float(start_face_x) - stop
    if travel <= 0.0:
        raise ValueError(
            f'the face is already at {start_face_x:.3f} m, at or inside the '
            f'{stop:.3f} m stop for the {side} face')
    if travel > float(params['max_travel']):
        raise ValueError(
            f'{travel:.3f} m of insertion travel is more than the '
            f'{float(params["max_travel"]):.3f} m limit; the robot is not at a '
            f'standoff')

    reach = float(params['carriage_x']) + float(params['tine_length'])
    depth = reach - stop
    if depth >= float(geometry['depth']):
        raise ValueError(
            f'stopping at {stop:.3f} m inserts {depth:.3f} m into a '
            f'{geometry["depth"]:.3f} m deep face: the tips come out of the back')
    return Plan(start_face_x=float(start_face_x), stop_face_x=stop,
                travel=travel, depth=depth, side=side)


def face_distance_ahead(
    points: np.ndarray | None,
    params: dict,
) -> tuple[float | None, float, str]:
    """Distance to the nearest surface ahead, from the LiDAR.

    Returns (distance, lateral, detail). `distance` is None when there is
    nothing to measure -- which is NOT "the way is clear" and must never be
    read as one; the caller counts consecutive Nones and aborts.

    `lateral` is the mean lateral position of the returns, logged so that lab
    session C has a drift number for the decision about whether insertion
    should track the face (ROADMAP stage 6). It is NaN when nothing was
    measured.

    The corridor is narrow (`half_width`) and low (`min_height` to
    `max_height`, i.e. floor-clear to just above a pallet), so the surface it
    finds is the pallet face or something standing in front of it. Either is a
    reason to stop, which is why no attempt is made to tell them apart.

    The statistic is a low percentile of x rather than the minimum: one stray
    return 0.1 m early would otherwise stop the insertion short every time.
    """
    if points is None or len(points) == 0:
        return None, float('nan'), 'no LiDAR data'
    p = np.asarray(points, dtype=float)
    heights = p[:, 2] - float(params['floor_z'])
    ahead = (
        (p[:, 0] > float(params['min_x']))
        & (np.abs(p[:, 1]) <= float(params['half_width']))
        & (heights >= float(params['min_height']))
        & (heights <= float(params['max_height']))
    )
    count = int(ahead.sum())
    if count < int(params['min_points']):
        return None, float('nan'), (
            f'only {count} returns in the corridor ahead, need '
            f'{int(params["min_points"])}')
    selected = p[ahead]
    distance = float(np.percentile(selected[:, 0], 10))
    lateral = float(selected[:, 1].mean())
    return distance, lateral, (
        f'{count} returns, face at {distance:.3f} m, '
        f'lateral {lateral * 1000:+.0f} mm')


@dataclass(frozen=True)
class Stop:
    """Whether to stop driving, and whether the run can still complete."""

    stop: bool
    abort: bool
    reason: str
    source: str
    """Which source fired: 'odometry', 'lidar', 'both' or '' while driving."""

    face_x: float
    """Best estimate of the face distance now, metres: the LiDAR's if it has
    one, otherwise odometry's."""


def stop_decision(
    travelled: float,
    lidar_face_x: float | None,
    unmeasured_frames: int,
    plan: Plan,
    params: dict,
) -> Stop:
    """D18, as one function. Called once per frame while inserting.

    `travelled` is odometry's path length since insertion began, always
    positive. `lidar_face_x` is this frame's measurement or None.
    `unmeasured_frames` is how many frames in a row have come back None.
    """
    odometry_face = plan.start_face_x - float(travelled)
    tolerance = float(params['stop_tolerance'])
    by_odometry = travelled >= plan.travel - tolerance
    by_lidar = (lidar_face_x is not None
                and lidar_face_x <= plan.stop_face_x + tolerance)

    if lidar_face_x is None:
        # The LiDAR is the only thing watching where the robot is going. A few
        # frames without a measurement is a sensor hiccup; a run of them during
        # a blind drive at a pallet is a reason to stop, not to keep going and
        # trust the wheels.
        if unmeasured_frames >= int(params['max_unmeasured_frames']):
            return Stop(stop=True, abort=True, source='lidar',
                        face_x=odometry_face,
                        reason=(f'no LiDAR measurement of the face for '
                                f'{unmeasured_frames} frames while inserting; '
                                f'odometry alone says {odometry_face:.3f} m'))
        if by_odometry:
            return Stop(stop=True, abort=True, source='odometry',
                        face_x=odometry_face,
                        reason=(f'odometry says the stop is reached at '
                                f'{odometry_face:.3f} m but the LiDAR has no '
                                f'measurement to confirm it'))
        return Stop(stop=False, abort=False, source='', face_x=odometry_face,
                    reason=f'{travelled:.3f} / {plan.travel:.3f} m, no LiDAR')

    disagreement = abs(odometry_face - lidar_face_x)
    if by_odometry or by_lidar:
        source = ('both' if by_odometry and by_lidar
                  else 'odometry' if by_odometry else 'lidar')
        if disagreement > float(params['max_disagreement']):
            return Stop(
                stop=True, abort=True, source=source, face_x=lidar_face_x,
                reason=(f'stopping on {source}, but odometry says the face is '
                        f'at {odometry_face:.3f} m and the LiDAR says '
                        f'{lidar_face_x:.3f} m -- {disagreement * 1000:.0f} mm '
                        f'apart, over the '
                        f'{float(params["max_disagreement"]) * 1000:.0f} mm limit'))
        return Stop(stop=True, abort=False, source=source, face_x=lidar_face_x,
                    reason=(f'stopped on {source}: face at {lidar_face_x:.3f} m, '
                            f'travelled {travelled:.3f} m, tips {plan.depth:.3f} m '
                            f'into the pallet'))

    return Stop(stop=False, abort=False, source='', face_x=lidar_face_x,
                reason=(f'{travelled:.3f} / {plan.travel:.3f} m, face at '
                        f'{lidar_face_x:.3f} m, sources '
                        f'{disagreement * 1000:+.0f} mm apart'))
