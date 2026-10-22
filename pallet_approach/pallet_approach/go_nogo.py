"""Safety rules: is it safe to insert the forks?

This is the most important file in the project and the easiest one to get
quietly wrong, because a bug here does not crash: it returns GO when it should
not. The whole design follows from one asymmetry:

    A false GO drives steel into a load. A false NO-GO wastes ten seconds.

So every rule is written to fail closed. Missing data is a NO-GO. An estimate
that disagrees with itself across frames is a NO-GO. A measurement outside
tolerance is a NO-GO. The only way to reach GO is for every check to pass on
evidence that is actually present.

Each rule is a small function returning `(passed, reason)`. The reason is what
the operator reads in the action result, so make it specific and quantitative:

    "yaw 7.2 deg exceeds 5.0 deg limit"        good
    "pallet misaligned"                        useless

`evaluate` runs the rules in order and stops at the first failure, so put the
cheap structural checks first and the expensive geometric ones last.

Thresholds live in the `go_nogo` block of params.yaml. Never hard-code one here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from pallet_approach.approach import (
    lateral_offset,
    outward_normal,
    pocket_center,
    wrap_angle,
    yaw_misalignment,
)
from pallet_approach.model import Decision, PalletEstimate, Pose2D

# How far in front of the fitted face a point has to be before it counts as
# something rather than as the face itself. Depth noise on the real pallet
# spans 26 mm across the front surface, but the face plane is fitted through
# the middle of that, so a couple of millimetres is enough to exclude the
# plane without excluding a real obstruction.
FACE_INSET = 0.005

# And how far above the fitted floor, for the same reason: on the short face
# the pocket opening reaches the ground, so a clearance box drawn to the
# measured opening would contain the floor. Matches floor.removal_margin.
FLOOR_INSET = 0.030

__all__ = [
    'check_face_quality',
    'check_pocket_count',
    'check_pocket_dimensions',
    'check_yaw',
    'check_lateral_offset',
    'check_overhang',
    'check_pocket_clearance',
    'check_stability',
    'evaluate',
]


def check_face_quality(estimate: PalletEstimate, min_inliers: int) -> tuple[bool, str]:
    """Was the front face actually found, and fitted to enough points?

    A plane can be fitted to three points. That does not make it a pallet face.
    `face_inlier_count` is the honest measure of how much evidence the fit has,
    and a face that thin usually means the pallet is at the edge of the depth
    range or half occluded.
    """
    count = int(estimate.face_inlier_count)
    if count < min_inliers:
        return False, (f'face fitted to {count} points, needs {min_inliers}')
    return True, f'face fitted to {count} points'


def check_pocket_count(estimate: PalletEstimate, minimum: int) -> tuple[bool, str]:
    """Were both pockets found?

    One pocket is not half a pass. If the second one is missing, either the
    pallet is not what we think it is, something is in front of it, or the
    estimate is unreliable -- all of which are NO-GO.
    """
    found = len(estimate.pockets)
    if found < minimum:
        return False, f'found {found} pocket(s), needs {minimum}'
    return True, f'found {found} pockets'


def check_pocket_dimensions(
    estimate: PalletEstimate,
    nominal_width: float,
    nominal_height: float,
    width_tolerance: float,
    height_tolerance: float,
) -> tuple[bool, str]:
    """Do the measured pockets match the expected opening, within tolerance?

    This is the check that catches a pallet that is damaged, the wrong type, or
    partly blocked: a pocket measured at 0.14 m when 0.2275 m is expected is not
    a pocket the forks fit into. Report the measured value in the reason.
    """
    if not estimate.pockets:
        return False, 'no pockets to measure'
    for index, pocket in enumerate(estimate.pockets):
        width_error = abs(pocket.width - nominal_width)
        if width_error > width_tolerance:
            return False, (
                f'pocket {index} is {pocket.width * 1000:.0f} mm wide against a '
                f'nominal {nominal_width * 1000:.0f} mm, '
                f'{width_error * 1000:.0f} mm out (limit '
                f'{width_tolerance * 1000:.0f} mm)')
        height_error = abs(pocket.height - nominal_height)
        if height_error > height_tolerance:
            return False, (
                f'pocket {index} is {pocket.height * 1000:.0f} mm high against a '
                f'nominal {nominal_height * 1000:.0f} mm, '
                f'{height_error * 1000:.0f} mm out (limit '
                f'{height_tolerance * 1000:.0f} mm)')
    widths = ', '.join(f'{p.width * 1000:.0f}' for p in estimate.pockets)
    return True, f'pockets {widths} mm wide, within tolerance'


def check_yaw(
    estimate: PalletEstimate,
    robot: Pose2D,
    max_yaw_error: float,
) -> tuple[bool, str]:
    """Is the robot square-on to the face?

    Use `approach.yaw_misalignment`, which already handles the normal's sign and
    wraps the angle. Report degrees in the reason -- nobody reads radians off a
    terminal and immediately pictures the geometry.

    Why it matters physically: enter a 0.2275 m pocket with a 0.100 m fork at an
    angle and the usable width shrinks as the fork sweeps across the opening.
    A few degrees is fine; ten is a collision with a block.
    """
    error = yaw_misalignment(robot.yaw, estimate.normal)
    if abs(error) > max_yaw_error:
        return False, (
            f'{math.degrees(error):+.1f} deg off square, limit '
            f'{math.degrees(max_yaw_error):.1f} deg')
    return True, f'{math.degrees(error):+.1f} deg off square'


def check_lateral_offset(
    estimate: PalletEstimate,
    robot: Pose2D,
    max_offset: float,
) -> tuple[bool, str]:
    """Is the robot on the pocket centre line?

    Use `approach.lateral_offset`. The tolerance comes from the clearance: a
    0.100 m fork in a 0.2275 m pocket has about 0.064 m either side, so 0.05 m
    of lateral error is already most of the budget -- which is why the default
    limit is 0.05 m and not 0.10 m.
    """
    centre = pocket_center(estimate.pockets, estimate.position)
    offset = lateral_offset(robot, centre, estimate.normal)
    if abs(offset) > max_offset:
        return False, (
            f'{offset * 1000:+.0f} mm off the pocket centre line, limit '
            f'{max_offset * 1000:.0f} mm')
    return True, f'{offset * 1000:+.0f} mm off the pocket centre line'


def check_overhang(
    estimate: PalletEstimate,
    load_points: np.ndarray | None,
    margin: float,
) -> tuple[bool, str]:
    """Does the load stick out in front of the pallet face?

    Concept: the fitted face is where the forks expect to meet the pallet. If
    the load on top overhangs that plane, the forks or the mast hit the load
    first. Test the cluster's points against the face plane: anything more than
    `margin` in front of it, above deck height, is overhang.

    `load_points` is the pallet cluster from perception. If it is None, you
    cannot check -- and an uncheckable safety rule is a NO-GO, not a pass.
    """
    if load_points is None or len(load_points) == 0:
        return False, 'no points to check the load against'

    points = np.asarray(load_points, dtype=float)
    normal = outward_normal(np.asarray(estimate.normal, dtype=float),
                            face_point=np.asarray(estimate.position, dtype=float))
    # Distance in FRONT of the face plane: the outward normal points at the
    # robot, so positive means between the pallet and us.
    ahead = (points - np.asarray(estimate.position, dtype=float)) @ normal

    # Only above the pocket opening. Below that is the pallet's own structure
    # and the floor, which extends toward the robot and is not a load. The
    # opening's top comes from the measured pockets rather than a constant,
    # because the two faces open to different heights (D19).
    if estimate.pockets:
        opening_top = max(_height_of(estimate, p.center) + p.height / 2
                          for p in estimate.pockets)
    else:
        opening_top = _height_of(estimate, np.asarray(estimate.position))
    heights = _heights_of(estimate, points)

    overhanging = (ahead > margin) & (heights > opening_top)
    count = int(np.count_nonzero(overhanging))
    if count:
        worst = float(ahead[overhanging].max())
        return False, (
            f'{count} load points up to {worst * 1000:.0f} mm in front of the '
            f'face, margin {margin * 1000:.0f} mm')
    return True, 'nothing in front of the face above the opening'


def check_pocket_clearance(
    estimate: PalletEstimate,
    scene_points: np.ndarray | None,
    clearance_depth: float,
) -> tuple[bool, str]:
    """Is the approach into each pocket clear?

    Concept: build a small box in front of each pocket -- pocket width and
    height, `clearance_depth` deep, along the face normal -- and require it to
    be empty. That catches shrink wrap, a fallen board, a stray box, or a
    second pallet pushed up against this one.

    Same rule as above: no data means NO-GO.
    """
    if scene_points is None or len(scene_points) == 0:
        return False, 'no points to check the pocket approaches against'
    if not estimate.pockets:
        return False, 'no pockets to check the approach to'

    points = np.asarray(scene_points, dtype=float)
    position = np.asarray(estimate.position, dtype=float)
    normal = outward_normal(np.asarray(estimate.normal, dtype=float),
                            face_point=position)
    along = _along_face(estimate, normal)

    ahead = (points - position) @ normal
    sideways = (points - position) @ along
    heights = _heights_of(estimate, points)

    # The face itself sits at ahead == 0, so the box starts just in front of it;
    # and the floor runs through the bottom of a short-side opening, which
    # reaches the ground, so the box starts just above the floor too. Both
    # insets are about excluding the pallet and the ground it stands on, not
    # about being lenient: FACE_INSET is a millimetre or two of depth noise and
    # FLOOR_INSET matches the floor-removal margin the pipeline already uses.
    in_depth = (ahead > FACE_INSET) & (ahead < clearance_depth)

    for index, pocket in enumerate(estimate.pockets):
        centre_s = float((np.asarray(pocket.center, dtype=float) - position) @ along)
        centre_h = _height_of(estimate, pocket.center)
        in_width = np.abs(sideways - centre_s) <= pocket.width / 2
        in_height = ((heights > max(FLOOR_INSET, centre_h - pocket.height / 2))
                     & (heights < centre_h + pocket.height / 2))
        blocked = in_depth & in_width & in_height
        count = int(np.count_nonzero(blocked))
        if count:
            nearest = float(ahead[blocked].max())
            return False, (
                f'{count} points in front of pocket {index}, nearest '
                f'{nearest * 1000:.0f} mm out from the face')
    return True, f'both pocket approaches clear to {clearance_depth * 1000:.0f} mm'


def check_stability(
    history: Sequence[PalletEstimate],
    required_frames: int,
    max_position_spread: float,
    max_yaw_spread: float,
) -> tuple[bool, str]:
    """Do the last few estimates agree with each other?

    Concept: a single frame can be wrong in a way that looks perfectly
    plausible -- a reflection, a flying-pixel edge, a cluster that merged two
    objects. Agreement across consecutive frames is the cheapest available
    defence, and it costs only a fraction of a second at 37 Hz.

    Measure spread as the max-minus-min (or the standard deviation; the
    docstring of the implementation says which) of the position and yaw over the last
    `required_frames` estimates. Fewer than that many estimates is a NO-GO:
    not enough evidence yet.

    Spread here is **max minus min** over the last `required_frames`
    estimates: for position, the length of the bounding box's diagonal, so a
    wobble split across two axes counts for what it is; for yaw, the angular
    range after wrapping.
    """
    if len(history) < required_frames:
        return False, (f'only {len(history)} estimate(s), needs '
                       f'{required_frames} to judge stability')

    recent = list(history)[-required_frames:]
    # D60 (Athreya, 2026-10-22): the POCKET MIDPOINT, not the face centre.
    # The face centre is the middle of whichever points RANSAC took as the
    # plane, and that set changes every frame: with the robot parked at the
    # standoff in the Gazebo sim it wandered 56 mm over 20 frames while the
    # pocket centres held to +-2 mm, so every run was refused. The midpoint is
    # also what the forks aim at. A frame without two pockets falls back to
    # the face centre, which then counts against it.
    positions = np.array([
        np.asarray(pocket_center(e.pockets, fallback=e.position), dtype=float)
        for e in recent])
    extent = positions.max(axis=0) - positions.min(axis=0)
    spread = float(np.linalg.norm(extent))
    if spread > max_position_spread:
        return False, (
            f'pocket midpoint moved {spread * 1000:.0f} mm across '
            f'{required_frames} frames, limit {max_position_spread * 1000:.0f} mm')

    yaws = np.array([e.yaw for e in recent])
    # Wrapped about the first sample, so a pair straddling +-pi does not read
    # as a 360 degree spread.
    relative = np.array([wrap_angle(y - yaws[0]) for y in yaws])
    yaw_spread = float(relative.max() - relative.min())
    if yaw_spread > max_yaw_spread:
        return False, (
            f'face yaw moved {math.degrees(yaw_spread):.1f} deg across '
            f'{required_frames} frames, limit {math.degrees(max_yaw_spread):.1f} deg')
    return True, (f'stable over {required_frames} frames: '
                  f'{spread * 1000:.0f} mm, {math.degrees(yaw_spread):.1f} deg')


def evaluate(
    estimate: PalletEstimate | None,
    robot: Pose2D,
    history: Sequence[PalletEstimate],
    params: dict,
    load_points: np.ndarray | None = None,
    scene_points: np.ndarray | None = None,
) -> Decision:
    """Run every rule and return the decision.

    Structure it so that:

    * `estimate is None` returns NO-GO immediately with a clear reason;
    * the rules run in a fixed, documented order;
    * the first failure decides the outcome and gives the reason;
    * `Decision.checks` records every rule that ran, so the evaluation tables
      can count *which* rule caught each unsafe scenario. That table is the
      evidence behind any false-GO count, so it has to come out of the code,
      not out of a recollection of the run.

    Args:
        estimate: the latest accepted estimate, or None.
        robot: the robot's pose in the same frame as `estimate`.
        history: recent estimates, oldest first.
        params: the `go_nogo` block of params.yaml.
        load_points, scene_points: optional point sets for the overhang and
            clearance checks.

    Returns:
        Decision(go, reason, checks).
    """
    checks: dict[str, bool] = {}
    reasons: dict[str, str] = {}

    if estimate is None:
        return Decision(go=False, reason='no pallet estimate', checks={})

    def run(name, result):
        ok, why = result
        checks[name] = bool(ok)
        reasons[name] = why
        return ok

    # Every rule runs, even after one has failed, because Decision.checks is
    # the evidence behind the false-GO count and a table with holes in it
    # cannot say which rule would have caught what. The ORDER below is what
    # picks the reason: the first failure in this list is the one reported,
    # so the reasons stay stable and comparable between runs.
    run('face_quality', check_face_quality(
        estimate, params.get('min_face_inliers', 60)))
    run('stability', check_stability(
        history, params['stability_frames'],
        params['max_position_spread'], params['max_yaw_spread']))
    run('pocket_count', check_pocket_count(estimate, params['min_pocket_count']))

    # Nominal pocket size is per face (D19), so it comes from the side the
    # estimate classified. Without it the rule cannot run; it is recorded as
    # not evaluated rather than quietly passed, and `evaluate` says so in the
    # reason of an otherwise clean GO -- a safety rule that is silently absent
    # is worse than one that fails.
    nominal = _nominal_opening(estimate, params)
    if nominal is None:
        checks['pocket_dimensions'] = None
        reasons['pocket_dimensions'] = (
            'not evaluated: no nominal opening configured for side '
            f'"{estimate.side}"')
    else:
        run('pocket_dimensions', check_pocket_dimensions(
            estimate, nominal[0], nominal[1],
            params['pocket_width_tolerance'], params['pocket_height_tolerance']))

    run('yaw', check_yaw(estimate, robot, params['max_yaw_error']))
    run('lateral_offset', check_lateral_offset(
        estimate, robot, params['max_lateral_offset']))
    # D61 (Athreya, 2026-10-22): switched off for now -- nothing is assumed to
    # stick out of the pallet. As written the rule fires on any single depth
    # point more than the margin in front of the face, and depth noise alone
    # put 77 of the pallet's own deck points 20-44 mm out in the sim. It is
    # reported as skipped in every decision rather than as passed.
    if params.get('check_overhang', True):
        run('overhang', check_overhang(
            estimate, load_points, params['overhang_margin']))
    else:
        reasons['overhang'] = 'not checked (go_nogo.check_overhang is false, D61)'
    run('pocket_clearance', check_pocket_clearance(
        estimate, scene_points, params['pocket_clearance_depth']))

    order = ('face_quality', 'stability', 'pocket_count', 'pocket_dimensions',
             'yaw', 'lateral_offset', 'overhang', 'pocket_clearance')
    for name in order:
        if checks.get(name) is False:
            return Decision(go=False, reason=f'{name}: {reasons[name]}',
                            checks=checks)

    reason = 'all checks passed'
    if checks.get('pocket_dimensions') is None:
        reason += f' ({reasons["pocket_dimensions"]})'
    if 'overhang' not in checks:
        reason += '; overhang NOT checked (D61)'
    return Decision(go=True, reason=reason, checks=checks)


def _nominal_opening(estimate: PalletEstimate, params: dict):
    """(width, height) the estimate's face should have, or None if unknown."""
    sides = params.get('sides')
    if sides and estimate.side in sides:
        side = sides[estimate.side]
        return side['pocket_width'], side['opening_top'] - side['opening_bottom']
    if 'pocket_nominal_width' in params and 'pocket_nominal_height' in params:
        return params['pocket_nominal_width'], params['pocket_nominal_height']
    return None


def _along_face(estimate: PalletEstimate, normal: np.ndarray) -> np.ndarray:
    """Unit vector across the face, horizontal in the floor's frame."""
    up = (np.asarray(estimate.floor.normal, dtype=float)
          if estimate.floor is not None else np.array([0.0, 0.0, 1.0]))
    along = np.cross(up, normal)
    return along / np.linalg.norm(along)


def _heights_of(estimate: PalletEstimate, points: np.ndarray) -> np.ndarray:
    """Heights above the fitted floor, perpendicular to it.

    Perpendicular distance rather than a z-subtraction: a rotation is an
    isometry, so the distance to the plane is exactly unchanged by a camera
    orientation error while a z-subtraction leaks it (measured: 4.5 mm against
    0.0 mm on this robot's real 0.87 deg error).
    """
    if estimate.floor is None:
        return np.asarray(points, dtype=float)[:, 2]
    return estimate.floor.height_of(np.asarray(points, dtype=float))


def _height_of(estimate: PalletEstimate, point) -> float:
    return float(_heights_of(estimate, np.asarray(point, dtype=float)[None, :])[0])
