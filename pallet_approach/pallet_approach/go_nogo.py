"""Safety rules: is it safe to insert the forks?

A false GO drives steel into a load; a false NO-GO wastes ten seconds. So every
rule fails closed: missing data, frames that disagree, or a measurement out of
tolerance is a NO-GO. Each rule returns `(passed, reason)`, with a specific,
quantitative reason ("yaw 7.2 deg exceeds 5.0 deg limit"), since that is what
the operator reads. Thresholds live in the `go_nogo` block of params.yaml.
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
    'evaluate_fused',
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
    """Is the robot square-on to the face? Reported in degrees.

    An angled fork sweeps across the pocket opening and uses up its width.
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
    """Does the load stick out in front of the face, above deck height, by more
    than `margin`? No `load_points` means the rule cannot be checked: NO-GO.
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
    """Is a box in front of each pocket (pocket width and height, `clearance_depth`
    deep along the normal) empty? Catches wrap, a fallen board, a second pallet.
    No data means NO-GO.
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

    # Start the box just in front of the face and just above the floor (the
    # short-side opening reaches the ground), to exclude the pallet itself.
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


def pocket_line_yaw(estimate: PalletEstimate) -> float:
    """Heading of the line through the two pocket centres, in (-pi/2, pi/2].

    Always taken pointing to +y, so frames compare whatever order the pockets
    were listed in. With fewer than two pockets, the face's along-face direction
    stands in.
    """
    if len(estimate.pockets) >= 2:
        a = np.asarray(estimate.pockets[0].center, dtype=float)[:2]
        b = np.asarray(estimate.pockets[-1].center, dtype=float)[:2]
        dx, dy = b - a
    else:
        n = np.asarray(estimate.normal, dtype=float)
        dx, dy = -n[1], n[0]
    if dy < 0 or (dy == 0 and dx < 0):
        dx, dy = -dx, -dy
    return math.atan2(dy, dx)


def check_stability(
    history: Sequence[PalletEstimate],
    required_frames: int,
    max_position_spread: float,
    max_yaw_spread: float,
) -> tuple[bool, str]:
    """Do the last `required_frames` estimates agree?

    A single frame can be plausibly wrong (a reflection, a flying-pixel edge, a
    merged cluster). Spread is max minus min: for position the bounding box's
    diagonal, for yaw the wrapped range. Fewer frames than required is a NO-GO.
    """
    if len(history) < required_frames:
        return False, (f'only {len(history)} estimate(s), needs '
                       f'{required_frames} to judge stability')

    recent = list(history)[-required_frames:]
    # D60: measured at the pocket midpoint, which the forks aim at; the face
    # centre wanders with RANSAC's inliers (56 mm parked, in the sim). Without
    # two pockets, the face centre stands in and counts against the frame.
    positions = np.array([
        np.asarray(pocket_center(e.pockets, fallback=e.position), dtype=float)
        for e in recent])
    extent = positions.max(axis=0) - positions.min(axis=0)
    spread = float(np.linalg.norm(extent))
    if spread > max_position_spread:
        return False, (
            f'pocket midpoint moved {spread * 1000:.0f} mm across '
            f'{required_frames} frames, limit {max_position_spread * 1000:.0f} mm')

    # D62 (Athreya, 2026-10-22): the direction of the line through the two
    # pocket centres, not the face plane's normal -- the pockets are the
    # reference. The face normal wobbled +-0.04 rad (+-2.3 deg) from frame to
    # frame in the sim and failed this rule on a parked robot; pocket centres
    # 0.365 m apart and steady to +-2 mm give the line to about +-0.3 deg.
    yaws = np.array([pocket_line_yaw(e) for e in recent])
    # Wrapped about the first sample, so a pair straddling +-pi does not read
    # as a 360 degree spread.
    relative = np.array([wrap_angle(y - yaws[0]) for y in yaws])
    yaw_spread = float(relative.max() - relative.min())
    if yaw_spread > max_yaw_spread:
        return False, (
            f'pocket line turned {math.degrees(yaw_spread):.1f} deg across '
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
    """Run every rule and return Decision(go, reason, checks).

    `estimate is None` is an immediate NO-GO. The rules run in a fixed order;
    the first failure gives the reason, and `checks` records every rule that
    ran, so evaluation tables can count which rule caught each case.
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
    # D63 (Athreya, 2026-10-22): switched off for now -- the pocket approaches
    # are assumed clear. As written it fires on any single point in a box that
    # starts 1-2 mm in front of the face and whose sides run along the block
    # edges; depth noise put 1-5 block-edge points 8-44 mm out in 10 of 10
    # Gazebo runs that passed every other rule.
    if params.get('check_pocket_clearance', True):
        run('pocket_clearance', check_pocket_clearance(
            estimate, scene_points, params['pocket_clearance_depth']))
    else:
        reasons['pocket_clearance'] = ('not checked '
                                       '(go_nogo.check_pocket_clearance is false, D63)')

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
    if 'pocket_clearance' not in checks:
        reason += '; pocket clearance NOT checked (D63)'
    return Decision(go=True, reason=reason, checks=checks)


def evaluate_fused(estimate: PalletEstimate | None, robot: Pose2D, track_ready: bool,
                   track_reason: str, params: dict) -> Decision:
    """GO/NO-GO with fused perception (D104): the end of a decision tree.

    Earlier stages have already answered face quality (LiDAR), pocket layout
    (camera) and stability and the camera-LiDAR cross-check (the filter's
    `track_ready`). What is left is alignment. Same shape as `evaluate`;
    overhang and clearance stay off (D61, D63) and the reason says so on GO.
    """
    checks: dict[str, bool | None] = {}
    reasons: dict[str, str] = {}
    if estimate is None:
        return Decision(go=False, reason='no pallet estimate', checks={})

    def run(name, result):
        ok, why = result
        checks[name] = bool(ok)
        reasons[name] = why

    run('track_ready', (bool(track_ready),
                        'camera and LiDAR agree on a healthy track' if track_ready
                        else f'track not ready: {track_reason}'))
    run('yaw', check_yaw(estimate, robot, params['max_yaw_error']))
    run('lateral_offset', check_lateral_offset(estimate, robot, params['max_lateral_offset']))
    for name in ('track_ready', 'yaw', 'lateral_offset'):
        if checks[name] is False:
            return Decision(go=False, reason=f'{name}: {reasons[name]}', checks=checks)
    return Decision(go=True, reason=('all checks passed (fused, D104); overhang NOT checked (D61); '
                                     'pocket clearance NOT checked (D63)'), checks=checks)


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
