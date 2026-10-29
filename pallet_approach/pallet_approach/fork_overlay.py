"""Draw the imaginary forks (and detected pockets) on the camera image.

Doubles as a visible calibration check: a wrong camera-to-base transform
makes the forks miss the pockets. Geometry is defined in `base_footprint`,
carried into the camera optical frame (tf at the image stamp), projected with
the pinhole intrinsics (rectified image, D = 0) and drawn as semi-transparent
polygons. Points behind the camera (z <= 0) are skipped; there is no
occlusion test against the scene.
"""

from __future__ import annotations

import numpy as np

from pallet_approach.model import PalletEstimate, Plane, Pocket
from pallet_approach.perception import transform_points
from pallet_approach.projection import face_corners, pocket_corners, project

__all__ = [
    'tine_corners',
    'to_camera_frame',
    'project_points',
    'draw_polygons',
    'draw_pockets',
    'render',
]


def tine_corners(
    tine_length: float,
    tine_width: float,
    tine_thickness: float,
    spacing: float,
    bottom_height: float,
    carriage_x: float,
    floor: Plane,
) -> np.ndarray:
    """The eight corners of each tine, (2, 8, 3) in `base_footprint`.

    Tines start at `carriage_x`, run `tine_length` forward, `spacing` apart about
    the centreline. `bottom_height` is above the floor PLANE (see `_at_height`),
    from `floor` when perception has one, else from the `floor_z` parameter.
    """
    half_width = tine_width / 2
    x_near, x_far = carriage_x, carriage_x + tine_length
    heights = (bottom_height, bottom_height + tine_thickness)

    tines = []
    for centre_y in (spacing / 2, -spacing / 2):
        corners = []
        for height in heights:
            for x, y in ((x_near, centre_y + half_width),
                         (x_near, centre_y - half_width),
                         (x_far, centre_y - half_width),
                         (x_far, centre_y + half_width)):
                corners.append(_at_height(x, y, height, floor))
        tines.append(corners)
    return np.array(tines, dtype=float)


def _at_height(x: float, y: float, height: float, floor: Plane) -> np.ndarray:
    """The point above (x, y) that is `height` from the floor plane
    {n.p + offset = 0}: z = (height - offset - n_x x - n_y y) / n_z.

    This keeps the tine parallel to a tilted floor, so the overlay can still
    show a camera orientation error rather than absorbing it.
    """
    normal = np.asarray(floor.normal, dtype=float)
    z = (height - floor.offset - normal[0] * x - normal[1] * y) / normal[2]
    return np.array([x, y, z], dtype=float)


def to_camera_frame(points: np.ndarray, base_to_camera: np.ndarray) -> np.ndarray:
    """Carry (..., 3) points from `base_footprint` into the camera optical frame.
    `base_to_camera` is camera_optical <- base_footprint (the inverse of
    perception's).
    """
    points = np.asarray(points, dtype=float)
    flat = transform_points(points.reshape(-1, 3), base_to_camera)
    return flat.reshape(points.shape)


def project_points(
    points_camera: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project optical-frame points to pixels: u = fx X/Z + cx, v = fy Y/Z + cy.

    Returns (pixels (..., 2), visible_mask (...)), the mask False where Z <= 0.
    """
    points = np.asarray(points_camera, dtype=float)
    flat = points.reshape(-1, 3)
    pixels, visible = project(flat, np.eye(4), {'fx': fx, 'fy': fy,
                                                'cx': cx, 'cy': cy})
    return (pixels.reshape(points.shape[:-1] + (2,)),
            visible.reshape(points.shape[:-1]))


def draw_polygons(
    image: np.ndarray,
    polygons: list[np.ndarray],
    color: tuple[int, int, int],
    alpha: float,
    thickness: int,
) -> np.ndarray:
    """Filled, semi-transparent polygons (blended with `alpha`) with full-opacity
    outlines. Returns a new BGR image.
    """
    import cv2

    usable = [np.asarray(p, dtype=np.int32) for p in polygons
              if p is not None and len(p) >= 3 and np.isfinite(p).all()]
    if not usable:
        return image.copy()

    filled = image.copy()
    for polygon in usable:
        cv2.fillPoly(filled, [polygon], color)
    # addWeighted, not a per-pixel loop: transparency in OpenCV is a blend of
    # two whole images, so the shapes go onto a copy and the copy is mixed back.
    blended = cv2.addWeighted(filled, alpha, image, 1.0 - alpha, 0.0)
    for polygon in usable:
        cv2.polylines(blended, [polygon], True, color, thickness, cv2.LINE_AA)
    return blended


# The six faces of a tine, as indices into the eight corners `tine_corners`
# returns: 0-3 underside, 4-7 top, in the same order round each.
TINE_FACES = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
              (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))


def tine_faces(corners_camera: np.ndarray, pixels: np.ndarray,
               visible: np.ndarray) -> list[np.ndarray]:
    """One tine's faces as pixel polygons, furthest first (painter's algorithm,
    so the tine occludes itself correctly).
    """
    faces = []
    for indices in TINE_FACES:
        if not visible[list(indices)].all():
            continue                      # any corner behind the camera: skip
        depth = float(np.mean(corners_camera[list(indices), 2]))
        faces.append((depth, pixels[list(indices)]))
    return [polygon for _, polygon in sorted(faces, key=lambda f: -f[0])]


def draw_pockets(
    image: np.ndarray,
    pockets: tuple[Pocket, ...],
    face_normal: np.ndarray,
    floor: Plane,
    base_to_camera: np.ndarray,
    intrinsics: dict,
    color: tuple[int, int, int],
    thickness: int,
    nominal: tuple[float, float] | None = None,
) -> np.ndarray:
    """Outline the measured pockets as rectangles in the face plane.

    With `nominal` = (width, height), also draw a dashed rectangle of the
    expected size at the same centre, so the measurement error is visible.
    """
    import cv2

    annotated = image.copy()
    for pocket in pockets:
        for size, dashed in ((None, False), (nominal, True)):
            if dashed and size is None:
                continue
            corners = pocket_corners(pocket, face_normal, floor, nominal=size)
            pixels, visible = project(corners, base_to_camera, intrinsics)
            if not visible.all() or not np.isfinite(pixels).all():
                continue
            points = pixels.astype(np.int32)
            if not dashed:
                cv2.polylines(annotated, [points], True, color, thickness,
                              cv2.LINE_AA)
            else:
                _dashed_polyline(annotated, points, color, max(1, thickness - 1))
    return annotated


def draw_face(image: np.ndarray, estimate: PalletEstimate, floor: Plane, height: float,
              base_to_camera: np.ndarray, intrinsics: dict, color: tuple[int, int, int],
              thickness: int) -> np.ndarray:
    """Outline the detected face: its measured width, from the floor up to
    `height`, in the face plane. Nothing is drawn without a finite width."""
    import cv2

    if not np.isfinite(estimate.face_width) or estimate.face_width <= 0:
        return image
    corners = face_corners(estimate.position, estimate.normal, estimate.face_width, height, floor)
    pixels, visible = project(corners, base_to_camera, intrinsics)
    if not visible.all() or not np.isfinite(pixels).all():
        return image
    annotated = image.copy()
    cv2.polylines(annotated, [pixels.astype(np.int32)], True, color, max(1, thickness - 1), cv2.LINE_AA)
    return annotated


def _dashed_polyline(image, points, color, thickness, dash=9, gap=6):
    """A dashed closed outline. OpenCV has no dash style, so walk each edge."""
    import cv2

    closed = np.vstack([points, points[:1]]).astype(float)
    for start, end in zip(closed[:-1], closed[1:]):
        span = end - start
        length = float(np.hypot(*span))
        if length < 1e-6:
            continue
        step = (dash + gap) / length
        position = 0.0
        while position < 1.0:
            a = start + span * position
            b = start + span * min(1.0, position + dash / length)
            cv2.line(image, tuple(a.astype(int)), tuple(b.astype(int)),
                     color, thickness, cv2.LINE_AA)
            position += step
    return image


def render(
    image: np.ndarray,
    estimate: PalletEstimate | None,
    state: str,
    base_to_camera: np.ndarray,
    intrinsics: dict,
    params: dict,
    floor: Plane,
) -> np.ndarray:
    """Build the annotated image for one frame.

    Args:
        image: BGR rectified left image.
        estimate: latest estimate, or None.
        state: a key of `fork_overlay.colors` ('idle', 'detecting',
            'approaching', 'go', 'no_go'); the node maps coordinator states.
        base_to_camera: (4, 4) camera_optical <- base_footprint at this stamp.
        intrinsics: {'fx', 'fy', 'cx', 'cy'}.
        params: the `/**` and `fork_overlay` blocks of params.yaml.
        floor: fallback "up" direction for the pockets when the estimate has no
            floor; the tines never use it.

    Returns:
        a new BGR image.
    """
    import cv2

    colors = params.get('colors', {})
    color = tuple(int(c) for c in colors.get(state, colors.get('idle',
                                                               (160, 160, 160))))
    forks = params['forks']
    alpha = float(params.get('alpha', 0.35))
    thickness = int(params.get('line_thickness', 2))

    # --- the tines, always, fixed to the robot: a level plane at the measured
    # floor_z, never the per-frame floor fit, which jitters, describes the floor
    # metres ahead, and would tilt with a camera orientation error and so hide it.
    floor_z = float(params.get('floor_z', -0.025))
    robot_fixed_floor = Plane(np.array([0.0, 0.0, 1.0]), -floor_z)
    corners = tine_corners(
        forks['tine_length'], forks['tine_width'], forks['tine_thickness'],
        forks['spacing'], forks['bottom_height'], forks['carriage_x'],
        robot_fixed_floor)
    in_camera = to_camera_frame(corners, base_to_camera)
    pixels, visible = project_points(in_camera, intrinsics['fx'], intrinsics['fy'],
                                     intrinsics['cx'], intrinsics['cy'])
    polygons = []
    for tine in range(corners.shape[0]):
        polygons.extend(tine_faces(in_camera[tine], pixels[tine], visible[tine]))
    annotated = draw_polygons(image, polygons, color, alpha, thickness)

    # --- what perception found, when it found anything
    lines = [state.upper()]
    if estimate is not None:
        if params.get('draw_face', True):
            annotated = draw_face(annotated, estimate, estimate.floor or floor,
                                  float(params.get('pallet_height', 0.145)),
                                  base_to_camera, intrinsics, color, thickness)
        if params.get('draw_pockets', True) and estimate.pockets:
            side = params.get('sides', {}).get(estimate.side, {})
            nominal = None
            if side.get('pocket_width') and side.get('opening_top') is not None:
                nominal = (side['pocket_width'],
                           side['opening_top'] - side.get('opening_bottom', 0.0))
            annotated = draw_pockets(
                annotated, estimate.pockets, estimate.normal,
                estimate.floor or floor, base_to_camera, intrinsics,
                color, thickness, nominal=nominal)
        lines.append(f'side {estimate.side}   face x {estimate.position[0]:.3f} m')
        lines.append(f'{len(estimate.pockets)} pocket(s)')
        reason = (estimate.side_detail or {}).get('reason', '')
        if reason:
            lines.append(reason)
    else:
        lines.append('no pallet')

    # Text on a translucent panel (per-glyph outlines smear at this scale).
    # text_scale: the node draws after downscaling and passes the factor.
    k = float(params.get('text_scale', 1.0))
    scale, pad, line_height = 0.6 * k, max(1, round(10 * k)), max(1, round(26 * k))
    sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0]
             for line in lines]
    panel_w = max(w for w, _ in sizes) + 2 * pad
    panel_h = line_height * len(lines) + pad
    panel = annotated.copy()
    cv2.rectangle(panel, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    annotated = cv2.addWeighted(panel, 0.55, annotated, 0.45, 0.0)
    for index, line in enumerate(lines):
        cv2.putText(annotated, line, (pad, pad + line_height * index + round(14 * k)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    return annotated
