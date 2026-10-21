"""Draw imaginary forklift forks on the camera image.

Two reasons this exists. The obvious one is that it makes the demo video
legible: anyone watching can see whether the forks would enter the pockets. The
real one is that it is a **calibration check you can see**. If the
camera-to-base transform is wrong, the forks visibly miss the pockets, and no
amount of plausible-looking numbers will hide it.

It is also the projection maths applied to something you can look at, so
the whole file is really one idea repeated: define geometry in `base_footprint`,
carry it into the camera optical frame, project it to pixels, draw it.

    fork boxes in base_footprint  (params.yaml `forks`)
      -> transform into the camera optical frame  (tf, at the image timestamp)
      -> project to pixels with the intrinsics     (cv2.projectPoints or by hand)
      -> draw filled, semi-transparent polygons on the rectified left image

Pitfalls, each of which has bitten everybody who has written one of these:

* Use the **rectified** image with the camera_info that matches it. Ours is
  rectified with D = 0, so `cv2.projectPoints` with zero distortion and the
  plain pinhole maths agree -- which makes for a good self-test.
* Mind the frame conventions: optical is z forward / y down, `base_footprint`
  is x forward / z up. The fork heights in params.yaml are heights above the
  floor, and `base_footprint` is near the floor -- but 25 mm above it, so add
  `floor_z` rather than using the heights raw.
* **Skip points behind the camera** (z <= 0 in the optical frame). Projecting
  them divides by a negative depth and they reappear mirrored on the image,
  which looks like a bug in your geometry and is not.
* The overlay draws over everything, with no occlusion test. A per-pixel depth
  test is an optional stretch; say out loud that you know it is missing.
"""

from __future__ import annotations

import numpy as np

from pallet_approach.model import PalletEstimate, Plane, Pocket
from pallet_approach.perception import transform_points
from pallet_approach.projection import pocket_corners, project

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
    """The eight corners of each tine, as a box, in `base_footprint`.

    Geometry: each tine is a rectangular box starting at `carriage_x` and
    extending `tine_length` forward. The pair is centred on the robot's
    centreline with `spacing` between their centres, which params.yaml sets to
    the pocket centre spacing (0.3725 m) so the overlay shows what a correctly
    set fork carriage would do.

    `bottom_height` 0.052 is a height above the *floor*, so the tine sits that
    far along the floor's normal from the floor plane -- not at that z. Build it
    from `floor`: the tine's underside lies on the plane offset by
    `bottom_height` along `floor.normal`. On a sloping floor that keeps the
    forks parallel to the floor, which is what a real carriage does.

    Getting this wrong breaks nothing loudly -- the forks just sit a couple of
    centimetres off in the image, which is exactly the kind of error the
    overlay exists to make visible.

    Heights in params.yaml are heights above the floor; `floor_z` says where
    the floor is in `base_footprint`, so a tine underside at 0.052 above the
    floor goes in at `floor_z + 0.052`. Pass perception's fitted value when
    there is one, and the `floor_z` parameter (-0.025, measured) otherwise --
    which is the case at the standoff, where only a 4 mm strip of floor is
    still in frame.

    Returns:
        (2, 8, 3) -- two tines, eight corners each, in `base_footprint`.

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
    """The point above (x, y) that is `height` from the floor PLANE.

    Not `z = floor_z + height`. The plane is {p : n.p + offset = 0}, so a point
    `height` above it satisfies `n.p + offset = height`, and solving for z

        z = (height - offset - n_x x - n_y y) / n_z

    keeps the tine parallel to the floor when the floor is tilted, which is
    what a real carriage does and what makes the overlay a fair check of the
    camera's orientation. Subtracting from z instead would tilt the forks
    against the floor by exactly the error being looked for, and the overlay
    would quietly stop being able to see it.
    """
    normal = np.asarray(floor.normal, dtype=float)
    z = (height - floor.offset - normal[0] * x - normal[1] * y) / normal[2]
    return np.array([x, y, z], dtype=float)


def to_camera_frame(points: np.ndarray, base_to_camera: np.ndarray) -> np.ndarray:
    """Carry (..., 3) points from `base_footprint` into the camera optical frame.

    `base_to_camera` is the 4x4 for camera_optical <- base_footprint. Note the
    direction: this is the inverse of what perception uses. Getting it backwards
    puts the forks behind the camera, which is a usefully obvious failure.

    Returns:
        the same shape as `points`, in the optical frame.

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
    """Project optical-frame points to pixels. Returns (pixels, visible_mask).

    Maths -- the inverse of what `depth_to_points` does:

        u = fx * X / Z + cx
        v = fy * Y / Z + cy

    Args:
        points_camera: (..., 3) in the optical frame.
        fx, fy, cx, cy: intrinsics from camera_info.

    Returns:
        pixels: (..., 2) float.
        visible_mask: (...) bool, False where Z <= 0 (behind the camera).

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
    """Draw filled, semi-transparent polygons with outlines.

    Transparency in OpenCV is a blend, not a draw: fill the shapes onto a copy
    of the image, then `cv2.addWeighted` the copy over the original with
    `alpha`. Draw the outlines on the result at full opacity so the edges stay
    crisp.

    Args:
        image: BGR, modified on a copy, not in place.
        polygons: each (M, 2) int32 pixel coordinates.
        color: BGR, from the `fork_overlay.colors` block for the current state.
        alpha: fill opacity.
        thickness: outline width in pixels.

    Returns:
        a new BGR image.
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
    """One tine's faces as pixel polygons, furthest away first.

    Sorting by mean depth and drawing back to front is the painter's
    algorithm, and it is what stops the box looking like a wireframe turned
    inside out. It handles the tine occluding ITSELF. It does nothing about the
    tine being occluded by the pallet -- the overlay has no depth test against
    the scene, so a tine draws over a block it would really be behind. That is
    a known limitation, not an oversight: a per-pixel test against the depth
    image is the stretch version.
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
    """Outline the detected pockets, so the overlay shows what perception saw.

    Each pocket becomes a rectangle in the face plane -- its measured width and
    height around its centre -- projected the same way as the tines. Drawing the
    *measured* pockets rather than nominal ones is the point: when detection is
    off, you see it here before you see it in a number.

    Pass `nominal` as (width, height) to add a dashed rectangle of the expected
    size at the SAME centre. The gap between the two outlines is the
    measurement error, which is a thing you can see rather than a number you
    have to look up. On the long face, where spacing reads +52 mm, the dashed
    box is visibly wider than the solid one.
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

    Pulls it together: look up `params['colors'][state]` (grey idle/detecting,
    yellow approaching, green GO, red NO-GO),
    draw the tines always, draw the pockets when there is an estimate, and put
    the state and the reason on as text so a still frame from the video is
    self-explanatory.

    Args:
        image: BGR rectified left image.
        estimate: latest estimate, or None.
        state: which colour to use -- one of the keys in the
            `fork_overlay.colors` parameter block ('idle', 'detecting',
            'approaching', 'go', 'no_go'). The node maps the coordinator's
            state name onto this, so you do not have to know the state machine.
        base_to_camera: (4, 4) camera_optical <- base_footprint at this stamp.
        intrinsics: {'fx', 'fy', 'cx', 'cy'}.
        params: the `/**` and `fork_overlay` blocks of params.yaml, as a dict.
        floor: perception's fitted floor plane, used ONLY as the fallback
            "up" direction for drawing pockets when the estimate carries no
            floor of its own. The tines never use it -- see the body.

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

    # --- the tines, always, and RIGIDLY FIXED TO THE ROBOT.
    #
    # Built against a level plane at the measured `floor_z`, never against
    # `estimate.floor`. A carriage is bolted to the truck: its pose in
    # base_footprint is a constant of the machine, and base_footprint already
    # follows whatever the robot is standing on. Three reasons the per-frame
    # fit is the wrong reference, in increasing order of how much it matters:
    #
    #   1. It moves. RANSAC redraws its samples every frame, so the forks would
    #      shiver by a millimetre or two even with the robot parked -- and an
    #      overlay you cannot trust to stay still is not a reference.
    #   2. It is fitted to floor METRES AHEAD, which is not the floor under the
    #      wheels. On a ramp those genuinely differ, and the forks belong to
    #      the wheels.
    #   3. It would hide the thing this overlay exists to show. A camera
    #      orientation error tilts the fitted floor; if the tines followed that
    #      tilt they would lean by exactly the error, while the pockets --
    #      measured against the same plane -- lean with them. The two errors
    #      cancel in the picture and the overlay reports that everything is
    #      fine. Keeping the tines robot-fixed makes the mismatch visible,
    #      which is the whole point.
    #
    # `floor_z` is -0.025, measured: base_footprint sits 25 mm above the real
    # floor, so a tine 0.040 above the floor is at z = +0.015 here.
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

    # Text, so a still frame taken out of the video explains itself. A demo
    # screenshot with no state on it is a picture of some forks.
    # A translucent panel behind the text rather than an outline around each
    # glyph: at this font scale a 4 px outline merges neighbouring letters into
    # a smear, which is how the first version came out. The panel also keeps
    # the text legible over a blown-out window, which half these frames have.
    scale, pad, line_height = 0.6, 10, 26
    sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0]
             for line in lines]
    panel_w = max(w for w, _ in sizes) + 2 * pad
    panel_h = line_height * len(lines) + pad
    panel = annotated.copy()
    cv2.rectangle(panel, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    annotated = cv2.addWeighted(panel, 0.55, annotated, 0.45, 0.0)
    for index, line in enumerate(lines):
        cv2.putText(annotated, line, (pad, pad + line_height * index + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    return annotated
