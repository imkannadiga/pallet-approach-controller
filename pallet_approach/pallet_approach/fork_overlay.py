"""Draw imaginary forklift forks on the camera image.

YOU WRITE THIS FILE. Skeletons only -- signature, reasoning, TODO.

Two reasons this exists. The obvious one is that it makes the demo video
legible: anyone watching can see whether the forks would enter the pockets. The
real one is that it is a **calibration check you can see**. If the
camera-to-base transform is wrong, the forks visibly miss the pockets, and no
amount of plausible-looking numbers will hide it.

It is also the Day 3 projection maths applied to something you can look at, so
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

    TODO: implement.
    Hint: build one tine's eight corners with a couple of `np.array`s, then
    offset copies by +/- spacing/2 in y. Keep a consistent corner order so the
    faces you draw later are the ones you meant.
    """
    raise NotImplementedError


def to_camera_frame(points: np.ndarray, base_to_camera: np.ndarray) -> np.ndarray:
    """Carry (..., 3) points from `base_footprint` into the camera optical frame.

    `base_to_camera` is the 4x4 for camera_optical <- base_footprint. Note the
    direction: this is the inverse of what perception uses. Getting it backwards
    puts the forks behind the camera, which is a usefully obvious failure.

    Returns:
        the same shape as `points`, in the optical frame.

    TODO: implement.
    Hint: `pallet_approach.perception.transform_points` does the (N, 3) case;
    reshape, reuse it, and reshape back rather than writing the maths twice.
    """
    raise NotImplementedError


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

    TODO: implement.
    Hint: compute the mask first and only divide where it is True, otherwise
    NumPy will warn about dividing by zero and hand you infinities to draw.
    """
    raise NotImplementedError


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

    TODO: implement.
    """
    raise NotImplementedError


def draw_pockets(
    image: np.ndarray,
    pockets: tuple[Pocket, ...],
    base_to_camera: np.ndarray,
    intrinsics: dict,
    color: tuple[int, int, int],
    thickness: int,
) -> np.ndarray:
    """Outline the detected pockets, so the overlay shows what perception saw.

    Each pocket becomes a rectangle in the face plane -- its measured width and
    height around its centre -- projected the same way as the tines. Drawing the
    *measured* pockets rather than nominal ones is the point: when detection is
    off, you see it here before you see it in a number.

    TODO: implement.
    """
    raise NotImplementedError


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
        floor: perception's fitted floor plane, or one built from the
            `floor_z` parameter (-0.025, measured) when there is no fit.

    Returns:
        a new BGR image.

    TODO: implement last.
    """
    raise NotImplementedError
