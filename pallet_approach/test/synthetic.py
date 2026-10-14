"""Synthetic EUR pallet point clouds and depth images, with known ground truth.

Used by the unit tests so that every perception stage can be checked against an
exact answer before it ever sees real data. Nothing here depends on ROS or on a
camera, so the tests run anywhere, in well under a second.

The pallet modelled is an EPAL 1 / EUR pallet seen from its 800 mm face, which
is the face the forks enter. Looking at that face, at pocket height, the block
layout across the 800 mm is:

    |<-100->|<--- 227.5 --->|<--145-->|<--- 227.5 --->|<-100->|
      block       pocket       block       pocket       block
    -0.400  -0.300        -0.0725    0.0725         0.300  0.400     (metres)

so the pocket centres land at -+0.18625 m from the centreline, 0.3725 m apart,
which is what params.yaml says and what `find_pockets` has to recover.

All clouds are in `base_footprint`: x forward, y left, z up. That is the frame
the pipeline works in, because every height the problem is stated in -- pallet
0.144 m, pocket centre 0.072 m, fork underside 0.052 m -- is a height above the
floor, and base_footprint is near the floor. `base_link` is the chassis body
frame 0.235 m higher up.

The floor itself is at `FLOOR_Z = -0.025`, not 0: base_footprint is 25 mm above
the real floor (see the constant below). The clouds use the measured value, so
anything that assumes z = 0 is the floor fails its tests here rather than in
the lab.
"""

from __future__ import annotations

import math

import numpy as np

# EPAL 1 nominal geometry, metres. Mirrors config/params.yaml.
PALLET_LENGTH = 1.200
PALLET_WIDTH = 0.800
PALLET_HEIGHT = 0.144
DECK_THICKNESS = 0.022
BLOCK_HEIGHT = 0.100
POCKET_WIDTH = 0.2275
POCKET_HEIGHT = 0.100
POCKET_CENTER_SPACING = 0.3725
POCKET_CENTER_HEIGHT = 0.072

# Blocks across the 800 mm face, as (start, end) along-face coordinates.
BLOCKS = ((-0.400, -0.300), (-0.0725, 0.0725), (0.300, 0.400))
# The gaps between them, i.e. the pockets.
POCKETS = ((-0.300, -0.0725), (0.0725, 0.300))

# Camera as measured on the robot, for the depth-image helpers.
FX = FY = 949.6732177734375
CX = 628.338134765625
CY = 369.74566650390625
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
# base_footprint -> camera optical, RPY (-90, 0, -90) deg. The measured
# base_link value is [0.390, 0.060, 0.163] and the URDF puts base_footprint
# 0.210 m below base_link, so this is what tf reports.
CAMERA_XYZ = (0.390, 0.060, 0.373)

# Where the real floor is, in base_footprint. NOT zero: the lens was
# tape-measured at 0.398 m above the floor on 2026-10-14, and the URDF puts it
# 0.163 m above base_link, so base_link is 0.235 m up -- which matches the
# wheel geometry (0.16459 radius, axle at -0.0702 -> 0.2348 m) to 0.2 mm, while
# base_footprint claims 0.210 m. base_footprint is 25 mm above the floor.
#
# The default floor in these clouds is this measured value, deliberately not 0,
# so that any code assuming z = 0 is the floor fails its tests.
FLOOR_Z = -0.025
CAMERA_HEIGHT_ABOVE_FLOOR = CAMERA_XYZ[2] - FLOOR_Z      # 0.398 m, measured


def _grid(u_range: tuple[float, float], v_range: tuple[float, float], spacing: float):
    """Two 1D sample arrays covering the given ranges at `spacing`."""
    u0, u1 = u_range
    v0, v1 = v_range
    nu = max(2, int(round((u1 - u0) / spacing)) + 1)
    nv = max(2, int(round((v1 - v0) / spacing)) + 1)
    return np.linspace(u0, u1, nu), np.linspace(v0, v1, nv)


def floor_points(
    x_range: tuple[float, float] = (0.5, 4.0),
    y_range: tuple[float, float] = (-1.5, 1.5),
    z: float = FLOOR_Z,
    spacing: float = 0.03,
) -> np.ndarray:
    """A flat floor at height `z` in `base_footprint`.

    Defaults to the measured floor, -0.025, not zero -- so a floor fit that
    snaps to z = 0 fails rather than passing by coincidence.
    """
    xs, ys = _grid(x_range, y_range, spacing)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    pts = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, z)], axis=1)
    return pts


def face_points(spacing: float = 0.006) -> np.ndarray:
    """Points on the pallet's 800 mm face, in face-local coordinates.

    Returns (N, 3) as (along_face, 0, height_above_floor): the face lies in the
    plane y = 0, so these are ready to be rotated and translated into place.

    The face is sampled as it physically is -- top deck and bottom boards across
    the full width, blocks only where there are blocks -- so the pockets are
    genuinely absent rather than marked.
    """
    parts = []

    # top deck board, full width
    u, h = _grid((-PALLET_WIDTH / 2, PALLET_WIDTH / 2),
                 (PALLET_HEIGHT - DECK_THICKNESS, PALLET_HEIGHT), spacing)
    uu, hh = np.meshgrid(u, h, indexing='ij')
    parts.append(np.stack([uu.ravel(), np.zeros(uu.size), hh.ravel()], axis=1))

    # bottom boards, full width
    u, h = _grid((-PALLET_WIDTH / 2, PALLET_WIDTH / 2), (0.0, DECK_THICKNESS), spacing)
    uu, hh = np.meshgrid(u, h, indexing='ij')
    parts.append(np.stack([uu.ravel(), np.zeros(uu.size), hh.ravel()], axis=1))

    # the three blocks, between the boards
    for u0, u1 in BLOCKS:
        u, h = _grid((u0, u1), (DECK_THICKNESS, DECK_THICKNESS + BLOCK_HEIGHT), spacing)
        uu, hh = np.meshgrid(u, h, indexing='ij')
        parts.append(np.stack([uu.ravel(), np.zeros(uu.size), hh.ravel()], axis=1))

    return np.concatenate(parts, axis=0)


def top_deck_points(spacing: float = 0.02) -> np.ndarray:
    """The top surface of the pallet, in face-local coordinates.

    The camera sits ~0.38 m up and looks level, so it does see some of the top
    deck. Including it keeps `select_pallet_cluster` and `fit_front_face`
    honest: the largest plane in the cluster is not necessarily the face.
    """
    u, d = _grid((-PALLET_WIDTH / 2, PALLET_WIDTH / 2), (0.0, PALLET_LENGTH), spacing)
    uu, dd = np.meshgrid(u, d, indexing='ij')
    return np.stack([uu.ravel(), dd.ravel(), np.full(uu.size, PALLET_HEIGHT)], axis=1)


def pallet_cloud(
    distance: float = 2.0,
    lateral: float = 0.0,
    yaw: float = 0.0,
    floor_z: float = FLOOR_Z,
    noise: float = 0.0,
    with_floor: bool = True,
    with_top_deck: bool = True,
    spacing: float = 0.006,
    floor_spacing: float = 0.03,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """A pallet at a known pose, plus the floor. Returns (points, truth).

    Args:
        distance: x of the pallet's front face in `base_footprint`, metres.
        lateral: y of the face centre, metres.
        yaw: rotation of the pallet about z, radians. 0 means the face is
            perpendicular to the robot's x axis, i.e. squarely facing it.
        floor_z: floor height in `base_footprint`; -0.025 is the measured floor.
        noise: standard deviation of isotropic Gaussian noise, metres. The ZED's
            real noise grows with range; 0.005 at 2 m is a reasonable stand-in
            until the Day 5 curve replaces it.
        with_floor: include floor points.
        with_top_deck: include the pallet's top surface.
        spacing: sample spacing on the face.
        floor_spacing: sample spacing on the floor; see `depth_image`
            about rendering density.
        seed: RNG seed, so tests are deterministic.

    Returns:
        points: (N, 3) in `base_footprint`.
        truth: the answers the pipeline should recover --
            'face_center', 'normal', 'yaw', 'pocket_centers', 'pocket_width',
            'pocket_height', 'pocket_center_spacing', 'floor_z'.
    """
    rng = np.random.default_rng(seed)

    local = face_points(spacing=spacing)
    if with_top_deck:
        local = np.concatenate([local, top_deck_points()], axis=0)

    # face-local -> base_footprint: the outward face normal points at the robot,
    # i.e. along -x when yaw is 0. Along-face +u maps to +y (left).
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    rot = np.array([
        [sin_y, cos_y, 0.0],     # local u (along face) and local depth -> x
        [cos_y, -sin_y, 0.0],    # -> y
        [0.0, 0.0, 1.0],         # local height -> z
    ])
    pts = local @ rot.T
    pts[:, 0] += distance
    pts[:, 1] += lateral
    pts[:, 2] += floor_z

    if with_floor:
        pts = np.concatenate([pts, floor_points(z=floor_z, spacing=floor_spacing)], axis=0)

    if noise > 0.0:
        pts = pts + rng.normal(0.0, noise, size=pts.shape)

    normal = np.array([-cos_y, sin_y, 0.0])
    normal = normal / np.linalg.norm(normal)
    face_center = np.array([distance, lateral, floor_z + PALLET_HEIGHT / 2.0])

    pocket_centers = []
    for u0, u1 in POCKETS:
        u_mid = 0.5 * (u0 + u1)
        local_c = np.array([u_mid, 0.0, POCKET_CENTER_HEIGHT])
        c = local_c @ rot.T
        pocket_centers.append(c + np.array([distance, lateral, floor_z]))

    truth = {
        'face_center': face_center,
        'normal': normal,
        'yaw': yaw,
        'pocket_centers': np.array(pocket_centers),
        'pocket_width': POCKET_WIDTH,
        'pocket_height': POCKET_HEIGHT,
        'pocket_center_spacing': POCKET_CENTER_SPACING,
        'floor_z': floor_z,
    }
    return pts, truth


def two_blobs(gap: float = 0.6, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two well-separated point blobs. Returns (points, true_labels).

    For testing the clustering stage on something with an unambiguous answer.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal((1.5, 0.0, 0.0), 0.05, size=(200, 3))
    b = rng.normal((1.5 + gap, 0.0, 0.0), 0.05, size=(200, 3))
    labels = np.concatenate([np.zeros(200, dtype=int), np.ones(200, dtype=int)])
    return np.concatenate([a, b], axis=0), labels


def base_to_camera_matrix() -> np.ndarray:
    """The measured camera extrinsic as a 4x4: camera_optical <- base_footprint.

    Built from the tf2_echo result in docs/SYSTEM.md, shifted to ground level.
    The rotation maps base_footprint (x forward, y left, z up) into the optical
    frame (x right, y down, z forward), which is the transpose of the matrix
    printed there. The rotation is identical in base_link and base_footprint --
    the joint between them is a pure translation -- so only z changes.
    """
    # base_footprint <- camera_optical, as tf2_echo prints it
    base_from_cam = np.eye(4)
    base_from_cam[:3, :3] = np.array([
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ])
    base_from_cam[:3, 3] = CAMERA_XYZ
    return np.linalg.inv(base_from_cam)


def camera_to_base_matrix() -> np.ndarray:
    """base_footprint <- camera_optical, as a 4x4. The inverse of the above."""
    return np.linalg.inv(base_to_camera_matrix())


def depth_image(
    points_base: np.ndarray,
    width: int = IMAGE_WIDTH,
    height: int = IMAGE_HEIGHT,
) -> np.ndarray:
    """Render `base_footprint` points into a float32 depth image, nearest wins.

    A crude z-buffer: project every point and keep the smallest depth per pixel.
    Pixels nothing lands on are NaN, exactly as the ZED leaves invalid pixels.

    **Sample the source cloud finely enough.** This splats points, so the image
    is only as dense as the cloud: at range d a cloud spaced `g` apart lands
    `g * fx / d` pixels apart, and anything over ~1 px leaves holes. A real ZED
    frame is 84 % valid (measured); a cloud spaced 0.004 m seen at 1.6 m renders
    to under 2 % and the pipeline then fails on sparsity rather than on
    anything real -- clustering in particular, which needs a surface dense
    relative to its `eps`. For a test at ~2 m use `spacing=0.0015` or finer, and
    assert the valid fraction rather than trusting it.
    """
    cam_from_base = base_to_camera_matrix()
    pts = points_base @ cam_from_base[:3, :3].T + cam_from_base[:3, 3]

    z = pts[:, 2]
    in_front = z > 1e-6
    pts, z = pts[in_front], z[in_front]

    u = np.round(FX * pts[:, 0] / z + CX).astype(int)
    v = np.round(FY * pts[:, 1] / z + CY).astype(int)
    on_image = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z = u[on_image], v[on_image], z[on_image]

    image = np.full((height, width), np.inf, dtype=np.float32)
    np.minimum.at(image, (v, u), z.astype(np.float32))
    image[np.isinf(image)] = np.nan
    return image
