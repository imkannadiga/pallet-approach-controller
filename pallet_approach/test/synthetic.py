"""Synthetic pallet point clouds and depth images, with known ground truth.

Used by the unit tests so every perception stage can be checked against an
exact answer before it ever sees real data. Nothing here depends on ROS or on a
camera, so the tests run anywhere in well under a second.

**The pallet modelled is the one that was measured, not an EPAL nominal.**
Until 2026-10-19 this file built an EPAL 1: 1200 x 800, pockets 227.5 wide at
372.5 centres, over a 22 mm bottom board. The real pallet is 1195 x 795 with
230 mm pockets at 365 centres and **no bottom board on the pocket face**, and
tolerances tuned against the wrong synthetic would have been wrong by exactly
the amount the measurement corrected (D4, D19).

Both faces are modelled, because they are different structures rather than two
sizes of one thing. A EUR pallet's three bottom boards run along the 1195 mm
direction:

  SHORT face, 795 mm -- the forks travel BETWEEN those boards, down clear
  channels, so the opening runs from the floor:

      |<-100->|<--- 230 --->|<-135->|<--- 230 --->|<-100->|
        block      pocket     block     pocket      block
      -0.3975  -0.2975   -0.0675  0.0675     0.2975  0.3975    (metres)

      height:  0.000 .. 0.100  opening        pocket centres -+0.1825
               0.100 .. 0.145  top deck       spacing 0.365

  LONG face, 1195 mm -- the forks cross those boards, so the opening starts
  above a 20 mm one:

      |<-150->|<--- 372.5 --->|<-150->|<--- 372.5 --->|<-150->|
      -0.5975  -0.4475     -0.075  0.075        0.4475  0.5975

      height:  0.000 .. 0.020  bottom board   pocket centres -+0.26125
               0.020 .. 0.125  opening        spacing 0.5225
               0.125 .. 0.145  top deck

All clouds are in `base_footprint`: x forward, y left, z up. That is the frame
the pipeline works in, because every height the problem is stated in is a
height above the floor and base_footprint is near the floor.

The floor is at `FLOOR_Z = -0.025`, not 0: base_footprint sits 25 mm above the
real floor. The clouds use the measured value, so anything that assumes z = 0
is the floor fails its tests here rather than in the lab.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class Face:
    """One face of the pallet: what you see, and at what heights."""

    name: str
    width: float
    """Across the face, metres."""
    depth: float
    """How far a fork travels once inserted."""
    blocks: tuple
    """(start, end) along-face spans of solid timber at pocket height."""
    pockets: tuple
    """The gaps between them."""
    opening_bottom: float
    opening_top: float
    bottom_board: float = 0.0
    """Thickness of the board under the opening; zero on the short face."""

    @property
    def pocket_width(self):
        return self.pockets[0][1] - self.pockets[0][0]

    @property
    def pocket_height(self):
        return self.opening_top - self.opening_bottom

    @property
    def pocket_center_height(self):
        return 0.5 * (self.opening_bottom + self.opening_top)

    @property
    def pocket_centers_u(self):
        return tuple(0.5 * (a + b) for a, b in self.pockets)

    @property
    def pocket_center_spacing(self):
        centres = self.pocket_centers_u
        return abs(centres[1] - centres[0])


PALLET_HEIGHT = 0.145

SHORT = Face(
    name='short', width=0.795, depth=1.195,
    blocks=((-0.3975, -0.2975), (-0.0675, 0.0675), (0.2975, 0.3975)),
    pockets=((-0.2975, -0.0675), (0.0675, 0.2975)),
    opening_bottom=0.000, opening_top=0.100, bottom_board=0.0)

LONG = Face(
    name='long', width=1.195, depth=0.795,
    blocks=((-0.5975, -0.4475), (-0.075, 0.075), (0.4475, 0.5975)),
    pockets=((-0.4475, -0.075), (0.075, 0.4475)),
    opening_bottom=0.020, opening_top=0.125, bottom_board=0.020)

FACES = {'short': SHORT, 'long': LONG}

# Back-compatible module constants: the short face, which is the one the robot
# normally enters and the one every existing test means.
PALLET_WIDTH = SHORT.width
PALLET_LENGTH = SHORT.depth
BLOCKS = SHORT.blocks
POCKETS = SHORT.pockets
POCKET_WIDTH = SHORT.pocket_width
POCKET_HEIGHT = SHORT.pocket_height
POCKET_CENTER_SPACING = SHORT.pocket_center_spacing
POCKET_CENTER_HEIGHT = SHORT.pocket_center_height
BLOCK_HEIGHT = SHORT.pocket_height
# The deck above the opening. On the short face that is the whole 45 mm between
# the opening and the top; there is no board below.
DECK_THICKNESS = PALLET_HEIGHT - SHORT.opening_top

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


def face_points(spacing: float = 0.006, side: Face | str = SHORT) -> np.ndarray:
    """Points on one face of the pallet, in face-local coordinates.

    Returns (N, 3) as (along_face, 0, height_above_floor): the face lies in the
    plane y = 0, so these are ready to be rotated and translated into place.

    Sampled as the face physically is -- boards across the full width, blocks
    only where there are blocks -- so the pockets are genuinely absent rather
    than marked. On the short face there is no board under the opening, which
    is the whole reason `find_pockets` cannot measure that edge and takes it
    from the known geometry instead (D25).
    """
    face = FACES[side] if isinstance(side, str) else side
    half = face.width / 2
    parts = []

    def slab(u_range, h_range):
        u, h = _grid(u_range, h_range, spacing)
        uu, hh = np.meshgrid(u, h, indexing='ij')
        parts.append(np.stack([uu.ravel(), np.zeros(uu.size), hh.ravel()], axis=1))

    # top deck, full width, above the opening
    slab((-half, half), (face.opening_top, PALLET_HEIGHT))
    # bottom board, full width -- the long face only
    if face.bottom_board > 0:
        slab((-half, half), (0.0, face.bottom_board))
    # the blocks, spanning the opening
    for u0, u1 in face.blocks:
        slab((u0, u1), (face.opening_bottom, face.opening_top))

    return np.concatenate(parts, axis=0)


def top_deck_points(spacing: float = 0.02, side: Face | str = SHORT) -> np.ndarray:
    """The top surface of the pallet, in face-local coordinates.

    The camera sits ~0.4 m up and looks level, so it does see some of the top
    deck. Including it keeps `select_pallet_cluster` and `fit_front_face`
    honest: the largest plane in the cluster is not necessarily the face.
    """
    face = FACES[side] if isinstance(side, str) else side
    u, d = _grid((-face.width / 2, face.width / 2), (0.0, face.depth), spacing)
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
    side: Face | str = SHORT,
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
        side: which face is presented -- SHORT (795 mm, opening from the floor)
            or LONG (1195 mm, opening above a 20 mm board). They are different
            structures, not two sizes of one (D19).

    Returns:
        points: (N, 3) in `base_footprint`.
        truth: the answers the pipeline should recover --
            'face_center', 'normal', 'yaw', 'pocket_centers', 'pocket_width',
            'pocket_height', 'pocket_center_spacing', 'floor_z'.
    """
    rng = np.random.default_rng(seed)
    face = FACES[side] if isinstance(side, str) else side

    local = face_points(spacing=spacing, side=face)
    if with_top_deck:
        local = np.concatenate([local, top_deck_points(side=face)], axis=0)

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
    for u_mid in face.pocket_centers_u:
        local_c = np.array([u_mid, 0.0, face.pocket_center_height])
        c = local_c @ rot.T
        pocket_centers.append(c + np.array([distance, lateral, floor_z]))

    truth = {
        'face_center': face_center,
        'normal': normal,
        'yaw': yaw,
        'side': face.name,
        'face_width': face.width,
        'pocket_centers': np.array(pocket_centers),
        'pocket_width': face.pocket_width,
        'pocket_height': face.pocket_height,
        'pocket_center_spacing': face.pocket_center_spacing,
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
