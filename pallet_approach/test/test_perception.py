"""Tests for perception.py, one or more per pipeline stage.

Every stage has its own tests, checked against exact answers from
`synthetic.py`, so a failure is a real disagreement rather than a tolerance
problem.

    # just one stage
    python -m pytest test/test_perception.py -k voxel -x -v

The expected answers come from `synthetic.py`, which models an EPAL 1 pallet
exactly, so a failure is a real disagreement, not a tolerance problem.
"""

import math

import numpy as np
import pytest
import synthetic as S

from pallet_approach import perception
from pallet_approach.model import Plane, Pocket


# ------------------------------------------------------- depth_to_points

def test_depth_to_points_matches_the_pinhole_formula():
    """X = (u - cx) Z / fx, Y = (v - cy) Z / fy, Z = depth."""
    fx = fy = 100.0
    cx = cy = 50.0
    depth = np.full((100, 100), np.nan, dtype=np.float32)
    depth[50, 50] = 2.0        # dead centre -> straight ahead
    depth[50, 150 - 100] = 2.0  # (v=50, u=50) again; keep the grid simple
    depth[50, 60] = 1.0        # 10 px right of centre at 1 m -> X = 0.1
    depth[60, 50] = 1.0        # 10 px below centre at 1 m -> Y = 0.1

    points = perception.depth_to_points(depth, fx, fy, cx, cy,
                                        stride=1, min_range=0.1, max_range=5.0)

    assert points.shape[1] == 3
    rows = {tuple(np.round(p, 6)) for p in points}
    assert (0.0, 0.0, 2.0) in rows
    assert (0.1, 0.0, 1.0) in rows
    assert (0.0, 0.1, 1.0) in rows


def test_depth_to_points_drops_invalid_and_out_of_range():
    depth = np.array([[np.nan, np.inf, 0.1, 2.0, 9.0]], dtype=np.float32)
    points = perception.depth_to_points(depth, 100.0, 100.0, 2.0, 0.0,
                                        stride=1, min_range=0.4, max_range=4.0)
    assert len(points) == 1
    assert np.isfinite(points).all()
    assert math.isclose(points[0, 2], 2.0, abs_tol=1e-6)


def test_depth_to_points_stride_decimates():
    depth = np.full((40, 40), 2.0, dtype=np.float32)
    full = perception.depth_to_points(depth, 100.0, 100.0, 20.0, 20.0, stride=1)
    half = perception.depth_to_points(depth, 100.0, 100.0, 20.0, 20.0, stride=2)
    assert len(half) == pytest.approx(len(full) / 4, rel=0.1)


# ------------------------------------------------------- transform_points

def test_transform_points_applies_rotation_and_translation():
    """A point 1 m in front of the camera is 1.390 m ahead of base_footprint
    and 0.373 m above it -- the camera height in the ground frame."""
    camera_to_base = S.camera_to_base_matrix()
    out = perception.transform_points(np.array([[0.0, 0.0, 1.0]]), camera_to_base)
    assert np.allclose(out[0], [1.390, 0.060, 0.373], atol=1e-9)


def test_transform_points_identity_is_a_no_op():
    points = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 0.5]])
    out = perception.transform_points(points, np.eye(4))
    assert np.allclose(out, points, atol=1e-12)


# ------------------------------------------------------------ crop_to_roi

def test_crop_to_roi_keeps_only_the_box():
    points = np.array([
        [1.0, 0.0, 0.0],      # inside
        [0.2, 0.0, 0.0],      # too close
        [5.0, 0.0, 0.0],      # too far
        [1.0, 2.0, 0.0],      # too far left
        [1.0, 0.0, 1.0],      # too high
    ])
    out = perception.crop_to_roi(points, 0.5, 4.0, -1.5, 1.5, -0.3, 0.6)
    assert len(out) == 1
    assert np.allclose(out[0], [1.0, 0.0, 0.0])


# -------------------------------------------------------- voxel_downsample

def test_voxel_downsample_collapses_duplicates():
    """Twenty points inside one 2 cm cube must come out as one."""
    rng = np.random.default_rng(0)
    cluster = rng.uniform(0.0, 0.019, size=(20, 3))
    out = perception.voxel_downsample(cluster, 0.02)
    assert len(out) == 1


def test_voxel_downsample_keeps_separate_cubes_separate():
    points = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.10, 0.0, 0.0]])
    out = perception.voxel_downsample(points, 0.02)
    assert len(out) == 3


def test_voxel_downsample_preserves_the_extent(square_pallet):
    points, _ = square_pallet
    out = perception.voxel_downsample(points, 0.02)
    assert len(out) < len(points)
    assert np.allclose(out.min(axis=0), points.min(axis=0), atol=0.03)
    assert np.allclose(out.max(axis=0), points.max(axis=0), atol=0.03)


# -------------------------------------------------------- fit_plane_ransac

def test_fit_plane_ransac_finds_a_clean_floor():
    """Deliberately not a floor at z = 0: a fit that happened to return d = 0
    would pass that vacuously. S.FLOOR_Z is the measured offset of the real
    floor below base_footprint, so this checks the fit recovers it."""
    floor = S.floor_points(z=S.FLOOR_Z)
    normal, offset, inliers = perception.fit_plane_ransac(
        floor, distance_threshold=0.01, iterations=100,
        rng=np.random.default_rng(0))
    assert abs(abs(normal[2]) - 1.0) < 1e-3          # horizontal plane
    assert inliers.sum() > 0.95 * len(floor)
    # n . p + d = 0 on the plane, so for a floor at z = -0.025: d = +-0.025
    assert math.isclose(abs(offset), abs(S.FLOOR_Z), abs_tol=0.005)


def test_fit_plane_ransac_survives_outliers():
    """This is the whole reason for RANSAC: least squares would be dragged off."""
    rng = np.random.default_rng(1)
    floor = S.floor_points(z=S.FLOOR_Z, spacing=0.05)
    outliers = rng.uniform([0.5, -1.5, 0.1], [4.0, 1.5, 1.0], size=(len(floor) // 2, 3))
    mixed = np.concatenate([floor, outliers])
    normal, offset, inliers = perception.fit_plane_ransac(
        mixed, distance_threshold=0.02, iterations=300,
        rng=np.random.default_rng(0))
    assert abs(abs(normal[2]) - 1.0) < 0.02
    assert math.isclose(abs(offset), abs(S.FLOOR_Z), abs_tol=0.02)
    # every floor point should be an inlier, and few of the outliers
    assert inliers[:len(floor)].sum() > 0.95 * len(floor)
    assert inliers[len(floor):].sum() < 0.25 * len(outliers)


def test_fit_plane_ransac_is_deterministic_with_a_seeded_rng():
    floor = S.floor_points(spacing=0.05)
    a = perception.fit_plane_ransac(floor, 0.01, 50, rng=np.random.default_rng(7))
    b = perception.fit_plane_ransac(floor, 0.01, 50, rng=np.random.default_rng(7))
    assert np.allclose(a[0], b[0])


# ------------------------------------------------------------ remove_floor

def test_remove_floor_keeps_the_pallet_and_drops_the_ground(square_pallet):
    points, truth = square_pallet
    floor = Plane(normal=np.array([0.0, 0.0, 1.0]), offset=-truth['floor_z'])
    out = perception.remove_floor(points, floor, margin=0.03)
    # nothing left near or below the floor
    assert (out[:, 2] > truth['floor_z'] + 0.02).all()
    # but the pallet survives, including most of its bottom boards' top edge
    assert len(out) > 1000
    assert out[:, 2].max() > truth['floor_z'] + 0.13


# ---------------------------------------------------------- cluster_points

def test_cluster_points_separates_two_blobs():
    points, true_labels = S.two_blobs(gap=0.6)
    labels = perception.cluster_points(points, eps=0.05, min_samples=10)
    assert len(labels) == len(points)
    real = labels[labels >= 0]
    assert len(set(real.tolist())) == 2
    # the split must agree with the truth, whatever the label numbering is
    first = labels[:200]
    second = labels[200:]
    assert len(set(first[first >= 0].tolist())) == 1
    assert len(set(second[second >= 0].tolist())) == 1
    assert set(first[first >= 0].tolist()) != set(second[second >= 0].tolist())


def test_cluster_points_marks_isolated_points_as_noise():
    points, _ = S.two_blobs()
    points = np.concatenate([points, np.array([[3.9, 1.4, 0.5]])])
    labels = perception.cluster_points(points, eps=0.05, min_samples=10)
    assert labels[-1] == -1


# ---------------------------------------------------- select_pallet_cluster

def test_select_pallet_cluster_picks_the_pallet_sized_one(square_pallet):
    points, truth = square_pallet
    above = points[points[:, 2] > truth['floor_z'] + 0.03]
    labels = np.zeros(len(above), dtype=int)
    chosen = perception.select_pallet_cluster(
        above, labels, pallet_width=S.PALLET_WIDTH, pallet_height=S.PALLET_HEIGHT)
    assert chosen is not None
    assert len(chosen) > 500


def test_select_pallet_cluster_rejects_a_small_box():
    """A 0.2 m cube is not a pallet, and saying so is how NO_PALLET happens."""
    rng = np.random.default_rng(0)
    box = rng.uniform([1.5, -0.1, 0.0], [1.7, 0.1, 0.2], size=(600, 3))
    labels = np.zeros(len(box), dtype=int)
    chosen = perception.select_pallet_cluster(
        box, labels, pallet_width=S.PALLET_WIDTH, pallet_height=S.PALLET_HEIGHT)
    assert chosen is None


def test_a_tall_panel_nearer_than_the_pallet_is_not_chosen(square_pallet):
    """D76, from the robot (2026-10-23): a partition panel standing edge-on,
    0.8 m long and 0.8 m tall, nearer than the pallet. It has a pallet-length
    side and is taller than a pallet, which the old rule accepted -- and being
    nearer, it won. With the height ceiling the pallet is chosen."""
    points, truth = square_pallet
    pallet = points[points[:, 2] > truth['floor_z'] + 0.03]
    rng = np.random.default_rng(0)
    near = 0.5 * float(np.linalg.norm(pallet[:, :2].mean(axis=0)))
    panel = rng.uniform([near, -1.2, truth['floor_z'] + 0.03],
                        [near + 0.04, -0.4, truth['floor_z'] + 0.80], size=(1500, 3))
    cloud = np.vstack([pallet, panel])
    labels = np.r_[np.zeros(len(pallet), int), np.ones(len(panel), int)]
    floor = Plane(np.array([0.0, 0.0, 1.0]), -truth['floor_z'])
    kwargs = dict(pallet_width=S.PALLET_WIDTH, pallet_height=S.PALLET_HEIGHT,
                  floor=floor, widths=(S.PALLET_WIDTH,))

    without = perception.select_pallet_cluster(cloud, labels, **kwargs)
    assert len(without) == len(panel)            # the bug: nearest tall thing wins
    chosen = perception.select_pallet_cluster(
        cloud, labels, **kwargs, max_height=S.PALLET_HEIGHT + 0.04)
    assert len(chosen) == len(pallet)


# -------------------------------------------------------- fit_front_face

@pytest.mark.parametrize('yaw_deg', [0, 15, -20])
def test_fit_front_face_recovers_the_normal(yaw_deg):
    points, truth = S.pallet_cloud(distance=2.0, yaw=math.radians(yaw_deg), noise=0.003)
    pallet = points[points[:, 2] > truth['floor_z'] + 0.03]
    result = perception.fit_front_face(
        pallet, distance_threshold=0.015, max_tilt_from_vertical=0.26,
        rng=np.random.default_rng(0))
    assert result is not None
    normal, centroid, inliers = result
    # the normal must be horizontal and parallel to the true face normal
    assert abs(normal[2]) < 0.1
    alignment = abs(float(np.dot(normal[:2] / np.linalg.norm(normal[:2]),
                                 truth['normal'][:2])))
    assert alignment > 0.99
    assert inliers.sum() > 300


def test_fit_front_face_rejects_a_horizontal_plane():
    """Handed only a floor, there is no vertical face to find."""
    floor = S.floor_points(spacing=0.03)
    result = perception.fit_front_face(
        floor, distance_threshold=0.015, max_tilt_from_vertical=0.26,
        rng=np.random.default_rng(0))
    assert result is None


# ---------------------------------------------------------------- face_yaw

@pytest.mark.parametrize('normal,expected_deg', [
    ((-1.0, 0.0, 0.0), 180.0),
    ((0.0, 1.0, 0.0), 90.0),
    ((0.0, -1.0, 0.0), -90.0),
    ((-0.866, 0.5, 0.0), 150.0),
])
def test_face_yaw(normal, expected_deg):
    got = math.degrees(perception.face_yaw(np.array(normal)))
    assert math.isclose(abs(got), abs(expected_deg), abs_tol=0.5)


# ------------------------------------------------------------- find_pockets

def test_find_pockets_recovers_two_epal_pockets(square_pallet):
    points, truth = square_pallet
    pallet = points[points[:, 2] > truth['floor_z'] + 0.03]
    band_points = pallet[np.abs(pallet[:, 0] - 2.0) < 0.02]   # the face only

    pockets = perception.find_pockets(
        band_points,
        face_normal=truth['normal'],
        face_centroid=truth['face_center'],
        floor=Plane(normal=np.array([0.0, 0.0, 1.0]), offset=-truth['floor_z']),
        band_half_height=0.03,
        min_gap_width=0.15,
        max_gap_width=0.30,
        pocket_center_height=S.POCKET_CENTER_HEIGHT,
    )

    assert len(pockets) == 2
    for pocket in pockets:
        assert math.isclose(pocket.width, S.POCKET_WIDTH, abs_tol=0.03)
    spacing = float(np.linalg.norm(pockets[0].center - pockets[1].center))
    assert math.isclose(spacing, S.POCKET_CENTER_SPACING, abs_tol=0.03)
    # ordered left to right as the robot sees them: +y first
    assert pockets[0].center[1] > pockets[1].center[1]


def test_find_pockets_returns_nothing_for_a_solid_face():
    """A solid wall has no pockets, and must not be reported as a pallet."""
    u, h = np.linspace(-0.4, 0.4, 120), np.linspace(0.0, 0.144, 25)
    uu, hh = np.meshgrid(u, h, indexing='ij')
    wall = np.stack([np.full(uu.size, 2.0), uu.ravel(),
                     hh.ravel() + S.FLOOR_Z], axis=1)
    pockets = perception.find_pockets(
        wall, np.array([-1.0, 0.0, 0.0]),
        np.array([2.0, 0.0, S.FLOOR_Z + 0.072]),
        floor=Plane(normal=np.array([0.0, 0.0, 1.0]), offset=-S.FLOOR_Z),
        band_half_height=0.03,
        min_gap_width=0.15, max_gap_width=0.30,
        pocket_center_height=S.POCKET_CENTER_HEIGHT)
    assert len(pockets) == 0


# ---------------------------------------------------------- estimate_pallet


def _face_window(truth, distance):
    """Pixel rows and columns the pallet's front face projects into.

    Derived from the face's corners in base_footprint through the same
    extrinsic and intrinsics the synthetic renderer uses, then inset slightly
    so the window is inside the face rather than straddling its edge.
    """
    half = truth['face_width'] / 2
    floor_z = truth['floor_z']
    corners = np.array([[distance, -half, floor_z],
                        [distance, half, floor_z],
                        [distance, -half, floor_z + S.PALLET_HEIGHT],
                        [distance, half, floor_z + S.PALLET_HEIGHT]])
    cam = S.base_to_camera_matrix()
    pts = corners @ cam[:3, :3].T + cam[:3, 3]
    u = S.FX * pts[:, 0] / pts[:, 2] + S.CX
    v = S.FY * pts[:, 1] / pts[:, 2] + S.CY
    inset = 4
    return (slice(int(v.min()) + inset, int(v.max()) - inset),
            slice(int(u.min()) + inset, int(u.max()) - inset))


def test_estimate_pallet_end_to_end(params, intrinsics):
    """The whole pipeline on a synthetic depth image, against known truth.

    The fine `spacing` is not decoration: `depth_image` splats points, so a
    coarse cloud renders to a sparse depth map and the pipeline fails on
    sparsity rather than on geometry. A real ZED frame is 84 % valid.
    """
    points, truth = S.pallet_cloud(distance=2.0, yaw=0.0, noise=0.0,
                                   spacing=0.0015, floor_spacing=0.012)
    depth = S.depth_image(points)
    # What matters is the density of the pallet FACE, not of the whole image --
    # the floor can be coarse because RANSAC needs support, not resolution,
    # while clustering needs the face dense relative to its `eps`. Check the
    # window the face projects into rather than the global fraction.
    #
    # Computed from the pallet's own geometry rather than hard-coded pixels.
    # The hard-coded window was calibrated against the EPAL synthetic and went
    # stale the moment the measured pallet replaced it -- 5 mm narrower, and an
    # opening that reaches the floor -- which showed up as a 49.9 % reading
    # against a 50 % threshold, looking like a sparsity problem rather than a
    # window that no longer framed the face.
    face_rows, face_cols = _face_window(truth, distance=2.0)
    face_valid = float(np.isfinite(depth[face_rows, face_cols]).mean())
    assert face_valid > 0.5, (
        f'only {100 * face_valid:.0f} % of the pallet face region is valid; the '
        f'synthetic cloud is too coarse to represent a real frame')

    estimate = perception.estimate_pallet(
        depth, intrinsics, S.camera_to_base_matrix(), params, stamp=1.0)

    assert estimate is not None
    assert np.linalg.norm(estimate.position[:2] - truth['face_center'][:2]) < 0.05
    true_yaw = math.atan2(truth['normal'][1], truth['normal'][0])
    assert abs(math.sin(estimate.yaw - true_yaw)) < 0.09      # within ~5 deg
    assert len(estimate.pockets) == 2
    assert math.isclose(estimate.stamp, 1.0, abs_tol=1e-9)


def test_estimate_pallet_returns_none_on_an_empty_scene(params, intrinsics):
    """An empty room is NO_PALLET, not a crash and not a guess."""
    depth = np.full((S.IMAGE_HEIGHT, S.IMAGE_WIDTH), np.nan, dtype=np.float32)
    assert perception.estimate_pallet(
        depth, intrinsics, S.camera_to_base_matrix(), params) is None


# ------------------------------------------------- D62: the pocket line

def test_pocket_line_normal_is_perpendicular_to_the_pockets_and_keeps_the_sign():
    floor = Plane(np.array([0.0, 0.0, 1.0]), 0.025)
    pockets = (Pocket(np.array([2.0, 0.18, 0.05]), 0.23, 0.10),
               Pocket(np.array([2.1, -0.18, 0.05]), 0.23, 0.10))
    face_normal = np.array([-1.0, 0.0, 0.0])            # towards the robot
    n = perception.pocket_line_normal(pockets, face_normal, floor)
    line = pockets[1].center - pockets[0].center
    assert abs(n @ line) < 1e-9 and n @ face_normal > 0
    assert perception.pocket_line_normal(pockets[:1], face_normal, floor) is None


def test_the_pallet_direction_holds_still_under_depth_noise(params, intrinsics):
    """D62. The face plane's normal wobbled +-2.3 deg frame to frame in the sim;
    the pocket line must not. Same pallet, eight independent noisy frames."""
    yaws = []
    for seed in range(8):
        points, truth = S.pallet_cloud(distance=1.8, yaw=0.10, noise=0.01,
                                       spacing=0.0015, floor_spacing=0.012,
                                       seed=seed)
        estimate = perception.estimate_pallet(
            S.depth_image(points), intrinsics, S.camera_to_base_matrix(), params)
        assert estimate is not None and len(estimate.pockets) == 2
        yaws.append(estimate.yaw)
    true_yaw = math.atan2(truth['normal'][1], truth['normal'][0])
    errors = [math.degrees(math.atan2(math.sin(y - true_yaw), math.cos(y - true_yaw)))
              for y in yaws]
    assert max(errors) - min(errors) < 2.0, errors       # the stability limit
    assert max(abs(e) for e in errors) < 2.0, errors


def _solid_box(x0, x1, y0, y1, z0, z1, spacing):
    """Points on the visible surfaces of a box (front, top, sides)."""
    def grid(a0, a1, b0, b1):
        a, b = np.meshgrid(np.arange(a0, a1, spacing), np.arange(b0, b1, spacing))
        return a.ravel(), b.ravel()
    y, z = grid(y0, y1, z0, z1)
    front = np.c_[np.full_like(y, x0), y, z]
    x, y2 = grid(x0, x1, y0, y1)
    top = np.c_[x, y2, np.full_like(x, z1)]
    x3, z3 = grid(x0, x1, z0, z1)
    sides = np.r_[np.c_[x3, np.full_like(x3, y0), z3], np.c_[x3, np.full_like(x3, y1), z3]]
    return np.r_[front, top, sides]


def test_a_nearer_pallet_sized_crate_without_pockets_is_not_chosen(params, intrinsics):
    """D87, from the sim (2026-10-23): a crate of pallet height and pallet-face
    width passes the size and height gates; when it is the nearest candidate it
    used to win. Now the nearest candidate with a POCKET PAIR is preferred."""
    pallet, truth = S.pallet_cloud(distance=2.0, lateral=0.45, noise=0.0,
                                   spacing=0.0015, floor_spacing=0.012)
    top = S.FLOOR_Z + S.PALLET_HEIGHT
    crate = _solid_box(1.6, 2.2, -0.95, -0.15, S.FLOOR_Z, top, 0.0015)
    depth = S.depth_image(np.vstack([pallet, crate]))
    est = perception.estimate_pallet(depth, intrinsics, S.camera_to_base_matrix(), params)
    assert est is not None
    assert len(est.pockets) == 2
    assert est.position[1] == pytest.approx(0.45, abs=0.06)      # the pallet, not the crate
