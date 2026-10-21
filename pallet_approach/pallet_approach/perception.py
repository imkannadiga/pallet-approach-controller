"""Depth image -> pallet pose and fork pockets.

Each stage is a small function that can be tested on its own, and
`test/test_perception.py` has a test per function using synthetic pallet clouds
with known answers. Run them with:

    cd pallet_approach && python -m pytest test/test_perception.py -x -v

The pipeline, which is just `estimate_pallet` calling the others in order:

    depth image (32FC1, metres)
      -> depth_to_points        back-projection with the camera intrinsics
      -> transform_points       camera optical frame -> base_footprint
      -> crop_to_roi            throw away everything that cannot be the pallet
      -> voxel_downsample       one point per 2 cm cube, for speed
      -> fit_plane_ransac       find the floor
      -> remove_floor           keep what stands on it
      -> cluster_points         separate objects from each other
      -> select_pallet_cluster  pick the one that looks like a pallet
      -> fit_front_face         the vertical plane facing the robot
      -> find_pockets           the two gaps in that face
      -> PalletEstimate

Conventions, which are the single most common source of bugs here:

* The camera optical frame has **z forward, x right, y down**.
* The working frame is **`base_footprint`**: x forward, y left, z up. Not
  `base_link` -- that is the chassis body frame 0.235 m up, where every height
  would be a large negative offset. At ground level the numbers read as what
  they are: pocket centre 0.050 above the floor, fork underside 0.030.
* **base_footprint is not exactly on the floor.** Measured 2026-10-14, the real
  floor is at z = -0.025 there: the URDF's base_footprint joint is 25 mm out
  (its wheel geometry is right to 0.2 mm). So do not write `z > 0` anywhere and
  call it "above the floor" -- fit the floor, then work relative to the fit.
  That is why `remove_floor` and `find_pockets` both take the floor as an
  argument rather than assuming it.
* The measured transform is in docs/SYSTEM.md (`[0.390, 0.060, 0.373]` from
  base_footprint, zero pitch). Never hard-code it: look it up from tf at the
  image timestamp, because one link in the chain is republished at 58 Hz.
* Intrinsics come from the camera_info message, not from a constant. The ZED
  publishes rectified images with D = 0, so there is no undistortion step.
"""

from __future__ import annotations

import numpy as np

from pallet_approach.model import PalletEstimate, Plane, Pocket
from sklearn.cluster import DBSCAN

__all__ = [
    'depth_to_points',
    'transform_points',
    'crop_to_roi',
    'voxel_downsample',
    'fit_plane_ransac',
    'remove_floor',
    'cluster_points',
    'select_pallet_cluster',
    'fit_front_face',
    'face_yaw',
    'find_pockets',
    'estimate_pallet',
]


def depth_to_points(
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    stride: int = 1,
    min_range: float = 0.4,
    max_range: float = 4.0,
) -> np.ndarray:
    """Back-project a depth image into 3D points in the camera optical frame.

    Concept: a depth image stores, for every pixel, how far away the surface it
    sees is. A pixel on its own is just a direction; the depth value says how
    far along that direction the surface sits. Undo the projection and you get
    the 3D point.

    Maths, for a rectified image (ours is, so no distortion terms):

        Z = depth[v, u]
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

    Args:
        depth: (H, W) float32 array of metres. Invalid pixels are NaN or inf.
        fx, fy, cx, cy: intrinsics from the camera_info message.
        stride: keep every Nth pixel in both directions. 1280x720 is 921k
            points; stride 2 gives 230k, which is plenty and four times cheaper.
        min_range, max_range: drop points outside this depth band.

    Returns:
        (N, 3) float array in the camera optical frame, finite values only.

    """

    z = depth[::stride, ::stride].astype(np.float64)
    
    v, u = np.indices(depth.shape)
    v = v[::stride, ::stride]
    u = u[::stride, ::stride]

    valid = np.isfinite(z) & (z >= min_range) & (z <= max_range)
    z = z[valid]
    u = u[valid]
    v = v[valid]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
 
    return np.column_stack((x, y, z))


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to (N, 3) points.

    Concept: a 4x4 transform packs a rotation and a translation together:

        T = [ R  t ]      R is 3x3, t is 3x1
            [ 0  1 ]

    Applying it to a point is `R @ p + t`. The 4x4 form exists so that chaining
    transforms is just matrix multiplication.

    Here it carries points from the camera optical frame to `base_footprint`,
    so `transform` is the one tf gives for base_footprint <- camera_optical.

    Args:
        points: (N, 3).
        transform: (4, 4).

    Returns:
        (N, 3) in the target frame.

    """
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    
    return points @ rotation.T + translation


def crop_to_roi(
    points: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> np.ndarray:
    """Keep only the points inside an axis-aligned box in `base_footprint`.

    Why first: it is the cheapest stage and it removes the ceiling, the walls,
    the robot's own sensor box and anything behind the robot. Everything
    downstream then runs on a fraction of the data. The defaults are in
    params.yaml under `pallet_perception.roi`.

    Returns:
        (M, 3) subset of `points`.
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    inside = (
        (x >= x_min) & (x <= x_max)
        & (y >= y_min) & (y <= y_max)
        & (z >= z_min) & (z <= z_max)
    )
    return points[inside]


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Reduce to roughly one point per `voxel_size` cube.

    Concept: divide space into a grid of cubes and keep one representative point
    per occupied cube. Depth images oversample near surfaces, so this costs
    almost no information while cutting the point count a lot -- and it makes
    RANSAC and DBSCAN behave consistently regardless of how close the pallet is.

    Keep the voxel well below the smallest thing you need to measure: the pocket
    tolerance is 4 cm, so 2 cm is a reasonable choice.

    Returns:
        (M, 3), M <= N.

    """
    if len(points) == 0:
        return points

    # Which cube is each point in? Divide by the cube size and round DOWN.
    # np.floor matters: plain int() rounds -0.5 and +0.5 both to 0, which would
    # merge the two cubes either side of zero into one.
    cube_index = np.floor(points / voxel_size).astype(np.int64)

    # Flatten the three cube indices into ONE number per point, the way a 3D
    # array is addressed in memory: (ix * ny + iy) * nz + iz. Two points get
    # the same number exactly when they are in the same cube, so finding the
    # distinct cubes becomes finding the distinct numbers in a plain 1D array.
    #
    # Why bother: `np.unique(cube_index, axis=0)` has to sort an (N, 3) array
    # row by row, which NumPy does by viewing it as a structured dtype and
    # falling back to a slow comparison sort. On a real frame -- 160k points --
    # that version of this function took 110 ms, half the entire pipeline.
    # Sorting one integer array instead takes 14 ms for the same answer, and
    # the results are identical, not merely close (there is a test).
    #
    # Shifting by the minimum keeps the numbers small and positive; the span is
    # the pipeline's ROI, a few metres, so a few hundred cubes per axis.
    base = cube_index.min(axis=0)
    span = (cube_index.max(axis=0) - base + 1).astype(np.int64)
    relative = cube_index - base
    key = (relative[:, 0] * span[1] + relative[:, 1]) * span[2] + relative[:, 2]

    # `inverse` says, for every point, which distinct cube it belongs to (as a
    # number 0, 1, 2, ...). Example: cubes A, B, A -> unique [A, B], inverse
    # [0, 1, 0].
    _, inverse = np.unique(key, return_inverse=True)
    inverse = inverse.ravel()  # shape differs between NumPy versions; make it 1D
    n_cubes = inverse.max() + 1

    # Average the points in each cube. `np.bincount` with weights is a summing
    # scatter: it adds every point's x (then y, then z) into the slot its cube
    # owns. The obvious spelling, `np.add.at(sums, inverse, points)`, computes
    # the same thing but is an *unbuffered* ufunc call -- NumPy's slowest
    # scatter-add, and it was most of the remaining time here.
    counts = np.bincount(inverse, minlength=n_cubes)
    sums = np.empty((n_cubes, 3))
    for axis in range(3):
        sums[:, axis] = np.bincount(inverse, weights=points[:, axis],
                                    minlength=n_cubes)
    # counts[:, None] turns the list of counts into a column, so each row of
    # sums is divided by its own count.
    return sums / counts[:, None]




def fit_plane_ransac(
    points: np.ndarray,
    distance_threshold: float,
    iterations: int = 200,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit a plane with RANSAC and return (normal, offset, inlier_mask).

    Concept: least squares fits every point, so a single bad cluster drags the
    answer off. RANSAC instead guesses: pick 3 random points, build the plane
    through them, count how many other points lie within `distance_threshold`
    of it, and keep the guess with the most supporters. Outliers never get a
    vote because they never agree with each other.

    Math: a plane is `n . p + d = 0` with `|n| = 1`. From three points,
    `n = normalise(cross(p2 - p1, p3 - p1))` and `d = -n . p1`. The distance
    from any point to the plane is then `|n . p + d|`.

    Args:
        points: (N, 3).
        distance_threshold: how close counts as an inlier, metres. Set it near
            the sensor noise at the working range.
        iterations: how many random guesses.
        rng: pass a seeded `np.random.default_rng(0)` in tests so results repeat.

    Returns:
        normal: (3,) unit vector.
        offset: d, such that `normal . p + d = 0` on the plane.
        inlier_mask: (N,) bool.

        `Plane(normal, offset)` wraps the first two, and that is what the rest
        of the pipeline wants -- `estimate_pallet` should build one here and
        pass it on rather than reducing the floor to a single height.

    
    """
    if rng is None:
        rng = np.random.default_rng()
    n_points = len(points)
    if n_points < 3:
        raise ValueError('need at least 3 points to fit a plane')
 
    best_count = -1
    best_normal = None
    best_offset = 0.0
 
    for _ in range(iterations):
        p1, p2, p3 = points[rng.choice(n_points, size=3, replace=False)]
 
        normal = np.cross(p2 - p1, p3 - p1)
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        normal = normal / length 
        offset = -normal @ p1 
 
        distances = np.abs(points @ normal + offset)
        count = np.count_nonzero(distances < distance_threshold)
 
        if count > best_count:
            best_count = count
            best_normal = normal
            best_offset = offset
 
    if best_normal is None:
        raise ValueError('RANSAC found no plane (all samples were collinear)')
 
    inliers = np.abs(points @ best_normal + best_offset) < distance_threshold
    inlier_points = points[inliers]
    if len(inlier_points) >= 3:
        centroid = inlier_points.mean(axis=0)
        _, _, vt = np.linalg.svd(inlier_points - centroid, full_matrices=False)
        refined_normal = vt[-1]
        if refined_normal @ best_normal < 0:
            refined_normal = -refined_normal
        best_normal = refined_normal
        best_offset = -best_normal @ centroid
        inliers = np.abs(points @ best_normal + best_offset) < distance_threshold
 
    return best_normal, float(best_offset), inliers
 


def remove_floor(
    points: np.ndarray,
    floor: Plane,
    margin: float,
) -> np.ndarray:
    """Drop every point within `margin` of the floor plane, and below it.

    Note the plane is fitted, not assumed: in `base_footprint` the floor is at
    about z = -0.025, not 0, and it moves with tyre compression and load. The
    fit is what makes every height downstream correct regardless.

    Why a margin rather than exactly the plane: the floor is not perfectly flat,
    the depth has noise, and flying pixels at the pallet's bottom edge sit just
    above the ground. A 3 cm margin removes those without eating into the
    pocket openings, whose lower lip is 2.2 cm above the floor.

    Careful: that margin is a real trade-off and you should be able to defend
    it. Too small and the floor survives into the clustering stage and welds the
    pallet to the ground; too large and you start deleting the bottom boards.

    Takes the fitted `Plane`, not a height, and tests `floor.height_of(points)`
    -- the perpendicular distance. See `Plane` for why that matters.

    Returns:
        (M, 3) points above the floor.

    """
    heights = floor.height_of(points)
    return points[heights > margin]


def cluster_points(
    points: np.ndarray,
    eps: float,
    min_samples: int,
) -> np.ndarray:
    """Group points into objects with DBSCAN. Returns a label per point.

    Concept: DBSCAN grows a cluster from any point that has at least
    `min_samples` neighbours within `eps`, then keeps absorbing the neighbours'
    neighbours. Points with too few neighbours are labelled noise (-1). Unlike
    k-means it does not need the number of clusters in advance, which is the
    whole point here: we do not know how many objects are in front of the robot.

    Once the floor is gone, objects standing on it are physically separated, so
    this cleanly splits the pallet from boxes and walls.

    Args:
        points: (N, 3).
        eps: neighbourhood radius, metres.
        min_samples: minimum neighbours to be a core point.

    Returns:
        (N,) int array of labels; -1 is noise.

    Uses scikit-learn's DBSCAN; Open3D is deliberately not a dependency on the
    Jetson (see docker/Dockerfile.jetson).
    """
    if len(points) == 0:
        return np.empty(0, dtype=int)
 
    # eps: two points closer than this are neighbours (same island).
    # min_samples: a point needs this many neighbours to start/grow an island.
    #
    # algorithm='ball_tree' is a pure speed choice and changes no result: DBSCAN
    # needs the exact set of neighbours within eps, and every algorithm here
    # returns exactly that set -- they differ only in how they search for it.
    # Measured on a real frame (16k points above the floor): the default 'auto'
    # picks a kd-tree and takes 77 ms, a ball tree takes 36 ms, with identical
    # labels (there is a test). Ball trees win because these points lie on a few
    # 2D surfaces inside a 3D box, which is exactly the uneven density a kd-tree
    # splits badly.
    return DBSCAN(eps=eps, min_samples=min_samples,
                  algorithm='ball_tree').fit_predict(points)


def select_pallet_cluster(
    points: np.ndarray,
    labels: np.ndarray,
    pallet_width: float,
    pallet_height: float,
    size_tolerance: float = 0.25,
    floor: Plane | None = None,
    min_points: int = 50,
    widths: tuple[float, ...] = (),
) -> np.ndarray | None:
    """Pick the cluster that could be a pallet, or None.

    This is the gate that stops the system hallucinating pallets out of boxes
    and table legs. Judge each cluster on geometry only -- no learned model, as
    the whole project insists:

    * its height above the floor should be close to `pallet_height` (0.144 m),
      or taller if it carries a load;
    * its footprint should be close to `pallet_width` x pallet length;
    * it should be the nearest such cluster to the robot;
    * it needs enough points to fit a face to.

    Returns:
        (M, 3) points of the chosen cluster, or None if nothing qualifies.
        Returning None is a perfectly good answer -- it becomes NO_PALLET.

    TODO: implement.
    """
    candidates = []
 
    for label in np.unique(labels):
        if label == -1:          # -1 means "noise", not an object
            continue
        cluster = points[labels == label]
 
        # Check 1: enough points to fit a face to.
        if len(cluster) < min_points:
            continue
 
        # Check 2: tall enough. At least ~pallet height; taller is fine (load).
        if floor is not None:
            height = floor.height_of(cluster).max()
        else:
            height = cluster[:, 2].max() - cluster[:, 2].min()
        if height < pallet_height * (1 - size_tolerance):
            continue
 
        # Check 3: the right width. The pallet may be turned, so measuring the
        # plain y-extent would be wrong. Instead find the cluster's own two
        # horizontal directions (SVD on x, y, as in the plane fit) and measure
        # its size along each. Depending on what the camera sees, that is
        # (front face width, small) or (deck length, front face width), so it
        # is enough that ONE of the two sizes matches the pallet width.
        xy = cluster[:, :2]
        centred = xy - xy.mean(axis=0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        along_axes = centred @ vt.T                     # coordinates along both axes
        extents = along_axes.max(axis=0) - along_axes.min(axis=0)
        # Match against EVERY face the pallet could be presenting, not just the
        # one we expect. This gate runs before the side is known -- it has to,
        # since classification needs a cluster to work on -- so gating on the
        # expected side's width alone throws away pallets that are merely
        # turned. Measured on pallet_s_129_45: a pallet at 45 deg gives extents
        # that match neither nominal cleanly and was rejected outright, so the
        # AMBIGUOUS path it exists to test could never be reached.
        possible_widths = widths if widths else (pallet_width,)
        width_ok = any(
            np.any(np.abs(extents - possible) <= possible * size_tolerance)
            for possible in possible_widths)
        if not width_ok:
            continue
 
        # Passed every check: remember it with its horizontal distance.
        distance = np.linalg.norm(cluster[:, :2].mean(axis=0))
        candidates.append((distance, cluster))
 
    if not candidates:
        return None
    # Several candidates: the nearest one is the one to approach.
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def fit_front_face(
    points: np.ndarray,
    distance_threshold: float,
    max_tilt_from_vertical: float,
    rng: np.random.Generator | None = None,
    iterations: int = 200,
    max_facing_angle: float = np.radians(60),
    min_inliers: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Fit the pallet's front face: the vertical plane facing the robot.

    Concept: of the pallet's surfaces, the one we care about is the face the
    forks enter. Reuse `fit_plane_ransac` on the cluster, then check the plane
    is nearly vertical (its normal nearly horizontal) and nearly facing the
    robot. A fit that comes back tilted is usually the top deck, or the floor
    leaking through the margin.

    Returns:
        normal: (3,) unit normal, flattened later by `approach.outward_normal`.
        centroid: (3,) centre of the inlier points.
        inlier_mask: (N,) bool over `points`.
        Or None if no acceptable face was found.

    "Nearly vertical" means `abs(normal[2])` is small: the normal has almost
    no z component, bounded by `max_tilt_from_vertical`.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_points = len(points)
    if n_points < 2:
        return None
    xy = points[:, :2]
 
    best_count = 0
    best_normal = None
    best_offset = 0.0
 
    for _ in range(iterations):
        p1, p2 = xy[rng.choice(n_points, size=2, replace=False)]
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length < 1e-6:                      # same spot twice: no line
            continue
        # Turn the line's direction by 90 degrees: that's the plane's normal,
        # lying flat (z = 0), so the plane is upright by construction.
        normal = np.array([-direction[1], direction[0], 0.0]) / length
        offset = -normal[:2] @ p1
 
        # Point the normal at the robot (the origin): from the face toward it.
        midpoint = 0.5 * (p1 + p2)
        if normal[:2] @ (-midpoint) < 0:
            normal, offset = -normal, -offset
        # Reject planes that don't face the robot, e.g. the pallet's side when
        # it is turned: the angle between the normal and the direction to the
        # robot must be small.
        to_robot = -midpoint / max(np.linalg.norm(midpoint), 1e-9)
        if normal[:2] @ to_robot < np.cos(max_facing_angle):
            continue
 
        count = np.count_nonzero(np.abs(xy @ normal[:2] + offset) < distance_threshold)
        if count > best_count:
            best_count, best_normal, best_offset = count, normal, offset
 
    if best_normal is None or best_count < min_inliers:
        return None
 
    mask = np.abs(xy @ best_normal[:2] + best_offset) < distance_threshold
    centroid = points[mask].mean(axis=0)
 
    # Refinement in full 3D (same SVD trick as in fit_plane_ransac). This also
    # lets us CHECK the assumption that the face is upright, instead of just
    # forcing it.
    _, _, vt = np.linalg.svd(points[mask] - centroid, full_matrices=False)
    normal = vt[-1]
    if normal @ best_normal < 0:
        normal = -normal
    # A plane tilted by angle a from vertical has |normal_z| = sin(a).
    if abs(normal[2]) > np.sin(max_tilt_from_vertical):
        return None
    return normal, centroid, mask


def face_yaw(normal: np.ndarray) -> float:
    """The face's heading: the angle of its normal in the ground plane.

    Maths: project the normal onto the ground plane and take
    `atan2(normal[1], normal[0])`. That is the whole conversion from "a plane
    in 3D" to "the one angle the controller needs".

    A plane normal is enough to recover yaw; pitch and roll are not
    recoverable from a single vertical plane.

    Returns:
        radians.

    TODO: implement.
    """
    return float(np.arctan2(normal[1], normal[0]))


def find_pockets(
    face_points: np.ndarray,
    face_normal: np.ndarray,
    face_centroid: np.ndarray,
    floor: Plane,
    band_half_height: float,
    min_gap_width: float,
    max_gap_width: float,
    pocket_center_height: float,
    bin_width: float = 0.02,
    min_points_per_bin: int = 1,
    opening_bottom: float = 0.0,
) -> tuple[Pocket, ...]:
    """Find the fork pockets as gaps in the face, left to right.

    Concept: the pockets are the places where the face is *missing*. On a EUR
    pallet seen from the 800 mm side, at pocket height you see block, gap,
    block, gap, block. So:

    1. Take the face points in a band around `pocket_center_height` (0.072 m
       above the floor), selected with `floor.height_of(points)` -- the
       perpendicular distance to the fitted floor, not a z-coordinate. That is
       what makes the measurement immune to the camera's extrinsic error; see
       `Plane`. It also means the band follows a sloping floor instead of
       drifting off the pockets at one end of the pallet.
    2. Express them in a 2D frame on the face: one axis along the face, one up.
       Along-face coordinate for a point p is `dot(p - centroid, along)` where
       `along = cross(up, normal)`.
    3. Histogram the along-face coordinate. Occupied bins are blocks; runs of
       empty bins between blocks are pockets.
    4. Keep runs whose width is between `min_gap_width` and `max_gap_width`,
       and measure each one's centre, width and vertical extent.

    The nominal answers for an EPAL 1 pallet are in params.yaml: two pockets,
    0.2275 m wide, 0.100 m high, centres 0.3725 m apart. Those are what to
    compare against -- not what to assume. If your gap analysis has to be told
    the answer to find it, it will not survive a damaged pallet.

    Returns:
        pockets ordered left to right as the robot sees them. Fewer than two is
        a valid outcome and becomes a NO-GO with a reason.

    TODO: implement.
    """
    if len(face_points) == 0:
        return ()
 
    # --- Set up coordinates on the face -----------------------------------
    # "up" = the floor's normal (not the z axis: the floor may be slightly
    # tilted relative to the robot).
    up = np.asarray(floor.normal, dtype=float)
    # "along" = horizontal direction along the face. It is perpendicular to
    # both up and the face normal, which is exactly what a cross product gives.
    along = np.cross(up, face_normal)
    # The face is only NEARLY vertical, so this vector is slightly shorter
    # than 1. Normalise it, or every measurement comes out slightly too small.
    along = along / np.linalg.norm(along)
 
    heights = floor.height_of(face_points)           # how high above the floor
    s = (face_points - face_centroid) @ along         # how far along the face
 
    # --- 1. Keep the points at pocket height --------------------------------
    in_band = np.abs(heights - pocket_center_height) <= band_half_height
    s_band = s[in_band]
    if len(s_band) == 0:
        return ()
 
    # --- 2. Sort them into buckets along the face ---------------------------
    # Buckets cover the whole face (all heights), so the pallet's outer ends
    # are known even if the band itself is ragged at the edges.
    s_min, s_max = s.min(), s.max()
    n_bins = max(1, int(np.ceil((s_max - s_min) / bin_width)))
    bin_of = np.clip(((s_band - s_min) / bin_width).astype(int), 0, n_bins - 1)
    counts = np.bincount(bin_of, minlength=n_bins)
    occupied = counts >= min_points_per_bin            # wood (True) or gap (False)
 
    # --- 3. Find runs of empty buckets --------------------------------------
    # Pad with "occupied" at both ends, then look where occupied/empty changes.
    # +1 in the difference = an empty run starts, -1 = it ends.
    empty = np.concatenate(([0], (~occupied).astype(int), [0]))
    changes = np.diff(empty)
    run_starts = np.flatnonzero(changes == 1)          # first empty bucket
    run_ends = np.flatnonzero(changes == -1)           # one past the last empty bucket
 
    pockets = []
    for start, end in zip(run_starts, run_ends):
        # A pocket has wood on BOTH sides. Empty runs touching the ends of the
        # face are just "beyond the pallet", not pockets.
        if start == 0 or end == n_bins:
            continue
 
        # Measure the gap from the actual points at its edges (more precise
        # than counting buckets): last point before it, first point after it.
        left_edge = s_band[bin_of == start - 1].max()
        right_edge = s_band[bin_of == end].min()
        width = right_edge - left_edge
 
        # --- 4. Keep only gaps of a plausible pocket size --------------------
        if not (min_gap_width <= width <= max_gap_width):
            continue
        s_center = 0.5 * (left_edge + right_edge)
 
        # Vertical size: look at face points (all heights) that lie inside the
        # gap horizontally. The highest one below the band is the bottom edge
        # of the opening, the lowest one above it the top edge. Stay one
        # bucket away from the gap's sides: the edges of the neighbouring
        # blocks are ragged, and their points would look like "inside" points.
        in_gap = (s > left_edge + bin_width) & (s < right_edge - bin_width)
        gap_heights = heights[in_gap]
        below = gap_heights[gap_heights < pocket_center_height]
        above = gap_heights[gap_heights > pocket_center_height]
        top = above.min() if len(above) else np.nan    # nothing above: unknown

        # The opening's LOWER edge is not observable on this pallet, from
        # either face, so it is taken from the known geometry rather than from
        # the points:
        #
        #   - short face: the opening starts at the floor. There is no lower
        #     edge to see -- the floor runs continuously into the pocket.
        #   - long face: the opening starts on a 20 mm bottom board, and
        #     `remove_floor` has already deleted everything within the 0.03 m
        #     floor margin, so the board's top edge went with it.
        #
        # So `opening_bottom` is the side's nominal (0.000 short, 0.020 long)
        # and the measured quantity is the opening's TOP edge. Measuring down
        # from the top to a known floor is the honest version of what the old
        # `bottom = 0.0` fallback was doing by accident -- right for the short
        # face, 20 mm short for the long one. If the lower edge ever has to be
        # measured, the floor margin has to come down first (2.4 mm of floor
        # RMS would allow ~0.015), and that is a change to argue for with data.
        bottom = below.max() if len(below) else opening_bottom
        pocket_height = top - bottom
        center_height = (0.5 * (top + bottom) if np.isfinite(top)
                         else pocket_center_height)
 
        # Back to 3D: start at the face centre, slide along the face to the
        # gap's middle, then move up/down to the opening's middle height.
        center = (face_centroid
                  + s_center * along
                  + (center_height - floor.height_of(face_centroid)) * up)
 
        # ADAPT TO model.py: field names of Pocket.
        pockets.append(Pocket(center=center, width=float(width),
                              height=float(pocket_height)))
 
    # Left to right as the robot sees it: left is +y in base_footprint.
    pockets.sort(key=lambda p: -p.center[1])
    return tuple(pockets)


def face_axis(face_normal: np.ndarray, floor: Plane) -> np.ndarray:
    """Unit vector along the face, horizontal in the floor's frame.

    Perpendicular to both the floor normal and the face normal, which is what a
    cross product gives. The face is only NEARLY vertical, so the result is
    slightly shorter than 1 and has to be normalised, or every length measured
    along the face comes out slightly too small.
    """
    along = np.cross(np.asarray(floor.normal, dtype=float), face_normal)
    return along / np.linalg.norm(along)


def face_extent(face_points: np.ndarray, face_normal: np.ndarray,
                face_centroid: np.ndarray, floor: Plane) -> float:
    """How wide the visible face is, measured along the face itself.

    Along the face, not along y: a turned pallet's y-extent is its true width
    times cos(yaw), which would read a square-on long face and a turned long
    face as different objects.

    This is a LOWER BOUND on the real width, never an upper one. Occlusion, the
    ROI crop and the image edge can all cut a face short, and nothing can make
    it look wider than it is. That asymmetry is why `classify_side` treats it as
    confirmation only (D22).
    """
    if len(face_points) == 0:
        return 0.0
    along = face_axis(face_normal, floor)
    s = (face_points - face_centroid) @ along
    return float(s.max() - s.min())


def flatten_side(side: dict) -> dict:
    """One side's geometry under the flat names the pipeline stages take.

    The stages each want one number -- `pocket_center_height`, not
    `sides.short.band_center` -- because at the point of use there is only one
    side in play. This is the single place that translates, so params.yaml can
    stay grouped by face and the stage signatures stay readable.
    """
    return {
        'pallet_width': side['face_width'],
        'pallet_length': side['depth'],
        'pocket_center_height': side['band_center'],
        'pocket_band_half_height': side['band_half_height'],
        'pocket_min_gap_width': side['min_gap_width'],
        'pocket_max_gap_width': side['max_gap_width'],
        'pocket_nominal_width': side['pocket_width'],
        'pocket_nominal_height': side['opening_top'] - side['opening_bottom'],
        'pocket_nominal_spacing': side['pocket_spacing'],
        'pocket_opening_bottom': side['opening_bottom'],
        'pocket_opening_top': side['opening_top'],
    }


def pocket_spacing(pockets, face_normal: np.ndarray,
                   floor: Plane) -> float | None:
    """Distance between two pocket centres, measured along the face.

    None unless there are exactly two: with one the question does not arise,
    and with three or more the scene is not a pallet face we understand, so
    picking a pair would be inventing an answer.
    """
    if len(pockets) != 2:
        return None
    along = face_axis(face_normal, floor)
    return float(abs((pockets[0].center - pockets[1].center) @ along))


AMBIGUOUS = 'ambiguous'


def classify_side(
    pocket_spacing: float | None,
    face_width: float,
    sides: dict,
    classify: dict,
) -> tuple[str, dict]:
    """Which face of the pallet is the robot looking at?

    Concept: the two faces of this pallet are different structures, not two
    sizes of one thing (D19), so four things differ between them. Two of them
    the pipeline already measures:

        =================  ==========  =========  ======
                            short       long       ratio
        face width          0.795       1.195      1.50
        pocket spacing      0.365       0.525      1.44
        =================  ==========  =========  ======

    **Pocket spacing decides; face width only confirms.** That is the opposite
    of the obvious choice, and the reason is that the two signals fail
    differently. Width is a lower bound -- occlusion, the ROI crop and the image
    edge all shorten it, nothing lengthens it -- so a 0.9 m reading could be a
    complete short face or a clipped long one, and there is no way to tell from
    the number. Spacing is a difference between two centres, so a clipped face
    either yields both pockets or fewer than two, and fewer than two is a case
    this function is told about rather than one it can be fooled by. Spacing
    also separates the faces by 160 mm against a measured error of about 13 mm.

    Both signals must agree. Disagreement means something is wrong with the
    scene -- a damaged pallet, a load overhanging the face, two pallets pushed
    together -- and guessing between two sets of nominals that differ by 160 mm
    is how forks end up driven at the wrong geometry.

    Args:
        pocket_spacing: distance between the two pocket centres along the face,
            or None if fewer than two pockets were found.
        face_width: the visible face width from `face_extent`.
        sides: {name: geometry}, as params.yaml's `pallet.sides`.
        classify: tolerances, as params.yaml's `pallet.classify`.

    Returns:
        (name, detail). `name` is a key of `sides`, or `AMBIGUOUS`. `detail`
        carries the evidence -- what each signal picked, and how far each side
        was from each measurement -- so a refusal can be explained after the
        run instead of being a bare "no".
    """
    spacing_tol = classify['spacing_tolerance']
    width_tol = classify['width_tolerance']

    spacing_fit = {name: (abs(pocket_spacing - cfg['pocket_spacing'])
                          if pocket_spacing is not None else float('inf'))
                   for name, cfg in sides.items()}
    width_fit = {name: abs(face_width - cfg['face_width'])
                 for name, cfg in sides.items()}

    def sole_match(fit, tolerance):
        """The one side within tolerance, or None if none or several are."""
        within = [name for name, error in fit.items() if error <= tolerance]
        return within[0] if len(within) == 1 else None

    by_spacing = sole_match(spacing_fit, spacing_tol)
    by_width = sole_match(width_fit, width_tol)

    detail = {'by_spacing': by_spacing, 'by_width': by_width,
              'spacing': pocket_spacing, 'face_width': face_width,
              'spacing_error': spacing_fit, 'width_error': width_fit}

    if by_spacing is None:
        detail['reason'] = ('no pocket pair to measure spacing from'
                            if pocket_spacing is None
                            else f'spacing {pocket_spacing:.3f} m matches no side')
        return AMBIGUOUS, detail
    if by_width is not None and by_width != by_spacing:
        detail['reason'] = (f'spacing says {by_spacing}, width says {by_width}')
        return AMBIGUOUS, detail

    detail['reason'] = ('spacing and width agree' if by_width == by_spacing
                        else 'spacing decided; width inconclusive')
    return by_spacing, detail


def _face_axis_of(normal: np.ndarray) -> np.ndarray:
    """Horizontal unit vector along a face, perpendicular to its normal."""
    axis = np.array([-normal[1], normal[0], 0.0])
    return axis / float(np.linalg.norm(axis))


def estimate_pallet(
    depth: np.ndarray,
    intrinsics: dict,
    camera_to_base: np.ndarray,
    params: dict,
    stamp: float = 0.0,
    decision_points: dict | None = None,
) -> PalletEstimate | None:
    """Run the whole pipeline on one depth frame. None means "no pallet here".

    This is the only function the ROS node calls. Keep it as a straight
    sequence of the stages above with no cleverness of its own: when something
    goes wrong on the robot you want to be able to log and plot every
    intermediate result, and that is much easier if this function is boring.

    Args:
        depth: (H, W) float32 metres.
        intrinsics: {'fx', 'fy', 'cx', 'cy'} from camera_info.
        camera_to_base: (4, 4) base_footprint <- camera_optical, from tf at
            `stamp`. Note the target frame: ground level, not the body frame.
        params: the `pallet_perception` block of params.yaml, as a dict.
        decision_points: optional dict, filled in with `cluster` (the selected
            pallet cluster) and `scene` (floor-removed points in a box around
            the face). Two GO/NO-GO rules need points rather than an estimate
            -- `check_overhang` and `check_pocket_clearance` -- and both treat
            missing data as a NO-GO rather than a pass, so without this the
            coordinator refuses every run. The node publishes what lands here;
            the coordinator passes it to the rules.
        stamp: image timestamp in seconds, copied into the estimate.

    Returns:
        PalletEstimate, or None.
    """
    rng = np.random.default_rng(params.get('seed'))
 
    # 1-2. Distances -> dots -> robot's point of view.
    points = depth_to_points(
        depth, intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy'],
        stride=params['stride'],
        min_range=params['min_range'], max_range=params['max_range'])
    points = transform_points(points, camera_to_base)
 
    # 3-4. Keep the box in front of the robot, then thin it out.
    roi = params['roi']
    points = crop_to_roi(points, roi['x_min'], roi['x_max'], roi['y_min'],
                         roi['y_max'], roi['z_min'], roi['z_max'])
    points = voxel_downsample(points, params['voxel_size'])
    if len(points) < 3:
        return None
 
    # 5. Find the floor.
    normal, offset, _ = fit_plane_ransac(
        points, params['floor_distance_threshold'],
        iterations=params['ransac_iterations'], rng=rng)
    # Make the floor normal point UP (remove_floor and find_pockets rely on it).
    if normal[2] < 0:
        normal, offset = -normal, -offset
    # Sanity check: RANSAC finds the BIGGEST plane. If that is steep, it is a
    # wall or a big box, not the floor; carrying on would give nonsense.
    if np.arccos(np.clip(normal[2], -1.0, 1.0)) > params['max_floor_tilt']:
        return None
    floor = Plane(normal, offset)
 
    # 6-8. Delete the floor, group into objects, pick the pallet.
    points = remove_floor(points, floor, params['floor_margin'])
    labels = cluster_points(points, params['cluster_eps'], params['cluster_min_samples'])
    # Every face the pallet could be showing. Which one it IS gets decided
    # below, from the pockets; this gate only has to not throw the cluster away.
    widths = tuple(cfg['face_width'] for cfg in params.get('sides', {}).values())
    pallet = select_pallet_cluster(
        points, labels, params['pallet_width'], params['pallet_height'],
        size_tolerance=params['size_tolerance'], floor=floor,
        min_points=params['min_cluster_points'], widths=widths)
    if pallet is None:
        return None
 
    # 9-10. Its front face and the angle it faces.
    face = fit_front_face(pallet, params['face_distance_threshold'],
                          params['max_tilt_from_vertical'], rng=rng)
    if face is None:
        return None
    face_normal, face_centroid, face_mask = face
    yaw = face_yaw(face_normal)

    if decision_points is not None:
        # The cluster, for the overhang rule, and a BOX AROUND THE FACE for the
        # pocket clearance rule -- not the whole scene. The clearance boxes are
        # `pocket_clearance_depth` deep and no wider than a pocket, so points
        # outside this box cannot affect either rule, and publishing the whole
        # ROI instead would be hundreds of kilobytes a second for information
        # nothing reads.
        decision_points['cluster'] = pallet
        lateral = max([cfg['face_width'] for cfg in params.get('sides', {}).values()]
                      or [params['pallet_width']]) / 2.0 + 0.20
        ahead = (points - face_centroid) @ face_normal
        sideways = (points - face_centroid) @ _face_axis_of(face_normal)
        near_face = ((ahead > -0.10) & (ahead < 0.60)
                     & (np.abs(sideways) <= lateral)
                     & (floor.height_of(points) < 0.60))
        decision_points['scene'] = points[near_face]
 
    # 11. The holes. Fewer than two is NOT "no pallet": the estimate is still
    # returned, and go_nogo turns it into a NO-GO with a reason.
    #
    # find_pockets works on a SLAB around the face plane, not on the plane's
    # RANSAC inliers. The real pallet's front surface spans 26 mm from the deck
    # boards to the lower blocks, so a 15 mm inlier band samples mostly deck and
    # starves the gap analysis -- it found one pocket instead of two on the
    # first real bag (D5). The pockets stay empty inside the slab because they
    # are deep.
    distance_from_face = (pallet - face_centroid) @ face_normal
    face_slice = pallet[np.abs(distance_from_face) < params['pocket_slab_half_depth']]

    min_per_bin = params.get('pocket_min_points_per_bin', 1)
    width = face_extent(pallet[face_mask], face_normal, face_centroid, floor)

    # 12. Which face are we looking at? The two differ enough that measuring
    # one against the other's nominals is meaningless, so this has to happen
    # before the pockets are measured -- but it needs pockets to decide with.
    # Hence two passes: a probe with a band and gap limits wide enough to suit
    # EITHER face, then the real measurement with the chosen face's own.
    side, detail = params.get('entry_face', 'unknown'), {}
    sides = params.get('sides')
    classify = params.get('classify')
    geometry = params
    if sides and classify and classify.get('enabled', True):
        probe = find_pockets(
            face_slice, face_normal, face_centroid, floor,
            classify['band_half_height'], classify['min_gap_width'],
            classify['max_gap_width'], classify['band_center'],
            bin_width=params['voxel_size'], min_points_per_bin=min_per_bin)
        spacing = pocket_spacing(probe, face_normal, floor)
        side, detail = classify_side(spacing, width, sides, classify)
        if side == AMBIGUOUS:
            # No defensible nominals to measure against, so nothing is claimed.
            # The estimate still goes out: "a pallet, turned so I cannot tell
            # which face" is different from "no pallet", and only the
            # coordinator gets to decide what to do about it (D22).
            return PalletEstimate(
                position=face_centroid, normal=face_normal, yaw=yaw,
                pockets=(), face_inlier_count=int(face_mask.sum()),
                floor=floor, side=AMBIGUOUS, side_detail=detail,
                face_width=width, stamp=stamp)
        geometry = {**params, **flatten_side(sides[side])}

    pockets = find_pockets(
        face_slice, face_normal, face_centroid, floor,
        geometry['pocket_band_half_height'], geometry['pocket_min_gap_width'],
        geometry['pocket_max_gap_width'], geometry['pocket_center_height'],
        bin_width=params['voxel_size'], min_points_per_bin=min_per_bin,
        opening_bottom=geometry.get('pocket_opening_bottom', 0.0))

    # `position` and `normal` describe the front face; `face_inlier_count` is
    # the support the face fit had, which go_nogo checks before trusting it.
    return PalletEstimate(position=face_centroid, normal=face_normal, yaw=yaw,
                          pockets=pockets,
                          face_inlier_count=int(face_mask.sum()),
                          floor=floor, side=side, side_detail=detail,
                          face_width=width, stamp=stamp)
 
