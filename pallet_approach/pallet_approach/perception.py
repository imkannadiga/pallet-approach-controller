"""Depth image -> pallet pose and fork pockets (the camera pipeline).

`estimate_pallet` runs the stages in order:

    depth_to_points -> transform_points (optical -> base_footprint)
    -> crop_to_roi -> voxel_downsample -> fit_plane_ransac (floor)
    -> remove_floor -> cluster_points -> select_pallet_cluster
    -> fit_front_face -> find_pockets -> PalletEstimate

Frames: camera optical is z forward, x right, y down; the working frame is
`base_footprint` (x forward, y left, z up). The floor there is at about
z = -0.025 (docs/SYSTEM.md), so heights are always measured from the fitted
floor plane, never from z = 0. The camera transform comes from tf at the image
stamp; intrinsics from camera_info (rectified, D = 0).
"""

from __future__ import annotations

import numpy as np

from pallet_approach.model import PalletEstimate, Plane, Pocket
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

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
    """Back-project a depth image to (N, 3) points in the camera optical frame:
    Z = depth[v, u], X = (u - cx) Z / fx, Y = (v - cy) Z / fy. Every `stride`th
    pixel, finite depths within [min_range, max_range] only.
    """

    z = depth[::stride, ::stride].astype(np.float64)
    
    # Only the strided pixels' coordinates: np.indices over the whole image
    # built two 1280x720 arrays a frame to throw three quarters away.
    v, u = np.meshgrid(np.arange(0, depth.shape[0], stride),
                       np.arange(0, depth.shape[1], stride), indexing='ij')

    valid = np.isfinite(z) & (z >= min_range) & (z <= max_range)
    z = z[valid]
    u = u[valid]
    v = v[valid]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
 
    return np.column_stack((x, y, z))


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to (N, 3) points: R p + t."""
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
    """Keep the points inside an axis-aligned box in `base_footprint`
    (`pallet_perception.roi`): the cheapest stage, so it runs first.
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    inside = (
        (x >= x_min) & (x <= x_max)
        & (y >= y_min) & (y <= y_max)
        & (z >= z_min) & (z <= z_max)
    )
    return points[inside]


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """About one point per `voxel_size` cube (2 cm, below the 4 cm pocket
    tolerance), so RANSAC and clustering behave the same at any range.
    """
    if len(points) == 0:
        return points

    # Which cube is each point in? Divide by the cube size and round DOWN.
    # np.floor matters: plain int() rounds -0.5 and +0.5 both to 0, which would
    # merge the two cubes either side of zero into one.
    cube_index = np.floor(points / voxel_size).astype(np.int64)

    # One integer per cube, (ix * ny + iy) * nz + iz, so finding distinct cubes
    # is a 1D unique: 14 ms on a real frame against 110 ms for
    # np.unique(axis=0), with identical results (tested).
    base = cube_index.min(axis=0)
    span = (cube_index.max(axis=0) - base + 1).astype(np.int64)
    relative = cube_index - base
    key = (relative[:, 0] * span[1] + relative[:, 1]) * span[2] + relative[:, 2]

    # `inverse` says, for every point, which distinct cube it belongs to (as a
    # number 0, 1, 2, ...). Example: cubes A, B, A -> unique [A, B], inverse
    # [0, 1, 0].
    cells = int(span[0] * span[1] * span[2])
    if cells <= 8_000_000:
        # The ROI at 2 cm is ~1.1 M cells: mark the occupied ones in a dense
        # array and number them in key order, instead of sorting ~200 k keys.
        # Same cubes in the same (sorted-key) order as np.unique, so the output
        # is identical.
        occupied = np.zeros(cells, dtype=bool)
        occupied[key] = True
        cubes = np.flatnonzero(occupied)
        rank = np.empty(cells, dtype=np.int32)
        rank[cubes] = np.arange(len(cubes), dtype=np.int32)
        inverse = rank[key]
    else:
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
    score_points: int | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """RANSAC plane fit: (normal, offset, inlier_mask), with n . p + offset = 0.

    Planes through random point triples; the one with most points within
    `distance_threshold` wins. Pass a seeded `rng` in tests.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_points = len(points)
    if n_points < 3:
        raise ValueError('need at least 3 points to fit a plane')
 
    best_count = -1
    best_normal = None
    best_offset = 0.0
 
    # All the samples in one draw. rng.choice(n, 3, replace=False) shuffles all
    # n points to pick three -- O(n) per iteration, 200 times a frame, and the
    # largest single cost on the Jetson (74 ms of a 186 ms frame, 2026-10-23).
    # A sample that repeats a point gives a zero cross product and is skipped
    # below, exactly as a collinear one is.
    samples = rng.integers(0, n_points, size=(iterations, 3))
    # Score the hypotheses on a random subset when the cloud is large: the
    # count only has to RANK planes, and 3000 points rank them as well as
    # 20 000 do. The winner is still refined on every point below. ~4x faster
    # for the floor on a real frame (2026-10-23).
    scored = points
    if score_points and n_points > score_points:
        scored = points[rng.integers(0, n_points, size=score_points)]
    for p1, p2, p3 in points[samples]:
 
        normal = np.cross(p2 - p1, p3 - p1)
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        normal = normal / length 
        offset = -normal @ p1 
 
        distances = np.abs(scored @ normal + offset)
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
    """Drop every point within `margin` of the fitted floor plane, or below it.
    Heights are perpendicular distances to the fit (`Plane.height_of`); the margin
    removes floor noise and edge pixels without eating into the pockets.
    """
    heights = floor.height_of(points)
    return points[heights > margin]


def cluster_points(
    points: np.ndarray,
    eps: float,
    min_samples: int,
) -> np.ndarray:
    """DBSCAN labels per point (-1 = noise): neighbours within `eps`, core points
    with at least `min_samples`. Built on SciPy's KD-tree (same clusters as
    scikit-learn's, a border point touching two may go to either; twice as fast).
    """
    if len(points) == 0:
        return np.empty(0, dtype=int)
 
    # eps: two points closer than this are neighbours (same island).
    # min_samples: a point needs this many neighbours (itself included, as in
    # scikit-learn) to start/grow an island.
    n = len(points)
    pairs = cKDTree(points).query_pairs(eps, output_type='ndarray')
    i, j = pairs[:, 0], pairs[:, 1]
    core = np.bincount(pairs.ravel(), minlength=n) + 1 >= min_samples
    both = core[i] & core[j]
    graph = coo_matrix((np.ones(int(both.sum()), dtype=bool), (i[both], j[both])),
                       shape=(n, n))
    _, component = connected_components(graph, directed=False)
    labels = np.full(n, -1, dtype=int)
    if core.any():
        # Number the clusters 0..k-1 by their core components.
        ids = np.full(component.max() + 1, -1, dtype=int)
        used = np.unique(component[core])
        ids[used] = np.arange(len(used))
        labels[core] = ids[component[core]]
        # A non-core point within eps of a core point is a border point of
        # that core's cluster.
        for a, b in ((i, j), (j, i)):
            border = core[a] & ~core[b]
            labels[b[border]] = labels[a[border]]
    return labels


def select_pallet_cluster(
    points: np.ndarray,
    labels: np.ndarray,
    pallet_width: float,
    pallet_height: float,
    size_tolerance: float = 0.25,
    floor: Plane | None = None,
    min_points: int = 50,
    widths: tuple[float, ...] = (),
    max_height: float | None = None,
) -> np.ndarray | None:
    """The nearest cluster that could be a pallet, or None (see
    `pallet_candidates`, which this wraps)."""
    candidates = pallet_candidates(points, labels, pallet_width, pallet_height,
                                   size_tolerance, floor, min_points, widths,
                                   max_height)
    return candidates[0] if candidates else None


def pallet_candidates(
    points: np.ndarray,
    labels: np.ndarray,
    pallet_width: float,
    pallet_height: float,
    size_tolerance: float = 0.25,
    floor: Plane | None = None,
    min_points: int = 50,
    widths: tuple[float, ...] = (),
    max_height: float | None = None,
) -> list[np.ndarray]:
    """Every cluster that could be a pallet, nearest first: height near
    `pallet_height` (and under `max_height` when given, D76), footprint near the
    pallet's, enough points for a face. Empty means NO_PALLET.
    """
    candidates = []
 
    for label in np.unique(labels):
        if label == -1:          # -1 means "noise", not an object
            continue
        cluster = points[labels == label]
 
        # Check 1: enough points to fit a face to.
        if len(cluster) < min_points:
            continue
 
        # Check 2: pallet height, and not much more (D76: an edge-on partition
        # panel won every frame without the upper bound).
        if floor is not None:
            height = floor.height_of(cluster).max()
        else:
            height = cluster[:, 2].max() - cluster[:, 2].min()
        if height < pallet_height * (1 - size_tolerance):
            continue
        if max_height is not None and height > max_height:
            continue
 
        # Check 3: width along the cluster's own horizontal axes (yaw-
        # invariant); it is enough that one of the two matches.
        xy = cluster[:, :2]
        centred = xy - xy.mean(axis=0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        along_axes = centred @ vt.T                     # coordinates along both axes
        extents = along_axes.max(axis=0) - along_axes.min(axis=0)
        # Against every face the pallet could present: the side is not known
        # yet, and a turned pallet must still reach classification.
        possible_widths = widths if widths else (pallet_width,)
        width_ok = any(
            np.any(np.abs(extents - possible) <= possible * size_tolerance)
            for possible in possible_widths)
        if not width_ok:
            continue
 
        # Passed every check: remember it with its horizontal distance.
        distance = np.linalg.norm(cluster[:, :2].mean(axis=0))
        candidates.append((distance, cluster))

    # Nearest first. select_pallet_cluster takes the first; estimate_pallet
    # prefers the nearest that shows two pockets (D87).
    candidates.sort(key=lambda item: item[0])
    return [cluster for _, cluster in candidates]


def fit_front_face(
    points: np.ndarray,
    distance_threshold: float,
    max_tilt_from_vertical: float,
    rng: np.random.Generator | None = None,
    iterations: int = 200,
    max_facing_angle: float = np.radians(60),
    min_inliers: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """The pallet's front face: an upright plane facing the robot, from RANSAC on
    the cluster (two points define an upright plane, so the top deck cannot win).
    Returns (normal, centroid, inlier_mask) or None. Upright = |normal z| within
    `max_tilt_from_vertical`.
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
 
    # One draw for all iterations, as in fit_plane_ransac; a repeated point is
    # caught by the zero-length check below.
    samples = rng.integers(0, n_points, size=(iterations, 2))
    for p1, p2 in xy[samples]:
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
    """The face normal's heading in the ground plane: atan2(n_y, n_x), radians."""
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
    frame_margin: float = 0.05,
) -> tuple[Pocket, ...]:
    """The fork pockets as gaps in the face, left to right as the robot sees them.

    Points in a band around `pocket_center_height` above the fitted floor
    (perpendicular distance), projected along the face and histogrammed: occupied
    bins are blocks, runs of empty bins between `min_gap_width` and
    `max_gap_width` are pockets, each with its centre, width and height. Fewer
    than two is a valid result (a NO-GO with a reason). Nominals are in
    params.yaml's `pallet.sides`; they are compared against, never assumed.
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

        # The opening's lower edge is not observable (the floor on the short
        # face; a 20 mm board inside remove_floor's margin on the long one), so
        # it is the side's nominal and the measured quantity is the TOP edge.
        bottom = below.max() if len(below) else opening_bottom
        pocket_height = top - bottom
        center_height = (0.5 * (top + bottom) if np.isfinite(top)
                         else pocket_center_height)
 
        # Back to 3D: start at the face centre, slide along the face to the
        # gap's middle, then move up/down to the opening's middle height.
        center = (face_centroid
                  + s_center * along
                  + (center_height - floor.height_of(face_centroid)) * up)

        # D62: each pocket gets its own depth, the median distance from the
        # plane of the wood framing it, so the line through the two pockets
        # measures the pallet's direction instead of the RANSAC plane's.
        near = np.abs(s - s_center) <= width / 2 + frame_margin
        if np.count_nonzero(near) >= 10:
            depth = float(np.median((face_points[near] - face_centroid) @ face_normal))
            center = center + depth * np.asarray(face_normal, dtype=float)
 
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
    """The visible face's width, measured along the face (so it does not shrink
    with yaw). A lower bound only: occlusion and the image edge shorten it,
    which is why `classify_side` uses it as confirmation (D22).
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
    """Which face the robot is looking at: (side name or AMBIGUOUS, detail).

    Pocket spacing decides (0.365 vs 0.525 m, ~13 mm measured error) and face
    width (0.795 vs 1.195 m, a lower bound) only confirms (D19, D22). If they
    disagree the answer is AMBIGUOUS: guessing between nominals 160 mm apart is
    how forks are driven at the wrong geometry. `detail` carries the evidence.
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


def pocket_line_normal(pockets, face_normal: np.ndarray, floor: Plane):
    """The pallet's direction from its two pocket centres (D62), or None.

    Line through the two centres, flattened onto the floor plane; its normal in
    that plane, pointing the same way as the fitted face's normal so every
    sign convention downstream is unchanged. Needs exactly the structure the
    forks need: two pockets.
    """
    if len(pockets) < 2:
        return None
    up = np.asarray(floor.normal, dtype=float)
    along = np.asarray(pockets[-1].center, dtype=float) - np.asarray(pockets[0].center, dtype=float)
    along = along - (along @ up) * up
    length = float(np.linalg.norm(along))
    if length < 1e-6:
        return None
    normal = np.cross(along / length, up)
    if normal @ np.asarray(face_normal, dtype=float) < 0:
        normal = -normal
    return normal / np.linalg.norm(normal)


def estimate_pallet(
    depth: np.ndarray,
    intrinsics: dict,
    camera_to_base: np.ndarray,
    params: dict,
    stamp: float = 0.0,
    decision_points: dict | None = None,
) -> PalletEstimate | None:
    """The whole pipeline on one depth frame; None means no pallet.

    `camera_to_base` is base_footprint <- camera optical from tf at `stamp`;
    `params` the `pallet_perception` block. `decision_points`, if given, is
    filled with `cluster` and `scene` for the GO/NO-GO point rules (D61, D63).
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
        iterations=params['ransac_iterations'], rng=rng,
        score_points=params.get('floor_score_points'))
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
    # Cluster only below the D76 ceiling (+0.10 m): taller objects are still
    # rejected, and walls mostly drop out of the costliest step.
    ceiling = params['pallet_height'] + params['max_height_above_pallet'] + 0.10
    low = floor.height_of(points) < ceiling
    labels = np.full(len(points), -1, dtype=int)
    labels[low] = cluster_points(points[low], params['cluster_eps'],
                                 params['cluster_min_samples'])
    # Every face the pallet could be showing. Which one it IS gets decided
    # below, from the pockets; this gate only has to not throw the cluster away.
    widths = tuple(cfg['face_width'] for cfg in params.get('sides', {}).values())
    candidates = pallet_candidates(
        points, labels, params['pallet_width'], params['pallet_height'],
        size_tolerance=params['size_tolerance'], floor=floor,
        min_points=params['min_cluster_points'], widths=widths,
        max_height=params['pallet_height'] + params['max_height_above_pallet'])
    # D87: the nearest candidate that shows a POCKET PAIR, else the nearest.
    # Size and height alone let a pallet-height crate through, and in the sim
    # it took over whenever noise made the pallet fail a gate for a frame
    # (~110 cm jumps that reset the tracker); a crate has no pockets.
    chosen, chosen_points = None, None
    for pallet in candidates[:int(params.get('max_candidates', 3))]:
        local = {} if decision_points is not None else None
        estimate = _estimate_from_cluster(pallet, points, floor, params, rng, stamp, local)
        if estimate is None:
            continue
        if len(estimate.pockets) == 2:
            chosen, chosen_points = estimate, local
            break
        if chosen is None:
            chosen, chosen_points = estimate, local
    if decision_points is not None and chosen_points:
        decision_points.update(chosen_points)
    return chosen


def _estimate_from_cluster(pallet, points, floor, params, rng, stamp, decision_points):
    """Steps 9-12 of `estimate_pallet` for one candidate cluster."""
    # 9-10. Its front face and the angle it faces. Fitted below the top deck
    # (D88): seen from above the deck is a big patch of points, and a vertical
    # plane through it rivals the real front face, so the random draw decided.
    below = pallet[floor.height_of(pallet)
                   < params['pallet_height'] - params.get('face_deck_margin', 0.0)]
    fit_on = below if len(below) >= params.get('face_min_points', 60) else pallet
    face = fit_front_face(fit_on, params['face_distance_threshold'],
                          params['max_tilt_from_vertical'], rng=rng)
    if face is None:
        return None
    face_normal, face_centroid, _ = face
    face_mask = np.abs((pallet - face_centroid) @ face_normal) < params['face_distance_threshold']
    yaw = face_yaw(face_normal)

    if decision_points is not None:
        # The cluster for the overhang rule, and a box around the face for the
        # clearance rule; nothing outside it affects either.
        decision_points['cluster'] = pallet
        lateral = max([cfg['face_width'] for cfg in params.get('sides', {}).values()]
                      or [params['pallet_width']]) / 2.0 + 0.20
        ahead = (points - face_centroid) @ face_normal
        sideways = (points - face_centroid) @ _face_axis_of(face_normal)
        near_face = ((ahead > -0.10) & (ahead < 0.60)
                     & (np.abs(sideways) <= lateral)
                     & (floor.height_of(points) < 0.60))
        decision_points['scene'] = points[near_face]
 
    # 11. The holes. Fewer than two still returns the estimate (go_nogo says
    # NO-GO). Searched in a slab around the face, not the plane's inliers: the
    # front surface spans 26 mm, so the inliers are mostly deck (D5).
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
        opening_bottom=geometry.get('pocket_opening_bottom', 0.0),
        frame_margin=params.get('pocket_frame_margin', 0.05))

    # D62: with both pockets found, the pallet's direction is the outward
    # normal of the line through their centres. The fitted face only stands
    # in when a pocket is missing (and such a frame is refused by GO/NO-GO).
    normal_from_pockets = pocket_line_normal(pockets, face_normal, floor)
    if normal_from_pockets is not None:
        face_normal = normal_from_pockets
        yaw = face_yaw(face_normal)

    # `position` and `normal` describe the front face; `face_inlier_count` is
    # the support the face fit had, which go_nogo checks before trusting it.
    return PalletEstimate(position=face_centroid, normal=face_normal, yaw=yaw,
                          pockets=pockets,
                          face_inlier_count=int(face_mask.sum()),
                          floor=floor, side=side, side_detail=detail,
                          face_width=width, stamp=stamp)
 
