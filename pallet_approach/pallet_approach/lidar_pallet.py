"""The pallet in the LiDAR, at pallet height, with no prior (D102 step a).

Only the slice 0.02-0.145 m above the fitted floor is used, so a load on the
pallet is invisible. In it: keep the nearest vertical surface at each bearing,
group returns into objects, fit the face line facing the robot (RANSAC, then
total least squares), accept it if it is a pallet face's width, and find the
pockets by sliding the side's block/pocket layout along the face's occupancy.
The face found is the block fronts (D102 item 3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from pallet_approach.model import Plane


@dataclass(frozen=True)
class LidarPallet:
    centre: np.ndarray          # (2,) base frame: the face's centre (block-front line)
    yaw: float                  # the face's outward normal, pointing at the robot; ~pi square-on
    width: float                # extent of the face returns along it, metres
    side: str                   # 'short', 'long', or '' when the width fits neither
    pocket_offset: float | None # where the pocket pair's centre sits along the face, from `centre`
    pocket_score: float         # mean occupancy at blocks minus at pockets; 0 = no structure
    points: int                 # returns on the face line
    rms: float                  # their perpendicular scatter, metres
    distance: float             # base origin to the face centre, metres
    in_range: bool              # inside the working range (D97)

    @property
    def tangent(self) -> np.ndarray:
        """Unit vector along the face, to the robot's left when square-on (yaw ~ pi)."""
        return np.array([math.sin(self.yaw), -math.cos(self.yaw)])

    @property
    def pocket_centre(self) -> np.ndarray | None:
        """(2,) the midpoint between the two pockets, or None if they were not found."""
        if self.pocket_offset is None:
            return None
        return self.centre + self.pocket_offset * self.tangent


def fit_floor(cloud: np.ndarray, p: dict, rng=None) -> Plane | None:
    """The floor near the robot as the LiDAR sees it, normal up."""
    z0 = p['floor_z']                  # where the floor should be, only to pick candidates
    cloud = cloud[::p['floor_stride']]  # the floor has returns to spare; the rest costs time
    r = np.hypot(cloud[:, 0], cloud[:, 1])
    cand = cloud[(np.abs(cloud[:, 2] - z0) < p['floor_search']) & (r > p['floor_min_range'])
                 & (r < p['floor_max_range'])]
    if len(cand) < 100:
        return None
    if len(cand) > 2000:
        cand = cand[np.random.default_rng(0).choice(len(cand), 2000, replace=False)]
    # RANSAC, all hypotheses at once (perception.fit_plane_ransac loops in
    # Python: 68 ms per sweep with the Jetson's NumPy 1.21).
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, len(cand), size=(150, 3))
    a, b, c = cand[idx[:, 0]], cand[idx[:, 1]], cand[idx[:, 2]]
    normals = np.cross(b - a, c - a)
    length = np.linalg.norm(normals, axis=1)
    keep = length > 1e-9
    normals, a = normals[keep] / length[keep, None], a[keep]
    normals[normals[:, 2] < 0] *= -1.0
    upright = normals[:, 2] >= math.cos(math.radians(5))
    if not upright.any():
        return None
    normals, a = normals[upright], a[upright]
    offsets = -np.einsum('ij,ij->i', normals, a)
    counts = (np.abs(cand @ normals.T + offsets) < p['floor_threshold']).sum(axis=0)
    best = int(np.argmax(counts))
    inliers = np.abs(cand @ normals[best] + offsets[best]) < p['floor_threshold']
    if inliers.sum() < 50:
        return None
    pts = cand[inliers]                                   # refine: least squares on the inliers
    centre = pts.mean(axis=0)
    normal = np.linalg.svd(pts - centre, full_matrices=False)[2][-1]
    if normal[2] < 0:
        normal = -normal
    if normal[2] < math.cos(math.radians(5)):
        return None
    return Plane(normal, float(-normal @ centre))


def first_surface(points: np.ndarray, heights: np.ndarray, p: dict) -> np.ndarray:
    """At each bearing, keep the nearest VERTICAL surface and `surface_depth` behind it.

    Removes the pallet's inner block rows and top deck (seen through and over
    the face). Vertical = returns within `surface_window` in range spanning
    `surface_min_height` in height, which floor returns read slightly high
    never do.
    """
    if len(points) == 0:
        return points
    bearing = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    rng = np.hypot(points[:, 0], points[:, 1])
    k = np.floor(bearing / p['surface_bin_deg']).astype(int)
    keep = np.zeros(len(points), bool)
    order = np.lexsort((rng, k))                      # by bin, then by range
    bins, starts = np.unique(k[order], return_index=True)
    ends = np.append(starts[1:], len(order))
    for a, b in zip(starts, ends):
        idx = order[a:b]
        r, h = rng[idx], heights[idx]
        for i in range(len(idx)):
            window = (r >= r[i]) & (r <= r[i] + p['surface_window'])
            if h[window].max() - h[window].min() >= p['surface_min_height']:
                keep[idx[r <= r[i] + p['surface_depth']]] = True
                break
    return points[keep]


def _clusters(xy: np.ndarray, link: float, cell: float = 0.02) -> list[np.ndarray]:
    """Indices of each group of points closer than `link` to one another.

    Linked by 2 cm cells rather than points: with a 0.45 m link over stacked
    sweeps, point pairs cost 120 of 190 ms.
    """
    if len(xy) == 0:
        return []
    keys = np.floor(xy / cell).astype(np.int64)
    cells, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    centres = (cells + 0.5) * cell
    pairs = cKDTree(centres).query_pairs(link, output_type='ndarray')
    n = len(centres)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    point_labels = labels[inverse]
    order = np.argsort(point_labels, kind='stable')
    bounds = np.flatnonzero(np.diff(point_labels[order])) + 1
    return np.split(order, bounds)


def _ransac_line(xy: np.ndarray, threshold: float, rng, iterations: int = 200,
                 max_yaw_deg: float = 90.0):
    """Best 2D line by inlier count among lines facing the robot (normal within
    `max_yaw_deg` of the line of sight), refined by total least squares."""
    n = len(xy)
    if n < 2:
        return None
    i = rng.integers(0, n, size=(iterations, 2))
    d = xy[i[:, 1]] - xy[i[:, 0]]
    length = np.linalg.norm(d, axis=1)
    ok = length > 0.05
    if not ok.any():
        return None
    d, a = d[ok] / length[ok, None], xy[i[ok, 0]]
    normals = np.stack([-d[:, 1], d[:, 0]], axis=1)
    sight = a / np.linalg.norm(a, axis=1, keepdims=True)
    facing = np.abs(np.einsum('kj,kj->k', normals, sight)) >= math.cos(math.radians(max_yaw_deg))
    if not facing.any():
        return None
    normals, a = normals[facing], a[facing]
    dist = np.abs(np.einsum('kj,knj->kn', normals, xy[None, :, :] - a[:, None, :]))
    best = int(np.argmax((dist < threshold).sum(axis=1)))
    inl = dist[best] < threshold
    for _ in range(2):
        pts = xy[inl]
        c = pts.mean(axis=0)
        direction = np.linalg.svd(pts - c, full_matrices=False)[2][0]
        normal = np.array([-direction[1], direction[0]])
        inl = np.abs((xy - c) @ normal) < threshold
    return c, direction, inl


def _pocket_search(along: np.ndarray, side: dict, half_span: float, p: dict):
    """Slide the block/pocket layout along the face; return (offset, score) or (None, 0)."""
    if len(along) < p['pocket_min_points']:
        return None, 0.0
    b = p['pocket_bin']
    edges = np.arange(-half_span, half_span + b, b)
    occupied = (np.histogram(along, bins=edges)[0] > 0).astype(float)
    centres = 0.5 * (edges[:-1] + edges[1:])
    w, s = side['pocket_width'], side['pocket_spacing']
    best, best_shift = -np.inf, None
    for shift in np.arange(-p['pocket_search'], p['pocket_search'] + 1e-9, b / 2):
        x = centres - shift
        in_pocket = (np.abs(np.abs(x) - s / 2) < w / 2 - b)
        in_block = ((np.abs(x) < s / 2 - w / 2 - b)
                    | ((np.abs(x) > s / 2 + w / 2 + b) & (np.abs(x) < side['face_width'] / 2 - b)))
        if in_pocket.sum() < 4 or in_block.sum() < 4:
            continue
        score = occupied[in_block].mean() - occupied[in_pocket].mean()
        if score > best:
            best, best_shift = score, shift
    if best_shift is None:
        return None, 0.0
    return float(best_shift), float(best)


def find_pallet(cloud: np.ndarray, params: dict, rng=None) -> tuple[LidarPallet | None, dict]:
    """The pallet in one or more LiDAR sweeps (N, 3) in the base frame.

    Returns (the best candidate or None, a small dict of diagnostics). The
    sweeps should be from a robot standing still; several can be stacked.
    """
    p = dict(params['lidar_acquisition'], floor_z=params['floor_z'])
    sides = params['pallet']['sides']
    rng = rng or np.random.default_rng(0)
    info = {'floor': False, 'band_points': 0, 'clusters': 0, 'rejected': []}
    floor = fit_floor(cloud, p, rng)
    if floor is None:
        return None, info
    info['floor'] = True
    info['floor_plane'] = floor
    h = floor.height_of(cloud)
    lo, hi = p['band']
    # The vertical test may use `surface_extra_height` above the slice, where
    # the deck edge gives a second beam; only slice points become the face.
    sel = ((h >= lo) & (h <= hi + p['surface_extra_height']) & (cloud[:, 0] > p['search_x'][0])
           & (cloud[:, 0] < p['search_x'][1]) & (np.abs(cloud[:, 1]) < p['search_y']))
    near = first_surface(cloud[sel], h[sel], p)
    band = near[floor.height_of(near) <= hi] if len(near) else near
    info['band_points'] = len(band)
    groups = [g for g in _clusters(band[:, :2], p['cluster_link']) if len(g) >= p['min_points']]
    info['clusters'] = len(groups)
    found = []
    for g in groups:
        pts = band[g]
        fit = _ransac_line(pts[:, :2], p['line_threshold'], rng, max_yaw_deg=p['max_face_yaw'])
        if fit is None:
            continue
        c, direction, inl = fit
        normal = np.array([-direction[1], direction[0]])
        if normal @ (-c) < 0:                       # outward normal points at the robot
            normal = -normal
        if abs(math.degrees(math.atan2(normal[1], -normal[0]))) > p['max_face_yaw']:
            info['rejected'].append(('faces away', len(g)))
            continue
        face = pts[inl]
        tangent = np.array([normal[1], -normal[0]])     # = LidarPallet.tangent
        along = (face[:, :2] - c) @ tangent
        lo_s, hi_s = np.percentile(along, [1, 99])
        width = float(hi_s - lo_s)
        side = next((name for name, sd in sides.items()
                     if abs(width - sd['face_width']) <= p['width_tolerance']), '')
        if not side:
            info['rejected'].append((f'width {width:.2f} m', len(g)))
            continue
        centre = c + tangent * (lo_s + hi_s) / 2
        sd = sides[side]
        face_h = floor.height_of(face)
        opening = (face_h >= sd['opening_bottom'] + 0.01) & (face_h <= sd['opening_top'] - 0.01)
        pocket_along = (face[opening, :2] - centre) @ tangent
        offset, score = _pocket_search(pocket_along, sd, width / 2 + 0.05, p)
        rms = float(np.sqrt(np.mean(((face[:, :2] - c) @ normal) ** 2)))
        distance = float(np.linalg.norm(centre))
        yaw = math.atan2(normal[1], normal[0])
        found.append(LidarPallet(centre, yaw, width, side, offset if score >= p['min_pocket_score'] else None,
                                 score, int(inl.sum()), rms, distance,
                                 p['working_range'][0] <= distance <= p['working_range'][1]))
    if not found:
        return None, info
    # The pallet the robot was brought to is the one most nearly straight ahead.
    best = min(found, key=lambda f: abs(math.atan2(f.centre[1], f.centre[0])))
    info['candidates'] = len(found)
    return best, info
