"""Make Gazebo's perfect depth look like the ZED 2i's, as measured on real bags.

Gazebo renders depth from the geometry, so it is exact except for whatever
noise is added. The sim used to add 1 cm of independent Gaussian noise per
pixel per frame, which is the wrong KIND of error: perception averages that
away. Measured on the robot's own bags (2026-10-23, D80; numbers from
`floor_cam_pose1/2`, `pallet_s_129`, `pallet_2m_0deg`), the ZED's error is:

* **a smooth warp, static while the view is.** Frame-to-frame flicker is 0.1-1.7 mm
  over a 6 s bag of a parked robot, but
  the error against the true floor is correlated across neighbouring pixels at
  r = 1.00 out to 32 px, and its spread grows with range:
  sd = 0.74 mm * exp(1.46 z) fits 1.2-2.4 m to 1.6 mm rms (10 mm at 1.75 m,
  29 mm at 2.5 m). Beyond 2.4 m it is an extrapolation, capped at `warp_max`.
  It changes as the VIEW changes, not with time: a fresh pattern per
  `warp_motion` of progress (0.5 m driven or 0.17 rad turned, from odometry).
  That it changes with viewpoint at all is an assumption -- the bags only show
  it is static when nothing moves. A first version redrew it every 5 s
  regardless, and the pallet's measured direction then drifted ~6 deg in 2 s
  with the robot standing still, which the real camera does not do.
* **filled cavities.** A pocket is 1.2 m deep, but the ZED reports a surface
  0.2-0.35 m behind the face inside it, with few points.
* **smeared edges.** Face-to-pocket transitions are sharp at 1.75 m and spread
  over up to ~8 cm at 2 m, with depths in between ("flying pixels").
* **no holes** on the floor: 99.6-100 % of pixels valid (the ZED fills them).

Pure NumPy and no ROS, so `test/test_zed_depth_model.py` can check each effect.
`DepthModel.apply` is called by `depth_self_filter` on every published frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


@dataclass
class ZedParams:
    # Smooth warp: sd(z) = warp_a * exp(warp_b * z), capped.
    warp_a: float = 0.00074           # m
    warp_b: float = 1.46              # 1/m
    warp_max: float = 0.12            # m; the fit is unmeasured past 2.4 m
    warp_cells: tuple = (5, 8)        # coarse grid, ~160 px per cell at 1280:
                                      # r ~0.99 at 16 px (measured: 1.00 at 32 px)
    warp_motion: float = 1.0          # progress units for one grid to blend into
                                      # the next; see DepthModel.progress
    # Per-frame flicker: sd = flicker * z^2 (0.06-0.09 mm/m^2 measured).
    flicker: float = 0.00009          # m / m^2
    # Cavity fill: a pixel this much behind the nearest surface to its left,
    # its right and above, within cavity_width, is inside an opening the ZED
    # cannot see into; it reports a surface fill_min..fill_max behind instead.
    cavity_jump: float = 0.35         # m
    cavity_width: float = 0.30        # m, at the reference range below
    cavity_reference_range: float = 2.0
    fill_min: float = 0.20            # m
    fill_max: float = 0.35            # m
    # Edge smearing across depth jumps larger than edge_jump. Half-width of the
    # smear = smear_scale * warp sd at the near side: ~4 cm at 2 m, ~2 cm at
    # 1.75 m, as measured.
    edge_jump: float = 0.10           # m
    smear_scale: float = 2.5
    downsample: int = 4               # cavity search resolution


def warp_sd(z: np.ndarray, p: ZedParams) -> np.ndarray:
    return np.minimum(p.warp_a * np.exp(p.warp_b * z), p.warp_max)


def _upsample(grid: np.ndarray, height: int, width: int) -> np.ndarray:
    """Separable bilinear interpolation of a coarse grid onto the image."""
    gh, gw = grid.shape
    rows = np.linspace(0.0, gh - 1.0, height)
    cols = np.linspace(0.0, gw - 1.0, width)
    tmp = np.empty((gh, width))
    for i in range(gh):
        tmp[i] = np.interp(cols, np.arange(gw), grid[i])
    out = np.empty((height, width))
    for j in range(width):
        out[:, j] = np.interp(rows, np.arange(gh), tmp[:, j])
    return out


class DepthModel:
    """Stateful: the warp changes with the view, so it is told how far the
    robot has moved (`progress`: metres / 0.5 + radians / 0.17, accumulated)."""

    def __init__(self, params: ZedParams | None = None, seed: int | None = 0):
        self.p = params or ZedParams()
        self.rng = np.random.default_rng(seed)
        self._grids = (self.rng.standard_normal(self.p.warp_cells),
                       self.rng.standard_normal(self.p.warp_cells))
        self._t0 = None
        self._field_cache = {}

    # ------------------------------------------------------------- effects

    def warp_field(self, progress: float, height: int, width: int) -> np.ndarray:
        """Unit-variance smooth field, blending one coarse grid into the next
        as the robot moves; constant while it does not."""
        if self._t0 is None:
            self._t0 = progress
        while progress - self._t0 >= self.p.warp_motion:
            self._grids = (self._grids[1], self.rng.standard_normal(self.p.warp_cells))
            self._t0 += self.p.warp_motion
            self._field_cache.clear()
        s = (progress - self._t0) / self.p.warp_motion
        key = (height, width)
        if key not in self._field_cache:
            self._field_cache[key] = tuple(_upsample(g, height, width) for g in self._grids)
        a, b = self._field_cache[key]
        return ((1.0 - s) * a + s * b) / math.hypot(1.0 - s, s)

    def fill_cavities(self, depth: np.ndarray, fx: float) -> np.ndarray:
        p = self.p
        k = max(1, int(p.downsample))
        small = depth[::k, ::k].astype(float)
        small = np.where(np.isfinite(small), small, np.inf)
        w = max(2, int(math.ceil(p.cavity_width * fx / p.cavity_reference_range / k)))
        h, wd = small.shape
        pad_lr = np.pad(small, ((0, 0), (w, w)), constant_values=np.inf)
        windows = sliding_window_view(pad_lr, w, axis=1)            # (h, wd + w + 1, w)
        left = windows[:, :wd].min(axis=2)                          # columns i-w .. i-1
        right = windows[:, w + 1:w + 1 + wd].min(axis=2)            # columns i+1 .. i+w
        pad_t = np.pad(small, ((w, 0), (0, 0)), constant_values=np.inf)
        top = sliding_window_view(pad_t, w, axis=0)[:h].min(axis=2)  # rows i-w .. i-1
        jump = p.cavity_jump
        cavity = ((small > left + jump) & (small > right + jump) & (small > top + jump)
                  & np.isfinite(left) & np.isfinite(right) & np.isfinite(top)
                  & (np.abs(left - right) < jump))
        if not cavity.any():
            return depth
        offset = self.rng.uniform(p.fill_min, p.fill_max)
        fill = np.maximum(left, right) + offset
        full_mask = np.repeat(np.repeat(cavity, k, 0), k, 1)[:depth.shape[0], :depth.shape[1]]
        full_fill = np.repeat(np.repeat(fill, k, 0), k, 1)[:depth.shape[0], :depth.shape[1]]
        out = depth.copy()
        out[full_mask] = full_fill[full_mask]
        return out

    def smear_edges(self, depth: np.ndarray, fx: float) -> np.ndarray:
        """Blend across horizontal depth jumps: the far side takes in-between
        values over `half` pixels, as a stereo matcher's window straddles both."""
        p = self.p
        out = depth.copy()
        a, b = depth[:, :-1], depth[:, 1:]
        jumps = np.isfinite(a) & np.isfinite(b) & (np.abs(b - a) > p.edge_jump)
        rows, cols = np.nonzero(jumps)
        width = depth.shape[1]
        for r, c in zip(rows, cols):
            near_c, far_c = (c, c + 1) if depth[r, c] < depth[r, c + 1] else (c + 1, c)
            near, far = float(depth[r, near_c]), float(depth[r, far_c])
            half = int(p.smear_scale * float(warp_sd(near, p)) * fx / near)
            if half < 1:
                continue
            step = 1 if far_c > near_c else -1
            idx = far_c + step * np.arange(half)
            idx = idx[(idx >= 0) & (idx < width)]
            seg = out[r, idx]
            ok = np.isfinite(seg) & (seg > near + p.edge_jump / 2)
            frac = 1.0 - (np.arange(len(idx)) + 1.0) / (half + 1.0)   # 1 at the edge -> 0
            out[r, idx[ok]] = (seg + (near - seg) * frac)[ok]
        return out

    # --------------------------------------------------------------- whole

    def apply(self, depth: np.ndarray, fx: float, progress: float) -> np.ndarray:
        d = np.asarray(depth, dtype=np.float32)
        d = self.fill_cavities(d, fx)
        d = self.smear_edges(d, fx)
        finite = np.isfinite(d)
        z = np.where(finite, d, 0.0)
        field = self.warp_field(progress, *d.shape)
        noise = (field * warp_sd(z, self.p)
                 + self.rng.standard_normal(d.shape) * self.p.flicker * z * z)
        return np.where(finite, d + noise, d).astype(np.float32)
