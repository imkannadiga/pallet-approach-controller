"""The sensors' range error against distance, from the range-error study (D109).

`lidar-camera-calib range-error` (2026-11-02) measured each sensor's distance to
a flat board against a tape at 20 positions, 1.0-3.9 m from the bumper-centre
mark, and fitted a bias and a noise for each as functions of the MEASURED
distance m from that mark: true = m - bias(m), and R(m) = sigma_total(m)^2.
Outside the measured range the end values hold; nothing is extrapolated.

A sensor's distance s from itself to a surface is m + sensor_minus_mark (the
sensor sits behind the mark). The numbers live in params.yaml, `range_error`.
"""

from __future__ import annotations

import numpy as np


def _clip(m, block: dict):
    lo, hi = block['valid_range']
    return np.clip(m, lo, hi)


def lidar_bias(m, block: dict):
    """Measured minus true, metres, at mark distance m: a + b m (linear fit)."""
    m = _clip(np.asarray(m, dtype=float), block)
    return block['bias_a'] + block['bias_b'] * m


def sigma_total(m, block: dict):
    """One standard deviation of a range measurement at mark distance m, metres."""
    m = _clip(np.asarray(m, dtype=float), block)
    return np.interp(m, block['sigma_m'], block['sigma_total'])


def correct_lidar(points: np.ndarray, origin: np.ndarray, block: dict) -> np.ndarray:
    """Remove the LiDAR's range bias from (N, 3) points, along each ray from
    `origin` (the LiDAR in the same frame)."""
    rays = points - origin
    r = np.linalg.norm(rays, axis=1)
    keep = r > 1e-6
    out = points.copy()
    m = r[keep] - block['sensor_minus_mark']
    out[keep] = origin + rays[keep] * ((r[keep] - lidar_bias(m, block)) / r[keep])[:, None]
    return out


def camera_true_depth(z, disparity_bias: float, offset: float = 0.0):
    """The ZED's depth with its error removed: s/(1 - beta s) - a.

    The study's model, error = a + s - s/(1 - beta s) with s the measured
    sensor-to-surface distance, is a constant disparity offset (beta, D103) plus
    a constant range offset (a). Applied per pixel to the depth along the axis.
    """
    z = np.asarray(z, dtype=float)
    return z / (1.0 - disparity_bias * z) - offset
