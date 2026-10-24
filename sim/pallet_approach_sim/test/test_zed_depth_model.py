"""Each measured ZED effect (D80) does what its measurement says, on a synthetic
depth image: a wall at 2.5 m with a pocket-sized opening into a 4 m room."""

import numpy as np
import pytest

from pallet_approach_sim.zed_depth_model import DepthModel, ZedParams, warp_sd

FX = 950.0
H, W = 360, 640


def wall(distance=2.5, hole_m=0.25, behind=4.0):
    depth = np.full((H, W), distance, dtype=np.float32)
    half = int(hole_m * FX / distance / 2)
    depth[200:300, W // 2 - half:W // 2 + half] = behind      # an opening with deck above
    return depth, half


def test_the_warp_has_the_measured_spread_and_is_smooth():
    spreads, correlation = [], []
    for seed in range(12):                       # several independent fields
        field = DepthModel(seed=seed).warp_field(0.0, H, W)
        spreads.append(field.std())
        correlation.append(np.corrcoef(field[:, :-8].ravel(), field[:, 8:].ravel())[0, 1])
    assert np.mean(correlation) > 0.98           # measured: 1.00 at 16-32 px
    assert 0.6 < np.mean(spreads) < 1.4          # unit variance, before scaling
    assert warp_sd(np.array(2.5), ZedParams()) == pytest.approx(0.00074 * np.exp(1.46 * 2.5))


def test_the_warp_is_static_while_the_view_is():
    """Measured: a parked robot's error does not change over a 6 s bag."""
    model = DepthModel(seed=3)
    a = model.warp_field(0.0, H, W)
    assert np.array_equal(a, model.warp_field(0.0, H, W))


def test_the_warp_changes_as_the_robot_moves():
    model = DepthModel(seed=3)
    a = model.warp_field(0.0, H, W)
    assert np.abs(model.warp_field(0.05, H, W) - a).max() < 0.2     # 2.5 cm driven
    assert np.abs(model.warp_field(2.0, H, W) - a).mean() > 0.3     # 1 m driven


def test_a_pocket_sized_opening_is_filled_a_little_behind_the_face():
    depth, half = wall()
    out = DepthModel(seed=0).fill_cavities(depth, FX)
    inside = out[230:290, W // 2 - half // 2:W // 2 + half // 2]
    assert np.all(inside - 2.5 >= 0.20 - 1e-6) and np.all(inside - 2.5 <= 0.35 + 1e-6)


def test_a_wide_opening_is_not_a_cavity():
    """The gap between two objects 1.5 m apart is not a pocket."""
    depth, _ = wall(hole_m=1.5)
    out = DepthModel(seed=0).fill_cavities(depth, FX)
    assert np.isclose(out[250, W // 2], 4.0)


def test_edges_are_smeared_at_range_with_in_between_depths():
    depth, half = wall(distance=2.5, hole_m=0.6, behind=3.0)
    out = DepthModel(seed=0).smear_edges(depth, FX)
    row = out[250]
    between = (row > 2.55) & (row < 2.95)
    assert 3 <= between.sum() <= 60              # a few cm of flying pixels per edge


def test_apply_keeps_the_scene_and_adds_bounded_error():
    depth, _ = wall(hole_m=0.6, behind=3.0)
    out = DepthModel(seed=0).apply(depth, FX, progress=0.0)
    face = out[50:150, 50:200]
    assert abs(np.median(face) - 2.5) < 0.1
    assert np.all(np.isfinite(out))
