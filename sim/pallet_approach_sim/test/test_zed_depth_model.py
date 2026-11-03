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


def _scene():
    """A level camera 0.4 m up, looking at a floor and at a wall 2.5 m away."""
    import numpy as np
    h, w, fx = 360, 640, 475.0
    v, u = np.mgrid[0:h, 0:w].astype(float)
    ray_y = (v - h / 2) / fx                               # optical y, down
    with np.errstate(divide='ignore'):
        floor_z = np.where(ray_y > 0, 0.4 / ray_y, np.inf)  # where each ray meets the floor
    depth = np.minimum(floor_z, 2.5).astype(np.float32)     # a wall at 2.5 m
    return depth, floor_z, u, v, w, h, fx


def test_vertical_faces_carry_the_range_studys_bias_and_noise_the_floor_keeps_d80():
    """D109: on the wall, the error's mean is the study's bias and its spread
    across views the study's sigma_position less the tape."""
    import math
    import numpy as np
    from pallet_approach_sim.zed_depth_model import DepthModel, ZedParams, warp_sd
    depth, floor_z, u, v, w, h, fx = _scene()
    model = DepthModel(ZedParams(), seed=3)
    flat = model.horizontal(depth, fx)
    assert flat[(floor_z < 2.0) & (np.abs(u - w / 2) < 200)].mean() > 0.95
    assert not flat[(depth == 2.5) & (v < h / 2 - 40)].any()
    p = ZedParams()
    s = 2.5
    expected_bias = p.vertical_bias_offset + s - s / (1 - p.vertical_bias_beta * s)
    errors = []
    for seed in range(200):                                  # many views
        m = DepthModel(ZedParams(), seed=seed)
        out = m.apply(depth, fx, progress=0.0)
        errors.append(float(np.median((out - depth)[(depth == 2.5) & (v < h / 2 - 40)])))
    errors = np.array(errors)
    # The model's measured depth is u / (1 + beta u); the study's error is
    # measured - true. At 2.5 m that is about -24 mm (camera reads short).
    assert np.mean(errors) == pytest.approx(-(s - (s + p.vertical_bias_offset)
                                              / (1 + p.vertical_bias_beta * (s + p.vertical_bias_offset))),
                                            abs=0.003)
    assert np.mean(errors) == pytest.approx(expected_bias, abs=0.003)
    assert np.std(errors) == pytest.approx(math.hypot(p.vertical_offset_sd, 0.0), rel=0.25)
    assert warp_sd(np.array([2.5]), ZedParams())[0] > 0.02   # the floor's model is unchanged


def test_the_sim_camera_and_lidar_use_the_numbers_in_params_yaml():
    """The sim image cannot import the stack, so its numbers are copies: same values."""
    import math
    import pathlib
    import yaml
    from pallet_approach_sim.zed_depth_model import ZedParams
    from pallet_approach_sim.lidar_error import LidarError
    params = pathlib.Path(__file__).resolve().parents[3] / 'pallet_approach' / 'config' / 'params.yaml'
    if not params.exists():
        pytest.skip('params.yaml not beside the sim package')
    e = yaml.safe_load(params.read_text())['/**']['ros__parameters']['range_error']
    c, li, p = e['camera'], e['lidar'], ZedParams()
    assert (p.vertical_bias_beta, p.vertical_bias_offset) == (c['disparity_bias'], c['offset'])
    assert (p.vertical_frame_sd0, p.vertical_frame_exponent) == (c['sigma_frame0'], c['sigma_frame_exponent'])
    assert p.sensor_minus_mark == c['sensor_minus_mark']
    lidar_tape = LidarError().tape_sigma
    assert p.vertical_offset_sd == pytest.approx(math.sqrt(c['sigma_position'] ** 2 - lidar_tape ** 2), abs=1e-5)
    le = LidarError()
    assert (le.bias_a, le.bias_b, le.sensor_minus_mark) == (li['bias_a'], li['bias_b'], li['sensor_minus_mark'])
    assert le.valid_range == tuple(li['valid_range'])


def test_the_lidar_error_moves_returns_along_their_rays_by_the_studys_bias():
    from pallet_approach_sim.lidar_error import LidarError
    e = LidarError()
    xyz = np.array([[3.0, 0.0, 0.0], [0.0, 1.5, 0.2], [0.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    out = e.apply(xyz)
    r0, r1 = np.linalg.norm(xyz, axis=1), np.linalg.norm(out, axis=1)
    assert r1[0] - r0[0] == pytest.approx(e.bias_a + e.bias_b * (3.0 - e.sensor_minus_mark))
    assert np.allclose(out[1] / r1[1], xyz[1] / r0[1])
    assert np.all(out[2] == 0.0)                                       # no return stays none
    assert r1[3] - r0[3] == pytest.approx(e.bias_a + e.bias_b * e.valid_range[1])   # held, not extrapolated


def test_the_lidar_error_rewrites_a_pointcloud_buffer_in_place_of_its_fields():
    from pallet_approach_sim.lidar_error import LidarError
    e = LidarError()
    dt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('pad', '<f4'), ('intensity', '<f4')])
    pts = np.zeros(3, dtype=dt)
    pts['x'], pts['y'], pts['z'], pts['intensity'] = [2.0, 0.0, 1.0], [0.0, 3.0, 1.0], [0.1, 0.0, 0.5], 7.0
    out = np.frombuffer(e.apply_to_buffer(pts.tobytes(), dt.itemsize, {'x': 0, 'y': 4, 'z': 8}), dtype=dt)
    xyz = np.stack([pts['x'], pts['y'], pts['z']], 1).astype(float)
    assert np.allclose(np.stack([out['x'], out['y'], out['z']], 1), e.apply(xyz), atol=1e-6)
    assert np.all(out['intensity'] == 7.0)                    # other fields untouched
