"""The overlay's geometry, which is the half that can be silently wrong.

Drawing errors announce themselves -- the picture looks wrong. Geometry errors
do not: forks a couple of centimetres off still look like forks, and the
overlay's whole job is to make a couple of centimetres visible. So these tests
are about the maths, not the pixels.
"""

import numpy as np
import pytest

from pallet_approach import fork_overlay as FO
from pallet_approach.model import Plane, Pocket

FORKS = dict(tine_length=1.150, tine_width=0.100, tine_thickness=0.040,
             spacing=0.365, bottom_height=0.040, carriage_x=0.4625)
LEVEL = Plane(np.array([0.0, 0.0, 1.0]), 0.025)      # floor at z = -0.025


def heights_above(points, floor):
    return floor.height_of(points.reshape(-1, 3))


def test_shape_and_spacing():
    corners = FO.tine_corners(floor=LEVEL, **FORKS)
    assert corners.shape == (2, 8, 3)
    centres = [tine[:, 1].mean() for tine in corners]
    assert abs(centres[0] - centres[1]) == pytest.approx(FORKS['spacing'])
    # each tine is tine_width across and tine_length long
    for tine in corners:
        assert np.ptp(tine[:, 1]) == pytest.approx(FORKS['tine_width'])
        assert np.ptp(tine[:, 0]) == pytest.approx(FORKS['tine_length'])


def test_the_tines_start_at_the_carriage_and_reach_the_tip():
    corners = FO.tine_corners(floor=LEVEL, **FORKS)
    assert corners[:, :, 0].min() == pytest.approx(FORKS['carriage_x'])
    assert corners[:, :, 0].max() == pytest.approx(
        FORKS['carriage_x'] + FORKS['tine_length'])


def test_heights_are_above_the_floor_not_z_values():
    """`bottom_height` is a distance from the floor PLANE. With the floor at
    z = -0.025, an underside 0.040 above it is at z = +0.015, not z = 0.040."""
    corners = FO.tine_corners(floor=LEVEL, **FORKS)
    assert corners[:, :4, 2].min() == pytest.approx(-0.025 + 0.040)
    heights = heights_above(corners, LEVEL)
    assert sorted(set(np.round(heights, 6))) == [
        pytest.approx(FORKS['bottom_height']),
        pytest.approx(FORKS['bottom_height'] + FORKS['tine_thickness'])]


@pytest.mark.parametrize('pitch_deg', [0.0, 1.0, -2.5, 5.0])
def test_the_tines_stay_parallel_to_a_SLOPING_floor(pitch_deg):
    """The one that matters, and the reason `_at_height` solves the plane
    equation instead of adding to z.

    On a tilted floor, `z = floor_z + height` tips the forks relative to the
    floor by exactly the tilt -- and the tilt the overlay exists to reveal is a
    camera orientation error of under a degree. An overlay that leans by the
    error it is looking for cannot show it. Every corner must be its nominal
    height from the PLANE, whatever the plane is doing.
    """
    pitch = np.radians(pitch_deg)
    normal = np.array([-np.sin(pitch), 0.0, np.cos(pitch)])
    floor = Plane(normal, 0.025 * np.cos(pitch))

    corners = FO.tine_corners(floor=floor, **FORKS)
    heights = heights_above(corners, floor)
    assert np.allclose(np.sort(np.unique(np.round(heights, 9))),
                       [FORKS['bottom_height'],
                        FORKS['bottom_height'] + FORKS['tine_thickness']],
                       atol=1e-9), (
        f'at {pitch_deg} deg of floor slope the tines are no longer parallel '
        f'to the floor: heights span {heights.min():.4f}-{heights.max():.4f}')


def test_points_behind_the_camera_are_masked_not_projected():
    """A negative Z divides to a finite pixel on the wrong side of the image,
    so the point reappears mirrored and looks like a geometry bug. It has to be
    masked out, not clamped."""
    points = np.array([[0.0, 0.0, 2.0],       # in front
                       [0.0, 0.0, -2.0],      # behind
                       [0.0, 0.0, 0.0]])      # exactly at the plane
    pixels, visible = FO.project_points(points, 900.0, 900.0, 640.0, 360.0)
    assert visible.tolist() == [True, False, False]
    assert pixels[0] == pytest.approx([640.0, 360.0])


def test_projection_is_the_inverse_of_back_projection():
    """The overlay's projection and perception's back-projection are the same
    pinhole model used in opposite directions; a point put through both must
    come back where it started, or one of them has a sign or an axis wrong."""
    from pallet_approach.perception import depth_to_points
    fx = fy, cx, cy = 900.0, 640.0, 360.0
    depth = np.full((720, 1280), np.nan, dtype=np.float32)
    depth[400, 700] = 2.5
    points = depth_to_points(depth, 900.0, 900.0, cx, cy, 1, 0.1, 5.0)
    assert len(points) == 1
    pixels, visible = FO.project_points(points, 900.0, 900.0, cx, cy)
    assert visible.all()
    assert pixels[0] == pytest.approx([700.0, 400.0], abs=1e-6)


def test_to_camera_frame_keeps_the_shape():
    corners = FO.tine_corners(floor=LEVEL, **FORKS)
    moved = FO.to_camera_frame(corners, np.eye(4))
    assert moved.shape == corners.shape
    assert np.allclose(moved, corners)


def test_every_face_of_a_tine_is_drawn_once():
    """Six faces, each using four of the eight corners, and between them every
    corner twice over -- a box with a face missing has a hole you see through."""
    assert len(FO.TINE_FACES) == 6
    used = [index for face in FO.TINE_FACES for index in face]
    assert sorted(set(used)) == list(range(8))
    assert all(used.count(index) == 3 for index in range(8)), (
        'each corner of a box belongs to exactly three of its faces')


def test_faces_are_ordered_furthest_first():
    """Painter's algorithm: the box draws inside out without it.

    Each corner is given a unique pixel so the returned polygons can be mapped
    back to the faces they came from -- otherwise this test can only count
    them, which is what the first version of it did.
    """
    corners = FO.tine_corners(floor=LEVEL, **FORKS)[0]
    in_camera = corners + np.array([0.0, 0.0, 3.0])      # push it in front
    pixels = np.column_stack([np.arange(8.0), np.arange(8.0)])
    visible = np.ones(8, dtype=bool)

    faces = FO.tine_faces(in_camera, pixels, visible)
    assert len(faces) == 6

    by_corners = {frozenset(indices): np.mean(in_camera[list(indices), 2])
                  for indices in FO.TINE_FACES}
    depths = [by_corners[frozenset(int(x) for x in polygon[:, 0])]
              for polygon in faces]
    assert depths == sorted(depths, reverse=True), (
        f'faces came back in depth order {depths}, which is not furthest-first')


def test_a_corner_behind_the_camera_drops_its_whole_face():
    """Half a projected face is worse than none: the visible corners still
    form a polygon, just not the one the face is."""
    corners = FO.tine_corners(floor=LEVEL, **FORKS)[0]
    in_camera = corners + np.array([0.0, 0.0, 3.0])
    pixels = np.column_stack([np.arange(8.0), np.arange(8.0)])
    visible = np.ones(8, dtype=bool)
    visible[0] = False
    faces = FO.tine_faces(in_camera, pixels, visible)
    assert len(faces) == 3, (
        'corner 0 belongs to three faces, so three should be dropped')


def test_the_tines_are_bolted_to_the_robot_not_to_the_floor_fit():
    """The forks must not move when perception's floor estimate moves.

    They are a rigid part of the machine, and the overlay is only a calibration
    reference while they stay put. This is also the case that would quietly
    defeat the whole tool: a camera orientation error tilts the fitted floor,
    and tines referenced to it would lean by exactly that error -- in the same
    direction as the pockets, which are measured against the same plane. The
    two would cancel on screen and the overlay would report that a miscalibrated
    camera is fine.
    """
    import cv2  # noqa: F401  (render needs it; skip cleanly if absent)

    image = np.zeros((360, 640, 3), dtype=np.uint8)
    intrinsics = {'fx': 450.0, 'fy': 450.0, 'cx': 320.0, 'cy': 180.0}
    params = {'floor_z': -0.025, 'forks': FORKS,
              'colors': {'idle': (160, 160, 160)}, 'alpha': 0.35,
              'line_thickness': 2, 'draw_pockets': False}
    # camera_optical <- base_footprint, roughly where the ZED sits
    base_to_camera = np.array([[0.0, -1.0, 0.0, 0.0],
                               [0.0, 0.0, -1.0, 0.398],
                               [1.0, 0.0, 0.0, -0.390],
                               [0.0, 0.0, 0.0, 1.0]])

    level = Plane(np.array([0.0, 0.0, 1.0]), 0.025)
    tilted = Plane(np.array([-np.sin(np.radians(3.0)), 0.0,
                             np.cos(np.radians(3.0))]), 0.060)

    a = FO.render(image, None, 'idle', base_to_camera, intrinsics, params, level)
    b = FO.render(image, None, 'idle', base_to_camera, intrinsics, params, tilted)
    assert np.array_equal(a, b), (
        'the drawn tines changed when the fitted floor changed; they must be '
        'built from params["floor_z"], which is a constant of the robot')


def test_changing_floor_z_does_move_the_tines():
    """The counterpart: `floor_z` is the one thing that may move them, because
    it says where the robot's own ground frame sits relative to the real floor.
    If this passed trivially, the test above would prove nothing."""
    import cv2  # noqa: F401

    image = np.zeros((360, 640, 3), dtype=np.uint8)
    intrinsics = {'fx': 450.0, 'fy': 450.0, 'cx': 320.0, 'cy': 180.0}
    base_to_camera = np.array([[0.0, -1.0, 0.0, 0.0],
                               [0.0, 0.0, -1.0, 0.398],
                               [1.0, 0.0, 0.0, -0.390],
                               [0.0, 0.0, 0.0, 1.0]])
    level = Plane(np.array([0.0, 0.0, 1.0]), 0.025)
    base = {'forks': FORKS, 'colors': {'idle': (160, 160, 160)}, 'alpha': 0.35,
            'line_thickness': 2, 'draw_pockets': False}
    a = FO.render(image, None, 'idle', base_to_camera, intrinsics,
                  dict(base, floor_z=-0.025), level)
    b = FO.render(image, None, 'idle', base_to_camera, intrinsics,
                  dict(base, floor_z=-0.075), level)
    assert not np.array_equal(a, b)


def test_drawing_at_half_size_puts_the_forks_on_exactly_half_the_pixels():
    """The overlay now shrinks the frame first and draws small, with fx, fy,
    cx, cy scaled by the same factor (2026-10-22, for a 25 fps dashboard).
    This is the factor-of-two error the old full-resolution render guarded
    against: the tines must land on exactly half the full-size pixels."""
    rotation = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)
    base_to_camera = np.eye(4)
    base_to_camera[:3, :3] = rotation
    base_to_camera[:3, 3] = -rotation @ np.array([0.39, 0.06, 0.373])
    corners = FO.tine_corners(1.15, 0.1, 0.04, 0.365, 0.05, 0.4625,
                              Plane(np.array([0.0, 0.0, 1.0]), 0.025))
    in_camera = FO.to_camera_frame(corners, base_to_camera)
    full, _ = FO.project_points(in_camera, 949.67, 949.67, 640.0, 360.0)
    half, _ = FO.project_points(in_camera, 949.67 / 2, 949.67 / 2, 320.0, 180.0)
    assert np.allclose(half, full / 2)


def test_the_text_panel_scales_with_the_drawing():
    import cv2  # noqa: F401
    rotation = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)
    base_to_camera = np.eye(4)
    base_to_camera[:3, :3] = rotation
    base_to_camera[:3, 3] = -rotation @ np.array([0.39, 0.06, 0.373])
    params = {'forks': dict(tine_length=1.15, tine_width=0.1, tine_thickness=0.04,
                            spacing=0.365, bottom_height=0.05, carriage_x=0.4625),
              'floor_z': -0.025}
    floor = Plane(np.array([0.0, 0.0, 1.0]), 0.025)
    small = np.zeros((360, 640, 3), np.uint8)
    out = FO.render(small, None, 'idle', base_to_camera,
                    dict(fx=474.8, fy=474.8, cx=320.0, cy=180.0),
                    dict(params, text_scale=0.5, line_thickness=1), floor)
    assert out.shape == small.shape      # draws at the small size without error
