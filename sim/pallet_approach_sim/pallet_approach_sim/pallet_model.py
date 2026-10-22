"""The measured pallet as Gazebo boxes, and where to put it.

Built from timber the way a EUR pallet is actually built, not as two painted
faces, so that BOTH faces come out right from one model and the depth camera
sees real gaps it can look into:

    z 0.125 .. 0.145  top deck, one plate over the whole footprint
    z 0.100 .. 0.125  three stringer boards running ACROSS (along y), one over
                      each column of blocks
    z 0.020 .. 0.100  nine blocks
    z 0.000 .. 0.020  three bottom boards running ALONG (along x), one under
                      each row of blocks

Model frame: origin on the floor at the centre of the footprint, x along the
1195 mm length, so the SHORT (795 mm, pocket) faces are the ones at x = +-0.5975.

Looking at the short face you see the bottom boards end-on under the blocks,
so the pockets run from the floor to 0.100 -- no board under them, which is the
measured pallet (docs/decisions.md, D4). Looking at the long face you see a bottom
board along the whole width and the stringers only end-on above the blocks, so
the opening runs 0.020 .. 0.125. Those are exactly the two faces
`test/synthetic.py` models; `test/test_pallet_model.py` checks the two agree by
slicing these boxes, so the simulation cannot drift from the synthetic data the
unit tests are tuned on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

LENGTH = 1.195          # along x: the long face's width
WIDTH = 0.795           # along y: the short (pocket) face's width
HEIGHT = 0.145

# Spans of solid timber, measured (cm tape, 2026-10-14) on the short face:
# block 10, pocket 23, block 13.5, pocket 23, block 10.
BLOCK_ROWS_Y = ((-0.3975, -0.2975), (-0.0675, 0.0675), (0.2975, 0.3975))
# Long face, as test/synthetic.py has it: 0.150 blocks and symmetric 0.3725
# pockets, which matches the D19 tape measurement (0.385 and 0.365, EPAL is
# symmetric by design) to within tape error. Change both together.
BLOCK_COLUMNS_X = ((-0.5975, -0.4475), (-0.075, 0.075), (0.4475, 0.5975))

BOTTOM_BOARD_TOP = 0.020
BLOCK_TOP = 0.100
STRINGER_TOP = 0.125


@dataclass(frozen=True)
class Box:
    name: str
    x: tuple
    y: tuple
    z: tuple

    @property
    def size(self):
        return tuple(b - a for a, b in (self.x, self.y, self.z))

    @property
    def center(self):
        return tuple(0.5 * (a + b) for a, b in (self.x, self.y, self.z))


def boxes() -> list[Box]:
    """Every piece of timber, in the model frame."""
    half_l, half_w = LENGTH / 2, WIDTH / 2
    parts = [Box('deck', (-half_l, half_l), (-half_w, half_w), (STRINGER_TOP, HEIGHT))]
    for i, xs in enumerate(BLOCK_COLUMNS_X):
        parts.append(Box(f'stringer_{i}', xs, (-half_w, half_w), (BLOCK_TOP, STRINGER_TOP)))
    for j, ys in enumerate(BLOCK_ROWS_Y):
        parts.append(Box(f'bottom_{j}', (-half_l, half_l), ys, (0.0, BOTTOM_BOARD_TOP)))
        for i, xs in enumerate(BLOCK_COLUMNS_X):
            parts.append(Box(f'block_{i}_{j}', xs, ys, (BOTTOM_BOARD_TOP, BLOCK_TOP)))
    return parts


def face_profile(face: str, height: float) -> list[tuple[float, float]]:
    """Solid spans along a face at one height, as a robot facing it sees them.

    `face` is 'short' (the +x end) or 'long' (the +y side). The along-face
    coordinate is model y for the short face and model x for the long one;
    both faces are symmetric about their centre, so which way it runs does not
    matter. A box counts if it reaches the face plane: the model is solid
    timber, so whatever is flush with the face is what the camera sees.
    """
    tol = 1e-9
    spans = []
    for box in boxes():
        if not box.z[0] - tol <= height <= box.z[1] + tol:
            continue
        if face == 'short' and box.x[1] >= LENGTH / 2 - tol:
            spans.append(box.y)
        elif face == 'long' and box.y[1] >= WIDTH / 2 - tol:
            spans.append(box.x)
    return _merge(spans)


def _merge(spans):
    merged = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1] + 1e-9:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


MASS = 20.0
"""kg. A wooden EUR pallet is 20-25 kg; this one was never weighed."""


def sdf(name: str = 'pallet', color=(0.62, 0.47, 0.30), static: bool = True) -> str:
    """An SDF model of the pallet.

    Static for the SCOUT, whose forks are virtual and must never touch it.
    Dynamic for the forklift (physical lift, Athreya 2026-10-22), with its mass
    at the footprint's centre, half way up.
    """
    r, g, b = color
    material = (f'<material><ambient>{r} {g} {b} 1</ambient>'
                f'<diffuse>{r} {g} {b} 1</diffuse></material>')
    links = []
    for box in boxes():
        cx, cy, cz = box.center
        sx, sy, sz = box.size
        geometry = f'<geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry>'
        links.append(
            f'<collision name="{box.name}_c"><pose>{cx:.4f} {cy:.4f} {cz:.4f} 0 0 0</pose>'
            f'{geometry}</collision>'
            f'<visual name="{box.name}_v"><pose>{cx:.4f} {cy:.4f} {cz:.4f} 0 0 0</pose>'
            f'{geometry}{material}</visual>')
    ixx = MASS * (WIDTH ** 2 + HEIGHT ** 2) / 12
    iyy = MASS * (LENGTH ** 2 + HEIGHT ** 2) / 12
    izz = MASS * (LENGTH ** 2 + WIDTH ** 2) / 12
    inertial = ('' if static else
                f'<inertial><pose>0 0 {HEIGHT / 2:.4f} 0 0 0</pose><mass>{MASS}</mass>'
                f'<inertia><ixx>{ixx:.4f}</ixx><iyy>{iyy:.4f}</iyy><izz>{izz:.4f}</izz>'
                '<ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>')
    return (f'<?xml version="1.0"?><sdf version="1.8"><model name="{name}">'
            f'<static>{"true" if static else "false"}</static>'
            f'<link name="body">{inertial}{"".join(links)}</link>'
            f'</model></sdf>')


def placement(face: str, face_x: float, face_y: float, face_yaw: float):
    """World pose (x, y, yaw) of the model so a face sits where asked.

    The robot spawns at the origin looking along +x. (face_x, face_y) is the
    centre of the requested face on the floor; face_yaw turns the face's
    outward normal away from pointing straight back at the robot
    (positive = counter-clockwise, seen from above). So face_yaw = 0 is a
    pallet squarely facing the robot.
    """
    if face not in ('short', 'long'):
        raise ValueError(f"face must be 'short' or 'long', not {face!r}")
    normal_yaw = math.pi + face_yaw            # outward normal, world frame
    if face == 'short':
        model_yaw = normal_yaw                 # model +x is the short face normal
        half_depth = LENGTH / 2
    else:
        model_yaw = normal_yaw - math.pi / 2   # model +y is the long face normal
        half_depth = WIDTH / 2
    cx = face_x - half_depth * math.cos(normal_yaw)
    cy = face_y - half_depth * math.sin(normal_yaw)
    return cx, cy, math.atan2(math.sin(model_yaw), math.cos(model_yaw))
