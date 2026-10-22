"""Split the Fuel forklift mesh into parts, place it on the SCOUT's frames, and
write the meshes and geometry the sim's URDF is built from.

    venv/bin/python sim/tools/build_forklift.py

Run once, on the laptop (needs `trimesh`, which is not in any robot image), and
commit the output. Input: sim/pallet_approach_sim/urdf/forklift/*_base_visual.glb
(the OpenRobotics "Forklift" from Gazebo Fuel, CC BY 4.0, Cole Biesemeyer).
Output: sim/pallet_approach_sim/meshes/forklift/*.obj (+ .mtl, texture) and
sim/pallet_approach_sim/config/forklift_geometry.json.

What it does, and the decision behind each step:

* Splits the single mesh into body, carriage (with the forks), and four wheels
  (D10: the model ships as one mesh whose parts are separate pieces inside).
  The mesh is in many small disconnected pieces, so parts are chosen by WHERE
  each piece is, not by connectivity alone: wheels are the pieces inside a
  wheel's cylinder, the carriage is everything in front of the mast, the forks
  are the carriage pieces inside a tine's footprint.
* Moves the two tines inward to 0.365 m between centres (D11; the model has
  0.526 m, where each fork would overlap an outer block by ~3 cm).
* Places it so the stack's frames mean what they mean on the SCOUT
  (Athreya, 2026-10-22): the fork heel 0.4625 m ahead of base_footprint (where
  the virtual forks start, forks.carriage_x), base_footprint 25 mm above the
  floor (floor_z: -0.025), base_link 0.21 m above base_footprint. The model
  faces +y; the robot faces +x.
* Exports OBJ: Gazebo Fortress loads Collada, OBJ and STL, not glTF. And
  WITHOUT texture coordinates: Fortress's Ogre 2.2 aborts importing an OBJ
  that has them ("Cannot lock this buffer, it is already locked!", measured
  2026-10-22 on the GPU machine, every part, both the window and the sensors).
  So the texture's colour under each face is sampled and reduced to a small
  palette, and each part is written with plain coloured materials instead.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import trimesh

HERE = pathlib.Path(__file__).resolve().parent
PACKAGE = HERE.parent / 'pallet_approach_sim'
SOURCE = next((PACKAGE / 'urdf' / 'forklift').glob('*_base_visual.glb'))
OUT = PACKAGE / 'meshes' / 'forklift'
GEOMETRY = PACKAGE / 'config' / 'forklift_geometry.json'

FORK_SPACING = 0.365       # D11, between tine centres
PALETTE_SIZE = 8           # colours the texture is reduced to
HEEL_X = 0.4625            # forks.carriage_x: the heel, ahead of base_footprint
FLOOR_Z = -0.025           # the floor in base_footprint (measured on the SCOUT)
BASE_LINK_Z = 0.21         # base_link above base_footprint, as on the SCOUT

# Model frame (+y forward, +z up), read off the mesh (see the module notes).
WHEELS = {                 # name: (axle y, axle z, radius, side sign of x)
    'front_left': (0.685, 0.285, 0.2847, -1),
    'front_right': (0.685, 0.285, 0.2847, +1),
    'rear_left': (-0.557, 0.2225, 0.2382, -1),
    'rear_right': (-0.557, 0.2225, 0.2382, +1),
}
CARRIAGE_FRONT_OF_Y = 1.0     # the carriage plate starts at y 1.012
TINE_X = (0.19, 0.34)         # |x| band of a tine (centre 0.2635, 0.127 wide)


def main():
    scene = trimesh.load(SOURCE)
    mesh = scene.to_geometry()
    labels = trimesh.graph.connected_component_labels(
        mesh.face_adjacency, node_count=len(mesh.faces))
    pieces = [np.where(labels == k)[0] for k in range(labels.max() + 1)]

    def bounds(faces):
        v = mesh.vertices[np.unique(mesh.faces[faces])]
        return v.min(0), v.max(0)

    groups = {name: [] for name in ['body', 'carriage', 'tine_left', 'tine_right', *WHEELS]}
    for faces in pieces:
        lo, hi = bounds(faces)
        centre = (lo + hi) / 2
        wheel = _wheel_of(lo, hi)
        if wheel:
            groups[wheel].append(faces)
        elif centre[1] > CARRIAGE_FRONT_OF_Y:
            in_tine = (TINE_X[0] <= min(abs(lo[0]), abs(hi[0]))
                       and max(abs(lo[0]), abs(hi[0])) <= TINE_X[1]
                       and np.sign(lo[0]) == np.sign(hi[0]))
            if in_tine:
                groups['tine_left' if centre[0] < 0 else 'tine_right'].append(faces)
            else:
                groups['carriage'].append(faces)
        else:
            groups['body'].append(faces)

    # Geometry of a tine, in the model frame, before moving it.
    tine = mesh.vertices[np.unique(mesh.faces[np.concatenate(groups['tine_right'])])]
    heel_y = float(tine[tine[:, 2] > 0.25][:, 1].max())         # shank front face
    blade = tine[tine[:, 1] > heel_y + 0.05]
    tine_bottom, tine_top = float(blade[:, 2].min()), float(blade[:, 2].max())
    tip_y = float(tine[:, 1].max())
    tine_centre = float((tine[:, 0].min() + tine[:, 0].max()) / 2)
    tine_width = float(tine[:, 0].max() - tine[:, 0].min())

    def to_base_footprint(v):
        """Model (x right, y forward, z up) -> base_footprint (x fwd, y left)."""
        out = np.empty_like(v)
        out[:, 0] = v[:, 1] - heel_y + HEEL_X
        out[:, 1] = -v[:, 0]
        out[:, 2] = v[:, 2] + FLOOR_Z
        return out

    OUT.mkdir(parents=True, exist_ok=True)
    geometry = {'source': SOURCE.name, 'fork_spacing': FORK_SPACING,
                'heel_x': HEEL_X, 'floor_z': FLOOR_Z, 'base_link_z': BASE_LINK_Z}
    shift = tine_centre - FORK_SPACING / 2

    face_colours = _face_colours(mesh)
    palette, colour_of_face = _palette(face_colours, PALETTE_SIZE)
    _write_mtl(OUT / 'forklift.mtl', palette)
    for stale in OUT.glob('*'):
        if stale.suffix in ('.png', '.mtl') and stale.name != 'forklift.mtl':
            stale.unlink()

    for name, face_groups in groups.items():
        faces = np.concatenate(face_groups)
        part = mesh.submesh([faces], append=True)
        part_colours = colour_of_face[faces]
        v = part.vertices.copy()
        if name == 'tine_right':
            v[:, 0] -= shift
        elif name == 'tine_left':
            v[:, 0] += shift
        v = to_base_footprint(v)
        if name in WHEELS:
            y, z, radius, side = WHEELS[name]
            centre = to_base_footprint(np.array([[side * 0.465, y, z]]))[0]
            centre[1] = float(v[:, 1].mean())
            v -= centre
            geometry[name] = {'xyz': centre.round(4).tolist(), 'radius': radius}
        else:
            v[:, 2] -= BASE_LINK_Z          # body and carriage hang off base_link
        part.vertices = v
        _write_obj(OUT / f'{name}.obj', part, part_colours)
        print(f'{name:12s} {len(faces):5d} faces')

    geometry['tine'] = {
        'centre_y': FORK_SPACING / 2,
        'width': round(tine_width, 4),
        'bottom_z': round(tine_bottom + FLOOR_Z, 4),      # in base_footprint, forks down
        'top_z': round(tine_top + FLOOR_Z, 4),
        'heel_x': HEEL_X,
        'tip_x': round(tip_y - heel_y + HEEL_X, 4),
        'bottom_above_floor': round(tine_bottom, 4),
    }
    body = to_base_footprint(mesh.vertices[np.unique(mesh.faces[np.concatenate(groups['body'])])])
    geometry['body'] = {'min': body.min(0).round(4).tolist(), 'max': body.max(0).round(4).tolist()}
    GEOMETRY.write_text(json.dumps(geometry, indent=2) + '\n')
    print(json.dumps(geometry['tine']), '\nbody', geometry['body'])


def _face_colours(mesh) -> np.ndarray:
    """The texture's RGB (0..1) under each face's UV centroid."""
    image = np.asarray(mesh.visual.material.baseColorTexture.convert('RGB'), dtype=float) / 255
    uv = mesh.visual.uv[mesh.faces].mean(axis=1)
    h, w = image.shape[:2]
    cols = np.clip((uv[:, 0] % 1.0) * (w - 1), 0, w - 1).astype(int)
    rows = np.clip((1.0 - uv[:, 1] % 1.0) * (h - 1), 0, h - 1).astype(int)
    return image[rows, cols]


def _palette(colours: np.ndarray, k: int):
    """k colours (plain k-means, deterministic start) and each face's index."""
    order = np.argsort(colours.sum(axis=1))
    centres = colours[order[np.linspace(0, len(order) - 1, k).astype(int)]].copy()
    for _ in range(30):
        index = np.argmin(((colours[:, None, :] - centres[None]) ** 2).sum(-1), axis=1)
        for c in range(k):
            if np.any(index == c):
                centres[c] = colours[index == c].mean(axis=0)
    return centres, index


def _write_mtl(path, palette):
    lines = ['# forklift palette, sampled from the Fuel model\'s texture']
    for i, (r, g, b) in enumerate(palette):
        lines += [f'newmtl c{i}', f'Ka {r:.4f} {g:.4f} {b:.4f}', f'Kd {r:.4f} {g:.4f} {b:.4f}',
                  'Ks 0.1000 0.1000 0.1000', 'Ns 10.0', '']
    path.write_text('\n'.join(lines))


def _write_obj(path, part, colours):
    """Vertices, vertex normals, faces grouped by palette colour; no UVs."""
    normals = part.vertex_normals
    lines = ['# split from the OpenRobotics Forklift (Gazebo Fuel, CC BY 4.0)',
             'mtllib forklift.mtl']
    lines += [f'v {x:.6f} {y:.6f} {z:.6f}' for x, y, z in part.vertices]
    lines += [f'vn {x:.5f} {y:.5f} {z:.5f}' for x, y, z in normals]
    for c in np.unique(colours):
        lines.append(f'usemtl c{c}')
        for a, b, d in part.faces[colours == c] + 1:
            lines.append(f'f {a}//{a} {b}//{b} {d}//{d}')
    path.write_text('\n'.join(lines) + '\n')


def _wheel_of(lo, hi):
    for name, (y, z, radius, side) in WHEELS.items():
        if np.sign(lo[0]) != side or np.sign(hi[0]) != side or min(abs(lo[0]), abs(hi[0])) < 0.33:
            continue
        corners = [(lo[1], lo[2]), (hi[1], hi[2])]
        if all(np.hypot(cy - y, cz - z) <= radius * 1.45 for cy, cz in corners):
            return name
    return None


if __name__ == '__main__':
    main()
