"""A ray-cast scene of the measured pallet for the LiDAR and camera tests (D102)."""

import math
import pathlib

import numpy as np
import yaml

CONFIG = pathlib.Path(__file__).resolve().parents[1] / 'config'
PARAMS = yaml.safe_load((CONFIG / 'params.yaml').read_text())['/**']['ros__parameters']
FLOOR_Z = PARAMS['floor_z']
LIDAR = np.array([0.173, 0.0, 0.584])          # beam origin, base_footprint (0.609 m above the floor)


def boxes(face_x, lateral, yaw_deg, side='short', load=False):
    """The pallet as axis-aligned boxes in its own frame, then placed: (lo, hi) corners.

    Pallet frame: x away from the robot (depth), y along the face, z up from
    the floor. Blocks under the face edge and the middle and back rows; a
    top deck; optionally a load on top. Measured layout (CLAUDE.md item 4).
    """
    s = PARAMS['pallet']['sides'][side]
    w, depth, h = s['face_width'], s['depth'], PARAMS['pallet']['pallet_height']
    pw, sp = s['pocket_width'], s['pocket_spacing']
    centre_block = sp - pw
    blocks_y = [(-w / 2, -sp / 2 - pw / 2), (-centre_block / 2, centre_block / 2), (sp / 2 + pw / 2, w / 2)]
    opening = s['opening_top']
    boxes = []
    for x0 in (0.0, depth / 2 - 0.05, depth - 0.10):           # front, middle, back rows of blocks
        for y0, y1 in blocks_y:
            boxes.append(((x0, y0, 0.0), (x0 + 0.10, y1, opening)))
    boxes.append(((0.0, -w / 2, opening), (depth, w / 2, h)))   # deck boards over the blocks
    if load:
        boxes.append(((0.0, -w / 2 - 0.05, h), (depth, w / 2 + 0.05, h + 0.6)))
    return boxes, (face_x, lateral, math.radians(yaw_deg))


def sweep(face_x, lateral=0.0, yaw_deg=0.0, side='short', load=False, pallet=True,
          noise=0.007, seed=0):
    """One 64 x 1024 LiDAR sweep (front 120 deg) of the scene, base frame."""
    rng = np.random.default_rng(seed)
    elev = np.radians(np.linspace(-45.2, 43.9, 64))
    az = np.radians(np.arange(-60, 60, 360 / 1024))
    e, a = np.meshgrid(elev, az)
    d = np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], -1).reshape(-1, 3)
    best = _cast(LIDAR, d, face_x, lateral, yaw_deg, side, load, pallet)
    ok = np.isfinite(best) & (best > 0.3)
    r = best[ok] + rng.normal(0, noise, ok.sum())
    return LIDAR + d[ok] * r[:, None]


def _cast(origin, dirs, face_x, lateral, yaw_deg, side, load, pallet, wall_x=6.0):
    """Range along each unit direction (base frame) to the floor, a wall, the pallet."""
    best = np.full(len(dirs), np.inf)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (FLOOR_Z - origin[2]) / dirs[:, 2]
        best = np.where((dirs[:, 2] < 0) & (t > 0), t, best)
        t = (wall_x - origin[0]) / dirs[:, 0]
        best = np.where((dirs[:, 0] > 0) & (t > 0), np.minimum(best, t), best)
    if pallet:
        bx, (fx, fy, yaw) = boxes(face_x, lateral, yaw_deg, side, load)
        c, s_ = math.cos(yaw), math.sin(yaw)
        rot = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
        o = rot.T @ (origin - np.array([fx, fy, FLOOR_Z]))
        d = dirs @ rot
        with np.errstate(divide='ignore', invalid='ignore'):
            for lo, hi in bx:
                t1, t2 = (np.array(lo) - o) / d, (np.array(hi) - o) / d
                tn = np.nanmax(np.minimum(t1, t2), axis=1)
                tf = np.nanmin(np.maximum(t1, t2), axis=1)
                best = np.where((tn <= tf) & (tn > 0), np.minimum(best, tn), best)
    return best


OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])   # optical axes in base
CAMERA = np.eye(4)
CAMERA[:3, :3], CAMERA[:3, 3] = OPTICAL, [0.39, 0.06, 0.373]
INTRINSICS = {'fx': 949.5, 'fy': 949.5, 'cx': 640.0, 'cy': 360.0}


def depth_image(face_x, lateral=0.0, yaw_deg=0.0, side='short', load=False, pallet=True,
                noise=0.0, seed=0, size=(1280, 720)):
    """The ZED's depth image (z along the optical axis) of the scene, from CAMERA."""
    rng = np.random.default_rng(seed)
    w, h = size
    v, u = np.mgrid[0:h, 0:w]
    rays = np.stack([(u - INTRINSICS['cx']) / INTRINSICS['fx'], (v - INTRINSICS['cy']) / INTRINSICS['fy'],
                     np.ones((h, w))], -1).reshape(-1, 3)
    norm = np.linalg.norm(rays, axis=1)
    dirs = (rays / norm[:, None]) @ CAMERA[:3, :3].T
    r = _cast(CAMERA[:3, 3], dirs, face_x, lateral, yaw_deg, side, load, pallet)
    z = r / norm
    z = z + rng.normal(0, noise, z.shape) * z
    z[~np.isfinite(z)] = np.nan
    return z.reshape(h, w).astype(np.float32)
