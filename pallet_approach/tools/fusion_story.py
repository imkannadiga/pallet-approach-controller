#!/usr/bin/env python3
"""Show the camera-LiDAR Kalman filter at work on one recorded run.

    python3 tools/fusion_story.py ~/bags/robot_fused_run5 --out docs/images/fusion_run5.png

Replays the bag through the real FusedPipeline, the way the robot's fusion node
feeds it (LiDAR at its forward points' time and bias-corrected, D109; depth at
stamp - latency, D107; odometry interpolated), and records every measurement the
filter receives with its estimate before and after. Everything is drawn in the
FACE's own axes, relative to the final estimate: range (along the face normal)
and lateral (along the face), in mm. The robot's own live estimate
(/pallet_fusion/tracked) is printed beside the replay's for comparison. No ROS.

The bag's depth was recorded at 3 Hz (the node ran at 15 Hz), so the camera
updates less often here than on the robot; the LiDAR is complete.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sqlite3
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import _bagutil as bag  # noqa: E402
from pallet_approach import fusion as F  # noqa: E402
from pallet_approach.fused_pipeline import FusedPipeline, forward_time, pose_at  # noqa: E402
from pallet_approach.pointcloud import transform_points, unpack_cloud  # noqa: E402
from pallet_approach.range_error import correct_lidar  # noqa: E402
from run_report import Bag, stamp as hstamp  # noqa: E402

LOG: list[dict] = []


def instrument() -> None:
    """Record every filter update: kind, time, measurement, prior and posterior."""
    def wrap(name):
        original = getattr(F.PalletFilter, name)

        def logged(self, *args, **kwargs):
            prior = (None if self.x is None else self.x.copy(), self.P.copy())
            ok = original(self, *args, **kwargs)
            LOG.append(dict(kind=name, t=args[2] if name == 'update_lidar' else args[1], z=np.array(args[0], float),
                            yaw=args[1] if name == 'update_lidar' else None, ok=bool(ok), prior=prior,
                            x=None if self.x is None else self.x.copy(), P=self.P.copy()))
            return ok
        setattr(F.PalletFilter, name, logged)
    wrap('update_lidar')
    wrap('update_camera')
    original_check = F.PalletFilter.check_camera

    def check(self, range_residual, yaw_residual):
        LOG.append(dict(kind='check', t=self.stamp, range_residual=range_residual, yaw_residual=yaw_residual))
        return original_check(self, range_residual, yaw_residual)
    F.PalletFilter.check_camera = check


def replay(path: pathlib.Path, shared: dict):
    data = bag.read(path, max_frames=1)
    k, cam_frame = data['intrinsics'], data['depth_frame'] or bag.CAMERA
    cam = bag.camera_to_ground(data, shared['frames']['extrinsic_correction'])
    b = Bag(path)
    odom = [(hstamp(m.header), (m.pose.pose.position.x, m.pose.pose.position.y,
                                2 * math.atan2(m.pose.pose.orientation.z, m.pose.pose.orientation.w)))
            for _, m in b.read('/odom')]
    odom.sort(key=lambda r: r[0])
    depth_topic = next(t for t in b.topics if 'depth_registered' in t)
    names = {'/pallet_approach/lidar': 'lidar', depth_topic: 'depth'}
    con = sqlite3.connect(next(path.glob('*.db3')))
    ids = {b.topics[n][0]: n for n in names}
    rows = con.execute(f'select topic_id, timestamp, data from messages where topic_id in '
                       f'({",".join(map(str, ids))}) order by timestamp')
    pipe = FusedPipeline(shared)
    to_ground, latency = None, float(shared['camera_depth'].get('latency', 0.0))
    for tid, _, raw in rows:
        name = ids[tid]
        msg = b.ts.deserialize_cdr(raw, b.topics[name][1])
        s = hstamp(msg.header)
        if names[name] == 'lidar':
            if to_ground is None:
                to_ground = bag.lidar_to_ground(data['tree'], msg.header.frame_id, cam_frame, shared)
            fields = {f.name for f in msg.fields}
            pts = unpack_cloud(msg, ('x', 'y', 'z', 't') if 't' in fields else ('x', 'y', 'z'))
            g = transform_points(pts[:, :3], to_ground)
            if pts.shape[1] == 4:
                s += forward_time(g, pts[:, 3] * 1e-9)
            g = correct_lidar(g, to_ground[:3, 3], shared['range_error']['lidar'])
            robot = pose_at(odom, s, 0.05)
            if robot is not None:
                pipe.on_lidar(g, robot, s)
        else:
            depth = np.frombuffer(bytes(msg.data), np.float32).reshape(msg.height, msg.width)
            measured = s - latency
            robot = pose_at(odom, measured, 0.05)
            if robot is not None:
                pipe.on_depth(depth, k, cam, robot, measured)
    return pipe, odom, b


def face_axes(yaw):
    return np.array([math.cos(yaw), math.sin(yaw)]), np.array([math.sin(yaw), -math.cos(yaw)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('bag', type=pathlib.Path)
    ap.add_argument('--out', type=pathlib.Path, default=pathlib.Path('fusion_story.png'))
    args = ap.parse_args()
    shared = bag.shared_params()
    instrument()
    pipe, odom, b = replay(args.bag, shared)
    states = []
    last = None
    for tm, m in b.read('/coordinator/state'):
        if m.data != last:
            states.append((tm, m.data))
            last = m.data
    # The fusion phase: from the first track to the start of the blind insertion,
    # which runs on odometry and the LiDAR stop, not on this filter.
    t_go = next((tm for tm, st in states if st == 'CHECKING'), None)
    t_end = next((tm for tm, st in states if st == 'INSERTING'), None)
    ups = [e for e in LOG if e['kind'] in ('update_lidar', 'update_camera') and e['x'] is not None
           and (t_end is None or e['t'] <= t_end)]
    if not ups or t_go is None:
        print('the filter never started, or the run never reached CHECKING')
        return 1
    final = next(e for e in reversed(ups) if e['t'] <= t_go)['x']       # the estimate GO was decided on
    n, t = face_axes(final[2])
    t0 = ups[0]['t']

    def face(p):                                   # odom point -> (lateral, range) mm, rel. final
        d = np.asarray(p[:2]) - final[:2]
        return 1000 * (d @ t), 1000 * (d @ n)

    live = [(hstamp(m.header), m) for _, m in b.read('/pallet_fusion/tracked') if m.estimate.valid]

    lidar = [e for e in ups if e['kind'] == 'update_lidar']
    camera = [e for e in ups if e['kind'] == 'update_camera']
    checks = [e for e in LOG if e['kind'] == 'check' and (t_end is None or e['t'] <= t_end)]

    def sig(e, v):
        return 1000 * math.sqrt(v @ e['P'][:2, :2] @ v)

    tl = np.array([e['t'] - t0 for e in lidar])
    lid_lat, lid_rng = np.array([face(e['z']) for e in lidar]).T
    # A camera measurement is one number: the pocket centre's position along the face.
    cam_t = np.array([e['t'] - t0 for e in camera])
    cam_lat = np.array([face(e['z'])[0] for e in camera])
    ok_l, ok_c = np.array([e['ok'] for e in lidar]), np.array([e['ok'] for e in camera])
    fus_t = np.array([e['t'] - t0 for e in ups])
    fus_lat, fus_rng = np.array([face(e['x']) for e in ups]).T
    fus_sl = np.array([sig(e, t) for e in ups])
    fus_sr = np.array([sig(e, n) for e in ups])
    fus_sy = np.array([math.degrees(math.sqrt(e['P'][2, 2])) for e in ups])

    fig = plt.figure(figsize=(17, 15))
    g = fig.add_gridspec(4, 3, height_ratios=[1.25, 1, 1, 0.8], hspace=0.42, wspace=0.25)
    fig.suptitle(f'{args.bag.name}: how the Kalman filter fuses camera and LiDAR (replayed on the real bag)',
                 fontsize=14, y=0.995)

    def shade(ax):
        palette = {'CLOSING': '#e8f1fb', 'CHECKING': '#fdf3d8', 'LOWERING': '#eeeeee',
                   'INSERTING': '#e6f5e6', 'LIFTING': '#f3e6f5'}
        x_hi = (t_end or ups[-1]['t']) - t0 + 0.5
        for (ta, s), (tb, _) in zip(states, states[1:] + [(states[-1][0] + 2, '')]):
            if s in palette and ta - t0 < x_hi:
                ax.axvspan(ta - t0, tb - t0, color=palette[s], zorder=0)
                mid = (max(ta - t0, -1) + min(tb - t0, x_hi)) / 2
                ax.text(mid, 1.0, s.lower(), transform=ax.get_xaxis_transform(),
                        ha='center', va='bottom', fontsize=8, color='0.35', clip_on=False)
        ax.axvline(t_go - t0, color='C3', ls='--', lw=1)
        ax.set_xlim(-1, x_hi)

    # ---- 1. the face's frame: what each sensor pins down (standing still, no odometry in it)
    t_move = next((tm for tm, st in states if st == 'CLOSING'), t_go)
    ref = next(e for e in reversed(ups) if e['t'] <= t_move)['x']
    nr, tr = face_axes(ref[2])

    def face_still(p):
        d = np.asarray(p[:2]) - ref[:2]
        return 1000 * (d @ tr), 1000 * (d @ nr)
    still_l = [e for e in lidar if e['ok'] and e['t'] <= t_move]
    still_c = [e for e in camera if e['ok'] and e['t'] <= t_move]
    i_still = max(i for i, e in enumerate(ups) if e['t'] <= t_move)
    ax = fig.add_subplot(g[0, 0:2])
    pts = np.array([face_still(e['z']) for e in still_l])
    ax.scatter(pts[:, 0], pts[:, 1], s=8, c='C0', alpha=0.5, label=f'LiDAR face readings ({len(still_l)})')
    for e in still_l[::12]:
        if e['ok']:
            la, ra = face_still(e['z'])
            ax.add_patch(Ellipse((la, ra), 2 * 1000 * shared['fusion']['lidar_sigma_lateral'],
                                 2 * 1000 * F.PalletFilter(shared).lidar_sigma_range(None), fill=False,
                                 ec='C0', lw=0.8, alpha=0.7))
    for e in still_c:
        ax.axvline(face_still(e['z'])[0], color='C2', alpha=0.25, lw=1)
    ax.plot([], [], color='C2', label=f'camera readings ({len(still_c)}): lateral only, so a line')
    i_go = max(i for i, e in enumerate(ups) if e['t'] <= t_go)
    sl, sr = fus_sl[i_still], fus_sr[i_still]
    ax.add_patch(Ellipse((0, 0), 2 * sl, 2 * sr, fill=False, ec='C3', lw=2.5, label='fused, 1 sigma'))
    ax.plot(0, 0, 'C3+', ms=12)
    ax.set(xlabel='lateral: along the face (mm)', ylabel='range: along the face normal (mm)',
           xlim=(-45, 45), ylim=(-25, 25), aspect='equal')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title(f'1. Standing still before the trigger ({t_move - t0:.0f} s), in the face\'s own axes. '
                 'LiDAR ellipses (1 sigma): tight in range, loose sideways.\nCamera: sideways only, so lines. '
                 'Fused (red): tight in both, each sensor covering the other\'s weak direction.', fontsize=10)

    ax = fig.add_subplot(g[0, 2])
    ax.semilogy(fus_t, fus_sl, 'C2', label='lateral')
    ax.semilogy(fus_t, fus_sr, 'C0', label='range')
    ax.set(xlabel='time since acquisition (s)', ylabel='1 sigma (mm)')
    ax2 = ax.twinx()
    ax2.plot(fus_t, fus_sy, 'C1', lw=1, label='yaw (deg)')
    ax2.set_ylabel('yaw 1 sigma (deg)', color='C1')
    shade(ax)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title('2. The uncertainty, acquisition to insertion\n(dashed: the GO decision)', fontsize=10)

    # ---- 3. lateral over time
    ax = fig.add_subplot(g[1, :])
    ax.plot(tl[ok_l], lid_lat[ok_l], '.', ms=3, color='C0', alpha=0.5, label='LiDAR (sigma 15 mm)')
    ax.plot(cam_t[ok_c], cam_lat[ok_c], 'o', ms=4, color='C2', label='camera (sigma 5 mm)')
    if (~ok_c).any():
        ax.plot(cam_t[~ok_c], cam_lat[~ok_c], 'x', color='k', label='camera, rejected by the gate')
    ax.plot(fus_t, fus_lat, 'C3', lw=1.5, label='fused')
    ax.fill_between(fus_t, fus_lat - fus_sl, fus_lat + fus_sl, color='C3', alpha=0.2)
    shade(ax)
    ax.set(ylabel='lateral (mm)', ylim=(-50, 50))
    ax.legend(loc='lower left', fontsize=9, ncol=4)
    ax.set_title('3. Lateral (relative to the estimate at GO): the LiDAR scatters by centimetres, the camera holds it to '
                 'millimetres. When the robot starts moving the estimate steps ~2 cm; two camera frames are gated out', fontsize=10)

    # ---- 4. range over time
    ax = fig.add_subplot(g[2, :])
    ax.plot(tl[ok_l], lid_rng[ok_l], '.', ms=3, color='C0', alpha=0.5, label='LiDAR')
    ax.plot(fus_t, fus_rng, 'C3', lw=1.5, label='fused')
    ax.fill_between(fus_t, fus_rng - fus_sr, fus_rng + fus_sr, color='C3', alpha=0.2)
    shade(ax)
    lo, hi = np.percentile(np.concatenate([lid_rng[ok_l], fus_rng]), [1, 99])
    ax.set(ylabel='range (mm)', ylim=(lo - 5, hi + 5))
    ax.legend(loc='lower left', fontsize=9)
    ax.set_title('4. Range: from the LiDAR alone. The pallet cannot move, yet in odom it drifts ~4 cm while the robot '
                 'drives 0.7 m: odometry, a range error changing with distance, or both (5 shows not odometry alone)', fontsize=10)

    # ---- 5. the camera as a witness
    ax = fig.add_subplot(g[3, :])
    if checks:
        ct = np.array([e['t'] - t0 for e in checks])
        rr = np.array([1000 * e['range_residual'] for e in checks])
        ax.plot(ct, rr, 'o-', ms=3, color='C2', lw=0.8, label='camera minus LiDAR, range')
    lim = 1000 * shared['fusion']['disagree_range']
    ax.axhspan(-lim, lim, color='C2', alpha=0.08, label=f'agreement limit +-{lim:.0f} mm (D110)')
    shade(ax)
    ax.set(xlabel='time since acquisition (s)', ylabel='mm', ylim=(-1.5 * lim, 1.5 * lim))
    ax.legend(loc='lower left', fontsize=9)
    ax.set_title('5. The camera\'s range is not fused; it is a witness: 5 frames in a row outside the band '
                 'and GO is blocked', fontsize=10)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=95, bbox_inches='tight')
    print(f'wrote {args.out}')
    print(f'LiDAR updates {len(lidar)} ({ok_l.sum()} accepted), camera {len(camera)} ({ok_c.sum()} accepted)')
    print(f'at the GO decision: sigma range {fus_sr[i_go]:.1f} mm, lateral {fus_sl[i_go]:.1f} mm, '
          f'yaw {fus_sy[i_go]:.2f} deg')
    first_cam = next((e['t'] - t0 for e in camera), float('nan'))
    print(f'first LiDAR update at 0 s, first camera update at {first_cam:.2f} s; GO at {t_go - t0:.2f} s')
    if live:
        ls, lm = min(live, key=lambda r: abs(r[0] - t_go))
        d = np.array([lm.center_odom.x, lm.center_odom.y]) - final[:2]
        print(f'robot\'s live estimate vs this replay at the GO decision: range {1000 * d @ n:+.1f} mm, '
              f'lateral {1000 * d @ t:+.1f} mm, yaw {math.degrees(F.wrap(lm.yaw_odom - final[2])):+.2f} deg')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
