#!/usr/bin/env python3
"""The figures of docs/kalman_filter.md, one per step of the explanation.

    python3 tools/kalman_figures.py ~/bags/robot_fused_run5 --outdir docs/images/kalman

Replays a recorded run through the real FusedPipeline, fed as the robot's
fusion node feeds it (LiDAR at its forward points' time and bias-corrected;
depth at stamp - latency; odometry interpolated), and logs every step of the
filter: each prediction, and for each update the measurement, innovation,
innovation covariance, Mahalanobis distance, gain and the estimate before and
after. Writes kalman_1_frames.png ... kalman_8_go.png and prints the numbers
the document quotes. No ROS.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse, FancyArrowPatch  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import _bagutil as bag  # noqa: E402
from pallet_approach import fusion as F  # noqa: E402
import sqlite3  # noqa: E402

from pallet_approach.fused_pipeline import FusedPipeline, forward_time, pose_at, to_base  # noqa: E402
from pallet_approach.pointcloud import transform_points, unpack_cloud  # noqa: E402
from pallet_approach.range_error import correct_lidar  # noqa: E402
from run_report import Bag, stamp as hstamp  # noqa: E402

STEPS: list[dict] = []          # every _update, initialise and check_camera, in order


def instrument() -> None:
    update = F.PalletFilter._update

    def logged_update(self, z, h, r, gate, angle_row=None):
        x0, p0 = self.x.copy(), self.P.copy()
        y = z - h @ x0
        if angle_row is not None:
            y[angle_row] = F.wrap(y[angle_row])
        s = h @ p0 @ h.T + r
        d2 = float(y @ np.linalg.solve(s, y))
        k = p0 @ h.T @ np.linalg.inv(s)
        ok = update(self, z, h, r, gate, angle_row)
        STEPS.append(dict(kind='lidar' if h.shape[0] == 3 else 'camera', t=self.stamp, z=np.array(z, float),
                          h=h.copy(), r=r.copy(), y=y, s=s, d2=d2, gate=gate, k=k, ok=bool(ok),
                          x0=x0, p0=p0, x=self.x.copy(), p=self.P.copy()))
        return ok
    F.PalletFilter._update = logged_update

    init = F.PalletFilter.initialise

    def logged_init(self, centre, yaw, stamp, *a, **kw):
        init(self, centre, yaw, stamp, *a, **kw)
        STEPS.append(dict(kind='init', t=stamp, x=self.x.copy(), p=self.P.copy()))
    F.PalletFilter.initialise = logged_init

    check = F.PalletFilter.check_camera

    def logged_check(self, range_residual, yaw_residual):
        out = check(self, range_residual, yaw_residual)
        STEPS.append(dict(kind='check', t=self.stamp, range_residual=range_residual,
                          yaw_residual=yaw_residual, run=self.disagree_run))
        return out
    F.PalletFilter.check_camera = logged_check


# ----------------------------------------------------------------------------- replay
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


# ----------------------------------------------------------------------------- geometry
def axes(yaw):
    """n: the face's outward normal (range); t: along the face (lateral)."""
    return np.array([math.cos(yaw), math.sin(yaw)]), np.array([math.sin(yaw), -math.cos(yaw)])


class Face:
    """Odom -> the face's own axes, in mm, relative to a reference estimate."""

    def __init__(self, ref):
        self.ref = np.asarray(ref, float)
        self.n, self.t = axes(self.ref[2])

    def __call__(self, p):
        d = np.asarray(p[:2], float) - self.ref[:2]
        return 1000 * float(d @ self.t), 1000 * float(d @ self.n)

    def cov(self, p):
        """A 2x2 odom covariance in the face's axes, mm^2 (lateral, range)."""
        rot = np.stack([self.t, self.n])
        return 1e6 * rot @ p[:2, :2] @ rot.T


def camera_point(e):
    """A camera update's measurement as an odom point: on its line t0 . p = z,
    at the prior's range. (z is along the face direction of that moment.)"""
    n0, t0 = axes(e['x0'][2])
    return float(e['z'][0]) * t0 + float(n0 @ e['x0'][:2]) * n0


def ellipse(ax, centre, cov, **kw):
    w, v = np.linalg.eigh(cov)
    angle = math.degrees(math.atan2(v[1, 1], v[0, 1]))
    ax.add_patch(Ellipse(centre, 2 * math.sqrt(w[1]), 2 * math.sqrt(w[0]), angle=angle, fill=False, **kw))


# ----------------------------------------------------------------------------- the run
class Run:
    def __init__(self, path: pathlib.Path):
        self.shared = bag.shared_params()
        instrument()
        self.pipe, self.odom, self.bag = replay(path, self.shared)
        self.name = path.name
        states, last = [], None
        for tm, m in self.bag.read('/coordinator/state'):
            if m.data != last:
                states.append((tm, m.data))
                last = m.data
        self.states = states
        self.t_move = next(tm for tm, s in states if s == 'CLOSING')
        self.t_go = next(tm for tm, s in states if s == 'CHECKING')
        self.t_end = next(tm for tm, s in states if s == 'INSERTING')
        first = next(i for i, e in enumerate(STEPS) if e['kind'] == 'init')
        self.steps = [e for e in STEPS[first:] if e['t'] <= self.t_end]
        self.t0 = self.steps[0]['t']
        self.updates = [e for e in self.steps if e['kind'] in ('lidar', 'camera')]
        self.lidar = [e for e in self.updates if e['kind'] == 'lidar']
        self.camera = [e for e in self.updates if e['kind'] == 'camera']
        self.at_go = next(e for e in reversed(self.updates) if e['t'] <= self.t_go)
        self.at_still = next(e for e in reversed(self.updates) if e['t'] <= self.t_move)

    def rel(self, t):
        return t - self.t0

    def shade(self, ax, labels=True):
        palette = {'CLOSING': '#e8f1fb', 'CHECKING': '#fdf3d8', 'LOWERING': '#eeeeee'}
        hi = self.rel(self.t_end) + 0.3
        for (ta, s), (tb, _) in zip(self.states, self.states[1:] + [(self.states[-1][0] + 2, '')]):
            if s in palette and self.rel(ta) < hi:
                ax.axvspan(self.rel(ta), self.rel(tb), color=palette[s], zorder=0)
                if labels:
                    ax.text((self.rel(ta) + min(self.rel(tb), hi)) / 2, 1.0, s.lower(),
                            transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=8, color='0.35')
        ax.axvline(self.rel(self.t_go), color='C3', ls='--', lw=1)
        ax.set_xlim(-0.5, hi)


# ----------------------------------------------------------------------------- figures
def fig1_frames(run: Run, out):
    """The same LiDAR readings in the robot's frame (moving) and in odom (static)."""
    base, odom_pts, ts = [], [], []
    for e in run.lidar:
        robot = pose_at(run.odom, e['t'], 0.05)
        if robot is None or not e['ok']:
            continue
        odom_pts.append(e['z'][:2])
        base.append(to_base(robot, e['z'][:2]))
        ts.append(run.rel(e['t']))
    base, odom_pts, ts = map(np.array, (base, odom_pts, ts))
    face = Face(run.at_go['x'])
    rel = np.array([face(q) for q in odom_pts])                 # lateral, range in mm
    fig, (a, b) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    a.plot(ts, base[:, 0], '.', ms=4, color='C0')
    a.set(ylabel='m', title=f'a. The face centre in the ROBOT\'s frame (distance ahead): it moves '
                            f'{np.ptp(base[:, 0]):.2f} m as the robot drives in')
    b.plot(ts, rel[:, 1], '.', ms=4, color='C0', label='range (along n)')
    b.plot(ts, rel[:, 0], '.', ms=4, color='C2', label='lateral (along t)')
    b.set(ylabel='mm', xlabel='time since acquisition (s)', ylim=(-60, 60),
          title='b. The SAME readings in odom, relative to the estimate at GO: they stay within centimetres, '
                'so the filter can estimate one constant')
    b.legend(loc='lower left', fontsize=9)
    for ax in (a, b):
        run.shade(ax, labels=ax is a)
    fig.suptitle(f'{run.name}: why the filter works in odom', fontsize=13)
    fig.savefig(out, dpi=100, bbox_inches='tight')
    return dict(base_travel_m=float(np.ptp(base[:, 0])),
                odom_range_ptp_mm=float(np.ptp(rel[:, 1])), odom_lateral_ptp_mm=float(np.ptp(rel[:, 0])))


def fig2_state(run: Run, out):
    """The state (px, py, psi) and the face's axes n and t, drawn on the run."""
    x = run.at_go['x']
    n, t = axes(x[2])
    w = run.shared['pallet']['sides']['short']['face_width']
    fig, ax = plt.subplots(figsize=(9, 8))
    path = np.array([p for tm, p in run.odom if run.t0 <= tm <= run.t_go])
    ax.plot(path[:, 1], path[:, 0], 'k-', lw=1.5, label='robot path until GO')
    robot = path[-1]
    ax.plot(robot[1], robot[0], 'ks', ms=9, label='robot at GO')
    a, b = x[:2] - w / 2 * t, x[:2] + w / 2 * t
    ax.plot([a[1], b[1]], [a[0], b[0]], color='saddlebrown', lw=6, solid_capstyle='butt', label='pallet face')
    ax.plot(x[1], x[0], 'C3o', ms=9, label='(px, py): the pocket pair\'s centre')
    for v, col, lab in ((n, 'C0', 'n = (cos psi, sin psi): range'), (t, 'C2', 't = (sin psi, -cos psi): lateral')):
        o = x[:2] + (0.08 * n if v is t else 0.0)          # t drawn just in front of the face
        ax.add_patch(FancyArrowPatch((o[1], o[0]), (o[1] + 0.4 * v[1], o[0] + 0.4 * v[0]),
                                     arrowstyle='-|>', mutation_scale=18, color=col, lw=2.5, zorder=5))
        ax.plot([], [], color=col, lw=2.5, label=lab)
    ax.set(xlabel='y in odom (m)', ylabel='x in odom (m)', aspect='equal',
           title=f'The state: x = (px, py, psi) = ({x[0]:.3f} m, {x[1]:+.3f} m, {math.degrees(x[2]):.1f} deg)\n'
                 'psi is the direction of n, the face\'s outward normal (towards the robot)')
    ax.set_xlim(x[1] + 0.6, x[1] - 0.6)
    ax.set_ylim(robot[0] - 0.3, x[0] + 0.3)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=9)
    fig.savefig(out, dpi=100, bbox_inches='tight')
    return dict(state=x.tolist())


def fig3_sensors(run: Run, out):
    """What each sensor measures, and how well: standing still, in the face's axes."""
    face = Face(run.at_still['x'])
    still_l = [e for e in run.lidar if e['ok'] and e['t'] <= run.t_move]
    still_c = [e for e in run.camera if e['ok'] and e['t'] <= run.t_move]
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw=dict(width_ratios=[1.6, 1]))
    pts = np.array([face(e['z']) for e in still_l])
    a.scatter(pts[:, 0], pts[:, 1], s=10, c='C0', alpha=0.6, label=f'LiDAR readings ({len(still_l)})')
    for e in still_l[::15]:
        ellipse(a, face(e['z']), face.cov(e['r']), ec='C0', lw=0.8, alpha=0.6)
    cam = [face(camera_point(e))[0] for e in still_c]
    for c in cam:
        a.axvline(c, color='C2', alpha=0.25, lw=1)
    a.plot([], [], color='C2', label=f'camera readings ({len(still_c)}): one number each, so a line')
    ellipse(a, (0, 0), face.cov(run.at_still['p']), ec='C3', lw=2.5)
    a.plot([], [], color='C3', lw=2.5, label='fused estimate, 1 sigma')
    a.set(xlabel='lateral: along t (mm)', ylabel='range: along n (mm)', xlim=(-45, 45), ylim=(-25, 25),
          aspect='equal', title='a. Standing still. LiDAR: 1-sigma ellipses, tight in range, loose sideways.\n'
                                'Camera: sideways only.')
    a.legend(loc='upper left', fontsize=9)
    r_l = still_l[-1]['r']
    sig = lambda v, m: 1000 * math.sqrt(v @ m @ v)  # noqa: E731
    rows = [('LiDAR', sig(face.n, r_l[:2, :2]), sig(face.t, r_l[:2, :2]), math.degrees(math.sqrt(r_l[2, 2]))),
            ('camera', float('nan'), 1000 * math.sqrt(still_c[-1]['r'][0, 0]), float('nan'))]
    lat_l, lat_c = pts[:, 0], np.array(cam)
    b.axis('off')
    txt = ['measurement noise R (from params.yaml)\n',
           f'{"":8s}{"range":>10s}{"lateral":>10s}{"yaw":>9s}']
    for name, r, l, y in rows:
        txt.append(f'{name:8s}{r:9.1f}mm{l:8.1f}mm{y:7.1f}deg'.replace('nanmm', ' -  ').replace('nandeg', ' -  '))
    txt += ['', 'what the readings did, standing still',
            f'LiDAR lateral: mean {lat_l.mean():+.1f} mm, sd {lat_l.std():.1f} mm',
            f'camera lateral: mean {lat_c.mean():+.1f} mm, sd {lat_c.std():.1f} mm',
            f'LiDAR range: sd {pts[:, 1].std():.1f} mm',
            '', 'fused, 1 sigma',
            f'range {sig(face.n, run.at_still["p"][:2, :2]):.1f} mm, lateral {sig(face.t, run.at_still["p"][:2, :2]):.1f} mm']
    b.text(0, 1, '\n'.join(txt), va='top', family='monospace', fontsize=10)
    fig.savefig(out, dpi=100, bbox_inches='tight')
    return dict(lidar_lat_mean=float(lat_l.mean()), lidar_lat_sd=float(lat_l.std()),
                cam_lat_mean=float(lat_c.mean()), cam_lat_sd=float(lat_c.std()),
                lidar_rng_sd=float(pts[:, 1].std()))


def fig4_rhythm(run: Run, out):
    """Predict grows P with time; each update shrinks it, along its own direction."""
    lo = run.t_move - 2.0
    hi = run.t_move - 0.5
    face = Face(run.at_still['x'])
    seq = [e for e in run.updates if lo <= e['t'] <= hi]
    fig, ax = plt.subplots(figsize=(13, 5))
    for key, col, lab in (('t', 'C2', 'lateral'), ('n', 'C0', 'range')):
        v = getattr(face, key)
        xs, ys = [], []
        for e in seq:
            xs += [run.rel(e['t']), run.rel(e['t'])]
            ys += [1000 * math.sqrt(v @ e['p0'][:2, :2] @ v), 1000 * math.sqrt(v @ e['p'][:2, :2] @ v)]
        ax.plot(xs, ys, color=col, lw=1.8, label=f'{lab} 1 sigma')
    for e in seq:
        ax.axvline(run.rel(e['t']), color='C0' if e['kind'] == 'lidar' else 'C2', alpha=0.25,
                   lw=0.8 if e['kind'] == 'lidar' else 2)
    ax.plot([], [], color='C0', alpha=0.5, label='a LiDAR update')
    ax.plot([], [], color='C2', alpha=0.5, lw=2, label='a camera update')
    ax.set(xlabel='time since acquisition (s)', ylabel='1 sigma (mm)',
           title='Predict and update, 1.5 s standing still: each vertical step down is an update, each slope up '
                 'between them is prediction (P += Q dt).\nThe camera mostly shrinks the lateral; the LiDAR the range.')
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    fig.savefig(out, dpi=100, bbox_inches='tight')
    return {}


def fig5_one_update(run: Run, out):
    """One LiDAR update and one camera update, with every number."""
    first_l = [e for e in run.lidar if e['ok']][1]       # [0] follows the initialise
    first_c = next(e for e in run.camera if e['ok'])
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    nums = {}
    for ax, e, title in ((axs[0], first_l, 'a. A LiDAR update (z = the face: px, py, psi)'),
                         (axs[1], first_c, 'b. A camera update (z = one number: the pockets\' position along t)')):
        face = Face(e['x0'])
        ellipse(ax, (0, 0), face.cov(e['p0']), ec='k', lw=1.8, ls='--')
        ax.plot([], [], 'k--', label='before (prior), 1 sigma')
        if e['kind'] == 'lidar':
            zc = face(e['z'])
            ellipse(ax, zc, face.cov(e['r']), ec='C0', lw=1.8)
            ax.plot(*zc, 'C0o')
            ax.plot([], [], 'C0', label='the measurement, 1 sigma (R)')
        else:
            zl = 1000 * (float(e['z'][0]) - e['x0'][:2] @ face.t)
            sd = 1000 * math.sqrt(e['r'][0, 0])
            ax.axvspan(zl - sd, zl + sd, color='C2', alpha=0.2)
            ax.axvline(zl, color='C2', lw=2, label='the measurement +- 1 sigma (R)')
        post = face(e['x'])
        ellipse(ax, post, face.cov(e['p']), ec='C3', lw=2.5)
        ax.plot(*post, 'C3o')
        ax.plot([], [], 'C3', lw=2.5, label='after (posterior), 1 sigma')
        lim = 1.6 * 1000 * math.sqrt(max(np.linalg.eigvalsh(e['p0'][:2, :2])))
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect='equal', xlabel='lateral (mm)', ylabel='range (mm)',
               title=title)
        ax.legend(loc='upper left', fontsize=9)
        y = e['y'] * np.array([1000, 1000, 180 / math.pi]) if e['kind'] == 'lidar' else 1000 * e['y']
        nums[e['kind']] = dict(t=run.rel(e['t']), innovation=np.round(y, 2).tolist(), d2=round(e['d2'], 2),
                               gate=e['gate'], gain=np.round(e['k'], 3).tolist(),
                               sigma_before=[round(1000 * math.sqrt(face.t @ e['p0'][:2, :2] @ face.t), 1),
                                             round(1000 * math.sqrt(face.n @ e['p0'][:2, :2] @ face.n), 1)],
                               sigma_after=[round(1000 * math.sqrt(face.t @ e['p'][:2, :2] @ face.t), 1),
                                            round(1000 * math.sqrt(face.n @ e['p'][:2, :2] @ face.n), 1)],
                               moved_mm=[round(v, 1) for v in post])
        ax.text(0.02, 0.02, f'd^2 = {e["d2"]:.2f} (gate {e["gate"]})\n'
                            f'sigma lateral {nums[e["kind"]]["sigma_before"][0]} -> {nums[e["kind"]]["sigma_after"][0]} mm\n'
                            f'sigma range   {nums[e["kind"]]["sigma_before"][1]} -> {nums[e["kind"]]["sigma_after"][1]} mm',
                transform=ax.transAxes, family='monospace', fontsize=9, va='bottom',
                bbox=dict(fc='white', ec='0.8'))
    fig.suptitle(f'{run.name}: one update of each kind, the first after acquisition', fontsize=13)
    fig.savefig(out, dpi=100, bbox_inches='tight')
    return nums


def fig6_gate(run: Run, out):
    """The gate: each measurement's Mahalanobis distance against its threshold."""
    fig, axs = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for ax, kind, col in ((axs[0], 'lidar', 'C0'), (axs[1], 'camera', 'C2')):
        es = [e for e in run.updates if e['kind'] == kind]
        t = np.array([run.rel(e['t']) for e in es])
        d2 = np.array([e['d2'] for e in es])
        ok = np.array([e['ok'] for e in es])
        ax.semilogy(t[ok], d2[ok], '.', color=col, ms=4, label='accepted')
        ax.semilogy(t[~ok], d2[~ok], 'kx', ms=7, label=f'rejected ({(~ok).sum()})')
        ax.axhline(es[0]['gate'], color='C3', lw=1.5,
                   label=f'gate {es[0]["gate"]} (chi-square, {3 if kind == "lidar" else 1} dof, 99 %)')
        run.shade(ax, labels=kind == 'lidar')
        ax.set(ylabel=f'{kind}: d^2', ylim=(1e-3, 300))
        ax.legend(loc='upper left', fontsize=9, ncol=3)
    axs[1].set_xlabel('time since acquisition (s)')
    fig.suptitle('The gate: d^2 = y^T S^-1 y, how surprising a measurement is given both uncertainties', fontsize=12)
    fig.savefig(out, dpi=100, bbox_inches='tight')
    return {k: int(sum(not e['ok'] for e in run.updates if e['kind'] == k)) for k in ('lidar', 'camera')}


def fig7_run(run: Run, out):
    """The whole approach: lateral and range, measurements and the fused estimate."""
    face = Face(run.at_go['x'])
    fig, axs = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    tl = [run.rel(e['t']) for e in run.lidar if e['ok']]
    ll, lr = np.array([face(e['z']) for e in run.lidar if e['ok']]).T
    tc = [run.rel(e['t']) for e in run.camera if e['ok']]
    cl = [face(camera_point(e))[0] for e in run.camera if e['ok']]
    tf = np.array([run.rel(e['t']) for e in run.updates])
    fl, fr = np.array([face(e['x']) for e in run.updates]).T
    sl = np.array([1000 * math.sqrt(face.t @ e['p'][:2, :2] @ face.t) for e in run.updates])
    sr = np.array([1000 * math.sqrt(face.n @ e['p'][:2, :2] @ face.n) for e in run.updates])
    a, b = axs
    a.plot(tl, ll, '.', ms=3, color='C0', alpha=0.5, label='LiDAR')
    a.plot(tc, cl, 'o', ms=4, color='C2', label='camera')
    a.plot(tf, fl, 'C3', lw=1.5, label='fused')
    a.fill_between(tf, fl - sl, fl + sl, color='C3', alpha=0.2)
    a.set(ylabel='lateral (mm)', ylim=(-45, 45))
    a.legend(loc='lower left', fontsize=9, ncol=3)
    a.set_title('a. Lateral, relative to the estimate at GO: the camera decides it', fontsize=11)
    b.plot(tl, lr, '.', ms=3, color='C0', alpha=0.5, label='LiDAR')
    b.plot(tf, fr, 'C3', lw=1.5, label='fused')
    b.fill_between(tf, fr - sr, fr + sr, color='C3', alpha=0.2)
    lo, hi = np.percentile(np.concatenate([lr, fr]), [1, 99])
    b.set(ylabel='range (mm)', ylim=(lo - 5, hi + 5), xlabel='time since acquisition (s)')
    b.legend(loc='lower left', fontsize=9)
    b.set_title('b. Range: the LiDAR decides it. The pallet cannot move, yet in odom it drifts while the robot '
                'drives (see the text)', fontsize=11)
    for ax in axs:
        run.shade(ax, labels=ax is a)
    fig.savefig(out, dpi=100, bbox_inches='tight')
    i = [j for j, e in enumerate(run.updates) if e['t'] <= run.t_move][-1]
    return dict(range_drift_mm=float(fr[i] - fr[-1]), lateral_step_mm=float(fl[i] - fl[-1]))


def fig8_go(run: Run, out):
    """Beyond the estimate: the checks GO needs."""
    checks = [e for e in run.steps if e['kind'] == 'check']
    p = run.shared['fusion']
    fig, axs = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    a, b = axs
    a.plot([run.rel(e['t']) for e in checks], [1000 * e['range_residual'] for e in checks], 'o-', ms=3,
           color='C2', lw=0.8, label='camera range minus LiDAR range')
    lim = 1000 * p['disagree_range']
    a.axhspan(-lim, lim, color='C2', alpha=0.08, label=f'agreement band +-{lim:.0f} mm (D110)')
    a.set(ylabel='mm', ylim=(-1.5 * lim, 1.5 * lim))
    a.legend(loc='lower left', fontsize=9)
    a.set_title('a. The camera\'s range is not fused, it is a witness: 5 frames in a row outside the band blocks GO',
                fontsize=11)
    grid = np.arange(run.t0, run.t_end, 0.02)
    lt = np.array([e['t'] for e in run.lidar if e['ok']])
    ct = np.array([e['t'] for e in run.camera if e['ok']])

    def age(ts):
        idx = np.searchsorted(ts, grid, side='right') - 1
        return np.where(idx >= 0, grid - ts[np.maximum(idx, 0)], np.inf)
    b.plot(grid - run.t0, age(lt), 'C0', label='time since the last accepted LiDAR reading')
    b.plot(grid - run.t0, age(ct), 'C2', label='... camera reading')
    b.axhline(p['lidar_max_age'], color='C3', lw=1.5, label=f'freshness limit {p["lidar_max_age"]} s')
    b.set(ylabel='s', ylim=(0, 0.8), xlabel='time since acquisition (s)')
    b.legend(loc='upper left', fontsize=9)
    b.set_title('b. Both sensors fresh: a gap over 0.5 s at the decision is a NO-GO ("LiDAR quiet")', fontsize=11)
    for ax in axs:
        run.shade(ax, labels=ax is a)
    fig.savefig(out, dpi=100, bbox_inches='tight')
    return dict(max_lidar_gap_s=float(np.max(np.diff(lt))), max_camera_gap_s=float(np.max(np.diff(ct))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('bag', type=pathlib.Path)
    ap.add_argument('--outdir', type=pathlib.Path, default=pathlib.Path('kalman_figures'))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    run = Run(args.bag)
    figures = (('kalman_1_frames.png', fig1_frames), ('kalman_2_state.png', fig2_state),
               ('kalman_3_sensors.png', fig3_sensors), ('kalman_4_predict_update.png', fig4_rhythm),
               ('kalman_5_one_update.png', fig5_one_update), ('kalman_6_gate.png', fig6_gate),
               ('kalman_7_run.png', fig7_run), ('kalman_8_go.png', fig8_go))
    for name, make in figures:
        print(f'{name}: {make(run, args.outdir / name)}')
    go = run.at_go
    face = Face(go['x'])
    print(f'at GO ({run.rel(run.t_go):.2f} s): sigma range {1000 * math.sqrt(face.n @ go["p"][:2, :2] @ face.n):.1f} mm, '
          f'lateral {1000 * math.sqrt(face.t @ go["p"][:2, :2] @ face.t):.1f} mm, '
          f'yaw {math.degrees(math.sqrt(go["p"][2, 2])):.2f} deg; updates: LiDAR {len(run.lidar)}, '
          f'camera {len(run.camera)}; standing still until {run.rel(run.t_move):.1f} s')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
