#!/usr/bin/env python3
"""Run the perception pipeline at wall-clock speed against a recorded bag.

`replay.py` answers "what did the pipeline do to this frame", one frame at a
time, as fast as the laptop can. This answers a different question: **what
would the node do on the robot, right now, at the rate the camera actually
produces frames.** It replays the bag's depth frames on their own timestamps,
puts them through the same `FrameGate` the node uses, and prints what the node
would publish -- including the frames it drops and how old the estimate is by
the time it exists.

That is the part you cannot see from a static replay, and it is where the real
failures live: a pipeline that has quietly fallen to 1 Hz still produces
estimates that look perfectly reasonable one at a time.

    python3 pallet_approach/tools/live.py ~/bags/pallet_s_129 --view --loop
    python3 pallet_approach/tools/live.py ~/bags/pallet_s_129
    python3 pallet_approach/tools/live.py ~/bags/pallet_l_129 --rate 2 --speed 0.5

`--view` opens a window with the camera image and the detected pockets drawn
back onto it, beside the same estimate seen from above. That projection is the
honest test of the geometry: a pocket measured correctly in metres lands on the
hole in the picture, and one measured wrongly does not. A table of numbers
cannot show you that, and neither can a plot of the point cloud -- only putting
the answer back where the question came from.

No ROS: it reads the bag with `rosbags` and calls `estimate_pallet` directly,
with the parameters and the extrinsic correction from `config/params.yaml`, so
the geometry is exactly what the robot would compute. What it does NOT model is
the Jetson's speed -- it runs at your laptop's, which on the measured numbers is
roughly 2-3x faster. Treat the frame time it prints as a lower bound.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PACKAGE = HERE.parent
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(HERE))

import _bagutil as bag                                            # noqa: E402
from pallet_approach import perception as P                       # noqa: E402
from pallet_approach.calibration import correct_transform, correction_matrix  # noqa: E402
from pallet_approach import fork_overlay                          # noqa: E402
from pallet_approach.rate_limit import FrameGate                  # noqa: E402

try:
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
except ImportError:                                               # pragma: no cover
    sys.exit('pip install rosbags numpy pyyaml  (in a venv)')

DEPTH_HINT = 'depth'
CLEAR = '\033[2J\033[H'
DIM, BOLD, RESET = '\033[2m', '\033[1m', '\033[0m'


CHANNELS = {'bgr8': 3, 'rgb8': 3, 'bgra8': 4, 'rgba8': 4}


def decode_colour(msg):
    """Rectified colour -> (H, W, 3) RGB. The ZED publishes bgra8."""
    n = CHANNELS.get(msg.encoding)
    if n is None:
        return None
    rows = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, -1)
    img = rows[:, :msg.width * n].reshape(msg.height, msg.width, n)[:, :, :3]
    return img[:, :, ::-1] if msg.encoding.startswith('bgr') else img


def read_frames(path: pathlib.Path, store: str, want_colour: bool):
    """Yield (stamp, depth, colour_or_None) in recorded order.

    Streams rather than loading the bag: a 10 s bag with colour in it is most
    of a gigabyte and only one frame is needed at a time.

    Colour is carried alongside rather than paired by timestamp. The two
    streams are throttled independently so their stamps never match exactly,
    and for a parked robot looking at a stationary pallet the most recent
    colour frame is the right one to draw on -- within a camera period, nothing
    has moved. Do NOT reuse this pairing once the robot drives.
    """
    typestore = get_typestore(getattr(Stores, store))
    latest_colour = None
    with AnyReader([path], default_typestore=typestore) as reader:
        images = [c for c in reader.connections if c.msgtype.endswith('msg/Image')]
        depth_conns = [c for c in images if DEPTH_HINT in c.topic]
        colour_conns = [c for c in images
                        if want_colour and 'rect/image' in c.topic]
        if not depth_conns:
            sys.exit(f'{path.name}: no depth image topic')
        for conn, _, raw in reader.messages(connections=depth_conns + colour_conns):
            msg = reader.deserialize(raw, conn.msgtype)
            if conn in colour_conns:
                decoded = decode_colour(msg)
                if decoded is not None:
                    latest_colour = decoded
                continue
            if msg.encoding != '32FC1':
                continue
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            yield (stamp,
                   np.frombuffer(bytes(msg.data), np.float32).reshape(
                       msg.height, msg.width),
                   latest_colour)


class LiveView:
    """A window showing what perception made of each frame, as the bag plays.

    Two panels, because they answer different questions:

      left   the camera image with the detected pockets drawn back onto it.
             This is the one that tells you the geometry is RIGHT -- a pocket
             measured correctly in metres lands on the hole in the picture, and
             one measured wrongly does not. Numbers cannot show you that.
      right  the same estimate from above, in base_footprint. This is what the
             controller will actually steer on: where the face is, which way it
             points, and where the pockets sit across it.

    Drawn from the PalletEstimate alone, not from the pipeline's internals, so
    what you see is exactly what the node would publish.
    """

    def __init__(self, bag_name, params, standoff, overlay_params=None,
                 state='detecting', save_dir=None):
        import matplotlib
        import matplotlib.pyplot as plt

        # Refuse rather than run blind. A non-interactive backend accepts every
        # drawing call, shows nothing, and prints one warning per frame -- so
        # the pipeline appears to work and the demo appears to be broken, which
        # is the wrong way round. This happened for a real reason worth
        # remembering: tools/replay.py calls matplotlib.use('Agg') at import,
        # and this file used to import it for two geometry helpers, which
        # pinned the backend before the window could exist. Those helpers now
        # live in pallet_approach.projection.
        # The list of interactive backends moved between matplotlib versions
        # (rcsetup.interactive_bk went away in 3.11), so test against the
        # headless ones instead: that set is short and has not changed.
        backend = matplotlib.get_backend()
        headless = {'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template'}
        if backend.lower() in headless and save_dir is None:
            sys.exit(
                f'--view needs an interactive matplotlib backend; this one is '
                f'"{backend}", which cannot open a window.\n'
                f'  - run it with the venv python, not a bare `python3`:\n'
                f'      venv/bin/python pallet_approach/tools/live.py ... --view\n'
                f'  - check MPLBACKEND is not pinned: echo $MPLBACKEND\n'
                f'  - or install a toolkit:  pip install PyQt5\n'
                f'Without --view the same run prints a live table instead.')

        self.plt = plt
        plt.ion()
        self.fig, (self.cam, self.top) = plt.subplots(
            1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1.45, 1]})
        self.fig.canvas.manager.set_window_title(f'pallet perception -- {bag_name}')
        self.params = params
        self.standoff = standoff
        self.overlay_params = overlay_params
        self.state = state
        self.save_dir = pathlib.Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        self.saved = 0
        self.image_artist = None
        self.noted_no_colour = False
        self.cam.set_axis_off()
        self.cam.set_title('camera, with the virtual forks and detected pockets')
        self.fallback_floor = None
        self.top.set_title('from above, in base_footprint')
        self.fig.tight_layout()

    def update(self, image, estimate, to_ground, intrinsics, status):
        import numpy as np
        for axis in (self.cam, self.top):
            for artist in list(axis.patches) + list(axis.lines) + list(axis.texts):
                artist.remove()

        # ---------------------------------------------------------- camera
        # The forks and pockets are drawn by fork_overlay.render -- the SAME
        # function the node runs on the robot, not a matplotlib reimplementation
        # of it. If the overlay is wrong here it is wrong there, which is the
        # only arrangement worth having: a viewer that draws its own version
        # can agree with the geometry while the robot's disagrees.
        if image is not None and self.overlay_params is not None:
            import cv2
            bgr = np.ascontiguousarray(image[:, :, ::-1])
            floor = estimate.floor if (estimate and estimate.floor is not None) \
                else self.fallback_floor
            annotated = fork_overlay.render(
                bgr, estimate, self.state, np.linalg.inv(to_ground), intrinsics,
                self.overlay_params, floor)
            image = annotated[:, :, ::-1]

        if image is not None:
            if self.image_artist is None:
                self.image_artist = self.cam.imshow(image)
            else:
                self.image_artist.set_data(image)
        elif not self.noted_no_colour:
            # A separate flag, not a sentinel in image_artist: a bag can start
            # without colour and get some later (the streams are throttled
            # independently), and stuffing False into a slot that later holds a
            # matplotlib artist breaks on the next frame that does have one.
            self.cam.text(0.5, 0.5, 'no colour image yet\n'
                                    '(the estimate on the right is still real)',
                          transform=self.cam.transAxes, ha='center', va='center',
                          fontsize=12, color='#757575')
            self.noted_no_colour = True
        if estimate is not None and not estimate.pockets and image is not None:
            self.cam.text(0.02, 0.04, 'no pockets measured',
                          transform=self.cam.transAxes, color='#d50000',
                          fontsize=13, fontweight='bold')

        # ------------------------------------------------------- from above
        self.top.clear()
        self.top.set_title('from above, in base_footprint')
        self.top.set_xlabel('y, left +  [m]')
        self.top.set_ylabel('x, forward  [m]')
        self.top.set_xlim(1.2, -1.2)
        self.top.set_ylim(0, 3.0)
        self.top.grid(alpha=0.3)
        self.top.plot([0], [0], marker='s', markersize=11, color='#1565c0')
        self.top.text(0.05, 0.08, 'robot', color='#1565c0', fontsize=9)
        self.top.axhline(self.standoff, color='#9e9e9e', linestyle=':', linewidth=1)
        self.top.text(1.15, self.standoff + 0.04,
                      f'standoff {self.standoff:.2f}', color='#616161', fontsize=8)

        if estimate is not None:
            side = self.params.get('sides', {}).get(estimate.side, {})
            half = side.get('face_width', 0.795) / 2
            along = np.cross([0, 0, 1.0], estimate.normal)
            along /= np.linalg.norm(along)
            a = estimate.position + half * along
            b = estimate.position - half * along
            self.top.plot([a[1], b[1]], [a[0], b[0]], color='#5d4037', linewidth=4)
            tip = estimate.position + 0.25 * estimate.normal
            self.top.plot([estimate.position[1], tip[1]],
                          [estimate.position[0], tip[0]],
                          color='#ff6f00', linewidth=2)
            for pocket in estimate.pockets:
                self.top.plot([pocket.center[1]], [pocket.center[0]], marker='o',
                              markersize=9, color='#00c853')
        self.top.text(0.02, 0.98, status, transform=self.top.transAxes,
                      va='top', ha='left', fontsize=9, family='monospace',
                      bbox=dict(boxstyle='round', facecolor='#fafafa',
                                edgecolor='#bdbdbd'))

    def save(self):
        """Write the current figure, when --save-frames is on."""
        if self.save_dir is None:
            return
        path = self.save_dir / f'frame_{self.saved:04d}.png'
        self.fig.savefig(path, dpi=90, bbox_inches='tight')
        self.saved += 1

    def pause(self, seconds):
        # With no window there is nothing to keep responsive, and pausing would
        # only make a headless capture take as long as the bag.
        if self.save_dir is not None and not self.alive_window:
            return
        self.plt.pause(max(seconds, 0.001))

    @property
    def alive_window(self):
        backend = __import__('matplotlib').get_backend().lower()
        return backend not in {'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg',
                               'template'}

    def alive(self):
        if not self.alive_window:
            return True            # headless capture runs to the end of the bag
        return self.plt.fignum_exists(self.fig.number)


def status_text(gate, estimate, age, params):
    """The same facts the terminal prints, sized for the plot."""
    import numpy as np
    lines = [f'{gate.rate(time.perf_counter()):5.2f} Hz   '
             f'frame {gate.frame_seconds * 1e3:5.1f} ms   '
             f'age {age * 1e3:4.0f} ms',
             f'dropped {gate.drop_fraction() * 100:3.0f} %']
    if estimate is None:
        lines.append('')
        lines.append('NO PALLET')
        return '\n'.join(lines)
    lines.append('')
    lines.append(f'side     {estimate.side}')
    lines.append(f'face     x {estimate.position[0]:6.3f}  '
                 f'y {estimate.position[1]:+6.3f}')
    lines.append(f'yaw      {np.degrees(estimate.yaw):+7.2f} deg')
    lines.append(f'width    {estimate.face_width:5.3f} m')
    lines.append(f'pockets  {len(estimate.pockets)}')
    if len(estimate.pockets) == 2:
        spacing = P.pocket_spacing(estimate.pockets, estimate.normal, estimate.floor)
        nominal = params.get('sides', {}).get(estimate.side, {}).get('pocket_spacing')
        if spacing is not None and nominal:
            lines.append(f'spacing  {spacing:.4f} m  '
                         f'({(spacing - nominal) * 1000:+.0f} mm)')
    reason = (estimate.side_detail or {}).get('reason', '')
    if reason:
        lines.append('')
        lines.append(reason[:38])
    return '\n'.join(lines)


def render(bag_name, gate, estimate, age, params, elapsed, published, no_pallet):
    """One screenful, overwritten in place."""
    out = [f'{BOLD}{bag_name}{RESET}   {elapsed:5.1f} s elapsed', '']
    rate = gate.rate(time.perf_counter())
    warn = '' if gate.frame_seconds <= gate.interval or not gate.interval else \
        f'  {BOLD}<- slower than the {gate.max_rate:.0f} Hz limit asks for{RESET}'
    out.append(f'  rate      {rate:5.2f} Hz        frame {gate.frame_seconds * 1e3:6.1f} ms'
               f'{warn}')
    out.append(f'  processed {gate.processed:5d}           dropped {gate.dropped:5d} '
               f'({gate.drop_fraction() * 100:.0f} %)')
    out.append(f'  published {published:5d}           no pallet {no_pallet:5d}')
    out.append(f'  estimate age {age * 1e3:6.0f} ms  {DIM}(watchdog limit 500){RESET}')
    out.append('')

    if estimate is None:
        out.append(f'  {BOLD}no pallet in this frame{RESET}')
        return '\n'.join(out)

    out.append(f'  face      x {estimate.position[0]:6.3f}  y {estimate.position[1]:+6.3f}'
               f'   yaw {np.degrees(estimate.yaw):+7.2f} deg')
    detail = estimate.side_detail or {}
    out.append(f'  side      {BOLD}{estimate.side:<10}{RESET} '
               f'width {estimate.face_width:5.3f} m   {DIM}{detail.get("reason", "")}{RESET}')
    if estimate.pockets:
        spacing = P.pocket_spacing(estimate.pockets, estimate.normal, estimate.floor)
        widths = '  '.join(f'{p.width:.3f}' for p in estimate.pockets)
        heights = '  '.join(f'{p.height:.3f}' for p in estimate.pockets)
        side = params.get('sides', {}).get(estimate.side, {})
        nominal_w = side.get('pocket_width')
        nominal_s = side.get('pocket_spacing')
        out.append(f'  pockets   {len(estimate.pockets)}          width  {widths}'
                   + (f'   {DIM}nominal {nominal_w:.3f}{RESET}' if nominal_w else ''))
        out.append(f'                       height {heights}')
        if spacing is not None and nominal_s:
            out.append(f'             spacing {spacing:.4f} m   '
                       f'{DIM}nominal {nominal_s:.3f} '
                       f'({(spacing - nominal_s) * 1000:+.0f} mm){RESET}')
    else:
        out.append(f'  pockets   {BOLD}none{RESET}')
    if estimate.floor is not None:
        tilt = np.degrees(np.arccos(np.clip(estimate.floor.normal[2], -1, 1)))
        out.append(f'  floor     tilt {tilt:4.2f} deg   z {estimate.floor_height:+.4f} m')
    return '\n'.join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bag', type=pathlib.Path)
    ap.add_argument('--rate', type=float, default=None,
                    help='processing limit in Hz (default: pallet_perception.max_rate)')
    ap.add_argument('--speed', type=float, default=1.0,
                    help='replay speed; 1.0 is real time, 0 is as fast as possible')
    ap.add_argument('--loop', action='store_true',
                    help='restart at the end, for watching a short bag')
    ap.add_argument('--no-correction', action='store_true',
                    help='skip the extrinsic correction, to see what it is worth')
    ap.add_argument('--store', default='ROS2_HUMBLE')
    ap.add_argument('--quiet', action='store_true',
                    help='summary only, no live display (for scripting)')
    ap.add_argument('--view', action='store_true',
                    help='open a window: the camera with the virtual forks and '
                         'detected pockets drawn on it, and the estimate from above')
    ap.add_argument('--save-frames', metavar='DIR', default=None,
                    help='write every processed frame to DIR as a PNG instead '
                         'of needing a window. For grabbing stills, and the '
                         'only way to run --view on a headless machine')
    ap.add_argument('--state', default='detecting',
                    choices=('idle', 'detecting', 'approaching', 'go', 'no_go'),
                    help='which colour the overlay draws in. Defaults to '
                         'detecting: this tool has no state machine, and GO is '
                         'a decision go_nogo.py has not been written to make, '
                         'so claiming it here would be a picture of a result '
                         'nothing produced')
    args = ap.parse_args()

    params = bag.load_params(PACKAGE)
    if args.rate is None:
        import yaml
        args.rate = yaml.safe_load((PACKAGE / 'config' / 'params.yaml').read_text()
                                   )['pallet_perception']['ros__parameters']['max_rate']

    # tf and intrinsics are static for a parked robot, so take them once.
    meta = bag.read(args.bag, store=args.store, max_frames=1)
    if meta['intrinsics'] is None:
        sys.exit(f'{args.bag.name}: no depth camera_info, so no intrinsics')
    to_ground = meta['tree'].lookup(bag.GROUND, meta['depth_frame'] or bag.CAMERA)
    if not args.no_correction:
        ec = params['extrinsic_correction']
        if ec.get('enabled', True):
            to_ground = correct_transform(to_ground, correction_matrix(
                ec['measured_floor_pitch'], ec['measured_floor_roll']))

    view = None
    if args.view:
        import yaml
        from pallet_approach.model import Plane
        raw = yaml.safe_load((PACKAGE / 'config' / 'params.yaml').read_text())
        shared = raw['/**']['ros__parameters']
        overlay = raw['fork_overlay']['ros__parameters']
        standoff = raw['coordinator']['ros__parameters']['approach'][
            'standoff_distance']
        view = LiveView(
            args.bag.name, params, standoff,
            overlay_params={
                # Robot-fixed reference for the tines (-0.025, measured).
                'floor_z': overlay['floor_z'],
                'forks': shared['forks'],
                'sides': shared['pallet']['sides'],
                'colors': overlay['colors'],
                'alpha': overlay['alpha'],
                'line_thickness': overlay['line_thickness'],
                'draw_pockets': overlay['draw_pockets'],
            },
            state=args.state, save_dir=args.save_frames)
        # Where the forks go when perception has no floor fit -- which is the
        # normal case at the standoff, where only a 4 mm strip of floor is
        # still in frame. floor_z is measured (-0.025), not assumed.
        view.fallback_floor = Plane(np.array([0.0, 0.0, 1.0]),
                                    -overlay['floor_z'])

    gate = FrameGate(max_rate=args.rate)
    gate.reset(time.perf_counter())
    published = no_pallet = 0
    frame_times: list[float] = []
    ages: list[float] = []
    estimate = None
    sides: dict[str, int] = {}

    print(f'replaying {args.bag.name} at {args.speed or float("inf"):g}x, '
          f'processing at up to {args.rate:g} Hz. Ctrl-C to stop.')
    start_wall = time.perf_counter()
    try:
        while True:
            first_stamp = None
            epoch = time.perf_counter()
            for stamp, depth, colour in read_frames(args.bag, args.store,
                                                    want_colour=bool(view)):
                if first_stamp is None:
                    first_stamp = stamp
                # Arrive when the camera would have produced it.
                if args.speed > 0:
                    due = epoch + (stamp - first_stamp) / args.speed
                    delay = due - time.perf_counter()
                    if delay > 0:
                        # plt.pause, not sleep, when there is a window: the GUI
                        # event loop has to run or the window stops redrawing
                        # and the desktop marks it unresponsive.
                        view.pause(delay) if view else time.sleep(delay)
                    arrived = due
                else:
                    arrived = time.perf_counter()
                if view and not view.alive():
                    raise KeyboardInterrupt

                now = time.perf_counter()
                if not gate.offer(now):
                    continue
                estimate = P.estimate_pallet(depth, meta['intrinsics'], to_ground,
                                             params, stamp=stamp)
                done = time.perf_counter()
                gate.finished(done)
                frame_times.append(gate.frame_seconds)
                # Age of the scene the estimate describes, by the time it exists.
                ages.append(done - arrived)
                if estimate is None:
                    no_pallet += 1
                    sides['none'] = sides.get('none', 0) + 1
                else:
                    published += 1
                    sides[estimate.side] = sides.get(estimate.side, 0) + 1
                if view:
                    view.update(colour, estimate, to_ground, meta['intrinsics'],
                                status_text(gate, estimate, ages[-1], params))
                    view.save()
                elif not args.quiet:
                    print(CLEAR + render(args.bag.name, gate, estimate, ages[-1],
                                         params, time.perf_counter() - start_wall,
                                         published, no_pallet), flush=True)
            if not args.loop:
                break
    except KeyboardInterrupt:
        print()

    if not frame_times:
        print('no frames were processed')
        return 1
    ft = np.array(frame_times) * 1e3
    ag = np.array(ages) * 1e3
    print(f'\n{BOLD}summary{RESET}  {args.bag.name}')
    print(f'  frames processed   {gate.processed}   dropped {gate.dropped} '
          f'({gate.drop_fraction() * 100:.0f} %)')
    print(f'  estimates          {published} with a pallet, {no_pallet} without')
    print(f'  sides              ' + ', '.join(f'{k} {v}' for k, v in sorted(sides.items())))
    print(f'  frame time ms      mean {ft.mean():.1f}  p50 {np.percentile(ft, 50):.1f}  '
          f'p95 {np.percentile(ft, 95):.1f}  max {ft.max():.1f}')
    print(f'  estimate age ms    mean {ag.mean():.0f}  p95 {np.percentile(ag, 95):.0f}  '
          f'max {ag.max():.0f}   {DIM}(watchdog limit 500){RESET}')
    budget = 1000.0 / args.rate if args.rate > 0 else float('inf')
    over = (ft > budget).mean() * 100
    print(f'  over the {budget:.0f} ms budget in {over:.0f} % of frames')
    if args.speed != 1.0:
        print(f'  {DIM}estimate age is only meaningful at --speed 1; this run '
              f'was {args.speed:g}x, so frames arrived faster than the camera '
              f'produces them{RESET}')
    elif ag.max() > 500:
        print(f'  {BOLD}NOTE{RESET} the estimate exceeded the 0.50 s staleness limit '
              f'(D9): the watchdog would have stopped the robot')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
