"""Composite the fork overlay with a text HUD: everything about a run, on one frame.

Pure drawing, like `fork_overlay.py`: this file only knows how to turn a
`Snapshot` into pixels. `nodes/dashboard_node.py` is the only thing that reads
a clock or a topic -- it collects the latest message from each of five topics
into a `Snapshot` and calls `render` once a frame.

Why a second file rather than growing `fork_overlay.py`: that file's whole job
is one specific calibration check (do the virtual tines line up with the real
pockets), and it already draws a small panel with state, side and pocket
count. This file is the opposite kind of overlay -- everything else about a
run that is NOT face geometry: health, the tracker's confidence and LiDAR
cross-check, what the controller is doing and measuring, the forks, and why.
Keeping the two apart means a change to one cannot silently shift the other's
pixels, and it mirrors the project's own split between `perception.py` (what
is there) and `go_nogo.py` (what to do about it). `fork_overlay_node.py` is
the one piece of shared state: it and this file use the same
`STATE_COLOR_KEYS` mapping and the same `colors` palette from params.yaml, so
a state never has two different colours depending on which overlay you are
looking at.

**Every value is shown as of when it arrived, never assumed current.** A
field whose source has never published shows '--' in grey; a field older than
its `stale_after` entry is shown dimmed with its age attached. This project
keeps reintroducing the same bug -- a reasonable default, chosen silently,
never reported (docs/decisions.md calls it out by name) -- and a dashboard is
exactly where that bug is most tempting, because a frozen number still LOOKS
like a number. So `Snapshot` tracks *when* each source last arrived, not just
what it last said, the same way `health.py` tracks age rather than a value.

**The output frame size is fixed, never data-dependent.** Every widget is
exactly a title line plus two value lines; a reason too long to fit is
truncated with an ellipsis rather than growing the panel. `frame_size` compute
the total canvas size from `params` alone, with no image content involved, so
a launch file that declares the video stream's resolution can match it exactly
-- `test_dashboard.py` checks that the checked-in default actually does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    'STATE_COLOR_KEYS', 'TRACKER_STATUS_COLORS', 'CoordinatorInfo',
    'HealthInfo', 'TrackedInfo', 'ControllerInfo', 'ForksInfo', 'Snapshot',
    'frame_size', 'render',
]

# Semantic colours, BGR -- distinct from the per-state identity palette in
# params.yaml (`colors.idle/.../no_go`). That palette answers "which phase is
# this"; these answer "is this particular number good, in progress, bad, or
# missing". The two happen to share amber for "transitional", which is
# intentional rather than a clash: both uses mean "something is under way".
GREY = (160, 160, 160)
WHITE = (255, 255, 255)
GREEN = (0, 200, 0)
AMBER = (0, 200, 255)
RED = (0, 0, 220)

# Coordinator state (eleven states + ABORTED) -> the key fork_overlay.colors
# uses. Moved here from fork_overlay_node.py on 2026-10-19 so this file and
# fork_overlay's own state panel can share one mapping without this module
# depending on anything that imports rclpy -- the node imports the mapping
# back under the name it used before (`from pallet_approach.dashboard import
# STATE_COLOR_KEYS as STATE_COLORS`), so moving it changed nothing it draws.
STATE_COLOR_KEYS = {
    'IDLE': 'idle',
    'CONFIRMING': 'detecting',
    'CLASSIFYING': 'detecting',
    'BACKING_OFF': 'approaching',
    'CLOSING': 'approaching',
    'CHECKING': 'approaching',
    'LOWERING': 'go',
    'INSERTING': 'inserting',
    'LIFTING': 'inserting',
    'BACKING_OUT': 'go',
    'DONE': 'idle',
    'ABORTED': 'no_go',
}

# pallet_tracker.Tracker.status() -> a semantic colour. CONFIRMED is the only
# state the coordinator may act on; CONFIRMING and SEARCHING are both
# "nothing wrong yet, just not there", which is why both are neutral rather
# than alarming; LOST and REJECTED are the two the operator should notice.
TRACKER_STATUS_COLORS = {
    'CONFIRMED': GREEN,
    'CONFIRMING': AMBER,
    'SEARCHING': GREY,
    'LOST': AMBER,
    'REJECTED': RED,
}

# Outcome -> colour. INSERTED reads as success; everything else reads as a stop.
GOOD_OUTCOMES = ('INSERTED',)
BAD_OUTCOMES = ('NO_GO', 'NO_PALLET', 'ABORTED')


# --------------------------------------------------------------- snapshot

@dataclass(frozen=True)
class CoordinatorInfo:
    """From `/coordinator/status` (CoordinatorStatus.msg)."""

    state: str = ''
    mode: str = ''
    side: str = ''
    reason: str = ''
    outcome: str = ''
    elapsed: float = float('nan')


@dataclass(frozen=True)
class HealthInfo:
    """From `/health` (Health.msg)."""

    ok: bool = False
    failures: tuple[str, ...] = ()
    stop_latched: bool = False


@dataclass(frozen=True)
class TrackedInfo:
    """From `/pallet_tracker/tracked` (TrackedPallet.msg).

    Only the fields the HUD needs that are NOT already drawn by
    `fork_overlay.render` -- side, face position and pocket count come from
    the estimate baked into the image this module composites onto, so
    repeating them here would be the same number drawn twice.
    """

    status: str = ''
    confidence: float = float('nan')
    consecutive_frames: int = 0
    total_frames: int = 0
    lidar_evaluated: bool = False
    lidar_agrees: bool = False
    lidar_offset: float = float('nan')
    lidar_points: int = 0


@dataclass(frozen=True)
class ControllerInfo:
    """From `/approach_controller/status` (DriveStatus.msg).

    Published only while the controller is ACTIVE (it is a lifecycle
    publisher): outside the four driving states there is genuinely no live
    command to report, and `controller_at` staying old is the correct way to
    say that, not a bug to work around.
    """

    mode: str = ''
    phase: str = ''
    arrived: bool = False
    blocked: bool = False
    abort: bool = False
    face_range: float = float('nan')
    lateral_offset: float = float('nan')
    heading_error: float = float('nan')
    rear_clearance: float = float('nan')


@dataclass(frozen=True)
class ForksInfo:
    """From `/virtual_forks/state` (ForkState.msg)."""

    state: str = ''
    height: float = float('nan')
    target_height: float = float('nan')
    moving: bool = False
    loaded: bool = False


@dataclass
class Snapshot:
    """The latest of each message, and WHEN each one last arrived.

    `now` is supplied by the caller at render time (never read from a clock in
    here, matching the rest of this project's pure modules), and every `*_at`
    is `None` until the first message -- the same rule `health.py` uses for
    age: a source that has never published has no age, and 0.0 is a reading,
    not a sentinel for "unset".
    """

    now: float = 0.0
    image_at: float | None = None

    coordinator: CoordinatorInfo = field(default_factory=CoordinatorInfo)
    coordinator_at: float | None = None

    health: HealthInfo = field(default_factory=HealthInfo)
    health_at: float | None = None

    tracked: TrackedInfo = field(default_factory=TrackedInfo)
    tracked_at: float | None = None

    controller: ControllerInfo = field(default_factory=ControllerInfo)
    controller_at: float | None = None

    forks: ForksInfo = field(default_factory=ForksInfo)
    forks_at: float | None = None

    def age(self, at: float | None) -> float | None:
        return None if at is None else self.now - at


# ------------------------------------------------------------------ layout
#
# Every size below is fixed by these constants and by `params`, never by how
# much text a widget happens to have this frame -- see the module docstring.
# `_LAYOUT` is computed once by `_layout` and used by both `frame_size` (so a
# caller can size a canvas, or a launch file, before any frame exists) and
# `render` (so the two can never disagree about where anything goes).

_TITLE_COLOR = (210, 210, 210)
_COLUMNS = 4
_WIDGET_ROWS = 2
_LINES_PER_WIDGET = 3          # title + two value lines, always, padded
_REASON_LINES = 2


@dataclass(frozen=True)
class _Layout:
    width: int
    panel_top: int              # y where the panel starts, in the final image
    height: int                 # total final image height
    column_width: int
    row_height: int
    reason_top: int
    line_height: int
    pad: int
    scale: float


def _layout(image_width: int, image_height: int, params: dict) -> _Layout:
    pad = int(params.get('pad', 8))
    line_height = int(params.get('line_height', 15))
    scale = float(params.get('font_scale', 0.40))
    row_height = _LINES_PER_WIDGET * line_height + pad
    panel_height = (_WIDGET_ROWS * row_height + _REASON_LINES * line_height
                    + 3 * pad)
    return _Layout(
        width=image_width, panel_top=image_height,
        height=image_height + panel_height,
        column_width=image_width // _COLUMNS, row_height=row_height,
        reason_top=image_height + _WIDGET_ROWS * row_height + 2 * pad,
        line_height=line_height, pad=pad, scale=scale)


def frame_size(image_width: int, image_height: int, params: dict) -> tuple[int, int]:
    """(width, height) of the composited frame, before any frame exists.

    Deterministic from `params` alone -- a launch file declaring the video
    stream's resolution can compute this once and match it exactly, which is
    what `test_dashboard.py` checks the checked-in default against.
    """
    layout = _layout(image_width, image_height, params)
    return layout.width, layout.height


# -------------------------------------------------------------- formatting

def _age_text(age: float | None) -> str:
    return 'never' if age is None else f'{age:.1f}s ago'


def _num(value: float, unit: str = '', digits: int = 0) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return '--'
    return f'{value:.{digits}f}{unit}'


def _mm(value: float) -> str:
    return _num(value * 1000.0 if value is not None and not math.isnan(value)
               else float('nan'), ' mm', 0)


def _stale(color, age: float | None, limit: float):
    """Fade a colour towards grey once `age` exceeds `limit`.

    `age is None` (never arrived) is handled by the caller, which has no
    value to show at all; this is for a source that HAS a value but it is
    old, which should look different from a source actively reporting red.
    """
    if age is None or age <= limit:
        return color
    return GREY


# ------------------------------------------------------------------ widgets

def _widget(title, line1, color1, line2, color2):
    """One widget: a title and exactly two value lines, every time."""
    return title, ((line1, color1), (line2, color2))


def _run_widget(info: CoordinatorInfo, age: float | None, limit: float,
               colors: dict):
    """`colors` is `params['colors']` -- the SAME per-state palette
    `fork_overlay.render` draws its own panel in, keyed through
    `STATE_COLOR_KEYS`. Looked up here rather than baked into
    `STATE_COLOR_KEYS` itself, because the palette is configuration
    (params.yaml) and the key mapping is not."""
    key = STATE_COLOR_KEYS.get(info.state, 'idle')
    state_color = tuple(int(c) for c in colors.get(key, GREY))
    return _widget(
        'RUN', f'mode {info.mode or "--"}', WHITE,
        f'state {info.state or "--"}', _stale(state_color, age, limit))


def _status_widget(info: CoordinatorInfo, age: float | None, limit: float):
    elapsed = _num(info.elapsed, 's')
    if info.outcome:
        color = (GREEN if info.outcome in GOOD_OUTCOMES else
                 RED if info.outcome in BAD_OUTCOMES else WHITE)
        line2 = f'outcome {info.outcome}'
    else:
        color = AMBER if age is not None else GREY
        line2 = 'running' if age is not None else 'no run yet'
    return _widget('STATUS', f'side {info.side or "--"}  t+{elapsed}', WHITE,
                   line2, _stale(color, age, limit))


def _health_widget(info: HealthInfo, age: float | None, limit: float):
    if age is None:
        return _widget('HEALTH', 'no data', GREY, '(watchdog not seen)', GREY)
    ok_color = _stale(GREEN if info.ok else RED, age, limit)
    line1 = 'OK' if info.ok else 'FAIL'
    if info.stop_latched:
        line1 += '  STOP LATCHED'
        ok_color = RED
    detail = info.failures[0] if info.failures else ''
    return _widget('HEALTH', line1, ok_color, detail, RED if detail else GREY)


def _tracker_widget(info: TrackedInfo, age: float | None, limit: float):
    if age is None:
        return _widget('TRACKER', 'no data', GREY, '', GREY)
    color = _stale(TRACKER_STATUS_COLORS.get(info.status, GREY), age, limit)
    line1 = f'{info.status or "--"}  conf {_num(info.confidence, "", 2)}'
    line2 = f'{info.consecutive_frames}/{info.total_frames} frames'
    return _widget('TRACKER', line1, color, line2, _stale(WHITE, age, limit))


def _lidar_widget(info: TrackedInfo, age: float | None, limit: float):
    if age is None:
        return _widget('LIDAR X-CHECK', 'no data', GREY, '', GREY)
    if not info.lidar_evaluated:
        return _widget('LIDAR X-CHECK', 'not evaluated', GREY,
                       f'{info.lidar_points} returns', GREY)
    color = _stale(GREEN if info.lidar_agrees else RED, age, limit)
    line1 = 'agrees' if info.lidar_agrees else 'DISAGREES'
    line2 = f'{_mm(info.lidar_offset)}  {info.lidar_points} pts'
    return _widget('LIDAR X-CHECK', line1, color, line2, _stale(WHITE, age, limit))


def _controller_widget(info: ControllerInfo, age: float | None, limit: float):
    if age is None:
        return _widget('CONTROLLER', 'not active', GREY,
                       '(not in a driving state)', GREY)
    if info.abort:
        flags, color = 'ABORT', RED
    elif info.arrived:
        flags, color = 'arrived', GREEN
    elif info.blocked:
        flags, color = 'blocked', AMBER
    else:
        flags, color = 'driving', AMBER
    line1 = f'{info.mode or "--"} {info.phase or ""}'.strip()
    return _widget('CONTROLLER', line1, _stale(WHITE, age, limit),
                   flags, _stale(color, age, limit))


def _geometry_widget(info: ControllerInfo, age: float | None, limit: float):
    if age is None:
        return _widget('GEOMETRY', 'no data', GREY, '', GREY)
    line1 = f'rng {_num(info.face_range, " m", 2)}  off {_mm(info.lateral_offset)}'
    # _num already turns NaN into '--', and math.degrees(nan) is nan -- so
    # there is no need to guard the conversion separately; doing so anyway
    # read as if it mattered and it did not.
    line2 = (f'yaw {_num(math.degrees(info.heading_error), "deg", 1)}'
             f'  rear {_num(info.rear_clearance, " m", 2)}')
    return _widget('GEOMETRY', line1, _stale(WHITE, age, limit),
                   line2, _stale(WHITE, age, limit))


def _forks_widget(info: ForksInfo, age: float | None, limit: float):
    if age is None:
        return _widget('FORKS', 'no data', GREY, '', GREY)
    arrow = ' -> ' + _mm(info.target_height) if info.moving else ''
    line1 = f'{info.state or "--"}  {_mm(info.height)}{arrow}'
    line2 = 'LOADED' if info.loaded else 'empty'
    color1 = AMBER if info.moving else WHITE
    color2 = GREEN if info.loaded else GREY
    return _widget('FORKS', line1, _stale(color1, age, limit),
                   line2, _stale(color2, age, limit))


# -------------------------------------------------------------- rendering

def _wrap(text: str, max_width: int, font, scale: float, cv2_module) -> list[str]:
    """Greedy word-wrap to fit `max_width` pixels, measured with `cv2_module`."""
    if not text:
        return ['']
    words = text.split()
    lines: list[str] = []
    current = ''
    for word in words:
        trial = f'{current} {word}'.strip()
        width = cv2_module.getTextSize(trial, font, scale, 1)[0][0]
        if width > max_width and current:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines or ['']


def render(image: np.ndarray | None, snapshot: Snapshot, params: dict) -> np.ndarray:
    """Build the composited frame: the base image, plus the HUD panel below it.

    `image` is normally `fork_overlay.render`'s own output -- the camera frame
    with the tines and pockets already drawn -- but may be `None`, which draws
    a placeholder instead of pretending there is a live picture. The
    placeholder's size is `params['placeholder_size']` so the output frame is
    the same size whether or not a camera frame has ever arrived.

    `params` is the `dashboard` block of params.yaml.
    """
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    base_width, base_height = params.get('placeholder_size', (640, 360))
    if image is None:
        base = np.full((base_height, base_width, 3), 40, dtype=np.uint8)
        text = 'NO CAMERA FEED (waiting for fork_overlay)'
        size = cv2.getTextSize(text, font, 0.6, 1)[0]
        cv2.putText(base, text, ((base_width - size[0]) // 2, base_height // 2),
                   font, 0.6, GREY, 1, cv2.LINE_AA)
    else:
        base = image
        base_height, base_width = image.shape[0], image.shape[1]
        image_age = snapshot.age(snapshot.image_at)
        if image_age is not None and image_age > float(params.get(
                'stale_after', {}).get('image', 1.0)):
            cv2.rectangle(base, (0, 0), (base_width - 1, base_height - 1),
                         RED, 3)
            tag = f'IMAGE {image_age:.1f}s OLD'
            cv2.putText(base, tag, (base_width - 180, base_height - 10),
                       font, 0.45, RED, 1, cv2.LINE_AA)

    layout = _layout(base_width, base_height, params)
    canvas = np.zeros((layout.height, layout.width, 3), dtype=np.uint8)
    canvas[:base_height, :base_width] = base
    cv2.rectangle(canvas, (0, layout.panel_top), (layout.width, layout.height),
                 (25, 25, 25), -1)

    stale_after = params.get('stale_after', {})
    now = snapshot.now
    ages = {
        'coordinator': snapshot.age(snapshot.coordinator_at),
        'health': snapshot.age(snapshot.health_at),
        'tracked': snapshot.age(snapshot.tracked_at),
        'controller': snapshot.age(snapshot.controller_at),
        'forks': snapshot.age(snapshot.forks_at),
    }
    colors = params.get('colors', {})
    widgets = [
        _run_widget(snapshot.coordinator, ages['coordinator'],
                   stale_after.get('coordinator', 1.0), colors),
        _status_widget(snapshot.coordinator, ages['coordinator'],
                       stale_after.get('coordinator', 1.0)),
        _health_widget(snapshot.health, ages['health'],
                       stale_after.get('health', 1.5)),
        _tracker_widget(snapshot.tracked, ages['tracked'],
                        stale_after.get('tracked', 1.0)),
        _lidar_widget(snapshot.tracked, ages['tracked'],
                     stale_after.get('tracked', 1.0)),
        _controller_widget(snapshot.controller, ages['controller'],
                           stale_after.get('controller', 0.5)),
        _geometry_widget(snapshot.controller, ages['controller'],
                         stale_after.get('controller', 0.5)),
        _forks_widget(snapshot.forks, ages['forks'],
                     stale_after.get('forks', 1.0)),
    ]

    pad, lh, scale = layout.pad, layout.line_height, layout.scale
    for index, (title, lines) in enumerate(widgets):
        row, col = divmod(index, _COLUMNS)
        x = col * layout.column_width + pad
        y = layout.panel_top + row * layout.row_height + pad + lh
        cv2.putText(canvas, title, (x, y), font, scale, _TITLE_COLOR, 1,
                   cv2.LINE_AA)
        for line_index, (text, color) in enumerate(lines):
            cv2.putText(canvas, text, (x, y + (line_index + 1) * lh), font,
                       scale, color, 1, cv2.LINE_AA)

    reason = snapshot.coordinator.reason
    reason_color = GREY if ages['coordinator'] is None else WHITE
    max_width = layout.width - 2 * pad
    wrapped = _wrap(reason or '(no reason reported)', max_width, font, scale, cv2)
    for line_index in range(_REASON_LINES):
        if line_index >= len(wrapped):
            break
        text = wrapped[line_index]
        if line_index == _REASON_LINES - 1 and len(wrapped) > _REASON_LINES:
            text = text.rstrip() + '...'
        cv2.putText(canvas, text, (pad, layout.reason_top + line_index * lh),
                   font, scale, reason_color, 1, cv2.LINE_AA)

    return canvas
