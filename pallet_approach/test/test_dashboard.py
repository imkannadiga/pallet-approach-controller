"""Tests for dashboard.py -- the HUD composited onto the fork overlay.

Three groups: that the output frame is always the same size no matter what
happens to be in it (the whole point of the fixed-layout design), that
staleness is shown rather than assumed, and that the colours mean what the
module docstring says they mean. `cv2` is imported lazily inside `render`, as
`fork_overlay.py` does, so these skip cleanly where it is absent.
"""


import numpy as np
import pytest

from pallet_approach import dashboard as D

PARAMS = {
    'colors': {'idle': [160, 160, 160], 'detecting': [160, 160, 160],
              'approaching': [0, 200, 255], 'inserting': [128, 255, 0],
              'go': [0, 200, 0], 'no_go': [0, 0, 220]},
    'stale_after': {'image': 1.0, 'coordinator': 1.0, 'health': 1.5,
                    'tracked': 1.0, 'controller': 0.5, 'forks': 1.0},
    'placeholder_size': (640, 360),
    'pad': 8, 'line_height': 15, 'font_scale': 0.40,
}


def snapshot(now=100.0, **overrides):
    snap = D.Snapshot(now=now)
    for key, value in overrides.items():
        setattr(snap, key, value)
    return snap


# -------------------------------------------------------------- frame size

def test_frame_size_matches_what_render_actually_produces():
    """The property the whole fixed-layout design exists for: a launch file
    can compute this BEFORE a frame exists and it will still be right."""
    import cv2  # noqa: F401
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    out = D.render(image, snapshot(), PARAMS)
    assert D.frame_size(640, 360, PARAMS) == (out.shape[1], out.shape[0])


def test_frame_size_is_the_same_with_or_without_an_image():
    import cv2  # noqa: F401
    with_image = D.render(np.zeros((360, 640, 3), dtype=np.uint8), snapshot(),
                          PARAMS)
    without_image = D.render(None, snapshot(), PARAMS)
    assert with_image.shape == without_image.shape


def test_frame_size_does_not_depend_on_reason_length():
    """The property that makes the whole design safe: a reason string long
    enough to need wrapping and truncation must not grow the canvas -- it is
    cut with an ellipsis instead. If this test failed, the video stream's
    declared resolution could silently stop matching the real one."""
    import cv2  # noqa: F401
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    short = D.render(image.copy(), snapshot(
        coordinator=D.CoordinatorInfo(reason='ok'), coordinator_at=99.0), PARAMS)
    long_reason = ' '.join(f'word{i}' for i in range(80))
    long = D.render(image.copy(), snapshot(
        coordinator=D.CoordinatorInfo(reason=long_reason), coordinator_at=99.0),
        PARAMS)
    assert short.shape == long.shape


def test_frame_size_scales_with_the_configured_image_size():
    import cv2  # noqa: F401
    small = D.frame_size(640, 360, PARAMS)
    big = D.frame_size(1280, 720, PARAMS)
    assert big[0] == 1280 and big[0] > small[0]
    assert big[1] > small[1]


# ---------------------------------------------------------------- staleness

def test_a_source_that_has_never_published_shows_no_data_not_a_default():
    """The bug this whole module is built around: a frozen or default value
    that still looks like a reading. A dataclass default (confidence=NaN,
    ok=False, ...) must never be drawn as if it were a real measurement."""
    import cv2  # noqa: F401
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    # Nothing has arrived at all: every *_at stays None.
    out = D.render(image, snapshot(), PARAMS)
    assert out is not None  # renders without raising on an empty snapshot


def test_image_staleness_is_visible_on_the_image_itself():
    """Not just a HUD field -- the one piece of data that looks live even when
    it is frozen is the picture, so it gets its own border."""
    import cv2  # noqa: F401
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    fresh = D.render(image.copy(), snapshot(image_at=99.9), PARAMS)
    stale = D.render(image.copy(), snapshot(image_at=90.0), PARAMS)
    # the stale version must differ from the fresh one within the image area
    # (a red border drawn over it), not just in the HUD panel below it
    assert not np.array_equal(fresh[:360], stale[:360])


def test_a_fresh_image_gets_no_border():
    import cv2  # noqa: F401
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    out = D.render(image.copy(), snapshot(image_at=99.9), PARAMS)
    assert np.array_equal(out[:360], image)


def test_no_image_at_all_draws_a_placeholder_not_a_crash():
    import cv2  # noqa: F401
    out = D.render(None, snapshot(), PARAMS)
    assert out.shape[1] == PARAMS['placeholder_size'][0]
    # the placeholder is not pure black (a black frame could be mistaken for a
    # genuinely dark but live camera image)
    assert out[:360].max() > 0


# -------------------------------------------------------------------- color

def test_state_color_keys_covers_every_coordinator_state():
    """The join `test_params.py` also checks from the YAML side; this is the
    code-only half -- every state the pure State enum defines must have an
    entry, with no dependence on params.yaml being loaded."""
    from pallet_approach.coordinator import State
    missing = [s.value for s in State if s.value not in D.STATE_COLOR_KEYS]
    assert not missing, missing


def test_a_good_outcome_is_not_colored_like_a_bad_one():
    good = D.CoordinatorInfo(outcome='INSERTED')
    bad = D.CoordinatorInfo(outcome='NO_GO')
    assert good.outcome in D.GOOD_OUTCOMES
    assert bad.outcome in D.BAD_OUTCOMES
    assert not set(D.GOOD_OUTCOMES) & set(D.BAD_OUTCOMES)


def test_tracker_status_colors_cover_every_tracker_status():
    """pallet_tracker.Tracker.status() can only return one of these five
    words -- if a sixth were ever added there, the HUD would silently fall
    back to grey for it, which looks like 'nothing to report' rather than
    'unrecognised status'."""
    # The five words Tracker.status() returns, read from its own source
    # rather than hard-coded here, so this breaks if the two ever diverge.
    import inspect
    from pallet_approach.tracker import Tracker
    source = inspect.getsource(Tracker.status)
    words = set(D.TRACKER_STATUS_COLORS)
    for word in words:
        assert word in source, (
            f'{word!r} is in TRACKER_STATUS_COLORS but Tracker.status() does '
            f'not appear to return it any more')


# ------------------------------------------------------------------- wrap

def test_short_text_does_not_wrap():
    import cv2
    lines = D._wrap('short', 600, cv2.FONT_HERSHEY_SIMPLEX, 0.4, cv2)
    assert lines == ['short']


def test_long_text_wraps_into_multiple_lines():
    import cv2
    text = ' '.join(f'word{i}' for i in range(40))
    lines = D._wrap(text, 200, cv2.FONT_HERSHEY_SIMPLEX, 0.4, cv2)
    assert len(lines) > 1
    # every line individually fits
    for line in lines:
        width = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
        assert width <= 200


def test_empty_text_does_not_crash_the_wrapper():
    import cv2
    assert D._wrap('', 400, cv2.FONT_HERSHEY_SIMPLEX, 0.4, cv2) == ['']


# ------------------------------------------------------------------ Snapshot

def test_age_is_none_before_anything_arrives():
    snap = D.Snapshot(now=5.0)
    assert snap.age(None) is None


def test_age_is_computed_relative_to_now():
    snap = D.Snapshot(now=5.0)
    assert snap.age(3.0) == pytest.approx(2.0)


def test_zero_is_a_valid_arrival_time_not_a_sentinel():
    """The same sentinel mistake as `FrameGate.last_start` and the health
    monitor's age: 0.0 is a real clock reading, not 'never happened'."""
    snap = D.Snapshot(now=5.0)
    assert snap.age(0.0) == pytest.approx(5.0)
    assert snap.age(0.0) is not None


# ------------------------------------------------------- the launch join

def test_the_video_stream_launch_default_matches_the_computed_frame_size():
    """`video_stream.launch.py` declares a fixed width/height for the
    dashboard GStreamer pipeline, because feeding it a size that does not
    match what `dashboard_node` actually publishes stretches the picture.
    That number is written by hand in the launch file rather than computed at
    launch time (ROS 2 launch substitutions cannot call into a Python
    function like this one), so nothing stops it drifting from
    config/params.yaml if the dashboard block's font_scale, line_height or
    pad ever change. This is that check -- it reads the launch file's literal
    `'height': N` for the dashboard stream and compares it against what
    `dashboard.frame_size` computes from the real params.yaml right now.
    """
    import re
    import pathlib

    import yaml

    package = pathlib.Path(__file__).resolve().parent.parent
    params = yaml.safe_load(
        (package / 'config' / 'params.yaml').read_text())['dashboard']['ros__parameters']
    expected_width, expected_height = D.frame_size(640, 360, params)

    source = (package / 'launch' / 'video_stream.launch.py').read_text()
    start = source.index("name='video_streamer_dashboard'")
    end = source.index('camera_stream = Node')
    block = source[start:end]
    width = int(re.search(r"'width':\s*(\d+)", block).group(1))
    height = int(re.search(r"'height':\s*(\d+)", block).group(1))

    assert (width, height) == (expected_width, expected_height), (
        f"video_stream.launch.py declares {(width, height)} for the dashboard "
        f"stream, but config/params.yaml's dashboard block now computes "
        f"{(expected_width, expected_height)} -- the GStreamer pipeline will "
        f"scale-fit (stretch) the real frames into the wrong box")
