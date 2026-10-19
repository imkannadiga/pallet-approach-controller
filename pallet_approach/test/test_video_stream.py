"""Tests for video_stream.py.

Everything here needs `gi` (PyGObject) and a GStreamer install, neither of
which exists off the Jetson -- this whole module does `import gi;
gi.require_version(...); Gst.init(None)` at MODULE level, so even importing it
fails without them. `pytest.importorskip` at the top makes that a clean skip
rather than a collection error, the same pattern `test_tools.py` uses for
`clock_audit.py`'s rclpy dependency. Run for real inside the container.
"""

import pytest

gi = pytest.importorskip('gi', reason='PyGObject not installed here; Jetson only')

from pallet_approach import video_stream as V  # noqa: E402


# ------------------------------------------------------------ nvenc_properties

def test_nvenc_properties_are_not_baked_into_the_pipeline_string():
    """The whole point of the fix: `build_pipeline_description` must not
    contain any encoder property for nvenc, because `gst_parse_launch` fails
    the WHOLE pipeline on the first one this L4T build does not recognise --
    measured on an Orin, JetPack 6.2.2: `preset-level` does not exist on this
    element build and crash-looped the node every time a frame arrived.
    """
    description = V.build_pipeline_description(
        'BGRx', 1280, 720, 1280, 720, 15, 3000, 'rtsp://localhost:8554/zed',
        'nvenc')
    for name in V.nvenc_properties(3000, 15):
        assert f'{name}=' not in description, (
            f'{name} is baked into the nvenc pipeline string; a property this '
            f'L4T build lacks would crash gst_parse_launch instead of being '
            f'skipped by VideoStreamer._configure_nvenc')
    assert 'name=enc' in description, (
        'the encoder element must be named so _configure_nvenc can find it')


def test_nvenc_properties_scale_with_configured_bitrate_and_fps():
    """Bitrate is in kbps in the API and bits/s on the element; fps sets both
    keyframe intervals. Wrong units here are not a crash, just quietly wrong
    video -- worth pinning down explicitly."""
    props = V.nvenc_properties(bitrate_kbps=1200, keyframe_interval=5)
    assert props['bitrate'] == 1200 * 1000
    assert props['iframeinterval'] == 5
    assert props['idrinterval'] == 5


def test_nvenc_properties_requests_cbr_baseline_no_bframes():
    """The decisions this module's docstring states: CBR for a predictable
    stream, baseline profile so every target browser decodes it, no B-frames
    so no frame waits on a later one."""
    props = V.nvenc_properties(3000, 15)
    assert props['control-rate'] == 1       # CBR
    assert props['profile'] == 0            # Baseline
    assert props['num-B-Frames'] == 0


# ------------------------------------------------------ pipeline construction

def test_x264_and_vp8_keep_their_properties_in_the_string():
    """Unlike nvenc, these are standard software GStreamer properties that do
    not vary by L4T build, so there is no reason to apply them defensively --
    this is the contrast that makes the nvenc fix a deliberate choice rather
    than a blanket policy change."""
    x264 = V.build_pipeline_description(
        'I420', 640, 360, 640, 360, 5, 1200, 'rtsp://x', 'x264')
    assert 'bitrate=1200' in x264 and 'key-int-max=5' in x264

    vp8 = V.build_pipeline_description(
        'I420', 640, 360, 640, 360, 5, 1200, 'rtsp://x', 'vp8')
    assert 'target-bitrate=1200000' in vp8


def test_an_unknown_encoder_is_refused():
    with pytest.raises(ValueError, match='unknown encoder'):
        V.build_pipeline_description(
            'BGRx', 640, 360, 640, 360, 5, 1200, 'rtsp://x', 'quicksync')


def test_nvenc_converts_when_the_input_format_needs_it():
    """BGRx and RGBA/GRAY8 can go into nvvidconv directly; BGR/RGB (no alpha)
    cannot, and need an explicit videoconvert first."""
    needs_convert = V.build_pipeline_description(
        'BGR', 640, 360, 640, 360, 5, 1200, 'rtsp://x', 'nvenc')
    assert 'videoconvert' in needs_convert

    no_convert = V.build_pipeline_description(
        'BGRx', 640, 360, 640, 360, 5, 1200, 'rtsp://x', 'nvenc')
    assert 'videoconvert' not in no_convert


# -------------------------------------------------------------- frame_from_buffer

def test_frame_from_buffer_passes_through_tightly_packed_rows():
    data = bytes(range(12))  # 2x2 BGR image, step == width*3
    assert V.frame_from_buffer(data, width=2, height=2, step=6,
                               bytes_per_pixel=3) == data


def test_frame_from_buffer_strips_row_padding():
    """ROS images may pad each row to a stride wider than the pixel data;
    GStreamer wants rows packed with no gap."""
    width, height, bpp = 2, 2, 3
    row_bytes = width * bpp
    step = row_bytes + 2               # 2 bytes of padding per row
    rows = [bytes(range(row_bytes)) + b'\\xff\\xff' for _ in range(height)]
    data = b''.join(rows)
    packed = V.frame_from_buffer(data, width, height, step, bpp)
    assert len(packed) == row_bytes * height
    assert b'\\xff' not in packed


# ------------------------------------------------------------- VideoStreamer

def test_an_unsupported_ros_encoding_is_refused_not_guessed():
    streamer = V.VideoStreamer(encoder='x264')
    with pytest.raises(ValueError, match='unsupported image encoding'):
        streamer.push(b'', 'yuv422', 4, 4, 16)


def test_frames_faster_than_fps_are_skipped_before_any_copy():
    """Measured cheaply: push() returns False without touching the data at
    all when called again inside one frame period."""
    streamer = V.VideoStreamer(encoder='x264', fps=5)
    streamer._last_push = __import__('time').monotonic()
    assert streamer.push(b'\\x00' * 48, 'bgr8', 4, 4, 12) is False
