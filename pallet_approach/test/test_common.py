"""Tests for nodes/_common.py -- the no-cv_bridge image and cloud decoding.

`image_to_array` has never had tests of its own, and that is how the two bugs
fixed here reached the robot before anyone wrote one down: a buffer a few
bytes larger than `height * step` (the ZED's bgra8 topic, which crashed the
reshape) and a 4-channel image handed straight to code built for 3-channel BGR
(every consumer -- fork_overlay's drawing, the video encoders -- assumes BGR,
and bgra8 is the ZED's actual encoding for the rectified left image).
"""

import numpy as np
import pytest

from pallet_approach.nodes._common import image_to_array


class Image:
    """A fake sensor_msgs/Image: just the fields image_to_array reads."""

    def __init__(self, array: np.ndarray, encoding: str, step: int | None = None,
                extra_bytes: int = 0):
        self.height, self.width = array.shape[0], array.shape[1]
        self.encoding = encoding
        row_bytes = array[0].nbytes if array.ndim == 3 else array[0].nbytes
        self.step = step if step is not None else row_bytes
        data = array.tobytes()
        if step is not None and step > row_bytes:
            # Insert padding per row to match a wider declared step.
            rows = [array[r].tobytes() + b'\x00' * (step - row_bytes)
                   for r in range(self.height)]
            data = b''.join(rows)
        self.data = data + b'\x00' * extra_bytes


def rgb_grid(height=3, width=4, channels=3, dtype=np.uint8):
    size = height * width * channels
    return (np.arange(size) % 256).astype(dtype).reshape(height, width, channels)


# -------------------------------------------------------------- basic shapes

def test_bgr8_passes_through_unchanged():
    original = rgb_grid(channels=3).astype(np.uint8)
    array = image_to_array(Image(original, 'bgr8'))
    assert array.shape == original.shape
    assert np.array_equal(array, original)


def test_mono8_drops_the_channel_axis():
    original = rgb_grid(channels=1).astype(np.uint8)
    array = image_to_array(Image(original, 'mono8'))
    assert array.ndim == 2
    assert array.shape == (3, 4)


def test_rgb8_is_reversed_to_bgr():
    original = rgb_grid(channels=3).astype(np.uint8)
    array = image_to_array(Image(original, 'rgb8'))
    assert np.array_equal(array, original[:, :, ::-1])


def test_an_unknown_encoding_is_refused():
    original = rgb_grid(channels=1).astype(np.uint8)
    with pytest.raises(ValueError, match='unhandled encoding'):
        image_to_array(Image(original, 'yuv422'))


# ------------------------------------------------------- the alpha channel

def test_bgra8_strips_the_alpha_channel():
    """The ZED's actual encoding for the rectified left image (docs/SYSTEM.md,
    video_stream.py's ROS_TO_GST_FORMAT table) -- every consumer of this
    function (fork_overlay's drawing, the video encoders) assumes 3-channel
    BGR, and a bare 4-channel array handed to them is a shape mismatch rather
    than a picture."""
    original = rgb_grid(channels=4).astype(np.uint8)
    array = image_to_array(Image(original, 'bgra8'))
    assert array.shape == (3, 4, 3)
    assert np.array_equal(array, original[:, :, :3]), (
        'bgra8 is already B,G,R,A in memory order, so dropping the fourth '
        'channel must leave B,G,R untouched -- no reversal needed')


def test_rgba8_strips_alpha_but_is_not_reversed_to_bgr():
    """Documents an asymmetry rather than fixing it: unlike the 3-channel
    rgb8 case, this function does not reverse a stripped rgba8 image to BGR --
    it is never actually produced by anything in this project (the ZED
    publishes bgra8), so there has been no real input to catch this on. If
    something ever does publish rgba8, this test is the one that should start
    failing once the reversal is added, not a surprise on the robot."""
    original = rgb_grid(channels=4).astype(np.uint8)
    array = image_to_array(Image(original, 'rgba8'))
    assert np.array_equal(array, original[:, :, :3]), (
        'still R,G,B -- NOT reversed to B,G,R')


# ------------------------------------------------------------- row padding

def test_a_padded_row_stride_is_unpacked_correctly():
    original = rgb_grid(channels=3).astype(np.uint8)
    padded_step = original[0].nbytes + 8
    array = image_to_array(Image(original, 'bgr8', step=padded_step))
    assert np.array_equal(array, original)


def test_padding_and_alpha_stripping_both_apply_together():
    original = rgb_grid(channels=4).astype(np.uint8)
    padded_step = original[0].nbytes + 16
    array = image_to_array(Image(original, 'bgra8', step=padded_step))
    assert np.array_equal(array, original[:, :, :3])


# --------------------------------------------------- a buffer with trailing bytes

def test_a_buffer_longer_than_height_times_step_does_not_crash():
    """The bug this test is built around: the ZED's bgra8 topic arrived with
    a few bytes more than `height * step`, and reshape raised on the
    mismatch before this function truncated the buffer first."""
    original = rgb_grid(channels=4).astype(np.uint8)
    array = image_to_array(Image(original, 'bgra8', extra_bytes=37))
    assert array.shape == (3, 4, 3)
    assert np.array_equal(array, original[:, :, :3])


def test_a_buffer_exactly_the_right_size_is_unaffected():
    """The truncation must be a no-op on the ordinary case -- it must not,
    for instance, quietly drop a real final row."""
    original = rgb_grid(channels=3).astype(np.uint8)
    array = image_to_array(Image(original, 'bgr8', extra_bytes=0))
    assert np.array_equal(array, original)
