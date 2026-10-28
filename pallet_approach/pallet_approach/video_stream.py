"""Low-latency H.264 streaming of camera frames to MediaMTX (-> WebRTC in a browser).

    VideoStreamer.push() -> appsrc -> scale/convert -> encoder -> RTSP
        -> MediaMTX on the Jetson -> http://<jetson-ip>:8889/<path> (WebRTC)

No ROS; `nodes/video_streamer.py` feeds it. Encoders: "nvenc" (Jetson
hardware), "x264", "vp8" (browsers without H.264), "auto" (nvenc if present).
For latency: a leaky one-frame queue, no B-frames, baseline profile, a
keyframe every second.
"""

from __future__ import annotations

import logging
import time

import numpy as np

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst  # noqa: E402

Gst.init(None)

log = logging.getLogger(__name__)

# ROS image encodings -> GStreamer raw formats. The 4-channel ones can go into
# the Jetson's hardware converter directly; 3-channel ones need videoconvert.
ROS_TO_GST_FORMAT = {
    'bgra8': 'BGRx',   # ZED default; alpha is ignored
    'rgba8': 'RGBx',
    'bgr8': 'BGR',
    'rgb8': 'RGB',
    'mono8': 'GRAY8',
}
BYTES_PER_PIXEL = {'BGRx': 4, 'RGBx': 4, 'BGR': 3, 'RGB': 3, 'GRAY8': 1}
NVVIDCONV_INPUT_FORMATS = {'BGRx', 'RGBA', 'GRAY8'}


def nvenc_available() -> bool:
    """True if the Jetson hardware encoder's GStreamer element can be created."""
    return Gst.ElementFactory.find('nvv4l2h264enc') is not None


def nvenc_properties(bitrate_kbps: int, keyframe_interval: int) -> dict:
    """The hardware encoder's tunable properties, as {name: value}.

    Applied one at a time with an exists-check after the pipeline is built, not
    in the launch string: the property set changes across L4T releases
    (`preset-level` is gone on JetPack 6.2.2) and an unknown name fails the
    whole pipeline. `control-rate=1` is CBR, `profile=0` Baseline.
    """
    return {
        'bitrate': bitrate_kbps * 1000,
        'control-rate': 1,              # CBR
        'profile': 0,                   # Baseline
        'maxperf-enable': True,
        'insert-sps-pps': True,
        'iframeinterval': keyframe_interval,
        'idrinterval': keyframe_interval,
        'num-B-Frames': 0,
    }


def build_pipeline_description(
    in_format: str,
    in_width: int,
    in_height: int,
    out_width: int,
    out_height: int,
    fps: int,
    bitrate_kbps: int,
    rtsp_url: str,
    encoder: str,
) -> str:
    """The GStreamer pipeline as a gst-launch style string.

    The nvenc encoder element is named (`name=enc`) and carries NO properties
    in the string itself -- see `nvenc_properties` for why, and
    `VideoStreamer._start` for where they are actually applied.
    """
    source = (
        f'appsrc name=src is-live=true format=time do-timestamp=true '
        f'caps=video/x-raw,format={in_format},width={in_width},'
        f'height={in_height},framerate={fps}/1 '
        f'! queue leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0'
    )
    keyframe_interval = fps  # one keyframe per second

    if encoder == 'nvenc':
        convert = ''
        if in_format not in NVVIDCONV_INPUT_FORMATS:
            convert = ' ! videoconvert ! video/x-raw,format=BGRx'
        encode = (
            f'{convert} ! nvvidconv '
            f'! video/x-raw(memory:NVMM),format=NV12,width={out_width},height={out_height} '
            f'! nvv4l2h264enc name=enc'
        )
    elif encoder == 'x264':
        encode = (
            f' ! videoconvert ! videoscale '
            f'! video/x-raw,format=I420,width={out_width},height={out_height} '
            f'! x264enc tune=zerolatency speed-preset=ultrafast bitrate={bitrate_kbps} '
            f'key-int-max={keyframe_interval} bframes=0 '
            f'! video/x-h264,profile=constrained-baseline'
        )
    elif encoder == 'vp8':
        # Software only (the Orin has no VP8 encoder). For browsers without
        # H.264 support, e.g. some Linux Chromium builds.
        encode = (
            f' ! videoconvert ! videoscale '
            f'! video/x-raw,format=I420,width={out_width},height={out_height} '
            f'! vp8enc deadline=1 target-bitrate={bitrate_kbps * 1000} '
            f'keyframe-max-dist={keyframe_interval} '
            f'! video/x-vp8,profile=(string)0'
        )
        return source + encode + f' ! rtspclientsink location={rtsp_url} protocols=tcp'
    else:
        raise ValueError(f'unknown encoder {encoder!r}')

    sink = f' ! h264parse config-interval=-1 ! rtspclientsink location={rtsp_url} protocols=tcp'
    return source + encode + sink


def frame_from_buffer(data, width: int, height: int, step: int, bytes_per_pixel: int) -> bytes:
    """Raw ROS image bytes -> tightly packed rows.

    ROS images may pad each row (step > width * bytes_per_pixel). GStreamer
    expects packed rows, so cut the padding off if there is any.
    """
    row_bytes = width * bytes_per_pixel
    if step == row_bytes:
        return bytes(data)
    rows = np.frombuffer(data, dtype=np.uint8).reshape(height, step)
    return rows[:, :row_bytes].tobytes()


class VideoStreamer:
    """Turns frames into an H.264 RTSP stream. Builds its pipeline on the first frame."""

    def __init__(
        self,
        rtsp_url: str = 'rtsp://localhost:8554/zed',
        out_width: int = 1280,
        out_height: int = 720,
        fps: int = 15,
        bitrate_kbps: int = 3000,
        encoder: str = 'auto',
        retry_seconds: float = 2.0,
    ) -> None:
        if encoder == 'auto':
            encoder = 'nvenc' if nvenc_available() else 'x264'
        self.encoder = encoder
        self.rtsp_url = rtsp_url
        self.out_width = out_width
        self.out_height = out_height
        self.fps = fps
        self.bitrate_kbps = bitrate_kbps
        self.retry_seconds = retry_seconds

        self._pipeline = None
        self._appsrc = None
        self._input_key = None          # (format, width, height) the pipeline was built for
        self._last_push = 0.0
        self._retry_after = 0.0
        self.frames_pushed = 0

    # -- public ---------------------------------------------------------------
    def push(self, data, encoding: str, width: int, height: int, step: int) -> bool:
        """Offer one frame. Returns True if it was sent to the encoder.

        Frames arriving faster than `fps` are skipped here, before any copy,
        so a 37 Hz camera costs only 15 Hz of work.
        """
        now = time.monotonic()
        if now - self._last_push < 1.0 / self.fps:
            return False
        if now < self._retry_after:
            return False

        gst_format = ROS_TO_GST_FORMAT.get(encoding)
        if gst_format is None:
            raise ValueError(f'unsupported image encoding {encoding!r}')

        key = (gst_format, width, height)
        if self._pipeline is None or key != self._input_key:
            self._start(gst_format, width, height)

        frame = frame_from_buffer(data, width, height, step, BYTES_PER_PIXEL[gst_format])
        result = self._appsrc.emit('push-buffer', Gst.Buffer.new_wrapped(frame))
        self._last_push = now
        if result != Gst.FlowReturn.OK:
            self._fail(f'push-buffer returned {result.value_nick}')
            return False
        self.frames_pushed += 1
        return True

    def poll(self) -> str | None:
        """Check the pipeline for errors; call regularly. Returns an error text or None.

        On an error (typically: MediaMTX not running yet) the pipeline is torn
        down and rebuilt on a later frame after `retry_seconds`.
        """
        if self._pipeline is None:
            return None
        bus = self._pipeline.get_bus()
        while True:
            msg = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
            if msg is None:
                return None
            if msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                text = f'{err.message} ({debug})'
            else:
                text = 'end of stream'
            self._fail(text)
            return text

    def stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.set_state(Gst.State.NULL)
        self._pipeline = None
        self._appsrc = None
        self._input_key = None

    # -- internals ------------------------------------------------------------
    def _start(self, gst_format: str, width: int, height: int) -> None:
        self.stop()
        description = build_pipeline_description(
            gst_format, width, height, self.out_width, self.out_height,
            self.fps, self.bitrate_kbps, self.rtsp_url, self.encoder)
        log.info('starting pipeline: %s', description)
        self._pipeline = Gst.parse_launch(description)
        self._appsrc = self._pipeline.get_by_name('src')
        self._input_key = (gst_format, width, height)
        if self.encoder == 'nvenc':
            self._configure_nvenc()
        self._pipeline.set_state(Gst.State.PLAYING)

    def _configure_nvenc(self) -> None:
        """Apply the hardware encoder's tunable properties one at a time.

        Not in the gst-launch string -- see `nvenc_properties`. A property
        this L4T build's `nvv4l2h264enc` does not expose is skipped and named
        in a single log line rather than taking the whole pipeline down, which
        is what baking it into the parsed string did on first contact with a
        real Orin (`preset-level`, measured 2026-10-19, JetPack 6.2.2).
        """
        encoder = self._pipeline.get_by_name('enc')
        if encoder is None:
            log.warning('nvenc requested but no "enc" element in the pipeline; '
                       'nothing to configure')
            return
        keyframe_interval = int(self.fps)
        applied, skipped = [], []
        for name, value in nvenc_properties(self.bitrate_kbps,
                                            keyframe_interval).items():
            try:
                encoder.set_property(name, value)
                applied.append(name)
            except TypeError:
                # GObject raises TypeError for a property this element build
                # does not have -- exactly the "no property X" error that a
                # baked-in gst-launch string turns into a crash instead.
                skipped.append(name)
            except Exception as error:  # noqa: BLE001
                # Anything else (a property that exists but rejects this
                # value, say) is also one property failing to apply, not a
                # reason to lose the whole stream -- but it is a different
                # failure from "not supported" and is named as one.
                skipped.append(f'{name} ({error})')
        log.info('nvenc: applied %s; skipped %s',
                sorted(applied), sorted(skipped) or 'none')

    def _fail(self, reason: str) -> None:
        log.warning('stream error, retrying in %.1f s: %s', self.retry_seconds, reason)
        self.stop()
        self._retry_after = time.monotonic() + self.retry_seconds