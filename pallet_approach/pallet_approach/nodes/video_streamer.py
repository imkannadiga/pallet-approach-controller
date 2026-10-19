"""ROS 2 node: stream a camera topic to the laptop as low-latency video.

Runs on the Jetson. It subscribes to the image topic LOCALLY (on the same
machine as the ZED node, so no camera data crosses the network as ROS
messages), encodes it to H.264, and publishes it to MediaMTX, which serves it
to browsers over WebRTC:

    laptop browser:  http://<jetson-ip>:8889/<stream_path>

Run one instance per stream (camera, later the fork overlay), each with its own
`stream_path`.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image

from pallet_approach.video_stream import VideoStreamer


class VideoStreamerNode(Node):

    def __init__(self) -> None:
        super().__init__('video_streamer')

        topic = self.declare_parameter(
            'image_topic', '/zed2/zed_node/rgb/color/rect/image').value
        path = self.declare_parameter('stream_path', 'zed').value
        rtsp_port = self.declare_parameter('rtsp_port', 8554).value
        self.streamer = VideoStreamer(
            rtsp_url=f'rtsp://localhost:{rtsp_port}/{path}',
            out_width=self.declare_parameter('width', 1280).value,
            out_height=self.declare_parameter('height', 720).value,
            fps=self.declare_parameter('fps', 15).value,
            bitrate_kbps=self.declare_parameter('bitrate_kbps', 3000).value,
            encoder=self.declare_parameter('encoder', 'auto').value,
        )

        # Best effort, newest frame only: if this node is slow, it should drop
        # frames, never make the camera wait (the 2026-10-13 lesson). A
        # best-effort subscriber is compatible with the ZED's reliable publisher.
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.create_subscription(Image, topic, self._on_image, qos)
        self.create_timer(0.5, self._check_pipeline)
        self.create_timer(10.0, self._report)
        self._frames_at_last_report = 0

        self.get_logger().info(
            f'streaming {topic} with encoder "{self.streamer.encoder}" '
            f'-> browser at http://<jetson-ip>:8889/{path}')

    def _on_image(self, msg: Image) -> None:
        try:
            self.streamer.push(msg.data, msg.encoding, msg.width, msg.height, msg.step)
        except ValueError as exc:   # unsupported encoding: say so once per report
            self.get_logger().error(str(exc), throttle_duration_sec=10.0)

    def _check_pipeline(self) -> None:
        error = self.streamer.poll()
        if error:
            self.get_logger().warning(
                f'stream error (is MediaMTX running?), retrying: {error}',
                throttle_duration_sec=5.0)

    def _report(self) -> None:
        sent = self.streamer.frames_pushed - self._frames_at_last_report
        self._frames_at_last_report = self.streamer.frames_pushed
        self.get_logger().info(f'{sent / 10.0:.1f} fps sent to the encoder')

    def destroy_node(self) -> None:
        self.streamer.stop()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VideoStreamerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()