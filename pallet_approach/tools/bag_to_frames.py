#!/usr/bin/env python3
"""Pull frames out of a rosbag without ROS (reads it with `rosbags`).

    python3 tools/bag_to_frames.py ~/bags/pallet_2m_0deg --out /tmp/frames
    python3 tools/bag_to_frames.py ~/bags/pallet_2m_0deg --list

Run in a venv (rosbags numpy pyyaml opencv-python), not the workstation's
system Python, where ROS Iron and conda break cv_bridge (docs/SYSTEM.md).

Outputs, per frame index:
    depth_000000.npy    (H, W) float32 metres, exactly as recorded
    image_000000.png    the rectified left image
    camera_info.yaml    fx, fy, cx, cy, distortion, frame_id -- written once
    index.csv           frame index, topic, timestamp, shape
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np

try:
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
except ImportError:
    sys.exit(
        'rosbags is not installed, or is too old for get_typestore. In a venv:\n'
        '    pip install -U rosbags numpy pyyaml opencv-python\n'
        'Do not use the system python here -- see the docstring.'
    )

# Humble's rosbag2 does not embed message definitions in the bag -- that only
# arrived in Iron. So the reader cannot work out what a sensor_msgs/msg/Image
# looks like from the file alone and has to be handed a typestore, or it fails
# with "Bag contains no type definitions". The robot runs Humble, so that is the
# store to use; the definitions we need (Image, CameraInfo, Odometry, TFMessage)
# are identical across the ROS 2 releases anyway.
TYPESTORES = {
    name.lower(): getattr(Stores, name)
    for name in dir(Stores) if name.startswith('ROS2_')
}

DEPTH_HINTS = ('depth_registered', '/depth')
IMAGE_HINTS = ('rect/image', 'image_rect', 'image_raw')
INFO_HINTS = ('camera_info',)

# encoding -> (numpy dtype, channels)
ENCODINGS = {
    '32FC1': (np.float32, 1),
    '16UC1': (np.uint16, 1),
    'mono8': (np.uint8, 1),
    'mono16': (np.uint16, 1),
    'bgr8': (np.uint8, 3),
    'rgb8': (np.uint8, 3),
    'bgra8': (np.uint8, 4),
    'rgba8': (np.uint8, 4),
}


def image_to_array(msg) -> np.ndarray:
    """Turn a sensor_msgs/Image into a NumPy array.

    This is all cv_bridge does for the simple encodings, and doing it by hand
    here means the offline tools need neither cv_bridge nor a matching NumPy.

    `step` is the row stride in bytes: it can be larger than width*channels*
    itemsize if the publisher pads rows, so slice per row rather than assuming
    the buffer is tightly packed.
    """
    encoding = msg.encoding
    if encoding not in ENCODINGS:
        raise ValueError(f'unhandled encoding {encoding!r}; add it to ENCODINGS')
    dtype, channels = ENCODINGS[encoding]
    itemsize = np.dtype(dtype).itemsize
    expected_step = msg.width * channels * itemsize

    buffer = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if msg.step == expected_step:
        array = buffer.view(dtype).reshape(msg.height, msg.width, channels)
    else:
        rows = buffer.reshape(msg.height, msg.step)[:, :expected_step]
        array = np.ascontiguousarray(rows).view(dtype).reshape(
            msg.height, msg.width, channels)

    if channels == 1:
        array = array[:, :, 0]
    if encoding == 'rgb8':
        array = array[:, :, ::-1]
    return array


def pick(topics, hints, explicit=None):
    """Choose a topic by name hints, or honour an explicit choice."""
    if explicit:
        if explicit not in topics:
            raise SystemExit(f'topic {explicit} is not in this bag')
        return explicit
    for hint in hints:
        for topic in topics:
            if hint in topic and 'compressed' not in topic and 'theora' not in topic:
                return topic
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bag', type=pathlib.Path)
    parser.add_argument('--out', type=pathlib.Path, default=pathlib.Path('frames'))
    parser.add_argument('--list', action='store_true', help='list topics and exit')
    parser.add_argument('--depth-topic')
    parser.add_argument('--image-topic')
    parser.add_argument('--max-frames', type=int, default=0, help='0 = all')
    parser.add_argument('--every', type=int, default=1, help='keep every Nth frame')
    parser.add_argument('--no-png', action='store_true',
                        help='skip the colour images (no OpenCV needed)')
    parser.add_argument('--typestore', default='ros2_humble',
                        choices=sorted(TYPESTORES),
                        help='message definitions to read the bag with; the '
                             'robot records on Humble (default: ros2_humble)')
    args = parser.parse_args()

    if not args.bag.exists():
        raise SystemExit(f'no such bag: {args.bag}')

    typestore = get_typestore(TYPESTORES[args.typestore])

    with AnyReader([args.bag], default_typestore=typestore) as reader:
        available = {c.topic: c for c in reader.connections}
        duration = getattr(reader, 'duration', 0) / 1e9

        if args.list:
            print(f'{args.bag}')
            print(f'  {duration:.2f} s, {getattr(reader, "message_count", 0)} messages, '
                  f'read with {args.typestore}')
            print()
            print(f'  {"msgs":>7}  {"Hz":>7}  topic')
            for topic, conn in sorted(available.items()):
                rate = conn.msgcount / duration if duration > 0 else float('nan')
                print(f'  {conn.msgcount:7d}  {rate:7.1f}  {topic}')
                print(f'  {"":7}  {"":7}    [{conn.msgtype}]')
            print()
            print('  Check the rates: a depth topic well below what you asked '
                  'for means\n  frames were dropped, not throttled.')
            return 0

        depth_topic = pick(available, DEPTH_HINTS, args.depth_topic)
        image_topic = pick(available, IMAGE_HINTS, args.image_topic)
        # prefer the camera_info that sits next to the depth topic -- that is
        # the one perception uses. They were identical on this robot, but do not
        # rely on that.
        info_topic = (pick(available, ('depth/camera_info',))
                      or pick(available, INFO_HINTS))
        print(f'depth : {depth_topic}')
        print(f'image : {image_topic}')
        print(f'info  : {info_topic}')
        if depth_topic is None and image_topic is None:
            raise SystemExit('found neither a depth nor an image topic; use --list')

        args.out.mkdir(parents=True, exist_ok=True)
        rows = []
        counts = {depth_topic: 0, image_topic: 0}
        kept = {depth_topic: 0, image_topic: 0}
        wrote_info = False

        wanted = [t for t in (depth_topic, image_topic, info_topic) if t]
        connections = [available[t] for t in wanted]

        for conn, timestamp, raw in reader.messages(connections=connections):
            msg = reader.deserialize(raw, conn.msgtype)
            stamp = timestamp * 1e-9

            if conn.topic == info_topic and not wrote_info:
                k = list(msg.k)
                info = {
                    'frame_id': msg.header.frame_id,
                    'width': int(msg.width), 'height': int(msg.height),
                    'fx': k[0], 'fy': k[4], 'cx': k[2], 'cy': k[5],
                    'distortion_model': msg.distortion_model,
                    'd': [float(v) for v in msg.d],
                }
                with open(args.out / 'camera_info.yaml', 'w') as fh:
                    for key, value in info.items():
                        fh.write(f'{key}: {value}\n')
                print(f'camera_info: fx={info["fx"]:.3f} cx={info["cx"]:.3f} '
                      f'cy={info["cy"]:.3f} frame={info["frame_id"]}')
                wrote_info = True
                continue

            if conn.topic not in counts:
                continue
            index = counts[conn.topic]
            counts[conn.topic] += 1
            if index % args.every:
                continue
            if args.max_frames and kept[conn.topic] >= args.max_frames:
                continue

            array = image_to_array(msg)
            n = kept[conn.topic]
            kept[conn.topic] += 1

            if conn.topic == depth_topic:
                path = args.out / f'depth_{n:06d}.npy'
                np.save(path, array)
            else:
                if args.no_png:
                    path = args.out / f'image_{n:06d}.npy'
                    np.save(path, array)
                else:
                    try:
                        import cv2
                    except ImportError:
                        raise SystemExit('need opencv-python for PNGs, or pass --no-png')
                    path = args.out / f'image_{n:06d}.png'
                    cv2.imwrite(str(path), array)

            rows.append((n, conn.topic, f'{stamp:.9f}', str(array.shape),
                         str(array.dtype), path.name))

        with open(args.out / 'index.csv', 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(['index', 'topic', 'stamp', 'shape', 'dtype', 'file'])
            writer.writerows(rows)

    print(f'\nwrote {len(rows)} files to {args.out}')
    if depth_topic and kept[depth_topic]:
        sample = np.load(args.out / 'depth_000000.npy')
        finite = np.isfinite(sample)
        print(f'first depth frame: {sample.shape} {sample.dtype}, '
              f'{finite.sum()} of {sample.size} pixels valid '
              f'({100 * finite.mean():.1f} %), '
              f'range {np.nanmin(sample):.3f} .. {np.nanmax(sample):.3f} m')
        print('\nThat percentage is worth looking at: invalid pixels are where '
              'the stereo match failed.\nDark, shiny and textureless surfaces '
              'produce holes, and they are a real result.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
