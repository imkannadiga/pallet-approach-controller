"""Write the robot's static transforms from one good bag to config/static_tf.yaml.

    python3 tools/export_static_tf.py ~/bags/floor_pose1

Static transforms are the rigid mounting: the chassis, the camera and the LiDAR
on their brackets. They do not change from bag to bag, so a recording does not
need to carry them. A bag sometimes arrives without some of them (the recorder
joined the latched /tf_static topic before it had discovered every publisher),
and the offline tools fill in what is missing from this file instead.

Regenerate it whenever the robot's hardware or its description changes: a
sensor remounted, a bracket changed, a new URDF. The file records the bag it
came from and the date, so a stale one is visible. Dynamic transforms (odom,
the ZED's self-calibration link) are NOT in it; every bag carries those.
"""

from __future__ import annotations

import argparse
import datetime
import pathlib
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _bagutil as bag  # noqa: E402
from rosbags.highlevel import AnyReader  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('bag', type=pathlib.Path)
    ap.add_argument('--out', type=pathlib.Path, default=bag.STATIC_TF)
    args = ap.parse_args()

    edges = {}
    with AnyReader([args.bag], default_typestore=bag.typestore()) as reader:
        conns = [c for c in reader.connections if c.topic == '/tf_static']
        for conn, _, raw in reader.messages(connections=conns):
            for tr in reader.deserialize(raw, conn.msgtype).transforms:
                t, q = tr.transform.translation, tr.transform.rotation
                edges[(tr.header.frame_id, tr.child_frame_id)] = {
                    'parent': tr.header.frame_id, 'child': tr.child_frame_id,
                    'translation': [float(t.x), float(t.y), float(t.z)],
                    'rotation': [float(q.x), float(q.y), float(q.z), float(q.w)]}
    if not edges:
        sys.exit(f'{args.bag}: no /tf_static messages')
    children = {c for _, c in edges}
    missing = [n for n in ('os_lidar', 'base_footprint') if n not in children]
    if missing:
        sys.exit(f'{args.bag} lacks {missing}: pick a bag with the whole static tree')

    doc = {'source_bag': args.bag.name,
           'exported': datetime.date.today().isoformat(),
           'note': 'Static transforms of the real robot. Regenerate if the hardware or '
                   'the robot description changes. Dynamic transforms are not here.',
           'edges': [edges[k] for k in sorted(edges)]}
    args.out.write_text(yaml.safe_dump(doc, sort_keys=False))
    print(f'wrote {len(edges)} static edges from {args.bag.name} to {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
