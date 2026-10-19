# Pallet Approach on a Real Robot

Geometry-driven pallet perception and fork-insertion approach on an AgileX
SCOUT 2.0 with a ZED 2i stereo camera, in ROS 2 Humble.

On a manual trigger the robot looks for a EUR pallet, estimates the pose of its
front face and the two fork pockets, drives itself into fork-insertion
alignment, and returns **GO** or **NO-GO** with a reason. No trained model: the
pipeline is plane fitting, clustering and gap analysis, so it works on pallets
it has never seen. A fork overlay draws a forklift's tines onto the camera image,
which doubles as a visual check on the camera-to-base calibration.

> Results and accuracy numbers are deliberately absent from this README until
> they come from a real evaluation run. The numbers that *are* here were
> measured on the robot on 2026-10-13 and are recorded in
> [docs/SYSTEM.md](docs/SYSTEM.md).

## Layout

```
pallet_approach_interfaces/     action + messages (ament_cmake)
pallet_approach/
  pallet_approach/
    model.py            shared data types: PalletEstimate, Pocket, Decision, Pose2D
    perception.py       depth -> pose + pockets                    [to write]
    go_nogo.py          the safety rules                           [to write]
    fork_overlay.py     projecting the forks onto the image        [to write]
    approach.py         standoff and line-up geometry              done
    controller.py       rotate-drive-rotate pose controller        done
    coordinator.py      the state machine                          done
    ros_conversions.py  messages <-> plain Python
    nodes/              ROS 2 wrappers (2 lifecycle, 2 plain)
  config/params.yaml    every tunable value, with its derivation
  test/                 pytest, on synthetic pallet clouds
  tools/                survey, bag recording, offline frame extraction
docker/Dockerfile.jetson
docs/SYSTEM.md          measured system facts
docs/survey/            raw survey output from both machines
docs/log.md             daily log
```

The three files marked `[to write]` are the core perception and safety logic and
are written by hand, not generated. They ship as skeletons with signatures,
explanations and TODOs, and with a full test suite that fails until they are
implemented.

## Architecture

```
operator ──goal──▶ coordinator ──activate──▶ pallet_perception ◀── ZED depth + tf
             ◀──feedback/result──┤                   │
                                 │          PalletEstimate
                                 ▼                   │
                         approach_controller ◀───────┘
                                 │
                              /cmd_vel ──▶ scout_base (robot PC) ──▶ wheels
```

`IDLE → DETECTING → APPROACHING → CHECKING → DONE`. The coordinator owns every
decision to move or stop. The controller is a lifecycle node and cannot publish
`/cmd_vel` until the coordinator activates it. Every terminal state stops the
robot with an explicit zero command — the base driver has no command timeout, so
nothing else will.

## Build and run

Everything runs on the Jetson, next to the camera: depth never crosses the
network, only `/cmd_vel`, `/tf` and odometry do.

```bash
./docker/build_and_run.sh build
./docker/build_and_run.sh launch
```

Nothing moves on launch. Start one run:

```bash
ros2 action send_goal /pallet_task \
    pallet_approach_interfaces/action/PalletTask \
    "{mode: pallet_orientation}" --feedback
```

Cancel at any time with Ctrl-C on that command; the robot stops.

## Tests

```bash
cd pallet_approach

# everything that should pass today
python -m pytest test -m "not unimplemented"

# the targets: one test per function still to write
python -m pytest test/test_perception.py -x -v
```

On the workstation, prefix with `env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`:
ROS Iron is on the path alongside a conda Python and the two collide. Better,
use a venv for offline work — see `pallet_approach/tools/bag_to_frames.py`.

## Working with bags

Raw depth is 136 MB/s and the Jetson has ~5.7 GB free, so bags are throttled and
cannot be left on the Jetson. One command from the laptop records on the Jetson,
copies the bag here, verifies it, and only then deletes the remote copy:

```bash
./pallet_approach/tools/fetch_bag.sh --check                     # test the path first
./pallet_approach/tools/fetch_bag.sh -o pallet_2m_0deg -d 20     # depth only, 5 Hz
./pallet_approach/tools/fetch_bag.sh -o pallet_2m_0deg -d 20 --with-image
```

Nothing on the robot has to be reconfigured: the bag is written inside the
already-running container and streamed out with `docker exec | tar`, so there is
no bind mount to add and no container to restart. `--fetch-only` retries a failed
transfer without re-recording.

From the robot's hotspot, ssh goes straight to `stuttgart_orin@192.168.1.1` —
no jump host, because the robot PC forwards between its AP subnet and the wired
switch. Use that wired address rather than the Jetson's own wifi address
(`10.42.0.40`): over the wired leg only the last hop is wifi, so it is about
twice as fast. Pass `--container <name>` (or set `CONTAINER=`).

Streaming the depth off the Jetson *while* recording is not possible: 10 Hz
depth plus colour is ~516 Mbit/s. Record local, then move.

Then read it offline, with no ROS needed:

```bash
python3 pallet_approach/tools/bag_to_frames.py ~/bags/pallet_2m_0deg --out /tmp/frames
```

Or replay it through the whole perception pipeline and draw every stage —
the fastest way to see *why* a frame produced what it did. The last panel
projects the detected pockets back onto the image they came from, measured
against expected, which is the check you can make by eye:

```bash
python3 pallet_approach/tools/replay.py ~/bags/pallet_2m_0deg --truth-x 2.0
python3 pallet_approach/tools/floor_plane_check.py --correct ~/bags/floor_*
```

Write the tape-measured ground truth for every bag into
[docs/dataset.md](docs/dataset.md). A bag without ground truth cannot be used for
evaluation.

## Safety

- Speed limits live in `config/params.yaml`: 0.20 m/s linear, 0.30 rad/s
  angular. They are the lab limits for autonomous runs and are not raised.
- Every command is clamped and acceleration-limited before it reaches the base.
- The controller refuses to configure if anything else publishes `/cmd_vel`.
- Controller and coordinator changes are replayed against a recorded bag before
  they run on the robot.
- Someone supervises every run with the e-stop in reach.
