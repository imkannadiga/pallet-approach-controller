# Pallet Approach on a Real Robot

Geometry-driven pallet perception and fork-insertion approach on an AgileX
SCOUT 2.0 with a ZED 2i stereo camera, in ROS 2 Humble.

On a manual trigger the robot looks for a EUR pallet, works out which of its two
faces it is looking at, backs off if it is too close, drives itself onto the
pocket centre line, decides **GO** or **NO-GO** with a reason, and on GO
performs a **virtual pickup**: lowers imaginary forks to pocket height, drives
them in, lifts, backs out, and hands control back to Nav2.

No trained model. The pipeline is plane fitting, clustering and gap analysis, so
it works on pallets it has never seen. A fork overlay draws a forklift's tines
onto the camera image, which doubles as a visual check on the camera-to-base
calibration — the tines are bolted to the robot rather than to the measured
pockets, precisely so that a mismatch between them is attributable.

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
    perception.py       depth -> face pose, pockets, which side
    tracker.py          is it the same pallet, frame after frame?
    go_nogo.py          the nine safety rules
    closing.py          the servo that drives onto the pocket centre line
    insertion.py        the blind drive, and its two-source stop
    drive.py            executing one drive request
    forks.py            the virtual carriage
    coordinator.py      the state machine for a whole pickup
    health.py           is the stack fit to drive?
    fork_overlay.py     projecting the forks onto the image
    approach.py         standoff geometry and 2D pose algebra
    controller.py       limits, deadband, acceleration bound
    model.py            shared data types
    pointcloud.py       PointCloud2 <-> NumPy
    ros_conversions.py  messages <-> plain Python
    nodes/              ROS 2 wrappers (2 lifecycle, 4 plain)
  config/params.yaml    every tunable value, with its derivation
  config/twist_mux.yaml velocity arbitration
  test/                 pytest: synthetic clouds, a simulated world, a whole run
  tools/                survey, bag recording, offline replay, a whole pickup
docker/Dockerfile.jetson
docs/decisions.md       every behaviour decision, with its reasoning
docs/SYSTEM.md          measured system facts
docs/ROADMAP.md         build order and open decisions
docs/survey/            raw survey output from both machines
docs/log.md             daily log
```

Nothing in `pallet_approach/*.py` imports ROS. The node wrappers in `nodes/`
subscribe, publish and call into them. That split is not tidiness: it is what
lets a whole virtual pickup run at a desk against a simulated robot
(`tools/sequence.py`), which is where most of the bugs in this project were
found.

## Architecture

```
       ZED depth + tf                  Ouster
             │                            │
             ▼                            ▼
     pallet_perception ──estimate──▶ pallet_tracker ──track──┐
             │  cluster + scene points        (LiDAR cross-check)
             │        │                                      ▼
             │        └──────────────────────────────▶  coordinator
             │                                          │   │    │
             ▼                                     drive│   │    │fork
        fork_overlay ◀──state, fork height──────────────┘   │    │
                                                            ▼    ▼
                                         approach_controller   virtual_forks
                                                     │
      watchdog ──/cmd_vel_safety──▶ twist_mux ◀───────┘ /cmd_vel_pallet
            (sensor ages, rates)        │     ◀── Nav2  /cmd_vel_nav
                                        ▼
                             /cmd_vel ──▶ scout base (robot PC) ──▶ wheels
```

```
IDLE → CONFIRMING → CLASSIFYING → [BACKING_OFF] → CLOSING → CHECKING
     → LOWERING → INSERTING → LIFTING → BACKING_OUT → DONE
Any state: cancel, unhealthy stack, lost pallet, controller abort → ABORTED
```

The coordinator owns every decision to move or stop, and the controller is a
lifecycle node that cannot publish a command until the coordinator activates it.
Three safety properties are worth stating because they have no exceptions:

- **Every terminal state stops the robot with an explicit zero command.** The
  base driver has no command timeout (`ros2 param dump /scout`), so nothing
  else will. That includes an aborted insertion, which stops where it is rather
  than reversing out.
- **The watchdog owns the stop independently of the coordinator.** It watches
  the sensors itself and drives the mux's highest-priority input, so a
  coordinator that has wedged still leaves a stationary robot.
- **The controller may be told to drive in exactly four states**, and the test
  suite asserts that over every event from every state — so a state added later
  without thinking about it cannot drive.

## Build and run

Everything runs on the Jetson, next to the camera: depth never crosses the
network, only `/cmd_vel`, `/tf` and odometry do.

One entry point. `up` starts every node — including MediaMTX and the video
streams, by default — and then waits: nothing moves, and nothing will until a
run is triggered:

```bash
./docker/build_and_run.sh build
./docker/build_and_run.sh dry      # FIRST TIME: wheels disconnected
./docker/build_and_run.sh up       # for real
./docker/build_and_run.sh logs
./docker/build_and_run.sh stop
```

The flow is sequential: Nav2 drives the robot in front of a pallet and
**finishes**, and only then is the pipeline triggered.

```bash
ros2 topic pub --once /pallet_approach/start  std_msgs/String "{data: approach_only}"
ros2 topic pub --once /pallet_approach/start  std_msgs/String "{data: pickup}"
ros2 topic pub --once /pallet_approach/cancel std_msgs/Empty  "{}"
ros2 topic echo /pallet_approach/result
```

`approach_only` runs the whole approach and stops at the GO/NO-GO decision,
inserting nothing. An unknown mode is refused, not defaulted. A run is refused
if the stack is unhealthy when the trigger arrives, naming the sensor that is
stale or slow, and the refusal is published on `/pallet_approach/result` as
well as logged — nothing is waiting on a return value.

Watch it from a laptop on the robot's Wi-Fi, no ROS install needed:

    http://<jetson-ip>:8889/dashboard   forks, pockets, health, tracker,
                                         controller, forks, and why
    http://<jetson-ip>:8889/zed         the raw camera, unannotated

Nothing on that page is a ROS topic over the network — the composite image is
built on the Jetson and sent out as H.264/WebRTC (`docs/SYSTEM.md` has the
pipeline and why, including the one time a remote ROS subscriber brought the
whole camera node to its knees).

The action server does the same thing with live feedback, and is the better way
to run one by hand:

```bash
ros2 action send_goal /pallet_task \
    pallet_approach_interfaces/action/PalletTask \
    "{mode: pickup}" --feedback
```

Ctrl-C on that command cancels; the robot stops.

## Running a whole pickup without a robot

```bash
cd pallet_approach
python3 tools/sequence.py                   # a clean run from 3.5 m
python3 tools/sequence.py --range 2.5       # too close: it reverses first
python3 tools/sequence.py --noise 0.005 --seeds 10
python3 tools/sequence.py --break-gain      # a sign error in the servo
```

This drives the real state machine, the real tracker, the real control laws,
the real fork simulator and the real safety rules through a simulated world,
and prints the phase-by-phase trace. It does **not** exercise the ROS layer —
discovery, QoS, lifecycle, tf and the clock — which is where every failure on
the real robot has come from so far.

## Tests

```bash
cd pallet_approach
python -m pytest test
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
