# TODO

Work that was deferred or left open, with where each item came from. Decisions
live in `decisions.md`; the build order in `ROADMAP.md`. An item here is not
decided until it reaches `decisions.md`.

## Perception

### Fused perception (D102, D104): open items
- The fused sideways sigma is optimistic (+-2 mm claimed, -12/-3 mm on the two
  taped bags): record more taped-lateral bags before trusting it.
- The full D96 sim acceptance in fused mode (seed 7, 10 runs) was stopped
  early (D106); run it to the end for a 9-of-10 count.
- The fusion node leaves TrackedPallet's frame counters at 0, so the
  dashboard's TRACKER widget reads "0/0 frames".

### Fused perception on the robot (needs the robot)
Rebuild the image; `./docker/build_and_run.sh dry` (fused is the default,
D106); read /pallet_fusion/stats (timing on the Jetson, rates, sigma) next to the old
pipeline's estimate through a few runs. Also: D101 dry run, D103 against the
calibration runs' PnP distances, taped-lateral and loaded-pallet bags.


- **Learned pallet detector in front of the geometry.** A small detector
  (YOLO class, TensorRT on the Orin's GPU; an open-vocabulary model such as
  YOLO-World prompted with "pallet", or one fine-tuned on a public pallet set)
  returns a box, and the geometry only considers clusters inside it. A filter,
  not a replacement: the pose still comes from depth. Replaces the height
  ceiling (D76) and makes loaded pallets possible. Needs an L4T/CUDA base
  image (ours is `ros:humble-ros-base`), TensorRT and the model on a Jetson
  with 6.3 GB free: ask before adding (hard rule 5). Measure misses and false
  boxes on recorded bags, and latency on the Jetson, before it drives.
  (2026-10-23)
- **Turn side classification back on** (D78; off, short face assumed). First
  decide how the tracker counts an "ambiguous" frame: D36 resets confirmation
  on any side other than "unknown", and at 2.4-2.5 m about one frame in
  several is ambiguous. Recommended then: treat it as neutral, like "unknown".
- **Loaded pallets.** D76 rejects any cluster more than 4 cm taller than an
  empty pallet. A rule for loads is needed with or without the detector.
- **Pocket detection beyond ~3 m.** At 3.35 m the face carries ~40 points at
  pocket height and pockets are found only intermittently; runs now start at
  ~2.5 m instead (D75). Options if longer range is ever needed: the face slab
  from un-voxelised points, finer voxels near the face, or accumulating
  several frames. (2026-10-23)
- **Re-enable the overhang rule** (D61) and **the pocket clearance rule**
  (D63), both switched off for now on the assumption that nothing sticks out
  of the pallet and the pockets are clear.
- **Per-frame position scatter on the real ZED**, to confirm the stability
  tolerance (D60; "Open, with numbers attached" in `decisions.md`).
- **LiDAR cross-check `min_points: 20`** is arithmetic (D37); replace with
  counts measured on bags.

## Calibration

- **Auto-calibration routine** (D74): on demand, turn in place to four
  headings, fit the floor with camera and LiDAR at each, solve height, pitch
  and roll per sensor; refuse on disagreement between headings or implausible
  corrections; results in a separate calibration file, not `params.yaml`;
  run-time warning when perception's floor drifts from it. Replaces the
  hand-entered LiDAR stopgap (D74) and the camera's measured correction. Build
  once perception works on the robot.
- **The LiDAR's forward offset** (D90): it reads ~14 mm too far on the
  pallet's front edge at the insertion stop (tape, 2026-10-24). The
  auto-calibration needs a known vertical target for x, y and yaw.
- **Report the URDF's mounting errors** to whoever maintains the robot
  description: `base_footprint` 25 mm, LiDAR 50 mm high and pitched 0.58°,
  camera 0.87° of tilt.

## Compute on the Jetson

- **Power mode.** The Jetson runs MODE_30W: 8 of 12 cores, capped clocks.
  MAXN (`sudo nvpmodel -m 0`, may need a reboot) is the cheapest speed-up for
  every node; costs battery and heat. Undecided. (2026-10-23)
- **Perception on the GPU** (Athreya, 2026-10-23: "as fast as possible, then
  on the GPU"). CPU work done: 80 -> 34 ms a frame on the laptop (docs/log.md);
  re-measure on the Jetson first. Then voxel/cluster/RANSAC on the GPU, which
  needs CuPy or similar in the image: check it installs on the Jetson and fits
  the disk, and ask before adding it.
- **Perception speed (before 2026-10-23's CPU work).** ~186 ms per frame in isolation, 250-350 ms with the
  stack running, so ~3.3 Hz against the watchdog's 2.5 Hz floor. Floor RANSAC
  (74 ms) and clustering (42 ms) dominate. Re-measure after MAXN; then GPU
  (CuPy) for those two stages if still needed. A vectorised RANSAC on the CPU
  was tried and is slower (3.4x on the laptop).
- **Raise the pipeline from 5 Hz to 10 Hz**, deferred; depends on the two
  items above.
- **Dashboard cost.** The fork overlay takes ~110 % CPU for the 24 fps
  dashboard. Option: a lower frame rate while a run is active.

## Robot and network

- **Odometry stalls** (open in D72). Most likely the LiDAR saturating the
  link, fixed 2026-10-22/23 (relay, 32 MB UDP buffers); confirm over a longer
  run.
- **Restart the ZED container** once so its sockets get the larger UDP
  buffers too (automatic from the next reboot).
- **Remap Nav2 to `/cmd_vel_nav`** so the nav lock is not inert (D55; "when
  convenient").
- **USB3 SSD for the Jetson**: the disk cannot hold the dataset or per-run
  recordings.

## Features

- **Run recorder** (D11, deferred).
- **Goal fields for an expected pallet pose and ID** (ROADMAP stage 0,
  deferred).
- **Repositioning to the pocket side via Nav2** (ROADMAP stage 9, stretch).

## Housekeeping

- **Commit** everything since `fbb5957` (trigger tool, relay, tf listener,
  watchdog fix, D72-D78).
- **Re-run the Gazebo sim** on the GPU machine after the 2026-10-22/23
  changes (relay, tf listener, LiDAR correction off in sim, run-up 2.5 m,
  height ceiling) and confirm 9 of 10 accepted (INSERTED) pickups from the
  shared starting positions, seed 7 (D96).
- **Evaluation bags**: a set of positions for the dataset (`dataset.md`).
