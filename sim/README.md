# Gazebo sim: the lab SCOUT and the measured pallet

The whole stack, unchanged, against a simulated robot. It sits between the
offline harness (`tools/sequence.py`, which tests the logic but none of the ROS
layer) and the robot. It exercises what the harness cannot: discovery, QoS,
lifecycle, tf, the clock, the launch file, the container and the dashboard.

```bash
./sim/run_sim.sh build          # once; see "Prerequisites"
./sim/run_sim.sh up             # Gazebo + the stack, detached
./sim/run_sim.sh status         # /coordinator/status and /health
./sim/run_sim.sh start approach_only
./sim/run_sim.sh result
./sim/run_sim.sh start pickup
./sim/run_sim.sh cancel
./sim/run_sim.sh logs [gazebo|stack]
./sim/run_sim.sh stop
```

Dashboard: <http://localhost:8889/dashboard>, the same page as on the robot.

Place the pallet with launch arguments (robot at the origin, looking along +x):

```bash
./sim/run_sim.sh up face_x:=2.5 face_y:=0.3 face_yaw:=0.2   # metres, metres, radians
./sim/run_sim.sh up face:=long                               # the long side towards the robot
./sim/run_sim.sh up gui:=true                                # the Gazebo window (X11)
```

## What is the same as the robot

| | robot | sim |
|---|---|---|
| stack container | `docker/Dockerfile.jetson` | the same Dockerfile, built for amd64 |
| launch | `pallet_approach.launch.py` | the same, plus `overlay_params` and `encoder:=x264` |
| params | `config/params.yaml` | the same, plus `config/sim_overlay.yaml` (sim time; no camera tilt correction) |
| topics | `/zed2/zed_node/...`, `/ouster/points`, `/odom`, `/cmd_vel` | the same names, types and frames |
| robot geometry | stuttgart_scout URDF | the same macros, from intelligent-fleet-sim-env |
| left optical frame from `base_footprint` | (0.390, 0.060, 0.373) | (0.390, 0.060, 0.373) |
| floor in `base_footprint` | −0.025 (measured) | −0.0248 (wheel geometry) |
| ZED dynamic tf link | republished ~58 Hz by the SDK | republished 58 Hz by `camera_tf.py` |
| pallet | measured short face, modelled long face | the same, checked against `test/synthetic.py` |

## What is not

These are the limits of what a sim run shows. A run that passes here has
passed the ROS layer and the control loop against a plant. It has not shown
anything about perception on real depth.

- **Depth is clean.** Gaussian noise of 1 cm, constant with range (`depth_noise:=`).
  Real ZED depth scatters by a few cm at 2 m, grows with range, smears at
  edges and fills holes. Perception thresholds that pass here may not pass
  on the robot.
- **No camera tilt.** The real camera is pitched 0.698° and rolled −0.516°;
  the simulated one is mounted exactly as the URDF says, so the correction is
  off. The correction code is tested in `test_calibration.py`, not here.
- **The LiDAR is the URDF's.** 64 × 1024, −44.8° to +43.9°, laid out as
  measured, but at the URDF height, which is 41 mm above the measured one.
- **Skid steer is a four-wheel differential drive.** Turning scrubs the tyres,
  so the odometry yaw drifts during turns, as on the robot. How much it drifts
  is not calibrated to the real base.
- **Intrinsics are an ideal pinhole** at the measured field of view, with the
  principal point at the centre and no distortion.
- **No Nav2.** The robot spawns where Nav2 would have left it, and the run is
  started on the topic, as in the real flow.

## Prerequisites

The Gazebo image builds on the intelligent-fleet-sim-env image, which has
Gazebo Fortress, `ros_gz` and the robot's `ugv_descriptions` at the absolute
path the robot meshes are referenced by. Build it in that repo first:

```bash
docker compose build hunter-ugv-avp-sim          # in intelligent-fleet-sim-env
```

Nothing in that repo is modified. Our top-level xacro includes its macros
and `robot_model.py` replaces its Gazebo plugins.

Rendering (camera and LiDAR) uses `/dev/dri` if it exists. On this laptop's
AMD iGPU the sim runs at a real-time factor of about 0.75. Everything runs on
sim time, so slower means a longer wall-clock wait, not a different result.

## Problems found while building it

Each of these was measured, and each fix is commented where it lives.

1. **Fortress publishes the wrong `camera_info`.** The `rgbd_camera` reports a
   320×240 camera's intrinsics (fx 277) next to 1280×720 images, even with
   `<lens><intrinsics>` set. The overlay drew the forks flipped into the sky,
   and perception would have back-projected every depth pixel 3.4× too wide.
   `camera_info_fix.py` republishes it with the intrinsics the images are
   rendered with.
2. **Shared memory and users.** The base image runs as uid 1000 and our stack
   as root. FastDDS shared-memory segments are writable only by their owner,
   so discovery worked but no message arrived. Both containers run as root.
3. **Large best-effort messages vanish.** FastDDS's default 0.5 MB shm
   segment cannot hold a 2.7 MB image, so the image goes out as fragmented
   UDP. A RELIABLE subscriber recovers the fragments; a BEST_EFFORT one got
   0 frames in 20 s. `fastdds_sim.xml` raises the segment to 32 MB. The
   pipeline's own image subscribers are RELIABLE; the two video streamers are
   best effort on purpose. **The same thing may happen on the robot**: if
   the browser's `/zed` stream is blank while perception works, this is the
   first thing to check.
4. **The video streamer aborted on amd64.** It first crashed creating its node
   once NumPy and an initialised GStreamer were loaded; the node is now
   created first (`video_streamer.py`). It then aborted in libproxy
   ("Unable to read configuration") when GStreamer's RTSP client asked GIO
   for an HTTP proxy, fixed by `GIO_USE_PROXY_RESOLVER=dummy`.

## Files

```
sim/
  run_sim.sh                       the entry point
  Dockerfile.gazebo                FROM the sim-env image, plus pallet_approach_sim
  fastdds_sim.xml                  32 MB shm segments, UDP on loopback
  pallet_approach_sim/
    urdf/scout_sim.urdf.xacro      their macros, the robot's frame names
    pallet_approach_sim/
      robot_model.py               their plugins out, our sensors and drive in
      pallet_model.py              the pallet as timber, and where to put it
      camera_tf.py                 the ZED's 58 Hz tf link
      camera_info_fix.py           the intrinsics Fortress gets wrong
    worlds/pallet_lab.sdf          a floor and a light
    config/bridge.yaml             Gazebo topics -> the robot's topic names
    launch/sim.launch.py
    test/                          pallet vs synthetic.py; URDF processing
pallet_approach/config/sim_overlay.yaml   the stack's only sim-specific config
```
