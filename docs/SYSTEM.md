# System facts (measured, not assumed)

Source: `pallet_approach/tools/survey_ros2.sh` run inside the ZED container on the Jetson and the
scout container on the robot PC, 2026-10-13. Raw output in `docs/survey/`.
Every number here came from that run. Nothing in this file is a guess; where
something is still open it says so.

## Machines

| | Jetson | Robot PC |
|---|---|---|
| hostname | `ubuntu` | `scout-stuttgart` |
| arch | aarch64, Orin, L4T R36.5.0 | x86_64 |
| OS / Python | Ubuntu 22.04.5 / 3.10.12 | Ubuntu 22.04.5 / 3.10.12 |
| cores / RAM | 8 / 61 GiB | 12 / 15 GiB |
| disk free | **5.7 G of 57 G (90 % full)** | 225 G of 439 G |
| ROS | Humble, `rmw_fastrtps_cpp` | Humble, RMW unset (= fastrtps) |
| `ROS_DOMAIN_ID` | 10 | 10 |
| extra | `FASTRTPS_DEFAULT_PROFILES_FILE=/ros2_ws/src/scout_nlp/configs/fastdds_jetson.xml` | none |
| ZED SDK | 5.4.1 (`zed_wrapper` 5.4.1) | — |

Links between them: wired `192.168.1.1` (Jetson `eno1`) ↔ `192.168.1.102`
(robot PC `enp43s0`), and WiFi `10.42.0.40` ↔ `10.42.0.1` (the robot PC runs the AP).

### Reaching the Jetson from a laptop

Join the robot's hotspot (you get `10.42.0.x`) and ssh straight to
**`stuttgart_orin@192.168.1.1`**. No jump host: the robot PC forwards between its
AP subnet and the wired switch.

The Jetson answers on two addresses and they are not equivalent for moving data:

| route | wifi transmissions | use it for |
|---|---|---|
| `192.168.1.1` (wired `eno1`) | 1 — AP → laptop only | **everything**; the Jetson→robot-PC leg is wired gigabit |
| `10.42.0.40` (its own wifi) | 2 — Jetson → AP → laptop | fallback if the robot PC stops forwarding |

Both legs of the `10.42.0.40` path share one radio, so the two transmissions
compete for airtime and roughly halve the usable throughput. Prefer the wired
address; `pallet_approach/tools/fetch_bag.sh` defaults to it.

**Multi-machine discovery works in both directions.** The two topic lists are
identical: the Jetson sees `/odom`, `/tf`, `/tf_static`, `/scout_status`, `/map`,
`/ouster/*`; the robot PC sees every `/zed2/*` topic. No extra configuration needed.

## Topics

| Role | Topic | Type | QoS | Rate |
|---|---|---|---|---|
| depth | `/zed2/zed_node/depth/depth_registered` | `sensor_msgs/Image` | RELIABLE, VOLATILE | ~37 Hz (2026-10-13); 44-56 Hz measured 2026-10-15 with `grab_frame_rate: 60` |
| depth camera info | `/zed2/zed_node/depth/camera_info` | `sensor_msgs/CameraInfo` | RELIABLE, VOLATILE | — |
| rectified left image | `/zed2/zed_node/rgb/color/rect/image` | `sensor_msgs/Image` | RELIABLE, VOLATILE | ~37 Hz |
| its camera info | `/zed2/zed_node/rgb/color/rect/camera_info` | `sensor_msgs/CameraInfo` | RELIABLE, VOLATILE | — |
| point cloud | `/zed2/zed_node/point_cloud/cloud_registered` | `sensor_msgs/PointCloud2` | — | — |
| wheel odometry | `/odom` | `nav_msgs/Odometry` | RELIABLE, VOLATILE | 50.0 Hz (std dev 0.2 ms) |
| velocity command | `/cmd_vel` | **`geometry_msgs/Twist`** (not stamped) | RELIABLE, VOLATILE | — |
| LiDAR | `/ouster/points` | `sensor_msgs/PointCloud2` | **BEST_EFFORT** | ~10 Hz |
| ZED visual odometry | `/zed2/zed_node/odom` | `nav_msgs/Odometry` | — | 58 Hz |

On the ZED wrapper, `rgb` **is** the left sensor; there is no `left/` topic. The
rectified left image and the depth image share one frame and one set of intrinsics.

Every ZED topic, `/odom`, `/tf` and `/cmd_vel` publish RELIABLE/VOLATILE, so
default rclpy QoS is compatible with those.

**`/ouster/points` is the exception: BEST_EFFORT.** Measured 2026-10-15 by
`tools/clock_audit.py`, which asked for RELIABLE and got

    New publisher discovered on topic '/ouster/points', offering incompatible
    QoS. No messages will be received from it. Last incompatible policy:
    RELIABILITY

A RELIABLE subscriber receives nothing at all from a BEST_EFFORT publisher --
not degraded delivery, none. So **the LiDAR subscription in the tracker (stage
2) and the insertion stop (stage 6) must be BEST_EFFORT**, unlike every other
subscription in this stack. `ros2 bag record` adapts automatically, which is
why the recorded bags have LiDAR in them and this went unnoticed until a
hand-written subscriber asked for the wrong thing.

The safe default for a diagnostic subscriber is BEST_EFFORT: it is compatible
with publishers of either kind, where RELIABLE is compatible with only one.

### `/cmd_vel` ownership

`Publisher count: 0`, one subscriber: node `scout` (the base driver). No teleop,
no joystick, no `twist_mux`, no Nav2 node is running. **Our controller will be the
only publisher**, as planned — no multiplexer needed. The startup check should
assert publisher count is 0 before activating.

Nav2 is not running at all. `slam_toolbox` *is* running and publishes `map → odom`
at 10 Hz. Driving targets stay in `odom` (never `map`, which jumps on loop closure).

### Base driver

`ros2 param dump /scout`: `base_frame: base_link`, `odom_frame: odom`,
`control_rate: 50`, `port_name: can0`, `use_sim_time: false`.

**There is no command-timeout or watchdog parameter.** Nothing to configure and
nothing to rely on, which settles the design: the controller publishes at a fixed
rate while active and sends an explicit zero `Twist` on every stop and in every
terminal state. It never assumes the base will stop itself.

A physical RC transmitter exists (`/rc_status`) and reaches the base over CAN,
outside `/cmd_vel`. It is an override path, not something we publish to.

## Frames

Live tf tree (`view_frames`):

```
map ──10 Hz──▶ odom ──50 Hz──▶ base_link ──▶ top_robot_link ──▶ bottom_box_link
                                     │                               ├──▶ top_box_link ──▶ … ──▶ os_lidar
                                     │                               ├──▶ imu_link
                                     ├──▶ base_footprint             └──▶ zed2_camera_link
                                     ├──▶ front_mount / rear_mount        └──▶ zed2_camera_center
                                     └──▶ inertial_link                        ├──58 Hz──▶ zed2_left_camera_frame
                                                                               │            └──▶ zed2_left_camera_frame_optical
                                                                               ├──58 Hz──▶ zed2_right_camera_frame
                                                                               └──▶ zed2_imu_link
```

The depth and left-image headers are stamped **`zed2_left_camera_frame_optical`**.
That frame is in the robot's URDF (`stuttgart.urdf.xacro`, camera named `zed2`),
so `base_link` → camera is a rigid chain, not an odometry composition.

Measured with `tf2_echo base_link zed2_left_camera_frame_optical` (see the
next section for the `base_footprint` version, which is the one the code uses):

```
Translation: [0.390, 0.060, 0.163]      Rotation RPY: [-90°, 0°, -90°]

  0  0  1   0.390        optical z (forward) → base x
 -1  0  0   0.060        optical x (right)   → base -y
  0 -1  0   0.163        optical y (down)    → base -z
  0  0  0   1
```

It reproduces exactly from the URDF joint chain: `0.113` (base→top_robot) +
`0.10` (→bottom_box) + `0.30` (→camera_link) + `0.015` (→camera_center) + `-0.01`
(→left_camera_frame) = `0.390` in x; `0.060` in y is half the 0.12 m stereo
baseline, so the left lens sits 6 cm left of the robot centreline. **Camera pitch
is exactly zero** — it looks straight ahead.

Two caveats:

- `zed2_camera_center → zed2_left_camera_frame` is republished at **58 Hz**, not
  static (the SDK's continuous self-calibration). So tf lookups must use the image
  timestamp with a short timeout, never `Time()`.
- `/zed2/zed_node/odom` is published with `frame_id: odom`, `child_frame_id:
  zed2_camera_link` — the ZED's visual odometry claims the same `odom` frame name
  the wheel odometry uses. It is **topic-only**; the ZED broadcasts no tf, so the
  tree is clean. Do not enable the wrapper's `publish_tf`: it would give
  `zed2_camera_link` a second parent and break the chain above.

### The working frame is `base_footprint`, not `base_link`

`base_link` is the chassis body frame. The wheels mount at `base_link` level
(axles at z = −0.0702 m, radius 0.16459 m), which lifts the body off the ground,
so `base_link` sits 0.21 m up. `base_footprint` is the robot's ground frame,
directly below it — and **the floor is at `base_footprint`, by definition**.

So perception works in `base_footprint`, where every height is a small number
close to what it physically is, instead of a large negative offset:

| | height above floor | `base_footprint` z | `base_link` z |
|---|---|---|---|
| floor | 0.000 | **−0.025** | −0.235 |
| pocket bottom lip | 0.022 | **−0.003** | −0.213 |
| pocket centre | 0.072 | **0.047** | −0.163 |
| fork tine underside | 0.052 | **0.027** | −0.183 |
| pallet top deck | 0.144 | **0.119** | −0.091 |
| camera optical centre | 0.398 | **0.373** | 0.163 |

Note the middle column is *not* the first: `base_footprint` is 25 mm above the
floor, measured (see below). Which is why nothing in the pipeline treats z = 0
as the floor — `remove_floor` and `find_pockets` both take the fitted floor
plane as an argument, so every height downstream is relative to a measurement.

The URDF joint is a pure translation (`xyz "0 0 -0.21"`, `rpy "0 0 0"`), so the
two frames share x, y and yaw **exactly**. Only z differs. That is what makes
the arrangement safe: perception reports in `base_footprint`, `/odom` reports
`base_link`, and the coordinator composes planar targets from one with the pose
of the other without any conversion — not by luck, but because the joint has no
rotation and no x/y offset.

Measured directly: `tf2_echo base_footprint zed2_left_camera_frame_optical` is
`[0.390, 0.060, 0.373]`, rotation RPY `[-90°, 0°, -90°]`, pitch exactly zero.

### How far down is the floor, really? — measured 2026-10-14

Tape-measured: the **left lens sits 0.398 m above the floor**. The URDF puts the
optical centre 0.163 m above `base_link`, so `base_link` is 0.235 m above the
floor. Against the two candidates:

| source | `base_link` height | error |
|---|---|---|
| wheel geometry (radius 0.16459, axle at −0.0702) | 0.2348 m | **−0.2 mm** |
| `base_footprint` joint (`xyz "0 0 -0.21"`) | 0.210 m | **−25 mm** |

**The wheel geometry is right and `base_footprint` is 25 mm out.** So
`base_footprint` is not on the floor: the floor is at **z = −0.025** in that
frame. That is `floor.expected_z` in `config/params.yaml`.

This changes nothing structural, because the pipeline never assumed otherwise —
the floor is fitted per frame and every height is taken relative to the fit. It
appears as an absolute number in exactly four places:

- `pallet_perception.floor.expected_z` — where to expect the fit, with
  `max_z_error: 0.08` to cover tyre compression, load and an uneven floor;
- `pallet_perception.roi.z_min: -0.05` — must reach below the floor, or RANSAC
  has no floor to fit;
- `fork_overlay.floor_z` — the fallback when no fit is available, which is the
  case at the standoff;
- the standoff derivation below.

It is also a calibration result in its own right: the URDF's own `base_footprint`
disagrees with its own wheel geometry by 25 mm, and the tape measure says which
one to believe.

## Extrinsic calibration: the camera is mis-mounted by 0.87°

Measured 2026-10-14 from four bags on bare floor — camera and LiDAR, at two
robot headings 173° apart (recovered from `/odom`, so the turn needed no
measuring). Reproduce with:

```bash
python3 pallet_approach/tools/floor_plane_check.py ~/bags/floor_{cam,lidar}_pose{1,2}
```

| bag | sensor | pitch | roll | height | RMS |
|---|---|---|---|---|---|
| floor_cam_pose1 | camera | +0.529° | −0.635° | −0.0263 m | 2.4 mm |
| floor_cam_pose2 | camera | +0.881° | −0.419° | −0.0230 m | 2.2 mm |
| floor_lidar_pose1 | LiDAR | −0.142° | −0.411° | +0.0127 m | 29 mm |
| floor_lidar_pose2 | LiDAR | +0.145° | +0.090° | +0.0133 m | 26 mm |

The floor gradient measured in `base_footprint` is the sum of a **mounting**
error, fixed to the robot, and the **floor's own slope**, fixed to the world and
therefore rotating with the robot's heading. Two headings separate them:

| | mounting tilt (robot-fixed) | floor slope (world-fixed) | floor height |
|---|---|---|---|
| camera | **0.868°** — pitch +0.698°, roll −0.516° | 0.207° | −0.0246 m |
| LiDAR | 0.152° — pitch −0.014°, roll −0.151° | 0.289° | +0.0130 m |

Three conclusions:

1. **The floor is level**, to about 0.2–0.3° (0.4 %). Both sensors agree on that
   within the measurement's own uncertainty, which is what the agreement bounds
   at roughly ±0.1°. It is not the explanation.
2. **The camera's URDF orientation is wrong by 0.868°**, mostly pitch: the camera
   is pitched down 0.70° more than the URDF claims, with 0.52° of roll. The
   LiDAR, on the same `bottom_box_link`, comes out level — so it is not the
   sensor box that is tilted, it is the camera on it. This is the number to
   calibrate out.
3. **The camera's mounting height is right.** Both poses put the floor at
   −0.0246 m, 1.7 mm from the −0.025 m that the 0.398 m tape-measured lens
   height implies. Two independent methods agreeing to under 2 mm.

### Why it has to be corrected

0.868° is a vertical error that grows with range:

| range | vertical error |
|---|---|
| 1.45 m (the standoff) | 22 mm |
| 2.0 m | 30 mm |
| 3.0 m | 45 mm |

The pocket opening is 100 mm and `pocket_height_tolerance` is 30 mm, so at the
standoff the error already eats three-quarters of the budget and at detection
range it exceeds it. Pocket geometry cannot be measured to tolerance until this
is corrected.

### The correction, and what it leaves behind

`frames.extrinsic_correction` in `config/params.yaml` holds the two measured
numbers; `pallet_approach/calibration.py` derives the levelling rotation and
both nodes apply it to the transform they look up from tf. Verified on the same
four bags:

```bash
python3 pallet_approach/tools/floor_plane_check.py --correct ~/bags/floor_*
```

| | before | after |
|---|---|---|
| camera mounting tilt | **0.868°** | **0.190°** |
| vertical error at 2 m | 30 mm | 6.6 mm |
| camera floor height | −0.0246 m | −0.0248 m |
| LiDAR mounting tilt (untouched) | 0.152° | 0.152° |

The residual 0.190° is the same size as the LiDAR's own 0.152°, and the two
sensors' estimates of the world-fixed floor slope differ by 0.1° — so we are at
the noise floor of what one session's bags can resolve, and correcting further
would be fitting noise.

Two caveats worth keeping in mind:

- The correction rotates the camera's **orientation only**; the translation is
  left alone, because the height was independently confirmed by the tape
  measure. Multiplying the whole 4×4 would have swung the camera's position by
  8 mm to fix its angle.
- A ground plane constrains exactly three of the six extrinsic degrees of
  freedom — the two tilts and the height. **Yaw, x and y are untouched**,
  because sliding or spinning a sensor along a plane does not change what the
  plane looks like. Those need an object at a taped position, which is the
  other half of the Day 6 task and still to do.

The quoted floor *height* is an extrapolation from an annulus 1–4 m out back to
the robot's own axis, so residual tilt is amplified into it: the per-pose spread
grew from 3.3 mm to 15.5 mm after correction even though the mean barely moved.
The mean is the number to trust.

### A separate finding: the LiDAR's height is 38 mm out

The LiDAR puts the floor at +0.0130 m where the camera puts it at −0.0246 m —
a 38 mm disagreement, consistent to 0.5 mm across both poses, so it is a
systematic error in the URDF's LiDAR mounting chain, not noise. Its
*orientation* is true. We do not depend on the LiDAR, but it is worth reporting
to whoever maintains the robot description, and it is a reminder that the
URDF's accessory mounts are nominal rather than measured — as `base_footprint`
being 25 mm out already showed.

## Intrinsics (rectified left, 1280 × 720)

```
fx = fy = 949.6732177734375
cx = 628.338134765625
cy = 369.74566650390625
D  = [0, 0, 0, 0, 0, 0, 0, 0]       R = I       P = [K | 0]
```

D is all zeros and R is identity because the image is already rectified, so
back-projection is the plain pinhole form with no undistortion step:

```
Z = depth[v, u]        X = (u - cx) · Z / fx        Y = (v - cy) · Z / fy
```

Depth encoding: **`32FC1`**, 1280 × 720, step 5120 — metres as float32, one channel.
Invalid pixels are NaN/±Inf (confirm on real data).

Field of view: horizontal 2·atan(640/949.67) = **68.0°**, vertical
2·atan(360/949.67) = **41.5°**.

### Derived: how close the camera can get before the pockets fall out of frame

The camera is 0.390 m ahead of `base_footprint` and **0.398 m above the floor**
(measured), pitched 0.70° nose-down against the URDF's zero (measured — see the
extrinsic section). Features low on the pallet fall out of the *bottom* of the
image as the robot closes in, and the bottom of the image is the row at
`cy + 719`, not the symmetric half-height: `atan((719 − 369.75) / 949.67)` =
**20.19°** below the optical axis, 20.89° below horizontal once the real pitch
is included. (An earlier version of this table used 20.76°, the symmetric
half-angle. That is the *top* edge; using it for the bottom made the camera look
7 mm more capable than it is.)

| feature on the pallet | height above floor | falls out of frame when the face is closer than |
|---|---|---|
| top deck edge | 0.145 m | 1.05 m |
| **pocket top edge** | 0.100 m | **1.17 m** |
| pocket centre | 0.050 m | 1.30 m |
| floor contact line | 0.000 m | 1.43 m |

**What the measured pallet changed here.** The old version of this table was
built on the EPAL nominal, where the pocket opening starts at a 22 mm bottom
lip, and argued that the lip had to stay in frame or the measured opening would
read short against a 0.030 m tolerance — which is what drove the standoff to
1.45 m. The real pallet has **no bottom lip on the pocket face**: the opening
runs from the floor to 0.100 m. So there is no lip to keep in frame, and the
opening's lower bound is the floor plane, which perception already fits from the
whole visible floor rather than from a strip at the face. The camera's
requirement collapses to **the pocket's top edge in frame, so the face must be
at 1.17 m or further** — plus margin, because pitch moves this limit by about
40 mm per degree. 0.10 m of margin covers 2.5° of pitch under braking or a
tyre-pressure change, giving **≈ 1.27 m** as the camera-side floor on the
standoff.

The camera is therefore no longer the binding constraint. The fork reach is
(below, and stage 5): the virtual tines must start outside the pallet at the
standoff, which puts it further out than 1.27 m for any realistic tine length.

**Consequence for floor fitting:** at any standoff below 1.43 m there is no
floor in frame at all near the pallet, so the floor estimate has to be carried
forward from the detection phase or fall back to `floor.expected_z`. The
fallback is only meaningful because that number has been measured.

## LiDAR coverage: a 0.83 m forward blind cone — measured 2026-10-14

The Ouster is an **OS0-070-64U-AX**: 64 beams over a ~90° vertical field, so it
looks almost straight down as well as out. That is the reason it can be the
insertion stop sensor at all. But the beams leave at fixed angles from a sensor
0.61 m up, so there is a cone in front of the robot that no beam enters, and
that cone is what limits how far the insertion can go under sensor supervision.

Measured from the two bare-floor LiDAR bags, no datasheet values used:

```bash
python3 pallet_approach/tools/lidar_coverage.py \
    ~/bags/floor_lidar_pose1 ~/bags/floor_lidar_pose2
```

| | measured |
|---|---|
| beams × azimuth columns | 64 × 1024 |
| vertical field of view | 88.7° / 89.1° (−44.8° … +43.9°) |
| beam spacing | 1.408° mean (1.19–1.95) — uniform, as the `U` in the part number says |
| axis position | x = +0.173 m in `base_footprint` |
| beam origin above the floor | **0.6121 m** (±1.8 mm between the two bags) |
| same height from the URDF chain | 0.6549 m — **41 mm out**, the error noted below |
| lowest beam usable straight ahead | row 62, **−43.6°** |
| **forward blind radius on the floor** | **0.832 m** from `base_footprint` (±1 mm) |

Two details worth keeping:

- **The lowest beam of all (row 63, −44.8°) sees nothing straight ahead.** It
  returns normally to the sides and behind, and in the ±10° sector it produced
  0 and 1 points across the two bags where every neighbouring beam produced
  55–72. The ZED sits at x = 0.390, 0.22 m below and ahead of the LiDAR, right
  in that beam's path. So the usable limit is row 62, not the nominal FOV edge:
  **self-occlusion, measured, not assumed.** Any geometry taken from the
  datasheet's ±45° would be one beam optimistic.
- **The height had to be fitted, not read.** The URDF chain to `os_lidar`
  overstates the sensor height by 41 mm (consistent with the 38 mm found from
  the floor fits), so the tool solves for it from where the eight steepest beams
  actually land on the bare floor. Using the URDF number would have moved every
  distance below by about 40 mm in the optimistic direction.

### What this means for the pallet

With those measured angles, the beams that cross the 0.145 m front face as the
robot closes in:

| face at x | bumper gap | beams on the face | of those, inside the pocket opening |
|---|---|---|---|
| 2.50 m | 2.04 m | 2 | 2 |
| 2.00 m | 1.54 m | 3 | 2 |
| 1.70 m | 1.24 m | 3 | 2 |
| 1.45 m | 0.99 m | 4 | 3 |
| 1.20 m | 0.74 m | 5 | 3 |
| 0.90 m | 0.44 m | 6 | 4 |
| 0.75 m | 0.29 m | 3 | 2 |
| **0.70 m** | **0.24 m** | **2** | 0 |
| 0.67 m | 0.20 m | 1 | 0 |
| 0.60 m | 0.14 m | **0** | 0 |

"bumper gap" is the clearance from the front face of the `base_link` collision
box (0.925 × 0.38 × 0.21, so 0.4625 m ahead of `base_footprint`) to the pallet.
That box is the real robot; the forks are imaginary.

Three conclusions, all of them things the plan depended on and none of them
previously known:

1. **The cross-check works, and it works on the pockets too.** At the standoff
   there are 3–4 beams on the face with 2–3 of them inside the 0–0.100 m pocket
   opening, where a beam either hits a block or flies into a pocket. At 1024
   columns over 360° the azimuth step is 8.9 mm at 1.45 m, so a 0.23 m pocket is
   26 columns wide and the 0.795 m face is 89. That is an independent
   measurement of the face plane *and* of the gap pattern — a genuine
   second-sensor check of pocket positions, not just of range.
2. **The LiDAR keeps the face in view down to a 0.24 m bumper gap** (2 beams) or
   0.20 m (1 beam), then loses it completely. So the insertion stop can be
   placed where the LiDAR still sees the face, and insertion need not be blind
   to it — which is a stronger version of D7 than D7 assumes. Below a 0.20 m
   gap nothing is observable and odometry is the only thing left.
3. **Pocket-level LiDAR returns fade below a 0.8 m face distance**, because the
   surviving beams are the ones that cross the face above 0.100 m. Range and
   face plane survive to 0.67 m; the gap pattern does not.

## Python environment inside the ZED container

| module | state |
|---|---|
| `numpy` | 2.2.6 |
| `yaml` | 5.4.1 |
| `cv2` | installed but **ImportError** |
| `cv_bridge` | imports, but its compiled module fails the same way |
| `scipy`, `sklearn`, `matplotlib`, `rosbags`, `transforms3d`, `numba`, `pandas` | missing |

`cv2` and `cv_bridge` were built against NumPy 1.x and the container has NumPy
2.2.6: `AttributeError: _ARRAY_API not found`. **Pinning `numpy<2` in our own
image fixes both.** We derive an image from `stereolabs/zed:5.4.1-devel-l4t-r36.5`
and install what we need there, so the ZED container stays untouched.

PyPI is reachable from inside the container.

### Open3D: available, but not used on the robot

`pip index versions open3d` on the Jetson lists 0.20.0, 0.18.0, 0.17.0, 0.16.0 —
so an aarch64 / cp310 wheel exists and it would install. We are **not** installing
it on the Jetson anyway:

- the Jetson has 5.7 G free and Open3D unpacks to several hundred MB;
- `numpy` and `scikit-learn` cover everything the pipeline needs (plane fit by
  RANSAC, DBSCAN clustering) at a fraction of the size;
- writing the plane fit directly is better interview material than calling
  `segment_plane`.

Open3D stays on the laptop for point-cloud visualisation during offline work.

## Recording bags: a real constraint

Raw depth is 1280 × 720 × 4 B = 3.69 MB per frame at ~37 Hz = **136 MB/s**. The
rectified colour image adds ~102 MB/s. A 20-second bag of both is ~4.8 GB, against
5.7 GB free on the Jetson. **Unthrottled recording on the Jetson is not viable.**

Options, cheapest first:

1. Throttle to 5–10 Hz with `topic_tools throttle` before recording. A pallet
   sequence does not need 37 Hz. 10 s at 10 Hz of depth + image ≈ 650 MB.
2. Record to an external USB3 SSD on the Jetson — the right fix for the Day 7
   static dataset and the Day 11 trials.
3. Record the ZED's own SVO file (`/zed2/zed_node/start_svo_rec`), which the SDK
   compresses, plus a small bag of `/tf`, `/tf_static`, `/odom`. Best quality per
   byte, and replaying the SVO through the wrapper regenerates depth exactly.
4. Record on the robot PC (225 G free) — but 240 MB/s has to cross the link first,
   so this only works together with (1).

`pallet_approach/tools/record_bag.sh` implements (1) and prints the size estimate before starting.
**Open: is a USB3 SSD available for the Jetson?** Everything else is settled.

## Still open

- USB3 SSD for the Jetson (see above). Blocks the Day 7 dataset, not Day 1.
- **The pallet's long side.** The 0.795 m pocket face is measured; the 1.195 m
  side is not. Side classification (stage 1) needs to know whether it has
  openings and how wide, or it has nothing to tell the two sides apart by.
- Which container hosts our stack: a new image from `docker/Dockerfile.jetson`.
  `docker build` on the Jetson is available (confirmed), disk headroom is not.
- **No bag contains the pallet and the LiDAR together.** `pallet_2m_0deg` has
  depth, colour, tf and odometry but no `/ouster/points`, and the LiDAR bags are
  of bare floor. The coverage numbers above are geometry from measured beam
  angles; the cross-check itself (does the gap pattern actually appear in the
  LiDAR returns?) has not been seen in data yet.

Closed since the first survey:

- ~~Measured dimensions of the actual pallet~~ — done 2026-10-14, D4.
- ~~Can the Ouster's lowest beams see a 14.5 cm pallet at close range?~~ —
  measured, see the coverage section: yes, down to a 0.20 m bumper gap, and the
  forward blind radius on bare floor is 0.832 m.
- ~~Does Nav2 publish on `/cmd_vel` while idle?~~ — no. It publishes only while
  a navigation goal is active (confirmed by Athreya on the robot, 2026-10-14),
  so `twist_mux` needs no special handling for an idle Nav2 stack.

---

## Velocity arbitration (added 2026-10-19)

Nothing in this stack publishes to `/cmd_vel` any more. `twist_mux` does, and
everything else drives one of its inputs:

| topic | priority | publisher |
|---|---|---|
| `/cmd_vel_safety` | 200 | `watchdog`, zeros only |
| `/cmd_vel_pallet` | 100 | `approach_controller` |
| `/cmd_vel_nav` | 10 | Nav2 |
| `/pallet_approach/nav_lock` | lock at 50 | `coordinator`, for the length of a run |

Config in `pallet_approach/config/twist_mux.yaml`; the launch file remaps the
mux's `cmd_vel_out` onto `/cmd_vel`. Every input has a 0.5 s timeout, because
`ros2 param dump /scout` shows the base has no command timeout of its own.

**Unverified on the robot.** The schema is the documented Humble one and
`test_arbitration.py` checks its shape against `params.yaml`, but nothing in
this repository runs `twist_mux`. Confirm in the lab that the mux reads this
file and relays `/cmd_vel_pallet` to `/cmd_vel`.

**What the mux is for under the sequential flow (D55).** Nav2 finishes before
the pipeline is triggered, so the priority between `/cmd_vel_pallet` and
`/cmd_vel_nav` never decides anything. What the mux still does is let the
watchdog's `/cmd_vel_safety` outrank our own controller — the mechanism by
which a wedged coordinator still leaves a stationary robot — and insure against
Nav2 publishing when nobody expected it to.

**The nav lock is inert until Nav2 is remapped to `/cmd_vel_nav`.** Nav2
currently publishes straight to `/cmd_vel`, which is the mux's *output*, so the
lock blocks an input nothing is using. Under the sequential flow that is
acceptable; it is recorded here because an inert safety mechanism that looks
active is worse than an absent one.

The controller refuses to configure if nothing subscribes to its output topic,
so the failure mode "twist_mux is not running and the robot silently never
moves" produces a loud refusal instead.

## Starting a run

Every node is up and waiting from the moment the container starts. A run begins
on a topic, because Nav2 has finished by then and the trigger comes from a
script with nothing to wait for (D55):

```bash
ros2 topic pub --once /pallet_approach/start  std_msgs/String "{data: pickup}"
ros2 topic pub --once /pallet_approach/start  std_msgs/String "{data: approach_only}"
ros2 topic pub --once /pallet_approach/cancel std_msgs/Empty  "{}"
ros2 topic echo /pallet_approach/result
```

`approach_only` runs the whole approach and stops at the GO/NO-GO decision,
inserting nothing. An unknown mode is refused, not defaulted — the difference
between the modes is whether the robot drives into the pallet. A run is also
refused if the stack is unhealthy when the trigger arrives, and the refusal is
published on `/pallet_approach/result` as well as logged, because nothing is
waiting on a return value.

The action server (`/pallet_task`) does the same thing with live feedback and
is the better way to run one by hand.

## Topics this stack publishes

| topic | type | rate | notes |
|---|---|---|---|
| `/pallet_perception/estimate` | `PalletEstimate` | 5 Hz | one frame's answer |
| `/pallet_perception/cluster_points` | `PointCloud2` | 5 Hz | the pallet cluster; **Jetson only** |
| `/pallet_perception/scene_points` | `PointCloud2` | 5 Hz | a box around the face; **Jetson only** |
| `/pallet_perception/diagnostics` | `DiagnosticArray` | 1 Hz | timing, achieved rate, drops |
| `/pallet_tracker/tracked` | `TrackedPallet` | 5 Hz | the track + the LiDAR cross-check |
| `/approach_controller/status` | `DriveStatus` | 20 Hz | what the controller sees and decided |
| `/virtual_forks/state` | `ForkState` | 10 Hz | carriage height, loaded |
| `/health` | `Health` | 2 Hz | sensor ages and rates |
| `/diagnostics` | `DiagnosticArray` | 2 Hz | the same, for rqt |
| `/coordinator/state` | `String` | on change | the sequence state alone, for fork_overlay's colour |
| `/coordinator/status` | `CoordinatorStatus` | 20 Hz | state, mode, side, **live reason**, outcome, elapsed (D57) |
| `/pallet_approach/result` | `PalletTask_Result` | per run | the outcome, for a topic-started run |
| `/fork_overlay/image` | `Image` | 5 Hz, 640×360 | forks + pockets drawn on the camera frame |
| `/dashboard/image` | `Image` | 5 Hz, 640×520 | the overlay image plus a HUD of everything else (D57) |

**None of the images or point clouds above may be subscribed to from the main
PC as ROS topics.** Measured 2026-10-13: one subscriber to a raw ZED image
topic from the other machine dropped the whole ZED node to about 0.2 Hz. The
two point clouds are purpose-built and small (a few thousand points each), and
exist because two GO/NO-GO rules take points rather than an estimate and treat
missing data as a NO-GO; they travel between two nodes a process apart on the
Jetson. Both images are downscaled and decimated for the same reason.

**Watching `/dashboard/image` from another machine is still fine** -- it goes
out over MediaMTX/WebRTC (`video_stream.launch.py`), not as a ROS topic
subscription. `dashboard_node` subscribes to `/fork_overlay/image` and the five
telemetry topics above LOCALLY, composites a HUD panel onto the camera frame
(health, the tracker's confidence and LiDAR cross-check, what the controller is
driving and measuring, the forks, and the live reason), and republishes the
result; `video_streamer` then re-encodes THAT to H.264 and hands it to
MediaMTX. The browser never touches a ROS topic at all. See
`pallet_approach/dashboard.py` for the HUD layout and `video_stream.py` for the
encoding pipeline.

    http://<jetson-ip>:8889/dashboard   everything: forks, pockets, health,
                                         tracker, controller, forks, why
    http://<jetson-ip>:8889/zed         the raw camera, for checking the
                                         camera itself

Every value on the dashboard is shown as of when it arrived, not assumed
current: a field whose source has never published reads `--` in grey, and one
older than its configured `stale_after` limit fades to grey with its age
rather than sitting there looking live. The camera frame itself gets the same
treatment -- a red border and an age tag if `/fork_overlay/image` has stopped
updating, because a frozen picture is the one piece of stale data that looks
most like it is still live.

## What has and has not been tested on the robot

As of 2026-10-19, **everything from the tracker onwards has been tested only in
simulation.** `tools/sequence.py` runs a whole virtual pickup through the real
state machine, the real tracker, the real control laws, the real fork simulator
and the real safety rules, and it completes. What it does not touch:

- discovery, QoS and the multi-homed FastDDS setup;
- lifecycle transitions and the services behind them;
- tf, including the ZED's self-calibration republishing on the camera's clock;
- the clock itself;
- the real base's response to a velocity command — no actuator lag, no slip.

Every failure on this robot so far has come from that list. The simulated
numbers below are therefore lower bounds, not predictions.

| quantity | simulated | measured on the robot |
|---|---|---|
| lateral error at the standoff | ≤ 17 mm worst case | — (lab session B) |
| yaw error at the standoff | ≤ 2.4° worst case | — |
| run time, 3.5 m start to backed out | 68 s | — |
| per-frame spread of the face position | assumed 5 mm | — |
| LiDAR returns on the face | assumed ≥ 20 | — |
