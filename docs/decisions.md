# Design decisions

One entry per behaviour decision: what was decided, why, when, and which
stage it belongs to. Athreya decides; Claude records and implements.

## 2026-10-14

**D1. Working model.** Athreya decides the logic of every stage; Claude writes
all the code, after presenting a logic sheet with recommended defaults.
*Why:* faster delivery of the full pickup, with the design reasoning owned and
documented by Athreya. (Stage: all)

**D2. Scope: full virtual pickup.** After GO: lower forks to pocket height,
insert, lift, back out, hand control back to Nav2. *Why:* mirrors a real
reach-truck pickup end to end. (ROADMAP)

**D3. Nav2 hands over instead of being disabled.** Nav2 drives to the pallet
and takes over afterwards; `twist_mux` arbitrates `/cmd_vel` (safety stop >
pallet controller > Nav2). *Why:* realistic deployment; general navigation and
precise docking are separate jobs. (Stage 0)

**D4. Measured pallet replaces EPAL nominals.** 119.5 × 79.5 × 14.5 cm; pocket
face block 10 / pocket 23 / block 13.5 / pocket 23 / block 10; pockets 23 × 10,
opening from the floor to 0.100 m; centre height ≈ 0.050 m; centres 0.365 m
apart. *Why:* the first real bag showed the nominal pocket band overlapping the
board edge at 0.100 m. (Stage 1)

**D5. Pocket finding on real data.** Band 0.035–0.075 m; ±6 cm slice of pallet
points around the face plane instead of plane inliers; at least 2 points per
bin to count as wood. *Why:* first real bag found only one pocket: band too
high, depth noise of a few cm at 2 m, single stray points splitting a pocket.
(Stage 1)

**D6. Two-leg approach.** Coarse leg to the line-up point on odometry (losing
sight expected); re-acquire at the line-up point; final leg on perception
(losing sight aborts). *Why:* the pallet leaves the camera's view while the
robot turns. (Stage 3)

**D7. Blind insertion on odometry with a LiDAR stop, robot body stays clear.**
*Why:* the pockets leave the camera's view about 0.9 m ahead of the robot
centre; the forks are imaginary but the robot is not. Exact distances: Stage 5.
(Stages 5–6)

**D8. Speed limits for autonomous runs.** 0.20 m/s and 0.30 rad/s normally;
0.10 m/s inside the final leg (closer than the line-up point); 0.05 m/s during
insertion and back-out. *Why:* the lab limit was already 0.20 m/s; each stage
closer to the pallet halves the energy at impact, and insertion is the one phase
where the robot moves toward an obstacle it cannot fully see. Defaults accepted
without change. (Stage 0)

**D9. Watchdog staleness limits.** Stop if depth is older than 0.30 s,
a perception estimate older than 0.50 s, odometry older than 0.20 s, or a tf
lookup older than 0.30 s. *Why:* each is roughly five missed messages at the
measured rate (depth ~37 Hz, odometry 50 Hz, perception target ~10 Hz) — long
enough to ride out a hiccup, short enough that at 0.20 m/s the robot travels
under 60 mm before stopping. Defaults accepted without change. (Stage 0)

**D10. `twist_mux` arbitration.** Priorities: safety stop > pallet controller >
Nav2. Each input gets a timeout so a crashed publisher stops driving.
*Why:* D3. **Measured 2026-10-14:** Nav2 publishes on `/cmd_vel` only while a
navigation goal is active, and nothing at all while idle, so no velocity
smoother or collision monitor is sitting on the topic between runs and the mux
needs no special case for an idle Nav2. (Stage 0)

**D11. Run recording: leave as it is for now.** `tools/record_bag.sh`'s 10 Hz
throttle and disk guard stand; the per-run recorder node and what it records are
deferred to the stage that builds it. *Why:* the Jetson has 5.7 GB free and the
policy should be chosen with the real bag sizes in hand, not before. (Stage 0,
revisited at its own stage)

**D12. Health check at goal time.** The coordinator refuses a goal if sensor
rates, perception time or tf freshness are outside limits, and reports which one
failed. *Why:* a run that starts unhealthy fails later and more expensively,
next to a pallet. Default accepted without change. (Stage 0)

**D13. LiDAR coverage measured; the insertion stop is bounded by it.** The
Ouster (OS0-070-64U-AX, 64 beams, 88.7° measured vertical FOV) has a **0.832 m
forward blind radius** on the floor, its lowest forward-usable beam is −43.6°
(the beam below that is blocked by the ZED), and it keeps 2 beams on a 0.145 m
pallet face down to a **0.24 m bumper gap**, 1 beam to 0.20 m, nothing below.
It also resolves the pocket gap pattern at the standoff (26 azimuth columns
across a 0.23 m pocket at 1.45 m). *Why it matters:* it answers the two open
questions this plan was waiting on — the LiDAR *can* be the insertion stop
sensor and *can* cross-check the pockets — and it fixes the limit that the stop
distance, and therefore the insertion depth, has to respect. Reproduce with
`tools/lidar_coverage.py`; numbers in `docs/SYSTEM.md`. (Stages 2, 5, 6)

**D14. Build order: fork and insertion geometry moves before the approach.**
ROADMAP stage 3 listed its standoff as coming "from fork geometry, stage 5",
which cannot be built in that order. The geometry decisions are split out as
**stage 5a** and taken first; the fork simulator node stays in stage 5b, after
GO/NO-GO. *Why:* the standoff is the one number stages 3, 4, 6 and 8 all read,
and it is not a perception choice — it is set by how far the virtual tines
reach. (ROADMAP)

**D15. Fork geometry: the carriage sits at the front bumper, tines 1.15 m.**
`carriage_x = 0.4625` (the front face of the `base_link` collision box, i.e.
where a real carriage would physically be), `tine_length = 1.150`,
`tine_width = 0.100`, `tine_thickness = 0.040`, `spacing = 0.365` (the measured
pocket centres). Tip reach **1.6125 m** from `base_footprint`.
*Why:* 1.15 m is the standard tine for a 1.195 m deep pallet, and the carriage
position is not a free choice once you ask where a carriage would actually bolt
on. The two only ever appear as their sum, so this is one decision, not two.
(Stage 5a)

**D16. Standoff 1.75 m; insertion stops at a 0.238 m bumper gap.**
Standoff **1.75 m** (face to `base_footprint`), which leaves the virtual tine
tips 0.1375 m clear of the face before insertion begins. Insertion stops with
the face at **0.700 m**, a **0.2375 m** gap to the real bumper. That gives
1.05 m of insertion travel and **0.9125 m of depth — 76 % of the pallet** — with
the tips 0.2825 m short of the far end, so they never poke out the back.
*Why:* two independent limits, and the stop takes the stricter one.
  - The standoff is set by the fork reach, not by the camera. The tines must
    start outside the pallet (1.6125 + clearance), and the camera's own limit is
    only 1.17 m now that the measured pallet's opening starts at the floor — see
    `docs/SYSTEM.md`. **This replaces the 1.45 m**, which came from keeping an
    EPAL bottom lip in frame that this pallet does not have.
  - The stop is set by the LiDAR, not by how close we dare go. It holds 2 beams
    on the face down to a 0.238 m gap and nothing below 0.204 m (D13), so at
    0.700 m the stop condition is a live measurement at the moment it fires
    rather than dead reckoning. Pushing to a 0.10 m gap would have bought 0.138 m
    of depth (~5 % of the pallet) in exchange for 0.104 m driven blind at an
    obstacle. Not worth it: the forks are imaginary, the robot is not.

  The camera keeps the pockets in view for the first 0.58 m of the insertion
  (down to 1.17 m) and the LiDAR holds the face for all 1.05 m, so "insertion is
  blind" (D7) is now only true of the pockets, and only for the last 0.47 m.
  (Stages 5a, 3, 6)

**D17. Fork underside 0.030 m, centred in the opening.** Tine underside at
0.030 m, so with a 0.040 m tine the centre is at 0.050 m and there is 0.030 m of
clearance above and below inside the 0–0.100 m opening. *Why:* symmetric is the
most tolerant arrangement, and 0.050 m is the measured geometric centre of the
opening. **This replaces the 0.052 m**, which assumed a 22 mm bottom lip.
Note this is deliberately *not* `pallet.pocket_center_height: 0.055`, which is
the centre of the detection band chosen in D5 to avoid the board edge at
0.100 m. The two numbers answer different questions and should not be made
equal. (Stage 5a)

**D18. If the LiDAR and odometry disagree about the stop, the earlier stop
wins.** Whichever source first says "stop" stops the robot. If they disagree by
more than 0.05 m when the stop fires, abort rather than complete the insertion
and report the disagreement. *Why:* fail-safe in the direction of not touching
the pallet, and a disagreement that large means one of the two is wrong, which
is not a condition to drive forward in. (Stages 5a, 6)

**D19. The long side, measured 2026-10-14 (tape, cm).** Height 14.5, of which a
2.0 board at the bottom and a 2.0 board at the top. Between them: block 15,
pocket 38.5, block 15, pocket 36.5, block 15. So the **opening runs 0.020 m to
0.125 m** above the floor (0.105 m high), pockets ≈ 0.375 m wide with **centres
0.525 m apart**, and the layout sums to 1.200 m against the 1.195 m tape — 5 mm,
inside tape error. The 2 cm difference between the two pockets is also tape-level:
the EPAL nominal for this face (blocks 0.145, pockets 0.3825, spacing 0.5275)
agrees with the measurement to 5 mm on the blocks and 2.5 mm on the spacing, and
EPAL is symmetric by design. *Why it matters:* the two faces are structurally
different, not just different sizes — see D20. (Stage 1b)

**D20. Long-side insertion is allowed, behind a configuration flag.** The default
is `pallet.long_side.insertable: true`. Set it false and CLASSIFYING returns
NO-GO "long side, insertion not permitted" instead of proceeding (and, once
stage 9 exists, repositions to a short side instead). *Why:* the measurement
shows this is a genuine four-way-entry pallet, so refusing the long side outright
would be modelling a limitation the pallet does not have. But whether a
*particular* site allows it is a policy question, not a geometry question — long-
side entry loads the 20 mm bottom boards rather than the blocks, and many
warehouses restrict it for that reason. A flag is the honest way to represent a
rule that lives outside the robot. Decided by Athreya 2026-10-14. (Stages 1b, 7)

**D21. Insertion geometry is per side, because the pallet is not square.** D16's
numbers are the short-side numbers. Applying them to the long side would drive
the tips **0.118 m out of the back of the pallet**, since the long side is only
0.795 m deep and the tip reach is 1.6125 m.

| | short (pocket) side | long side |
|---|---|---|
| depth once inserted | 1.195 m | 0.795 m |
| insertion stop, face at | **0.700 m** | **0.950 m** |
| bumper gap at the stop | 0.2375 m | 0.4875 m |
| insertion depth | 0.9125 m = 76 % | 0.6625 m = 83 % |
| tips short of the far edge | 0.2825 m | 0.1325 m |
| fork spacing | 0.365 m | **0.525 m** |
| fork underside | 0.030 m | **0.0525 m** |
| standoff | 1.75 m | 1.75 m (same — set by tip reach) |

*Why:* the rule is unchanged from D16 — stop where the LiDAR still sees the face,
and never let the tips leave the pallet. Only the inputs change. 0.950 m is a
round number above the 0.9175 m hard minimum for 0.10 m of tip clearance, and at
that range the LiDAR has 5 beams on the face rather than 2, so the long side is
the easier of the two to stop safely. The fork spacing and underside are centred
in whichever opening applies; a real forklift adjusts tine spacing for exactly
this reason (manual carriage slide or a hydraulic fork positioner), and ours are
virtual, so it costs a parameter. Note the long side's **0.0525 m underside is
almost exactly the 0.052 m that `params.yaml` has held all along** — that number
came from the EPAL bottom lip, which is real on this face and absent on the
other. (Stages 1b, 5a, 6)

**D22. Side classification has four independent discriminators.** Measured ratios
between the faces:

| | short side | long side | ratio |
|---|---|---|---|
| face width | 0.795 | 1.195 | 1.50 |
| pocket width | 0.230 | 0.375 | 1.63 |
| pocket spacing | 0.365 | 0.525 | 1.44 |
| **opening bottom edge** | **0.000 (floor)** | **0.020** | — |

*Why it matters:* the roadmap's stage 1b question was "face width alone, or width
+ gap pattern?" and assumed width was the primary signal. It is the *weakest* of
the four, because a partially occluded or clipped face measures short and there
is nothing to say whether a 0.9 m face is a foreshortened long side or a
complete short side. Pocket spacing needs only the two pocket centres that
perception already finds, and separates the faces by 160 mm against a measured
spacing error of ~13 mm — more than a 10σ gap. **Default: classify on pocket
spacing, confirm with face width, and require both to agree or report
ambiguous.** The opening's bottom edge is a third check held in reserve: the
long side has a 20 mm lip and the short side does not, which is a different
*kind* of evidence (a height, not a width) and so fails differently. (Stage 1b)

**D23. The robot's own dimensions are confirmed, and the URDF is right this
time.** Measured 2026-10-14: 0.920 m front bumper to rear bumper, 0.690 m wide
over the wheels; lens height re-confirmed at 0.398 m.

| | measured | URDF | difference |
|---|---|---|---|
| length | 0.920 m | 0.925 m | 5 mm |
| width over the wheels | 0.690 m | 0.6996 m | 10 mm |
| front overhang | 0.460 m | 0.4625 m | 2.5 mm |

*Why it matters:* every clearance number in D16 and D21 rests on the front
overhang, and after the `base_footprint` joint (25 mm out) and the LiDAR chain
(41 mm out) it could not be assumed. **`base_link`'s collision box is accurate**
— so the bumper gaps stand as decided. Keeping the URDF's 0.4625 m rather than
the measured 0.460 m, because the larger value is the conservative one. Assumes
the 0.920 m is symmetric about the wheel centres, which the symmetric wheelbase
(±0.249 m) makes likely; worth a glance if the front bumper ever looks deeper
than the rear. (Stages 5a, 6)

**D24. Perception processes the newest frame and drops the rest, at 5 Hz.**
Depth arrives at ~37 Hz; one frame costs ~90 ms after the speed work below, so
most frames cannot be looked at. The node keeps only the latest and starts work
at most `max_rate` times a second. *Why:* the alternative — letting rclpy queue
them — loses no frame and is far worse: the backlog grows ~26 frames/s, so
within ten seconds the published estimate describes the scene seven seconds ago
while the robot acts on it. Dropping costs nothing, because a stationary pallet
looks the same in every frame. 5 Hz keeps the estimate under ~200 ms old
against D9's 0.50 s staleness limit, with headroom for the Jetson being slower
than the laptop. Decided by Athreya 2026-10-15. (Stage 0/1a)

**D25. The pocket opening's lower edge comes from the side's geometry, not from
the points.** `find_pockets` takes `opening_bottom` (0.000 short, 0.020 long).
*Why:* it is not observable from either face. On the short side the opening
starts at the floor, so there is no edge to see — the floor runs continuously
into the pocket. On the long side it starts on a 20 mm board that
`remove_floor` has already deleted with the 0.03 m floor margin. The measured
quantity is the opening's **top** edge; the height is measured down from it to
a known floor. This replaces a `bottom = 0.0` fallback that was right for the
short face by accident and 20 mm short for the long one — worse than wrong,
because it looked right half the time. Measuring the lower edge instead would
need the floor margin reduced (2.4 mm of floor RMS would allow ~0.015), which
is a change to argue for with data. (Stage 1a)

**D26. Side classification refuses rather than guesses, including when the face
is merely clipped.** A consequence of D22 worth stating because it refuses good
pallets: occlusion or the image edge can cut a 1.195 m face down to something
measuring like a 0.795 m one, and the spacing then says `long` while the width
says `short`. That is AMBIGUOUS, not `long`. *Why:* width disagreeing with
spacing means the face is not fully visible, and measuring pocket geometry
against nominals on a face you cannot fully see is how a damaged pallet passes.
Resolving it belongs to the tracker (stage 2), which sees several frames from
slightly different positions. (Stages 1b, 2)

**D27. One fork setting for both faces, not one per face.** `forks.spacing:
0.365`, `forks.bottom_height: 0.040`, used whichever face is entered. This
supersedes D21's per-side `fork_spacing` / `fork_bottom_height`.
*Why:* decided by Athreya 2026-10-19 — a carriage without a hydraulic
positioner keeps one setting, and having one removes a way for the overlay, the
simulator and the insertion to disagree about what the forks are.

It is only allowed because a single setting genuinely fits both. With a
0.100 × 0.040 tine:

| | window where one setting fits both |
|---|---|
| spacing | 0.250 – 0.495 m (short face 0.235–0.495, long 0.250–0.800) |
| underside | 0.020 – 0.060 m (short 0.000–0.060, long 0.020–0.085) |

0.365 is the measured short-side pocket spacing, so the tines sit dead centre
in the face normally entered; on the long face they land 80 mm inboard of the
pocket centres with 57 mm to the nearest edge. 0.040 is the middle of the
height window, clearing the long face's 20 mm bottom board by 20 mm and the
short face's board above by 20 mm. **Worst clearance anywhere: 20 mm.**
`test_params.py` derives those windows from the measured pocket geometry rather
than checking the numbers against themselves, so shrinking a pocket cannot
silently make the forks stop fitting. The per-side `insertion_stop_face_x`
stays per side — that is about depth, which the two faces really do differ in.
(Stages 5a, 6, 8)

**D28. The overlay draws measured pockets solid and nominal dashed.** At the
same centre, so the gap between the two outlines is the measurement error.
*Why:* the overlay's job is to make an error visible rather than readable. On
`pallet_l_129` the dashed box is plainly wider than the solid one, which is the
−51 mm pocket-width bias from D19 as a picture instead of a log line. Decided
by Athreya 2026-10-19. (Stage 8)

**D29. The virtual forks are rigidly fixed to the robot, never referenced to
the per-frame floor fit.** The tines are built against a level plane at the
measured `floor_z` (-0.025), a constant of the machine. *Why:* raised by
Athreya 2026-10-19 — a carriage is bolted to the truck, and the truck's
orientation already follows whatever it stands on, which is what
`base_footprint` encodes. Referencing perception's fitted plane adds nothing
and costs three things:

1. it moves — RANSAC redraws its samples every frame, measured at 1.0 px of
   tine-tip wobble on a parked robot;
2. that plane is fitted to floor metres ahead, not the floor under the wheels,
   and on a ramp those differ;
3. **it hides the error the overlay exists to show.** A camera orientation
   error tilts the fitted floor, so tines referenced to it lean by the error --
   in the same direction as the pockets, which are measured against the same
   plane. Measured against the real 0.87° mounting error on this robot: the
   fork-to-pocket gap grew by 5.4 px with the fitted floor and 16.8 px
   robot-fixed. The floor-fit reference swallowed two-thirds of the evidence.

The pockets keep using the fitted floor, because that is what they were
measured against. Tines robot-fixed, pockets perception-fixed: that asymmetry
is what makes a mismatch between them attributable.
(Stage 8; supersedes the reasoning in `fork_overlay.tine_corners`' original
docstring, which argued the opposite and was wrong.)

## 2026-10-19 (stage 0: arbitration and safety)

**D30. `twist_mux` arbitration is configured, and our controller no longer
publishes at the base.** `config/twist_mux.yaml`: `/cmd_vel_safety` at priority
200, `/cmd_vel_pallet` at 100, `/cmd_vel_nav` at 10, every input with a 0.5 s
timeout, output remapped onto `/cmd_vel`. Plus a **lock** at priority 50 on
`/pallet_approach/nav_lock`, held at 10 Hz by the coordinator for the length of
a run. *Why:* implements D3/D10. Three points that are decisions rather than
transcription:

- **A lock as well as priorities.** Priority only decides who wins while both
  are publishing, and "Nav2 finished its goal" and "a pallet run started" are
  independent events -- a half-finished Nav2 goal still has a controller that
  believes it is driving. The lock makes "a run is in progress" explicit
  instead of inferred from who happens to be talking.
- **The lock expires in 0.5 s.** A lock with no timeout is held for ever once
  published, so a coordinator that died mid-run would lock the robot out of
  navigation until a human noticed. Expiring means a dead coordinator gives
  control back.
- **Input timeouts (0.5 s) are deliberately looser than the watchdog's
  staleness limits (0.20-0.50 s, D9).** The watchdog is meant to notice first
  and stop the robot with a reason the operator can read; the mux timeout is
  the backstop for the watchdog itself dying.

**Open on the robot:** Nav2 still publishes straight to `/cmd_vel`, which is
the mux's output. Until its controller server is remapped to `/cmd_vel_nav`,
Nav2 bypasses arbitration and two publishers share the base's input. That is
survivable only because Nav2 is measured silent while idle (D10) -- a window,
not a design. Remap before the first driving session; it is on the lab session A
list. (Stage 0)

**D31. The controller refuses to configure if nothing subscribes to its command
topic.** This replaces `require_sole_cmd_vel_publisher`, which was dead code
(the coordinator never declared it) encoding the superseded "shut Nav2 down for
a run" design. *Why:* moving our output from `/cmd_vel` to `/cmd_vel_pallet`
created a new silent failure -- with `twist_mux` not running, every command goes
nowhere and the whole stack reports that it is driving while the robot sits
still. That is the fifth instance of this project's recurring bug shape: a
reasonable default, chosen automatically, never reported. The fix is the same as
the other four: say what was chosen, or refuse. The launch file also starts the
mux by default, so the normal way of running the stack cannot produce it.
(Stage 0)

**D32. A watchdog node, independent of the coordinator, owns the stop.** New
node `watchdog`: it measures the age and rate of the camera, the perception
estimate, odometry, the LiDAR and tf; publishes `/health` at 2 Hz and
`/diagnostics`; and, while a run is in progress, publishes a zero command on
the mux's highest-priority input the moment a required source goes outside its
limits. *Why:* the coordinator aborting a run is the normal path and it depends
on the coordinator being alive and correct. This node depends on neither, so a
coordinator that has wedged still ends up with a stationary robot. The two
mechanisms overlap on purpose. Four choices inside it:

- **It never subscribes to depth.** Camera liveness comes from
  `depth/camera_info` -- same publisher, same callback, same rate, a few dozen
  bytes against 3.7 MB a frame. The network rule (CLAUDE.md item 7) then holds
  by construction wherever the node runs.
- **The LiDAR subscription is BEST_EFFORT.** `/ouster/points` publishes
  BEST_EFFORT and a RELIABLE subscriber receives nothing at all from it
  (measured 2026-10-15). A watchdog that silently saw no LiDAR would latch a
  stop on every run.
- **The stop latches for the rest of the run.** A sensor that flickers would
  otherwise produce a stutter of stop and go, which is worse than either. It
  releases when the run ends, and the reason reaches the operator on `/health`.
- **What is required changes with the phase.** Perception is a managed
  lifecycle node and publishes nothing between runs, so requiring its estimate
  always would make the stack permanently unhealthy and D12's goal-time check
  would refuse every goal. The estimate is required only while a run is in
  progress; the sensors always. `/health` carries a `required[]` array so which
  answer is being given is visible rather than implied.

Two measurement rules inside `health.py` that are worth stating because they
are the bugs this project keeps producing: **a source that has never published
has no age** (`None`, not 0.0, which passes every staleness limit ever written),
and **a rate is not measurable until the window holds enough history** (`None`,
not 0.0, which would trip the rate floor on every run inside the first second).
There is also a **future-stamp check**: a negative age is reported as clock skew
rather than as freshness, because the 24.5 % fast camera clock measured on
2026-10-15 would otherwise have read as perfectly fresh tf. (Stage 0)

**D33. Rate floors come from what the pipeline consumes, not from what the
sensor produces.** `camera_info` 10 Hz (perception looks at 5 frames a second,
D24), `estimate` 2.5 Hz, `odom` 20 Hz, `lidar` 5 Hz -- each roughly half of
what the consumer needs rather than half of the nominal sensor rate. *Why:*
depth arrives at ~37 Hz and perception uses 5 of them, so a camera at 12 Hz
costs the pipeline nothing and a floor set near the nominal rate would abort
runs over a hiccup that does not matter. (Stage 0)

## Pending (to be decided at the start of each stage)

See the stage lists in `docs/ROADMAP.md`. Stage 5a is closed (D15–D18); the
numbers it decided reach `config/params.yaml` in stage 1a, together with
retiring the lip-based standoff tests that encoded the old 1.45 m.