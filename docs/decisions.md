# Design decisions

One entry per behaviour decision: what was decided, why, when, and which
stage it belongs to. Every decision that changes what the robot does is taken
before the code for it is written, and a decision is never changed silently:
when testing shows one does not work, a new entry supersedes it and says why.

## 2026-10-14

**D1. Working model.** Each stage starts with a logic sheet: every behaviour
decision in plain language, with a default and the reason for it. Only then is
it implemented and tested. *Why:* the behaviour of a robot that drives at a
load should be a written, reviewable decision, not a side effect of whatever
the code happened to do. (Stage: all)

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
*Why:* the pockets leave the camera's view before the forks are in; the forks
are imaginary but the robot is not. (The first estimate here was ~0.9 m; D16
replaces it with 1.17 m, computed from the measured camera pose: the pocket's
top edge leaves the frame once the face is closer than that.)
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
rule that lives outside the robot. Decided 2026-10-14. (Stages 1b, 7)

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
than the laptop. Decided 2026-10-15. (Stage 0/1a)

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
*Why:* decided 2026-10-19 — a carriage without a hydraulic
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
2026-10-19. (Stage 8)

**D29. The virtual forks are rigidly fixed to the robot, never referenced to
the per-frame floor fit.** The tines are built against a level plane at the
measured `floor_z` (-0.025), a constant of the machine. *Why:* raised
2026-10-19 — a carriage is bolted to the truck, and the truck's
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
  bytes against 3.7 MB a frame. The network rule (docs/SYSTEM.md) then holds
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

## 2026-10-19 (stage 2: the tracker)

**D34. The tracker does not smooth the estimate.** It carries the newest
accepted estimate and uses the window to decide whether to trust it, not to
alter it. *Why:* averaging poses over a window lags the truth by about half the
window times the closing speed -- 10 frames at 5 Hz and 0.10 m/s is 0.10 m --
and the closing loop's whole design is that the measurement improves as the
remaining correction shrinks. A lagged average would undo exactly that. The
confidence is a count of accepted frames, not a filter gain: a count is what
can be explained in one sentence and read off a log. (Stage 2)

**D35. The track lives in `odom`, with the robot's motion taken out.** Each
estimate's pocket centre is placed in `odom` using the odometry sample nearest
the frame's stamp, and association is judged there: 0.15 m of position and
0.20 rad of heading. *Why:* in `base_footprint` a stationary pallet moves by
the robot's own travel -- 20 mm a frame at the closing speed, 0.20 m over the
10-frame window -- so an association gate in the robot frame would have to
swallow the travel and would stop discriminating. In `odom` a stationary pallet
is stationary, which is what makes the gate mean measurement error.
`test_tracker.py` asserts both directions: the same drive that passes with the
pose taken out fails without it. The odometry sample has to be the one at the
frame's stamp rather than the latest, because taking the motion out at the
wrong instant undoes it silently; a frame with no sample within 0.05 s is
rejected with that as its reason. (Stage 2)

**D36. A side that changes between frames rejects both readings.** A pallet
does not turn over in the 200 ms between two frames, so a track that classified
`short` and then sees `long` has one wrong classification and no way to tell
which. Neither is trusted: the frame is rejected and the consecutive count
restarts. `unknown` (classification switched off) is an absence of evidence and
does not count as a change. *Why:* the same reasoning as D26 one level up -- the
honest answer to contradictory evidence is to refuse, and the tracker is where
several frames can be compared. (Stage 2)

**D37. The LiDAR cross-check compares the nearest surface in a window round
the camera's plane, and "too few returns" is not agreement.** Window: half the
*measured* face width laterally, `floor_margin` to just above the pallet
vertically, 0.60 m towards the robot and 1.00 m away from it in depth.
Statistic: the 90th percentile of the signed offsets. Limit 0.10 m. Fewer than
`min_points` returns in the window reports NOT EVALUATED, which does not block
a run and must never read as agreement. *Why:*

- **The camera's plane is the reference** and the LiDAR is asked about it,
  rather than fitting a second plane and comparing two fits. One fit has one
  failure mode; two have three, and the comparison then needs its own
  correspondence argument.
- **The lower height bound is the load-bearing one.** Without it the floor
  immediately in front of the face falls in the window, and floor points lie
  almost exactly on a vertical plane's own extension -- the check would agree
  about a face that is bare ground. `test_tracker.py` has that case.
- **The depth window is asymmetric.** The first version was symmetric at
  +-0.15 m, chosen to keep a loaded pallet's cargo out, and a test found it
  excluded the disagreement the check exists to catch: a camera hallucinating a
  face 0.4 m from the real one read as NOT EVALUATED. Widening it behind the
  plane fixes that; it stays narrow in front so the robot's own structure
  cannot become the measured surface. A positive offset -- the LiDAR seeing
  something nearer than the camera's plane -- is reported as a disagreement
  because between us and the pallet is precisely where that matters.
- **Nearest surface, not mean or median**, because a loaded pallet returns from
  its cargo too and the face is what is being checked.
- **0.10 m is loose on purpose.** The LiDAR's own URDF chain is 41 mm out
  (docs/SYSTEM.md), so a tight limit here would be measuring our extrinsics
  rather than checking the camera.

**`min_points: 20` is arithmetic, not a measurement**, and is the one number
here that needs the lab: the beam geometry predicts 1-2 beams by ~44 azimuth
columns at 3 m and 2-3 by ~75 at the standoff, but nothing has counted the real
returns yet. The tier-1 bags carry `/ouster/points`; the achieved counts go in
`docs/dataset.md` and this value is set from them. (Stage 2)

## 2026-10-19 (stage 3: the closing loop)

**D38. The two-leg approach is replaced by a continuous closing servo.** This
supersedes D6. There is no line-up point and no rotate-drive-rotate: the robot
backs off until it has enough run-up, then drives slowly forward while servoing
onto the pocket centre line, and stops at the standoff. Decided
2026-10-19. *Why:* the measurement improves as the robot closes, so correcting
continuously means every correction is made with better information than the
one before. Aligning once at 3 m and driving straight freezes in whatever error
the 3 m measurement had. It also removes a failure mode rather than adding one:
the two-leg design expected to lose sight of the pallet while turning and had
to re-acquire it, which is why D6 needed a separate rule for when losing sight
aborts.

`approach.lineup_pose` and `approach.approach_targets` are **deleted**, not left
unused. Dead code with passing tests behind it reads as a design that is still
in force. (Stage 3)

**D39. Variable standoff with a 3 m minimum, and the run-up is computed from
the lateral error.** Decided 2026-10-19 after a simulation: starting
at 1.75 m left 182 mm of residual error and 2.25 m left 112 mm, against 65 mm
of room for a 100 mm tine in a 230 mm pocket; 3.00 m was the first distance
that fitted. The implementation makes the floor a floor and computes the rest:

    run_up = max(3.0, standoff + distance the servo needs for this offset)

`required_run_up` models the servo in **two regimes**, and getting that wrong is
the most instructive bug of this stage. While the aim is pinned at
`max_approach_angle` the offset falls linearly at `sin(angle)` per metre; only
below `max_approach_angle / k_cross` does the proportional term take over and
the decay become exponential with a length of `1 / k_cross`. The first version
used the exponential term alone, claimed 3.04 m was enough to take out 0.50 m
of offset, and the simulated run finished **250 mm off the line with 11 degrees
of yaw**. With both terms the honest answer is just over 4 m.
`test_closing.py` measures the distance the loop actually needs and fails if the
formula promises better. (Stage 3)

**D40. The servo is a cascade with two independent gains, and small angular
commands are dropped rather than inflated.** Outer loop:
`aim = -k_cross * offset`, capped at `max_approach_angle` and zeroed inside
`straighten_offset`. Inner loop: `omega = k_heading * (aim - heading)`.
`k_cross = 2.5` (a 0.40 m decay length), `k_heading = 4.0`. *Why each piece:*

- **Separate gains.** The first implementation was pure pursuit chasing a
  carrot on the centre line, which is the same algebra with the two gains
  locked together at `k_cross = k_heading / lookahead`. Locking them is what
  made it unusable: a lookahead short enough to converge in 1.25 m implied an
  approach angle near 60 degrees, so the robot dog-legged -- turn in place,
  drive, turn back -- and arrived square only by luck. Separated, the decay
  length and the approach angle are chosen independently, which is what they
  are.
- **A cap on the approach angle** (0.25 rad). Without it a large offset asks
  for an aim the base reaches by turning most of the way sideways, and the
  pallet leaves the camera's view: the servo would blind itself.
- **A deadzone on the aim** (15 mm). The arrival yaw is otherwise *determined*
  by the arrival offset, since aiming `k_cross * offset` at the line means
  arriving at that angle -- at `k_cross = 2.5` a 35 mm residual is the entire
  5 degree GO/NO-GO yaw budget. Inside the deadzone the robot straightens and
  arrives square.
- **Angular commands below `min_angular` are ZEROED, not raised.** This is the
  single change that gave the loop any fine authority. The base cannot turn
  more slowly than `min_angular`, so for a finer correction the choice is to
  turn too much or not to turn, and not turning is the stable one. Raising it
  -- which is what `apply_deadband` does, and what this code did first -- makes
  every command hard over or hard back: measured in simulation, the residual
  then scaled with the lookahead and **ignored the gain entirely**, which is
  the signature of a saturated loop. The linear deadband is still raised,
  because below it the base does not move at all.
- **`k_heading = 4.0` rather than the 6.0 that simulated slightly better.** The
  simulation has no actuator lag; the gentler gain is the one that survives
  having some.

The cost of zeroing small commands is a granularity: the smallest heading
correction the base will make is `min_angular / k_heading` = 0.025 rad, 1.4
degrees. **That, and not the gain, is what sets the achievable accuracy.**
(Stage 3)

**D41. A square-up phase at the standoff.** The loop stops driving when it
reaches the standoff and then turns in place until it is within
`arrival_yaw` (0.035 rad, 2.0 degrees) of square. *Why:* the loop can be caught
mid-correction when it arrives -- one 0.2 s frame at the 0.30 rad/s limit is
3.4 degrees -- and simulation with noise showed arrival yaw of 4.4 to 5.9
degrees against a 5 degree limit, which would have failed GO/NO-GO on yaw after
a perfectly good approach. Turning in place costs about a second, does not
disturb the lateral offset at all (turning about `base_footprint` does not move
`base_footprint`), and brought the worst simulated arrival to 2.4 degrees and
17 mm. `arrival_yaw` has to be larger than the base's turn granularity above or
the square-up cannot finish; `test_params.py` checks that relationship.

**Simulated, not measured.** `test/simulate.py` models unicycle kinematics,
the two rates (5 Hz estimates, 20 Hz commands) and measurement noise. It does
not model actuator lag, wheel slip or an uneven floor, so its numbers are a
lower bound on the error rather than a prediction. The real figure comes from
lab session B. (Stage 3)

**D42. Our own reverse controller, with a LiDAR check and an explicit
allowance.** Decided 2026-10-19 over handing the reverse to Nav2.
The reverse runs at 0.05 m/s with angular exactly zero, and the distance it is
allowed to cover is

    allowance = distance driven forward this run + max_unverified (0.50 m)

*Why it is stated as an allowance rather than a clearance:* the ground
immediately behind the robot cannot be checked, so the justification is an
argument, not a measurement -- `travelled_forward` is ground the robot's own
body was standing on seconds ago and is therefore known to have been clear. The
0.50 m is the admitted gap in that argument, kept small deliberately.

**What this choice costs, plainly:** Nav2 would have had the costmap, which
remembers what the LiDAR saw while the robot drove past, and could have planned
a reverse over verified ground. We do not have it. `rear_clearance` measures
what the LiDAR can see in a corridor the robot's width behind the bumper and
refuses the reverse if an obstacle is inside the needed distance plus a margin,
but it returns **None for "found nothing", never "clear"** -- the distinction
matters because at close range behind the robot there is nothing to find.
(Stages 3, 6)

## 2026-10-19 (stages 5b and 6: the forks and the insertion)

**D43. The fork simulator may not claim anything the sequence did not do.**
`loaded` is set only when a lift *completes* from the insertion height, and
`lift()` refuses outright from any other height. Lowering and lifting take real
time at `speed`, so a phase whose job is to take time while the robot stands
still cannot be skipped. *Why:* there is no hardware behind any of this, so the
simulator is the only thing standing between "the sequence reached LIFTING" and
"the pallet came off the floor". A simulator that sets a flag because a state
machine said a word is a simulator that produces a successful demo of nothing.
(Stage 5b)

**D44. `lift_travel` is carriage travel, and the pallet lift is reported
separately per face.** 0.070 m of carriage travel from the 0.040 m insertion
height. The tine top first takes up the slack to the top of the pocket --
0.020 m on the short face, 0.045 m on the long one -- before the pallet moves,
so the same travel lifts the short face **50 mm** off the floor and the long
face **25 mm**. `forks.pallet_lift` reports which, and the tests require both
to be positive. *Why:* "the forks rose by 70 mm" is a fact about the carriage;
"the pallet came off the floor" is the claim a virtual pickup is making, and
conflating them would let a lift that never moved the pallet read as a success.
(Stage 5b)

**D45. The insertion plan is built from the MEASURED face distance, and the
LiDAR is the source for both the plan and the stop.** `plan_insertion` takes
the face distance measured when insertion begins, not the configured 1.75 m,
because the closing loop arrives within `range_tolerance` of the standoff
rather than at it; planning from the nominal would spend that tolerance on the
stop distance. The measurement comes from the LiDAR, with perception's estimate
as a logged fallback. *Why the same sensor for both:* planning from one source
and stopping on another is how a 50 mm disagreement becomes a surprise at the
moment of the stop rather than a known offset at the start of the drive.
`plan_insertion` **refuses** rather than clamps when the face is already inside
the stop, when the travel exceeds `max_travel`, when the side has no configured
geometry (which is what `ambiguous` reaches it as), or when the side is marked
not insertable (D20). (Stage 6)

**D46. During insertion, no LiDAR measurement for `max_unmeasured_frames`
aborts, and so does finishing on odometry alone.** Five frames at 10 Hz is half
a second, 25 mm of travel at 0.05 m/s. *Why:* this is the one phase where the
robot deliberately drives at something it cannot fully see, and odometry alone
is dead reckoning. D18 made a *disagreement* abort; this closes the gap D18
left, which is that an absent measurement cannot disagree with anything. A run
that completes its travel with nothing having confirmed where the face was has
not measured an insertion and must not report one. (Stage 6)

**D47. The insertion's lateral drift is measured and logged, not acted on.**
`face_distance_ahead` returns the mean lateral position of the returns alongside
the distance. *Why:* D7 and ROADMAP stage 6 keep insertion blind-on-odometry
with angular exactly zero, because the tines are inside the pallet and steering
drags them sideways through the blocks. But D13 found the LiDAR holds the face
for the whole travel, so the drift *is* observable, and the decision about
whether to correct it needs a number. Lab session C produces that number; until
then the behaviour is unchanged and the data is collected. (Stage 6)

## 2026-10-19 (stage 7: the coordinator, and what the harness found)

**D48. The full sequence, as eleven states plus ABORTED.** IDLE, CONFIRMING,
CLASSIFYING, BACKING_OFF, CLOSING, CHECKING, LOWERING, INSERTING, LIFTING,
BACKING_OUT, DONE, ABORTED. The controller may be told to move in exactly four
of them (`DRIVING_STATES`), and `test_coordinator.py` asserts that over every
event from every state -- so a state added later without thinking about the
list cannot drive, which is the right default. (Stage 7)

**D49. An aborted insertion stops where it is. It does not reverse out.** The
robot's body is at least 0.204 m clear of the face at any legal stop (D13,
D16), so standing still is safe, and driving after an abort contradicts the one
rule with no exceptions. Recovery is the operator's, with the reason in the
result. *Why this is worth stating:* the tempting behaviour is to back out
automatically, and it would usually work. But "the robot must stop in every
terminal state" cannot have a special case for the state where the robot is
closest to the pallet and something has just gone wrong. (Stage 7)

**D50. Only NO-GOs about where the robot is standing are retried.** `yaw`,
`lateral_offset` and `stability` -- `retryable_checks` in params.yaml. A pocket
that measures the wrong size, a face with no support behind the fit, an
obstruction in a pocket and a load overhanging the face are all properties of
the *pallet*, and driving the same approach again would measure the same
pallet. A mixed failure is not retried either: if the pallet is wrong as well
as the robot, the pallet does not change. One retry (2026-10-19), and
it always reverses at least `min_retry_backoff` so that "retry" means a
genuinely new approach rather than a twitch. (Stage 7)

**D51. Perception publishes the two point sets the GO/NO-GO rules need.** New
topics `~/cluster_points` and `~/scene_points`, filled by an optional
`decision_points` dict that `estimate_pallet` now populates. *Why:*
`check_overhang` and `check_pocket_clearance` take points rather than an
estimate, and both treat missing data as a NO-GO rather than a pass -- a
decided behaviour, and the conservative one. Nothing was supplying them, so
**every run returned NO_GO with "no points to check the load against"**, found
by the offline harness on its first complete run. The clouds are purpose-built
and small: the pallet cluster (a few thousand voxels) and a box around the face
rather than the ROI, because the clearance boxes are 0.10 m deep and no wider
than a pocket, so nothing outside that box can affect either rule. Both stay on
the Jetson between two nodes a process apart; nothing subscribes to them from
the main PC and nothing should. (Stages 1a, 4, 7)

**D52. The stability rule is evaluated only on frames taken standing still.**
The coordinator clears its estimate buffer on entering CHECKING. *Why:* the
rule compares the face position across five frames against a 30 mm limit, and
five frames taken while closing at 0.10 m/s span 100 mm of real travel. The
harness measured exactly that -- a clean noisy run refused with "stability:
face position moved 118 mm across 5 frames". An earlier version of the node
dodged this by repeating the latest estimate five times, which made the rule
measure one frame against itself and therefore **unable to fail**: a safety
rule that cannot fail is worse than one that is absent, because it reports a
pass. (Stages 4, 7)

**D53. The divergence detector needs eight frames, not five, and ignores
errors inside `residual_target`.** The mechanism is unchanged -- abort
after N consecutive growing frames (2026-10-19) -- but N and its exposure are
now measured rather than guessed. **At five frames and 10 mm of per-frame
noise, the offline harness aborted 3 runs in 12 from a 9 m run-up and 1 in 12
from 6 m, with the robot converging perfectly.** Monotone growth is a fragile
statistic: five strictly increasing samples happen by chance once in 120
windows of white noise, and a converged robot on a long run-up offers hundreds
of windows. `min_divergence_growth` barely helped, because the growth
accumulates across the window. At eight frames plus a floor at
`residual_target`: none, in every case tried.

The cost is 1.6 s rather than 1.0 s to notice a real divergence, which at the
closing speed is 160 mm of extra travel. **Still open:** if that is too
slow, the alternative is a model-based check -- compare the error against what
the servo's own decay model predicts after the distance travelled -- which is
robust to noise over a long baseline and would catch a divergence in half the
frames. That is a different mechanism, so it is a decision, not a tweak.
(Stage 3)

**D54. The run-up requirement has a tolerance, and the reverse overshoots
it.** `closing.run_up_tolerance: 0.05`, applied both ways. *Why:* the reverse
stops within its own 10 mm window and odometry path length is never shorter
than the displacement, so a reverse computed exactly finishes about a
centimetre short of the requirement. Without the tolerance the robot then asks
for another centimetre, and after `max_backoffs` the run dies with
"still only 2.99 m from the pallet after 2 reverse(s); 3.00 m is needed" --
having done everything right. Found by the offline harness. (Stage 3)

## 2026-10-19 (operational flow)

**D55. The flow is strictly sequential, and the trigger is a topic.** Decided
2026-10-19. Nav2 drives the robot in front of a pallet and
**finishes its goal**; only then is this pipeline triggered, and Nav2 does not
participate in the run. Every node is already up and waiting.

    docker run (one entry point)  ->  nine nodes up, nothing moving
    Nav2 goal  ->  Nav2 finishes
    /pallet_approach/start  ->  the run
    /pallet_approach/result  ->  the outcome

*Why it is a topic:* the thing doing the triggering is a script or a behaviour
tree that has just finished a navigation goal and has nothing to wait for. An
action is the wrong shape for that. The action server stays, because it is the
better way to run one by hand -- it streams feedback and Ctrl-C cancels -- and
both paths go through one `_start_run`, so the health gate (D12) cannot apply
to only one of them.

Three consequences that are not optional:

- **`/pallet_approach/cancel` exists.** A topic trigger is fire-and-forget, so
  a run started that way has no action goal to cancel -- and "the robot must
  stop on cancel" has no exceptions. This is a safety requirement, not a
  convenience.
- **`/pallet_approach/result` exists**, carrying the action's own Result
  message rather than a second definition of one. One definition means the
  topic and the action cannot drift apart, and it means a refusal (unknown
  mode, unhealthy stack) is reported somewhere, which a fire-and-forget trigger
  otherwise has no way to do.
- **The start subscription is VOLATILE, depth 1.** Latched, it would start a
  run the moment the coordinator came up, from a message published before it
  existed. Deep, it would start a second run the instant the first finished.

**What this changes about arbitration.** D3 assumed Nav2 and this stack might
overlap, and D30 built `twist_mux` for it. With a sequential flow they never
overlap in time, so the mux's *priority* between `/cmd_vel_pallet` and
`/cmd_vel_nav` stops mattering. Two things it does keep doing, both worth the
config file:

1. the watchdog's `/cmd_vel_safety` outranks our own controller, which is the
   mechanism by which a wedged coordinator still leaves a stationary robot;
2. it is insurance against Nav2 publishing when nobody expected it to -- an
   operator sending a goal mid-run, or a recovery behaviour firing.

**But (2) only works once Nav2 is remapped to `/cmd_vel_nav`.** Until then Nav2
publishes straight to `/cmd_vel`, which is the mux's *output*, and the nav lock
blocks an input nothing is using: it is inert. That is acceptable under this
flow and it was a blocker under the old one, so the remap moves from "before
the first driving session" to "when convenient". Stated here rather than left
implied, because an inert safety mechanism that looks active is worse than an
absent one.

**D56. `approach_only` is a goal mode, not a configuration flag.** `mode:
approach_only` runs the whole approach and stops at the GO/NO-GO decision with
outcome `GO`, inserting nothing. `pickup` runs the whole sequence. An unknown
mode is **refused**, not defaulted. *Why a mode:* ROADMAP's lab session B is
"first driving: approach and GO/NO-GO only, insertion **not** enabled", and
that is a property of one run rather than of the robot -- the same stack,
triggered twice, should be able to do a dry approach and then a full pickup
without a restart. *Why refused rather than defaulted:* the difference between
the two modes is whether the robot drives into the pallet, so a typo must not
fall back to the one that does.

This was a decided requirement (ROADMAP, lab session B) that the stage 7 work
left unimplemented. (Stage 7)

## 2026-10-19 (the video dashboard)

**D57. One video stream, watchable from the laptop, carries everything -- not
just the camera.** The transport (GStreamer -> MediaMTX -> WebRTC,
`video_stream.py`/`video_streamer.py`/`config/mediamtx.yaml`) carries one
screen, and that screen should show everything about the pipeline. Decided here is what that becomes in practice: `dashboard_node`
subscribes LOCALLY to `/fork_overlay/image` and five telemetry topics, draws a
HUD panel underneath the existing fork/pocket overlay, and republishes the
composite on `/dashboard/image`, which is the stream `video_stream.launch.py`
points a browser at by default. The camera stream (`/zed`) stays available
separately for checking the camera itself.

Five things worth recording about how it is built, because each one is a
place a HUD like this normally goes quietly wrong:

- **Every value is shown as of when it arrived, not assumed current.**
  `dashboard.Snapshot` tracks a `*_at` timestamp per source alongside its
  value, the same shape as `health.py`'s age tracking, and a value older than
  `dashboard.stale_after.<source>` fades to grey with its age rather than
  sitting there looking live. This project has reintroduced the same bug
  (silent fallback, D-many) often enough that a HUD -- the place it is most
  tempting, because a frozen number still looks like a number -- gets it built
  in from the start rather than added after someone is fooled by it.
- **The camera frame itself gets the same treatment.** If `/fork_overlay/image`
  stops updating, the picture is still a picture; nothing else about the
  design would show that. A red border and an age tag appear directly on the
  image when it goes stale, and a grey placeholder with "NO CAMERA FEED"
  replaces it if none has arrived at all -- so the dashboard degrades to
  telemetry-only rather than to a frozen or blank frame that looks fine.
- **The output frame size is fixed, never data-dependent.** Every widget is
  exactly a title line and two value lines; a reason too long to fit is
  truncated with an ellipsis rather than growing the panel. `dashboard.
  frame_size` computes the total canvas size from configuration alone, with no
  image or text content involved, so `video_stream.launch.py`'s declared
  stream resolution can match it exactly -- `test_dashboard.py` checks the
  checked-in default against that computation and fails if they drift.
- **The state-colour palette moved to the shared `/**` block in params.yaml,**
  because two differently-named nodes (`fork_overlay`, `dashboard`) now draw
  in it, and a node's own parameter block is only visible to that node --
  `fork_overlay` and `dashboard` would otherwise have needed two copies that
  could silently disagree. While doing this, `fork_overlay_node` turned out to
  be missing `colors.inserting` entirely: `_params()` read it with an
  f-string-built key (`get(f'colors.{k}')`), which the static declared/read
  scraper in `test_params.py` cannot see, so the gap was invisible to the one
  test built to catch exactly this. The overlay's own guard against a
  rendering bug taking the node down ("an overlay bug must not take the node
  down") caught the resulting exception every time, which meant **the fork
  overlay silently stopped updating during INSERTING and LIFTING** -- the two
  states where its colour change matters most -- since the first frame after
  Day 8. Fixed two ways: the missing declaration, and rewriting every colour
  read in both nodes as one explicit literal `get('colors.X')` call instead of
  a loop over an f-string, so the scraper can verify every one of them and
  this class of bug cannot reach either node again unnoticed.
- **A new topic, `/coordinator/status`, publishes the live reason continuously
  (20 Hz).** `/coordinator/state` (plain `String`, just the state name) stays
  as it is, because `fork_overlay` already keys its colour off it. Before this,
  the *reason* a state was what it was only reached an action goal's own
  feedback -- invisible to anything that was not the caller of that specific
  goal, including a HUD, rqt, or an operator who started watching mid-run.

Treated as an engineering decision rather than a behaviour decision: nothing
here changes what the robot does or any safety behaviour: it is
read-only observability built on topics that already existed, plus one new
read-only status topic. (Stage 8, video streaming)

**D58. Perception gets 2 s after a run starts before its silence counts.**
`watchdog.first_estimate_grace: 2.0`. A run's own start activates perception,
which needs a frame and up to a second of processing before its first
estimate; the watchdog judges at 2 Hz and used to call "nothing received since
startup" a failure immediately, so a run aborted whenever that judgement came
first. *Why:* measured 2026-10-20 in the Gazebo sim, 2 aborts in 4 runs, about
0.1 s after the start. The grace covers only the first estimate of a run (once
one has arrived, `max_age.estimate` applies unchanged) and only CONFIRMING,
where the robot does not drive. Amends D9. (2026-10-20)

**D59. The closing loop steers onto the pallet's averaged pose, recomputed at
20 Hz against odometry.** Refines D34 and the "act on what it can see now"
rule in `closing.Geometry`. The tracker keeps a weighted average of the pallet's
pose in `odom` (`pallet_tracker.fusion_memory: 0.95`, about 20 frames; weight
1 / range^2, floored at `fusion_min_range: 0.5`), and the controller places
that line against the robot's odometry pose on every tick
(`closing.geometry_from_track`). INSERT and every single-frame judgement
(GO/NO-GO, the run-up decision) still use the newest frame. The divergence
detector is still fed once per camera frame, so its 8 frames still mean
1.6 s. *Why:* measured 2026-10-20 in the Gazebo sim: one frame's face normal
wobbles by +-0.04 rad, which at 3 m swings the pocket centre line by
+-120 mm five times a second; the angular command saturated and flipped sign
20 times in a 20 s approach while the measured offset jumped by up to 100 mm
between frames. D34's objection was lag from averaging poses relative to a
moving robot; in `odom` the pallet stands still, so there is none, and
1 / range^2 keeps its point that close frames are better. The cost is
odometry drift, which the bounded memory limits. **Gains unchanged**
(k_cross 2.5, k_heading 4.0, zeta = 2 at 0.10 m/s): an offline sweep with
sim-sized normal noise (8 seeds each) gave a median of 6 angular sign flips
at k_cross 2.5, 13 at 5.0 (zeta 1.4) and 13.5 at 10.0 (zeta 1.0), with no
better final offset (24, 31, 33 mm). On noisy data, less damping only buys
more wobble. Not yet confirmed in the Gazebo sim (the GPU machine was down).
(Athreya, 2026-10-20)

**D60. The stability rule watches the pocket midpoint, not the face centre.**
`go_nogo.check_stability` measures the spread of the midpoint between the two
pocket centres over the last `stability_frames` estimates (limit unchanged,
30 mm). *Why:* the face centre is the middle of whichever points the plane
fit took, which changes every frame. Measured 2026-10-20 in the Gazebo sim
with the robot parked at the standoff: the face centre wandered 56 mm over 20
frames while the pocket centres held to +-2 mm, so every run ended NO_GO at
CHECKING. The midpoint is also the point the forks aim at, so a stable
midpoint is the stability that matters. Synthetic numbers: re-check the real
ZED's scatter on a bag before relying on the 30 mm. (Athreya, 2026-10-22)

**D61. The overhang rule is switched off for now.** `go_nogo.check_overhang:
false`: nothing is assumed to stick out of the pallet towards the robot.
Every GO says "overhang NOT checked", so a GO is never read as "checked and
clear". *Why:* the rule fires on any single point above the pocket opening
more than `overhang_margin` (20 mm) in front of the face plane, and depth
noise alone does that: in the Gazebo sim, with 1 cm of noise and no load,
77 of the pallet's own deck points sat 20-44 mm out and the run was refused
(2026-10-22). The real ZED scatters more. To switch it back on, first make
it noise-robust; the option discussed was per-strip medians across the face
width (5 cm strips, at least 20 points each, the same 20 mm margin).
(Athreya, 2026-10-22)

## Open, with numbers attached

Two predictions the offline harness makes about the real robot. Both are
measurements for lab session B, and both would change a decided tolerance if
they come out badly -- so neither has been pre-emptively changed.

**The stability rule may be too tight.** `go_nogo.max_position_spread: 0.030`
over five frames. Swept in the harness, the rule refuses runs at **8 mm** of
independent per-frame position noise (11 of 12 completed) and **every** run at
15 mm; up to 5 mm it refuses none. What is unknown is whether the real sensor
is that noisy *frame to frame*: `docs/dataset.md` records accuracy against a
tape (face position within 5-10 mm), which is a different quantity, because
depth noise is strongly correlated between consecutive frames of a stationary
scene. Nobody has measured the per-frame spread. If it is above ~5 mm, the
tolerance is wrong and that is a decision to take with the number in hand.

**The LiDAR cross-check's `min_points: 20` is arithmetic, not a measurement.**
See D37. The tier-1 bags carry `/ouster/points`; the achieved counts go in
`docs/dataset.md`.

## Pending (to be decided at the start of each stage)

See the stage lists in `docs/ROADMAP.md`. Stage 5a is closed (D15–D18); the
numbers it decided reach `config/params.yaml` in stage 1a, together with
retiring the lip-based standoff tests that encoded the old 1.45 m.