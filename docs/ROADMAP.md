# Roadmap: from current state to full pallet pickup

Working model: Athreya decides the logic, Claude writes the code. Each stage
starts with a **logic sheet** (the decisions below, with recommended defaults),
then implementation, then offline tests on synthetic data and recorded bags,
then a robot test by Athreya. Decisions go into `docs/decisions.md`.

## Scenario

Nav2 has already brought the robot to a spot where a pallet is somewhere in
the camera's view (not necessarily straight ahead). Available: the map, the
pallet visible to LiDAR and camera, and the drive controls. Task: pick up the
pallet: align with the pockets, insert the (virtual) forks, lift, back out,
hand control back to Nav2.

## Target flow

```
IDLE
 │ action goal (optional: expected pallet pose, pallet ID)
 ▼
CONFIRMING ── same pallet over N frames, tracker confident ─────► not found: NO_PALLET
 ▼
CLASSIFYING ─ pocket side / long side / ambiguous
 │   long or ambiguous → REPOSITIONING (viewpoint goal to Nav2) → CONFIRMING   [stage 9]
 ▼
APPROACHING ─ coarse leg to line-up point (odometry) → re-acquire → final leg (perception)
 ▼
CHECKING ──── stopped; alignment, pockets, fork height → GO / NO-GO
 │   NO-GO retryable → back to APPROACHING (max retries)
 │   NO-GO not retryable → result NO_GO + reason
 ▼ GO
LOWERING ──── virtual forks to pocket height
 ▼
INSERTING ─── straight, slow, odometry + LiDAR distance stop
 ▼
LIFTING ───── virtual forks up, mark loaded
 ▼
BACKING_OUT ─ reverse straight to standoff
 ▼
DONE: result INSERTED; control returns to Nav2

Any state: cancel, stale sensor, obstacle, lost pallet on final leg → stop → ABORTED
```

## Background processes

| Process | Job | Implementation |
|---|---|---|
| Localization | Robot pose in `map` and `odom` | Existing (base odometry, SLAM/Nav2 container) via tf; clocks synced (chrony) |
| Pallet tracker | Steady pallet pose + confidence over frames, in `odom` | New node |
| Sensor cross-check | LiDAR vs camera agreement on the face | Part of tracker |
| Safety | Obstacle zone, watchdogs on stale data, speed limits, base command timeout | Nav2 collision monitor + new watchdog |
| Arbitration | Exactly one source drives the base | `twist_mux`: safety > pallet controller > Nav2 |
| Behaviour layer | Sequencing, decisions, action feedback | Coordinator state machine |
| Virtual forks | Fork height, "loaded" flag, timed motion | New fork simulator node |
| Recorder | Rosbag of every run | New, started by coordinator |
| Health | Sensor rates, perception time, tf freshness | New diagnostics node; coordinator refuses to start if unhealthy |

## Build order

Revised 2026-10-14 after the LiDAR coverage measurement (D13) and the Nav2 idle
check. Two things forced a re-order:

- **Stage 3 could not be built where it was.** It listed its standoff as coming
  "from fork geometry, stage 5" — two stages later. The standoff is read by
  stages 3, 4, 6 and 8, and it is not a perception number: it is set by how far
  the virtual tines reach, because they must start outside the pallet. So the
  geometry decisions split off as **stage 5a** and go first (D14).
- **The two questions that could have invalidated design work are now answered.**
  The Ouster does see a 145 mm pallet at close range, with a measured 0.832 m
  forward blind radius and the face held to a 0.24 m bumper gap (D13); Nav2 is
  silent on `/cmd_vel` while idle (D10). Nothing downstream is waiting on them.

What remains blocked is physical, not analytical: the pallet's long side is
unmeasured, and **no bag contains the pallet and the LiDAR together**, so the
tracker's cross-check cannot be tested on data yet. Both are gathered in lab
session A below, which is why the stages that need them sit after it.

Order, with what each one waits on:

| # | Stage | Waits on | Where |
|---|---|---|---|
| 5a | Fork and insertion **geometry** (decisions only) | — | desk |
| 0 | Foundation: action interface, `twist_mux`, watchdog/health, recorder | — | desk |
| 1a | Perception: measured-pallet params, the two open bugs, synthetic rewrite | — | desk |
| 4 | GO/NO-GO | 5a (tolerances reference the standoff) | desk |
| — | **Lab session A** — measurements and bags, no driving | 1a | **lab** |
| 1b | Side classification | lab A (the long side) | desk |
| 2 | Pallet tracker and LiDAR cross-check | lab A (a bag with both sensors) | desk |
| 3 | Approach: two-leg controller | 5a, 2 | desk |
| — | **Lab session B** — first driving: approach and GO/NO-GO only | 0, 3, 4 | **lab** |
| 5b | Virtual fork simulator node | 5a | desk |
| 6 | Insert, lift, back out | 5a, 5b, lab B | desk |
| 7 | Coordinator: full flow and handover | 6 | desk |
| — | **Lab session C** — insertion and the full run | 6, 7 | **lab** |
| 8 | Fork overlay and demo stream | 7 | desk |
| 10 | Evaluation | 7, lab C | desk + **lab D** |
| 9 | *Stretch:* repositioning to the pocket side | 7 | desk + lab |

The lab sessions are the critical path, not the code. Each one gates three or
four desk stages, and surprises there cost a whole cycle. Hence one rule for
them: **batch everything physical into the earliest session that can answer
it**, even if the stage that needs the answer is days away.

## Effort estimate

Day 0–2 are done (survey and scaffold, floor and extrinsic calibration,
perception running end to end). Days 3–14 remain.

| Stage | Desk | Notes on the estimate |
|---|---|---|
| 5a | ½ day | Decisions and arithmetic; no code. |
| 0 | 1 day | `twist_mux` is cheap now that Nav2 is known to be silent when idle. |
| 1a | 1 day | Params are mostly decided (D4, D5); the work is the two open bugs and rewriting `test/synthetic.py` to the measured pallet. |
| 4 | 1 day | 26 tests for it already exist and are deselected — they define the shape. |
| 1b | ½ day | Small once the long side is known. |
| 2 | 1 day | New node, plus the cross-check against a real two-sensor bag. |
| 3 | 1½ days | Two-leg controller with odometry targets; the most logic of any stage. |
| 5b | ½ day | A timed-motion node with no real hardware behind it. |
| 6 | 1 day | Short, but every failure mode is a stop. |
| 7 | 1½ days | The state machine grows from 5 states to 11 plus ABORTED. |
| 8 | 1 day | Downscaled debug stream, per the network rule. |
| 10 | 1 day | Metrics tooling and the results table. |
| | **~11½ days** | against 12 remaining — and that counts no lab time. |

So the honest position: **the desk work fits the remaining days with no slack,
and the four lab sessions are not in the number.** That is not a schedule, it is
a warning. Two consequences, decided now rather than discovered on day 13:

1. **Stage 9 (repositioning) is out** unless everything else lands early. It is
   the one stage whose absence costs nothing but a sentence in the README —
   CLASSIFYING already has a defined answer for the wrong side (NO-GO, "unclear
   side").
2. **Stage 10 is not negotiable.** The portfolio weight sits in measured
   accuracy and a false-GO count, not in the number of states in the machine. If
   time runs short, stage 8 gets a plain overlay and stage 7 gets fewer recovery
   paths, and the trials still happen.

## Lab sessions

### Lab session A — measurements and bags. No driving.

Everything physical that any later stage needs. Nothing here moves the robot
under our control, so it can be done at any time.

- [x] **Measure the pallet's long side** — done, D19. Four-way entry: the
      opening runs 0.020–0.125 m over a 20 mm bottom board, pockets ~0.375 m
      wide with centres 0.525 m apart.
- [x] **Confirm the robot's front overhang** — done, D23. 0.920 m long and
      0.690 m wide as measured, against the URDF's 0.925 and 0.6996. The
      collision box is accurate; the clearance numbers stand.
- [ ] **The bags.** The full plan, with the setup for each one and distances
      given as tape from the front bumper, is in **`docs/dataset.md`** — 25 bags
      in four tiers, every one with `/ouster/points` included. Tier 1 is seven
      bags and unblocks stages 1b, 2, 4 and 6; the rest is the accuracy sweep,
      the tolerance set and the false-GO set. Keep that file as the single list;
      do not duplicate it here.
- [ ] **Nav2 handover smoke test:** send a goal, let it finish, confirm
      `/cmd_vel` goes quiet, confirm the publisher count drops back. Confirms
      D10 on the real stack rather than from a description.

### Lab session B — first driving. Approach and GO/NO-GO only.

The controller drives for the first time. Insertion is **not** enabled.

- [ ] Coarse leg to the line-up point, re-acquire, final leg to the standoff.
- [ ] Verify the stop behaviours actually stop: cancel, pallet lost on the final
      leg, stale depth, unhealthy start.
- [ ] Record every run. These bags are the offline test set for stages 6 and 7.
- [ ] Measure where the robot actually ends up against where perception said the
      pallet was — the first real approach accuracy number.

### Lab session C — insertion and the full run.

- [ ] Insertion with the LiDAR stop, starting with the stop distance set
      deliberately short, then stepped in.
- [ ] Full flow: CONFIRMING through BACKING_OUT, then handover back to Nav2.
- [ ] Lateral drift over the blind portion, measured — the main risk in 5a.

### Lab session D — evaluation trials.

- [ ] The trial protocol from stage 10, including the unsafe scenarios where the
      target is zero false GOs.

## Stages

Each stage lists the decisions Athreya makes. Defaults are proposals only.

### Stage 5a: Fork and insertion geometry — DONE 2026-10-14 (D15–D18)
Taken first because the standoff depends on it and stages 3, 4, 6 and 8 all read
the standoff (D14). Decided:

| | |
|---|---|
| carriage | at the front bumper, x = 0.4625 m |
| tines | 1.150 long × 0.100 wide × 0.040 thick, 0.365 apart, underside 0.030 |
| tip reach | 1.6125 m from `base_footprint` |
| **standoff** | **1.75 m** (tips 0.1375 m clear of the face) |
| **insertion stop** | face at **0.700 m**, a **0.2375 m** bumper gap |
| insertion travel | 1.05 m — pockets in camera view for the first 0.58 m, face in LiDAR view for all of it |
| insertion depth | 0.9125 m = **76 %** of the pallet; tips 0.2825 m short of the far end |

Still open here: the line-up distance, which belongs with the approach
controller (stage 3).

### Stage 0: Foundation
Action interface update (goal prior + pallet ID, new phases and outcomes);
`twist_mux` config; health/watchdog node; run recorder; time sync check.
Decisions settled: D8 (speeds), D9 (watchdogs), D10 (arbitration), D11
(recording, deferred), D12 (health check).

### Stage 1a: Perception on the measured pallet — DONE 2026-10-15
Carries the stage 5a numbers into `config/params.yaml`: `standoff_distance`
1.45 → **1.75**, `forks.carriage_x` 0.0 → **0.4625**, `forks.bottom_height`
0.052 → **0.030**, and a new insertion stop distance of 0.700 m. The lip-based
standoff tests in `test_params.py` and `test_approach.py` encode the old
derivation and are retired with it; the replacement assertions are that the
standoff clears the tip reach and that the stop distance stays inside the
LiDAR's measured coverage.

Then params from D4 and D5 (already in `params.yaml`), plus the two bugs the first
real bag exposed: `find_pockets` is called without `min_points_per_bin`, so
`pocket_min_points_per_bin: 2` is dead; and the pocket-height `bottom = 0.0`
fallback, which on this pallet is not a fallback at all but the correct answer,
since the opening starts at the floor. `test/synthetic.py` still builds an EPAL
pallet with a bottom board and has to be rewritten to the measured one, and the
lip-based standoff tests in `test_params.py` and `test_approach.py` retired.
- Whether the pocket's lower bound is taken as the fitted floor by construction,
  or still measured per pocket (default: the fitted floor — there is nothing
  else to see, and it is what the real pallet is)

### Stage 1b: Side classification — DONE 2026-10-15 (D19–D22, D26)
The long side is measured, and the two faces differ in four independent ways
(D22). Decided: classify on **pocket spacing** (0.365 vs 0.525 m, a 160 mm gap
against ~13 mm of measurement error), confirm with face width, require both to
agree. Long-side insertion is allowed behind `pallet.long_side.insertable`
(D20), and the insertion geometry is per side (D21) — the short-side stop
distance would drive the tips 0.118 m out of the back of the long side.
- Still open: what counts as ambiguous when the two discriminators disagree, and
  what to do then (NO-GO "unclear side" for now; reposition in stage 9)

### Stage 2: Pallet tracker
- How many frames to confirm a pallet
- How far apart (position, angle) two frames may be and still count as the same
  pallet
- How confidence is computed, and the threshold to start approaching
- LiDAR cross-check: allowed disagreement before the estimate is rejected. D13
  measured what is available to compare — face plane and, at the standoff, the
  pocket gap pattern — so this decision can name which of the two it uses.

### Stage 3: Approach
- Line-up distance (the standoff comes from 5a)
- Controller: rotate–drive–rotate (default) or smooth curve
- Position and angle tolerances for "arrived"
- Re-acquire at the line-up point: how long to look, what if not found
- When losing sight aborts (final leg only, default)

### Stage 4: GO/NO-GO
- The checks and their tolerances (lateral offset, yaw, both pockets, pocket size
  vs nominal, overhang, obstruction, estimate stability)
- Which NO-GOs are retryable, and how many retries

### Stage 5b: Virtual fork simulator
The node: fork height, the "loaded" flag, timed motion. Geometry comes from 5a.
- Lowering height: pocket centre from perception, or nominal
- Lift height and the motion speeds

### Stage 6: Insert, lift, back out
- Insertion speed (D8 default 0.05 m/s); the LiDAR stop distance from 5a
- **Whether insertion also tracks the face laterally.** D13 found the LiDAR
  holds 2 beams on the face through the whole insertion, so lateral and yaw
  error stay observable where D7 assumed they would not. Default: keep D7's
  blind-on-odometry behaviour and use the LiDAR only as a stop, with the face
  measurement logged; revisit with the lab C drift number in hand.
- Back-out distance and speed

### Stage 7: Coordinator (full flow) and handover
- Behaviour on each failure; what is reported to the operator
- Handover back to Nav2 after each outcome

### Stage 8: Fork overlay and demo — rendering done 2026-10-19 (D27, D28)
`fork_overlay.py` is implemented and runs offline through
`tools/live.py --view`, drawing the tines, the measured pockets solid and the
nominal dashed, with the state on the image. The geometry is shared with the
robot: the viewer calls the same `render()` the node will.
- Still open: the ROS side — the node's debug topic, downscaled and low-rate
  per the network rule (640×360 at 5 Hz, never the full stream), and the
  mapping from coordinator state to colour.

### Stage 9 (stretch): Repositioning to the pocket side
- Which short side to pick; free-space check (map, LiDAR); viewpoint distance
- Nav2 drives to the viewpoint; routine resumes at CONFIRMING

### Stage 10: Evaluation
- Metrics and protocol (trials, ground truth); results table for README
