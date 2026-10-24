# Dataset: bags and their ground truth

A bag without tape-measured ground truth cannot be used for evaluation, so every
bag gets a row in the table at the bottom the moment it is recorded. Fill it in
at the robot, not from memory afterwards.

Bags themselves are not in git (see `.gitignore`) — they live on the laptop under
`~/bags`. This file is the index and the recording plan.

## Distances are measured from the front bumper

Everything in this file is **tape from the front bumper to the pallet's front
face**, because that is what you can actually put a tape on. The code works in
`base_footprint` (the point on the floor below the middle of the wheelbase), and
the two differ by a constant:

```
face_x (in base_footprint) = bumper tape + 0.4625 m
```

0.4625 m is the front face of `base_link`'s collision box — confirmed by tape on
2026-10-14 as 0.920 m bumper to bumper against the URDF's 0.925 m (D23).

The distances that matter, in both units:

| bumper tape | `face_x` | what it is |
|---|---|---|
| **24 cm** | 0.70 m | short-side insertion stop (D16) |
| **49 cm** | 0.95 m | long-side insertion stop (D21) |
| 59 cm | 1.05 m | below this the camera sees **none** of the pallet |
| **71 cm** | 1.17 m | last distance with the pocket top edge in frame |
| 97 cm | 1.43 m | last distance with the floor line at the face in frame |
| **129 cm** | 1.75 m | **the standoff** — where CHECKING and GO/NO-GO run |

The camera limits are worth knowing while you work, because they explain what
you will and will not see in each bag: the pallet is only 0.145 m tall and the
bottom of the image reaches the floor 1.43 m out, so **closer than 59 cm of
bumper tape the whole pallet has dropped out of the bottom of the frame.** Bags
closer than that are LiDAR bags; the camera's blindness there is the finding,
not a fault.

## Four tape readings per bag

Do these the same way every time and all three ground-truth numbers fall out.
Lay a straightedge (or a second tape) flat against the bumper face as the zero
line, and measure **perpendicular from that line**, not diagonally.

1. **`dL`** — from the bumper line to the pallet's **left** front corner.
2. **`dR`** — from the bumper line to the pallet's **right** front corner.
3. **`lat`** — sideways offset between a mark at the centre of the front bumper
   and a mark at the centre of the pallet's front face (39.75 cm from either
   edge on the short face, 59.75 cm on the long face). Positive = pallet to the
   **left**.
4. **lens height** — only when the load or tyre pressure has changed. 0.398 m as
   of 2026-10-14.

Then:

```
distance to the face  = (dL + dR) / 2        <- this is the "bumper tape" column
angle                 = asin((dL - dR) / face_width)
```

`face_width` is 0.795 m on the short (pocket) side, 1.195 m on the long side. So
to **set** an angle, slew the pallet until the two corner readings differ by:

| angle | short face (0.795 m) | long face (1.195 m) |
|---|---|---|
| 10° | 13.8 cm | 20.8 cm |
| 20° | 27.2 cm | 40.9 cm |
| 30° | 39.8 cm | 59.7 cm |
| 45° | 56.2 cm | 84.5 cm |

Two tape readings give the angle to a fraction of a degree, which beats a
protractor on a concrete floor.

**Cross-check for the squared-up, centred case:** with the pallet centred, the
sideways gap from each front wheel's outer edge to the corresponding pallet edge
is **5.25 cm** on the short face and **25.25 cm** on the long face. If those two
gaps are equal, `lat` is zero.

## How to record

```bash
C=<zed container>                       # ./pallet_approach/tools/fetch_bag.sh --check lists them

./pallet_approach/tools/fetch_bag.sh --container $C -o <name> \
    -d 10 --with-image --with-lidar
```

- **`-d 10` is enough.** The pallet is not moving; 10 s at 5 Hz is 50 depth
  frames, and the replay tool uses 5.
- **`--with-lidar` on every bag.** It roughly triples the size (~0.4 GB
  compressed), but a bag without it can never be used for the stage 2
  cross-check, and re-recording means re-setting the pallet. 25 bags is about
  10 GB against 35 GB free on the laptop.
- **Push the robot by hand.** Nothing here involves driving — the controller is
  not built yet, and at 24 cm of bumper tape you do not want it to be.
- **After the first bag**, confirm the LiDAR did not throttle:
  `ros2 topic hz /ouster/points` should still read ~10 Hz. If it dropped, see
  the note `fetch_bag.sh` prints about recording two bags instead.

## Park the robot once; move the pallet

**The robot stays where it is for the whole session.** Every distance and angle
below is set by moving the *pallet*, not the robot.

Only the relative pose matters to the sensors, so this is free — and it is
better than driving the robot between bags: its pose relative to the floor stays
identical across every bag, so the floor fit and the extrinsic correction are
the same in all of them, and any variation between bags is attributable to the
pallet rather than to the robot having been nudged. No odometry drift either.

Turning the pallet to show a different face:

- **0°** — short (pocket) face toward the robot. 0.795 m wide, pockets 0.23 m
  wide at 0.365 m centres, opening from the floor.
- **90°** — long face toward the robot. 1.195 m wide, pockets ~0.375 m at
  0.525 m centres, opening from 0.020 m over a 20 mm bottom board.
- **180° is not a new bag.** It shows the opposite short face, which on a EUR
  pallet is identical to the first one.

After every rotation, **re-measure**: turning the pallet in place moves the face
midpoint, so `(dL + dR)/2` changes. Set it back to the target before recording.

## Tier 1 — the seven bags that unblock the most (about 20 minutes)

Stop here if that is all the time there is. These validate the decisions the
rest of the build rests on.

| # | name | side | bumper tape | angle | lat | what it is for |
|---|---|---|---|---|---|---|
| 1 | `pallet_s_129` | short | 129 cm | 0° | 0 | **The standoff.** The one bag with the pallet and the LiDAR together at the distance where CHECKING runs. Tests D13's claim that the pocket gap pattern appears in LiDAR returns. |
| 2 | `pallet_s_071` | short | 71 cm | 0° | 0 | The camera's pocket-top limit. Both pockets should be fully in frame and *only just*. |
| 3 | `pallet_s_024` | short | 24 cm | 0° | 0 | **The insertion stop (D16).** Expect the camera to show nothing of the pallet and the LiDAR to hold 2 beams on the face. That contrast is the point. |
| 4 | `pallet_l_129` | **long** | 129 cm | 0° | 0 | Side classification (D22) at the standoff. Pocket spacing should read 0.525 m, not 0.365 m. |
| 5 | `pallet_l_049` | **long** | 49 cm | 0° | 0 | The long-side insertion stop (D21). Should show 5 LiDAR beams on the face, not 2. |
| 6 | `pallet_s_129_45` | 45° | 129 cm | **45°** | 0 | **The ambiguous case.** Neither face square on. Stage 1b has to detect this and refuse; it is the only bag where the two discriminators should disagree. |
| 7 | `empty_129` | — | — | — | — | Pallet removed, robot untouched. The NO_PALLET test, with both sensors. |

For #6, "45°" means the pallet slewed so a corner points at the robot. **Do not
chase precision on this one** — it exists in order to be ambiguous, so rotate
until a corner faces the robot, measure the distance to that nearest corner, and
write down what you got. It is the one bag where the exact number does not
matter.

Working order, which costs three inward moves, two rotations and one removal:

| | pallet | tape | bag |
|---|---|---|---|
| 1 | short face square on | 129 | `pallet_s_129` |
| 2 | pull it in | 71 | `pallet_s_071` |
| 3 | pull it in | 24 | `pallet_s_024` |
| 4 | push back out, rotate **45°** | 129 | `pallet_s_129_45` |
| 5 | rotate to **90°** (long face) | 129 | `pallet_l_129` |
| 6 | pull it in | 49 | `pallet_l_049` |
| 7 | take the pallet away | — | `empty_129` |

## Tier 2 — the distance sweep (7 bags)

Short side, square on, centred. This is the accuracy-versus-range curve for
stage 10, and the training set for the tracker's confidence.

| name | bumper tape | note |
|---|---|---|
| `pallet_s_060` | 60 cm | right at the edge — the deck is about to leave the frame |
| `pallet_s_080` | 80 cm | camera sees the deck, not the pockets |
| `pallet_s_105` | 105 cm | |
| `pallet_s_155` | 155 cm | |
| `pallet_s_205` | 205 cm | matches the existing `pallet_2m_0deg`, which has no LiDAR |
| `pallet_s_255` | 255 cm | |
| `pallet_s_305` | 305 cm | the far end of useful range; add 355 cm if it still detects |

The exact distance matters less here than in tier 1 — what matters is that the
number you **write down** is the number you measured. Round tape values are a
convenience, not a requirement.

## Tier 3 — angle and lateral tolerance at the standoff (8 bags)

All at **129 cm** of bumper tape, short side, because that is where GO/NO-GO
runs and so where the tolerances have to hold.

| name | angle | corner difference to set it | lat |
|---|---|---|---|
| `pallet_s_129_m30` | −30° | 39.8 cm | 0 |
| `pallet_s_129_m20` | −20° | 27.2 cm | 0 |
| `pallet_s_129_m10` | −10° | 13.8 cm | 0 |
| `pallet_s_129_p10` | +10° | 13.8 cm | 0 |
| `pallet_s_129_p20` | +20° | 27.2 cm | 0 |
| `pallet_s_129_p30` | +30° | 39.8 cm | 0 |
| `pallet_s_129_latm30` | 0° | — | −30 cm |
| `pallet_s_129_latp30` | 0° | — | +30 cm |

Sign convention: **positive angle turns the pallet's left edge away** from the
robot (so `dL > dR`), and **positive lat puts the pallet to the left**.

## Tier 4 — the false-GO set (3 bags)

These are the bags the README number comes from, and the target is **zero false
GOs**. All at 129 cm, short side.

| name | setup |
|---|---|
| `unsafe_blocked_pocket` | A board or box pushed into one pocket so it is not clear. A GO here would drive a fork into timber. |
| `unsafe_box` | The pallet removed, a cardboard box of roughly pallet height in its place. Tests that "two gaps in a flat face" is not the whole detector. |
| `unsafe_overhang` | Pallet in place with a box on top overhanging the front face by ~10 cm, so the face perception sees is not the face the forks meet. |

Tier 3's −30° and ±30 cm bags double as unsafe scenarios, so they are not
repeated here.

## Later, not this session

- **Conditions:** lights off, blinds open, direct sun on the face, clutter either
  side of the pallet. Worth a second session once the pipeline is known-good, so
  a failure can be attributed to the lighting rather than to the code.
- **Long-side sweep:** `pallet_l_080` / `_155` / `_205`, once side
  classification is working and worth characterising over range.
- **Run recordings:** every driven run from lab sessions B and C, which become
  the offline test set for stages 6 and 7.

## Evaluating a bag

`tools/evaluate_bags.py` runs the robot's perception (same parameters, same
extrinsic correction) on every depth frame of each bag, and compares the
pockets with the tape readings in **`docs/bag_truth.yaml`** -- enter `dL`, `dR`
and `lat` there for every new bag, in metres, with `measured: true`:

    cd pallet_approach
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ../venv/bin/python tools/evaluate_bags.py ~/bags/pallet_s_* --csv /tmp/eval.csv

It reports, per bag: frames with a pallet and with both pockets, the pocket
midpoint's error against the tape (median and 95th percentile), the yaw error,
spacing / width / height against the measured pallet, and the frame-to-frame
scatter (which needs no tape at all). Rows whose truth is only a target
distance are flagged and are not accuracy numbers.

## Bags

**The eight bags from 2026-10-15 carry their _target_ distances, not measured
ones.** They were recorded to the targets below; the `dL`/`dR`/`lat` readings
have not been transcribed yet. Until they are, these bags are good for
behaviour ("does it find two pockets?") but **not for an accuracy number**,
because the error would be against an intended distance rather than a measured
one. Mark the row ✓ once the tape readings are in.

| name | bumper tape (cm) | `face_x` (m) | side | angle (°) | lat (cm) | rate | topics | lens height (m) | notes |
|---|---|---|---|---|---|---|---|---|---|
| `pallet_s_129` | 129 *(target)* | 1.752 | short | 0 | 0 | 5 | depth+image+LiDAR | 0.398 | face +5 mm, 2 pockets, spacing −3 mm |
| `pallet_s_071` | 71 *(target)* | 1.172 | short | 0 | 0 | ~9 | depth+image+LiDAR | 0.398 | face +10 mm, **0 pockets — band below the frame, expected** |
| `pallet_s_024` | 24 *(target)* | 0.703 | short | 0 | 0 | ~13 | depth+image+LiDAR | 0.398 | camera blind; LiDAR face +14 mm |
| `pallet_s_129_45` | 129 *(target)* | 1.752 | 45° | ~45 | 0 | ~18 | depth+image+LiDAR | 0.398 | the ambiguous case, for stage 1b |
| `pallet_l_129` | 129 *(target)* | 1.752 | **long** | 0 | 0 | ~22 | depth+image+LiDAR | 0.398 | face +10 mm; 2 pockets **only with long nominals** |
| `pallet_l_049` | 49 *(target)* | 0.953 | **long** | 0 | 0 | ~26 | depth+image+LiDAR | 0.398 | camera blind; LiDAR face +10 mm |
| `pallet_l_024` | 24 *(target)* | 0.703 | **long** | 0 | 0 | ~28 | depth+image+LiDAR | 0.398 | extra bag; LiDAR face +6 mm |
| `empty_129` | — | — | — | — | — | ~32 | depth+image+LiDAR | 0.398 | NO_PALLET test — **but the room is not empty**, see below |
| `pallet_2m_0deg` | 154 | 2.000 | short | 0 | 0 | 5 | depth+image, **no LiDAR** | 0.398 | first real bag, 2026-10-14 |
| `floor_cam_pose1` | — | — | — | — | — | 5 | depth | 0.398 | bare floor, heading 1 |
| `floor_cam_pose2` | — | — | — | — | — | 5 | depth | 0.398 | bare floor, heading 2 (173° from pose1) |
| `floor_lidar_pose1` | — | — | — | — | — | 10 | LiDAR | 0.398 | bare floor; `lidar_coverage.py` input |
| `floor_lidar_pose2` | — | — | — | — | — | 10 | LiDAR | 0.398 | bare floor, second heading |

### Two faults in these eight bags, both in the recording script

**1. The depth rate climbs through the session: 4.8 → 8.7 → 12.8 → 18.2 → 22.2
→ 25.7 → 27.6 → 32.4 Hz**, on a topic that `topic_tools throttle` was told to
pin at 5. One stray throttle node survived each run and they all publish to the
same `..._throttled` topic, so the recorded rate is a multiple of the intended
one. Cause: `ros2 run` launches the node as a *child* process, so signalling the
pid the script held did not reach it. Fixed in `record_bag.sh` — the throttles
now start under `setsid` and are signalled as a process group, strays are swept
before each run, the publisher count is checked before recording starts, and the
achieved rates are printed at the end. The depth frames themselves are fine;
the bags are just larger than intended.

**2. The Ouster drops from 9.4 Hz on the first bag to 2.6–4.6 Hz on the rest.**
This is the risk flagged before the session: `/ouster/points` is RELIABLE and
crosses the network from the robot PC into the Jetson's container, so a
subscriber that cannot keep up slows the publisher — made much worse by the
stray throttle nodes eating the Jetson's CPU. **For these bags it costs
nothing** (the pallet is static, so 21–37 scans is plenty), but it is a warning
for lab sessions B and C, where the insertion stop depends on the LiDAR. What
matters there is *latency*, not rate, and latency has not been measured.
Re-check the rate once the throttle fix is in: if it returns to ~10 Hz, the
strays were the cause.

### The room is not empty in `empty_129`

502 returns remain at pallet height in front of the robot — x 0.49–1.92 m,
y −0.90 to +0.35 (mostly to the robot's right), up to 0.25 m above the floor.
So this bag tests "reject clutter that is not a pallet", which is a better test
than "see nothing", but it has to be labelled as such rather than treated as a
clean negative. Worth a photo of the lab next time.
