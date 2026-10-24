# Daily log

One short entry per day: what got done, what broke, what the numbers were.
What broke and how it was fixed is the most useful part, so it is written
down while it is fresh rather than reconstructed later.

---

## Day 1 — 2026-10-14 — the floor, measured

**Done**

- Tape-measured the camera lens height above the floor: **0.398 m**.
- Propagated it: `floor.expected_z`, the overlay's fallback floor, the synthetic
  clouds' default floor, and the standoff.
- Bag tooling finished: `fetch_bag.sh` records on the Jetson and moves the bag
  to the laptop over the hotspot, verified before deleting the remote copy.

**What the measurement settled**

The URDF disagreed with itself about the floor by 25 mm. The tape measure says
which half was right:

| source | implied `base_link` height | error |
|---|---|---|
| wheel geometry | 0.2348 m | −0.2 mm |
| `base_footprint` joint | 0.210 m | **−25 mm** |

So `base_footprint` is **not** on the floor — it is 25 mm above it, and the
floor is at `z = -0.025` in that frame. The wheel geometry was right all along.

**What broke / what I learned**

- I had been saying "z = 0 is the floor in base_footprint". That was wrong, and
  the correction is now in `docs/SYSTEM.md` and every docstring
  that claimed it. What saved it from being a real bug is that the pipeline
  never used z = 0: `remove_floor` and `find_pockets` take the *fitted* floor as
  an argument, so heights are relative to a measurement. The absolute number
  appears in only four places, all of them parameters.
- **The 25 mm mattered for the standoff.** At 1.40 m the bottom lip of the
  pocket opening is only 7 mm inside the frame — tyre compression or a little
  pitch under braking would push it out, the measured opening would read short
  against a 30 mm tolerance, and a sound pallet would get a spurious NO-GO.
  Moved to **1.45 m**, which gives 26 mm and brings a 4 mm strip of floor back
  into frame. `test_params.py` now fails if anyone tightens it below 15 mm.
- The synthetic clouds default to a floor at −0.025 rather than 0, so anything
  that assumes z = 0 is the floor fails in the test suite instead of in the lab.
- `fetch_bag.sh` had `-o BatchMode=yes` in its ssh options, which disables
  password and passphrase prompts — so it failed on a host that ssh'd fine by
  hand, and I had suppressed the error that would have said so. Removed, plus
  one shared ControlMaster connection so any password is asked once.
- ROS is not on PATH in a non-interactive shell in those containers. Sourcing
  now happens once in `fetch_bag.sh`'s remote-command helper, and
  `record_bag.sh` sources it itself so it works piped in or run by hand.

**Numbers**

| | |
|---|---|
| camera lens above the floor | **0.398 m** (measured) |
| `base_link` above the floor | 0.235 m |
| floor in `base_footprint` | **−0.025 m** |
| `base_footprint` joint error | 25 mm |
| pocket bottom lip in frame from | 1.38 m |
| standoff | **1.45 m** (26 mm lip margin, 89 px pocket height) |

- Recorded the first pallet bag (2.0 m, square, 800 mm face) and four bare-floor
  bags — camera and LiDAR, two headings 173 deg apart.
- Wrote `tools/floor_plane_check.py`: fits the floor in `base_footprint` from
  either sensor and separates a robot-fixed mounting tilt from a world-fixed
  floor slope.

**The extrinsic result**

| | mounting tilt | floor slope | floor height |
|---|---|---|---|
| camera | **0.868 deg** (pitch +0.698, roll -0.516) | 0.207 deg | -0.0246 m |
| LiDAR | 0.152 deg | 0.289 deg | +0.0130 m |

The floor is level to ~0.2 deg, so it was never the floor. The **camera's URDF
orientation is wrong by 0.87 deg**, while the LiDAR on the same box comes out
level — so the box is fine and the camera on it is not. That is 22 mm of
vertical error at the standoff and 30 mm at 2 m, against a 30 mm pocket-height
tolerance, so it has to be corrected before pocket geometry means anything.

The camera's mounting *height* is right: both poses put the floor at -0.0246 m,
within 1.7 mm of the -0.025 m the tape measure implied. Two independent methods
agreeing to 2 mm.

Bonus: the LiDAR's height in the URDF is 38 mm out (its orientation is fine).
Not ours to fix, but another sign the accessory mounts are nominal, not measured.

**What I learned**

- My first tilt estimate from the pallet bag ranged from 0.86 to 1.49 deg
  depending on how I selected points, because the pallet contaminated the
  far-field bins. On bare floor the same fit has a 2.4 mm RMS residual instead
  of 7.8 mm and the answer is stable. Measure the thing you want to measure on
  a scene that contains only that thing.
- `/odom` gave the turn as 173 deg, so "almost 180, not measured" was good
  enough. Deriving the heading from data beat measuring it by hand.
- The LiDAR's per-point noise is 10x the camera's (26-29 mm vs 2.4 mm) at
  grazing incidence on the floor, but with 43k points the plane parameters are
  still precise. Noise per point and precision of a fit are different things.

**The correction**

Built `pallet_approach/calibration.py` plus a `frames.extrinsic_correction`
block in params.yaml holding the two *measured* numbers (not the correction
itself, so the sign cannot be entered backwards). Both nodes apply it to the
transform they get from tf. Verified on the same four bags with
`floor_plane_check.py --correct`:

| | before | after |
|---|---|---|
| camera mounting tilt | 0.868 deg | **0.190 deg** |
| vertical error at 2 m | 30 mm | 6.6 mm |

The residual 0.190 deg matches the LiDAR's own 0.152 deg, and the two sensors'
floor-slope estimates differ by 0.1 deg, so that is the noise floor for one
session. Correcting further would be fitting noise.

Only the rotation is corrected -- the translation stays, because the tape
measure confirmed the height. Rotating the whole 4x4 would have swung the
camera's position 8 mm sideways to fix its angle.

**The floor is a plane, not a height**

Changed `remove_floor` and `find_pockets` to take the fitted floor as a `Plane`
(normal + offset) and measure perpendicular distance to it, instead of taking a
scalar height and subtracting z. Reason, demonstrated and now a test: an
extrinsic rotation error turns the whole observed scene, and a rotation is an
isometry -- so point-to-plane distance is *exactly* unchanged, while a
z-subtraction leaks the error. Measured on the real error: 4.5 mm leaked into a
z-based pocket height, 0.0 mm into the perpendicular one. The test holds even
with a deliberately absurd 3 deg error.

So the robustness comes from the invariance, not from the calibration. Keep the
calibration anyway -- it makes the overlay truthful and removes a known bias
from the driving geometry -- but the pipeline no longer depends on it being
right. Also corrected my earlier overstatement: I had claimed 30 mm of error at
2 m, which is the error in an *absolute* height above base_footprint, not in
anything the pipeline actually measures.

**Open**

- Check the fork overlay lines up once it is written: that is the visual
  confirmation of this calibration.
- Yaw, x and y are still uncalibrated -- a plane cannot see them. Needs a box
  at a taped position.
- Day 4 intrinsics calibration not done. We only read camera_info; fx drifted
  949.673 -> 949.409 between two sessions, which is a good motivating example.
- Day 5 depth noise curve not done, though the plane-fit RMS of 2.4 mm over
  1-4 m is a first hint.
- Real pallet dimensions still to check against the EPAL nominals.

## Day 4 — 2026-10-15 — eight bags, and the cross-check confirmed

**Done**

- Eight tier-1 bags recorded, camera and LiDAR in one bag for the
  first time. Ran the pipeline on all of them.
- Fixed two faults in `record_bag.sh` that the session exposed.

**The headline: the standoff bag is the best result so far**

`pallet_s_129`, short side at the new 1.75 m standoff:

| | measured | vs tape | previous best (2.0 m bag) |
|---|---|---|---|
| face x | 1.7566 | **+5 mm** | −31 mm |
| pockets found | 2 of 2 | | 2 in 4 frames of 5 |
| pocket width | 0.2173 / 0.2318 | −13 / +2 mm | −5 mm mean |
| spacing | 0.3619 | **−3 mm** | +13 mm |

And **that kills my own explanation of the old −31 mm.** On day 2 I attributed
it to an uncalibrated camera x offset, reasoning that a ground plane cannot
constrain x. But a constant offset would show up identically at every range,
and at 1.75 m the error is +5 mm, at 1.17 m it is +10 mm, at 1.76 m on the long
side +10 mm. Three consistent small positives do not contain a −31 mm constant.
The likelier explanation is the day-2 tape reading itself: this session used a
defined protocol (straightedge on the bumper, perpendicular, two corners
averaged) and the old one did not. A hypothesis that only ever had one data
point behind it, and I stated it as the probable cause.

**A label in my own plan was wrong**

I had written that 71 cm of bumper tape was "the last distance with the pocket top
edge in frame", and it is -- but that is not the same as the last distance where
the pockets can be *measured*. The detection band is 0.035-0.075 above the
floor, and the band's top leaves the frame at a face distance of 1.24 m, the
whole band by 1.34 m. So `pallet_s_071` finds the face (+10 mm) and **zero
pockets**, correctly. The real camera limit for pocket measurement is about
88 cm of tape, not 71. The 1.75 m standoff is comfortably clear of it either
way, so nothing downstream moves -- but the number in `docs/dataset.md` was
describing the wrong boundary.

**The long side needs its own nominals, exactly as D21/D22 said**

`pallet_l_129` found the face (+10 mm) and zero pockets, because
`pocket_max_gap_width: 0.32` rejects the long side's 0.375 m pockets outright.
Swap in the D19 nominals and both pockets appear. So side classification is not
optional polish: with one set of constants the pipeline silently reports a
pallet with no pockets, which a naive GO/NO-GO would read as "damaged pallet"
rather than "wrong face".

The long side's geometry then reads ~50 mm off the tape (width −51, spacing
+52, i.e. the inner block edges pushed outward about 50 mm each). Against a
0.02 m histogram bin that is 2-3 bins, the same quantisation that made the
short side read +13 mm on day 2, just on a bigger pallet feature. Not yet
explained to my satisfaction and it is a stage 1b item, not a mystery to leave
in a docstring.

**The two sensors doing the job they were split for**

| bag | face x, camera | face x, LiDAR | vs tape |
|---|---|---|---|
| `pallet_s_024` | nothing -- below the frame | 0.717 | **+14 mm** |
| `pallet_l_049` | nothing -- below the frame | 0.963 | **+10 mm** |
| `pallet_l_024` | nothing -- below the frame | 0.708 | **+6 mm** |

That is D16 and D21 demonstrated rather than argued: at the insertion stop the
camera is blind and the LiDAR holds the face to about a centimetre. The
geometry predicted it; the bags show it.

**What broke / what I learned**

- **`record_bag.sh` leaked a throttle node per run.** The recorded depth rate
  climbed 4.8 -> 8.7 -> 12.8 -> ... -> 32.4 Hz across the eight bags, because
  `ros2 run` launches the node as a *child* and signalling the pid the script
  held never reached it, while every survivor republished to the same
  `..._throttled` topic. Now: `setsid` plus process-group signals, a sweep for
  strays before recording, a publisher-count check that refuses to start, and
  the achieved rates printed at the end. The last of those is the real fix --
  the bug was invisible for eight bags because nothing ever looked.
- **The Ouster fell to 2.6-4.6 Hz** after the first bag. Predicted risk
  (RELIABLE QoS across the network into the Jetson), aggravated by the strays
  eating CPU. Harmless for static bags, not harmless for the insertion stop, and
  what matters there is latency, which is still unmeasured.
- **My LiDAR column classifier was wrong twice before it was right.** First I
  counted floor returns as pallet, because the LiDAR's floor sits ~30 mm high in
  `base_footprint` and my band started 30 mm up. Then I classified an azimuth
  column as wood if its *nearest* return was at the face, so a single stray point
  filled a pocket. Even after switching to a majority vote it disagreed with the
  camera on the long side, where the camera's unrolled-face plot is unambiguous.
  Conclusion recorded rather than papered over: **that ad-hoc classifier is not
  evidence at 1.75 m on the long side**, and the LiDAR gap analysis is stage 2
  work with a real algorithm. I was close to reporting its answer as a finding.
- **`empty_129` is not empty.** 502 returns at pallet height, 0.5-1.9 m ahead
  and mostly to the right, up to 0.25 m high. It is a clutter-rejection test, not
  a clean negative, and needs labelling as such.
- **The bags cannot produce an accuracy number yet.** They were recorded to
  *target* distances; the `dL`/`dR`/`lat` readings have not been transcribed. The
  errors above are against intended distances, which is fine for "does it find
  two pockets" and not fine for a results table.

**Open**

- Transcribe the tape readings into `docs/dataset.md` before any of these bags
  is used for an accuracy claim.
- Re-check `/ouster/points` rate with the throttle fix in; measure its *latency*
  before lab session C.
- Explain the long side's ~50 mm width/spacing bias (stage 1b).

## Day 2 — 2026-10-14 — perception runs end to end

**Done**

- `perception.py` complete. 29/29 of its tests pass and it runs on the
  real bag, returning a full `PalletEstimate` on all five frames.
- `tools/replay.py`: replays a bag through every stage, prints a table and draws
  nine panels, ending with the detected pockets projected back onto the image.
  `tools/_bagutil.py` holds the bag reading, shared with `floor_plane_check.py`.

**First measurements, pallet_2m_0deg, five frames**

| | mean | vs measured pallet |
|---|---|---|
| pocket width | 0.2245 | -5 mm (tol +-40) |
| pocket height | 0.0869 | **-13 mm, worst -41** (tol +-30) |
| pocket spacing | 0.3780 | +13 mm |
| face x | 1.9694 | -31 mm vs the 2.000 m tape |

Two pockets found in four frames of five. Frame 2 is a 19 deg yaw outlier --
exactly what the coordinator's confirm_frames gate exists to reject.

**What broke / what I learned**

- Four of my params.yaml values were wrong, all the same way: picked by eye and
  inconsistent with `voxel_size`. `cluster.min_samples` 30 (nothing can reach it
  on a 2 cm surface -- every point became noise), `cluster.min_cluster_points`
  300 (rejected a cluster containing the whole face, which voxelises to 228),
  `face.min_points` 300 (a real face fit gets 139-146). All now derived from the
  geometry and pinned by tests in `test_params.py` so changing `voxel_size`
  cannot silently break them again. Eyeballed parameters need validating end to
  end, not just reading plausibly.
- My synthetic depth image rendered 1.6 % valid pixels against a real frame's
  84 %, so the end-to-end test was failing on sparsity rather than geometry.
  `depth_image` splats points, so the source cloud has to be sampled finer than
  a pixel at the test range.
- The real pallet's front surface spans **26 mm** from deck boards to lower
  blocks, so a 15 mm RANSAC inlier band samples mostly deck and starves the gap
  analysis -- one pocket instead of two. Fixed by slicing a +-60 mm slab around
  the fitted face plane instead of using the inliers. Measure the thing, then
  pick the threshold.
- Pocket spacing is +13 mm consistently against a 20 mm histogram bin: gap edges
  are quantised outward by up to a bin. Bin width is a measurement choice, not
  just a performance one.
- Face x is -31 mm consistently. Probably not perception: a ground plane cannot
  constrain camera x, y or yaw, and a constant camera x offset looks exactly
  like this. The box-at-a-taped-position test is still outstanding.

**Open**

- `find_pockets` is called without `min_points_per_bin`, so
  `pallet.pocket_min_points_per_bin: 2` is dead and the default 1 is used.
- Pocket height reads short: the `bottom = 0.0` fallback, since `remove_floor`
  deletes the bottom board and leaves nothing below the band inside a gap.
- `go_nogo.py` not written -- 26 of the 27 remaining test failures.
- Still zero commits.

## Day 5 — 2026-10-15 — it runs in real time, and it knows which face it sees

**Done**

- Stage 1a and 1b built. 175 tests pass, up from 148; the 26 `go_nogo` failures
  are unchanged and still waiting on that file.
- The pipeline went from **4.4 Hz to 11.2 Hz** with no change to any answer.
- `tools/live.py`: replays a bag at wall-clock speed through the node's own
  rate-limiting policy, so the real-time behaviour can be watched without the
  robot.

**The speed work, and why it was not an algorithm problem**

Measured per stage on a real frame first, which said 84 % of the time was in
two functions:

| stage | before | after | what changed |
|---|---|---|---|
| `voxel_downsample` | 110 ms | **14 ms** | one int64 key instead of `np.unique(..., axis=0)`; `bincount` instead of `np.add.at` |
| `cluster_points` | 77 ms | **36 ms** | DBSCAN `algorithm='ball_tree'` |
| total frame | 229 ms | **90 ms** | |

Both are the same lesson. `np.unique(x, axis=0)` sorts an (N, 3) array row by
row, which NumPy does by viewing it as a structured dtype and falling back to a
slow comparison sort; packing the three cube indices into one integer makes it
sort a 1-D array for an identical answer. `np.add.at` is an *unbuffered* ufunc
and the slowest scatter-add NumPy has. Neither is a better algorithm — the
complexity is the same — they are the same algorithm spelled in a way NumPy can
actually run fast. Both are pinned by tests against a naive reference, because
"faster and identical" is only worth having if the second half stays true.

**Side classification works on both faces of the real pallet**

| bag | expected | got | face x | width | pockets | spacing |
|---|---|---|---|---|---|---|
| `pallet_s_129` | short | **short** | +5 mm | 0.781 | 2 | 0.3619 |
| `pallet_l_129` | long | **long** | +10 mm | 1.204 | 2 | 0.5774 |
| `pallet_s_071` | short | ambiguous | +10 mm | 0.790 | 0 | — |
| `empty_129` | none | no pallet | — | — | — | — |

`pallet_s_071` returning ambiguous is correct and is the reason the band
constraint is now a test: at a face distance of 1.17 m the detection band has
left the bottom of the frame, so there are no pockets to measure spacing from.

**What broke / what I learned**

- **A test caught a real bug in code I had just written.** `FrameGate` used
  `0.0` as the "no frame yet" sentinel, which is also a perfectly valid reading
  from a monotonic clock, so the first frame after t = 0 behaved differently
  from every other. Falsy sentinels that are also valid values.
- **Then the fix exposed a second one.** The rate test failed on
  `3 * 0.1 == 0.30000000000000004`, and my first instinct was to call it a test
  artifact and write the test around it. It is not: set `max_rate` to the
  camera's own 37 Hz and arrivals land 27.027 ms apart against a 27.027 ms
  interval, where the comparison goes either way on the last bit and the
  achieved rate silently halves. The gate now allows a frame a thousandth of an
  interval early. I nearly papered over a real robustness hole because it
  surfaced as an ugly test.
- **I wrote a test asserting what I had assumed rather than what was decided.**
  `test_width_alone_cannot_decide` asserted that a clipped long face still
  classifies as `long`. D22 says both signals must agree, so it is AMBIGUOUS —
  a refusal on a good pallet, and the conservative direction on purpose. The
  test is now named for the consequence and D26 records it, because it will
  look like a bug the first time it stops a run.
- **The 45 deg bag cannot test what it was recorded for.** `pallet_s_129_45`
  returns "no pallet": the pallet is not in any cluster. The clusters at pallet
  height have 13-21 points against the 2308 a square-on pallet gives. That is
  not a gate being too strict — it is ZED stereo returning almost nothing off
  bare wood at grazing incidence. So the AMBIGUOUS path still has no real-data
  test, and the LiDAR cross-check matters more than I thought: at 45 deg it
  sees those faces perfectly well.
- **`pallet_s_024` reports a pallet that is not there** — a face at 1.481 m with
  yaw 135 deg, which is the wall behind the real pallet. It comes out AMBIGUOUS
  (no pockets), so GO/NO-GO would refuse it and nothing unsafe follows, but
  perception is claiming a pallet-shaped thing where there is none. The LiDAR
  would kill it instantly: it sees the real pallet at 0.70 m. Stage 2.
- **I widened the cluster gate for the wrong reason and kept the change.**
  `select_pallet_cluster` matched the *expected* side's width, which cannot be
  right when the gate runs before the side is known. Fixing it did not rescue
  the 45 deg bag — that was the depth, not the gate — but gating on an assumption
  the pipeline has not made yet was wrong regardless.

**The container had never been built**

The first real `docker build` on the Jetson failed after **259 seconds**, on a
backslash. `rosidl_adapter` copies `.msg` comments into the generated IDL and
that text is decoded with Python's `unicodeescape` codec, so the shell
line-continuation in `PalletTask.action`'s usage example --

    #   ros2 action send_goal /pallet_task \\

-- is an invalid escape and the whole interfaces package fails:

    UnicodeDecodeError processing template 'struct.idl.em':
    'unicodeescape' codec can't decode byte 0x5c in position 37

Written on day 0, never exercised until today. A definition file is not
compiled by anything on a laptop; it is compiled inside the container, on the
robot, four minutes into an image build, and the error names a CMake file and
an emPy template rather than the line that is wrong.

`test_interfaces.py` now covers it three ways: a scan for backslashes, a scan
for non-ASCII, and -- when `rosidl_adapter` is importable -- the real
conversion. Verified by putting the backslash back: the laptop reproduces the
Jetson's error verbatim in under a second instead of 259.

**And that test immediately found a second bug**

`floor_height` was declared in `PalletEstimate.msg` and never assigned by
`estimate_to_msg`, so every estimate ever published reported the floor at
**0.000** -- precisely the value this project spent day 1 proving wrong (it is
-0.025). A field that is declared and never filled publishes a plausible
number forever, which is worse than a crash because nothing looks broken. It
now carries the real value, or NaN when no floor was fitted.

**Then the container saw no topics at all**

The node came up active and published diagnostics, but `rate_hz 0` and "no
frames processed". `ros2 topic echo` inside our container said the ZED's depth
topic "does not appear to be published yet", while the same echo inside the ZED
container showed data. Both containers run `--network host` on the same machine
with `ROS_DOMAIN_ID=10`, so this is **discovery** failing, not transport.

The difference is the FastDDS profile. The Jetson is multi-homed -- wifi
10.42.0.40, wired 192.168.1.1, docker0 -- so the ZED container runs with an
explicit `FASTRTPS_DEFAULT_PROFILES_FILE` that picks an interface, and a
participant started without it does not necessarily match.

`build_and_run.sh` had a `DDS_DIR` default of `/ros2_ws/src/scout_nlp/configs`,
which is a path inside the ZED container, not on the host -- so it never found
it, printed `note: ... not found, running without an explicit FastDDS profile`,
and carried on.

**That is the third time a script has quietly continued past a missing
prerequisite.** `fetch_bag.sh` suppressed ssh's own error with `BatchMode=yes`;
`record_bag.sh` never printed the rate it achieved, so eight bags were recorded
wrong; now this. Every one cost more to diagnose than the check would have
cost to write. The script now reads the domain, the RMW and the profile path
off the running ZED container, maps the profile back to a host path through
that container's mounts, and **refuses to start** if it cannot -- printing what
it inspected and the exact `docker cp` to work around it. `DDS_DIR=none` is the
deliberate escape hatch. There is also a `dds` subcommand that shows what it
would use without starting anything.

A note it now emits rather than a refusal: if the ZED container does not run
with `--ipc host`, the two have separate `/dev/shm`, so discovery succeeds and
bulk data over shared memory does not -- a different symptom with the same
flavour, worth recognising before chasing it.

**Three runtime failures in a row, all the same shape**

Once DDS was fixed the node ran and failed three times, each a *join between two
declarations that nothing checked*:

| failure | one side | other side | symptom |
|---|---|---|---|
| `UnicodeDecodeError ... 0x5c` | a comment in `PalletTask.action` | rosidl_adapter's IDL template | build dies after 259 s |
| `ParameterNotDeclaredException('seed')` | `_declare` in the node | `_params` reading them back | node alive, pipeline dead |
| `AttributeError: no attribute 'floor_valid'` | `PalletEstimate.msg` | `ros_conversions` assigning | no estimate ever published |

None of them is subtle. All three survived because the two sides live in
different files and no test looked at both.

And the parameter one had a second cause worth recording: `test_params.py`'s
scraper for `(name, default)` tuples was anchored with `^`, so in

    ('depth.min_range', 0.40), ('depth.max_range', 4.00), ('depth.stride', 2),

it saw the first and silently missed the other two. That test has been
under-checking since day 0 -- it passed while looking at a fraction of what the
nodes declare. A test that is quietly weaker than it claims is worse than no
test, because it is counted as covered.

Both directions are now pinned, for parameters and for messages:

- every declared parameter is in the YAML **and** every parameter read is
  declared (4 nodes each way);
- every `.msg` field is assigned by the converter **and** every field the
  converter assigns exists in the `.msg`;
- the coordinator sets only `PalletTask` fields that exist;
- and the definitions are run through the real `rosidl_adapter` when it is
  importable, which on this laptop it is.

Checking only one direction is never enough, and the two directions fail
differently: declared-but-unused publishes a plausible wrong number forever
(that is how `floor_height` shipped reading 0.000 when the floor is at -0.025);
used-but-undeclared throws. The first is the dangerous one.

**What the failures proved worked**

Each error located the fault precisely enough to clear everything upstream of
it. `_params()` is evaluated at the `estimate_pallet(...)` call, which is after
the tf lookup -- so a `ParameterNotDeclaredException` there meant camera_info
had arrived, the image had decoded, and **tf had resolved**, all of which were
open questions at the time. The 3-4 ms frame time said the same thing: too long
for the camera_info bail-out, far too short for a 150 ms tf timeout.

**The clock: measured, and gone**

Three hypotheses, all killed by the next round of data: an uptime clock (would
give slope 1.0), SVO playback (the camera is live), a frame-count/rate mismatch
(the camera runs at 56 Hz, which the model cannot turn into 1.24). I kept
reasoning from symptoms a layer removed from the cause. The fix was to stop and
measure every publisher at once: `tools/clock_audit.py` fits each topic's
header stamps against arrival time and reports a slope and an offset.

Run on the robot, every topic came back healthy -- depth at slope 1.0003 and
-0.066 s, which is just pipeline latency. **The ZED node had been restarted in
between, and the fault went with it.** So the 1.24x was real, confined to one
node instance, and is not reproducible on demand. Not satisfying, but the audit
is the thing that can tell in thirty seconds next time, and it is worth running
straight after any ZED restart.

Two faults in my own tool in that one run:

- it subscribed RELIABLE, so **`/ouster/points` delivered nothing** -- it
  publishes BEST_EFFORT, and a RELIABLE subscriber gets *nothing at all* from a
  BEST_EFFORT publisher. A diagnostic that silently misses a topic is worse
  than no diagnostic. Now BEST_EFFORT, which is compatible with both.
- it reported `/tf_static` running at **1354x**. tf_static is latched, so the
  whole backlog arrives the instant you subscribe: real stamp spread divided by
  near-zero wall time. The tool now says "latched, rate not measurable, offset
  is". It had invented a spectacular number and presented it exactly like the
  real finding, which is the same sin as the scraper and the drop fraction
  earlier today -- a measurement claiming more than it measures.

**And the Ouster QoS finding is a real one for later.** `docs/SYSTEM.md` said
"everything publishes RELIABLE/VOLATILE, no sensor-data-QoS mismatch to work
around", measured on day 0. That is false for the LiDAR. It never showed up
because `ros2 bag record` adapts QoS automatically, so the bags have LiDAR data
in them -- the first hand-written subscriber to ask for RELIABLE was this tool.
The tracker (stage 2) and the insertion stop (stage 6) must subscribe
BEST_EFFORT, alone in this stack. Corrected in SYSTEM.md and params.yaml.

**A demo you can run from a bag (2026-10-16)**

`tools/live.py --view` replays a bag at wall-clock speed through the real
pipeline and shows two panels: the camera image with the detected pockets
projected back onto it, and the same estimate from above in `base_footprint`.
`pallet_s_129` comes out `short` with both boxes sitting on the holes;
`pallet_l_129` comes out `long`, and the +52 mm spacing error is **visible** --
the boxes' inner edges stop short of the real openings.

That projection is the only check that shows the geometry is right rather than
merely self-consistent. A pocket measured correctly in metres lands on the hole
in the picture; one measured wrongly does not, and no table of numbers says
which you have.

**It did not work the first time, for a reason worth keeping.** The window
never appeared: matplotlib had fallen back to `Agg`. Not a missing toolkit --
`tools/replay.py` calls `matplotlib.use('Agg')` at import because it writes
PNGs, and I had made `live.py` import it to reuse `pocket_corners` and
`project`. Importing a plotting tool for two lines of geometry pinned the
backend before the window could exist.

Both functions now live in `pallet_approach/projection.py`, where they belong:
geometry has no business in a module that chooses a plotting backend, and three
things need them -- the replay, the viewer, and the fork overlay node in stage
8. And `--view` now REFUSES on a headless backend with instructions, instead of
accepting every drawing call, showing nothing, and printing one warning per
frame. That makes the pipeline look fine and the demo look broken, which is
exactly backwards.

Fifth silent fallback of the week, after ssh BatchMode, the unreported bag
rate, the DDS profile note and the RELIABLE subscription that received nothing.
The shape is always the same: a default that is reasonable in isolation, chosen
automatically, and never reported. The fix is always the same too -- say what
was chosen, or refuse.

**Fork overlay (2026-10-19)**

`fork_overlay.py` implemented -- all six functions -- and visible offline
through `tools/live.py --view`. The viewer calls the same `render()` the robot
node will, rather than drawing its own version in matplotlib, so an overlay
that is wrong here is wrong there too.

Two decisions: **one fork setting for both faces** (D27, replacing
D21's per-side geometry) and **nominal pockets dashed beside the measured
ones** (D28). The first needed checking before building on it -- a single
0.365 / 0.040 setting does fit both faces, with 20 mm as the worst clearance
anywhere, and `test_params.py` now derives that window from the measured pocket
geometry rather than checking the configured numbers against themselves.

**And a bug caught by reading the picture, not the code.** The tines
were built against `estimate.floor`, the per-frame RANSAC fit, so they floated.
The original docstring had argued for this -- "keeps the forks parallel to the
floor, which is what a real carriage does" -- and that reasoning is wrong: a
carriage is bolted to the truck, and the truck already follows the floor it
stands on, which is exactly what `base_footprint` encodes. The fitted plane
adds nothing.

What makes it a real bug rather than a stylistic one is the third consequence
(D29): the pockets are measured against that same plane, so a camera
orientation error tilts both the tines and the pockets *the same way* and
cancels on screen. Measured against this robot's real 0.87 deg error: the
fork-to-pocket gap moved 5.4 px with the fitted floor and 16.8 px robot-fixed.
**The overlay was discarding two-thirds of the evidence it exists to collect.**

Tines robot-fixed, pockets perception-fixed. That asymmetry is the whole
diagnostic: the forks are a known rigid object, the pockets are the
measurement, and any mismatch is attributable to one or the other.

**Open**

- `test/synthetic.py` still builds an EPAL pallet with a bottom board; it needs
  the measured geometry and a long-side variant.
- No real-data test of the AMBIGUOUS path (see the 45 deg bag above).
- Jetson timing unmeasured. The laptop does 90 ms/frame; 5 Hz assumes the
  Jetson stays under 200 ms, which nobody has checked.

## Day 3 — 2026-10-14 — the LiDAR's blind cone, and a re-ordered plan

**Done**

- Answered the two open questions the plan was waiting on, then re-ordered the
  roadmap around the answers (`docs/ROADMAP.md`, D13 and D14).
- `tools/lidar_coverage.py`: reads the organised 64 × 1024 cloud, measures each
  beam's altitude angle, fits the sensor's height from where the steepest beams
  land on bare floor, and prints what the LiDAR can see of the pallet at every
  approach distance.
- Recorded D8–D12 (Stage 0 defaults, accepted unchanged) in `docs/decisions.md`.

**What the LiDAR measurement settled**

| | measured |
|---|---|
| beams × columns, vertical FOV | 64 × 1024, 88.7° — it is an OS0, it looks nearly straight down |
| beam spacing | 1.408° |
| beam origin above the floor | **0.6121 m** (±1.8 mm across two bags) |
| same height via the URDF chain | 0.6549 m — **41 mm out** |
| lowest beam usable straight ahead | row 62, **−43.6°** |
| forward blind radius on the floor | **0.832 m** from `base_footprint` |
| face held down to | **0.24 m** bumper gap (2 beams), 0.20 m (1 beam) |

So yes to both uses: the LiDAR can be the insertion stop sensor, and at the
standoff it puts 3–4 beams on the face with 2–3 of them inside the pocket
opening, 26 azimuth columns across a 0.23 m pocket. That is an independent
measurement of the gap pattern, not just of range.

**What broke / what I learned**

- **The lowest beam of all sees nothing straight ahead.** Row 63 (−44.8°)
  returns normally to the sides and behind, but in the ±10° forward sector it
  gave 0 and 1 points across the two bags where its neighbours gave 55–72. The
  ZED sits 0.22 m below and ahead of the LiDAR, right in its path. Taking the
  datasheet's ±45° would have been one beam — about 20 mm of stop distance —
  optimistic. The robot occludes its own sensors and only the data says where.
- **I had to fit the sensor height rather than read it.** The URDF chain
  overstates it by 41 mm, matching the 38 mm the floor fits found on day 1. Using
  the URDF number pushes every distance in this analysis ~40 mm optimistic, all
  in the direction of driving closer to the pallet. Second time the URDF's
  accessory mounts have been wrong; they are nominal, not measured.
- **Fitting the height from all 22 ground intercepts was worse than using 8.**
  A beam at 5° lands 7 m out, where floor slope and a long lever arm swamp the
  height it is supposed to measure. Restricting the fit to the eight steepest
  beams cut the spread from 12.0 mm to 5.7 mm. More data is not more information
  when the extra data has a worse conditioning number.
- **The roadmap had an ordering inversion I had written myself.** Stage 3 listed
  its standoff as coming "from fork geometry, stage 5". The standoff is not a
  perception number — it is set by how far the virtual tines reach, because they
  have to start outside the pallet. Split out as stage 5a and taken first.
- **The measured pallet made the camera-FOV section in `SYSTEM.md` wrong, and
  looser, not tighter.** The whole 1.45 m standoff argument rested on keeping a
  22 mm bottom lip in frame. This pallet has no lip: the opening runs from the
  floor up. The opening's lower bound is the floor plane, which is fitted from
  the whole visible floor, so only the pocket's *top* edge needs to be in
  frame — 1.17 m, not 1.38 m. The camera no longer binds the standoff at all.
- While rewriting it I found the old table used 20.76° for the bottom of the
  image, which is the *symmetric* half-angle. The real bottom row is at
  `atan((719 − cy)/fy)` = 20.19°, because `cy` is 369.75, not 360. It made the
  camera look 7 mm more capable than it is — small, but in the optimistic
  direction again.

**The estimate, honestly**

Stages 5a–10 come to about 11½ desk days against 12 remaining, and that counts
no lab time at all. Four lab sessions are the critical path. Stage 9
(repositioning) is therefore out unless everything lands early; stage 10
(evaluation) is not negotiable, because the measured accuracy and the false-GO
count are what the project is judged on, not the state count.

**Stage 5a decided (D15-D18)**

Carriage at the front bumper with 1.15 m tines, so the tip reaches 1.6125 m.
Standoff **1.75 m**, insertion stops with the face at **0.700 m** - a 0.2375 m
gap to the real bumper, which is where the LiDAR still holds two beams on the
face. 1.05 m of travel, **0.9125 m of insertion = 76 % of the pallet**, tips
0.2825 m short of the far end.

Both of the old numbers died of the same cause: 1.45 m and the 0.052 m fork
underside were derived from an EPAL bottom lip this pallet does not have.
Replacing them, the standoff went *up* and the underside went *down*, which is
not what I expected from "the lip is gone" - the lip was holding the camera
limit up and the fork height up at the same time.

The thing I would not have thought of without the measurement: insertion is
only blind in the sense D7 meant for the **last 0.47 m**. The camera keeps the
pockets to 1.17 m and the LiDAR keeps the face all the way to the stop. "Blind
insertion" was a reasonable guess that measurement made more precise.

**The long side, measured at the lab (D19-D23)**

Height 14.5 with a 2.0 board top and bottom; between them 15 / 38.5 / 15 /
36.5 / 15. So the opening runs **0.020 to 0.125 m** with pockets 0.375 wide and
centres **0.525 apart**, against 0.000-0.100, 0.230 and 0.365 on the short side.

That is not two sizes of the same thing, it is two different structures, and the
measurement explains why. A EUR pallet's three bottom boards run along the
1200 mm direction. Enter from the 800 mm face and the forks travel *between*
them, down clear channels, so there is nothing under the pocket and the opening
starts at the floor. Enter from the 1200 mm face and the forks cross those
boards, so the opening has to start above them - the 2 cm lip. Both faces are
enterable, which is the definition of four-way entry. I had the short-side
absence of a lip filed as "this pallet is unusual"; it is not unusual, it is the
side I happened to measure first.

Three consequences:

- **Side classification got easy, and by the signal I had ranked last.** The
  roadmap asked "face width alone, or width + gap pattern?" assuming width was
  primary. Width is the *weakest* of the four discriminators: a clipped or
  occluded face measures short, and nothing says whether a 0.9 m face is a
  foreshortened long side or a complete short one. **Pocket spacing** separates
  the faces by 160 mm against a measured spacing error of ~13 mm - over 10 sigma
  - and it uses the two pocket centres perception already finds.
- **The insertion geometry had to become per-side.** D16's stop distance applied
  to the long side drives the tips **0.118 m out of the back**, because that face
  is only 0.795 m deep against a 1.6125 m tip reach. Long-side stop moves out to
  0.950 m: 83 % depth, and 5 LiDAR beams on the face instead of 2.
- **The 0.052 m fork underside that `params.yaml` has held all along was right -
  for the wrong face.** It came from the EPAL bottom lip. The lip is real on the
  long side (0.0525 m centred) and absent on the short side (0.030 m). I had
  called that number simply wrong on day 3; it was face-specific, which is worse
  than wrong because it looks right half the time.

**And the URDF is right for once**

| | measured | URDF | diff |
|---|---|---|---|
| length | 0.920 m | 0.925 m | 5 mm |
| width over the wheels | 0.690 m | 0.6996 m | 10 mm |
| front overhang | 0.460 m | 0.4625 m | 2.5 mm |

After the `base_footprint` joint (25 mm) and the LiDAR chain (41 mm), I expected
a third. `base_link`'s collision box is accurate, so every bumper gap in D16 and
D21 stands. Keeping the URDF's 0.4625 because the larger number is the
conservative one. Lens height re-confirmed at 0.398 m, so `floor.expected_z`
stays at -0.025.

**Open**

- No bag has the pallet and `/ouster/points` together, so D13's cross-check
  claim is geometry from measured beam angles, not yet seen in data.

## Day 0 — 2026-10-13 — survey and scaffold

**Done**

- Surveyed both machines with `pallet_approach/tools/survey_ros2.sh` (read-only). Findings in
  `docs/SYSTEM.md`; the raw output stays off the repository (lab network details).
- Repo scaffolded: `pallet_approach_interfaces` (action + 2 messages) and
  `pallet_approach` (logic, nodes, launch, config, tests, tools), plus
  `docker/Dockerfile.jetson`.
- `config/params.yaml` filled with measured values, not guesses.
- 83 unit tests passing for the modules that are written; 55 failing on purpose
  against `perception.py` and `go_nogo.py`, which were still skeletons.

**What the survey settled**

- Multi-machine DDS works in both directions, no configuration needed: both
  machines on `ROS_DOMAIN_ID=10`, `rmw_fastrtps_cpp`, identical topic lists.
- `/cmd_vel` is `geometry_msgs/Twist` with **zero** publishers and one
  subscriber (`scout`). No teleop, joystick, mux or Nav2 in the way.
- Depth is `32FC1`, 1280x720 at ~37 Hz, stamped
  `zed2_left_camera_frame_optical`; the rectified left image is under
  `rgb/color/rect/`. Intrinsics `fx=fy=949.673`, `cx=628.338`, `cy=369.746`,
  `D = 0`.
- `base_link -> camera` is a rigid URDF chain: translation
  `[0.390, 0.060, 0.163]`, zero pitch.
- The base driver has **no command timeout parameter**, which settles the
  heartbeat design: fixed publish rate, explicit zero on every stop.

**What broke / what I learned**

- The deployed URDF names the camera `zed2`, not `sensor_d456` as the older
  copy on the workstation does. Reading a stale checkout nearly cost a day of
  wrong frame names in `params.yaml`. Read the live system, not the checkout.
- The ZED's visual odometry publishes `/zed2/zed_node/odom` with `frame_id:
  odom` and `child_frame_id: zed2_camera_link`, which collides by name with the
  wheel odometry. It is topic-only -- the wrapper broadcasts no tf -- so the tree
  is clean. Enabling the wrapper's `publish_tf` would give `zed2_camera_link`
  two parents and break the camera chain.
- **Got the working frame wrong at first and had to redo it.** I had perception
  working in `base_link`, which is the chassis body frame 0.21 m above the
  ground, so every height was a negative offset from a floor at -0.210. The
  floor is at `base_footprint`, by definition, and that is where perception
  belongs: pocket centre 0.072, fork underside 0.052, fitted floor ~0. The two
  frames differ by a pure z translation, so x, y and yaw are identical and the
  driving targets compose with `/odom` unchanged.
- That mistake was hiding a real bug. Anchored at the floor, the standoff
  arithmetic shows the camera clips the **bottom** of the pocket opening at
  1.30 m: it would measure 94 mm of the 100 mm opening, or 69 mm if the real
  floor is at the wheel contact patch rather than `base_footprint`. Against a
  30 mm height tolerance that is a spurious NO-GO on a sound pallet. Standoff
  moved to 1.40 m, the smallest value that keeps the whole opening in frame
  under either floor height. In `base_link` the 2.5 cm ambiguity would have
  ridden along inside a negative offset and I would not have seen it.
- `base_footprint` (0.210 m below `base_link`) and the wheel contact patch
  (0.2348 m) differ by 2.5 cm. Not a contradiction -- one is the declared ground
  frame, the other an approximate model of it -- but the RANSAC floor fit
  measures which, and the residual is a result worth reporting.
- `cv2` and `cv_bridge` are broken inside the ZED container: they are built
  against NumPy 1.x and the container has 2.2.6. Our own image pins `numpy<2`.
  The nodes decode images with four lines of buffer arithmetic instead, so they
  do not depend on either.
- Same class of problem on the workstation: ROS Iron on the path plus a conda
  Python 3.13. Offline tooling goes in a venv; tests need
  `env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

**Numbers**

| | |
|---|---|
| camera height above the floor | 0.373 m declared; 0.398 m by wheel geometry |
| camera FOV | 68.0° horizontal, 41.5° vertical |
| pocket centre leaves the frame closer than | 1.18 m from `base_footprint` |
| whole pocket opening needs | 1.32 m (1.38 m on the lower floor) |
| standoff | **1.40 m**, derived from the above |
| depth rate | ~37 Hz |
| odometry rate | 50.0 Hz, std dev 0.2 ms |
| raw depth bandwidth | 136 MB/s; +102 MB/s for colour |
| Jetson free disk | 5.7 G of 57 G (90 % full) |

**Open**

- No USB3 SSD for the Jetson yet. 5.7 G free will not hold the Day 7 static
  dataset; `pallet_approach/tools/record_bag.sh` throttles to 10 Hz and refuses to fill the disk.
- The real pallet still needs measuring against the EPAL 1 nominals now in
  `params.yaml`.


---

## Day 6 — 2026-10-19 (afternoon): the rest of the pipeline

Built the five stages that were left, in one pass, and the offline harness that
was supposed to come last turned out to be the most useful thing in it.

**Arbitration and the watchdog (stage 0, D30–D33).** `twist_mux` config with
the safety stop above our controller above Nav2, plus a lock the coordinator
holds for the length of a run so "a run is in progress" is stated rather than
inferred. Our commands moved off `/cmd_vel` onto `/cmd_vel_pallet`, which
created a new silent failure — with no mux running every command vanishes and
the stack reports that it is driving — so the controller now refuses to
configure when nothing is listening. Fifth instance of the same shape.

The watchdog owns the stop independently of the coordinator. It takes the
camera's liveness from `camera_info` rather than the depth image, which keeps it
off a 3.7 MB topic for a timestamp that arrives free next door.

**The tracker (stage 2, D34–D37).** Association in `odom` with the robot's
motion taken out — two tests, one each way, because the same drive that passes
with the pose taken out fails without it. The cross-check's first window was
±0.15 m of depth, chosen to keep a loaded pallet's cargo out of the comparison,
and a test found it excluded the disagreement the check exists to catch: a
camera hallucinating a face 0.4 m from the real one read as NOT EVALUATED.

**The closing servo (stage 3, D38–D42).** Three versions.

1. Pure pursuit chasing a carrot on the centre line. The residual scaled with
   the lookahead and *ignored the gain entirely* — the signature of a saturated
   loop. Cause: the angular deadband raises any small command to the base's
   minimum turn rate, so every command came out hard over or hard back. Small
   angular commands are dropped now, not inflated.
2. Pure pursuit still locks two gains together. A lookahead short enough to
   converge in 1.25 m implies a 60° approach angle, so the robot dog-legged.
   Split into an outer loop (which way to aim) and an inner loop (turn to it).
3. The run-up formula modelled only the exponential regime and claimed 3.04 m
   was enough to take out 0.50 m of offset. The run finished 250 mm off the
   line with 11° of yaw. While the aim is capped the error falls *linearly*.
   Honest answer: just over 4 m.

With a square-up at the standoff, the worst simulated arrival over offsets to
0.8 m and 10 mm of noise is 17 mm and 2.4°, against 65 mm of pocket slack.

**The forks and the insertion (stages 5b, 6, D43–D47).** Writing the insertion
tests found the gap D18 left: a *disagreement* aborts, but an absent
measurement cannot disagree with anything. Losing the LiDAR for half a second
aborts now, and so does finishing the travel with nothing having confirmed
where the face was.

**The coordinator and the harness (stage 7, D48–D54).** The harness wires the
real state machine, tracker, executor, fork simulator and safety rules into
`simulate.py`'s world. On its first few runs it found four real bugs:

- nothing supplied the two GO/NO-GO rules that take points, so **every** run
  returned NO_GO. Perception publishes two purpose-built clouds now.
- the stability rule was being judged on frames taken while the robot was still
  moving — five frames at the closing speed span 100 mm of travel, which reads
  as a 118 mm unstable estimate. Worse, the node had been dodging it by
  repeating the latest estimate five times, which made the rule **unable to
  fail**.
- the divergence detector aborted 3 runs in 12 from a 9 m run-up, converging
  perfectly. Monotone growth is a fragile statistic under noise. Eight frames
  and a floor at the residual target.
- the run-up requirement had no hysteresis, so a reverse that stopped a
  centimetre short asked for another centimetre and died on `max_backoffs`.

Two findings left standing rather than tuned away, both in `docs/decisions.md`
under "Open, with numbers attached": the stability tolerance refuses runs above
about 5 mm of per-frame scatter and nobody has measured the real scatter, and
the LiDAR cross-check's `min_points` is still arithmetic.

**Where this leaves things.** 800 tests pass. A whole virtual pickup completes
in 68 s. And none of the last five stages has run on the robot — the harness
does not touch discovery, QoS, lifecycle, tf or the clock, which is where every
real failure has come from. Lab session B is next, and it should be expected to
find things.

## Day 9 — 2026-10-22 — stale sensors on the robot were the network

After D72 and D73 the robot still reported "unhealthy while idle": odometry
up to 0.7 s old, now and then camera_info. The sensors were fine; the link was
full. The Jetson's port received 520 Mbit/s at idle, because each process
subscribing to `/ouster/points` gets its own copy of a 3 MB cloud. A
`lidar_relay` node now brings the cloud across once (257 Mbit/s) and the
other nodes read its local copy. That fixed odometry outright. The LiDAR still
dropped to 6.5 Hz, from a 212 KB kernel UDP buffer that overflowed on the
fragments of each cloud; at 32 MB it arrives at 10.0 Hz with a worst gap of
0.12 s, and 90 s at idle produced no warning. Numbers in `docs/SYSTEM.md`,
"Network". The buffer setting is not yet persistent on the Jetson.

2026-10-23: a reboot reset it and the run aborted on "lidar: 5.0 Hz". Now in
`/etc/sysctl.d/90-ros-dds.conf` on the Jetson; after a restart of the stack
the LiDAR is back at 10.0 Hz and 90 s at idle gave no warning.


Then the next abort: "estimate: 1.8 Hz, limit 2.5 Hz". Perception was fine;
its tf buffer was not. Activated on its own (no run, controller inactive),
perception's buffer fell 0.15-0.4 s behind the depth stamps within two
seconds, while a separate process saw the same transforms 44 ms old, so
almost every stamped lookup failed. The stock `TransformListener(spin_thread=
True)` spins the whole node on its thread (so tf queues behind depth frames)
and decodes all ~200 /tf messages a second in Python. `CameraTransformListener`
(`nodes/_common.py`) spins a helper node with only the tf subscriptions and
decodes only messages naming `frames.moving_links`. Lookups stay stamped.
Result over 30 s: no tf failures, an estimate from every processed frame.
It also fixes a thread leak: each configure/cleanup cycle used to leave one
more listener thread spinning perception.

What that uncovered: a frame really costs 250-350 ms on the robot with the
whole stack running, not ~155 ms (that figure was the 150 ms tf timeout on
frames that failed). Perception runs at ~3.3 Hz (2.9-3.7), inside the
watchdog's 2.5 Hz and 0.5 s limits but not by much. The Jetson sits at a load
average of ~8.5 on 8 cores at idle: ZED SDK ~130 %, fork overlay ~110 % (the
24 fps dashboard), watchdog ~50 %, tracker ~40 %.

### Perception speed, 2026-10-23 (no behaviour change except D88)

Laptop, one thread, `tools/profile_perception.py pallet_s_129`: **80 -> 34 ms**
a frame. Each step was checked on 31 real frames from five bags
(pocket positions, detections) and with a 10-seed sweep.

- `depth_to_points`: index grid built only for the strided pixels.
- `voxel_downsample`: occupied cubes numbered in a dense array instead of
  sorting the keys (identical output).
- `cluster_points`: DBSCAN on SciPy's C KD-tree (`query_pairs` + connected
  components) instead of scikit-learn's, 2x faster; same clusters, a border
  point touching two clusters may go to the other.
- RANSAC: all samples drawn in one call (`rng.choice(..., replace=False)`
  shuffled every point every iteration -- 74 ms of 186 on the Jetson), and the
  floor's hypotheses scored on 3000 points (`floor.score_points`).
- Only points below the D76 ceiling + 0.10 m are clustered; anything taller
  still shows a top above the ceiling and is rejected as before.
- `FrameGate` keeps its average rate (it rounded 3.3 Hz down to 2.5 Hz from a
  10 Hz input).

Changing the random draws exposed D88. Not yet measured on the Jetson (it was
unreachable); GPU next (docs/TODO.md).

