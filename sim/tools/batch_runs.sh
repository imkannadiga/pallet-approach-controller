#!/usr/bin/env bash
# N independent runs, each on a freshly started sim; one RESULT line per run.
#   ./sim/tools/batch_runs.sh approach_only 10 [launch args]
#   ./sim/tools/batch_runs.sh approach_only 10 --random 7 [launch args]
#
# --random SEED (D83): each run starts where Nav2 might leave the robot: 2.5 m
# from the face along the pallet's centre line (D75), up to +-0.15 m off that
# line and turned up to +-10 deg from it. Drawn as (offset, heading error) and
# converted to where the pallet appears from the robot, which sits at the
# origin; rotating the PALLET by 10 deg instead would swing its centre line
# ~0.4 m off the robot at 2.5 m, a different and much larger error. Each run
# also gets its own depth-model seed; everything comes from SEED, so a batch
# repeats exactly. Pass criterion: GO in at least 9 of 10.
set -u
mode=${1:-approach_only}; n=${2:-10}; shift 2 || true
random=''
if [ "${1:-}" = "--random" ]; then random=${2:?--random needs a seed}; shift 2; fi
cd "$(dirname "$0")/../.."
for i in $(seq 1 "$n"); do
  pose=(); note=''
  if [ -n "$random" ]; then
    # e: robot's offset LEFT of the centre line; psi: its heading, CCW from
    # the line. The face centre in the robot frame is 2.5 u - e v, with u the
    # line direction and v its left normal, both seen from the robot; the face
    # is turned by -psi.
    read -r fx fy fyaw e psi < <(python3 -c "
import math, random; r = random.Random($random * 1000 + $i)
e, psi, d = r.uniform(-0.15, 0.15), math.radians(r.uniform(-10.0, 10.0)), 2.5
print(f'{d*math.cos(psi) - e*math.sin(psi):.3f} {-d*math.sin(psi) - e*math.cos(psi):.3f} '
      f'{-psi:.4f} {e:+.3f} {math.degrees(psi):+.1f}')")
    pose=("face_x:=$fx" "face_y:=$fy" "face_yaw:=$fyaw" "seed:=$((random * 1000 + i))")
    note="offset ${e} m, heading ${psi} deg" 
  fi
  ./sim/run_sim.sh stop >/dev/null 2>&1
  [ "$i" -gt 1 ] && export SIM_OPEN_BROWSER=false
  ./sim/run_sim.sh up "${pose[@]}" "$@" >/dev/null 2>&1
  sleep 35
  line=$(./sim/run_sim.sh run "$mode" 900 2>&1 | grep '^RESULT' | cut -c1-230)
  echo "run $i ${note:+($note)}: ${line:-RESULT none}"
done
./sim/run_sim.sh stop >/dev/null 2>&1
