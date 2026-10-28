#!/usr/bin/env bash
# N independent pickups from the shared starting positions (D96); one RESULT
# line per run, then the acceptance count.
#   ./sim/tools/batch_runs.sh 7 10 [launch args]
#
# SEED picks the starting positions. The same seed gives the same poses on the
# real robot: `pallet_approach/tools/start_positions.py --seed SEED` prints the
# tape readings to set the pallet to for each run.
#
# Each run starts where Nav2 might leave the robot: 2.5 m from the robot origin
# along the pallet's centre line (D75), up to +-0.15 m off that line and turned
# up to +-10 deg from it (D83). Each run also gets its own depth-model seed, and
# everything comes from SEED, so a batch repeats exactly.
#
# Acceptance (D96): a run is accepted when it ends INSERTED. The batch passes
# when at least 9 of 10 runs are accepted; it exits 0 if it does, 1 if not.
set -u
seed=${1:?usage: batch_runs.sh SEED [COUNT] [launch args]}; n=${2:-10}
shift 2 || true
[ "${1:-}" = "--" ] && shift
need=$(( (n * 9 + 9) / 10 ))    # 9 of 10; the same fraction for any count
cd "$(dirname "$0")/../.."

mapfile -t poses < <(python3 pallet_approach/tools/start_positions.py \
                       --seed "$seed" --count "$n" | tail -n +2)
[ "${#poses[@]}" -eq "$n" ] || { echo "could not generate $n poses" >&2; exit 2; }

accepted=0
for i in $(seq 1 "$n"); do
  read -r _ fx fy fyaw dL dR lat offset heading <<< "${poses[$((i - 1))]}"
  ./sim/run_sim.sh stop >/dev/null 2>&1
  [ "$i" -gt 1 ] && export SIM_OPEN_BROWSER=false
  ./sim/run_sim.sh up "face_x:=$fx" "face_y:=$fy" "face_yaw:=$fyaw" \
                      "seed:=$((seed * 1000 + i))" "$@" >/dev/null 2>&1
  sleep 35
  line=$(./sim/run_sim.sh run pickup 900 2>&1 | grep '^RESULT' | cut -c1-230)
  echo "run $i (offset ${offset} m, heading ${heading} deg, dL ${dL} dR ${dR}): ${line:-RESULT none}"
  case "$line" in 'RESULT INSERTED'*) accepted=$((accepted + 1)) ;; esac
done
./sim/run_sim.sh stop >/dev/null 2>&1

echo "accepted $accepted of $n (need $need): $([ "$accepted" -ge "$need" ] && echo PASS || echo FAIL)"
[ "$accepted" -ge "$need" ]
