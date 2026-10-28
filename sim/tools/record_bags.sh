#!/usr/bin/env bash
# Record a set of sim bags, each from a fresh sim, with the ground truth beside it.
#   ./sim/tools/record_bags.sh SPEC [SECONDS]
#
# Run where the sim runs (sim/remote.sh bags SPEC does that and copies the bags
# back). SPEC is a text file, one bag per line, columns:
#   name face_x face_y face_yaw block_protrusion load_height load_overhang seed
# Lines starting with # and blank lines are skipped. A bag whose name already
# exists in sim/bags is skipped, so an interrupted set can be resumed.
#
# Beside each bag goes truth.yaml: the launch arguments, which ARE the truth in
# a sim (the pallet is placed exactly there), and the commit.
set -u
spec=${1:?usage: record_bags.sh SPEC [SECONDS]}; seconds=${2:-8}
cd "$(dirname "$0")/../.."
case "$spec" in /*) ;; *) [ -f "$spec" ] || spec="sim/tools/$spec" ;; esac
[ -f "$spec" ] || { echo "no such spec: $spec" >&2; exit 2; }
mkdir -p sim/bags
while read -r name fx fy fyaw prot lh lo seed; do
  case "$name" in ''|'#'*) continue ;; esac
  if [ -d "sim/bags/$name" ]; then echo "skip $name (exists)"; continue; fi
  ./sim/run_sim.sh stop >/dev/null 2>&1
  ./sim/run_sim.sh up robot:=scout face_x:="$fx" face_y:="$fy" face_yaw:="$fyaw" \
      block_protrusion:="$prot" load_height:="$lh" load_overhang:="$lo" seed:="$seed" \
      >/dev/null 2>&1
  sleep 40            # the robot spawns at ~20 s; the stack needs the rest
  ./sim/run_sim.sh record "$name" "$seconds"
  {
    echo "name: $name"
    echo "robot: scout"
    echo "face: short"
    echo "face_x: $fx"
    echo "face_y: $fy"
    echo "face_yaw: $fyaw"
    echo "block_protrusion: $prot"
    echo "load_height: $lh"
    echo "load_overhang: $lo"
    echo "seed: $seed"
    echo "commit: ${REPO_COMMIT:-unknown}"
  } > "sim/bags/$name/truth.yaml"
done < "$spec"
./sim/run_sim.sh stop >/dev/null 2>&1
echo "done: $(ls sim/bags | wc -l) bags in sim/bags"
