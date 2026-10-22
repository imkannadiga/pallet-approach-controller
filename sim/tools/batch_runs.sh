#!/usr/bin/env bash
# N independent runs, each on a freshly started sim; one RESULT line per run.
#   ./sim/tools/batch_runs.sh approach_only 10 [launch args]
set -u
mode=${1:-approach_only}; n=${2:-10}; shift 2 || true
cd "$(dirname "$0")/../.."
for i in $(seq 1 "$n"); do
  ./sim/run_sim.sh stop >/dev/null 2>&1
  [ "$i" -gt 1 ] && export SIM_OPEN_BROWSER=false
  ./sim/run_sim.sh up "$@" >/dev/null 2>&1
  sleep 35
  line=$(./sim/run_sim.sh run "$mode" 900 2>&1 | grep '^RESULT' | cut -c1-230)
  echo "run $i: ${line:-RESULT none}"
done
./sim/run_sim.sh stop >/dev/null 2>&1
