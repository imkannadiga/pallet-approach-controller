#!/usr/bin/env bash
# Run the sim on the GPU machine and watch it from this laptop.
#
#   ./sim/remote.sh up [launch args]    # sync, build if needed, start
#   ./sim/remote.sh run pickup         # start a run and follow it here
#   ./sim/remote.sh watch               # tunnel; then open http://localhost:8889/dashboard
#   ./sim/remote.sh stop
#   ./sim/remote.sh bags SPEC [seconds]  # record a set of bags there, copy them to ~/bags/sim
#   ./sim/remote.sh <anything else>     # passed to run_sim.sh over there
#
# `watch` blocks while the tunnel is open; Ctrl-C closes it. Run it in a second
# terminal. The tunnel forwards 8889 (the page and signalling) and 8189 (the
# video, WebRTC over TCP), both to the remote's localhost.
#
# The machine comes from REMOTE_HOST (user@host) and, if it sits behind a
# bastion, REMOTE_JUMP -- set in the environment or in sim/remote.env, which
# is git-ignored:
#
#   REMOTE_HOST=user@gpu-box.example.org
#   REMOTE_JUMP=user@bastion.example.org
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
[ -f "$REPO_ROOT/sim/remote.env" ] && . "$REPO_ROOT/sim/remote.env"
: "${REMOTE_HOST:?set REMOTE_HOST=user@host (environment or sim/remote.env)}"
REMOTE_JUMP=${REMOTE_JUMP:-}
REMOTE_DIR=${REMOTE_DIR:-pallet-approach-controller}
JUMP=()
[ -n "$REMOTE_JUMP" ] && JUMP=(-J "$REMOTE_JUMP")
SSH=(ssh -A "${JUMP[@]}" "$REMOTE_HOST")

sync() {
  rsync -az --delete --exclude .git --exclude __pycache__ --exclude venv --exclude sim/bags \
    -e "ssh ${JUMP[*]}" "$REPO_ROOT/" "$REMOTE_HOST:$REMOTE_DIR/"
}

case "${1:-}" in
  up)
    shift
    sync
    # Rebuild every time: docker's cache makes an unchanged build take seconds,
    # and a stale image is the one mistake this script exists to prevent.
    "${SSH[@]}" "cd $REMOTE_DIR && ./sim/run_sim.sh stop >/dev/null && ./sim/run_sim.sh build >/tmp/sim_build.log 2>&1 \
      || { tail -20 /tmp/sim_build.log; exit 1; }; ./sim/run_sim.sh up $*"
    echo "then: ./sim/remote.sh watch   (in another terminal)"
    ;;
  bags)
    # SPEC is a file under sim/tools (see bag_set_baseline.txt). Each bag is a
    # fresh sim, so a set of 20 takes the better part of half an hour.
    shift
    sync
    commit=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)
    git -C "$REPO_ROOT" diff --quiet 2>/dev/null || commit="$commit+uncommitted"
    "${SSH[@]}" "cd $REMOTE_DIR && ./sim/run_sim.sh stop >/dev/null 2>&1; ./sim/run_sim.sh build >/tmp/sim_build.log 2>&1 \
      || { tail -20 /tmp/sim_build.log; exit 1; }; REPO_COMMIT='$commit' ./sim/tools/record_bags.sh $*"
    mkdir -p "$HOME/bags/sim"
    rsync -az -e "ssh ${JUMP[*]}" "$REMOTE_HOST:$REMOTE_DIR/sim/bags/" "$HOME/bags/sim/"
    echo "bags are in $HOME/bags/sim"
    ;;
  watch)
    for port in 8889 8189; do
      if ss -ltn 2>/dev/null | grep -q ":$port "; then
        echo "port $port is already in use here (a local sim or another tunnel?)" >&2; exit 1
      fi
    done
    echo "open http://localhost:8889/dashboard   (raw camera: /zed)   Ctrl-C to close"
    ssh -N "${JUMP[@]}" -L 8889:localhost:8889 -L 8189:localhost:8189 "$REMOTE_HOST"
    ;;
  "")
    sed -n '2,15p' "$0"; exit 2 ;;
  *)
    "${SSH[@]}" "cd $REMOTE_DIR && ./sim/run_sim.sh $*" ;;
esac
