#!/usr/bin/env bash
# Record a bag on the Jetson, then move it to this laptop. Run this on the laptop.
#
# Nothing on the robot has to be reconfigured: the bag is written inside the
# already-running container and then streamed out through `docker exec | tar`, so
# there is no bind mount to add, no container to restart, and no second copy
# sitting on the Jetson's disk while it transfers.
#
# The Jetson has ~5.7 GB free, so the bag cannot stay there. The order is:
# record -> stream here -> verify file by file -> only then delete the remote
# copy. If anything fails the remote copy is untouched and `--fetch-only` retries
# the transfer without re-recording.
#
# Network: ssh straight to the Jetson at stuttgart_orin@192.168.1.1 from the
# hotspot. No jump host -- the robot PC forwards between its AP subnet
# (10.42.0.x, which your laptop joins) and the wired switch (192.168.1.x).
# Use that wired address rather than the Jetson's own wifi address
# (10.42.0.40): over the wired leg only the last hop is wifi, so it is about
# twice as fast.
#
# Authentication: the script runs several commands over ssh, so it opens ONE
# shared connection up front (ssh ControlMaster) and reuses it. You are asked
# for a password or passphrase at most once. An ssh key is nicer still:
#     ssh-copy-id stuttgart_orin@192.168.1.1
#
# Usage:
#   ./fetch_bag.sh --check                              # test everything, record nothing
#   ./fetch_bag.sh -o pallet_2m_0deg -d 20              # depth only, 5 Hz, zstd
#   ./fetch_bag.sh -o pallet_2m_0deg -d 20 --with-image # add the colour image
#   ./fetch_bag.sh -o pallet_2m_0deg -d 5 --full-rate   # no throttle (no topic_tools);
#                                                       # ~56 Hz, so keep it short
#   ./fetch_bag.sh -o pallet_175_lidar -d 10 \
#       --with-image --with-lidar                       # camera AND LiDAR in one bag
#   JETSON=user@192.168.1.102 CONTAINER=<scout> \
#     ./fetch_bag.sh -o floor_lidar -d 20 --lidar-only  # LiDAR, on the robot PC
#   ./fetch_bag.sh -o pallet_2m_0deg --fetch-only       # retry a failed transfer
#   ./fetch_bag.sh --list-remote                        # what is still on the Jetson
#   ./fetch_bag.sh --check --debug                      # print every remote command
#
# Settings, as environment variables:
#   JETSON=stuttgart_orin@192.168.1.1   ssh target
#   CONTAINER=<name>                    container that can see the ZED topics
#   REMOTE_BAG_DIR=/tmp/pallet_bags     where to write inside that container
#   LOCAL_BAG_DIR=~/bags                where bags land here

set -uo pipefail

JETSON="${JETSON:-stuttgart_orin@192.168.1.1}"
CONTAINER="${CONTAINER:-}"
REMOTE_BAG_DIR="${REMOTE_BAG_DIR:-/tmp/pallet_bags}"
LOCAL_BAG_DIR="${LOCAL_BAG_DIR:-$HOME/bags}"
VIA=""
NAME=""
DURATION=20
RATE=5
WITH_IMAGE=0
WITH_LIDAR=0
LIDAR_ONLY=0
COMPRESS=1
CHECK_ONLY=0
FETCH_ONLY=0
LIST_REMOTE=0
KEEP_REMOTE=0
FULL_RATE=0
DEBUG=0

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while [ $# -gt 0 ]; do
  case "$1" in
    -o|--out)      NAME="$2"; shift 2 ;;
    -d|--duration) DURATION="$2"; shift 2 ;;
    -r|--rate)     RATE="$2"; shift 2 ;;
    --full-rate)   FULL_RATE=1; shift ;;
    --with-image)  WITH_IMAGE=1; shift ;;
    --with-lidar)  WITH_LIDAR=1; shift ;;
    --lidar-only)  LIDAR_ONLY=1; shift ;;
    --no-compress) COMPRESS=0; shift ;;
    --via)         VIA="$2"; shift 2 ;;
    --container)   CONTAINER="$2"; shift 2 ;;
    --keep-remote) KEEP_REMOTE=1; shift ;;
    --fetch-only)  FETCH_ONLY=1; shift ;;
    --list-remote) LIST_REMOTE=1; shift ;;
    --check)       CHECK_ONLY=1; shift ;;
    --debug)       DEBUG=1; shift ;;
    -h|--help)     sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

say()  { printf '\n== %s\n' "$*"; }
fail() { printf '\n%s\n' "$*" >&2; exit 1; }
dbg()  { [ "$DEBUG" = "1" ] && printf '   $ %s\n' "$*" >&2; return 0; }

# --------------------------------------------------- one shared ssh connection
# NO BatchMode here: it would disable password and passphrase prompts, so the
# script would fail on a host you can ssh to perfectly well by hand.
CTL="${TMPDIR:-/tmp}/pa-ssh-$(id -u)-$$"
SSH_OPTS=(-o ControlPath="$CTL")

cleanup() {
  ssh -O exit -o ControlPath="$CTL" "$JETSON" >/dev/null 2>&1
  rm -f "$CTL"
}
trap cleanup EXIT

on_jetson() { dbg "ssh $JETSON $*"; ssh "${SSH_OPTS[@]}" "$JETSON" "$@"; }

say "connecting to $JETSON"
echo "  opening one shared connection, so any password is asked for once only"
MASTER=(-f -N -M -o ControlPath="$CTL" -o ControlPersist=600 -o ConnectTimeout=15)
[ -n "$VIA" ] && MASTER+=(-J "$VIA")
dbg "ssh ${MASTER[*]} $JETSON"
if ! ssh "${MASTER[@]}" "$JETSON"; then
  fail "$(cat <<MSG
Could not open an ssh connection to $JETSON${VIA:+ via $VIA}.
The error above is ssh's own -- it says what went wrong. Common causes:

  - wrong user or address. Plain ssh working by hand means this should too, so
    check you are using exactly the same target:
        ssh $JETSON
  - not on the robot's hotspot. Your address should be 10.42.0.x:
        ip -brief addr | grep 10.42
  - no route to the switch subnet:
        ip route get 192.168.1.1
  - the Jetson's own wifi address is a slower fallback:
        JETSON=stuttgart_orin@10.42.0.40 $0 ...
  - or tunnel through the robot PC:  --via <user>@192.168.1.102
MSG
)"
fi
echo "  connected"

# ------------------------------------------------------------ docker access
DOCKER="docker"
if on_jetson "docker ps -q" >/dev/null 2>&1; then
  echo "  docker: ok"
elif on_jetson "sudo -n docker ps -q" >/dev/null 2>&1; then
  DOCKER="sudo docker"
  echo "  docker: needs sudo, and passwordless sudo works -- using 'sudo docker'"
else
  fail "$(cat <<MSG
Cannot run docker on the Jetson as this user.

  $ ssh $JETSON docker ps
$(on_jetson "docker ps" 2>&1 | sed 's/^/  /' | head -5)

Either add the user to the docker group (then log out and back in):
    sudo usermod -aG docker \$USER
or allow passwordless sudo for docker, or set DOCKER_SUDO=1 and run with a
sudo password available.
MSG
)"
fi

# ROS is not on PATH in a non-interactive shell in these containers, so every
# remote command is prefixed with the environment setup. Doing it here means no
# call site has to remember, and there is one place to change if the workspace
# moves. Override with REMOTE_ROS_SETUP=... if it lives somewhere else.
#
# Deliberately free of `$` and quotes: the string is parsed by the remote login
# shell before `bash -lc` ever sees it, so a `$var` here would be expanded (to
# nothing) on the way through.
DEFAULT_ROS_ENV='. /opt/ros/humble/setup.bash >/dev/null 2>&1; . /ros2_ws/install/setup.bash >/dev/null 2>&1; true'
ROS_ENV="${REMOTE_ROS_SETUP:-$DEFAULT_ROS_ENV}"

in_box() { dbg "$DOCKER exec -i $CONTAINER bash -lc \"<ros env>; $1\""
           on_jetson "$DOCKER exec -i $CONTAINER bash -lc \"$ROS_ENV; $1\""; }

# ------------------------------------------------------------------ container
if [ -z "$CONTAINER" ]; then
  say "containers running on the Jetson"
  on_jetson "$DOCKER ps --format '  {{.Names}}\t{{.Image}}'"
  fail "Pick one with --container <name> (or set CONTAINER=). It needs to see the
ZED topics, so the container already running the wrapper is the obvious choice."
fi
if ! on_jetson "$DOCKER ps --format '{{.Names}}'" | grep -qx "$CONTAINER"; then
  say "containers running on the Jetson"
  on_jetson "$DOCKER ps --format '  {{.Names}}\t{{.Image}}'"
  fail "container '$CONTAINER' is not among them"
fi
echo "  container '$CONTAINER' is up"

# ------------------------------------------------------- tools in the container
say "tools inside '$CONTAINER'"
in_box "command -v tar >/dev/null" || fail "no tar in the container; cannot stream the bag out"
echo "  tar: ok"
in_box "ros2 bag --help >/dev/null 2>&1" \
  || fail "$(cat <<MSG
'ros2 bag' is not available in '$CONTAINER'. rosbag2 is not installed there.
  apt-get update && apt-get install -y ros-humble-rosbag2 ros-humble-rosbag2-storage-default-plugins
or pick a different container with --container.
MSG
)"
echo "  ros2 bag: ok"
if in_box "ros2 pkg executables topic_tools 2>/dev/null | grep -q throttle"; then
  echo "  topic_tools throttle: ok"
else
  echo "  topic_tools throttle: MISSING -- needed to record below full rate."
  echo "    apt-get install -y ros-humble-topic-tools    (or use -d 5 --full-rate)"
fi
if [ "$COMPRESS" = "1" ]; then
  if in_box "ros2 bag record --help 2>/dev/null | grep -q compression-mode"; then
    echo "  zstd compression: ok"
  else
    echo "  zstd compression: unavailable, turning it off"
    COMPRESS=0
  fi
fi
PROBES=()
[ "$LIDAR_ONLY" = "0" ] && PROBES+=(/zed2/zed_node/depth/depth_registered)
{ [ "$LIDAR_ONLY" = "1" ] || [ "$WITH_LIDAR" = "1" ]; } && PROBES+=(/pallet_approach/lidar)
for PROBE in "${PROBES[@]}"; do
  if in_box "ros2 topic list 2>/dev/null | grep -q $PROBE"; then
    echo "  $PROBE visible from this container: ok"
  else
    echo "  $PROBE NOT visible from '$CONTAINER' -- check ROS_DOMAIN_ID and the RMW"
    echo "    there, or pick the container running that driver"
  fi
done

# The Ouster driver runs on the robot PC, so --with-lidar means its cloud crosses
# the network into the Jetson's container. That is the mirror image of the trap in
# docs/SYSTEM.md's network rule, where a remote subscriber throttled the ZED to 0.2 Hz: these
# topics are RELIABLE, so a subscriber that cannot keep up slows the publisher.
# The difference is scale and medium -- 31 MB/s over the wired switch rather than
# 136 MB/s over wifi -- so it is expected to work, but it has not been measured.
# Check the rate afterwards and fall back to two bags if the Ouster has dropped.
if [ "$WITH_LIDAR" = "1" ] && [ "$LIDAR_ONLY" = "0" ]; then
  cat <<'MSG'

  NOTE: --with-lidar subscribes to /pallet_approach/lidar across the network, from the
  robot PC into the Jetson. Unmeasured: watch for the Ouster's rate dropping.
  After this bag, confirm it is still ~10 Hz:
      ros2 topic hz /pallet_approach/lidar
  If it has dropped, record two bags instead (camera on the Jetson, LiDAR with
  --lidar-only on the robot PC) and align them offline by timestamp -- the
  clocks are chrony-synced.
MSG
fi

# ---------------------------------------------------------------------- space
say "space"
echo "  container: $(in_box "df -Pm '$REMOTE_BAG_DIR' 2>/dev/null || df -Pm /" | awk 'NR==2{print $4" MB free"}')"
echo "  here:      $(df -Pm "$(dirname "$LOCAL_BAG_DIR")" | awk 'NR==2{print $4" MB free"}')"
RAW_MB=$(( DURATION * RATE * 4 ))
[ "$WITH_IMAGE" = "1" ] && RAW_MB=$(( DURATION * RATE * 7 ))
# the LiDAR is not throttled -- 3.15 MB per scan at ~10 Hz is 31 MB/s on its own,
# which dominates a 5 Hz camera bag.
[ "$WITH_LIDAR" = "1" ] && RAW_MB=$(( RAW_MB + DURATION * 32 ))
echo "  this bag:  ~${RAW_MB} MB before compression, on the Jetson only until it transfers"

if [ "$LIST_REMOTE" = "1" ]; then
  say "bags still on the Jetson, in $REMOTE_BAG_DIR"
  in_box "du -sh $REMOTE_BAG_DIR/* 2>/dev/null" || echo "  none"
  exit 0
fi
if [ "$CHECK_ONLY" = "1" ]; then
  say "all checks passed, nothing recorded"
  exit 0
fi
[ -n "$NAME" ] || fail "need -o <name>"

REMOTE_BAG="$REMOTE_BAG_DIR/$NAME"
mkdir -p "$LOCAL_BAG_DIR"

# -------------------------------------------------------------------- record
if [ "$FETCH_ONLY" = "0" ]; then
  if in_box "test -e '$REMOTE_BAG'"; then
    fail "$REMOTE_BAG already exists on the Jetson. Pull it with --fetch-only, or pick another name."
  fi
  [ -e "$LOCAL_BAG_DIR/$NAME" ] && fail "$LOCAL_BAG_DIR/$NAME already exists here"

  RECORD_ARGS=(-o "$NAME" -d "$DURATION" -r "$RATE" --dir "$REMOTE_BAG_DIR")
  if [ "$LIDAR_ONLY" = "1" ]; then
    RECORD_ARGS+=(--lidar-only)
  else
    [ "$WITH_IMAGE" = "0" ] && RECORD_ARGS+=(--no-image)
    [ "$WITH_LIDAR" = "1" ] && RECORD_ARGS+=(--with-lidar)
    # No throttle: for a container without topic_tools. record_bag.sh sizes the
    # bag at ~56 Hz and refuses if it does not fit on the Jetson.
    [ "$FULL_RATE" = "1" ] && RECORD_ARGS+=(--full-rate)
  fi
  [ "$COMPRESS" = "1" ] && RECORD_ARGS+=(--compress)

  say "recording ${DURATION}s on the Jetson"
  echo "  nothing moves -- this only records. Keep the pallet still."
  # record_bag.sh is piped into `bash -s`, so it cannot go through in_box --
  # it sources the ROS environment itself, which also lets it be run by hand
  # inside the container.
  dbg "$DOCKER exec -i $CONTAINER bash -s -- ${RECORD_ARGS[*]}  < record_bag.sh"
  ssh "${SSH_OPTS[@]}" "$JETSON" \
      "$DOCKER exec -i $CONTAINER bash -s -- ${RECORD_ARGS[*]}" < "$HERE/record_bag.sh" \
    || fail "recording failed; nothing to transfer"
else
  in_box "test -d '$REMOTE_BAG'" || fail "$REMOTE_BAG is not on the Jetson"
  rm -rf "$LOCAL_BAG_DIR/$NAME"
fi

# ----------------------------------------------- stream it out of the container
say "streaming $REMOTE_BAG to $LOCAL_BAG_DIR/$NAME"
echo "  the bag is already zstd-compressed, so tar does not compress again"
# No ROS environment and no `bash -lc` here, on purpose: tar needs stdout to
# itself, and a setup script printing even one line would corrupt the archive.
dbg "$DOCKER exec $CONTAINER tar -cf - -C $REMOTE_BAG_DIR $NAME | tar -xf -"
ssh "${SSH_OPTS[@]}" "$JETSON" \
    "$DOCKER exec $CONTAINER tar -cf - -C '$REMOTE_BAG_DIR' '$NAME'" \
  | tar -xf - -C "$LOCAL_BAG_DIR"
status=("${PIPESTATUS[@]}")
if [ "${status[0]}" != "0" ] || [ "${status[1]}" != "0" ]; then
  fail "transfer failed (ssh=${status[0]}, tar=${status[1]}).
The Jetson's copy is intact at $REMOTE_BAG -- retry with:
    $0 --container $CONTAINER -o $NAME --fetch-only"
fi

# ---------------------------------------------------------------------- verify
say "verifying"
remote_list=$(in_box "cd '$REMOTE_BAG' && find . -type f -printf '%P %s\n' | sort")
local_list=$(cd "$LOCAL_BAG_DIR/$NAME" && find . -type f -printf '%P %s\n' | sort)
if [ "$remote_list" != "$local_list" ]; then
  echo "the two copies differ. Keeping both. Diff (< remote, > local):" >&2
  diff <(echo "$remote_list") <(echo "$local_list") >&2
  fail "not deleting anything. Retry with --fetch-only."
fi
echo "  $(echo "$local_list" | grep -c .) files match by name and size"

# ---------------------------------------------------------------------- delete
if [ "$KEEP_REMOTE" = "1" ]; then
  say "keeping the Jetson's copy (--keep-remote) -- it is eating the 5.7 GB"
else
  say "deleting the Jetson's copy"
  if in_box "rm -rf '$REMOTE_BAG'"; then
    echo "  done, $(du -sh "$LOCAL_BAG_DIR/$NAME" | cut -f1) freed on the Jetson"
  else
    echo "  could NOT delete $REMOTE_BAG -- it is still using space on the Jetson." >&2
    echo "  Your copy here is verified and complete; remove the remote one with:" >&2
    echo "    ssh $JETSON \"$DOCKER exec -i $CONTAINER rm -rf '$REMOTE_BAG'\"" >&2
  fi
fi

echo
du -sh "$LOCAL_BAG_DIR/$NAME"
cat <<MSG

Write the ground truth into docs/dataset.md now, while you remember it:

  | $NAME | <distance m> | <lateral m> | <angle deg> | $RATE | $([ "$WITH_IMAGE" = 1 ] && echo "depth+image" || echo "depth") | <lens height m> | |

Then look at what you got:

  python3 pallet_approach/tools/bag_to_frames.py $LOCAL_BAG_DIR/$NAME --list
MSG
