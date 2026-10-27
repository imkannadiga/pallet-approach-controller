#!/usr/bin/env bash
# Record a pallet dataset bag on the Jetson.
#
# Why this is not just `ros2 bag record -a`: raw depth is 1280x720 float32 at
# ~37 Hz = 136 MB/s, the rectified colour image adds ~102 MB/s, and the Jetson
# has about 5.7 GB free (docs/SYSTEM.md). Unthrottled, 20 seconds fills the disk.
#
# So by default this throttles depth and image to 10 Hz with topic_tools, which
# is plenty for a pallet that is not moving fast, and refuses to start if the
# estimated size does not fit in the free space with a margin.
#
# Usage:
#   ./record_bag.sh -o pallet_2m_0deg -d 20
#   ./record_bag.sh -o pallet_2m_0deg -d 20 -r 5         # 5 Hz, smaller
#   ./record_bag.sh -o pallet_2m_0deg -d 20 --no-image   # depth only, half the size
#   ./record_bag.sh -o floor_lidar -d 20 --lidar-only    # LiDAR + tf + odom, no camera
#                                                        # (run this one on the robot PC,
#                                                        #  where the Ouster driver is local)
#   ./record_bag.sh -o pallet_2m_0deg -d 20 --compress    # zstd as it writes
#   ./record_bag.sh -o pallet_2m_0deg -d 20 --full-rate  # no throttle, be careful
#   ./record_bag.sh --list                               # what it would record
#
# To record and pull it straight to your laptop in one command, use
# tools/fetch_bag.sh from the laptop instead -- it calls this script over ssh,
# verifies the copy and only then deletes the Jetson's copy.
#
# Write the ground truth for each bag into docs/dataset.md as you go: tape
# measured distance, lateral offset and angle. A bag without ground truth cannot
# be used for evaluation.

set -euo pipefail

# ROS is not on PATH in a non-interactive shell in these containers, so source
# it here. That makes this script self-sufficient: it works the same whether it
# is run by hand inside the container or piped into `bash -s` over ssh by
# fetch_bag.sh. The setup scripts reference unset variables and return non-zero,
# so -e and -u come off around them.
if ! command -v ros2 >/dev/null 2>&1; then
  set +eu
  if [ -f /opt/ros/humble/setup.bash ]; then . /opt/ros/humble/setup.bash; fi
  if [ -f /ros2_ws/install/setup.bash ]; then . /ros2_ws/install/setup.bash; fi
  set -eu
  if ! command -v ros2 >/dev/null 2>&1; then
    echo "ros2 is not on PATH and sourcing /opt/ros/humble and /ros2_ws did not" >&2
    echo "find it. Source the right workspace first, or set it up in this shell." >&2
    exit 1
  fi
fi

NAME=""
DURATION=20
RATE=10
FULL_RATE=0
LIST_ONLY=0
WITH_IMAGE=1
LIDAR_ONLY=0
COMPRESS=0
STORAGE=""
OUT_DIR="${BAG_DIR:-/tmp/pallet_bags}"

DEPTH=/zed2/zed_node/depth/depth_registered
DEPTH_INFO=/zed2/zed_node/depth/camera_info
IMAGE=/zed2/zed_node/rgb/color/rect/image
IMAGE_INFO=/zed2/zed_node/rgb/color/rect/camera_info
EXTRA=(/tf /tf_static /odom /joint_states /scout_status)
# The Jetson's local copy, from lidar_relay. NEVER record the raw /ouster/points
# from a machine other than the robot PC: a second network subscriber to a
# 3 MB / 10 Hz cloud throttled the Ouster to 2.6-4.6 Hz (docs/dataset.md) and
# queued everything else from the robot PC behind it (lidar_relay_node.py).
# --lidar-only is for the robot PC itself, where the raw topic is local, and
# switches to it. The stack must be up for the relay to exist.
LIDAR=/pallet_approach/lidar
WITH_LIDAR=0

while [ $# -gt 0 ]; do
  case "$1" in
    -o|--out)       NAME="$2"; shift 2 ;;
    -d|--duration)  DURATION="$2"; shift 2 ;;
    -r|--rate)      RATE="$2"; shift 2 ;;
    --full-rate)    FULL_RATE=1; shift ;;
    --with-lidar)   WITH_LIDAR=1; shift ;;
    --no-image)     WITH_IMAGE=0; shift ;;
    --lidar-only)   LIDAR_ONLY=1; WITH_LIDAR=1; WITH_IMAGE=0; LIDAR=/ouster/points; shift ;;
    --lidar-topic)  LIDAR="$2"; shift 2 ;;
    --compress)     COMPRESS=1; shift ;;
    --mcap)         STORAGE="mcap"; shift ;;
    --dir)          OUT_DIR="$2"; shift 2 ;;
    --list)         LIST_ONLY=1; shift ;;
    -h|--help)      sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------- size estimate
# 1280*720*4 bytes for depth, *3 for bgr8 colour.
DEPTH_FRAME_MB=3.69
IMAGE_FRAME_MB=2.76
# measured from the survey: /ouster/points is 64 x 1024, point_step 48,
# so 3,145,728 bytes per frame. Assumes a 10 Hz scan rate -- double it at 20 Hz.
LIDAR_FRAME_MB=3.15
LIDAR_RATE=10
EFF_RATE=$RATE
# [measured 2026-10-15] ~56 Hz with grab_frame_rate 60 at HD720. The survey
# recorded ~37 Hz on 2026-10-13, so the camera's configuration changed between
# them -- reconcile before trusting either. 56 is the safe one to size against:
# at 37 a 20 s unthrottled bag of depth+colour estimates 4.8 GB and really
# takes 7.3 GB, which is more than the Jetson has free.
[ "$FULL_RATE" = "1" ] && EFF_RATE=56

[ "$WITH_IMAGE" = "0" ] && IMAGE_FRAME_MB=0
[ "$LIDAR_ONLY" = "1" ] && DEPTH_FRAME_MB=0
EST_MB=$(awk -v d="$DURATION" -v r="$EFF_RATE" -v df="$DEPTH_FRAME_MB" -v if_="$IMAGE_FRAME_MB" \
  'BEGIN { printf "%.0f", d * r * (df + if_) }')
[ "$WITH_LIDAR" = "1" ] && EST_MB=$(awk -v e="$EST_MB" -v d="$DURATION" \
  -v lf="$LIDAR_FRAME_MB" -v lr="$LIDAR_RATE" 'BEGIN{printf "%.0f", e + d*lr*lf}')

# df on a directory that does not exist yet fails, which left FREE_MB at 0 and
# silently switched the "would leave under 1 GB free" refusal off for the first
# bag of a session. Ask about the nearest directory that does exist.
CHECK_DIR="$OUT_DIR"
while [ ! -d "$CHECK_DIR" ] && [ "$CHECK_DIR" != "/" ]; do CHECK_DIR=$(dirname "$CHECK_DIR"); done
FREE_MB=$(df -Pm "$CHECK_DIR" 2>/dev/null | awk 'NR==2{print $4}') || true
FREE_MB=${FREE_MB:-0}

if [ "$LIST_ONLY" = "1" ]; then
  echo "would record, for ${DURATION}s at ${EFF_RATE} Hz:"
  if [ "$LIDAR_ONLY" = "0" ]; then
    printf '  %s\n' "$DEPTH" "$DEPTH_INFO"
    [ "$WITH_IMAGE" = "1" ] && printf '  %s\n' "$IMAGE" "$IMAGE_INFO"
  fi
  printf '  %s\n' "${EXTRA[@]}"
  [ "$WITH_LIDAR" = "1" ] && printf '  %s\n' "$LIDAR"
  echo "estimated size: ${EST_MB} MB (uncompressed)    free at ${OUT_DIR}: ${FREE_MB} MB"
  [ "$COMPRESS" = "1" ] && echo "zstd is on, so expect well under that -- measure it"
  exit 0
fi

[ -n "$NAME" ] || { echo "need -o <name>" >&2; exit 2; }
mkdir -p "$OUT_DIR"
BAG="$OUT_DIR/$NAME"
[ -e "$BAG" ] && { echo "refusing to overwrite $BAG" >&2; exit 1; }

echo "estimated size: ${EST_MB} MB    free at ${OUT_DIR}: ${FREE_MB} MB"
if [ "$FREE_MB" -gt 0 ] && [ "$EST_MB" -gt $((FREE_MB - 1024)) ]; then
  cat >&2 <<MSG

REFUSING: ${EST_MB} MB would leave under 1 GB free at ${OUT_DIR}.
Options: lower -r, shorten -d, record to an external disk with --dir,
or record an SVO instead:
    ros2 service call /zed2/zed_node/start_svo_rec zed_msgs/srv/StartSvoRec \\
        "{svo_filename: '/mnt/ssd/pallet.svo2'}"
    # plus a small bag of /tf /tf_static /odom, then stop_svo_rec
MSG
  exit 1
fi

# ---------------------------------------------------------------- throttling
TOPICS=("${EXTRA[@]}")
if [ "$LIDAR_ONLY" = "0" ]; then
  TOPICS+=("$DEPTH_INFO")
  [ "$WITH_IMAGE" = "1" ] && TOPICS+=("$IMAGE_INFO")
fi
[ "$WITH_LIDAR" = "1" ] && TOPICS+=("$LIDAR")
THROTTLE_PIDS=()

# `ros2 run` is a Python launcher that spawns the node as a CHILD process, so
# SIGTERM to the pid we hold does not necessarily reach the node. Measured
# 2026-10-15: eight bags recorded in one session left one throttle node running
# each time, and because every one of them republishes to the SAME
# `..._throttled` topic, the recorded rate climbed 4.8 -> 8.7 -> 12.8 -> ... ->
# 32 Hz across the session while the Jetson's CPU went to the strays. So: start
# each throttle in its own process group with setsid and signal the whole group.
THROTTLE_PATTERN='topic_tools.*throttle|throttle messages'

sweep_strays() {
  command -v pkill >/dev/null 2>&1 || return 0
  pkill -f "$THROTTLE_PATTERN" 2>/dev/null || true
}

cleanup() {
  for pid in "${THROTTLE_PIDS[@]:-}"; do
    kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  done
  sleep 0.5
  for pid in "${THROTTLE_PIDS[@]:-}"; do
    kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  done
  # belt and braces: anything of ours that outlived its group
  if [ "${#THROTTLE_PIDS[@]}" -gt 0 ]; then sweep_strays; fi
}
trap cleanup EXIT

# Captured first, then searched. `ros2 pkg executables ... | grep -q throttle`
# under `set -o pipefail` reports FAILURE when grep exits at its match and the
# writer, still printing, gets SIGPIPE (status 141): an installed topic_tools
# then reads as "not installed", although the same probe without pipefail, in
# fetch_bag.sh, passed.
TOPIC_TOOLS_EXES=$(ros2 pkg executables topic_tools 2>/dev/null || true)

if [ "$LIDAR_ONLY" = "1" ]; then
  echo "LiDAR only: no camera topics, no throttling (the Ouster runs at ~10 Hz)"
elif [ "$FULL_RATE" = "1" ]; then
  echo "WARNING: recording at full rate. Watch the disk."
  TOPICS+=("$DEPTH")
  [ "$WITH_IMAGE" = "1" ] && TOPICS+=("$IMAGE")
elif grep -q throttle <<<"$TOPIC_TOOLS_EXES"; then
  # A throttle left over from a previous run publishes to the same output topic,
  # so the recorded rate would be a multiple of RATE and the bag unusable for
  # anything rate-sensitive. Clear them out first and say so.
  if command -v pgrep >/dev/null 2>&1 && pgrep -f "$THROTTLE_PATTERN" >/dev/null 2>&1; then
    echo "WARNING: throttle nodes from an earlier run are still alive:"
    pgrep -af "$THROTTLE_PATTERN" | sed 's/^/    /'
    echo "  killing them -- left alone they would multiply the recorded rate"
    sweep_strays
    sleep 1
  fi

  echo "throttling to ${RATE} Hz"
  setsid ros2 run topic_tools throttle messages "$DEPTH" "$RATE" \
      "${DEPTH}_throttled" >/dev/null 2>&1 & THROTTLE_PIDS+=($!)
  TOPICS+=("${DEPTH}_throttled")
  if [ "$WITH_IMAGE" = "1" ]; then
    setsid ros2 run topic_tools throttle messages "$IMAGE" "$RATE" \
        "${IMAGE}_throttled" >/dev/null 2>&1 & THROTTLE_PIDS+=($!)
    TOPICS+=("${IMAGE}_throttled")
  fi
  sleep 2

  # One publisher per throttled topic, or the rate is wrong. Cheap to check and
  # it catches the failure at the start of the bag rather than on the laptop.
  for t in "${DEPTH}_throttled" $([ "$WITH_IMAGE" = "1" ] && echo "${IMAGE}_throttled"); do
    n=$(ros2 topic info "$t" 2>/dev/null | awk '/Publisher count/{print $3}')
    if [ -n "$n" ] && [ "$n" != "1" ]; then
      echo "REFUSING: $t has $n publishers, so the bag would record about" >&2
      echo "  ${n}x ${RATE} Hz. Kill the strays and try again:" >&2
      echo "    pkill -f 'topic_tools.*throttle'" >&2
      exit 1
    fi
  done
else
  cat >&2 <<'MSG'
topic_tools is not installed, so depth cannot be throttled. Either install it
(apt install ros-humble-topic-tools) or pass --full-rate and keep the bag very
short. Not recording.
MSG
  exit 1
fi

RECORD_OPTS=(-o "$BAG" --max-cache-size 268435456)
if [ -n "$STORAGE" ]; then
  RECORD_OPTS+=(-s "$STORAGE")
fi
if [ "$COMPRESS" = "1" ]; then
  # `message` mode, not `file`: it writes compressed as it goes, where `file`
  # writes the bag uncompressed and compresses it afterwards -- which needs
  # roughly twice the peak disk, and disk is the thing we are short of.
  RECORD_OPTS+=(--compression-mode message --compression-format zstd)
fi

echo
echo "recording ${DURATION}s to ${BAG}"
echo "topics:"; printf '  %s\n' "${TOPICS[@]}"
echo
ros2 bag record "${RECORD_OPTS[@]}" "${TOPICS[@]}" &
RECORD_PID=$!
sleep "$DURATION"
kill -INT "$RECORD_PID" 2>/dev/null || true
wait "$RECORD_PID" 2>/dev/null || true

echo
du -sh "$BAG" 2>/dev/null || true

# Rates as recorded, not as intended. This is the check that would have caught
# the stray-throttle bug on the first bag instead of the eighth.
if command -v python3 >/dev/null 2>&1; then
  echo
  echo "recorded rates:"
  export RATE
  python3 - "$BAG" <<'PY' 2>/dev/null || echo "  (could not read the bag back)"
import glob, sqlite3, sys
db = (glob.glob(f'{sys.argv[1]}/*.db3') + glob.glob(f'{sys.argv[1]}/*.mcap'))
if not db or db[0].endswith('.mcap'):
    raise SystemExit(0)
c = sqlite3.connect(db[0])
for name, n, span in c.execute(
        'select t.name, count(m.id), (max(m.timestamp)-min(m.timestamp))/1e9 '
        'from topics t left join messages m on m.topic_id=t.id group by t.id'):
    if not n or not span:
        continue
    hz = n / span
    flag = ''
    if 'throttled' in name and hz > 1.5 * float(__import__('os').environ.get('RATE', '10')):
        flag = '   <-- HIGHER THAN ASKED FOR: stray throttle node?'
    if name in ('/ouster/points', '/pallet_approach/lidar') and hz < 7:
        flag = '   <-- the Ouster is being throttled; see docs/SYSTEM.md'
    print(f'  {name:<58} {n:>5} msgs  {hz:>6.2f} Hz{flag}')
PY
fi

cat <<MSG

Done. Now write the ground truth for this bag into docs/dataset.md:

  | $NAME | <distance m> | <lateral m> | <angle deg> | <notes> |

A bag without tape-measured ground truth cannot be used for evaluation.
MSG
