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
LIDAR=/ouster/points
WITH_LIDAR=0

while [ $# -gt 0 ]; do
  case "$1" in
    -o|--out)       NAME="$2"; shift 2 ;;
    -d|--duration)  DURATION="$2"; shift 2 ;;
    -r|--rate)      RATE="$2"; shift 2 ;;
    --full-rate)    FULL_RATE=1; shift ;;
    --with-lidar)   WITH_LIDAR=1; shift ;;
    --no-image)     WITH_IMAGE=0; shift ;;
    --lidar-only)   LIDAR_ONLY=1; WITH_LIDAR=1; WITH_IMAGE=0; shift ;;
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
[ "$FULL_RATE" = "1" ] && EFF_RATE=37

[ "$WITH_IMAGE" = "0" ] && IMAGE_FRAME_MB=0
[ "$LIDAR_ONLY" = "1" ] && DEPTH_FRAME_MB=0
EST_MB=$(awk -v d="$DURATION" -v r="$EFF_RATE" -v df="$DEPTH_FRAME_MB" -v if_="$IMAGE_FRAME_MB" \
  'BEGIN { printf "%.0f", d * r * (df + if_) }')
[ "$WITH_LIDAR" = "1" ] && EST_MB=$(awk -v e="$EST_MB" -v d="$DURATION" \
  -v lf="$LIDAR_FRAME_MB" -v lr="$LIDAR_RATE" 'BEGIN{printf "%.0f", e + d*lr*lf}')

FREE_MB=$(df -Pm "$OUT_DIR" 2>/dev/null | awk 'NR==2{print $4}' || echo 0)

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

cleanup() {
  for pid in "${THROTTLE_PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT

if [ "$LIDAR_ONLY" = "1" ]; then
  echo "LiDAR only: no camera topics, no throttling (the Ouster runs at ~10 Hz)"
elif [ "$FULL_RATE" = "1" ]; then
  echo "WARNING: recording at full rate. Watch the disk."
  TOPICS+=("$DEPTH")
  [ "$WITH_IMAGE" = "1" ] && TOPICS+=("$IMAGE")
elif ros2 pkg executables topic_tools 2>/dev/null | grep -q throttle; then
  echo "throttling to ${RATE} Hz"
  ros2 run topic_tools throttle messages "$DEPTH" "$RATE" "${DEPTH}_throttled" \
      >/dev/null 2>&1 & THROTTLE_PIDS+=($!)
  TOPICS+=("${DEPTH}_throttled")
  if [ "$WITH_IMAGE" = "1" ]; then
    ros2 run topic_tools throttle messages "$IMAGE" "$RATE" "${IMAGE}_throttled" \
        >/dev/null 2>&1 & THROTTLE_PIDS+=($!)
    TOPICS+=("${IMAGE}_throttled")
  fi
  sleep 2
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
cat <<MSG

Done. Now write the ground truth for this bag into docs/dataset.md:

  | $NAME | <distance m> | <lateral m> | <angle deg> | <notes> |

A bag without tape-measured ground truth cannot be used for evaluation.
MSG
