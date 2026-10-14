#!/usr/bin/env bash
# Build and run our container on the Jetson.
#
#   ./docker/build_and_run.sh build      # build the image
#   ./docker/build_and_run.sh shell      # interactive shell, stack not launched
#   ./docker/build_and_run.sh launch     # launch the stack
#
# Host networking is required: DDS discovery between the Jetson and the robot PC
# will not work through docker's bridge. Nothing else is shared with the host.
#
# Disk: the Jetson was at 90 % full (5.7 G free) when surveyed, so check before
# building. `docker system prune` first if the build fails on space.

set -euo pipefail

IMAGE=pallet_approach:jetson
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DDS_DIR=${DDS_DIR:-/ros2_ws/src/scout_nlp/configs}

case "${1:-}" in
  build)
    echo "free space:"; df -h /var/lib/docker 2>/dev/null || df -h /
    docker build -f "$REPO_ROOT/docker/Dockerfile.jetson" -t "$IMAGE" "$REPO_ROOT"
    ;;
  shell|launch)
    ARGS=(--rm -it --network host --ipc host
          -e ROS_DOMAIN_ID=10 -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp
          -v "$REPO_ROOT:/workspace:ro")
    if [ -d "$DDS_DIR" ]; then
      ARGS+=(-v "$DDS_DIR:/dds:ro"
             -e FASTRTPS_DEFAULT_PROFILES_FILE=/dds/fastdds_jetson.xml)
    else
      echo "note: $DDS_DIR not found, running without an explicit FastDDS profile" >&2
    fi
    if [ "$1" = "shell" ]; then
      docker run "${ARGS[@]}" "$IMAGE" bash
    else
      docker run "${ARGS[@]}" "$IMAGE"
    fi
    ;;
  *)
    sed -n '2,14p' "$0"; exit 2 ;;
esac
