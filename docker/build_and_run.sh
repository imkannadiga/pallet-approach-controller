#!/usr/bin/env bash
# Build and run our container on the Jetson.
#
#   ./docker/build_and_run.sh build      # build the image
#   ./docker/build_and_run.sh up         # start it and leave it running
#   ./docker/build_and_run.sh dry        # the same, with the wheels disconnected
#   ./docker/build_and_run.sh logs       # follow a running stack
#   ./docker/build_and_run.sh stop       # stop it
#   ./docker/build_and_run.sh launch     # launch in the foreground (Ctrl-C ends it)
#   ./docker/build_and_run.sh shell      # interactive shell, stack not launched
#   ./docker/build_and_run.sh dds        # show the DDS settings it would use
#   ./docker/build_and_run.sh start approach_only|pickup   # start a run, confirm it began
#   ./docker/build_and_run.sh cancel     # stop a run, confirm it stopped
#   ./docker/build_and_run.sh status     # /coordinator/status and /health, once
#
# ONE ENTRY POINT. `up` starts every node and then waits: nothing moves, and
# nothing will until a run is triggered on a topic. The sequence is
#
#   Nav2 drives the robot in front of a pallet and FINISHES
#   ./docker/build_and_run.sh start pickup
#   ros2 topic echo /pallet_approach/result
#
# and `cancel` stops it at any point:
#   ./docker/build_and_run.sh cancel
#
# `up` restarts the container if it exits, so a crashed stack comes back
# waiting rather than staying down. It does NOT resume a run: a restart leaves
# the robot stopped and IDLE, which is the only safe thing a restart can do.
#
# USE `dry` FOR THE FIRST BRING-UP. Everything runs and twist_mux arbitrates
# exactly as it will for real, but its output goes to /cmd_vel_dry, which the
# base is not listening to. The robot cannot move. See the launch file.
#
# Host networking is required: DDS discovery between the Jetson and the robot PC
# will not work through docker's bridge. Nothing else is shared with the host.
#
# Disk: the Jetson was at 90 % full (5.7 G free) when surveyed, so check before
# building. `docker system prune` first if the build fails on space.
#
# DDS: this container has to end up on the same DDS configuration as the ZED
# container, or it discovers nothing -- `ros2 topic echo` then says "does not
# appear to be published yet" even though the ZED is publishing happily two
# processes away. The Jetson is multi-homed (wifi 10.42.0.40, wired
# 192.168.1.1, docker0), so the ZED container runs with an explicit FastDDS
# profile that picks an interface, and a participant started without it may not
# match. This script now reads that configuration off the running ZED container
# rather than guessing a path, and REFUSES to start if it cannot -- an earlier
# version printed a note and carried on, which cost an evening.
#
# Override if you know better:
#   DDS_DIR=/path/to/configs DDS_PROFILE=fastdds_jetson.xml ./build_and_run.sh shell
#   DDS_DIR=none ./build_and_run.sh shell      # deliberately no profile
#
# GPU: the container runs with --runtime nvidia by default -- Jetson's own
# Docker is normally already configured with it (that is what the SDK Manager
# flow sets up), and every node here still runs fine without a GPU if it is
# not.
#
# --runtime nvidia alone only grants DEVICE access; it is NOT what makes the
# H.264 hardware encoder (the nvv4l2h264enc GStreamer element) exist inside
# the container. On x86 you would docker build an L4T base image or apt-install
# NVIDIA's L4T GStreamer packages to get it. On Jetson there is a second,
# usually-simpler path: nvidia-container-runtime on Jetson runs in "csv" mode,
# which reads manifests under
#   /etc/nvidia-container-runtime/host-files-for-container.d/*.csv
# on the HOST (installed by JetPack alongside nvidia-l4t-gstreamer /
# nvidia-l4t-multimedia) and bind-mounts the listed files -- the plugin .so's
# among them -- into any container started with --runtime nvidia. No L4T base
# image, no matching that exact L4T apt repo inside our own Dockerfile. This
# is why the image stays ros:humble-ros-base with only stock Ubuntu
# gstreamer1.0-* packages (docker/Dockerfile.jetson): if the CSV mounts are
# present on this Jetson, the element shows up anyway.
#
# NVIDIA_VISIBLE_DEVICES / NVIDIA_DRIVER_CAPABILITIES below are the standard
# incantation that tells the runtime to actually perform those mounts, not
# just hand over the raw devices.
#
# UNVERIFIED FROM HERE -- there is no Jetson in this environment to test on.
# What determines whether it actually works: whether SDK Manager's JetPack
# install populated those CSV files (normal on a standard flash) and whether
# the mounted plugin's ABI matches this image's stock GStreamer core closely
# enough to load (both are GStreamer 1.20.x on Ubuntu 22.04 for JetPack 6.x,
# which is the specific thing that makes this approach workable at all on
# JetPack 6 where it often was not on JetPack 4). Confirm on the robot:
#   ./docker/build_and_run.sh shell
#   gst-inspect-1.0 nvv4l2h264enc      # found -> hardware encode is live
# or just watch the video_streamer log line after `up`/`dry`: it states which
# encoder "auto" actually picked.
#
# If the element is not found: the fallback is installing nvidia-l4t-gstreamer
# in the Dockerfile from NVIDIA's own L4T apt repo, pinned to this exact L4T
# r36.5 build -- genuinely more fragile (a second apt source, version-pinned,
# arm64-only) and not done here. Ask if the CSV path does not pan out.
#   DOCKER_RUNTIME=none ./build_and_run.sh up     # no --runtime flag at all
#   DOCKER_RUNTIME=nvidia ./build_and_run.sh up   # the default, explicit

set -euo pipefail

IMAGE=pallet_approach:jetson
CONTAINER=pallet_approach
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------------------------------------------------------- DDS discovery

zed_container() {
  # The ZED wrapper, by image name. Falls back to any container whose name
  # mentions zed, because people rename things.
  docker ps --format '{{.Names}}\t{{.Image}}' \
    | awk -F'\t' 'tolower($2) ~ /zed/ || tolower($1) ~ /zed/ { print $1; exit }'
}

container_env() {   # container, var name
  docker inspect "$1" --format '{{range .Config.Env}}{{println .}}{{end}}' \
    | sed -n "s/^$2=//p" | head -1
}

host_path_for() {   # container, path inside it -> path on the host
  local container=$1 inside=$2
  docker inspect "$container" \
      --format '{{range .Mounts}}{{.Destination}}|{{.Source}}{{"\n"}}{{end}}' \
    | while IFS='|' read -r dest src; do
        [ -n "$dest" ] || continue
        case "$inside" in
          "$dest"|"$dest"/*) echo "${src}${inside#"$dest"}"; return 0 ;;
        esac
      done
}

# Returns via globals: DDS_HOST_DIR, DDS_PROFILE_NAME (both empty if none).
resolve_dds() {
  DDS_HOST_DIR=""; DDS_PROFILE_NAME=""

  if [ "${DDS_DIR:-}" = "none" ]; then
    echo "DDS: explicitly disabled (DDS_DIR=none)"
    return 0
  fi
  if [ -n "${DDS_DIR:-}" ]; then
    DDS_HOST_DIR=$DDS_DIR
    DDS_PROFILE_NAME=${DDS_PROFILE:-fastdds_jetson.xml}
    echo "DDS: using the override $DDS_HOST_DIR/$DDS_PROFILE_NAME"
    [ -f "$DDS_HOST_DIR/$DDS_PROFILE_NAME" ] \
      || { echo "  but that file does not exist" >&2; return 1; }
    return 0
  fi

  local zed profile host_profile
  zed=$(zed_container)
  if [ -z "$zed" ]; then
    cat >&2 <<MSG

REFUSING: no running ZED container found, so the DDS configuration this
container must match cannot be read.

  docker ps --format '{{.Names}}\t{{.Image}}'
$(docker ps --format '  {{.Names}}\t{{.Image}}')

Start the ZED wrapper first, or pass the profile yourself:
  DDS_DIR=/path/to/configs $0 ${1:-shell}
  DDS_DIR=none            $0 ${1:-shell}     # if no profile is needed
MSG
    return 1
  fi
  echo "DDS: reading the configuration off container '$zed'"

  # Domain and RMW have to match too, and a mismatch in either looks exactly
  # like a transport problem from the outside.
  local their_domain their_rmw
  their_domain=$(container_env "$zed" ROS_DOMAIN_ID)
  their_rmw=$(container_env "$zed" RMW_IMPLEMENTATION)
  echo "  ZED: ROS_DOMAIN_ID=${their_domain:-<unset, so 0>} RMW=${their_rmw:-<default>}"
  if [ -n "$their_domain" ] && [ "$their_domain" != "10" ]; then
    echo "REFUSING: the ZED container is on domain $their_domain, we use 10." >&2
    echo "  Set ROS_DOMAIN_ID=$their_domain here, or fix it there." >&2
    return 1
  fi
  if [ -n "$their_rmw" ] && [ "$their_rmw" != "rmw_fastrtps_cpp" ]; then
    echo "REFUSING: the ZED container uses $their_rmw, we use rmw_fastrtps_cpp." >&2
    return 1
  fi

  profile=$(container_env "$zed" FASTRTPS_DEFAULT_PROFILES_FILE)
  if [ -z "$profile" ]; then
    echo "  no FASTRTPS_DEFAULT_PROFILES_FILE there either, so neither side"
    echo "  uses a profile -- running without one, which matches."
    return 0
  fi
  echo "  profile inside that container: $profile"

  host_profile=$(host_path_for "$zed" "$profile")
  if [ -z "$host_profile" ] || [ ! -f "$host_profile" ]; then
    cat >&2 <<MSG

REFUSING: the ZED container uses the FastDDS profile
    $profile
but that path is inside ITS filesystem and no bind mount maps it back to this
host, so it cannot be mounted here. Its mounts are:

$(docker inspect "$zed" --format '{{range .Mounts}}    {{.Source}} -> {{.Destination}}{{"\n"}}{{end}}')

Copy the file out and point at it:
    docker cp $zed:$profile /tmp/dds/$(basename "$profile")
    DDS_DIR=/tmp/dds DDS_PROFILE=$(basename "$profile") $0 ${1:-shell}
MSG
    return 1
  fi

  DDS_HOST_DIR=$(dirname "$host_profile")
  DDS_PROFILE_NAME=$(basename "$host_profile")
  echo "  on this host: $host_profile"
}

# ---------------------------------------------------------------------- actions

case "${1:-}" in
  build)
    echo "free space:"; df -h /var/lib/docker 2>/dev/null || df -h /
    docker build -f "$REPO_ROOT/docker/Dockerfile.jetson" -t "$IMAGE" "$REPO_ROOT"
    ;;

  dds)
    resolve_dds shell
    echo
    echo "would run with:"
    if [ -n "${DDS_HOST_DIR:-}" ]; then
      echo "  -v $DDS_HOST_DIR:/dds:ro"
      echo "  -e FASTRTPS_DEFAULT_PROFILES_FILE=/dds/$DDS_PROFILE_NAME"
    else
      echo "  (no FastDDS profile)"
    fi
    ;;

  start|cancel)
    # Inside the stack's own container, so it is on exactly the DDS setup the
    # coordinator is. tools/trigger.py waits for the subscription to be matched
    # before publishing (a bare `ros2 topic pub --once` from another machine
    # often sends to nobody, and the coordinator then logs nothing at all) and
    # confirms the coordinator reacted.
    # Copied in from this checkout on every call, so it works with an image
    # built before the tool existed, without a rebuild.
    what=$1
    [ "$what" = start ] && what=${2:?usage: start approach_only|pickup}
    trigger="$REPO_ROOT/pallet_approach/tools/trigger.py"
    [ -f "$trigger" ] || { echo "missing $trigger: update this checkout" >&2; exit 1; }
    docker ps --format '{{.Names}}' | grep -qx "$CONTAINER" \
      || { echo "the stack container '$CONTAINER' is not running: ./docker/build_and_run.sh up" >&2; exit 1; }
    docker cp "$trigger" "$CONTAINER:/tmp/trigger.py" >/dev/null
    docker exec "$CONTAINER" bash -lc \
      "source /ros2_ws/install/setup.bash && python3 /tmp/trigger.py $what"
    ;;

  status)
    docker exec "$CONTAINER" bash -lc \
      'source /ros2_ws/install/setup.bash && ros2 topic echo --once /coordinator/status \
       && ros2 topic echo --once /health'
    ;;

  logs)
    docker logs -f "$CONTAINER"
    ;;

  stop)
    # Stop, then remove, so `up` can be run again without a name clash. A
    # stopped container with a restart policy comes back on daemon restart,
    # which is not what "stop" should mean.
    docker stop "$CONTAINER" >/dev/null 2>&1 || true
    docker rm "$CONTAINER" >/dev/null 2>&1 || true
    echo "stopped and removed $CONTAINER"
    ;;

  shell|launch|up|dry)
    resolve_dds "$1"
    ARGS=(--network host --ipc host
          -e ROS_DOMAIN_ID=10 -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp
          -v "$REPO_ROOT:/workspace:ro")
    # --runtime alone is device access; these two env vars are what tell
    # Jetson's nvidia-container-runtime to also perform the CSV-manifest
    # library mounts that make nvv4l2h264enc possible -- see the long comment
    # near the top of this file.
    RUNTIME=${DOCKER_RUNTIME:-nvidia}
    if [ "$RUNTIME" != "none" ]; then
      echo "GPU: --runtime $RUNTIME (DOCKER_RUNTIME=none to disable)"
      ARGS+=(--runtime "$RUNTIME"
             -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all)
    else
      echo "GPU: none requested (DOCKER_RUNTIME=none)"
    fi
    case "$1" in
      up|dry)
        # Detached, named, and restarted if it exits. One entry point: start
        # this and the whole stack is up and waiting for a trigger.
        ARGS+=(-d --name "$CONTAINER" --restart unless-stopped)
        if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
          echo "REFUSING: a container named $CONTAINER already exists." >&2
          echo "  ./docker/build_and_run.sh stop    # then try again" >&2
          echo "  ./docker/build_and_run.sh logs    # or just look at it" >&2
          exit 1
        fi
        ;;
      *) ARGS+=(--rm -it) ;;
    esac
    if [ -n "${DDS_HOST_DIR:-}" ]; then
      ARGS+=(-v "$DDS_HOST_DIR:/dds:ro"
             -e "FASTRTPS_DEFAULT_PROFILES_FILE=/dds/$DDS_PROFILE_NAME")
    fi
    # --ipc host shares /dev/shm, which FastDDS uses for same-host transport.
    # If the ZED container does NOT share it, the two have different shared
    # memory and large messages never arrive even once discovery works.
    zed=$(zed_container)
    if [ -n "$zed" ]; then
      ipc=$(docker inspect "$zed" --format '{{.HostConfig.IpcMode}}')
      if [ "$ipc" != "host" ]; then
        echo "note: the ZED container has IpcMode='$ipc', not 'host', so it has" >&2
        echo "  its own /dev/shm. Discovery will work; bulk data over shared" >&2
        echo "  memory may not. Restart it with --ipc host if topics are" >&2
        echo "  discovered but carry no data." >&2
      fi
    fi
    case "$1" in
      shell)
        docker run "${ARGS[@]}" "$IMAGE" bash ;;
      dry)
        docker run "${ARGS[@]}" "$IMAGE" bash -lc \
          'source /ros2_ws/install/setup.bash && ros2 launch pallet_approach pallet_approach.launch.py cmd_vel:=/cmd_vel_dry'
        JETSON_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
        echo
        echo "started $CONTAINER with the wheels DISCONNECTED."
        echo "  commands go to /cmd_vel_dry; the base will not see them."
        echo "  ./docker/build_and_run.sh logs"
        echo
        echo "watch it: http://${JETSON_IP:-<jetson-ip>}:8889/dashboard"
        ;;
      up)
        docker run "${ARGS[@]}" "$IMAGE"
        JETSON_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
        echo
        echo "started $CONTAINER. Nothing moves until a run is triggered:"
        echo "  ./docker/build_and_run.sh start approach_only"
        echo "  ros2 topic echo /pallet_approach/result"
        echo "  ./docker/build_and_run.sh logs"
        echo
        echo "watch it: http://${JETSON_IP:-<jetson-ip>}:8889/dashboard"
        echo "  (video_stream:=false to skip MediaMTX if it is not wanted)"
        ;;
      *)
        docker run "${ARGS[@]}" "$IMAGE" ;;
    esac
    ;;

  *)
    sed -n '2,51p' "$0"; exit 2 ;;
esac
