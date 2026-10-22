#!/usr/bin/env bash
# The whole stack against a Gazebo SCOUT and the measured pallet, on a laptop.
#
#   ./sim/run_sim.sh build                 # both images (the stack's is the robot's own Dockerfile)
#   ./sim/run_sim.sh up [launch args]      # Gazebo, then the stack, both detached
#   ./sim/run_sim.sh up face_x:=2.5 face_y:=0.3 face_yaw:=0.2
#   ./sim/run_sim.sh start approach_only   # trigger a run, as Nav2's caller would
#   ./sim/run_sim.sh start pickup
#   ./sim/run_sim.sh run approach_only    # start and follow it to the result
#   ./sim/run_sim.sh cancel
#   ./sim/run_sim.sh result                # wait for and print /pallet_approach/result
#   ./sim/run_sim.sh status                # /coordinator/status and /health, once
#   ./sim/run_sim.sh logs [gazebo|stack]
#   ./sim/run_sim.sh shell                 # a ROS shell on the sim's domain
#   ./sim/run_sim.sh stop
#
# Dashboard: http://localhost:8889/dashboard, the same page as on the robot.
#
# SAFETY. The sim runs on ROS_DOMAIN_ID 42 with ROS_LOCALHOST_ONLY=1. The
# robot is on domain 10, so if this laptop is on the robot's Wi-Fi, the sim's
# /cmd_vel cannot reach the real base, and the real ZED cannot leak into the
# sim. Do not change either without thinking about that.
#
# The stack container is the SAME image the Jetson runs, built for amd64 from
# docker/Dockerfile.jetson, started with the same launch file. Two things are
# added: overlay_params:=config/sim_overlay.yaml (sim time, no camera tilt
# correction) and encoder:=x264 (no Jetson encoder here).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIM_IMAGE=pallet_approach:gazebo
STACK_IMAGE=pallet_approach:sim-stack
GAZEBO=pallet_sim_gazebo
STACK=pallet_sim_stack
DOMAIN=42

# fastdds_sim.xml: big shared-memory segments so camera images do not fall
# back to fragmented UDP (where best-effort subscribers get nothing), with UDP
# pinned to loopback. GIO_USE_PROXY_RESOLVER: GStreamer's RTSP client asks GIO
# for an HTTP proxy, GIO asks libproxy, and libproxy in this image throws
# "Unable to read configuration" and aborts the video streamer.
ROS_ENV=(-e ROS_DOMAIN_ID=$DOMAIN -e ROS_LOCALHOST_ONLY=1
         -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp
         -v "$REPO_ROOT/sim/fastdds_sim.xml:/sim_dds/fastdds_sim.xml:ro"
         -e FASTRTPS_DEFAULT_PROFILES_FILE=/sim_dds/fastdds_sim.xml
         -e GIO_USE_PROXY_RESOLVER=dummy)

in_stack() {   # run a ROS command inside the stack container
  docker exec "$STACK" bash -lc "source /ros2_ws/install/setup.bash && $*"
}

case "${1:-}" in
  build)
    docker image inspect sim-env/hunter-ugv-avp-sim:1.1 >/dev/null 2>&1 || {
      echo "the base image sim-env/hunter-ugv-avp-sim:1.1 is missing." >&2
      echo "  build it in intelligent-fleet-sim-env: docker compose build hunter-ugv-avp-sim" >&2
      exit 1; }
    docker build -f "$REPO_ROOT/sim/Dockerfile.gazebo" -t "$SIM_IMAGE" "$REPO_ROOT"
    docker build -f "$REPO_ROOT/docker/Dockerfile.jetson" -t "$STACK_IMAGE" "$REPO_ROOT"
    ;;

  up)
    shift
    for c in $GAZEBO $STACK; do
      if docker ps -a --format '{{.Names}}' | grep -qx "$c"; then
        echo "REFUSING: $c already exists. ./sim/run_sim.sh stop first." >&2; exit 1
      fi
    done
    # /dev/dri: the camera and the LiDAR are rendered (ogre2), on the iGPU if
    # it is passed through, in software otherwise -- much slower, and the
    # real-time factor drops. Everything runs on sim time, so slower is
    # slower, not wrong.
    # The image runs as a non-root user, so it also needs the groups that own
    # the device nodes (video for card*, render for renderD*), by the HOST's
    # numeric ids -- without them EGL fails with "amdgpu_device_initialize
    # failed" and Gazebo segfaults creating its offscreen window.
    # An NVIDIA GPU, if the host has a working one: `graphics` in the
    # capabilities is what mounts the EGL driver that ogre2 renders with
    # headless. Otherwise /dev/dri (Mesa), as on an AMD/Intel laptop.
    GPU=()
    if nvidia-smi >/dev/null 2>&1; then
      GPU=(--gpus all -e NVIDIA_DRIVER_CAPABILITIES=all)
    elif [ -e /dev/dri ]; then
      GPU=(--device /dev/dri)
      for gid in $(stat -c %g /dev/dri/card* /dev/dri/renderD* 2>/dev/null | sort -u); do
        GPU+=(--group-add "$gid")
      done
    fi
    X11=()
    # gui:=true shows the Gazebo window, and opens the dashboard in a browser,
    # on the screen of the machine running this. It works from an SSH session
    # too (sim/remote.sh up gui:=true): the display is the first local X
    # server (/tmp/.X11-unix/X1 -> :1) unless SIM_DISPLAY or DISPLAY says
    # otherwise, and the desktop's own cookie (gdm's) is what lets xhost allow
    # local containers to draw on it.
    GUI=false
    if printf '%s\n' "$@" | grep -qx 'gui:=true'; then
      GUI=true
      SHOW_DISPLAY=${SIM_DISPLAY:-${DISPLAY:-}}
      if [ -z "$SHOW_DISPLAY" ]; then
        sock=$(ls /tmp/.X11-unix 2>/dev/null | sed -n 's/^X//p' | head -1)
        SHOW_DISPLAY=":${sock:-0}"
      fi
      SHOW_AUTH=${XAUTHORITY:-}
      [ -z "$SHOW_AUTH" ] && [ -f "/run/user/$(id -u)/gdm/Xauthority" ] \
        && SHOW_AUTH=/run/user/$(id -u)/gdm/Xauthority
      [ -z "$SHOW_AUTH" ] && SHOW_AUTH=$HOME/.Xauthority
      if ! DISPLAY=$SHOW_DISPLAY XAUTHORITY=$SHOW_AUTH xhost +local: >/dev/null 2>&1; then
        echo "warning: could not open display $SHOW_DISPLAY (xhost failed);" >&2
        echo "  the Gazebo window may not appear. Set SIM_DISPLAY / XAUTHORITY." >&2
      fi
      X11=(-e DISPLAY="$SHOW_DISPLAY" -e QT_X11_NO_MITSHM=1
           -v /tmp/.X11-unix:/tmp/.X11-unix:ro)
    fi
    # As root, like the stack container. The base image's default user is uid
    # 1000, and FastDDS's shared-memory segments in /dev/shm are writable only
    # by their owner: discovery then works (UDP) but no message from Gazebo
    # ever reaches the stack -- `ros2 topic list` shows /clock and every
    # sensor, and nothing arrives.
    # IGN_IP: Ignition transport (Gazebo <-> the ROS bridge, both in this
    # container) finds its peers by multicast on the host's network interface.
    # With the laptop off the network that fails ("Exception sending a
    # multicast message: Network is unreachable") and no sensor reaches ROS.
    # Loopback works with or without a network.
    docker run -d --name "$GAZEBO" -u 0 --network host --ipc host "${GPU[@]}" "${X11[@]}" \
      -e IGN_IP=127.0.0.1 -e GZ_IP=127.0.0.1 \
      "${ROS_ENV[@]}" "$SIM_IMAGE" bash -lc \
      "source /sim_ws/install/setup.bash && ros2 launch pallet_approach_sim sim.launch.py $*"
    # WebRTC over TCP as well as UDP, advertising 127.0.0.1: then the video
    # can be watched through an SSH tunnel (sim/remote.sh watch), which only
    # carries TCP. MediaMTX reads MTX_* variables as config overrides, so the
    # robot's config/mediamtx.yaml is untouched.
    docker run -d --name "$STACK" --network host --ipc host "${ROS_ENV[@]}" \
      -e MTX_WEBRTCLOCALTCPADDRESS=:8189 -e MTX_WEBRTCADDITIONALHOSTS=127.0.0.1 \
      "$STACK_IMAGE" bash -lc \
      'source /ros2_ws/install/setup.bash && ros2 launch pallet_approach pallet_approach.launch.py \
         overlay_params:=/ros2_ws/install/pallet_approach/share/pallet_approach/config/sim_overlay.yaml \
         encoder:=x264'
    echo "started $GAZEBO and $STACK on domain $DOMAIN (localhost only)."
    echo "  give Gazebo ~20 s to spawn the robot, then:"
    echo "  ./sim/run_sim.sh status"
    echo "  ./sim/run_sim.sh start approach_only"
    echo "  http://localhost:8889/dashboard"
    if [ "$GUI" = true ] && [ "${SIM_OPEN_BROWSER:-true}" = true ]; then
      # The dashboard next to the Gazebo window: once MediaMTX is serving.
      # SIM_OPEN_BROWSER=false skips it (batch_runs.sh after its first run:
      # the open page reconnects to the restarted stream by itself).
      ( sleep 25; DISPLAY=$SHOW_DISPLAY XAUTHORITY=$SHOW_AUTH \
          xdg-open http://localhost:8889/dashboard >/dev/null 2>&1 ) &
      echo "  Gazebo window and dashboard on display $SHOW_DISPLAY"
    fi
    ;;

  start)
    mode=${2:?usage: start approach_only|pickup}
    in_stack "ros2 topic pub --once /pallet_approach/start std_msgs/msg/String '{data: $mode}'"
    ;;

  run)
    # Start a run and follow it to its result, one line per state change.
    mode=${2:?usage: run approach_only|pickup [timeout_s]}
    docker cp "$REPO_ROOT/sim/tools/follow_run.py" "$STACK:/tmp/follow_run.py" >/dev/null
    in_stack "python3 -u /tmp/follow_run.py $mode ${3:-1800}"
    ;;

  cancel)
    in_stack "ros2 topic pub --once /pallet_approach/cancel std_msgs/msg/Empty '{}'"
    ;;

  result)
    in_stack "ros2 topic echo --once /pallet_approach/result"
    ;;

  status)
    in_stack "ros2 topic echo --once /coordinator/status; ros2 topic echo --once /health"
    ;;

  logs)
    case "${2:-stack}" in
      gazebo) docker logs -f "$GAZEBO" ;;
      *)      docker logs -f "$STACK" ;;
    esac
    ;;

  shell)
    docker exec -it "$STACK" bash -lc 'source /ros2_ws/install/setup.bash && exec bash'
    ;;

  stop)
    for c in $STACK $GAZEBO; do
      docker rm -f "$c" >/dev/null 2>&1 || true
    done
    echo "stopped and removed $STACK and $GAZEBO"
    ;;

  *)
    sed -n '2,27p' "$0"; exit 2 ;;
esac
