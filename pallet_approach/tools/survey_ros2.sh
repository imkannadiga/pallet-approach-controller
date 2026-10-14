#!/usr/bin/env bash
# Read-only survey of one ROS 2 machine or container.
#
# Answers the discoverable open questions in CLAUDE.md: topic names and types,
# depth encoding and QoS, tf tree and camera-to-base_link transform, the URDF,
# cmd_vel publishers, base driver parameters, multi-machine visibility,
# and whether Open3D can be installed here.
#
# It only reads: lists, info, one-shot echoes, parameter dumps. It publishes no
# topic, calls no service and sends no command to the base. Safe with the robot on.
#
# Usage (inside the container):   bash survey_ros2.sh > survey.txt 2>&1
# Usage (from your laptop):       ssh HOST 'docker exec -i CONTAINER bash -s' < pallet_approach/tools/survey_ros2.sh > docs/survey/HOST.txt 2>&1

set +u
export LC_ALL=C
TMPD=$(mktemp -d)
trap 'rm -rf "$TMPD"' EXIT

sec() { printf '\n\n========================= %s =========================\n' "$*"; }
run() {
  printf '\n$ %s\n' "$1"
  timeout "${2:-15}" bash -c "$1" 2>&1 | sed 's/^/    /'
  rc=${PIPESTATUS[0]}
  [ "$rc" = "124" ] && printf '    [timed out after %ss]\n' "${2:-15}"
  return 0
}

printf '### pallet-approach system survey\n'
printf 'generated : %s\n' "$(date -Is)"
printf 'host      : %s\n' "$(hostname)"

# ---------------------------------------------------------------- machine
sec "MACHINE"
run 'uname -srm'
run 'grep -E "^(NAME|VERSION)=" /etc/os-release'
run 'python3 --version; python3 -c "import platform;print(platform.machine())"'
run 'nproc; free -h | head -2; df -h / /tmp 2>/dev/null | sort -u'
if [ -f /.dockerenv ]; then printf '\n    inside a container: YES\n'; else printf '\n    inside a container: no (host shell)\n'; fi
run 'cat /etc/nv_tegra_release 2>/dev/null || echo "not a Tegra/Jetson"'
run 'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || echo "nvidia-smi not available"'
run 'docker ps --format "{{.Names}}\t{{.Image}}\t{{.Status}}" 2>/dev/null || echo "no docker CLI here (expected inside a container)"'
run 'ip -brief addr 2>/dev/null | grep -v "^lo"'

# ---------------------------------------------------------------- ros env
sec "ROS 2 ENVIRONMENT  (multi-machine: these must match on both machines)"
if ! command -v ros2 >/dev/null 2>&1; then
  printf '\n    ros2 not on PATH, sourcing:\n'
  for s in /opt/ros/*/setup.bash; do [ -f "$s" ] && . "$s" && printf '    sourced %s\n' "$s"; done
  for s in /ros2_ws/install/setup.bash /root/ros2_ws/install/setup.bash /colcon_ws/install/setup.bash /workspaces/*/install/setup.bash; do
    [ -f "$s" ] && . "$s" && printf '    sourced %s\n' "$s"
  done
fi
run 'env | grep -E "^(ROS_|RMW_|AMENT_PREFIX|COLCON_PREFIX|CYCLONEDDS|FASTRTPS|FASTDDS)" | sort'
run 'ros2 doctor --report 2>/dev/null | sed -n "1,60p"' 40

# ---------------------------------------------------------------- graph
sec "NODES"
run 'ros2 node list' 25

sec "TOPICS AND TYPES"
run 'ros2 topic list -t' 25

sec "SERVICES  (base control, e-stop, lifecycle)"
run 'ros2 service list -t | grep -viE "/(describe|get|set|list)_parameter|parameter_events|logger" | sort' 25

sec "ACTIONS"
run 'ros2 action list -t' 20

timeout 20 ros2 topic list > "$TMPD/topics" 2>/dev/null
pick() { grep -E "$1" "$TMPD/topics" 2>/dev/null | grep -vE "${2:-ZZZnomatchZZZ}" | head -n "${3:-4}"; }

DEPTH=$(pick 'depth'                          'camera_info|metadata|points|compressed|theora|confidence|nitros')
LEFT=$(pick  'left.*(image|rect)|image.*rect' 'camera_info|compressed|theora|nitros')
CINFO=$(pick 'camera_info'                    'ZZZ' 6)
CLOUD=$(pick 'point_?cloud|/points'           'camera_info' 3)
ODOM=$(pick  'odom'                           'camera_info' 6)
CMD=$(pick   'cmd_vel|cmd_ctrl|/command'      'ZZZ' 8)
SCAN=$(pick  'scan|/os_|lidar|ouster'         'metadata|imu|reflec' 6)
WATCH=$(printf '%s\n%s\n%s\n%s\n%s\n%s\n%s\n/tf\n/tf_static\n' "$DEPTH" "$LEFT" "$CINFO" "$CLOUD" "$ODOM" "$CMD" "$SCAN" | awk 'NF' | sort -u)

printf '\n\n    topics selected for inspection:\n'
printf '%s\n' "$WATCH" | sed 's/^/      /'

sec "QoS, PUBLISHERS AND SUBSCRIBERS  (who else publishes cmd_vel?)"
for t in $WATCH; do run "ros2 topic info -v $t" 15; done

sec "MESSAGE SAMPLES  (one message, arrays hidden -> depth encoding, step, frame_id)"
for t in $DEPTH $LEFT $ODOM $SCAN $CLOUD; do run "ros2 topic echo --once --no-arr $t" 20; done

sec "CAMERA INFO  (full: K, D, P, distortion model)"
for t in $CINFO; do run "ros2 topic echo --once --full-length $t" 20; done

sec "PUBLISH RATES"
for t in $DEPTH $LEFT $ODOM; do run "timeout 6 ros2 topic hz $t" 12; done

# ---------------------------------------------------------------- tf
sec "TF TREE"
cd "$TMPD" 2>/dev/null
run 'timeout 10 ros2 run tf2_ros tf2_monitor' 20
run 'ros2 run tf2_tools view_frames' 40
run "cat $TMPD/frames*.gv 2>/dev/null || echo 'no .gv produced (graphviz/pydot missing?) - use the tf_static dump below'"
timeout 10 ros2 topic echo /tf_static --full-length > "$TMPD/tf_static.txt" 2>&1
run "grep -E 'frame_id|translation|rotation|^  [xyzw]:' $TMPD/tf_static.txt | head -120"

sec "CAMERA -> BASE_LINK TRANSFORM  (does it match the tape measure?)"
FRAMES=$(grep -E 'child_frame_id' "$TMPD/tf_static.txt" 2>/dev/null | awk '{print $2}' | tr -d "'\"" | sort -u)
printf '\n    static frames found:\n'
printf '%s\n' "$FRAMES" | sed 's/^/      /'
CAM=$(printf '%s\n' "$FRAMES" | grep -iE 'cam|zed|optical' | head -6)
for f in $CAM; do run "timeout 5 ros2 run tf2_ros tf2_echo base_link $f" 12; done

sec "URDF  (is the camera mount defined here?)"
timeout 25 ros2 param get /robot_state_publisher robot_description > "$TMPD/urdf.txt" 2>&1
run "wc -c $TMPD/urdf.txt"
run "grep -oE '<(link|joint)[^>]*' $TMPD/urdf.txt | head -100"
run "grep -oE '<origin[^>]*' $TMPD/urdf.txt | head -50"

sec "BASE DRIVER PARAMETERS  (look for a command/watchdog timeout)"
for n in $(timeout 15 ros2 node list 2>/dev/null | grep -iE 'scout|base|chassis|ugv|robot_base'); do
  run "ros2 param dump $n" 25
done
run 'ros2 node list 2>/dev/null | grep -iE "teleop|joy|twist_mux|nav2|controller_server|velocity_smoother" || echo "no teleop/joy/mux/nav2 node visible right now"' 20

# ---------------------------------------------------------------- python deps
sec "PYTHON DEPENDENCIES FOR PERCEPTION"
python3 - <<'PY'
import importlib
mods = ["numpy","scipy","sklearn","cv2","open3d","matplotlib","yaml","rosbags",
        "cv_bridge","tf_transformations","transforms3d","numba","pandas"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"    {m:18s} OK       {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"    {m:18s} MISSING  ({type(e).__name__})")
PY

sec "CAN OPEN3D BE INSTALLED HERE?  (nothing is installed, query only)"
run 'python3 -m pip --version'
run 'timeout 25 python3 -m pip index versions open3d 2>&1 | head -5' 40
run 'timeout 60 python3 -m pip install --dry-run --no-deps open3d 2>&1 | tail -5' 90
run 'apt-cache policy python3-sklearn python3-scipy libpcl-dev 2>/dev/null | head -24'
run 'timeout 8 python3 -c "import urllib.request;urllib.request.urlopen(\"https://pypi.org/simple/\",timeout=6);print(\"pypi reachable\")" 2>&1 | tail -2' 20

# ---------------------------------------------------------------- appendix
sec "APPENDIX: FULL robot_description"
cat "$TMPD/urdf.txt" 2>/dev/null

printf '\n\n### end of survey\n'
