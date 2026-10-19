# Whether the stack is healthy enough to drive. Published by the watchdog at a
# steady rate; read by the coordinator, which refuses a goal when ok is false
# (D12) and aborts a run when it goes false mid-run (D9).
#
# NO BACKSLASHES ANYWHERE IN THIS FILE, not even in a comment: rosidl_adapter
# copies these comments into the generated IDL and decodes the result with
# Python's 'unicodeescape' codec, so one line-continuation fails the whole
# package build. test_interfaces.py checks this.
std_msgs/Header header

# True when every source REQUIRED IN THE CURRENT PHASE is inside its limits.
# What is required changes: perception publishes nothing between runs, so it is
# required only while a run is in progress. The required[] array below says
# which sources this particular ok refers to.
bool ok
string phase                   # "always" between runs, "running" during one

# One line per required source that is outside its limits, with the numbers in
# it, e.g. "odom: 0.62 s old, limit 0.20 s". Empty when ok.
string[] failures

# Every source, in a fixed order, whether required or not. Parallel arrays
# rather than a sub-message: this is for logs and rqt, and a flat set of arrays
# plots directly.
string[] names
float32[] ages                 # seconds since the last message; NaN = never seen
float32[] rates                # Hz over the window; NaN = not yet measurable
bool[] required                # whether this source is required in this phase
bool[] source_ok

# The watchdog is publishing a zero command on the safety input of twist_mux,
# which outranks both the pallet controller and Nav2. Latched for the rest of
# the run once it fires, so a sensor that flickers cannot produce a stutter of
# stop and go.
bool stop_latched
