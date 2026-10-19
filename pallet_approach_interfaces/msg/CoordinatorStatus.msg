# Continuous telemetry for the sequence: what it is doing and why, published
# every tick (20 Hz) so a passive observer -- the dashboard, rqt, a log file --
# can see the live state and reason without waiting on an action goal's
# feedback, which only reaches whoever sent that particular goal.
#
# /coordinator/state (plain std_msgs/String, the state name alone) stays as it
# is: fork_overlay already keys its colour off it and nothing about that is
# wrong. This is the richer sibling, for anything that wants to know WHY.
#
# NO BACKSLASHES ANYWHERE IN THIS FILE, not even in a comment. rosidl_adapter
# copies these comments into the generated IDL and decodes the result with
# Python's 'unicodeescape' codec, so one line-continuation fails the whole
# package build. test_interfaces.py checks this.
std_msgs/Header header

string state                   # the eleven-state machine's current state
string mode                    # "pickup" or "approach_only"
string side                    # "short", "long", "ambiguous", "unknown", or ""
string reason                  # why the state is what it is, with numbers in it
string outcome                 # INSERTED, GO, NO_GO, NO_PALLET or ABORTED;
                               # empty while a run is in progress or none has
                               # happened yet
float32 elapsed                # seconds since the current run started; NaN
                               # while idle and no run has ever been triggered
