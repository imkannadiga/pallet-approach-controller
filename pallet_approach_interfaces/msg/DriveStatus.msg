# What the controller is doing and what it can see. Published while active, at
# the command rate, so a run can be understood from `ros2 topic echo` alone.
#
# NO BACKSLASHES ANYWHERE IN THIS FILE, not even in a comment. See Health.msg.
std_msgs/Header header

string mode                    # the mode being executed
string phase                   # CLOSING, SQUARING or ARRIVED, for CLOSE
bool arrived                   # this move is complete
bool blocked                   # not driving, and not because it finished
bool abort                     # something is wrong enough to end the run
string reason                  # always populated, with the numbers in it

# Odometry. `travelled` is path length since the move began and `displacement`
# is the straight-line distance covered. For a straight move they should agree;
# a growing gap between them is the robot curving when it was told not to, so
# both are published rather than just the one the stop rule uses.
float32 travelled
float32 displacement

# What the controller measured this frame. NaN for anything not measured in
# this mode -- never 0.0, which reads as a measurement of zero.
float32 face_range             # distance to the pallet face
float32 lateral_offset         # signed offset from the pocket centre line
float32 heading_error          # signed yaw error, radians
float32 rear_clearance         # nearest obstacle behind the bumper
