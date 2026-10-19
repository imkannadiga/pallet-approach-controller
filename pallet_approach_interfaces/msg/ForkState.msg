# The virtual forks. There is no carriage on this robot: the forks are a
# simulated part of the machine, and this message is what the rest of the stack
# and the overlay read instead of a hardware feedback topic.
#
# NO BACKSLASHES ANYWHERE IN THIS FILE, not even in a comment. See Health.msg.
std_msgs/Header header

# Height of the tine UNDERSIDE above the floor, metres -- the same quantity as
# forks.bottom_height in params.yaml, which is what the overlay draws and what
# the pocket clearance is measured from. Not the tine centre and not a
# z-coordinate in any frame.
float32 height
float32 target_height           # where it is moving to; equal to height at rest

string state                    # PARKED, MOVING, INSERT_HEIGHT or LIFTED
bool moving

# Set when the forks have lifted with a pallet on them, cleared when they are
# lowered again. Nothing is weighed: this is the simulator asserting that the
# sequence reached LIFTING with the forks inside a pallet, which is the whole
# claim a virtual pickup can make.
bool loaded
