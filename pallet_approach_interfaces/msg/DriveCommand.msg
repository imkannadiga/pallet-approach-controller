# What the coordinator wants the controller to do. The controller owns HOW --
# the gains, the limits, the stop rules -- and this says only which behaviour
# and how far.
#
# NO BACKSLASHES ANYWHERE IN THIS FILE, not even in a comment. See Health.msg.
std_msgs/Header header

# STOP     publish zero and stay stopped
# CLOSE    servo onto the pocket centre line and drive to the standoff (D38)
# REVERSE  straight back by `distance`, with the rear LiDAR check (D42)
# INSERT   straight forward into the pockets, stopping on the earlier of
#          odometry and the LiDAR (D18). `side` picks the stop distance.
# BACKOUT  straight back by `distance`, out of the pallet
string mode

# Metres of travel for REVERSE and BACKOUT. Always positive: the mode says the
# direction, so a sign here could only ever contradict it.
float32 distance

# INSERT only: which face is being entered. The stop distance is per side
# (D21) and the controller looks it up in its own parameters rather than being
# told it, so there is one place that number comes from.
string side
