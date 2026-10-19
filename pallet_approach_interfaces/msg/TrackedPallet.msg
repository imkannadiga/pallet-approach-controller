# One pallet, tracked across frames. Published by pallet_tracker, consumed by
# the coordinator, which will not move the robot until confirmed is true.
#
# NO BACKSLASHES ANYWHERE IN THIS FILE, not even in a comment. See Health.msg.
std_msgs/Header header

# The NEWEST accepted estimate, not an average of the window. Averaging poses
# lags the truth by half the window times the closing speed, and the closing
# loop is built on the measurement improving as the remaining correction
# shrinks (D34). The window decides whether to trust this estimate; it does not
# alter it.
PalletEstimate estimate

bool confirmed                 # enough consecutive frames, and no disagreement
float32 confidence             # accepted fraction of the window; 0.0 until the
                               # window has enough frames to mean anything
int32 consecutive_frames
int32 total_frames             # accepted frames since the track started
string status                  # SEARCHING, CONFIRMING, CONFIRMED, LOST, REJECTED
string reason                  # why the status is what it is, with numbers

# Where the track sits in the odom frame. Tracked there rather than in
# base_footprint because a stationary pallet has a stationary position there
# even while the robot drives, which is exactly when this has to work.
geometry_msgs/Point center_odom
float32 yaw_odom

# The LiDAR cross-check (D13, D35). evaluated false means there were not enough
# returns on the face to judge, which is NOT agreement and must not be read as
# any. offset is the median signed distance from the returns to the camera's
# face plane, positive when the LiDAR sees the face further away than the
# camera does; NaN when not evaluated.
bool lidar_evaluated
bool lidar_agrees
float32 lidar_offset
int32 lidar_points
string lidar_detail
