# One frame's estimate of the pallet's front face and its pockets.
# Published by pallet_perception, consumed by the coordinator and fork_overlay.
std_msgs/Header header              # stamp = the depth image's stamp; frame = base_footprint

geometry_msgs/Point position        # centre of the fitted front face
geometry_msgs/Vector3 normal        # unit outward face normal, pointing at the robot
float32 yaw                         # atan2(normal.y, normal.x), radians

Pocket[] pockets                    # left to right as the robot sees them
int32 face_inlier_count             # points supporting the face fit
# Height of the fitted floor plane along the frame's z axis. Expect about
# -0.025, NOT zero: base_footprint sits 25 mm above the real floor
# (measured). NaN when no floor was fitted, which happens legitimately at
# the standoff. For measuring pallet geometry use floor_normal/floor_offset
# and take perpendicular distances; this one is for logs and displays.
float32 floor_height

# The fitted floor as a PLANE, which is what pallet heights must be measured
# against. Taking a height as "z minus floor_height" leaks any camera
# orientation error into the answer; a perpendicular distance to this plane
# does not, because a rotation is an isometry and leaves it exactly unchanged.
# Measured on the real 0.868 deg mounting error: 4.5 mm leaked into a z-based
# pocket height, 0.0 mm into the perpendicular one. So use these two, not
# floor_height, for anything that is then compared against a tolerance.
bool floor_valid                    # false means no floor was fitted this frame
geometry_msgs/Vector3 floor_normal  # unit, pointing up
float32 floor_offset                # plane is {p : dot(normal, p) + offset = 0}

bool valid                          # false means "no pallet in this frame"

# Which face of the pallet this is: "short", "long", "ambiguous", or "unknown"
# when classification was not run. Not a label -- it selects the pocket
# nominals, fork spacing and insertion stop everything downstream uses, and the
# two faces differ by 160 mm in pocket spacing (D19/D21/D22).
string side
# Why `side` came out that way, e.g. "spacing says short, width says long".
# Carried so a refusal can be explained after the run instead of being a bare
# "no".
string side_reason
# Visible width of the face, measured along the face itself. A LOWER bound:
# occlusion and the image edge shorten it, nothing lengthens it.
float32 face_width
