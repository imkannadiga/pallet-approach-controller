# One frame's estimate of the pallet's front face and its pockets.
# Published by pallet_perception, consumed by the coordinator and fork_overlay.
std_msgs/Header header              # stamp = the depth image's stamp; frame = base_link

geometry_msgs/Point position        # centre of the fitted front face
geometry_msgs/Vector3 normal        # unit outward face normal, pointing at the robot
float32 yaw                         # atan2(normal.y, normal.x), radians

Pocket[] pockets                    # left to right as the robot sees them
int32 face_inlier_count             # points supporting the face fit
float32 floor_height                # z of the fitted floor plane, in base_link
bool valid                          # false means "no pallet in this frame"
